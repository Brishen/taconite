// SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
// SPDX-License-Identifier: Apache-2.0

//! Host-side f32 math: threading helpers, linear layers, LayerNorm, the
//! activations and multi-head attention. Everything is written so the
//! inner loops vectorise (8 independent accumulators, no branches) and
//! spread over scoped threads.

use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

use taconite::{bf16_to_f32, fast_exp};

static THREADS: AtomicUsize = AtomicUsize::new(0);

/// Host threads to use (default: the machine's, up to 16). Fixed once the
/// pool below has started.
pub fn threads() -> usize {
    match THREADS.load(Ordering::Relaxed) {
        0 => std::thread::available_parallelism().map_or(4, |n| n.get()).min(16),
        n => n,
    }
}

/// Sets the thread count; takes effect if called before the first parallel
/// operation.
pub fn set_threads(n: usize) {
    THREADS.store(n, Ordering::Relaxed);
}

thread_local! {
    static INLINE: Cell<bool> = const { Cell::new(false) };
}

/// Runs `f` with every parallel operation on this thread executed inline
/// (single-threaded), for work done on a side thread while the pool serves
/// the main one.
pub fn run_inline<R>(f: impl FnOnce() -> R) -> R {
    let prev = INLINE.with(|c| c.replace(true));
    let r = f();
    INLINE.with(|c| c.set(prev));
    r
}

/// A pool of `threads() - 1` workers that, with the calling thread, run the
/// `n` indexed tasks of a job; jobs run one at a time, the caller blocking
/// until every worker has left the job, so a task closure only needs to
/// outlive the call. A forward makes a few hundred parallel calls; spawning
/// and joining 16 threads for each cost more than many of them compute.
struct Pool {
    shared: Arc<Shared>,
    workers: usize,
}

type Task<'a> = dyn Fn(usize) + Sync + 'a;

struct Shared {
    state: Mutex<State>,
    wake: Condvar,
    /// the next task index of the current job
    next: AtomicUsize,
    /// workers still inside the current job
    remaining: AtomicUsize,
    done_lock: Mutex<()>,
    done: Condvar,
    panicked: AtomicBool,
}

struct State {
    generation: u64,
    job: Option<(*const Task<'static>, usize)>,
}

// The raw pointer is only dereferenced while its job runs, which the
// submitting thread outlives.
unsafe impl Send for State {}

static POOL: OnceLock<Pool> = OnceLock::new();

fn pool() -> &'static Pool {
    POOL.get_or_init(|| Pool::new(threads().saturating_sub(1)))
}

impl Pool {
    fn new(workers: usize) -> Pool {
        let shared = Arc::new(Shared {
            state: Mutex::new(State { generation: 0, job: None }),
            wake: Condvar::new(),
            next: AtomicUsize::new(0),
            remaining: AtomicUsize::new(0),
            done_lock: Mutex::new(()),
            done: Condvar::new(),
            panicked: AtomicBool::new(false),
        });
        for _ in 0..workers {
            let sh = Arc::clone(&shared);
            std::thread::Builder::new()
                .name("taconite-sam3 worker".into())
                .spawn(move || Self::work(&sh))
                .expect("spawning a worker thread");
        }
        Pool { shared, workers }
    }

    fn work(sh: &Shared) {
        let mut seen = 0u64;
        loop {
            let (task, n) = {
                let mut st = sh.state.lock().unwrap_or_else(|e| e.into_inner());
                while st.generation == seen {
                    st = sh.wake.wait(st).unwrap_or_else(|e| e.into_inner());
                }
                seen = st.generation;
                st.job.expect("a job with the generation")
            };
            let f: &Task<'static> = unsafe { &*task };
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| Self::drain(sh, n, f)));
            if r.is_err() {
                sh.panicked.store(true, Ordering::Relaxed);
            }
            if sh.remaining.fetch_sub(1, Ordering::AcqRel) == 1 {
                let _g = sh.done_lock.lock().unwrap_or_else(|e| e.into_inner());
                sh.done.notify_one();
            }
        }
    }

    fn drain(sh: &Shared, n: usize, f: &Task<'_>) {
        loop {
            let i = sh.next.fetch_add(1, Ordering::Relaxed);
            if i >= n {
                return;
            }
            f(i);
        }
    }

    /// Runs `f(0..n)` over the workers and this thread.
    fn run(&self, n: usize, f: &Task<'_>) {
        if n == 0 {
            return;
        }
        let sh = &*self.shared;
        // the lifetime is erased for the workers; this call outlives them
        let task: *const Task<'static> = unsafe { std::mem::transmute(f as *const Task<'_>) };
        sh.next.store(0, Ordering::Relaxed);
        sh.remaining.store(self.workers, Ordering::Release);
        {
            let mut st = sh.state.lock().unwrap_or_else(|e| e.into_inner());
            st.generation += 1;
            st.job = Some((task, n));
        }
        sh.wake.notify_all();
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| Self::drain(sh, n, f)));
        let mut g = sh.done_lock.lock().unwrap_or_else(|e| e.into_inner());
        while sh.remaining.load(Ordering::Acquire) != 0 {
            g = sh.done.wait(g).unwrap_or_else(|e| e.into_inner());
        }
        drop(g);
        if let Err(e) = r {
            std::panic::resume_unwind(e);
        }
        if sh.panicked.swap(false, Ordering::Relaxed) {
            panic!("a parallel task panicked");
        }
    }
}

/// Splits `out` (rows of `row` elements; the last may be short) over the
/// threads; `f(first_row, rows)` fills each piece. Pieces are a few per
/// thread so uneven ones balance.
pub fn par_rows<T: Send>(out: &mut [T], row: usize, f: impl Fn(usize, &mut [T]) + Sync) {
    let row = row.max(1);
    let rows = out.len().div_ceil(row);
    if rows == 0 {
        return;
    }
    let nt = threads();
    if nt <= 1 || rows < 2 || INLINE.with(Cell::get) {
        f(0, out);
        return;
    }
    let tasks = rows.min(4 * nt);
    let per = rows.div_ceil(tasks);
    let tasks = rows.div_ceil(per);
    let len = out.len();
    let ptr = out.as_mut_ptr() as usize;
    pool().run(tasks, &|i| {
        let start = i * per * row;
        if start >= len {
            return;
        }
        let n = (per * row).min(len - start);
        // disjoint pieces of `out`, each handed to exactly one task
        let piece = unsafe { std::slice::from_raw_parts_mut((ptr as *mut T).add(start), n) };
        f(i * per, piece);
    });
}

// The hot kernels below are written once, generic over whether fused
// multiply-adds are available (`F`), and compiled twice on x86-64: for the
// baseline (SSE2) and for AVX2 + FMA, chosen at run time. `mul_add` only
// belongs in the FMA build -- without the instruction it is a libm call.
#[inline(always)]
fn fma<const F: bool>(a: f32, b: f32, c: f32) -> f32 {
    if F { a.mul_add(b, c) } else { a * b + c }
}

#[cfg(target_arch = "x86_64")]
fn have_avx2() -> bool {
    static F: OnceLock<bool> = OnceLock::new();
    *F.get_or_init(|| std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma"))
}

/// `$name(args)` -> `$imp::<true>` compiled for AVX2 + FMA when the CPU
/// has them, else `$imp::<false>` for the baseline target.
macro_rules! dispatch {
    ($(#[$m:meta])* $vis:vis fn $name:ident($($arg:ident: $ty:ty),* $(,)?) $(-> $ret:ty)? = $imp:ident) => {
        $(#[$m])*
        #[inline]
        $vis fn $name($($arg: $ty),*) $(-> $ret)? {
            #[cfg(target_arch = "x86_64")]
            {
                #[target_feature(enable = "avx2,fma")]
                fn fast($($arg: $ty),*) $(-> $ret)? {
                    $imp::<true>($($arg),*)
                }
                if have_avx2() {
                    // SAFETY: the features were detected on this CPU
                    return unsafe { fast($($arg),*) };
                }
            }
            $imp::<false>($($arg),*)
        }
    };
}

#[inline(always)]
fn dot_impl<const F: bool>(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let mut acc = [0f32; 8];
    let chunks = n / 8;
    for c in 0..chunks {
        let (x, y) = (&a[c * 8..c * 8 + 8], &b[c * 8..c * 8 + 8]);
        for l in 0..8 {
            acc[l] = fma::<F>(x[l], y[l], acc[l]);
        }
    }
    let mut s: f32 = acc.iter().sum();
    for i in chunks * 8..n {
        s += a[i] * b[i];
    }
    s
}

/// Dot product of the common prefix of `a` and `b`.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if have_avx2() {
        // SAFETY: the features were detected on this CPU
        return unsafe { avx::dot(a, b) };
    }
    dot_impl::<false>(a, b)
}

#[inline]
pub fn dot_bf16(a: &[f32], b: &[u16]) -> f32 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let mut acc = [0f32; 8];
    let chunks = n / 8;
    for c in 0..chunks {
        for l in 0..8 {
            acc[l] += a[c * 8 + l] * bf16_to_f32(b[c * 8 + l]);
        }
    }
    let mut s: f32 = acc.iter().sum();
    for i in chunks * 8..n {
        s += a[i] * bf16_to_f32(b[i]);
    }
    s
}

/// A weight matrix `[out, in]` (a torch Linear's layout), f32 or bf16.
#[derive(Clone, Copy)]
pub enum W<'a> {
    F32(&'a [f32]),
    Bf16(&'a [u16]),
}

impl W<'_> {
    fn len(&self) -> usize {
        match self {
            W::F32(w) => w.len(),
            W::Bf16(w) => w.len(),
        }
    }

}

impl W<'_> {
    /// Row `o` of W as f32 (converted into `tmp` for bf16 weights).
    #[inline]
    fn row_f32<'t>(&'t self, o: usize, n_in: usize, tmp: &'t mut [f32]) -> &'t [f32] {
        match self {
            W::F32(w) => &w[o * n_in..(o + 1) * n_in],
            W::Bf16(w) => {
                let src = &w[o * n_in..(o + 1) * n_in];
                for (t, &v) in tmp.iter_mut().zip(src) {
                    *t = bf16_to_f32(v);
                }
                &tmp[..n_in]
            }
        }
    }
}

/// Four rows of `x` (each `n_in` wide) against one row of W: four dots
/// sharing every load of the weights, each over 8 lanes of accumulators.
#[inline(always)]
fn dot4_impl<const F: bool>(x: &[f32], n_in: usize, w: &[f32]) -> [f32; 4] {
    let n = n_in.min(w.len());
    let (x0, x1, x2, x3) = (&x[..n], &x[n_in..n_in + n], &x[2 * n_in..2 * n_in + n], &x[3 * n_in..3 * n_in + n]);
    let mut acc = [[0f32; 8]; 4];
    let chunks = n / 8;
    for c in 0..chunks {
        let wv = &w[c * 8..c * 8 + 8];
        let (a, b, cc, d) = (&x0[c * 8..c * 8 + 8], &x1[c * 8..c * 8 + 8], &x2[c * 8..c * 8 + 8], &x3[c * 8..c * 8 + 8]);
        for l in 0..8 {
            acc[0][l] = fma::<F>(a[l], wv[l], acc[0][l]);
            acc[1][l] = fma::<F>(b[l], wv[l], acc[1][l]);
            acc[2][l] = fma::<F>(cc[l], wv[l], acc[2][l]);
            acc[3][l] = fma::<F>(d[l], wv[l], acc[3][l]);
        }
    }
    let mut s = [0f32; 4];
    for (r, xr) in [x0, x1, x2, x3].into_iter().enumerate() {
        s[r] = acc[r].iter().sum();
        for i in chunks * 8..n {
            s[r] += xr[i] * w[i];
        }
    }
    s
}

#[inline]
fn dot4(x: &[f32], n_in: usize, w: &[f32]) -> [f32; 4] {
    #[cfg(target_arch = "x86_64")]
    if have_avx2() {
        // SAFETY: the features were detected on this CPU
        return unsafe { avx::dot4(x, n_in, w) };
    }
    dot4_impl::<false>(x, n_in, w)
}

/// `y[rows, out] = x[rows, in] W^T + b`. Rows go four at a time through
/// each row of W (loaded once per four rows, not once per row: the W
/// traffic, not the arithmetic, is what bounds these small layers). With
/// few rows (a handful of text tokens) the work is split over output
/// features, so every thread streams its own slice of W once; with many
/// (the 201 decoder queries, thousands of tokens) over blocks of rows.
pub fn linear(x: &[f32], n_in: usize, w: W, b: Option<&[f32]>) -> Vec<f32> {
    let n_out = w.len() / n_in;
    let rows = x.len() / n_in;
    let mut y = vec![0f32; rows * n_out];
    let bias = |o: usize| b.map_or(0.0, |b| b[o]);
    // y for rows `r..r + nr` (nr <= 4) and output `o`, `stride` apart in `out`
    let block = |xb: &[f32], nr: usize, o: usize, wrow: &[f32], out: &mut [f32], stride: usize| {
        if nr == 4 {
            let d = dot4(xb, n_in, wrow);
            for (r, dr) in d.into_iter().enumerate() {
                out[r * stride] = dr + bias(o);
            }
        } else {
            for r in 0..nr {
                out[r * stride] = dot(&xb[r * n_in..(r + 1) * n_in], wrow) + bias(o);
            }
        }
    };
    if n_in <= 8 && rows >= 4 * threads() {
        // a tiny input (the box bias MLP's 2 coordinates): each output is
        // a few multiply-adds, so the blocked path's per-output overhead
        // would be the whole cost
        let wf: Vec<f32> = match w {
            W::F32(w) => w.to_vec(),
            W::Bf16(w) => w.iter().map(|&v| bf16_to_f32(v)).collect(),
        };
        par_rows(&mut y, n_out, |r0, out| {
            for (ri, yr) in out.chunks_mut(n_out).enumerate() {
                let xr = &x[(r0 + ri) * n_in..(r0 + ri + 1) * n_in];
                for (o, (yo, wr)) in yr.iter_mut().zip(wf.chunks(n_in)).enumerate() {
                    let mut acc = bias(o);
                    for (a, b) in xr.iter().zip(wr) {
                        acc += a * b;
                    }
                    *yo = acc;
                }
            }
        });
    } else if rows >= 4 * threads() {
        // pieces of 4 rows
        par_rows(&mut y, 4 * n_out, |b0, out| {
            let mut tmp = vec![0f32; n_in];
            for (bi, yb) in out.chunks_mut(4 * n_out).enumerate() {
                let r = 4 * (b0 + bi);
                let nr = (rows - r).min(4);
                let xb = &x[r * n_in..(r + nr) * n_in];
                for o in 0..n_out {
                    let wrow = w.row_f32(o, n_in, &mut tmp);
                    block(xb, nr, o, wrow, &mut yb[o..], n_out);
                }
            }
        });
    } else {
        // transposed [out, rows], then back
        let mut yt = vec![0f32; n_out * rows];
        par_rows(&mut yt, rows, |o0, out| {
            let mut tmp = vec![0f32; n_in];
            for (oi, col) in out.chunks_mut(rows).enumerate() {
                let o = o0 + oi;
                let wrow = w.row_f32(o, n_in, &mut tmp);
                for r in (0..rows).step_by(4) {
                    let nr = (rows - r).min(4);
                    block(&x[r * n_in..(r + nr) * n_in], nr, o, wrow, &mut col[r..], 1);
                }
            }
        });
        for o in 0..n_out {
            for r in 0..rows {
                y[r * n_out + o] = yt[o * rows + r];
            }
        }
    }
    y
}

/// LayerNorm over rows of `dim`, into a new buffer.
pub fn layer_norm(x: &[f32], dim: usize, w: &[f32], b: &[f32], eps: f32) -> Vec<f32> {
    let mut y = vec![0f32; x.len()];
    par_rows(&mut y, dim, |r0, out| {
        for (ri, yr) in out.chunks_mut(dim).enumerate() {
            let xr = &x[(r0 + ri) * dim..(r0 + ri + 1) * dim];
            ln_row(xr, yr, w, b, eps);
        }
    });
    y
}

#[inline]
pub fn ln_row(x: &[f32], y: &mut [f32], w: &[f32], b: &[f32], eps: f32) {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let inv = 1.0 / (var + eps).sqrt();
    for i in 0..x.len() {
        y[i] = (x[i] - mean) * inv * w[i] + b[i];
    }
}

/// erf to 1.2e-7 (Numerical Recipes' erfc Chebyshev fit); std has none.
#[inline]
/// erf to ~1e-6 (Numerical Recipes' erfc Chebyshev fit), in f32 with the
/// branch-free `fast_exp` so a loop over it vectorises: the neck's GELU
/// runs it 10M times a forward, and libm's f64 exp was that pass.
pub fn erf(x: f32) -> f32 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let r = t
        * fast_exp(
            -z * z - 1.265_512_23
                + t * (1.000_023_68
                    + t * (0.374_091_96
                        + t * (0.096_784_18
                            + t * (-0.186_288_06
                                + t * (0.278_868_07
                                    + t * (-1.135_203_98
                                        + t * (1.488_515_87 + t * (-0.822_152_23 + t * 0.170_872_77)))))))),
        );
    // branch-free sign
    let pos = 1.0 - r;
    let neg = r - 1.0;
    if x >= 0.0 { pos } else { neg }
}

/// The exact (erf) GELU, as torch's default.
#[inline]
pub fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x * std::f32::consts::FRAC_1_SQRT_2))
}

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn relu_(x: &mut [f32]) {
    for v in x {
        *v = v.max(0.0);
    }
}

pub fn add_(x: &mut [f32], y: &[f32]) {
    for (a, b) in x.iter_mut().zip(y) {
        *a += b;
    }
}

/// In-place softmax of one row; `-inf` entries become 0. Three passes --
/// max, exp, sum -- each written to vectorise: the reductions over 8
/// lanes, the exp on its own (an exp fused with a serial sum runs scalar).
#[inline(always)]
fn softmax_impl<const F: bool>(row: &mut [f32]) {
    let mut mx = [f32::NEG_INFINITY; 8];
    for v in row.chunks_exact(8).remainder() {
        mx[0] = mx[0].max(*v);
    }
    for c in row.chunks_exact(8) {
        for l in 0..8 {
            mx[l] = mx[l].max(c[l]);
        }
    }
    let m = mx.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if m == f32::NEG_INFINITY {
        row.fill(0.0);
        return;
    }
    for v in row.iter_mut() {
        *v = fast_exp(*v - m);
    }
    let mut acc = [0f32; 8];
    for v in row.chunks_exact(8).remainder() {
        acc[0] += *v;
    }
    for c in row.chunks_exact(8) {
        for l in 0..8 {
            acc[l] += c[l];
        }
    }
    let inv = 1.0 / acc.iter().sum::<f32>();
    for v in row.iter_mut() {
        *v *= inv;
    }
}

dispatch!(pub fn softmax_(row: &mut [f32]) = softmax_impl);

/// Multi-head attention options: `valid` masks keys; `bias` is added to the
/// scores, `[H, lq, lk]`; `causal` masks keys after the query.
#[derive(Default)]
pub struct Attn<'a> {
    pub heads: usize,
    pub valid: Option<&'a [bool]>,
    pub bias: Option<&'a [f32]>,
    pub causal: bool,
}

/// `q [lq, dim]`, `k`/`v [lk, dim]` (heads side by side, `dim = H * hd`)
/// -> `[lq, dim]`, scale 1/sqrt(hd). `k` and `v` may be bf16 or f32 rows of
/// width `kv_stride` whose first `dim` elements are used (a slice of a wider
/// GEMM output).
pub fn attention(q: &[f32], dim: usize, k: Rows, v: Rows, a: &Attn) -> Vec<f32> {
    let h = a.heads;
    let hd = dim / h;
    let lq = q.len() / dim;
    let lk = k.rows;
    let scale = 1.0 / (hd as f32).sqrt();
    let mut out = vec![0f32; lq * dim];
    // one task per query row, all heads
    par_rows(&mut out, dim, |i0, piece| {
        let mut s = vec![0f32; lk];
        for (ii, orow) in piece.chunks_mut(dim).enumerate() {
            let i = i0 + ii;
            for hh in 0..h {
                let qh = &q[i * dim + hh * hd..i * dim + (hh + 1) * hd];
                for j in 0..lk {
                    let masked = a.valid.is_some_and(|m| !m[j]) || (a.causal && j > i);
                    s[j] = if masked {
                        f32::NEG_INFINITY
                    } else {
                        dot(qh, k.at(j, hh * hd, hd)) * scale + a.bias.map_or(0.0, |b| b[(hh * lq + i) * lk + j])
                    };
                }
                softmax_(&mut s);
                let oh = &mut orow[hh * hd..(hh + 1) * hd];
                oh.fill(0.0);
                for j in 0..lk {
                    let p = s[j];
                    if p != 0.0 {
                        for (o, &x) in oh.iter_mut().zip(v.at(j, hh * hd, hd)) {
                            *o += p * x;
                        }
                    }
                }
            }
        }
    });
    out
}

/// One task of [`attention_dec`]: `QB` queries `qs [QB, hd]` (scaled; the
/// unused ones zero) of one head against its keys `k [hd, lk]` and values
/// `v [lk, hd]`, with the bias rows `by`/`bx [nq, g]`, into `o [QB, hd]`;
/// `s [QB, lk]` is scratch.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn dec_block_impl<const F: bool>(
    qs: &[f32],
    hd: usize,
    nq: usize,
    k: &[f32],
    v: &[f32],
    by: &[f32],
    bx: &[f32],
    g: usize,
    s: &mut [f32],
    o: &mut [f32],
) {
    const QB: usize = 8;
    const KB: usize = 8;
    let lk = g * g;
    for jb in (0..lk).step_by(KB) {
        let mut acc = [[0f32; KB]; QB];
        for d in 0..hd {
            let kv = &k[d * lk + jb..][..KB];
            for qi in 0..QB {
                let qd = qs[qi * hd + d];
                for l in 0..KB {
                    acc[qi][l] = fma::<F>(qd, kv[l], acc[qi][l]);
                }
            }
        }
        for qi in 0..QB {
            s[qi * lk + jb..][..KB].copy_from_slice(&acc[qi]);
        }
    }
    for qi in 0..nq {
        let (byr, bxr) = (&by[qi * g..][..g], &bx[qi * g..][..g]);
        let srow = &mut s[qi * lk..(qi + 1) * lk];
        for (iy, sr) in srow.chunks_mut(g).enumerate() {
            let b = byr[iy];
            for (sj, &bxx) in sr.iter_mut().zip(bxr) {
                *sj += b + bxx;
            }
        }
        softmax_impl::<F>(srow);
    }
    for qi in nq..QB {
        s[qi * lk..(qi + 1) * lk].fill(0.0);
    }
    o.fill(0.0);
    for (j, vrow) in v.chunks(hd).enumerate() {
        for qi in 0..QB {
            let p = s[qi * lk + j];
            let oq = &mut o[qi * hd..(qi + 1) * hd];
            for (od, &x) in oq.iter_mut().zip(vrow) {
                *od = fma::<F>(p, x, *od);
            }
        }
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn dec_block(qs: &[f32], hd: usize, nq: usize, k: &[f32], v: &[f32], by: &[f32], bx: &[f32], g: usize, s: &mut [f32], o: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if have_avx2() {
        // SAFETY: the features were detected on this CPU
        return unsafe { avx::dec_block(qs, hd, nq, k, v, by, bx, g, s, o) };
    }
    dec_block_impl::<false>(qs, hd, nq, k, v, by, bx, g, s, o)
}

/// The DETR decoder's vision cross-attention: `q [lq, H*hd]` against
/// transposed keys `kt [H, hd, lk]` and values `vh [H, lk, hd]`, with the
/// box relative-position bias in its separable form, `by [H, lq, g]` +
/// `bx [H, lq, g]` over the `g x g` key grid (`lk == g * g`).
///
/// A task is one head and a block of `QB` queries: the scores are built
/// 8 keys at a time with the `QB x 8` accumulators in registers across
/// the key dimensions (one load of the keys per block, no per-key
/// reduction), the bias is added as `by[iy] + bx[ix]` on the way (the
/// `[H, lq, lk]` bias is never materialised), and the values are streamed
/// once per block for all `QB` queries. One head's keys and values are
/// 1.3 MB; streaming them per query, not per block, was what bounded the
/// earlier version.
pub fn attention_dec(
    q: &[f32],
    dim: usize,
    heads: usize,
    kt: &[f32],
    vh: &[f32],
    by: &[f32],
    bx: &[f32],
    g: usize,
) -> Vec<f32> {
    const QB: usize = 8;
    const KB: usize = 8;
    let hd = dim / heads;
    let lk = g * g;
    let lq = q.len() / dim;
    assert_eq!(lk % KB, 0, "the key count must be a multiple of {KB}");
    debug_assert_eq!(kt.len(), heads * hd * lk);
    debug_assert_eq!(vh.len(), heads * lk * hd);
    let scale = 1.0 / (hd as f32).sqrt();
    let blocks = lq.div_ceil(QB);
    let mut oh = vec![0f32; heads * blocks * QB * hd]; // [H, block, QB, hd]
    par_rows(&mut oh, QB * hd, |b0, piece| {
        let mut s = vec![0f32; QB * lk];
        let mut qs = vec![0f32; QB * hd];
        for (bi, o) in piece.chunks_mut(QB * hd).enumerate() {
            let (h, qb) = ((b0 + bi) / blocks, (b0 + bi) % blocks);
            let i0 = qb * QB;
            let nq = (lq - i0).min(QB);
            qs.fill(0.0);
            for qi in 0..nq {
                for d in 0..hd {
                    qs[qi * hd + d] = q[(i0 + qi) * dim + h * hd + d] * scale;
                }
            }
            let k = &kt[h * hd * lk..(h + 1) * hd * lk];
            let v = &vh[h * lk * hd..(h + 1) * lk * hd];
            let (byr, bxr) = (&by[(h * lq + i0) * g..][..nq * g], &bx[(h * lq + i0) * g..][..nq * g]);
            dec_block(&qs, hd, nq, k, v, byr, bxr, g, &mut s, o);
        }
    });
    let mut out = vec![0f32; lq * dim];
    for h in 0..heads {
        for i in 0..lq {
            let src = (h * blocks * QB + i) * hd;
            out[i * dim + h * hd..][..hd].copy_from_slice(&oh[src..src + hd]);
        }
    }
    out
}

#[inline(always)]
fn gelu_bf16_impl<const F: bool>(src: &[u16], dst: &mut [u16]) {
    for (o, &v) in dst.iter_mut().zip(src) {
        *o = taconite::f32_to_bf16(gelu(bf16_to_f32(v)));
    }
}

dispatch!(
    /// `dst = bf16(gelu(src))`, exact (erf) GELU over bf16 rows.
    pub fn gelu_bf16(src: &[u16], dst: &mut [u16]) = gelu_bf16_impl
);


/// The hot kernels in AVX2 + FMA intrinsics: the portable versions above
/// are written to vectorise, but the compiler leaves their 8-lane
/// accumulator patterns at a fraction of the machine's rate (a 4 x 256 dot
/// at 220 ns where these take 20).
#[cfg(target_arch = "x86_64")]
#[allow(unsafe_op_in_unsafe_fn)]
mod avx {
    use std::arch::x86_64::*;

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn hsum(v: __m256) -> f32 {
        let lo = _mm256_castps256_ps128(v);
        let hi = _mm256_extractf128_ps(v, 1);
        let s = _mm_add_ps(lo, hi);
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
        _mm_cvtss_f32(s)
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let mut acc = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut i = 0;
        while i + 16 <= n {
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), acc);
            acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i + 8)), _mm256_loadu_ps(pb.add(i + 8)), acc2);
            i += 16;
        }
        if i + 8 <= n {
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), acc);
            i += 8;
        }
        let mut s = hsum(_mm256_add_ps(acc, acc2));
        while i < n {
            s += a[i] * b[i];
            i += 1;
        }
        s
    }

    /// Four rows of `x` (`n_in` apart) against `w`.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot4(x: &[f32], n_in: usize, w: &[f32]) -> [f32; 4] {
        let n = n_in.min(w.len());
        debug_assert!(x.len() >= 3 * n_in + n);
        let px = x.as_ptr();
        let pw = w.as_ptr();
        let (mut a0, mut a1, mut a2, mut a3) =
            (_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps());
        let mut i = 0;
        while i + 8 <= n {
            let wv = _mm256_loadu_ps(pw.add(i));
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(px.add(i)), wv, a0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(px.add(n_in + i)), wv, a1);
            a2 = _mm256_fmadd_ps(_mm256_loadu_ps(px.add(2 * n_in + i)), wv, a2);
            a3 = _mm256_fmadd_ps(_mm256_loadu_ps(px.add(3 * n_in + i)), wv, a3);
            i += 8;
        }
        let mut s = [hsum(a0), hsum(a1), hsum(a2), hsum(a3)];
        while i < n {
            for (r, sr) in s.iter_mut().enumerate() {
                *sr += x[r * n_in + i] * w[i];
            }
            i += 1;
        }
        s
    }

    /// [`super::dec_block`]: scores 8 keys x 8 queries in registers across
    /// the key dimensions; values streamed once per pair of queries.
    #[target_feature(enable = "avx2,fma")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn dec_block(
        qs: &[f32],
        hd: usize,
        nq: usize,
        k: &[f32],
        v: &[f32],
        by: &[f32],
        bx: &[f32],
        g: usize,
        s: &mut [f32],
        o: &mut [f32],
    ) {
        const QB: usize = 8;
        let lk = g * g;
        debug_assert!(lk % 8 == 0 && hd % 8 == 0 && k.len() >= hd * lk && v.len() >= lk * hd);
        debug_assert!(s.len() >= QB * lk && o.len() >= QB * hd && qs.len() >= QB * hd);
        let (pk, pq, ps) = (k.as_ptr(), qs.as_ptr(), s.as_mut_ptr());
        let mut jb = 0;
        while jb < lk {
            let mut acc = [_mm256_setzero_ps(); QB];
            for d in 0..hd {
                let kv = _mm256_loadu_ps(pk.add(d * lk + jb));
                for (qi, a) in acc.iter_mut().enumerate() {
                    *a = _mm256_fmadd_ps(_mm256_broadcast_ss(&*pq.add(qi * hd + d)), kv, *a);
                }
            }
            for (qi, a) in acc.iter().enumerate() {
                _mm256_storeu_ps(ps.add(qi * lk + jb), *a);
            }
            jb += 8;
        }
        for qi in 0..nq {
            let (byr, bxr) = (&by[qi * g..][..g], &bx[qi * g..][..g]);
            let srow = &mut s[qi * lk..(qi + 1) * lk];
            for (iy, sr) in srow.chunks_mut(g).enumerate() {
                let b = byr[iy];
                for (sj, &bxx) in sr.iter_mut().zip(bxr) {
                    *sj += b + bxx;
                }
            }
            super::softmax_impl::<true>(srow);
        }
        for qi in nq..QB {
            s[qi * lk..(qi + 1) * lk].fill(0.0);
        }
        // values: two queries at a time, hd / 8 accumulators each
        let pv = v.as_ptr();
        let po = o.as_mut_ptr();
        let nv = hd / 8;
        debug_assert!(nv <= 4);
        let mut qi = 0;
        while qi < QB {
            let mut a0 = [_mm256_setzero_ps(); 4];
            let mut a1 = [_mm256_setzero_ps(); 4];
            let (s0, s1) = (ps.add(qi * lk), ps.add((qi + 1) * lk));
            for j in 0..lk {
                let p0 = _mm256_broadcast_ss(&*s0.add(j));
                let p1 = _mm256_broadcast_ss(&*s1.add(j));
                let vr = pv.add(j * hd);
                for c in 0..nv {
                    let vv = _mm256_loadu_ps(vr.add(c * 8));
                    a0[c] = _mm256_fmadd_ps(p0, vv, a0[c]);
                    a1[c] = _mm256_fmadd_ps(p1, vv, a1[c]);
                }
            }
            for c in 0..nv {
                _mm256_storeu_ps(po.add(qi * hd + c * 8), a0[c]);
                _mm256_storeu_ps(po.add((qi + 1) * hd + c * 8), a1[c]);
            }
            qi += 2;
        }
    }
}

/// Row-major key/value rows, `stride` elements apart, read from column
/// `off` on (e.g. one layer's slice of a wider GEMM output).
#[derive(Clone, Copy)]
pub struct Rows<'a> {
    pub data: &'a [f32],
    pub rows: usize,
    pub stride: usize,
    pub off: usize,
}

impl<'a> Rows<'a> {
    pub fn f32(data: &'a [f32], stride: usize) -> Self {
        Rows { data, rows: data.len() / stride, stride, off: 0 }
    }

    pub fn strided(data: &'a [f32], rows: usize, stride: usize, off: usize) -> Self {
        Rows { data, rows, stride, off }
    }

    #[inline]
    fn at(&self, j: usize, c: usize, n: usize) -> &'a [f32] {
        let s = j * self.stride + self.off + c;
        &self.data[s..s + n]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `cargo test --release -- --ignored --nocapture bench`: the host
    /// kernels' cost at the decoder's shapes.
    #[test]
    #[ignore]
    fn bench() {
        use std::time::Instant;
        let time = |name: &str, reps: usize, mut f: Box<dyn FnMut()>| {
            f();
            let t0 = Instant::now();
            for _ in 0..reps {
                f();
            }
            println!("{name:40} {:8.3} ms", t0.elapsed().as_secs_f64() * 1e3 / reps as f64);
        };
        let x = |n: usize| -> Vec<f32> { (0..n).map(|i| ((i * 7919) % 1000) as f32 / 500.0 - 1.0).collect() };
        let (a, w1, w2) = (x(201 * 256), x(2048 * 256), x(256 * 2048));
        time("linear 201x256 -> 2048", 20, Box::new(|| { linear(&a, 256, W::F32(&w1), None); }));
        let h = x(201 * 2048);
        time("linear 201x2048 -> 256", 20, Box::new(|| { linear(&h, 2048, W::F32(&w2), None); }));
        let (r2, wr1, wr2, hr) = (x(14472 * 2), x(256 * 2), x(8 * 256), x(14472 * 256));
        time("linear 14472x2 -> 256", 20, Box::new(|| { linear(&r2, 2, W::F32(&wr1), None); }));
        time("linear 14472x256 -> 8", 20, Box::new(|| { linear(&hr, 256, W::F32(&wr2), None); }));
        let mut row = x(5184);
        time("softmax 5184 (x1000)", 5, Box::new(|| { for _ in 0..1000 { softmax_(&mut row); } }));
        time("par_rows empty x100", 5, Box::new(|| { let mut buf = vec![0f32; 64 * 4096]; for _ in 0..100 { par_rows(&mut buf, 4096, |_, _| {}); } }));
        time("par_rows touch x100", 5, Box::new(|| { let mut buf = vec![0f32; 64 * 4096]; for _ in 0..100 { par_rows(&mut buf, 4096, |_, p| p[0] += 1.0); } }));
        let (q, kt, vh, by, bx) = (x(201 * 256), x(256 * 5184), x(256 * 5184), x(8 * 201 * 72), x(8 * 201 * 72));
        time("attention_dec 201 q, 8 h, 5184 k", 10, Box::new(|| { attention_dec(&q, 256, 8, &kt, &vh, &by, &bx, 72); }));
        let xr = x(4 * 256);
        time("dot4 256 (x10000)", 10, Box::new(|| { let mut s = 0.0; for _ in 0..10000 { s += dot4(&xr, 256, &w1[..256])[0]; } std::hint::black_box(s); }));
        let ln_x = x(5184 * 256);
        let (lw, lb) = (x(256), x(256));
        time("layer_norm 5184x256", 20, Box::new(|| { layer_norm(&ln_x, 256, &lw, &lb, 1e-5); }));
    }

    #[test]
    fn erf_matches_known_values() {
        for (x, e) in
            [(0.0f32, 0.0f32), (0.5, 0.520_499_9), (1.0, 0.842_700_8), (-2.0, -0.995_322_3), (3.0, 0.999_977_9)]
        {
            assert!((erf(x) - e).abs() < 3e-7, "erf({x}) = {} vs {e}", erf(x));
        }
    }

    #[test]
    fn linear_both_split_strategies_agree() {
        let n_in = 37;
        let w: Vec<f32> = (0..n_in * 11).map(|i| ((i * 7) % 13) as f32 * 0.1 - 0.6).collect();
        let b: Vec<f32> = (0..11).map(|i| i as f32 * 0.01).collect();
        for rows in [2usize, 200] {
            let x: Vec<f32> = (0..rows * n_in).map(|i| ((i * 5) % 17) as f32 * 0.05).collect();
            let y = linear(&x, n_in, W::F32(&w), Some(&b));
            for r in 0..rows {
                for o in 0..11 {
                    let e: f32 = (0..n_in).map(|i| x[r * n_in + i] * w[o * n_in + i]).sum::<f32>() + b[o];
                    assert!((y[r * 11 + o] - e).abs() < 1e-4);
                }
            }
        }
    }

    #[test]
    fn attention_masks_and_normalises() {
        // one head, two queries, three keys; key 2 masked, query 0 causal
        let q = vec![1.0, 0.0, 0.0, 1.0];
        let k = vec![1.0, 0.0, 0.0, 1.0, 5.0, 5.0];
        let v = vec![1.0, 2.0, 3.0, 4.0, 100.0, 100.0];
        let valid = [true, true, false];
        let a = Attn { heads: 1, valid: Some(&valid), causal: true, ..Default::default() };
        let o = attention(&q, 2, Rows::f32(&k, 2), Rows::f32(&v, 2), &a);
        assert_eq!(&o[..2], &[1.0, 2.0]); // query 0 sees key 0 only
        let s = 1.0 / 2f32.sqrt();
        let (p0, p1) = (1.0 / (1.0 + s.exp()), s.exp() / (1.0 + s.exp()));
        assert!((o[2] - (p0 * 1.0 + p1 * 3.0)).abs() < 1e-4);
    }
}
