// SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
// SPDX-License-Identifier: Apache-2.0

//! Host-side f32 math: threading helpers, linear layers, LayerNorm, the
//! activations and multi-head attention. Everything is written so the
//! inner loops vectorise (8 independent accumulators, no branches) and
//! spread over scoped threads.

use std::sync::atomic::{AtomicUsize, Ordering};

use taconite::{bf16_to_f32, fast_exp};

static THREADS: AtomicUsize = AtomicUsize::new(0);

/// Host threads to use (default: the machine's, up to 16).
pub fn threads() -> usize {
    match THREADS.load(Ordering::Relaxed) {
        0 => std::thread::available_parallelism().map_or(4, |n| n.get()).min(16),
        n => n,
    }
}

pub fn set_threads(n: usize) {
    THREADS.store(n, Ordering::Relaxed);
}

/// Splits `out` (rows of `row` elements; the last may be short) over the
/// threads; `f(first_row, rows)` fills each piece.
pub fn par_rows<T: Send>(out: &mut [T], row: usize, f: impl Fn(usize, &mut [T]) + Sync) {
    let rows = out.len().div_ceil(row.max(1));
    if rows == 0 {
        return;
    }
    let per = rows.div_ceil(threads());
    if per >= rows {
        f(0, out);
        return;
    }
    std::thread::scope(|s| {
        for (i, piece) in out.chunks_mut(per * row).enumerate() {
            let f = &f;
            s.spawn(move || f(i * per, piece));
        }
    });
}

#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let mut acc = [0f32; 8];
    let chunks = n / 8;
    for c in 0..chunks {
        for l in 0..8 {
            acc[l] += a[c * 8 + l] * b[c * 8 + l];
        }
    }
    let mut s: f32 = acc.iter().sum();
    for i in chunks * 8..n {
        s += a[i] * b[i];
    }
    s
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

    #[inline]
    fn row_dot(&self, o: usize, n_in: usize, x: &[f32]) -> f32 {
        match self {
            W::F32(w) => dot(x, &w[o * n_in..(o + 1) * n_in]),
            W::Bf16(w) => dot_bf16(x, &w[o * n_in..(o + 1) * n_in]),
        }
    }
}

/// `y[rows, out] = x[rows, in] W^T + b`. With few rows (a handful of text
/// tokens, the 201 decoder queries) the work is split over output features,
/// so every thread streams its own slice of W once.
pub fn linear(x: &[f32], n_in: usize, w: W, b: Option<&[f32]>) -> Vec<f32> {
    let n_out = w.len() / n_in;
    let rows = x.len() / n_in;
    let mut y = vec![0f32; rows * n_out];
    if rows >= 4 * threads() {
        par_rows(&mut y, n_out, |r0, out| {
            for (ri, yr) in out.chunks_mut(n_out).enumerate() {
                let xr = &x[(r0 + ri) * n_in..(r0 + ri + 1) * n_in];
                for (o, v) in yr.iter_mut().enumerate() {
                    *v = w.row_dot(o, n_in, xr) + b.map_or(0.0, |b| b[o]);
                }
            }
        });
    } else {
        // transposed [out, rows], then back
        let mut yt = vec![0f32; n_out * rows];
        par_rows(&mut yt, rows, |o0, out| {
            for (oi, col) in out.chunks_mut(rows).enumerate() {
                let o = o0 + oi;
                for (r, v) in col.iter_mut().enumerate() {
                    *v = w.row_dot(o, n_in, &x[r * n_in..(r + 1) * n_in]) + b.map_or(0.0, |b| b[o]);
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
pub fn erf(x: f32) -> f32 {
    let z = (x as f64).abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let r = t
        * (-z * z - 1.265_512_23
            + t * (1.000_023_68
                + t * (0.374_091_96
                    + t * (0.096_784_18
                        + t * (-0.186_288_06
                            + t * (0.278_868_07
                                + t * (-1.135_203_98
                                    + t * (1.488_515_87 + t * (-0.822_152_23 + t * 0.170_872_77)))))))))
            .exp();
    (if x >= 0.0 { 1.0 - r } else { r - 1.0 }) as f32
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

/// In-place softmax of one row; `-inf` entries become 0.
#[inline]
pub fn softmax_(row: &mut [f32]) {
    let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if m == f32::NEG_INFINITY {
        row.fill(0.0);
        return;
    }
    let mut s = 0.0;
    for v in row.iter_mut() {
        *v = fast_exp(*v - m);
        s += *v;
    }
    let inv = 1.0 / s;
    for v in row.iter_mut() {
        *v *= inv;
    }
}

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

/// Attention against many keys with head-major keys and values
/// (`kh`/`vh [H, lk, hd]`) and an additive `bias [H, lq, lk]`:
/// `q [lq, H*hd]` -> `[lq, H*hd]`. The work is split over (head, query)
/// pairs in head-major order, so each thread streams one head's keys --
/// a few hundred KB, L2-resident -- for all of its queries.
pub fn attention_hm(q: &[f32], dim: usize, heads: usize, kh: &[f32], vh: &[f32], bias: Option<&[f32]>) -> Vec<f32> {
    let hd = dim / heads;
    let lk = kh.len() / dim;
    let lq = q.len() / dim;
    let scale = 1.0 / (hd as f32).sqrt();
    let mut oh = vec![0f32; heads * lq * hd]; // [H, lq, hd]
    par_rows(&mut oh, hd, |r0, piece| {
        let mut s = vec![0f32; lk];
        for (ri, o) in piece.chunks_mut(hd).enumerate() {
            let (h, i) = ((r0 + ri) / lq, (r0 + ri) % lq);
            let qh = &q[i * dim + h * hd..][..hd];
            let k = &kh[h * lk * hd..(h + 1) * lk * hd];
            let b = bias.map(|b| &b[(h * lq + i) * lk..][..lk]);
            for j in 0..lk {
                s[j] = dot(qh, &k[j * hd..(j + 1) * hd]) * scale + b.map_or(0.0, |b| b[j]);
            }
            softmax_(&mut s);
            let v = &vh[h * lk * hd..(h + 1) * lk * hd];
            o.fill(0.0);
            for j in 0..lk {
                let p = s[j];
                for (od, &x) in o.iter_mut().zip(&v[j * hd..(j + 1) * hd]) {
                    *od += p * x;
                }
            }
        }
    });
    let mut out = vec![0f32; lq * dim];
    for h in 0..heads {
        for i in 0..lq {
            out[i * dim + h * hd..][..hd].copy_from_slice(&oh[(h * lq + i) * hd..][..hd]);
        }
    }
    out
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
