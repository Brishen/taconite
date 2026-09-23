// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Embeddable Rust runtime for the AdaFace IR face embedders: NPU
//! convolutions + CPU glue/head, no Python.
//!
//! [`IrEmbedder`] replays a bundle exported by
//! `iron/applications/adaface_ir18/export_ir18.py` (or the IR-101 wrapper
//! `export_ir101.py`): the shared IRON bundle format (read with
//! `taconite-bundle`), whose manifest lists the network's steps in order. The
//! steps encode the architecture, so this one runtime serves every exported
//! IR backbone; the `run_ir18` / `run_ir101` binaries are thin wrappers over
//! [`cli`], which is itself a consumer of the public API:
//!
//! ```no_run
//! use taconite_adaface::IrEmbedder;
//!
//! # fn main() -> Result<(), taconite_adaface::Error> {
//! let mut emb = IrEmbedder::load("/path/to/bundle")?;
//! // An aligned 112x112 face crop, CHW f32, cvlface preprocessing
//! // ((rgb/255) - 0.5) / 0.5. Channels beyond the image's (the bundle pads
//! // the stem to a multiple of 8) are zero-filled internally.
//! let chw = vec![0f32; 3 * 112 * 112];
//! let embedding = emb.embed(&chw)?; // 512-d
//! # let _ = embedding;
//! # Ok(())
//! # }
//! ```
//!
//! Loading reads every weight into RAM and checks every step against them.
//! The first forward creates one XRT hardware context per distinct conv
//! kernel (16 for IR-18/IR-101) and the session keeps them ALL resident (LRU
//! cache in the C++ shim, sized to the NPU2's 16-concurrent-context cap), so
//! every later forward reloads nothing. Keep ONE `IrEmbedder` alive for the
//! process and reuse it; a second simultaneous session would fight over the
//! context cap.
//!
//! Pipeline per forward (all in Rust):
//!   * every convolution on the NPU via the XRT shim, in the Conv2d
//!     channel-tiled [H, C/8, W, 8] bf16 layout; wide convs are summed over
//!     input-channel groups sized to fit L1, and the width%4 stages (14x14 /
//!     7x7) zero-pad the input width to a multiple of 4 and crop the extra
//!     output columns;
//!   * the per-channel glue -- leading BN0 affine, fused conv-bias + PReLU
//!     (prelu(x + bias)), plain conv bias -- and the residual add and stride-2
//!     subsample on the CPU (these are tiny and dispatch-bound, and float32 on
//!     the host is if anything more accurate);
//!   * the embedding head (both head BatchNorms folded into the Linear) as an
//!     exact f32 mat-vec on the CPU (M=1) -- keeping it off the NPU lets all
//!     16 conv kernels stay resident within the concurrent-context cap.
//!
//! Threading: an [`IrEmbedder`] is `Send` (move it to a worker thread) but
//! deliberately not `Sync` -- all methods take `&mut self`, one forward at a
//! time. Share it behind a `Mutex` if several threads need embeddings.

use std::collections::HashMap;
use std::ffi::{c_char, c_int, c_void, CString};
use std::fmt;
use std::path::{Path, PathBuf};

use taconite_bundle::{DType, Manifest, Record, Store};

/// The bundle format version this runtime reads (`export_ir18.VERSION`).
pub const BUNDLE_VERSION: u32 = 1;

// ----------------------------------------------------------------------------
// Errors
// ----------------------------------------------------------------------------

/// Everything that can go wrong loading a bundle or running a forward. The
/// library never prints or exits; the host application decides.
#[derive(Debug)]
pub enum Error {
    /// A bundle file could not be read.
    Io(PathBuf, std::io::Error),
    /// The bundle is malformed or inconsistent: a bad manifest record, a
    /// missing or mis-sized tensor, a step reading an undefined buffer.
    Plan(String),
    /// The XRT shim reported a device/kernel error.
    Xrt(String),
    /// The caller-supplied input tensor has the wrong shape.
    Input(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(p, e) => write!(f, "cannot read {}: {e}", p.display()),
            Error::Plan(s) => write!(f, "bad bundle: {s}"),
            Error::Xrt(s) => write!(f, "XRT: {s}"),
            Error::Input(s) => write!(f, "bad input: {s}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<taconite_bundle::Error> for Error {
    fn from(e: taconite_bundle::Error) -> Self {
        match e {
            taconite_bundle::Error::Io(p, e) => Error::Io(p, e),
            taconite_bundle::Error::Format(m) => Error::Plan(m),
        }
    }
}

// ----------------------------------------------------------------------------
// FFI to the XRT shim (persistent session; LRU cache of resident hw_contexts)
// ----------------------------------------------------------------------------

#[repr(C)]
struct Session {
    _p: [u8; 0],
}

extern "C" {
    fn iron_xrt_open(device_index: c_int, err: *mut c_char, err_len: usize) -> *mut Session;
    fn iron_xrt_load(
        s: *mut Session,
        key: *const c_char,
        xclbin: *const c_char,
        insts: *const c_char,
        kernel: *const c_char,
        err: *mut c_char,
        err_len: usize,
    ) -> c_int;
    fn iron_xrt_run_conv(
        s: *mut Session,
        key: *const c_char,
        input: *const c_void,
        in_bytes: usize,
        weights: *const c_void,
        wt_bytes: usize,
        out: *mut c_void,
        out_bytes: usize,
        err: *mut c_char,
        err_len: usize,
    ) -> c_int;
    fn iron_xrt_close(s: *mut Session);
}

struct Xrt {
    s: *mut Session,
    err: Vec<i8>,
}

impl Xrt {
    fn open() -> Result<Xrt, Error> {
        let mut err = vec![0i8; 512];
        let s = unsafe { iron_xrt_open(0, err.as_mut_ptr() as *mut c_char, 512) };
        if s.is_null() {
            return Err(Error::Xrt(format!("open failed: {}", err_str(&err))));
        }
        Ok(Xrt { s, err })
    }

    /// Ensure `key`'s kernel is resident (no-op if already cached; otherwise
    /// created, evicting the LRU context if the cache is full).
    fn ensure(
        &mut self,
        key: &str,
        xclbin: &Path,
        insts: &Path,
        kernel: &str,
    ) -> Result<(), Error> {
        let ck = cstr(key)?;
        let cx = cstr(xclbin.to_str().ok_or_else(|| bad_path(xclbin))?)?;
        let ci = cstr(insts.to_str().ok_or_else(|| bad_path(insts))?)?;
        let cn = cstr(kernel)?;
        let rc = unsafe {
            iron_xrt_load(
                self.s,
                ck.as_ptr(),
                cx.as_ptr(),
                ci.as_ptr(),
                cn.as_ptr(),
                self.err.as_mut_ptr() as *mut c_char,
                512,
            )
        };
        if rc != 0 {
            return Err(Error::Xrt(format!("load({key}): {}", err_str(&self.err))));
        }
        Ok(())
    }

    /// Run a kernel with the (input, weights, output) ABI (Conv2d or GEMM).
    fn run(
        &mut self,
        key: &str,
        input: &[u16],
        weights: &[u16],
        out_len: usize,
    ) -> Result<Vec<u16>, Error> {
        let ck = cstr(key)?;
        let mut out = vec![0u16; out_len];
        let rc = unsafe {
            iron_xrt_run_conv(
                self.s,
                ck.as_ptr(),
                input.as_ptr() as *const c_void,
                input.len() * 2,
                weights.as_ptr() as *const c_void,
                weights.len() * 2,
                out.as_mut_ptr() as *mut c_void,
                out.len() * 2,
                self.err.as_mut_ptr() as *mut c_char,
                512,
            )
        };
        if rc != 0 {
            return Err(Error::Xrt(format!("run({key}): {}", err_str(&self.err))));
        }
        Ok(out)
    }
}

impl Drop for Xrt {
    fn drop(&mut self) {
        unsafe { iron_xrt_close(self.s) };
    }
}

fn cstr(s: &str) -> Result<CString, Error> {
    CString::new(s).map_err(|_| Error::Plan(format!("string contains NUL: {s}")))
}

fn bad_path(p: &Path) -> Error {
    Error::Plan(format!("non-UTF-8 path: {}", p.display()))
}

fn err_str(buf: &[i8]) -> String {
    let b: Vec<u8> = buf
        .iter()
        .take_while(|&&c| c != 0)
        .map(|&c| c as u8)
        .collect();
    String::from_utf8_lossy(&b).into_owned()
}

// ----------------------------------------------------------------------------
// bf16 helpers
// ----------------------------------------------------------------------------

#[inline]
fn bf16_to_f32(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}

#[inline]
fn f32_to_bf16(f: f32) -> u16 {
    let bits = f.to_bits();
    if (bits & 0x7fff_ffff) > 0x7f80_0000 {
        return 0x7fc0; // NaN
    }
    let rounding_bias = 0x7fff + ((bits >> 16) & 1); // round to nearest even
    ((bits + rounding_bias) >> 16) as u16
}

// ----------------------------------------------------------------------------
// Tiled buffers: flat [H, C/8, W, 8] of bf16
// ----------------------------------------------------------------------------

#[derive(Clone)]
struct Buf {
    v: Vec<u16>,
    c: usize,
    h: usize,
    w: usize,
}

#[inline]
fn tidx(c: usize, w: usize, ch: usize, y: usize, x: usize) -> usize {
    (((y * (c / 8)) + ch / 8) * w + x) * 8 + (ch % 8)
}

/// Extract input channel-groups [g0, g0+ngroups) into a fresh tiled buffer,
/// zero-padding the width from `src.w` up to `wp` (a multiple of 4) so the
/// width%4 conv shapes are feasible. Columns x >= src.w stay zero.
fn slice_groups_wpad(src: &Buf, g0: usize, ngroups: usize, wp: usize) -> Vec<u16> {
    let icg = ngroups * 8;
    let mut out = vec![0u16; src.h * icg * wp];
    for y in 0..src.h {
        for gg in 0..ngroups {
            for x in 0..src.w {
                for lane in 0..8 {
                    let s = (((y * (src.c / 8)) + (g0 + gg)) * src.w + x) * 8 + lane;
                    let d = (((y * ngroups) + gg) * wp + x) * 8 + lane;
                    out[d] = src.v[s];
                }
            }
        }
    }
    out
}

/// Leading BN0 as a per-channel affine scale*x + shift (tiled -> new tiled).
fn affine_tiled(src: &Buf, scale: &[u16], shift: &[u16]) -> Buf {
    let mut v = vec![0u16; src.v.len()];
    for y in 0..src.h {
        for ch in 0..src.c {
            for x in 0..src.w {
                let i = tidx(src.c, src.w, ch, y, x);
                let r = bf16_to_f32(src.v[i]) * bf16_to_f32(scale[ch]) + bf16_to_f32(shift[ch]);
                v[i] = f32_to_bf16(r);
            }
        }
    }
    Buf {
        v,
        c: src.c,
        h: src.h,
        w: src.w,
    }
}

/// Fused per-channel bias add + PReLU in place: y = prelu(x + bias). The bias
/// add rounds to bf16 before the PReLU, matching the recorder's two host ops.
fn prelu_bias_tiled(buf: &mut Buf, alpha: &[u16], bias: &[u16]) {
    for y in 0..buf.h {
        for ch in 0..buf.c {
            for x in 0..buf.w {
                let i = tidx(buf.c, buf.w, ch, y, x);
                let t = bf16_to_f32(f32_to_bf16(bf16_to_f32(buf.v[i]) + bf16_to_f32(bias[ch])));
                let r = if t < 0.0 {
                    bf16_to_f32(alpha[ch]) * t
                } else {
                    t
                };
                buf.v[i] = f32_to_bf16(r);
            }
        }
    }
}

/// Per-channel bias add in place.
fn bias_tiled(buf: &mut Buf, bias: &[u16]) {
    for y in 0..buf.h {
        for ch in 0..buf.c {
            for x in 0..buf.w {
                let i = tidx(buf.c, buf.w, ch, y, x);
                buf.v[i] = f32_to_bf16(bf16_to_f32(buf.v[i]) + bf16_to_f32(bias[ch]));
            }
        }
    }
}

fn add_tiled(a: &Buf, b: &Buf) -> Buf {
    let v =
        a.v.iter()
            .zip(&b.v)
            .map(|(&x, &y)| f32_to_bf16(bf16_to_f32(x) + bf16_to_f32(y)))
            .collect();
    Buf {
        v,
        c: a.c,
        h: a.h,
        w: a.w,
    }
}

fn subsample_tiled(src: &Buf, stride: usize) -> Buf {
    let (oh, ow) = (src.h / stride, src.w / stride);
    let mut out = vec![0u16; oh * (src.c / 8) * ow * 8];
    for y in 0..oh {
        for g in 0..(src.c / 8) {
            for x in 0..ow {
                for lane in 0..8 {
                    let s = (((stride * y) * (src.c / 8) + g) * src.w + stride * x) * 8 + lane;
                    let d = ((y * (src.c / 8) + g) * ow + x) * 8 + lane;
                    out[d] = src.v[s];
                }
            }
        }
    }
    Buf {
        v: out,
        c: src.c,
        h: oh,
        w: ow,
    }
}

/// Convert a tiled [h, c/8, w, 8] bf16 buffer to channel-major f32 [c, h, w].
fn tiled_to_nchw_f32(b: &Buf) -> Vec<f32> {
    let mut out = vec![0f32; b.c * b.h * b.w];
    for ch in 0..b.c {
        for y in 0..b.h {
            for x in 0..b.w {
                out[(ch * b.h + y) * b.w + x] = bf16_to_f32(b.v[tidx(b.c, b.w, ch, y, x)]);
            }
        }
    }
    out
}

/// Cosine similarity between two embeddings (0.0 if either is all-zero).
pub fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        dot += x as f64 * y as f64;
        na += x as f64 * x as f64;
        nb += y as f64 * y as f64;
    }
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na.sqrt() * nb.sqrt())
    }
}

// ----------------------------------------------------------------------------
// The bundle's steps
// ----------------------------------------------------------------------------

/// One NPU dispatch: the input-channel group `[8 * c0, 8 * c0 + icg)` of a
/// convolution, accumulated into `out` from the `first` dispatch to the
/// `last` (which crops the padded width `owp` back to `ow`).
struct Conv {
    input: u32,
    out: u32,
    c0: usize,
    icg: usize,
    wp: usize,
    oc: usize,
    oh: usize,
    ow: usize,
    owp: usize,
    ctx: String,
    xclbin: PathBuf,
    kernel: String,
    insts: PathBuf,
    wt: String,
    first: bool,
    last: bool,
}

/// One manifest step, parsed and checked against the tensor store at load.
/// Buffer ids name tiled bf16 buffers; `String`s name tensors.
enum Step {
    Input {
        buf: u32,
        c: usize,
        h: usize,
        w: usize,
        tensor: String,
    },
    Conv(Conv),
    Affine {
        input: u32,
        out: u32,
        scale: String,
        shift: String,
    },
    PreluBias {
        buf: u32,
        alpha: String,
        bias: String,
    },
    Bias {
        buf: u32,
        bias: String,
    },
    Subsample {
        input: u32,
        out: u32,
        stride: usize,
    },
    Add {
        a: u32,
        b: u32,
        out: u32,
    },
    Head {
        input: u32,
        k: usize,
        n: usize,
        bt: String,
        b2: String,
    },
}

/// (channels, height, width) of a buffer.
type Shape = (usize, usize, usize);

/// A buffer a step reads: `key=<id>`, which an earlier step must define.
fn read(shapes: &HashMap<u32, Shape>, r: &Record, key: &str) -> Result<(u32, Shape), Error> {
    let id: u32 = r.get(key)?;
    match shapes.get(&id) {
        Some(&s) => Ok((id, s)),
        None => Err(r
            .error(format!("{key}={id} is not a defined buffer here"))
            .into()),
    }
}

fn dims(r: &Record) -> Result<Shape, Error> {
    Ok((r.get("c")?, r.get("h")?, r.get("w")?))
}

fn same(r: &Record, what: &str, got: Shape, want: Shape) -> Result<(), Error> {
    if got != want {
        return Err(r
            .error(format!(
                "{what} is {got:?} (c, h, w), the step says {want:?}"
            ))
            .into());
    }
    Ok(())
}

/// Tensor `key=<name>`, checked to be `elems` x `dtype`.
fn tensor(
    store: &Store,
    r: &Record,
    key: &str,
    dtype: DType,
    elems: usize,
) -> Result<String, Error> {
    let name = r.str(key)?;
    store.expect(name, dtype, elems).map_err(|e| r.error(e))?;
    Ok(name.to_string())
}

/// Parse the manifest's steps, checking each against the buffers the steps
/// before it define and against the tensors, so a forward cannot fail on
/// a malformed bundle.
fn parse_steps(m: &Manifest, store: &Store) -> Result<Vec<Step>, Error> {
    let mut shapes: HashMap<u32, Shape> = HashMap::new();
    let mut steps = Vec::new();
    for r in m.records() {
        let step = match r.tag.as_str() {
            "input" => {
                let (c, h, w) = dims(r)?;
                if c == 0 || c % 8 != 0 || h == 0 || w == 0 {
                    return Err(r
                        .error("c must be a positive multiple of 8, h and w positive")
                        .into());
                }
                let tensor = tensor(store, r, "tensor", DType::Bf16, c * h * w)?;
                let buf = r.get("buf")?;
                shapes.insert(buf, (c, h, w));
                Step::Input {
                    buf,
                    c,
                    h,
                    w,
                    tensor,
                }
            }
            "conv" => {
                let (input, s) = read(&shapes, r, "in")?;
                let (ict, h, w): Shape = (r.get("ict")?, r.get("h")?, r.get("w")?);
                same(r, "the input", s, (ict, h, w))?;
                let (c0, icg, wp): (usize, usize, usize) =
                    (r.get("c0")?, r.get("icg")?, r.get("wp")?);
                let (oc, oh, ow, owp): (usize, usize, usize, usize) =
                    (r.get("oc")?, r.get("oh")?, r.get("ow")?, r.get("owp")?);
                if icg == 0
                    || icg % 8 != 0
                    || 8 * c0 + icg > ict
                    || oc % 8 != 0
                    || wp < w
                    || owp < ow
                {
                    return Err(r
                        .error("inconsistent channel group or padded widths")
                        .into());
                }
                let x = m.xclbin(r.str("ctx")?).map_err(|e| r.error(e))?;
                let wt = r.str("wt")?;
                let e = store.entry(wt).map_err(|e| r.error(e))?;
                if e.dtype != DType::Bf16 {
                    return Err(r
                        .error(format!("tensor {wt} is {:?}, wanted Bf16", e.dtype))
                        .into());
                }
                let (out, last) = (r.get("out")?, r.flag("last")?);
                if last {
                    shapes.insert(out, (oc, oh, ow));
                }
                Step::Conv(Conv {
                    input,
                    out,
                    c0,
                    icg,
                    wp,
                    oc,
                    oh,
                    ow,
                    owp,
                    ctx: x.key.clone(),
                    xclbin: x.path.clone(),
                    kernel: x.kernel.clone(),
                    insts: m.path(r.str("insts")?),
                    wt: wt.to_string(),
                    first: r.flag("first")?,
                    last,
                })
            }
            "affine" => {
                let (input, s) = read(&shapes, r, "in")?;
                let d = dims(r)?;
                same(r, "the input", s, d)?;
                let scale = tensor(store, r, "scale", DType::Bf16, d.0)?;
                let shift = tensor(store, r, "shift", DType::Bf16, d.0)?;
                let out = r.get("out")?;
                shapes.insert(out, d);
                Step::Affine {
                    input,
                    out,
                    scale,
                    shift,
                }
            }
            "prelu_bias" => {
                let (buf, s) = read(&shapes, r, "buf")?;
                same(r, "the buffer", s, dims(r)?)?;
                let alpha = tensor(store, r, "alpha", DType::Bf16, s.0)?;
                let bias = tensor(store, r, "bias", DType::Bf16, s.0)?;
                Step::PreluBias { buf, alpha, bias }
            }
            "bias" => {
                let (buf, s) = read(&shapes, r, "buf")?;
                same(r, "the buffer", s, dims(r)?)?;
                let bias = tensor(store, r, "bias", DType::Bf16, s.0)?;
                Step::Bias { buf, bias }
            }
            "subsample" => {
                let (input, s) = read(&shapes, r, "in")?;
                let (c, h, w) = dims(r)?;
                same(r, "the input", s, (c, h, w))?;
                let stride: usize = r.get("stride")?;
                if stride == 0 {
                    return Err(r.error("stride must be positive").into());
                }
                let out = r.get("out")?;
                shapes.insert(out, (c, h / stride, w / stride));
                Step::Subsample { input, out, stride }
            }
            "add" => {
                let (a, sa) = read(&shapes, r, "a")?;
                let (b, sb) = read(&shapes, r, "b")?;
                let d = dims(r)?;
                same(r, "a", sa, d)?;
                same(r, "b", sb, d)?;
                let out = r.get("out")?;
                shapes.insert(out, d);
                Step::Add { a, b, out }
            }
            "head" => {
                let (input, s) = read(&shapes, r, "in")?;
                let d = dims(r)?;
                same(r, "the input", s, d)?;
                let (k, n): (usize, usize) = (r.get("k")?, r.get("n")?);
                if d.0 * d.1 * d.2 != k {
                    return Err(r
                        .error(format!("k={k} is not the input's c * h * w"))
                        .into());
                }
                let bt = tensor(store, r, "bT", DType::Bf16, k * n)?;
                let b2 = tensor(store, r, "b2", DType::F32, n)?;
                Step::Head {
                    input,
                    k,
                    n,
                    bt,
                    b2,
                }
            }
            other => return Err(r.error(format!("unknown step {other}")).into()),
        };
        steps.push(step);
    }
    Ok(steps)
}

// ----------------------------------------------------------------------------
// The embedder: replay the recorded IR steps on the NPU
// ----------------------------------------------------------------------------

/// A resident NPU session replaying one exported IR bundle.
///
/// Construct once with [`IrEmbedder::load`], then call [`IrEmbedder::embed`]
/// per face. The first forward creates the hardware contexts; every later
/// forward reuses them (warm path).
pub struct IrEmbedder {
    xrt: Xrt,
    store: Store,
    steps: Vec<Step>,
    /// Each head's `bT`, widened to f32 once.
    head_bt: HashMap<String, Vec<f32>>,
    model: Option<String>,
    in_c: usize, // stem channels as recorded (image channels padded to %8)
    in_h: usize,
    in_w: usize,
    embed_dim: usize,
}

// SAFETY: the session pointer owns a heap object in the C++ shim with no
// thread-affine state (XRT device/context handles may be used from any
// thread), and every method takes &mut self, so calls are serialized. The
// type is deliberately NOT Sync: wrap it in a Mutex to share across threads.
unsafe impl Send for IrEmbedder {}

impl IrEmbedder {
    /// Read and check the bundle (manifest, every tensor), then open the NPU
    /// device. Hardware contexts load lazily on the first forward.
    pub fn load(bundle_dir: impl AsRef<Path>) -> Result<Self, Error> {
        let dir = bundle_dir.as_ref();
        if !dir.join(Manifest::FILE).exists() && dir.join("plan.txt").exists() {
            return Err(Error::Plan(format!(
                "{} is an old plan.txt bundle; re-export it with export_ir18.py / export_ir101.py",
                dir.display()
            )));
        }
        let m = Manifest::load(dir, BUNDLE_VERSION)?;
        let store = Store::load(dir)?;
        let steps = parse_steps(&m, &store)?;
        let inputs: Vec<Shape> = steps
            .iter()
            .filter_map(|s| match s {
                Step::Input { c, h, w, .. } => Some((*c, *h, *w)),
                _ => None,
            })
            .collect();
        let [(in_c, in_h, in_w)] = inputs[..] else {
            return Err(Error::Plan(
                "the bundle needs exactly one input step".into(),
            ));
        };
        let mut head_bt = HashMap::new();
        let mut embed_dim = None;
        for s in &steps {
            if let Step::Head { bt, n, .. } = s {
                embed_dim = Some(*n);
                if !head_bt.contains_key(bt) {
                    let v: Vec<f32> = store.bf16(bt)?.iter().map(|&x| bf16_to_f32(x)).collect();
                    head_bt.insert(bt.clone(), v);
                }
            }
        }
        if !matches!(steps.last(), Some(Step::Head { .. })) {
            return Err(Error::Plan("the bundle's last step is not a head".into()));
        }
        Ok(IrEmbedder {
            xrt: Xrt::open()?,
            store,
            steps,
            head_bt,
            model: m.param("model").ok().map(str::to_string),
            in_c,
            in_h,
            in_w,
            embed_dim: embed_dim.unwrap_or(0),
        })
    }

    /// Stem input dims as recorded in the bundle: (channels-padded-to-%8, h, w).
    pub fn input_dims(&self) -> (usize, usize, usize) {
        (self.in_c, self.in_h, self.in_w)
    }

    /// Output embedding dimension (512 for the AdaFace IR models).
    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    /// The network the bundle holds (its `model` param, e.g. `IR-101`).
    pub fn model(&self) -> Option<&str> {
        self.model.as_deref()
    }

    /// NPU conv dispatches per forward (a fixed property of the bundle).
    pub fn num_conv_dispatches(&self) -> usize {
        self.steps
            .iter()
            .filter(|s| matches!(s, Step::Conv(_)))
            .count()
    }

    /// Embed a caller-supplied image: channel-major (CHW) f32, spatial size
    /// exactly the bundle's h*w, 1..=in_c channels (an RGB face is 3; the
    /// remaining stem channels are zero -- the recorded conv weights for them
    /// are zero too, so this is exact). For the AdaFace models the input is
    /// an aligned 112x112 face with cvlface preprocessing
    /// `((rgb/255) - 0.5) / 0.5`. Values round to bf16 (round-to-nearest-even,
    /// bit-identical to torch) before hitting the NPU.
    pub fn embed(&mut self, chw: &[f32]) -> Result<Vec<f32>, Error> {
        let plane = self.in_h * self.in_w;
        if chw.is_empty() || chw.len() % plane != 0 {
            return Err(Error::Input(format!(
                "length {} is not a multiple of h*w = {plane}",
                chw.len()
            )));
        }
        let ch = chw.len() / plane;
        if ch > self.in_c {
            return Err(Error::Input(format!(
                "{ch} channels, but the bundle's stem takes at most {}",
                self.in_c
            )));
        }
        let mut tiled = vec![0u16; self.in_h * (self.in_c / 8) * self.in_w * 8];
        for c in 0..ch {
            for y in 0..self.in_h {
                for x in 0..self.in_w {
                    tiled[tidx(self.in_c, self.in_w, c, y, x)] =
                        f32_to_bf16(chw[(c * self.in_h + y) * self.in_w + x]);
                }
            }
        }
        self.run(Some(&tiled))
    }

    /// Embed the bundle's own recorded input (`ref.input`) -- a self-check
    /// that must match `ref.npu_embedding` exactly, and the benchmark workload.
    pub fn embed_recorded(&mut self) -> Result<Vec<f32>, Error> {
        self.run(None)
    }

    fn run(&mut self, input: Option<&[u16]>) -> Result<Vec<f32>, Error> {
        let Self {
            xrt,
            store,
            steps,
            head_bt,
            ..
        } = self;
        forward(xrt, store, steps, head_bt, input)
    }
}

/// One forward pass over the parsed steps. NPU convs dispatch through the
/// resident-context session `xrt`; the per-channel glue and residual add run
/// on the host. `input` overrides the input step's recorded tensor with a
/// caller-built tiled buffer. Every buffer a step reads exists (checked at
/// load), so the `bufs[..]` lookups cannot fail.
fn forward(
    xrt: &mut Xrt,
    store: &Store,
    steps: &[Step],
    head_bt: &HashMap<String, Vec<f32>>,
    input: Option<&[u16]>,
) -> Result<Vec<f32>, Error> {
    let mut bufs: HashMap<u32, Buf> = HashMap::new();
    let mut acc: HashMap<u32, Vec<f32>> = HashMap::new();
    let mut embed: Vec<f32> = Vec::new();

    for step in steps {
        match step {
            Step::Input {
                buf,
                c,
                h,
                w,
                tensor,
            } => {
                let (c, h, w) = (*c, *h, *w);
                let v = match input {
                    Some(t) => {
                        if t.len() != h * (c / 8) * w * 8 {
                            return Err(Error::Input(format!(
                                "tiled input length {} != {}",
                                t.len(),
                                h * (c / 8) * w * 8
                            )));
                        }
                        t.to_vec()
                    }
                    None => store.bf16(tensor)?.to_vec(),
                };
                bufs.insert(*buf, Buf { v, c, h, w });
            }
            Step::Conv(cv) => {
                let (oc, oh, ow, owp) = (cv.oc, cv.oh, cv.ow, cv.owp);
                xrt.ensure(&cv.ctx, &cv.xclbin, &cv.insts, &cv.kernel)?;
                let inp = slice_groups_wpad(&bufs[&cv.input], cv.c0, cv.icg / 8, cv.wp);
                let part = xrt.run(&cv.ctx, &inp, store.bf16(&cv.wt)?, oc * oh * owp)?;
                let a = acc
                    .entry(cv.out)
                    .or_insert_with(|| vec![0f32; oc * oh * owp]);
                if cv.first {
                    a.iter_mut().for_each(|x| *x = 0.0);
                }
                for (dst, &p) in a.iter_mut().zip(&part) {
                    *dst += bf16_to_f32(p);
                }
                if cv.last {
                    // Crop the padded output width owp -> ow (tiled layout).
                    let mut v = vec![0u16; oh * (oc / 8) * ow * 8];
                    for y in 0..oh {
                        for g in 0..(oc / 8) {
                            for x in 0..ow {
                                for lane in 0..8 {
                                    let si = (((y * (oc / 8)) + g) * owp + x) * 8 + lane;
                                    let di = (((y * (oc / 8)) + g) * ow + x) * 8 + lane;
                                    v[di] = f32_to_bf16(a[si]);
                                }
                            }
                        }
                    }
                    bufs.insert(
                        cv.out,
                        Buf {
                            v,
                            c: oc,
                            h: oh,
                            w: ow,
                        },
                    );
                }
            }
            Step::Affine {
                input,
                out,
                scale,
                shift,
            } => {
                let r = affine_tiled(&bufs[input], store.bf16(scale)?, store.bf16(shift)?);
                bufs.insert(*out, r);
            }
            Step::PreluBias { buf, alpha, bias } => {
                let buf = bufs.get_mut(buf).expect("checked at load");
                prelu_bias_tiled(buf, store.bf16(alpha)?, store.bf16(bias)?);
            }
            Step::Bias { buf, bias } => {
                let buf = bufs.get_mut(buf).expect("checked at load");
                bias_tiled(buf, store.bf16(bias)?);
            }
            Step::Subsample { input, out, stride } => {
                let r = subsample_tiled(&bufs[input], *stride);
                bufs.insert(*out, r);
            }
            Step::Add { a, b, out } => {
                let r = add_tiled(&bufs[a], &bufs[b]);
                bufs.insert(*out, r);
            }
            Step::Head {
                input,
                k,
                n,
                bt,
                b2,
            } => {
                // Embedding head on the CPU: an exact f32 mat-vec with the
                // folded head weights (both head BatchNorms folded into the
                // Linear). M=1, so this is far cheaper than the padded M=32 NPU
                // GEMM, and it keeps the head off the NPU so all 16 conv kernels
                // stay resident (the NPU2 concurrent-context cap is 16).
                let (k, n) = (*k, *n);
                let flat = tiled_to_nchw_f32(&bufs[input]); // [k]
                let bt = &head_bt[bt]; // [k * n], row-major [k][n]
                let mut e = store.f32(b2)?.to_vec(); // [n], folded bias
                for i in 0..k {
                    let f = flat[i];
                    let row = &bt[i * n..(i + 1) * n];
                    for j in 0..n {
                        e[j] += f * row[j];
                    }
                }
                embed = e;
            }
        }
    }
    Ok(embed)
}

// ----------------------------------------------------------------------------
// CLI (the run_ir18 / run_ir101 binaries)
// ----------------------------------------------------------------------------

/// Full CLI entry point, shared by the per-network binaries (run_ir18,
/// run_ir101). `net_name` labels the output when the bundle has no `model`
/// param; the replayed network is whatever the bundle contains. Prints
/// per-forward latency (cold vs warm) and the embedding's cosine to the CPU
/// reference (`ref.cpu_embedding`) and the recorder's embedding
/// (`ref.npu_embedding`); when the bundle has the raw f32 input
/// (`ref.input_chw`), also exercises the public [`IrEmbedder::embed`] API and
/// checks it reproduces the recorded replay.
pub fn cli(net_name: &str) {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 2 {
        let prog = a
            .first()
            .map(|p| p.rsplit('/').next().unwrap_or(p).to_string())
            .unwrap_or_else(|| "run_ir".to_string());
        eprintln!("usage: {prog} <bundle_dir> [reps]");
        std::process::exit(2);
    }
    let dir = PathBuf::from(&a[1]);
    // Repeated forwards demonstrate steady-state throughput once every kernel's
    // hw-context is resident (reps from arg 2 or $NREPS, default 1).
    let reps: usize = a
        .get(2)
        .cloned()
        .or_else(|| std::env::var("NREPS").ok())
        .and_then(|s| s.parse().ok())
        .unwrap_or(1)
        .max(1);
    if let Err(e) = cli_run(net_name, &dir, reps) {
        eprintln!("{e}");
        std::process::exit(1);
    }
}

fn cli_run(net_name: &str, dir: &Path, reps: usize) -> Result<(), Error> {
    use std::time::Instant;

    let mut emb = IrEmbedder::load(dir)?;
    let mut embed = Vec::new();
    let mut cold = 0f64;
    let mut warm_sum = 0f64;
    for i in 0..reps {
        let t = Instant::now();
        embed = emb.embed_recorded()?;
        let dt = t.elapsed().as_secs_f64() * 1e3;
        if i == 0 {
            cold = dt;
        } else {
            warm_sum += dt;
        }
    }

    println!(
        "{} forward: {} conv dispatches on the NPU + CPU head",
        emb.model().unwrap_or(net_name),
        emb.num_conv_dispatches()
    );
    println!("embedding: {} dims", embed.len());
    if reps == 1 {
        println!("latency: {cold:.0} ms");
    } else {
        println!(
            "latency: cold {:.0} ms, warm {:.0} ms (mean of {} reps; contexts resident)",
            cold,
            warm_sum / (reps - 1) as f64,
            reps - 1
        );
    }

    // Self-check against the recorder's own embedding (should be ~identical).
    if let Ok(exp) = emb.store.f32("ref.npu_embedding") {
        println!(
            "cosine(Rust, recorder embedding)     = {:.5}",
            cosine(&embed, exp)
        );
    }
    // Exercise the public embed() API: feeding the raw f32 image through the
    // library's pad + bf16-round + tile path must reproduce the recorded
    // replay bit-for-bit (torch and the library both round to nearest even).
    if let Ok(chw) = emb.store.f32("ref.input_chw") {
        let chw = chw.to_vec();
        let e2 = emb.embed(&chw)?;
        println!(
            "cosine(embed() API, recorded replay) = {:.5}",
            cosine(&e2, &embed)
        );
    }
    // Ground-truth check against the CPU reference embedding.
    if let Ok(r) = emb.store.f32("ref.cpu_embedding") {
        let cos = cosine(&embed, r);
        println!("cosine(Rust, CPU reference)          = {cos:.5}");
        println!(
            "{}",
            if cos > 0.99 {
                "RESULT: MATCH (Rust embedding agrees with the CPU reference)"
            } else {
                "RESULT: embedding differs from the CPU reference"
            }
        );
    }
    Ok(())
}
