// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Embeddable Rust runtime for the AdaFace IR face embedders: NPU
//! convolutions + CPU glue/head, no Python.
//!
//! [`IrEmbedder`] replays a bundle exported by
//! `iron/applications/adaface_ir18/export_ir18.py` (or the IR-101 wrapper
//! `export_ir101.py`). The plan format is architecture-agnostic, so this one
//! runtime serves every exported IR backbone; the `run_ir18` / `run_ir101`
//! binaries are thin wrappers over [`cli`], which is itself a consumer of the
//! public API:
//!
//! ```no_run
//! use adaface_ir_runtime::IrEmbedder;
//!
//! # fn main() -> Result<(), adaface_ir_runtime::Error> {
//! let mut emb = IrEmbedder::load("/path/to/bundle")?;
//! // An aligned 112x112 face crop, CHW f32, cvlface preprocessing
//! // ((rgb/255) - 0.5) / 0.5. Channels beyond the image's (the plan pads the
//! // stem to a multiple of 8) are zero-filled internally.
//! let chw = vec![0f32; 3 * 112 * 112];
//! let embedding = emb.embed(&chw)?; // 512-d
//! # let _ = embedding;
//! # Ok(())
//! # }
//! ```
//!
//! The first forward creates one XRT hardware context per distinct conv
//! kernel (16 for IR-18/IR-101) and the session keeps them ALL resident (LRU
//! cache in the C++ shim, sized to the NPU2's 16-concurrent-context cap), so
//! every later forward reloads nothing. Weights/params are cached in RAM on
//! first use. Keep ONE `IrEmbedder` alive for the process and reuse it; a
//! second simultaneous session would fight over the context cap.
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

// ----------------------------------------------------------------------------
// Errors
// ----------------------------------------------------------------------------

/// Everything that can go wrong loading a bundle or running a forward. The
/// library never prints or exits; the host application decides.
#[derive(Debug)]
pub enum Error {
    /// A bundle file could not be read.
    Io(PathBuf, std::io::Error),
    /// plan.txt is malformed or references an undefined buffer.
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
            Error::Plan(s) => write!(f, "bad plan: {s}"),
            Error::Xrt(s) => write!(f, "XRT: {s}"),
            Error::Input(s) => write!(f, "bad input: {s}"),
        }
    }
}

impl std::error::Error for Error {}

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
    fn ensure(&mut self, key: &str, xclbin: &Path, insts: &Path, kernel: &str) -> Result<(), Error> {
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

fn read_bf16(path: &Path) -> Result<Vec<u16>, Error> {
    let bytes = std::fs::read(path).map_err(|e| Error::Io(path.to_path_buf(), e))?;
    Ok(bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
}

fn read_f32(path: &Path) -> Result<Vec<f32>, Error> {
    let bytes = std::fs::read(path).map_err(|e| Error::Io(path.to_path_buf(), e))?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
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

// ----------------------------------------------------------------------------
// Plan parsing
// ----------------------------------------------------------------------------

fn kv(line: &str) -> (String, HashMap<String, String>) {
    let mut it = line.split_whitespace();
    let op = it.next().unwrap_or("").to_string();
    let mut m = HashMap::new();
    for tok in it {
        if let Some((k, v)) = tok.split_once('=') {
            m.insert(k.to_string(), v.to_string());
        }
    }
    (op, m)
}

fn gets<'a>(m: &'a HashMap<String, String>, k: &str) -> Result<&'a str, Error> {
    m.get(k)
        .map(|s| s.as_str())
        .ok_or_else(|| Error::Plan(format!("missing key {k}")))
}

fn geti(m: &HashMap<String, String>, k: &str) -> Result<i64, Error> {
    gets(m, k)?
        .parse()
        .map_err(|_| Error::Plan(format!("bad integer for key {k}")))
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
// The embedder: replay the recorded IR plan on the NPU
// ----------------------------------------------------------------------------

type Op = (String, HashMap<String, String>);

/// Load a bf16 blob into `cache` if not already present (weights/params are
/// reused verbatim across forwards, so they are read from disk only once).
fn cache_bf16<'a>(
    cache: &'a mut HashMap<String, Vec<u16>>,
    dir: &Path,
    name: &str,
) -> Result<&'a [u16], Error> {
    if !cache.contains_key(name) {
        cache.insert(name.to_string(), read_bf16(&dir.join(name))?);
    }
    Ok(&cache[name])
}

fn getbuf<'a>(bufs: &'a HashMap<u32, Buf>, id: u32, op: &str) -> Result<&'a Buf, Error> {
    bufs.get(&id)
        .ok_or_else(|| Error::Plan(format!("{op}: buffer {id} undefined")))
}

/// A resident NPU session replaying one exported IR bundle.
///
/// Construct once with [`IrEmbedder::load`], then call [`IrEmbedder::embed`]
/// per face. The first forward creates the hardware contexts and loads the
/// weights from disk; every later forward reuses all of it (warm path).
pub struct IrEmbedder {
    xrt: Xrt,
    dir: PathBuf,
    ops: Vec<Op>,
    bcache: HashMap<String, Vec<u16>>,
    fcache: HashMap<String, Vec<f32>>,
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
    /// Parse `<bundle>/plan.txt` and open the NPU device. Hardware contexts
    /// and weights load lazily on the first forward.
    pub fn load(bundle_dir: impl AsRef<Path>) -> Result<Self, Error> {
        let dir = bundle_dir.as_ref().to_path_buf();
        let plan_path = dir.join("plan.txt");
        let plan =
            std::fs::read_to_string(&plan_path).map_err(|e| Error::Io(plan_path.clone(), e))?;
        let ops: Vec<Op> = plan
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(kv)
            .collect();
        let (mut in_c, mut in_h, mut in_w, mut embed_dim) = (0usize, 0usize, 0usize, 0usize);
        for (op, m) in &ops {
            if op == "INPUT" {
                in_c = geti(m, "c")? as usize;
                in_h = geti(m, "h")? as usize;
                in_w = geti(m, "w")? as usize;
            } else if op == "HEAD" {
                embed_dim = geti(m, "n")? as usize;
            }
        }
        if in_c == 0 || in_c % 8 != 0 || in_h == 0 || in_w == 0 {
            return Err(Error::Plan("no valid INPUT op".to_string()));
        }
        if embed_dim == 0 {
            return Err(Error::Plan("no HEAD op".to_string()));
        }
        Ok(IrEmbedder {
            xrt: Xrt::open()?,
            dir,
            ops,
            bcache: HashMap::new(),
            fcache: HashMap::new(),
            in_c,
            in_h,
            in_w,
            embed_dim,
        })
    }

    /// Stem input dims as recorded in the plan: (channels-padded-to-%8, h, w).
    pub fn input_dims(&self) -> (usize, usize, usize) {
        (self.in_c, self.in_h, self.in_w)
    }

    /// Output embedding dimension (512 for the AdaFace IR models).
    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    /// NPU conv dispatches per forward (a fixed property of the plan).
    pub fn num_conv_dispatches(&self) -> usize {
        self.ops.iter().filter(|(o, _)| o == "CONV").count()
    }

    /// Embed a caller-supplied image: channel-major (CHW) f32, spatial size
    /// exactly the plan's h*w, 1..=in_c channels (an RGB face is 3; the
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
                "{ch} channels, but the plan's stem takes at most {}",
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

    /// Embed the bundle's own recorded input (`input.bin`) -- a self-check
    /// that must match `expected.bin` exactly, and the benchmark workload.
    pub fn embed_recorded(&mut self) -> Result<Vec<f32>, Error> {
        self.run(None)
    }

    fn run(&mut self, input: Option<&[u16]>) -> Result<Vec<f32>, Error> {
        let Self {
            xrt,
            dir,
            ops,
            bcache,
            fcache,
            ..
        } = self;
        forward_ops(xrt, dir, ops, bcache, fcache, input)
    }
}

/// One forward pass over the parsed plan. NPU convs dispatch through the
/// resident-context session `xrt`; the per-channel glue and residual add run
/// on the host. Blobs are served from `bcache`/`fcache` (populated on the
/// first pass), so repeated forwards touch the disk for nothing. `input`
/// overrides the INPUT op's recorded file with a caller-built tiled buffer.
fn forward_ops(
    xrt: &mut Xrt,
    dir: &Path,
    ops: &[Op],
    bcache: &mut HashMap<String, Vec<u16>>,
    fcache: &mut HashMap<String, Vec<f32>>,
    input: Option<&[u16]>,
) -> Result<Vec<f32>, Error> {
    let mut bufs: HashMap<u32, Buf> = HashMap::new();
    let mut acc: HashMap<u32, Vec<f32>> = HashMap::new();
    let mut embed: Vec<f32> = Vec::new();

    for (op, m) in ops {
        match op.as_str() {
            "INPUT" => {
                let (c, h, w) = (
                    geti(m, "c")? as usize,
                    geti(m, "h")? as usize,
                    geti(m, "w")? as usize,
                );
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
                    None => cache_bf16(bcache, dir, gets(m, "file")?)?.to_vec(),
                };
                bufs.insert(geti(m, "buf")? as u32, Buf { v, c, h, w });
            }
            "CONV" => {
                let in_id = geti(m, "in")? as u32;
                let out_id = geti(m, "out")? as u32;
                let c0 = geti(m, "c0")? as usize;
                let icg = geti(m, "icg")? as usize;
                let (oc, oh) = (geti(m, "oc")? as usize, geti(m, "oh")? as usize);
                let (ow, owp) = (geti(m, "ow")? as usize, geti(m, "owp")? as usize);
                let wp = geti(m, "wp")? as usize;
                let key = gets(m, "xclbin")?;
                xrt.ensure(
                    key,
                    &dir.join(key),
                    &dir.join(gets(m, "insts")?),
                    gets(m, "kernel")?,
                )?;
                let inp = slice_groups_wpad(getbuf(&bufs, in_id, "CONV")?, c0, icg / 8, wp);
                let wt = cache_bf16(bcache, dir, gets(m, "wt")?)?;
                let part = xrt.run(key, &inp, wt, oc * oh * owp)?;
                let a = acc
                    .entry(out_id)
                    .or_insert_with(|| vec![0f32; oc * oh * owp]);
                if geti(m, "first")? == 1 {
                    a.iter_mut().for_each(|x| *x = 0.0);
                }
                for (dst, &p) in a.iter_mut().zip(&part) {
                    *dst += bf16_to_f32(p);
                }
                if geti(m, "last")? == 1 {
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
                        out_id,
                        Buf {
                            v,
                            c: oc,
                            h: oh,
                            w: ow,
                        },
                    );
                }
            }
            "AFFINE" => {
                let in_id = geti(m, "in")? as u32;
                let out_id = geti(m, "out")? as u32;
                let scale = gets(m, "scale")?.to_string();
                let shift = gets(m, "shift")?.to_string();
                cache_bf16(bcache, dir, &scale)?;
                cache_bf16(bcache, dir, &shift)?;
                let r = affine_tiled(
                    getbuf(&bufs, in_id, "AFFINE")?,
                    &bcache[&scale],
                    &bcache[&shift],
                );
                bufs.insert(out_id, r);
            }
            "PRELU_BIAS" => {
                let id = geti(m, "buf")? as u32;
                let alpha = cache_bf16(bcache, dir, gets(m, "alpha")?)?.to_vec();
                let bias = cache_bf16(bcache, dir, gets(m, "bias")?)?.to_vec();
                let buf = bufs
                    .get_mut(&id)
                    .ok_or_else(|| Error::Plan(format!("PRELU_BIAS: buffer {id} undefined")))?;
                prelu_bias_tiled(buf, &alpha, &bias);
            }
            "BIAS" => {
                let id = geti(m, "buf")? as u32;
                let bias = cache_bf16(bcache, dir, gets(m, "bias")?)?.to_vec();
                let buf = bufs
                    .get_mut(&id)
                    .ok_or_else(|| Error::Plan(format!("BIAS: buffer {id} undefined")))?;
                bias_tiled(buf, &bias);
            }
            "SUBSAMPLE" => {
                let (in_id, out_id) = (geti(m, "in")? as u32, geti(m, "out")? as u32);
                let r = subsample_tiled(
                    getbuf(&bufs, in_id, "SUBSAMPLE")?,
                    geti(m, "stride")? as usize,
                );
                bufs.insert(out_id, r);
            }
            "ADD" => {
                let (a_id, b_id, out_id) = (
                    geti(m, "a")? as u32,
                    geti(m, "b")? as u32,
                    geti(m, "out")? as u32,
                );
                let r = add_tiled(getbuf(&bufs, a_id, "ADD")?, getbuf(&bufs, b_id, "ADD")?);
                bufs.insert(out_id, r);
            }
            "HEAD" => {
                // Embedding head on the CPU: an exact f32 mat-vec with the
                // folded head weights (both head BatchNorms folded into the
                // Linear). M=1, so this is far cheaper than the padded M=32 NPU
                // GEMM, and it keeps the head off the NPU so all 16 conv kernels
                // stay resident (the NPU2 concurrent-context cap is 16).
                let in_id = geti(m, "in")? as u32;
                let (k, n) = (geti(m, "k")? as usize, geti(m, "n")? as usize);
                let src = getbuf(&bufs, in_id, "HEAD")?;
                if src.c * src.h * src.w != k {
                    return Err(Error::Plan(format!(
                        "HEAD input K mismatch: {} != {k}",
                        src.c * src.h * src.w
                    )));
                }
                let flat = tiled_to_nchw_f32(src); // [k]
                let bt_name = gets(m, "bT")?.to_string();
                let b2_name = gets(m, "b2")?.to_string();
                // bT = W'' transposed [k, n], cached as f32 once.
                if !fcache.contains_key(&bt_name) {
                    let bf = read_bf16(&dir.join(&bt_name))?;
                    fcache.insert(bt_name.clone(), bf.iter().map(|&x| bf16_to_f32(x)).collect());
                }
                if !fcache.contains_key(&b2_name) {
                    let v = read_f32(&dir.join(&b2_name))?;
                    fcache.insert(b2_name.clone(), v);
                }
                let bt = &fcache[&bt_name]; // [k * n], row-major [k][n]
                let mut e = fcache[&b2_name].clone(); // [n], folded bias
                for i in 0..k {
                    let f = flat[i];
                    let row = &bt[i * n..(i + 1) * n];
                    for j in 0..n {
                        e[j] += f * row[j];
                    }
                }
                embed = e;
            }
            other => {
                return Err(Error::Plan(format!("unknown op {other}")));
            }
        }
    }
    if embed.is_empty() {
        return Err(Error::Plan("plan produced no embedding (no HEAD op)".into()));
    }
    Ok(embed)
}

// ----------------------------------------------------------------------------
// CLI (the run_ir18 / run_ir101 binaries)
// ----------------------------------------------------------------------------

/// Full CLI entry point, shared by the per-network binaries (run_ir18,
/// run_ir101). `net_name` only labels the output; the replayed plan is
/// whatever the bundle contains. Prints per-forward latency (cold vs warm)
/// and the embedding's cosine to the CPU reference (ref.bin) and the
/// recorder's embedding (expected.bin); when the bundle has the raw f32 input
/// (input_chw.bin), also exercises the public [`IrEmbedder::embed`] API and
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
        "{net_name} forward: {} conv dispatches on the NPU + CPU head",
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
    if let Ok(exp) = read_f32(&dir.join("expected.bin")) {
        println!(
            "cosine(Rust, recorder embedding)     = {:.5}",
            cosine(&embed, &exp)
        );
    }
    // Exercise the public embed() API: feeding the raw f32 image through the
    // library's pad + bf16-round + tile path must reproduce the recorded
    // replay bit-for-bit (torch and the library both round to nearest even).
    if let Ok(chw) = read_f32(&dir.join("input_chw.bin")) {
        let e2 = emb.embed(&chw)?;
        println!(
            "cosine(embed() API, recorded replay) = {:.5}",
            cosine(&e2, &embed)
        );
    }
    // Ground-truth check against the CPU reference embedding.
    if let Ok(r) = read_f32(&dir.join("ref.bin")) {
        let cos = cosine(&embed, &r);
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
