// SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
// SPDX-License-Identifier: Apache-2.0

//! The NPU side: every kernel of the bundle -- the prefill GEMMs (an
//! instruction stream a chunk count) on one hardware context, the decode
//! GEMVs on another -- and helpers for the device buffers.
//!
//! Both contexts stay resident when the device has the slots (NPU2 has 16,
//! shared by every process). When it has not -- other programs holding
//! them -- a context's kernels are dropped to make room for the other's
//! and loaded again when next needed: one swap each way between a prompt
//! and its answer, instead of a failure.

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

use taconite_bundle::Manifest;
// The NPU path: XRT (feature `xrt`, the default), or the driver's ioctls
// with no XRT (feature `direct`, which wins when both are on).
#[cfg(feature = "direct")]
pub use taconite::direct::{Buffer, Kernel, Session};
#[cfg(all(feature = "xrt", not(feature = "direct")))]
pub use taconite::{Buffer, Kernel, Session};

use crate::Error;

/// One compiled `flm.GEMM` over `chunk` rows (a `gemm` record), and the
/// A / C view sizes of its stream over n chunks (index n - 1).
#[derive(Debug, Clone)]
pub struct GemmSpec {
    pub m: usize,
    pub k: usize,
    pub n: usize,
    /// row stride of A in the buffer it reads (K, or wider)
    pub lda: usize,
    /// row stride of C (N, or wider for a paired epilogue)
    pub c_stride: usize,
    pub b_bytes: usize,
    pub a_elems: Vec<usize>,
    pub c_elems: Vec<usize>,
}

/// One `GEMVbfp16` (a `gemv` record): y [M] = W [M, K] x [K].
#[derive(Debug, Clone)]
pub struct GemvSpec {
    pub m: usize,
    pub k: usize,
    pub w_words: usize,
}

/// A kernel as the manifest names it: its context key, xclbin, kernel
/// name, instruction stream, and ops a run (for the activity counters).
struct KernelSrc {
    ctx: String,
    xclbin: PathBuf,
    name: String,
    insts: PathBuf,
    ops: u64,
}

pub struct Npu {
    pub session: Session,
    srcs: HashMap<String, KernelSrc>,
    /// context key -> its loaded kernels (absent: not resident)
    loaded: RefCell<HashMap<String, HashMap<String, Kernel>>>,
    /// contexts dropped to make room for another
    pub swaps: std::cell::Cell<usize>,
    pub gemms: HashMap<String, GemmSpec>,
    pub gemvs: HashMap<String, GemvSpec>,
    /// distinct hardware contexts (xclbins) loaded
    pub contexts: usize,
}

/// The kernel key of GEMM `key`'s stream over `n` chunks.
pub fn gemm_key(key: &str, n: usize) -> String {
    format!("gemm.{key}.x{n}")
}

pub fn gemv_key(key: &str) -> String {
    format!("gemv.{key}")
}

impl Npu {
    /// Opens the NPU and loads every `gemm`, `gemm_chunks` and `gemv` of
    /// the manifest.
    pub fn open(m: &Manifest) -> Result<Self, Error> {
        let session = Session::open(0)?;
        let mut srcs = HashMap::new();
        let mut gemms: HashMap<String, GemmSpec> = HashMap::new();
        // a GEMM's context: its multi-chunk streams run on the same xclbin
        let mut gemm_ctx: HashMap<String, String> = HashMap::new();
        let mut gemvs = HashMap::new();
        let mut ctxs: Vec<PathBuf> = Vec::new();
        for r in m.records() {
            let (key, ctx, ops) = match r.tag.as_str() {
                "gemm" => {
                    let g = GemmSpec {
                        m: r.get("M")?,
                        k: r.get("K")?,
                        n: r.get("N")?,
                        lda: r.get("lda")?,
                        c_stride: r.get("c_stride")?,
                        b_bytes: r.get("b_bytes")?,
                        a_elems: vec![r.get("a_elems")?],
                        c_elems: vec![r.get("c_elems")?],
                    };
                    let ops = (2 * g.m * g.k * g.n) as u64;
                    gemms.insert(r.field(0)?.to_string(), g);
                    gemm_ctx.insert(r.field(0)?.to_string(), r.str("ctx").unwrap_or("gemm").to_string());
                    (gemm_key(r.field(0)?, 1), "gemm", ops)
                }
                "gemm_chunks" => {
                    let (key, n): (&str, usize) = (r.field(0)?, r.field_as(1)?);
                    let g = gemms
                        .get_mut(key)
                        .ok_or_else(|| Error::Bundle(format!("gemm_chunks {key} before its gemm")))?;
                    if g.a_elems.len() + 1 != n {
                        return Err(Error::Bundle(format!("gemm_chunks {key} {n} out of order")));
                    }
                    g.a_elems.push(r.get("a_elems")?);
                    g.c_elems.push(r.get("c_elems")?);
                    let ops = (2 * n * g.m * g.k * g.n) as u64;
                    (gemm_key(key, n), gemm_ctx[key].as_str(), ops)
                }
                "gemv" => {
                    let g = GemvSpec { m: r.get("M")?, k: r.get("K")?, w_words: r.get("w_words")? };
                    let ops = (2 * g.m * g.k) as u64;
                    gemvs.insert(r.field(0)?.to_string(), g);
                    (gemv_key(r.field(0)?), "gemv", ops)
                }
                _ => continue,
            };
            let ctx = r.str("ctx").unwrap_or(ctx).to_string();
            let x = m.xclbin(&ctx)?;
            if !ctxs.contains(&x.path) {
                ctxs.push(x.path.clone());
            }
            let src =
                KernelSrc { ctx, xclbin: x.path.clone(), name: x.kernel.clone(), insts: m.path(r.str("insts")?), ops };
            srcs.insert(key, src);
        }
        let npu = Npu {
            session,
            srcs,
            loaded: RefCell::new(HashMap::new()),
            swaps: std::cell::Cell::new(0),
            gemms,
            gemvs,
            contexts: ctxs.len(),
        };
        // every context up front where the device has the slots; the
        // rest load on first use
        let mut keys: Vec<String> = npu.srcs.values().map(|s| s.ctx.clone()).collect();
        keys.sort();
        keys.dedup();
        for (i, ctx) in keys.iter().enumerate() {
            match npu.load_ctx(ctx) {
                Ok(()) => {}
                Err(_) if i > 0 => break,
                Err(e) => return Err(e),
            }
        }
        Ok(npu)
    }

    /// Loads every kernel of context `ctx`.
    fn load_ctx(&self, ctx: &str) -> Result<(), Error> {
        let mut ks = HashMap::new();
        for (key, s) in self.srcs.iter().filter(|(_, s)| s.ctx == ctx) {
            let k = self
                .session
                .load_kernel(&s.xclbin, &s.insts, Some(&s.name), s.ops)
                .map_err(|e| Error::Npu(format!("loading {key}: {e}")))?;
            ks.insert(key.clone(), k);
        }
        self.loaded.borrow_mut().insert(ctx.to_string(), ks);
        Ok(())
    }

    pub fn gemm(&self, key: &str) -> Result<&GemmSpec, Error> {
        self.gemms.get(key).ok_or_else(|| Error::Bundle(format!("no GEMM {key} in the bundle")))
    }

    pub fn gemv(&self, key: &str) -> Result<&GemvSpec, Error> {
        self.gemvs.get(key).ok_or_else(|| Error::Bundle(format!("no GEMV {key} in the bundle")))
    }

    /// A device buffer holding `bytes` (packed weights).
    pub fn upload(&self, bytes: &[u8]) -> Result<Buffer, Error> {
        let mut b = self.session.alloc(bytes.len())?;
        b.write(bytes)?;
        Ok(b)
    }

    /// A zeroed device buffer of `elems` bf16 values.
    pub fn zeros(&self, elems: usize) -> Result<Buffer, Error> {
        let mut b = self.session.alloc(elems * 2)?;
        b.as_mut_slice::<u16>().fill(0);
        b.sync_to_device()?;
        Ok(b)
    }

    /// Runs kernel `key` once over `args`, in its argument order.
    pub fn run(&self, key: &str, args: &[&Buffer]) -> Result<Duration, Error> {
        let ctx = &self.srcs.get(key).ok_or_else(|| Error::Bundle(format!("no kernel {key} in the bundle")))?.ctx;
        if !self.loaded.borrow().contains_key(ctx) {
            if let Err(e) = self.load_ctx(ctx) {
                // no free slot: give up the other contexts' and retry
                let others = self.loaded.borrow().len();
                if others == 0 {
                    return Err(e);
                }
                self.loaded.borrow_mut().clear();
                self.swaps.set(self.swaps.get() + others);
                self.load_ctx(ctx)?;
            }
        }
        let loaded = self.loaded.borrow();
        let k = &loaded[ctx][key];
        k.run(args).map_err(|e| Error::Npu(format!("{key}: {e}")))
    }
}
