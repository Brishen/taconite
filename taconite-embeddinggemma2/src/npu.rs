// SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
// SPDX-License-Identifier: Apache-2.0

//! The NPU side: the bundle's kernels (kernels naming the same xclbin
//! share its hardware context: the vision GEMMs one, the text GEMMs
//! another, the MHA length buckets a third -- 3 of NPU2's 16, shared by
//! every process), device row buffers, and the GEMM / MHA dispatches.
//!
//! A kernel loads on first use and stays loaded. When its context cannot
//! be created -- other programs holding the device's slots -- the other
//! contexts' kernels are dropped to make room and load again when next
//! needed: correct but slower (an image swaps ~33 times between them), so
//! [`Npu::swaps`] counts them.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::path::PathBuf;

use taconite::{Timing, bf16_to_f32, f32_to_bf16};
use taconite_bundle::Manifest;
// The NPU path: XRT (feature `xrt`, the default), or the driver's ioctls
// with no XRT (feature `direct`, which wins when both are on).
#[cfg(feature = "direct")]
pub use taconite::direct::{Buffer, Kernel, Session};
#[cfg(all(feature = "xrt", not(feature = "direct")))]
pub use taconite::{Buffer, Kernel, Session};

use crate::Error;

const BF16: usize = 2;

/// One compiled `flm.GEMM` (a `gemm` record) and its multi-chunk streams
/// (`gemm_chunks`): A / C view sizes over n chunks at index n - 1.
#[derive(Debug, Clone)]
pub struct GemmSpec {
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub lda: usize,
    pub c_stride: usize,
    pub a_elems: Vec<usize>,
    pub c_elems: Vec<usize>,
}

/// One MHA length bucket (an `mha` record): its view sizes and the
/// instruction words holding the valid key count and the window.
#[derive(Debug, Clone)]
pub struct MhaSpec {
    pub rows: usize,
    pub elems: [usize; 4],
    pub kv_words: Vec<usize>,
    pub win_words: Vec<usize>,
}

/// A bf16 device buffer of `rows` x `width`.
pub struct Rows {
    pub buf: Buffer,
    pub width: usize,
}

impl Rows {
    /// x `[rows, cols]` (f32) as bf16 into the first rows, from column 0,
    /// synced to the device.
    pub fn put(&mut self, x: &[f32], cols: usize) -> Result<(), Error> {
        let rows = x.len() / cols;
        let w = self.width;
        let dst = self.buf.as_mut_slice::<u16>();
        for (r, src) in x.chunks(cols).enumerate() {
            for (o, &v) in dst[r * w..r * w + cols].iter_mut().zip(src) {
                *o = f32_to_bf16(v);
            }
        }
        self.buf.sub(0, rows * w * BF16)?.sync_to_device()?;
        Ok(())
    }

    /// The first `rows` rows' first `cols` columns, synced from the device.
    pub fn get(&self, rows: usize, cols: usize) -> Result<Vec<f32>, Error> {
        let w = self.width;
        self.buf.sub(0, rows * w * BF16)?.sync_from_device()?;
        let src = self.buf.as_slice::<u16>();
        let mut out = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            out.extend(src[r * w..r * w + cols].iter().map(|&v| bf16_to_f32(v)));
        }
        Ok(out)
    }

    /// A view of `elems` elements from column `col` of the first row.
    pub fn view(&self, col: usize, elems: usize) -> Result<Buffer, Error> {
        Ok(self.buf.sub(col * BF16, elems * BF16)?)
    }
}

/// Where a kernel comes from: its context key, xclbin, kernel name,
/// instruction stream, and ops a run.
struct Src {
    ctx: String,
    xclbin: PathBuf,
    name: String,
    insts: PathBuf,
    ops: u64,
}

pub struct Npu {
    pub session: Session,
    srcs: HashMap<String, Src>,
    kernels: RefCell<HashMap<String, Kernel>>,
    /// contexts dropped to make room for another
    pub swaps: Cell<usize>,
    pub gemms: HashMap<String, GemmSpec>,
    /// the vision attention's length buckets, shortest first
    pub mhas: Vec<MhaSpec>,
    /// the (valid keys, window) each bucket's stream holds now
    mha_cur: RefCell<HashMap<usize, (u32, u32)>>,
    /// distinct hardware contexts (xclbins) loaded
    pub contexts: usize,
}

fn gemm_kernel(key: &str, n: usize) -> String {
    if n == 1 { key.to_string() } else { format!("{key}_x{n}") }
}

impl Npu {
    /// Opens the NPU and reads every `gemm`, `gemm_chunks` and `mha`
    /// kernel of the manifest (loaded on first use).
    pub fn open(m: &Manifest) -> Result<Self, Error> {
        let session = Session::open(0)?;
        let mut srcs = HashMap::new();
        let mut gemms: HashMap<String, GemmSpec> = HashMap::new();
        let mut gemm_ctx: HashMap<String, String> = HashMap::new();
        let mut mhas = Vec::new();
        let mut ctxs: Vec<PathBuf> = Vec::new();
        let words = |s: &str| -> Result<Vec<usize>, Error> {
            s.split(',').map(|w| w.parse().map_err(|_| Error::Bundle(format!("bad instruction word {w}")))).collect()
        };
        for r in m.records() {
            let (key, ctx, ops) = match r.tag.as_str() {
                "gemm" => {
                    let g = GemmSpec {
                        m: r.get("M")?,
                        k: r.get("K")?,
                        n: r.get("N")?,
                        lda: r.get("lda")?,
                        c_stride: r.get("c_stride")?,
                        a_elems: vec![r.get("a_elems")?],
                        c_elems: vec![r.get("c_elems")?],
                    };
                    let ops = (2 * g.m * g.k * g.n) as u64;
                    let key = r.field(0)?.to_string();
                    gemms.insert(key.clone(), g);
                    gemm_ctx.insert(key.clone(), r.str("ctx")?.to_string());
                    (key, r.str("ctx")?.to_string(), ops)
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
                    (gemm_kernel(key, n), gemm_ctx[key].clone(), ops)
                }
                "mha" => {
                    let spec = MhaSpec {
                        rows: r.field_as(0)?,
                        elems: [r.get("q_elems")?, r.get("k_elems")?, r.get("v_elems")?, r.get("o_elems")?],
                        kv_words: words(r.str("kv_words")?)?,
                        win_words: words(r.str("win_words")?)?,
                    };
                    let key = format!("mha_{}", spec.rows);
                    mhas.push(spec);
                    (key, r.str("ctx")?.to_string(), 0)
                }
                _ => continue,
            };
            let x = m.xclbin(&ctx)?;
            if !ctxs.contains(&x.path) {
                ctxs.push(x.path.clone());
            }
            let src = Src { ctx, xclbin: x.path.clone(), name: x.kernel.clone(), insts: m.path(r.str("insts")?), ops };
            srcs.insert(key, src);
        }
        mhas.sort_by_key(|s| s.rows);
        Ok(Npu {
            session,
            srcs,
            kernels: Default::default(),
            swaps: Cell::new(0),
            gemms,
            mhas,
            mha_cur: Default::default(),
            contexts: ctxs.len(),
        })
    }

    pub fn gemm(&self, key: &str) -> Result<&GemmSpec, Error> {
        self.gemms.get(key).ok_or_else(|| Error::Bundle(format!("no GEMM {key} in the bundle")))
    }

    /// A device buffer holding `bytes` (packed weights).
    pub fn upload(&self, bytes: &[u8]) -> Result<Buffer, Error> {
        let mut b = self.session.alloc(bytes.len())?;
        b.write(bytes)?;
        Ok(b)
    }

    /// A zeroed bf16 device buffer of `rows` x `width`.
    pub fn rows(&self, rows: usize, width: usize) -> Result<Rows, Error> {
        let mut buf = self.session.alloc(rows * width * BF16)?;
        buf.as_mut_slice::<u16>().fill(0);
        buf.sync_to_device()?;
        Ok(Rows { buf, width })
    }

    /// Runs `f` on kernel `key`, loading it first if it is not.
    fn with_kernel<T>(&self, key: &str, f: impl FnOnce(&Kernel) -> Result<T, Error>) -> Result<T, Error> {
        if !self.kernels.borrow().contains_key(key) {
            let s = self.srcs.get(key).ok_or_else(|| Error::Bundle(format!("no kernel {key} in the bundle")))?;
            let load = || {
                self.session
                    .load_kernel(&s.xclbin, &s.insts, Some(&s.name), s.ops)
                    .map_err(|e| Error::Npu(format!("loading {key}: {e}")))
            };
            let k = match load() {
                Ok(k) => k,
                Err(e) => {
                    // no free slot: give up the other contexts' kernels
                    let mut ks = self.kernels.borrow_mut();
                    let others: Vec<String> = ks.keys().filter(|k| self.srcs[*k].ctx != s.ctx).cloned().collect();
                    if others.is_empty() {
                        return Err(e);
                    }
                    let mut ctxs: Vec<&str> = others.iter().map(|k| self.srcs[k].ctx.as_str()).collect();
                    ctxs.sort();
                    ctxs.dedup();
                    self.swaps.set(self.swaps.get() + ctxs.len());
                    for k in &others {
                        ks.remove(k);
                    }
                    // a reloaded MHA stream holds its compiled words again
                    self.mha_cur.borrow_mut().retain(|rows, _| !others.contains(&format!("mha_{rows}")));
                    drop(ks);
                    load()?
                }
            };
            self.kernels.borrow_mut().insert(key.to_string(), k);
        }
        let ks = self.kernels.borrow();
        f(&ks[key])
    }

    /// GEMM `key` over the first `rows` rows of A (from column `col`), one
    /// dispatch: C's first rows (row stride N) get the product.
    #[allow(clippy::too_many_arguments)]
    pub fn run_gemm(
        &self,
        key: &str,
        a: &Rows,
        col: usize,
        w: &Buffer,
        c: &Rows,
        rows: usize,
        t: &mut Timing,
    ) -> Result<(), Error> {
        let g = self.gemm(key)?;
        let n = rows.div_ceil(g.m);
        let (ae, ce) = match (g.a_elems.get(n - 1), g.c_elems.get(n - 1)) {
            (Some(&a), Some(&c)) => (a, c),
            _ => return Err(Error::Input(format!("{key}: {rows} rows (the bundle has streams for {})", g.a_elems.len() * g.m))),
        };
        let (av, cv) = (a.view(col, ae)?, c.view(0, ce)?);
        let d = self.with_kernel(&gemm_kernel(key, n), |k| {
            k.run(&[&av, w, &cv]).map_err(|e| Error::Npu(format!("{key}: {e}")))
        })?;
        t.add(&format!("npu:{key}"), d);
        Ok(())
    }

    /// The vision attention over `n` rows: q / k / v staged in `qkv`
    /// (`[rows, 3 D]`), the output into `o` (`[rows, D]`).
    pub fn run_mha(&self, qkv: &Rows, o: &Rows, n: usize, no_window: u32, t: &mut Timing) -> Result<(), Error> {
        let spec = self
            .mhas
            .iter()
            .find(|s| s.rows >= n)
            .ok_or_else(|| Error::Input(format!("{n} patches: the largest MHA bucket is {:?}", self.mhas.last().map(|s| s.rows))))?;
        let key = format!("mha_{}", spec.rows);
        let want = (n as u32, no_window);
        let [qe, ke, ve, oe] = spec.elems;
        let views = [qkv.view(0, qe)?, qkv.view(0, ke)?, qkv.view(0, ve)?, o.view(0, oe)?];
        let d = self.with_kernel(&key, |k| {
            if self.mha_cur.borrow().get(&spec.rows) != Some(&want) {
                let mut words: Vec<(usize, u32)> = spec.kv_words.iter().map(|&i| (i, want.0)).collect();
                words.extend(spec.win_words.iter().map(|&i| (i, want.1)));
                k.set_insts_words(&words).map_err(|e| Error::Npu(format!("{key}: {e}")))?;
                self.mha_cur.borrow_mut().insert(spec.rows, want);
            }
            k.run(&[&views[0], &views[1], &views[2], &views[3]]).map_err(|e| Error::Npu(format!("{key}: {e}")))
        })?;
        t.add("npu:mha", d);
        Ok(())
    }
}
