// SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
// SPDX-License-Identifier: Apache-2.0

//! The NPU side: the bundle's kernels (kernels naming the same xclbin
//! share its hardware context: a context an `flm.GEMM` K, and the
//! attention's -- 6 of NPU2's 16, shared by every process), flat bf16
//! device buffers, and the GEMM / MHA dispatches.
//!
//! A kernel loads on first use and stays loaded. When its context cannot
//! be created -- other programs holding the device's slots -- the least
//! recently used other context's kernels are dropped to make room (and
//! load again when next needed): correct but slower, so [`Npu::swaps`]
//! counts them.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::path::PathBuf;

use taconite::cpu::par_rows;
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
/// Elements a worker converts at a time.
const PIECE: usize = 1 << 16;
/// 100 ms waits for a hardware context when every slot is taken by others
/// (another program's run may end soon; a model this process keeps
/// resident won't, so the wait is short).
const WAITS: usize = 50;

/// One compiled `flm.GEMM` at its row count (a `gemm` record).
#[derive(Debug, Clone)]
pub struct GemmSpec {
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub lda: usize,
    pub c_stride: usize,
    pub rows: usize,
    pub a_elems: usize,
    pub c_elems: usize,
}

/// A flat bf16 device buffer.
pub struct Flat {
    pub buf: Buffer,
    pub elems: usize,
}

impl Flat {
    /// x (f32) as bf16 into the first `x.len()` elements, synced.
    pub fn put(&mut self, x: &[f32]) -> Result<(), Error> {
        self.fill(x.len(), |i0, dst| {
            for (o, &v) in dst.iter_mut().zip(&x[i0..]) {
                *o = f32_to_bf16(v);
            }
        })
    }

    /// The first `n` elements written by `f(first index, piece)` in parallel
    /// (bf16 bits), then synced to the device.
    pub fn fill(&mut self, n: usize, f: impl Fn(usize, &mut [u16]) + Sync) -> Result<(), Error> {
        if n > self.elems {
            return Err(Error::Input(format!("{n} elements into a buffer of {}", self.elems)));
        }
        let dst = &mut self.buf.as_mut_slice::<u16>()[..n];
        par_rows(dst, PIECE.min(n.max(1)), |r0, piece| f(r0 * PIECE.min(n.max(1)), piece));
        self.buf.sub(0, n * BF16)?.sync_to_device()?;
        Ok(())
    }

    /// `rows` rows of `width` from the start, row r written by `f(r, row)`
    /// in parallel (bf16 bits), then synced to the device.
    pub fn fill_rows(&mut self, rows: usize, width: usize, f: impl Fn(usize, &mut [u16]) + Sync) -> Result<(), Error> {
        let n = rows * width;
        if n > self.elems {
            return Err(Error::Input(format!("{n} elements into a buffer of {}", self.elems)));
        }
        let dst = &mut self.buf.as_mut_slice::<u16>()[..n];
        par_rows(dst, width, |r0, piece| {
            for (ri, row) in piece.chunks_mut(width).enumerate() {
                f(r0 + ri, row);
            }
        });
        self.buf.sub(0, n * BF16)?.sync_to_device()?;
        Ok(())
    }

    /// The first `n` elements, synced from the device, as bf16 bits.
    pub fn bits(&self, n: usize) -> Result<&[u16], Error> {
        self.buf.sub(0, n * BF16)?.sync_from_device()?;
        Ok(&self.buf.as_slice::<u16>()[..n])
    }

    /// The first `n` elements as f32.
    pub fn get(&self, n: usize) -> Result<Vec<f32>, Error> {
        let src = self.bits(n)?;
        let mut out = vec![0f32; n];
        par_rows(&mut out, PIECE.min(n.max(1)), |r0, piece| {
            let i0 = r0 * PIECE.min(n.max(1));
            for (o, &b) in piece.iter_mut().zip(&src[i0..]) {
                *o = bf16_to_f32(b);
            }
        });
        Ok(out)
    }

    pub fn view(&self, elems: usize) -> Result<Buffer, Error> {
        if elems > self.elems {
            return Err(Error::Bundle(format!("a view of {elems} elements over a buffer of {}", self.elems)));
        }
        Ok(self.buf.sub(0, elems * BF16)?)
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
    /// context -> when a kernel of it last ran
    used: RefCell<HashMap<String, u64>>,
    tick: Cell<u64>,
    pub gemms: HashMap<String, GemmSpec>,
    /// the attention's q, k, v, o view sizes
    pub mha: [usize; 4],
    /// distinct hardware contexts (xclbins)
    pub contexts: usize,
}

impl Npu {
    /// Opens the NPU and reads every `gemm` and `mha` kernel of the
    /// manifest (loaded on first use).
    pub fn open(m: &Manifest) -> Result<Self, Error> {
        let session = Session::open(0)?;
        let mut srcs = HashMap::new();
        let mut gemms = HashMap::new();
        let mut mha = None;
        let mut ctxs: Vec<PathBuf> = Vec::new();
        for r in m.records() {
            let (key, ops) = match r.tag.as_str() {
                "gemm" => {
                    let g = GemmSpec {
                        m: r.get("M")?,
                        k: r.get("K")?,
                        n: r.get("N")?,
                        lda: r.get("lda")?,
                        c_stride: r.get("c_stride")?,
                        rows: r.get("rows")?,
                        a_elems: r.get("a_elems")?,
                        c_elems: r.get("c_elems")?,
                    };
                    let ops = (2 * g.rows * g.k * g.n) as u64;
                    let key = r.field(0)?.to_string();
                    gemms.insert(key.clone(), g);
                    (key, ops)
                }
                "mha" => {
                    mha = Some([r.get("q_elems")?, r.get("k_elems")?, r.get("v_elems")?, r.get("o_elems")?]);
                    ("mha".to_string(), 0)
                }
                _ => continue,
            };
            let ctx = r.str("ctx")?.to_string();
            let x = m.xclbin(&ctx)?;
            if !ctxs.contains(&x.path) {
                ctxs.push(x.path.clone());
            }
            let src = Src { ctx, xclbin: x.path.clone(), name: x.kernel.clone(), insts: m.path(r.str("insts")?), ops };
            srcs.insert(key, src);
        }
        let mha = mha.ok_or_else(|| Error::Bundle("no mha record".into()))?;
        Ok(Npu {
            session,
            srcs,
            kernels: Default::default(),
            swaps: Cell::new(0),
            used: Default::default(),
            tick: Cell::new(0),
            gemms,
            mha,
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

    /// A zeroed flat bf16 device buffer.
    pub fn flat(&self, elems: usize) -> Result<Flat, Error> {
        let mut buf = self.session.alloc(elems * BF16)?;
        buf.as_mut_slice::<u16>().fill(0);
        buf.sync_to_device()?;
        Ok(Flat { buf, elems })
    }

    /// Runs `f` on kernel `key`, loading it first if it is not.
    fn with_kernel<T>(&self, key: &str, f: impl FnOnce(&Kernel) -> Result<T, Error>) -> Result<T, Error> {
        let s = self.srcs.get(key).ok_or_else(|| Error::Bundle(format!("no kernel {key} in the bundle")))?;
        self.tick.set(self.tick.get() + 1);
        self.used.borrow_mut().insert(s.ctx.clone(), self.tick.get());
        if !self.kernels.borrow().contains_key(key) {
            let load = || {
                self.session
                    .load_kernel(&s.xclbin, &s.insts, Some(&s.name), s.ops)
                    .map_err(|e| Error::Npu(format!("loading {key}: {e}")))
            };
            let mut waits = 0;
            let k = loop {
                match load() {
                    Ok(k) => break k,
                    Err(e) => {
                        // no free slot: give up the least recently used
                        // other context's kernels
                        let mut ks = self.kernels.borrow_mut();
                        let used = self.used.borrow();
                        let victim = ks
                            .keys()
                            .map(|k| self.srcs[k].ctx.as_str())
                            .filter(|c| *c != s.ctx)
                            .min_by_key(|c| used.get(*c).copied().unwrap_or(0))
                            .map(str::to_string);
                        let Some(victim) = victim else {
                            // nothing of ours left to give up: other
                            // programs hold every slot; wait for one
                            if waits >= WAITS {
                                return Err(Error::Npu(format!(
                                    "{e}: every NPU hardware context is taken by other models or \
                                     programs (NPU2 has 16); free some (e.g. another model's \
                                     `release_contexts`) and try again"
                                )));
                            }
                            waits += 1;
                            drop((ks, used));
                            std::thread::sleep(std::time::Duration::from_millis(100));
                            continue;
                        };
                        ks.retain(|k, _| self.srcs[k].ctx != victim);
                        self.swaps.set(self.swaps.get() + 1);
                    }
                }
            };
            self.kernels.borrow_mut().insert(key.to_string(), k);
        }
        let ks = self.kernels.borrow();
        f(&ks[key])
    }

    /// Drops every loaded kernel, freeing this model's hardware contexts
    /// for others (they load again on next use); the number freed.
    pub fn release(&self) -> usize {
        let mut ks = self.kernels.borrow_mut();
        let mut ctxs: Vec<&str> = ks.keys().map(|k| self.srcs[k].ctx.as_str()).collect();
        ctxs.sort();
        ctxs.dedup();
        let n = ctxs.len();
        ks.clear();
        n
    }

    /// Loads the kernels now rather than on first use, as many as the
    /// device's free hardware contexts hold (the rest load on use).
    pub fn preload(&self) -> Result<(), Error> {
        let mut keys: Vec<&String> = self.srcs.keys().collect();
        keys.sort_by_key(|k| (&self.srcs[*k].ctx, *k));
        for k in keys {
            let s = &self.srcs[k];
            if self.kernels.borrow().contains_key(k) {
                continue;
            }
            match self.session.load_kernel(&s.xclbin, &s.insts, Some(&s.name), s.ops) {
                Ok(kn) => {
                    self.kernels.borrow_mut().insert(k.clone(), kn);
                }
                Err(_) => break,
            }
        }
        Ok(())
    }

    /// GEMM `key` over all its rows, one dispatch: A from `a`'s start (row
    /// stride lda), C into `c`'s (row stride N).
    pub fn run_gemm(&self, key: &str, a: &Flat, w: &Buffer, c: &Flat, t: &mut Timing) -> Result<(), Error> {
        let g = self.gemm(key)?;
        let (av, cv) = (a.view(g.a_elems)?, c.view(g.c_elems)?);
        let d = self.with_kernel(key, |k| k.run(&[&av, w, &cv]).map_err(|e| Error::Npu(format!("{key}: {e}"))))?;
        t.add(&format!("npu:{key}"), d);
        Ok(())
    }

    /// The attention: q / k / v staged in `qkv` (`[rows, 3 D]`), the output
    /// into `o` (`[rows, D]`).
    pub fn run_mha(&self, qkv: &Flat, o: &Flat, t: &mut Timing) -> Result<(), Error> {
        let [qe, ke, ve, oe] = self.mha;
        let views = [qkv.view(qe)?, qkv.view(ke)?, qkv.view(ve)?, o.view(oe)?];
        let d = self.with_kernel("mha", |k| {
            k.run(&[&views[0], &views[1], &views[2], &views[3]]).map_err(|e| Error::Npu(format!("mha: {e}")))
        })?;
        t.add("npu:mha", d);
        Ok(())
    }
}
