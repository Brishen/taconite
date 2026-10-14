// SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
// SPDX-License-Identifier: Apache-2.0

//! The NPU side: the bundle's kernels (kernels naming the same xclbin
//! share its hardware context: one an `flm.GEMM` shape group and one a
//! MambaScan phase, 7 of NPU2's 16, shared by every process), flat bf16
//! device buffers, and the dispatches.
//!
//! A GEMM runs over any row count as power-of-two chunk counts (of
//! `chunk` rows, up to `max_chunks` a dispatch), each on sub-buffers of A
//! and C at its first row; a MambaScan phase at one of the exported
//! (sequences, length) shapes.
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

#[cfg(not(any(feature = "xrt", feature = "direct")))]
compile_error!("taconite-reuse needs the `xrt` or the `direct` feature");

use crate::Error;

const BF16: usize = 2;
/// Elements a worker converts at a time.
const PIECE: usize = 1 << 16;
/// 100 ms waits for a hardware context when every slot is taken by others.
const WAITS: usize = 50;

/// One compiled `flm.GEMM` at a chunk count (a `gemm` record).
#[derive(Debug, Clone)]
pub struct GemmSpec {
    pub k: usize,
    pub n: usize,
    pub lda: usize,
    pub rows: usize,
    pub a_elems: usize,
    pub c_elems: usize,
}

/// One compiled MambaScan phase at a shape (a `mamba` record).
#[derive(Debug, Clone)]
pub struct MambaSpec {
    pub seqs: usize,
    pub l: usize,
    pub rows: usize,
    pub in_elems: usize,
    pub out_elems: usize,
}

/// A flat bf16 device buffer.
pub struct Flat {
    pub buf: Buffer,
    pub elems: usize,
}

impl Flat {
    /// The first `n` elements as mutable bf16 bits (follow with
    /// [`sync`](Self::sync)).
    pub fn bits_mut(&mut self, n: usize) -> &mut [u16] {
        &mut self.buf.as_mut_slice::<u16>()[..n]
    }

    /// Syncs the first `n` elements to the device.
    pub fn sync(&self, n: usize) -> Result<(), Error> {
        self.buf.sub(0, n.min(self.elems) * BF16)?.sync_to_device()?;
        Ok(())
    }

    /// `rows` rows of `width` from the start, row r written by `f(r, row)`
    /// in parallel (bf16 bits), then the first `sync_elems` (at least the
    /// rows) synced to the device.
    pub fn fill_rows(&mut self, rows: usize, width: usize, sync_elems: usize, f: impl Fn(usize, &mut [u16]) + Sync) -> Result<(), Error> {
        let n = rows * width;
        if n > self.elems || sync_elems > self.elems {
            return Err(Error::Input(format!("{n} elements into a buffer of {}", self.elems)));
        }
        let dst = &mut self.buf.as_mut_slice::<u16>()[..n];
        par_rows(dst, width, |r0, piece| {
            for (ri, row) in piece.chunks_mut(width).enumerate() {
                f(r0 + ri, row);
            }
        });
        self.sync(sync_elems.max(n))
    }

    /// Zeroes the first `n` elements (host side; not synced).
    pub fn zero(&mut self, n: usize) {
        let dst = &mut self.buf.as_mut_slice::<u16>()[..n];
        par_rows(dst, PIECE, |_, p| p.fill(0));
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
        par_rows(&mut out, PIECE, |r0, piece| {
            let i0 = r0 * PIECE;
            for (o, &b) in piece.iter_mut().zip(&src[i0..]) {
                *o = bf16_to_f32(b);
            }
        });
        Ok(out)
    }

    /// A view of `elems` elements from element `offset`.
    pub fn view(&self, offset: usize, elems: usize) -> Result<Buffer, Error> {
        if offset + elems > self.elems {
            return Err(Error::Bundle(format!("a view of [{offset}, {}) over a buffer of {}", offset + elems, self.elems)));
        }
        Ok(self.buf.sub(offset * BF16, elems * BF16)?)
    }
}

/// x as bf16 bits.
pub fn to_bits(x: f32) -> u16 {
    f32_to_bf16(x)
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
    /// `<group>.<chunks>`
    pub gemms: HashMap<String, GemmSpec>,
    /// `<phase>.<seqs>x<L>`
    pub mambas: HashMap<String, MambaSpec>,
    pub chunk: usize,
    pub max_chunks: usize,
    /// distinct hardware contexts (xclbins)
    pub contexts: usize,
}

impl Npu {
    /// Opens the NPU and reads every `gemm` and `mamba` kernel of the
    /// manifest (loaded on first use).
    pub fn open(m: &Manifest) -> Result<Self, Error> {
        let session = Session::open(0)?;
        let mut srcs = HashMap::new();
        let mut gemms = HashMap::new();
        let mut mambas = HashMap::new();
        let mut ctxs: Vec<PathBuf> = Vec::new();
        for r in m.records() {
            let key = format!("{}:{}", r.tag, r.field(0)?);
            let ops = match r.tag.as_str() {
                "gemm" => {
                    let g = GemmSpec {
                        k: r.get("K")?,
                        n: r.get("N")?,
                        lda: r.get("lda")?,
                        rows: r.get("rows")?,
                        a_elems: r.get("a_elems")?,
                        c_elems: r.get("c_elems")?,
                    };
                    let ops = (2 * g.rows * g.k * g.n) as u64;
                    gemms.insert(r.field(0)?.to_string(), g);
                    ops
                }
                "mamba" => {
                    let s = MambaSpec {
                        seqs: r.get("seqs")?,
                        l: r.get("L")?,
                        rows: r.get("rows")?,
                        in_elems: r.get("in_elems")?,
                        out_elems: r.get("out_elems")?,
                    };
                    mambas.insert(r.field(0)?.to_string(), s);
                    0
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
        Ok(Npu {
            session,
            srcs,
            kernels: Default::default(),
            swaps: Cell::new(0),
            used: Default::default(),
            tick: Cell::new(0),
            gemms,
            mambas,
            chunk: m.param_as("chunk")?,
            max_chunks: m.param_as("max_chunks")?,
            contexts: ctxs.len(),
        })
    }

    /// A device buffer holding `bytes` (packed weights, parameters).
    pub fn upload(&self, bytes: &[u8]) -> Result<Buffer, Error> {
        let mut b = self.session.alloc(bytes.len())?;
        b.write(bytes)?;
        Ok(b)
    }

    /// A zeroed flat bf16 device buffer.
    pub fn flat(&self, elems: usize) -> Result<Flat, Error> {
        let mut buf = self.session.alloc(elems.max(1) * BF16)?;
        buf.as_mut_slice::<u16>().fill(0);
        buf.sync_to_device()?;
        Ok(Flat { buf, elems })
    }

    /// `slot` grown (zeroed, re-allocated) to hold at least `elems`.
    pub fn ensure<'a>(&self, slot: &'a mut Option<Flat>, elems: usize) -> Result<&'a mut Flat, Error> {
        if slot.as_ref().is_none_or(|f| f.elems < elems) {
            // a little headroom, so a slightly longer next input reuses it
            *slot = Some(self.flat(elems + elems / 8)?);
        }
        Ok(slot.as_mut().unwrap())
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
                            if waits >= WAITS {
                                return Err(Error::Npu(format!(
                                    "{e}: every NPU hardware context is taken by other models or \
                                     programs (NPU2 has 16); free some and try again"
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

    fn run(&self, key: &str, args: &[&Buffer], t: &mut Timing, label: &str) -> Result<(), Error> {
        let d = self.with_kernel(key, |k| k.run(args).map_err(|e| Error::Npu(format!("{key}: {e}"))))?;
        t.add(label, d);
        Ok(())
    }

    /// GEMM group `g` over `rows` rows: A from row `a_row` of `a` (at the
    /// group's row stride), C into row `c_row` of `c` (row stride N), as
    /// power-of-two chunk counts. The caller has synced A.
    pub fn gemm(&self, g: &str, a: &Flat, a_row: usize, w: &Buffer, c: &Flat, c_row: usize, rows: usize, t: &mut Timing) -> Result<(), Error> {
        let mut left = rows.div_ceil(self.chunk);
        let mut r = 0;
        while left > 0 {
            let n = if left >= self.max_chunks { self.max_chunks } else { 1 << (usize::BITS - 1 - left.leading_zeros()) };
            let key = format!("{g}.{n}");
            let s = self.gemms.get(&key).ok_or_else(|| Error::Bundle(format!("no GEMM {key} in the bundle")))?;
            let av = a.view((a_row + r) * s.lda, s.a_elems)?;
            let cv = c.view((c_row + r) * s.n, s.c_elems)?;
            self.run(&format!("gemm:{key}"), &[&av, w, &cv], t, &format!("npu:{g}"))?;
            r += n * self.chunk;
            left -= n;
        }
        Ok(())
    }

    /// The `gemm` record of group `g` at one chunk (its K, N, row stride).
    pub fn gemm_spec(&self, g: &str) -> Result<&GemmSpec, Error> {
        self.gemms.get(&format!("{g}.1")).ok_or_else(|| Error::Bundle(format!("no GEMM {g} in the bundle")))
    }

    pub fn mamba_spec(&self, phase: &str, seqs: usize, l: usize) -> Option<&MambaSpec> {
        self.mambas.get(&format!("{phase}.{seqs}x{l}"))
    }

    /// MambaScan `phase` at shape (seqs, l): `inp` -> `out` with the
    /// direction parameters `prm` (the caller has synced the input).
    pub fn mamba(&self, phase: &str, seqs: usize, l: usize, inp: &Flat, prm: &Buffer, out: &Flat, t: &mut Timing) -> Result<(), Error> {
        let key = format!("{phase}.{seqs}x{l}");
        let s = self.mambas.get(&key).ok_or_else(|| Error::Bundle(format!("no MambaScan {key} in the bundle")))?;
        let (iv, ov) = (inp.view(0, s.in_elems)?, out.view(0, s.out_elems)?);
        self.run(&format!("mamba:{key}"), &[&iv, prm, &ov], t, &format!("npu:{phase}"))
    }
}
