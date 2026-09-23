// SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
// SPDX-License-Identifier: Apache-2.0

//! The NPU side: every kernel of the bundle loaded once (kernels naming the
//! same xclbin share its hardware context -- 13 in all, under NPU2's 16),
//! and the device buffers they run over.
//!
//! A GEMM is compiled for `M` rows; a stage with more rows owns one `A` and
//! one `C` buffer of the padded height and runs the kernel once per
//! `M`-row chunk over sub-buffer views of them, as the Python app does.

use std::collections::HashMap;
use std::time::Duration;

// The NPU path: XRT (feature `xrt`, the default), or the driver's ioctls
// with no XRT (feature `direct`, which wins when both are on).
#[cfg(feature = "direct")]
pub use taconite::direct::{Buffer, Kernel, Session};
#[cfg(all(feature = "xrt", not(feature = "direct")))]
pub use taconite::{Buffer, Kernel, Session};
#[cfg(not(any(feature = "xrt", feature = "direct")))]
compile_error!("no NPU path: enable feature `xrt` (the default) or `direct`");

use crate::Error;
use crate::bundle::{GemmSpec, Manifest, MhaSpec};
use crate::cpu::par_rows;

pub fn round_up(x: usize, m: usize) -> usize {
    x.div_ceil(m) * m
}

const BF16: usize = 2;

/// Where a kernel comes from: xclbin (its hardware context), instruction
/// stream, kernel name, ops per run.
struct Source {
    ctx: String,
    xclbin: std::path::PathBuf,
    insts: std::path::PathBuf,
    name: String,
    ops: u64,
}

pub struct Npu {
    pub session: Session,
    sources: HashMap<String, Source>,
    loaded: HashMap<String, Kernel>,
    /// resident contexts, least recently used first
    lru: Vec<String>,
    /// contexts created / evicted to make room (another process may hold
    /// some of NPU2's 16)
    pub loads: usize,
    pub evictions: usize,
    /// at most this many resident contexts (`SAM3_MAX_CONTEXTS`; default:
    /// as many as the driver grants)
    cap: usize,
    pub gemms: HashMap<String, GemmSpec>,
    pub mhas: HashMap<String, MhaSpec>,
}

/// A GEMM's operands for a given row count: `a` (rows `lda` apart) and `c`,
/// and the per-dispatch views of both.
pub struct Io {
    pub key: String,
    pub a: Buffer,
    pub c: Buffer,
    chunks: Vec<(Buffer, Buffer)>,
    /// `a` is another GEMM's `c` (device-produced; never written here)
    chained: bool,
    pub rows: usize,
    pub n: usize,
}

/// Copies `src` into the start of `b` with wide, parallel copies (`b`'s
/// host mapping is uncached: element-wise access to it is many times
/// slower than streaming whole rows) and syncs it to the device.
pub fn push(src: &[u16], b: &mut Buffer) -> Result<(), Error> {
    let dst = &mut b.as_mut_slice::<u16>()[..src.len()];
    par_rows(dst, COPY, |r0, piece| piece.copy_from_slice(&src[r0 * COPY..][..piece.len()]));
    Ok(b.sync_to_device()?)
}

/// The first `n` bf16 of `b` (synced from the device) in cached memory.
pub fn pull(b: &Buffer, n: usize) -> Result<Vec<u16>, Error> {
    b.sync_from_device()?;
    let src = &b.as_slice::<u16>()[..n];
    let mut dst = vec![0u16; n];
    par_rows(&mut dst, COPY, |r0, piece| piece.copy_from_slice(&src[r0 * COPY..][..piece.len()]));
    Ok(dst)
}

const COPY: usize = 1 << 16;

impl Io {
    /// Fills the start of A (bf16 bits) and syncs it.
    pub fn set_a(&mut self, src: &[u16]) -> Result<(), Error> {
        assert!(!self.chained, "{}: A is another kernel's output", self.key);
        push(src, &mut self.a)
    }

    pub fn sync_a(&self) -> Result<(), Error> {
        Ok(self.a.sync_to_device()?)
    }

    /// The first `rows` rows of C (bf16 bits), copied out.
    pub fn get_c(&self, rows: usize) -> Result<Vec<u16>, Error> {
        pull(&self.c, rows * self.n)
    }
}

pub struct MhaIo {
    pub key: String,
    pub q: Buffer,
    pub k: Buffer,
    pub v: Buffer,
    pub o: Buffer,
}

impl Npu {
    /// Opens the NPU and loads every kernel whose context fits; the rest
    /// load on first use (see `Npu::kernel`).
    pub fn open(m: &Manifest) -> Result<Self, Error> {
        let session = Session::open(0)?;
        let mut sources = HashMap::new();
        let mut order = vec![];
        for g in m.gemms.values() {
            let x = m.xclbin(&g.ctx)?;
            let ops = (2 * g.m * g.k * g.n) as u64;
            let s = Source {
                ctx: g.ctx.clone(),
                xclbin: x.path.clone(),
                insts: g.insts.clone(),
                name: x.kernel.clone(),
                ops,
            };
            sources.insert(g.key.clone(), s);
            order.push(g.key.clone());
        }
        for o in m.ops.values() {
            let s = Source {
                ctx: o.key.clone(),
                xclbin: o.xclbin.clone(),
                insts: o.insts.clone(),
                name: o.name.clone(),
                ops: 0,
            };
            sources.insert(o.key.clone(), s);
            order.push(o.key.clone());
        }
        for h in m.mhas.values() {
            let ops = (4 * h.heads * h.seq * h.seq * h.d) as u64;
            let s = Source {
                ctx: h.key.clone(),
                xclbin: h.xclbin.clone(),
                insts: h.insts.clone(),
                name: h.name.clone(),
                ops,
            };
            sources.insert(h.key.clone(), s);
            order.push(h.key.clone());
        }
        let mut npu = Npu {
            session,
            sources,
            loaded: HashMap::new(),
            lru: vec![],
            loads: 0,
            evictions: 0,
            cap: std::env::var("SAM3_MAX_CONTEXTS").ok().and_then(|v| v.parse().ok()).unwrap_or(usize::MAX),
            gemms: m.gemms.clone(),
            mhas: m.mhas.clone(),
        };
        order.sort();
        for key in order {
            let ctx = &npu.sources[&key].ctx;
            if !npu.lru.contains(ctx) && npu.lru.len() >= npu.cap {
                break;
            }
            match npu.try_load(&key) {
                Ok(()) => {}
                Err(e) if is_full(&e) => break,
                Err(e) => return Err(Error::Npu(format!("loading {key}: {e}"))),
            }
        }
        Ok(npu)
    }

    fn try_load(&mut self, key: &str) -> Result<(), taconite::Error> {
        let s = &self.sources[key];
        let k = self.session.load_kernel(&s.xclbin, &s.insts, Some(&s.name), s.ops)?;
        let ctx = s.ctx.clone();
        self.loaded.insert(key.to_string(), k);
        if !self.lru.contains(&ctx) {
            self.lru.push(ctx);
            self.loads += 1;
        }
        Ok(())
    }

    /// Kernel `key`, loaded if it is not resident -- evicting the least
    /// recently used contexts while the device has no room for its own.
    fn kernel(&mut self, key: &str) -> Result<&Kernel, Error> {
        let ctx = self.sources.get(key).ok_or_else(|| Error::Bundle(format!("no kernel {key}")))?.ctx.clone();
        if !self.loaded.contains_key(key) {
            while !self.lru.contains(&ctx) && self.lru.len() >= self.cap {
                self.evict_lru(&ctx);
            }
            loop {
                match self.try_load(key) {
                    Ok(()) => break,
                    Err(e) if is_full(&e) => {
                        if !self.evict_lru(&ctx) {
                            return Err(Error::Npu(format!("{key}: no hardware context available: {e}")));
                        }
                    }
                    Err(e) => return Err(Error::Npu(format!("loading {key}: {e}"))),
                }
            }
        }
        if let Some(i) = self.lru.iter().position(|c| *c == ctx) {
            let c = self.lru.remove(i);
            self.lru.push(c);
        }
        Ok(&self.loaded[key])
    }

    pub fn spec(&self, key: &str) -> Result<&GemmSpec, Error> {
        self.gemms.get(key).ok_or_else(|| Error::Bundle(format!("no kernel {key}")))
    }

    /// A device buffer holding `bytes` (packed weights).
    pub fn upload(&self, bytes: &[u8]) -> Result<Buffer, Error> {
        let mut b = self.session.alloc(bytes.len())?;
        b.write(bytes)?;
        Ok(b)
    }

    /// A weight buffer sized for `key`'s packed B, to be filled per forward.
    pub fn weight_slot(&self, key: &str) -> Result<Buffer, Error> {
        Ok(self.session.alloc(self.spec(key)?.b_bytes)?)
    }

    /// Operands of `key` for `rows` rows (plain row-major A, `lda == K`).
    pub fn io(&self, key: &str, rows: usize) -> Result<Io, Error> {
        let g = self.spec(key)?;
        assert_eq!(g.lda, g.k, "{key}: overlapping A, use conv_io");
        let rows = round_up(rows, g.m);
        let a = self.session.alloc(rows * g.k * BF16)?;
        let c = self.session.alloc(rows * g.n * BF16)?;
        let mut chunks = Vec::new();
        for i in 0..rows / g.m {
            chunks
                .push((a.sub(i * g.m * g.k * BF16, g.m * g.k * BF16)?, c.sub(i * g.m * g.n * BF16, g.m * g.n * BF16)?));
        }
        Ok(Io { key: key.into(), a, c, chunks, chained: false, rows, n: g.n })
    }

    /// Operands of `key` whose A is `src`'s C, read in place.
    pub fn io_chained(&self, key: &str, src: &Io) -> Result<Io, Error> {
        let g = self.spec(key)?;
        assert_eq!(src.n, g.k, "{key}: A width {} != K {}", src.n, g.k);
        let rows = src.rows;
        let a = src.c.sub(0, rows * g.k * BF16)?;
        let c = self.session.alloc(rows * g.n * BF16)?;
        let mut chunks = Vec::new();
        for i in 0..rows / g.m {
            chunks.push((
                src.c.sub(i * g.m * g.k * BF16, g.m * g.k * BF16)?,
                c.sub(i * g.m * g.n * BF16, g.m * g.n * BF16)?,
            ));
        }
        Ok(Io { key: key.into(), a, c, chunks, chained: true, rows, n: g.n })
    }

    /// Operands of the overlapping-view 3x3 convolution `key` for an
    /// `h x w` image: A is `Y` (`(rows * P + 2) x D`, see neck.rs), each
    /// dispatch's view reaching `(M - 1) * lda + K` elements on.
    pub fn conv_io(&self, key: &str, h: usize, w: usize) -> Result<Io, Error> {
        let g = self.spec(key)?;
        let d = (g.k - g.lda) / 2; // K = (P + 2) D, lda = P D
        let p = g.lda / d;
        let rows = round_up((h * (w + 2)).div_ceil(p), g.m);
        let a = self.session.alloc((rows * p + 2) * d * BF16)?;
        let c = self.session.alloc(rows * g.n * BF16)?;
        let ext = (g.m - 1) * g.lda + g.k;
        let mut chunks = Vec::new();
        for i in 0..rows / g.m {
            chunks.push((a.sub(i * g.m * g.lda * BF16, ext * BF16)?, c.sub(i * g.m * g.n * BF16, g.m * g.n * BF16)?));
        }
        Ok(Io { key: key.into(), a, c, chunks, chained: false, rows, n: g.n })
    }

    pub fn mha_io(&self, key: &str) -> Result<MhaIo, Error> {
        let h = self.mhas.get(key).ok_or_else(|| Error::Bundle(format!("no MHA {key}")))?;
        let buf = || self.session.alloc(h.elems * BF16);
        Ok(MhaIo { key: key.into(), q: buf()?, k: buf()?, v: buf()?, o: buf()? })
    }

    /// Drops the least recently used context other than `keep` (and its
    /// kernels); false if there is none.
    fn evict_lru(&mut self, keep: &str) -> bool {
        let Some(victim) = self.lru.iter().find(|c| *c != keep).cloned() else {
            return false;
        };
        self.loaded.retain(|k, _| self.sources[k].ctx != victim);
        self.lru.retain(|c| *c != victim);
        self.evictions += 1;
        true
    }

    /// Runs `io`'s kernel over every chunk with weights `w`; A must be synced.
    pub fn run(&mut self, io: &Io, w: &Buffer) -> Result<Duration, Error> {
        let k = self.kernel(&io.key)?;
        let mut t = Duration::ZERO;
        for (a, c) in &io.chunks {
            t += k.run(&[a, w, c]).map_err(|e| Error::Npu(format!("{}: {e}", io.key)))?;
        }
        Ok(t)
    }

    /// Runs `io` with its A synced first -- unless A is another kernel's
    /// output, where a sync would push the host's stale copy over it.
    pub fn run_synced(&mut self, io: &Io, w: &Buffer) -> Result<Duration, Error> {
        if !io.chained {
            io.sync_a()?;
        }
        self.run(io, w)
    }

    /// Runs kernel `key` once over `args` (in its argument order), none
    /// synced: for kernels chained on device-produced buffers.
    pub fn run_args(&mut self, key: &str, args: &[&Buffer]) -> Result<Duration, Error> {
        self.kernel(key)?.run(args).map_err(|e| Error::Npu(format!("{key}: {e}")))
    }

    /// Runs the MHA over its q/k/v (already [`push`]ed).
    pub fn run_mha(&mut self, m: &MhaIo) -> Result<Duration, Error> {
        self.kernel(&m.key)?.run(&[&m.q, &m.k, &m.v, &m.o]).map_err(|e| Error::Npu(format!("{}: {e}", m.key)))
    }
}

/// The driver's answer when every hardware context is taken.
fn is_full(e: &taconite::Error) -> bool {
    e.to_string().contains("CREATE_HWCTX")
}
