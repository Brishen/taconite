// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The NPU side: every kernel of the bundle loaded once, kernels naming the
//! same xclbin sharing its hardware context (8 in all, of NPU2's 16), and
//! the device buffers they run over.
//!
//! A GEMM is compiled for `M` rows; a stage with more rows owns one `A` and
//! one `C` buffer of the padded height and runs the kernel once per
//! `M`-row chunk over sub-buffer views of them, as the Python app does.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

use iron_bundle::Manifest;
use iron_xrt::{Buffer, Kernel, Session};

use crate::Error;

pub const BF16: usize = 2;

/// One compiled `flm.GEMM`: `C[M, N] = A[M, K] B[K, N]` (+ bias), B
/// pre-packed (`b_bytes`).
#[derive(Debug, Clone)]
pub struct GemmSpec {
    pub key: String,
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub b_bytes: usize,
}

pub struct Npu {
    pub session: Session,
    kernels: HashMap<String, Kernel>,
    pub gemms: HashMap<String, GemmSpec>,
    /// distinct hardware contexts (xclbins) loaded
    pub contexts: usize,
}

/// A GEMM's operands for `rows` rows (a multiple of M): `a`, `c`, and each
/// dispatch's views of both.
pub struct Io {
    pub key: String,
    pub a: Buffer,
    pub c: Buffer,
    chunks: Vec<(Buffer, Buffer)>,
    pub rows: usize,
    pub k: usize,
    pub n: usize,
}

impl Npu {
    /// Opens the NPU and loads every `gemm` and `op` of the manifest.
    pub fn open(m: &Manifest) -> Result<Self, Error> {
        let session = Session::open(0)?;
        let mut kernels = HashMap::new();
        let mut gemms = HashMap::new();
        let mut ctxs: Vec<PathBuf> = Vec::new();
        for r in m.records() {
            let (key, xclbin, insts, name, ops) = match r.tag.as_str() {
                "gemm" => {
                    let g = GemmSpec {
                        key: r.field(0)?.to_string(),
                        m: r.get("M")?,
                        k: r.get("K")?,
                        n: r.get("N")?,
                        b_bytes: r.get("b_bytes")?,
                    };
                    let x = m.xclbin(r.str("ctx")?)?;
                    let ops = (2 * g.m * g.k * g.n) as u64;
                    gemms.insert(g.key.clone(), g);
                    (r.field(0)?, x.path.clone(), m.path(r.str("insts")?), x.kernel.clone(), ops)
                }
                "op" => (r.field(0)?, m.path(r.str("xclbin")?), m.path(r.str("insts")?), r.str("name")?.to_string(), 0),
                _ => continue,
            };
            let k = session
                .load_kernel(&xclbin, &insts, Some(&name), ops)
                .map_err(|e| Error::Npu(format!("loading {key}: {e}")))?;
            if !ctxs.contains(&xclbin) {
                ctxs.push(xclbin);
            }
            kernels.insert(key.to_string(), k);
        }
        Ok(Npu { session, kernels, gemms, contexts: ctxs.len() })
    }

    pub fn spec(&self, key: &str) -> Result<&GemmSpec, Error> {
        self.gemms.get(key).ok_or_else(|| Error::Bundle(format!("no GEMM {key} in the bundle")))
    }

    /// A device buffer holding `bytes` (packed weights).
    pub fn upload(&self, bytes: &[u8]) -> Result<Buffer, Error> {
        let mut b = self.session.alloc(bytes.len())?;
        b.write(bytes)?;
        Ok(b)
    }

    /// A zeroed device buffer of `elems` bf16.
    pub fn zeros(&self, elems: usize) -> Result<Buffer, Error> {
        let b = self.session.alloc(elems * BF16)?;
        b.sync_to_device()?;
        Ok(b)
    }

    fn chunks(&self, g: &GemmSpec, a: &Buffer, c: &Buffer, rows: usize) -> Result<Vec<(Buffer, Buffer)>, Error> {
        if rows % g.m != 0 {
            return Err(Error::Bundle(format!("{}: {rows} rows is not a multiple of M = {}", g.key, g.m)));
        }
        (0..rows / g.m)
            .map(|i| {
                Ok((a.sub(i * g.m * g.k * BF16, g.m * g.k * BF16)?, c.sub(i * g.m * g.n * BF16, g.m * g.n * BF16)?))
            })
            .collect()
    }

    /// Operands of `key` for `rows` rows, zeroed.
    pub fn io(&self, key: &str, rows: usize) -> Result<Io, Error> {
        let g = self.spec(key)?.clone();
        let a = self.zeros(rows * g.k)?;
        let c = self.zeros(rows * g.n)?;
        let chunks = self.chunks(&g, &a, &c, rows)?;
        Ok(Io { key: key.into(), a, c, chunks, rows, k: g.k, n: g.n })
    }

    /// Operands of `key` whose A is `src`'s C, read in place.
    pub fn io_chained(&self, key: &str, src: &Io) -> Result<Io, Error> {
        let g = self.spec(key)?.clone();
        if src.n != g.k {
            return Err(Error::Bundle(format!("{key}: A width {} != K {}", src.n, g.k)));
        }
        let a = src.c.sub(0, src.rows * g.k * BF16)?;
        let c = self.zeros(src.rows * g.n)?;
        let chunks = self.chunks(&g, &a, &c, src.rows)?;
        Ok(Io { key: key.into(), a, c, chunks, rows: src.rows, k: g.k, n: g.n })
    }

    fn kernel(&self, key: &str) -> Result<&Kernel, Error> {
        self.kernels.get(key).ok_or_else(|| Error::Bundle(format!("no kernel {key} in the bundle")))
    }

    /// Runs `io`'s kernel over every chunk with weights `w` (A as the
    /// device has it).
    pub fn gemm(&self, io: &Io, w: &Buffer) -> Result<Duration, Error> {
        let k = self.kernel(&io.key)?;
        let mut t = Duration::ZERO;
        for (a, c) in &io.chunks {
            t += k.run(&[a, w, c]).map_err(|e| Error::Npu(format!("{}: {e}", io.key)))?;
        }
        Ok(t)
    }

    /// Runs kernel `key` once over `args`, in its argument order.
    pub fn op(&self, key: &str, args: &[&Buffer]) -> Result<Duration, Error> {
        self.kernel(key)?.run(args).map_err(|e| Error::Npu(format!("{key}: {e}")))
    }
}
