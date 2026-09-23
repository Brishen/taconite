// SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
// SPDX-License-Identifier: Apache-2.0

//! The bundle `iron/applications/sam3/export_sam3.py` writes: `manifest.txt`
//! (kernels, model constants, tests), `tensors.txt` + `tensors.bin` (every
//! weight and reference tensor), `kernels/`, the tokenizer files and test
//! cases. See that script's docstring for the format; the records every
//! IRON bundle shares (`version`, `param`, `xclbin`) and the tensor store
//! are `taconite-bundle`'s.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub use taconite_bundle::{DType, Entry, Store, Xclbin};

use crate::Error;

pub const VERSION: u32 = 1;

/// One compiled `flm.GEMM`: `C[M, N] = A[M, K] B[K, N]` (+ bias), A rows
/// `lda` apart (overlapping when `lda < K`), B pre-packed (`b_bytes`).
#[derive(Debug, Clone)]
pub struct GemmSpec {
    pub key: String,
    /// The xclbin (hardware context) this kernel runs on.
    pub ctx: String,
    pub insts: PathBuf,
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub lda: usize,
    pub tile_k: usize,
    pub tile_n: usize,
    pub ct_k: usize,
    pub bias: bool,
    pub b_bytes: usize,
}

/// One compiled non-causal MHA: q/k/v/o are `elems` bf16 each, laid out
/// `[heads, seq, d]` or, `token_major`, `[seq, heads * d]`.
#[derive(Debug, Clone)]
pub struct MhaSpec {
    pub key: String,
    pub xclbin: PathBuf,
    pub insts: PathBuf,
    pub name: String,
    pub heads: usize,
    pub seq: usize,
    pub d: usize,
    pub token_major: bool,
    pub elems: usize,
}

/// Any other compiled kernel, launched over buffers the runtime already
/// holds (the device-resident ViT's RoPE and residual add + LayerNorm).
#[derive(Debug, Clone)]
pub struct OpSpec {
    pub key: String,
    pub xclbin: PathBuf,
    pub insts: PathBuf,
    pub name: String,
}

/// An end-to-end test case: an image, a prompt and the reference
/// instances (`case.<idx>.*` tensors).
#[derive(Debug, Clone)]
pub struct Case {
    pub idx: usize,
    pub image: PathBuf,
    pub prompt: String,
}

#[derive(Debug, Clone)]
pub struct Manifest {
    pub dir: PathBuf,
    /// The `version`, `param` and `xclbin` records.
    pub base: taconite_bundle::Manifest,
    pub gemms: HashMap<String, GemmSpec>,
    pub mhas: HashMap<String, MhaSpec>,
    pub ops: HashMap<String, OpSpec>,
    pub ref_image: Option<PathBuf>,
    pub ref_prompt: Option<String>,
    pub tok_tests: Vec<(usize, String)>,
    pub cases: Vec<Case>,
}

/// Python's `unicode_escape` of a prompt, undone (the escapes it writes for
/// text: `\t \n \r \\ \xNN \uNNNN \UNNNNNNNN`).
pub fn unescape(s: &str) -> String {
    let mut out = String::new();
    let mut it = s.chars();
    while let Some(c) = it.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        let hex = |it: &mut std::str::Chars, n: usize| -> Option<char> {
            let h: String = it.by_ref().take(n).collect();
            u32::from_str_radix(&h, 16).ok().and_then(char::from_u32)
        };
        match it.next() {
            Some('t') => out.push('\t'),
            Some('n') => out.push('\n'),
            Some('r') => out.push('\r'),
            Some('\\') => out.push('\\'),
            Some('x') => out.extend(hex(&mut it, 2)),
            Some('u') => out.extend(hex(&mut it, 4)),
            Some('U') => out.extend(hex(&mut it, 8)),
            Some(o) => {
                out.push('\\');
                out.push(o);
            }
            None => out.push('\\'),
        }
    }
    out
}

impl Manifest {
    pub fn load(dir: &Path) -> Result<Self, Error> {
        let base = taconite_bundle::Manifest::load(dir, VERSION)?;
        let mut m = Manifest {
            dir: dir.to_path_buf(),
            base: base.clone(),
            gemms: HashMap::new(),
            mhas: HashMap::new(),
            ops: HashMap::new(),
            ref_image: None,
            ref_prompt: None,
            tok_tests: Vec::new(),
            cases: Vec::new(),
        };
        let name = |r: &taconite_bundle::Record| r.str("name").unwrap_or("MLIR_AIE").to_string();
        for r in base.records() {
            match r.tag.as_str() {
                "gemm" => {
                    let g = GemmSpec {
                        key: r.field(0)?.to_string(),
                        ctx: r.str("ctx")?.to_string(),
                        insts: base.path(r.str("insts")?),
                        m: r.get("M")?,
                        k: r.get("K")?,
                        n: r.get("N")?,
                        lda: r.get("lda")?,
                        tile_k: r.get("tile_k")?,
                        tile_n: r.get("tile_n")?,
                        ct_k: r.get("ct_k")?,
                        bias: r.flag("bias")?,
                        b_bytes: r.get("b_bytes")?,
                    };
                    if base.xclbin(&g.ctx).is_err() {
                        return Err(r.error(format!("gemm {} runs on unknown xclbin {}", g.key, g.ctx)).into());
                    }
                    m.gemms.insert(g.key.clone(), g);
                }
                "mha" => {
                    let h = MhaSpec {
                        key: r.field(0)?.to_string(),
                        xclbin: base.path(r.str("xclbin")?),
                        insts: base.path(r.str("insts")?),
                        name: name(r),
                        heads: r.get("heads")?,
                        seq: r.get("seq")?,
                        d: r.get("d")?,
                        token_major: r.flag("token_major")?,
                        elems: r.get("elems")?,
                    };
                    m.mhas.insert(h.key.clone(), h);
                }
                "op" => {
                    let o = OpSpec {
                        key: r.field(0)?.to_string(),
                        xclbin: base.path(r.str("xclbin")?),
                        insts: base.path(r.str("insts")?),
                        name: name(r),
                    };
                    m.ops.insert(o.key.clone(), o);
                }
                "ref_image" => m.ref_image = Some(dir.join(r.rest(0))),
                "ref_prompt" => m.ref_prompt = Some(r.rest(0)),
                "tok_test" => m.tok_tests.push((r.field_as(0)?, unescape(&r.rest(1)))),
                "case" => {
                    r.field(2)?;
                    m.cases.push(Case { idx: r.field_as(0)?, image: dir.join(r.field(1)?), prompt: r.rest(2) });
                }
                _ => {}
            }
        }
        Ok(m)
    }

    pub fn param(&self, k: &str) -> Result<&str, Error> {
        Ok(self.base.param(k)?)
    }

    pub fn usize(&self, k: &str) -> Result<usize, Error> {
        Ok(self.base.param_as(k)?)
    }

    pub fn f32(&self, k: &str) -> Result<f32, Error> {
        Ok(self.base.param_as(k)?)
    }

    pub fn list(&self, k: &str) -> Result<Vec<usize>, Error> {
        Ok(self.base.list(k)?)
    }

    pub fn xclbin(&self, k: &str) -> Result<&Xclbin, Error> {
        Ok(self.base.xclbin(k)?)
    }

    pub fn gemm(&self, k: &str) -> Result<&GemmSpec, Error> {
        self.gemms.get(k).ok_or_else(|| Error::Bundle(format!("kernel {k} missing from the manifest")))
    }
}
