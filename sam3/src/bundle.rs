// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The bundle `iron/applications/sam3/export_sam3.py` writes: `manifest.txt`
//! (kernels, model constants, tests), `tensors.txt` + `tensors.bin` (every
//! weight and reference tensor), `kernels/`, the tokenizer files and test
//! cases. See that script's docstring for the format.

use std::collections::HashMap;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

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
    pub params: HashMap<String, String>,
    /// key -> (xclbin path, kernel name)
    pub xclbins: HashMap<String, (PathBuf, String)>,
    pub gemms: HashMap<String, GemmSpec>,
    pub mhas: HashMap<String, MhaSpec>,
    pub ops: HashMap<String, OpSpec>,
    pub ref_image: Option<PathBuf>,
    pub ref_prompt: Option<String>,
    pub tok_tests: Vec<(usize, String)>,
    pub cases: Vec<Case>,
}

fn kv(fields: &[&str]) -> HashMap<String, String> {
    fields.iter().filter_map(|f| f.split_once('=')).map(|(k, v)| (k.to_string(), v.to_string())).collect()
}

fn bad(what: impl Into<String>) -> Error {
    Error::Bundle(what.into())
}

fn field<T: std::str::FromStr>(m: &HashMap<String, String>, k: &str, line: &str) -> Result<T, Error> {
    m.get(k).and_then(|v| v.parse().ok()).ok_or_else(|| bad(format!("`{k}=` missing or malformed in: {line}")))
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
        let path = dir.join("manifest.txt");
        let text = fs::read_to_string(&path).map_err(|e| bad(format!("{}: {e}", path.display())))?;
        let mut m = Manifest {
            dir: dir.to_path_buf(),
            params: HashMap::new(),
            xclbins: HashMap::new(),
            gemms: HashMap::new(),
            mhas: HashMap::new(),
            ops: HashMap::new(),
            ref_image: None,
            ref_prompt: None,
            tok_tests: Vec::new(),
            cases: Vec::new(),
        };
        let mut version = None;
        for line in text.lines() {
            let f: Vec<&str> = line.split_whitespace().collect();
            let Some(&tag) = f.first() else { continue };
            // the text after the first `n` fields, spacing intact
            let rest = |n: usize| -> String {
                let mut s = line;
                for _ in 0..n {
                    s = s.trim_start();
                    s = s.find(char::is_whitespace).map_or("", |i| &s[i..]);
                }
                s.strip_prefix(' ').unwrap_or(s).to_string()
            };
            match tag {
                "version" => version = f.get(1).and_then(|v| v.parse::<u32>().ok()),
                "param" if f.len() >= 3 => {
                    m.params.insert(f[1].to_string(), f[2].to_string());
                }
                "xclbin" if f.len() >= 4 => {
                    m.xclbins.insert(f[1].to_string(), (dir.join(f[2]), f[3].to_string()));
                }
                "gemm" if f.len() >= 2 => {
                    let a = kv(&f[2..]);
                    let g = GemmSpec {
                        key: f[1].to_string(),
                        ctx: a.get("ctx").cloned().ok_or_else(|| bad(format!("no ctx in: {line}")))?,
                        insts: dir.join(a.get("insts").ok_or_else(|| bad(format!("no insts in: {line}")))?),
                        m: field(&a, "M", line)?,
                        k: field(&a, "K", line)?,
                        n: field(&a, "N", line)?,
                        lda: field(&a, "lda", line)?,
                        tile_k: field(&a, "tile_k", line)?,
                        tile_n: field(&a, "tile_n", line)?,
                        ct_k: field(&a, "ct_k", line)?,
                        bias: field::<u8>(&a, "bias", line)? != 0,
                        b_bytes: field(&a, "b_bytes", line)?,
                    };
                    m.gemms.insert(g.key.clone(), g);
                }
                "mha" if f.len() >= 2 => {
                    let a = kv(&f[2..]);
                    let h = MhaSpec {
                        key: f[1].to_string(),
                        xclbin: dir.join(a.get("xclbin").ok_or_else(|| bad(format!("no xclbin in: {line}")))?),
                        insts: dir.join(a.get("insts").ok_or_else(|| bad(format!("no insts in: {line}")))?),
                        name: a.get("name").cloned().unwrap_or_else(|| "MLIR_AIE".into()),
                        heads: field(&a, "heads", line)?,
                        seq: field(&a, "seq", line)?,
                        d: field(&a, "d", line)?,
                        token_major: field::<u8>(&a, "token_major", line)? != 0,
                        elems: field(&a, "elems", line)?,
                    };
                    m.mhas.insert(h.key.clone(), h);
                }
                "op" if f.len() >= 2 => {
                    let a = kv(&f[2..]);
                    let o = OpSpec {
                        key: f[1].to_string(),
                        xclbin: dir.join(a.get("xclbin").ok_or_else(|| bad(format!("no xclbin in: {line}")))?),
                        insts: dir.join(a.get("insts").ok_or_else(|| bad(format!("no insts in: {line}")))?),
                        name: a.get("name").cloned().unwrap_or_else(|| "MLIR_AIE".into()),
                    };
                    m.ops.insert(o.key.clone(), o);
                }
                "ref_image" => m.ref_image = Some(dir.join(rest(1))),
                "ref_prompt" => m.ref_prompt = Some(rest(1)),
                "tok_test" if f.len() >= 2 => {
                    let j = f[1].parse().map_err(|_| bad(line))?;
                    m.tok_tests.push((j, unescape(&rest(2))));
                }
                "case" if f.len() >= 4 => {
                    let idx = f[1].parse().map_err(|_| bad(line))?;
                    m.cases.push(Case { idx, image: dir.join(f[2]), prompt: rest(3) });
                }
                _ => {}
            }
        }
        match version {
            Some(VERSION) => {}
            v => return Err(bad(format!("{}: bundle version {v:?}, this runtime reads {VERSION}", path.display()))),
        }
        for g in m.gemms.values() {
            if !m.xclbins.contains_key(&g.ctx) {
                return Err(bad(format!("gemm {} runs on unknown xclbin {}", g.key, g.ctx)));
            }
        }
        Ok(m)
    }

    pub fn param(&self, k: &str) -> Result<&str, Error> {
        self.params.get(k).map(String::as_str).ok_or_else(|| bad(format!("param {k} missing from the manifest")))
    }

    pub fn usize(&self, k: &str) -> Result<usize, Error> {
        self.param(k)?.parse().map_err(|_| bad(format!("param {k} is not an integer")))
    }

    pub fn f32(&self, k: &str) -> Result<f32, Error> {
        self.param(k)?.parse().map_err(|_| bad(format!("param {k} is not a number")))
    }

    pub fn list(&self, k: &str) -> Result<Vec<usize>, Error> {
        self.param(k)?
            .split(',')
            .filter(|s| !s.is_empty())
            .map(|s| s.parse().map_err(|_| bad(format!("param {k}: bad list"))))
            .collect()
    }

    pub fn gemm(&self, k: &str) -> Result<&GemmSpec, Error> {
        self.gemms.get(k).ok_or_else(|| bad(format!("kernel {k} missing from the manifest")))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    Bf16,
    U8,
    I32,
}

#[derive(Debug, Clone)]
pub struct Entry {
    pub dtype: DType,
    pub shape: Vec<usize>,
    off: usize,
    len: usize,
}

/// Every tensor of the bundle, in memory. The backing store is `u64`s so
/// the 64-byte-aligned offsets `tensors.txt` records are aligned in memory
/// too, and the typed views below are sound.
pub struct Store {
    data: Vec<u64>,
    map: HashMap<String, Entry>,
}

impl Store {
    pub fn load(dir: &Path) -> Result<Self, Error> {
        let idx = dir.join("tensors.txt");
        let text = fs::read_to_string(&idx).map_err(|e| bad(format!("{}: {e}", idx.display())))?;
        let mut map = HashMap::new();
        for line in text.lines() {
            let f: Vec<&str> = line.split_whitespace().collect();
            if f.len() != 5 {
                return Err(bad(format!("tensors.txt: {line}")));
            }
            let dtype = match f[1] {
                "f32" => DType::F32,
                "bf16" => DType::Bf16,
                "u8" => DType::U8,
                "i32" => DType::I32,
                d => return Err(bad(format!("tensors.txt: dtype {d}"))),
            };
            let shape = f[2].split(',').map(|s| s.parse().map_err(|_| bad(line))).collect::<Result<_, _>>()?;
            let off: usize = f[3].parse().map_err(|_| bad(line))?;
            let len: usize = f[4].parse().map_err(|_| bad(line))?;
            if off % 8 != 0 {
                return Err(bad(format!("tensors.txt: {} is not 8-byte aligned", f[0])));
            }
            map.insert(f[0].to_string(), Entry { dtype, shape, off, len });
        }
        let bin = dir.join("tensors.bin");
        let mut file = fs::File::open(&bin).map_err(|e| bad(format!("{}: {e}", bin.display())))?;
        let bytes = file.metadata().map_err(|e| bad(e.to_string()))?.len() as usize;
        let mut data = vec![0u64; bytes.div_ceil(8)];
        // SAFETY: a u64 buffer viewed as its bytes.
        let raw = unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, bytes) };
        file.read_exact(raw).map_err(|e| bad(format!("{}: {e}", bin.display())))?;
        for (name, e) in &map {
            if e.off + e.len > bytes {
                return Err(bad(format!("tensor {name} runs past the end of tensors.bin")));
            }
        }
        Ok(Store { data, map })
    }

    pub fn has(&self, name: &str) -> bool {
        self.map.contains_key(name)
    }

    pub fn entry(&self, name: &str) -> Result<&Entry, Error> {
        self.map.get(name).ok_or_else(|| bad(format!("tensor {name} missing from the bundle")))
    }

    pub fn shape(&self, name: &str) -> Result<&[usize], Error> {
        Ok(&self.entry(name)?.shape)
    }

    fn typed<T: Copy>(&self, name: &str, want: DType) -> Result<&[T], Error> {
        let e = self.entry(name)?;
        if e.dtype != want {
            return Err(bad(format!("tensor {name} is {:?}, wanted {want:?}", e.dtype)));
        }
        // SAFETY: in bounds (checked at load), aligned (8-byte offsets into a
        // u64 buffer), and every bit pattern is a valid f32/u16/u8/i32.
        Ok(unsafe {
            std::slice::from_raw_parts(
                (self.data.as_ptr() as *const u8).add(e.off) as *const T,
                e.len / std::mem::size_of::<T>(),
            )
        })
    }

    /// Any tensor's raw bytes.
    pub fn bytes(&self, name: &str) -> Result<&[u8], Error> {
        let e = self.entry(name)?;
        // SAFETY: in bounds (checked at load).
        Ok(unsafe { std::slice::from_raw_parts((self.data.as_ptr() as *const u8).add(e.off), e.len) })
    }

    pub fn f32(&self, name: &str) -> Result<&[f32], Error> {
        self.typed(name, DType::F32)
    }

    /// bf16 as raw bits.
    pub fn bf16(&self, name: &str) -> Result<&[u16], Error> {
        self.typed(name, DType::Bf16)
    }

    pub fn u8(&self, name: &str) -> Result<&[u8], Error> {
        self.typed(name, DType::U8)
    }

    pub fn i32(&self, name: &str) -> Result<&[i32], Error> {
        self.typed(name, DType::I32)
    }
}
