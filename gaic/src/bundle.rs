// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The bundle `iron/applications/gaic/export_gaic.py` writes: a plain-text
//! `manifest.txt` naming the compiled kernels, the packed weights and the
//! host-side parameters. See that script's docstring for the format.

use std::fs;
use std::path::{Path, PathBuf};

use crate::Error;

pub const VERSION: u32 = 1;

/// One compiled flm.GEMM: `C[M, N] = A[M, K] B[K, N]`, A read from
/// `a_elems` bf16 (rows may overlap), B pre-packed (`b_bytes`).
#[derive(Debug, Clone)]
pub struct KernelSpec {
    pub key: String,
    pub xclbin: PathBuf,
    pub insts: PathBuf,
    pub name: String,
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub a_elems: usize,
    pub b_bytes: usize,
    pub c_elems: usize,
}

/// One 3x3 conv (pad 1, stride 1) as a GEMM over `P` output pixels a row;
/// see `gaic_npu.ConvSpec`.
#[derive(Debug, Clone)]
pub struct ConvSpec {
    pub name: String,
    pub kernel: String,
    pub c: usize,
    pub oc: usize,
    pub pool_before: bool,
    pub m_chunk: usize,
    pub p: usize,
    /// The stem: A is a materialised im2col, P windows of 9C a row.
    pub window: bool,
    /// Y's row width (view) / one pixel's [dy][c] run (window, = 3C).
    pub d: usize,
    pub b: PathBuf,
    pub bias: PathBuf,
}

#[derive(Debug, Clone)]
pub struct Fc1Spec {
    pub kernel: String,
    pub b: PathBuf,
    pub bias: PathBuf,
    pub k: usize,
    pub k_pad: usize,
    pub n: usize,
    pub m: usize,
}

#[derive(Debug, Clone)]
pub struct RefSpec {
    pub image: String,
    pub w: usize,
    pub h: usize,
    pub n_anchors: usize,
    pub src_w: usize,
    pub src_h: usize,
}

#[derive(Debug, Clone)]
pub struct Manifest {
    pub dir: PathBuf,
    pub kernels: Vec<KernelSpec>,
    pub convs: Vec<ConvSpec>,
    pub f3: String,
    pub f4: String,
    pub dimred_w: PathBuf,
    pub dimred_b: PathBuf,
    pub reddim: usize,
    pub dimred_in: usize,
    pub fc1: Fc1Spec,
    pub fc2_w: PathBuf,
    pub fc2_b: PathBuf,
    pub fc2_in: usize,
    pub fc2_out: usize,
    pub fc3_w: PathBuf,
    pub fc3_b: PathBuf,
    pub align_size: usize,
    pub spatial_scale: f32,
    pub reference: Option<RefSpec>,
}

fn bad(line: usize, msg: impl std::fmt::Display) -> Error {
    Error::Bundle(format!("manifest.txt:{line}: {msg}"))
}

fn num<T: std::str::FromStr>(line: usize, s: &str) -> Result<T, Error> {
    s.parse().map_err(|_| bad(line, format!("not a number: {s:?}")))
}

impl Manifest {
    pub fn load(dir: &Path) -> Result<Self, Error> {
        let path = dir.join("manifest.txt");
        let text = fs::read_to_string(&path).map_err(|e| Error::Bundle(format!("{}: {e}", path.display())))?;
        let f = |s: &str| dir.join(s);
        let mut version = None;
        let mut kernels = Vec::new();
        let mut convs = Vec::new();
        let (mut f3, mut f4) = (None, None);
        let mut dimred = None;
        let mut fc1 = None;
        let mut fc2 = None;
        let mut fc3 = None;
        let mut align = None;
        let mut reference = None;
        for (i, raw) in text.lines().enumerate() {
            let ln = i + 1;
            let t: Vec<&str> = raw.split_whitespace().collect();
            let Some(&tag) = t.first() else { continue };
            let want = |n: usize| {
                if t.len() == n { Ok(()) } else { Err(bad(ln, format!("{tag}: {} fields, expected {n}", t.len()))) }
            };
            match tag {
                "gaic" => {
                    want(2)?;
                    version = Some(num::<u32>(ln, t[1])?);
                }
                "kernel" => {
                    want(11)?;
                    kernels.push(KernelSpec {
                        key: t[1].into(),
                        xclbin: f(t[2]),
                        insts: f(t[3]),
                        name: t[4].into(),
                        m: num(ln, t[5])?,
                        k: num(ln, t[6])?,
                        n: num(ln, t[7])?,
                        a_elems: num(ln, t[8])?,
                        b_bytes: num(ln, t[9])?,
                        c_elems: num(ln, t[10])?,
                    });
                }
                "conv" => {
                    want(12)?;
                    convs.push(ConvSpec {
                        name: t[1].into(),
                        kernel: t[2].into(),
                        c: num(ln, t[3])?,
                        oc: num(ln, t[4])?,
                        pool_before: num::<u8>(ln, t[5])? != 0,
                        m_chunk: num(ln, t[6])?,
                        p: num(ln, t[7])?,
                        window: num::<u8>(ln, t[8])? != 0,
                        d: num(ln, t[9])?,
                        b: f(t[10]),
                        bias: f(t[11]),
                    });
                }
                "f3" => {
                    want(2)?;
                    f3 = Some(t[1].to_string());
                }
                "f4" => {
                    want(2)?;
                    f4 = Some(t[1].to_string());
                }
                "dimred" => {
                    want(5)?;
                    dimred = Some((f(t[1]), f(t[2]), num(ln, t[3])?, num(ln, t[4])?));
                }
                "fc1" => {
                    want(8)?;
                    fc1 = Some(Fc1Spec {
                        kernel: t[1].into(),
                        b: f(t[2]),
                        bias: f(t[3]),
                        k: num(ln, t[4])?,
                        k_pad: num(ln, t[5])?,
                        n: num(ln, t[6])?,
                        m: num(ln, t[7])?,
                    });
                }
                "fc2" => {
                    want(5)?;
                    fc2 = Some((f(t[1]), f(t[2]), num(ln, t[3])?, num(ln, t[4])?));
                }
                "fc3" => {
                    want(4)?;
                    fc3 = Some((f(t[1]), f(t[2])));
                }
                "align" => {
                    want(3)?;
                    align = Some((num(ln, t[1])?, num(ln, t[2])?));
                }
                "ref" => {
                    want(7)?;
                    reference = Some(RefSpec {
                        image: t[1].into(),
                        w: num(ln, t[2])?,
                        h: num(ln, t[3])?,
                        n_anchors: num(ln, t[4])?,
                        src_w: num(ln, t[5])?,
                        src_h: num(ln, t[6])?,
                    });
                }
                _ => return Err(bad(ln, format!("unknown record {tag:?}"))),
            }
        }
        match version {
            Some(VERSION) => {}
            Some(v) => return Err(Error::Bundle(format!("bundle version {v}, this runtime reads {VERSION}"))),
            None => return Err(Error::Bundle("manifest.txt has no `gaic <version>` line".into())),
        }
        let missing = |what: &str| Error::Bundle(format!("manifest.txt has no {what} record"));
        let (dimred_w, dimred_b, reddim, dimred_in) = dimred.ok_or_else(|| missing("dimred"))?;
        let (fc2_w, fc2_b, fc2_in, fc2_out) = fc2.ok_or_else(|| missing("fc2"))?;
        let (fc3_w, fc3_b) = fc3.ok_or_else(|| missing("fc3"))?;
        let (align_size, spatial_scale) = align.ok_or_else(|| missing("align"))?;
        let m = Manifest {
            dir: dir.to_path_buf(),
            kernels,
            convs,
            f3: f3.ok_or_else(|| missing("f3"))?,
            f4: f4.ok_or_else(|| missing("f4"))?,
            dimred_w,
            dimred_b,
            reddim,
            dimred_in,
            fc1: fc1.ok_or_else(|| missing("fc1"))?,
            fc2_w,
            fc2_b,
            fc2_in,
            fc2_out,
            fc3_w,
            fc3_b,
            align_size,
            spatial_scale,
            reference,
        };
        m.validate()?;
        Ok(m)
    }

    pub fn kernel(&self, key: &str) -> Result<&KernelSpec, Error> {
        self.kernels
            .iter()
            .find(|k| k.key == key)
            .ok_or_else(|| Error::Bundle(format!("no kernel {key:?} in the manifest")))
    }

    /// The shapes the runtime's glue assumes, checked once up front so a
    /// mismatched bundle is an error rather than garbage.
    fn validate(&self) -> Result<(), Error> {
        let mut c_in = 3;
        for c in &self.convs {
            let k = self.kernel(&c.kernel)?;
            let err = |m: String| Err(Error::Bundle(format!("{}: {m}", c.name)));
            if c.c != c_in {
                return err(format!("takes {} channels, the previous layer makes {c_in}", c.c));
            }
            if k.m != c.m_chunk || k.n != c.p * c.oc {
                return err(format!(
                    "kernel {} is [{}, {}], expected [{}, {}]",
                    k.key,
                    k.m,
                    k.n,
                    c.m_chunk,
                    c.p * c.oc
                ));
            }
            let (want_k, want_a) =
                if c.window { (k.k, c.m_chunk * k.k) } else { ((c.p + 2) * c.d, (c.m_chunk * c.p + 2) * c.d) };
            if c.window && (c.d != 3 * c.c || c.p * 9 * c.c > k.k) {
                return err(format!("window layer with D {} / K {}", c.d, k.k));
            }
            if k.k != want_k || k.a_elems != want_a || k.c_elems != c.m_chunk * c.p * c.oc {
                return err(format!("kernel {} does not match the conv's layout", k.key));
            }
            if !c.window && c.d < 3 * c.c {
                return err(format!("D {} < 3C", c.d));
            }
            c_in = c.oc;
        }
        let k = self.kernel(&self.fc1.kernel)?;
        if k.m != self.fc1.m || k.k != self.fc1.k_pad || k.n != self.fc1.n {
            return Err(Error::Bundle("fc1 kernel shape mismatch".into()));
        }
        let s = self.align_size;
        if self.fc1.k != 2 * self.reddim * s * s || self.fc2_in != self.fc1.n {
            return Err(Error::Bundle("fc layer sizes do not chain".into()));
        }
        for name in [&self.f3, &self.f4] {
            if !self.convs.iter().any(|c| &c.name == name) {
                return Err(Error::Bundle(format!("f3/f4 name an unknown conv {name:?}")));
            }
        }
        Ok(())
    }
}

/// Little-endian f32s from a file (the exporter's `.f32`s).
pub fn read_f32(path: &Path) -> Result<Vec<f32>, Error> {
    let bytes = fs::read(path).map_err(|e| Error::Bundle(format!("{}: {e}", path.display())))?;
    if bytes.len() % 4 != 0 {
        return Err(Error::Bundle(format!("{}: {} bytes is not whole f32s", path.display(), bytes.len())));
    }
    Ok(bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect())
}

pub fn read_f32_n(path: &Path, n: usize) -> Result<Vec<f32>, Error> {
    let v = read_f32(path)?;
    if v.len() != n {
        return Err(Error::Bundle(format!("{}: {} values, expected {n}", path.display(), v.len())));
    }
    Ok(v)
}

pub fn read_bytes(path: &Path) -> Result<Vec<u8>, Error> {
    fs::read(path).map_err(|e| Error::Bundle(format!("{}: {e}", path.display())))
}
