// SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
// SPDX-License-Identifier: Apache-2.0

//! The bundle `iron/applications/gaic/export_gaic.py` writes, in the format
//! every IRON bundle shares (read with [`taconite_bundle`]): `manifest.txt`
//! naming the compiled kernels and the layer table, the packed weights,
//! host-side parameters and the self-check's references in the tensor
//! store. See that script's docstring for every record and tensor.
//!
//! [`Bundle::load`] parses every record into the typed specs below and
//! checks them against each other and against the tensors, so a mismatched
//! bundle is an error naming the manifest line rather than garbage later.

use std::path::{Path, PathBuf};

use taconite_bundle::{DType, Manifest, Record, Store};

use crate::Error;

/// The bundle format version this runtime reads (`export_gaic.VERSION`).
pub const VERSION: u32 = 1;

impl From<taconite_bundle::Error> for Error {
    fn from(e: taconite_bundle::Error) -> Self {
        Error::Bundle(e.to_string())
    }
}

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
    /// Y's row width (view) / one pixel's `[dy][c]` run (window, = 3C).
    pub d: usize,
    /// Tensor: the packed B.
    pub b: String,
    /// Tensor: `[OC]` f32.
    pub bias: String,
}

#[derive(Debug, Clone)]
pub struct Fc1Spec {
    pub kernel: String,
    /// Tensor: the packed B.
    pub b: String,
    /// Tensor: `[n]` f32.
    pub bias: String,
    pub k: usize,
    pub k_pad: usize,
    pub n: usize,
    pub m: usize,
}

/// The self-check image; its data are the `ref.*` tensors.
#[derive(Debug, Clone)]
pub struct RefSpec {
    pub image: String,
    pub w: usize,
    pub h: usize,
    pub n_anchors: usize,
    pub src_w: usize,
    pub src_h: usize,
}

pub struct Bundle {
    pub dir: PathBuf,
    pub kernels: Vec<KernelSpec>,
    pub convs: Vec<ConvSpec>,
    pub f3: String,
    pub f4: String,
    /// Tensors: `[reddim, dimred_in]` / `[reddim]` f32.
    pub dimred_w: String,
    pub dimred_b: String,
    pub reddim: usize,
    pub dimred_in: usize,
    pub fc1: Fc1Spec,
    /// Tensors: `[fc2_out, fc2_in]` / `[fc2_out]` / `[fc2_out]` / `[1]` f32.
    pub fc2_w: String,
    pub fc2_b: String,
    pub fc2_in: usize,
    pub fc2_out: usize,
    pub fc3_w: String,
    pub fc3_b: String,
    pub align_size: usize,
    pub spatial_scale: f32,
    pub reference: Option<RefSpec>,
    /// Every tensor the records name.
    pub store: Store,
}

fn once<T>(slot: &mut Option<T>, r: &Record, v: T) -> Result<(), Error> {
    if slot.replace(v).is_some() {
        return Err(r.error(format!("second {} record", r.tag)).into());
    }
    Ok(())
}

impl Bundle {
    /// Read and check `<dir>`'s manifest and tensors.
    pub fn load(dir: &Path) -> Result<Self, Error> {
        let manifest_path = dir.join(Manifest::FILE);
        if !dir.join("tensors.txt").exists()
            && std::fs::read_to_string(&manifest_path).is_ok_and(|t| t.starts_with("gaic "))
        {
            return Err(Error::Bundle(format!(
                "{} is an old-format GAIC bundle (`gaic <version>` manifest, one file a tensor); \
                 re-export it with python -m iron.applications.gaic.export_gaic",
                dir.display()
            )));
        }
        let m = Manifest::load(dir, VERSION)?;
        let store = Store::load(dir)?;

        let mut kernels: Vec<KernelSpec> = Vec::new();
        let mut convs: Vec<ConvSpec> = Vec::new();
        let (mut dimred, mut fc1, mut fc2, mut fc3, mut reference) = (None, None, None, None, None);
        for r in m.records() {
            match r.tag.as_str() {
                "kernel" => {
                    let key = r.field(0)?.to_string();
                    if kernels.iter().any(|k| k.key == key) {
                        return Err(r.error(format!("kernel {key} defined twice")).into());
                    }
                    let x = m.xclbin(r.str("ctx")?).map_err(|e| r.error(e))?;
                    kernels.push(KernelSpec {
                        key,
                        xclbin: x.path.clone(),
                        insts: m.path(r.str("insts")?),
                        name: x.kernel.clone(),
                        m: r.get("M")?,
                        k: r.get("K")?,
                        n: r.get("N")?,
                        a_elems: r.get("a_elems")?,
                        b_bytes: r.get("b_bytes")?,
                        c_elems: r.get("c_elems")?,
                    });
                }
                "conv" => {
                    let name = r.field(0)?.to_string();
                    if convs.iter().any(|c| c.name == name) {
                        return Err(r.error(format!("conv {name} defined twice")).into());
                    }
                    convs.push(ConvSpec {
                        name,
                        kernel: r.str("kernel")?.into(),
                        c: r.get("C")?,
                        oc: r.get("OC")?,
                        pool_before: r.flag("pool_before")?,
                        m_chunk: r.get("m_chunk")?,
                        p: r.get("P")?,
                        window: r.flag("window")?,
                        d: r.get("D")?,
                        b: r.str("b")?.into(),
                        bias: r.str("bias")?.into(),
                    });
                }
                "dimred" => {
                    let v = (r.str("w")?.to_string(), r.str("b")?.to_string(), r.get("out")?, r.get("in")?);
                    once(&mut dimred, r, v)?;
                }
                "fc1" => {
                    let v = Fc1Spec {
                        kernel: r.str("kernel")?.into(),
                        b: r.str("b")?.into(),
                        bias: r.str("bias")?.into(),
                        k: r.get("k")?,
                        k_pad: r.get("k_pad")?,
                        n: r.get("n")?,
                        m: r.get("m")?,
                    };
                    once(&mut fc1, r, v)?;
                }
                "fc2" => {
                    let v = (r.str("w")?.to_string(), r.str("b")?.to_string(), r.get("in")?, r.get("out")?);
                    once(&mut fc2, r, v)?;
                }
                "fc3" => {
                    let v = (r.str("w")?.to_string(), r.str("b")?.to_string(), r.get::<usize>("in")?);
                    once(&mut fc3, r, v)?;
                }
                "ref" => {
                    let v = RefSpec {
                        image: r.str("image")?.into(),
                        w: r.get("w")?,
                        h: r.get("h")?,
                        n_anchors: r.get("anchors")?,
                        src_w: r.get("src_w")?,
                        src_h: r.get("src_h")?,
                    };
                    once(&mut reference, r, v)?;
                }
                _ => return Err(r.error(format!("unknown record {:?}", r.tag)).into()),
            }
        }
        let missing = |what: &str| Error::Bundle(format!("{}: no {what} record", manifest_path.display()));
        let (dimred_w, dimred_b, reddim, dimred_in) = dimred.ok_or_else(|| missing("dimred"))?;
        let (fc2_w, fc2_b, fc2_in, fc2_out) = fc2.ok_or_else(|| missing("fc2"))?;
        let (fc3_w, fc3_b, fc3_in) = fc3.ok_or_else(|| missing("fc3"))?;
        if fc3_in != fc2_out {
            return Err(Error::Bundle(format!("fc3 takes {fc3_in} inputs, fc2 makes {fc2_out}")));
        }
        let b = Bundle {
            dir: dir.to_path_buf(),
            kernels,
            convs,
            f3: m.param("f3")?.to_string(),
            f4: m.param("f4")?.to_string(),
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
            align_size: m.param_as("align_size")?,
            spatial_scale: m.param_as("spatial_scale")?,
            reference,
            store,
        };
        b.validate()?;
        b.validate_tensors()?;
        Ok(b)
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

    /// Every tensor a record names exists with the size the record implies.
    fn validate_tensors(&self) -> Result<(), Error> {
        let st = &self.store;
        let packed = |name: &str, bytes: usize| -> Result<(), Error> {
            let got = st.bytes(name)?.len();
            if got != bytes {
                return Err(Error::Bundle(format!("tensor {name} is {got} bytes, the kernel takes {bytes}")));
            }
            Ok(())
        };
        for c in &self.convs {
            packed(&c.b, self.kernel(&c.kernel)?.b_bytes)?;
            st.expect(&c.bias, DType::F32, c.oc)?;
        }
        packed(&self.fc1.b, self.kernel(&self.fc1.kernel)?.b_bytes)?;
        st.expect(&self.fc1.bias, DType::F32, self.fc1.n)?;
        st.expect(&self.dimred_w, DType::F32, self.reddim * self.dimred_in)?;
        st.expect(&self.dimred_b, DType::F32, self.reddim)?;
        st.expect(&self.fc2_w, DType::F32, self.fc2_out * self.fc2_in)?;
        st.expect(&self.fc2_b, DType::F32, self.fc2_out)?;
        st.expect(&self.fc3_w, DType::F32, self.fc2_out)?;
        st.expect(&self.fc3_b, DType::F32, 1)?;
        if let Some(r) = &self.reference {
            let red = self.reddim * (r.h / 16) * (r.w / 16);
            st.expect("ref.src_rgb", DType::U8, r.src_h * r.src_w * 3)?;
            st.expect("ref.input_chw", DType::F32, 3 * r.h * r.w)?;
            st.expect("ref.red_cpu", DType::F32, red)?;
            st.expect("ref.red_npu", DType::F32, red)?;
            st.expect("ref.anchors", DType::I32, r.n_anchors * 4)?;
            st.expect("ref.scores_cpu", DType::F32, r.n_anchors)?;
            st.expect("ref.scores_npu", DType::F32, r.n_anchors)?;
        }
        Ok(())
    }

    /// An f32 tensor, copied.
    pub fn f32_vec(&self, name: &str) -> Result<Vec<f32>, Error> {
        Ok(self.store.f32(name)?.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn old_bundle_is_rejected_with_a_hint() {
        let d = std::env::temp_dir().join(format!("gaic-old-bundle-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("manifest.txt"), "gaic 1\nf3 conv4_3\n").unwrap();
        let e = Bundle::load(&d).err().unwrap().to_string();
        assert!(e.contains("old-format GAIC bundle") && e.contains("export_gaic"), "{e}");
        std::fs::remove_dir_all(&d).unwrap();
    }
}
