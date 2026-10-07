// SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
// SPDX-License-Identifier: Apache-2.0

//! Sapiens2-Pose 0.4B (`facebook/sapiens2-pose-0.4b`) on an AMD XDNA NPU:
//! a person box in an image -> 308 keypoints (body, feet, hands, face).
//!
//! The bundle `iron/applications/sapiens2_pose/export_sapiens2.py` writes
//! holds every compiled IRON kernel and the weights (the NPU ones
//! pre-packed); this crate replays the forward the Python app runs
//! (`sapiens2_common.py` / `sapiens2_npu.py`):
//!
//! | | NPU | host (here) |
//! |---|---|---|
//! | backbone (ViT, 24 layers, 1024 wide, 3081 tokens) | every projection (`flm.GEMM`s: patch embedding, qkv, o, the SwiGLU gate+up, down) and the attention (the MHA operator) | the box crop (`preprocess.rs`), RMSNorms, q / k norms, 2D RoPE, residual adds |
//! | head (2 transposed convs to 256 x 192, 3 1 x 1 convs, the predictor) | every convolution as an `flm.GEMM` (a transposed conv as one GEMM over its input's 2 x 2 windows) | the window layout, InstanceNorm + SiLU |
//! | keypoints | | argmax + DARK refinement, back through the crop (`post.rs`) |
//!
//! [`Sapiens2::pose`] gives a box's keypoints in image coordinates and
//! their heatmap scores, as HF's `post_process_pose_estimation`.

use std::fmt;
use std::path::Path;

use taconite_bundle::{Manifest, Store};

pub use taconite::Timing;

pub mod model;
pub mod npu;
pub mod post;
pub mod preprocess;

use model::Model;
use npu::Npu;
pub use post::Keypoint;
pub use preprocess::BBox;

pub const VERSION: u32 = 1;

#[derive(Debug)]
pub enum Error {
    Bundle(String),
    Npu(String),
    Input(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Bundle(m) => write!(f, "bundle: {m}"),
            Error::Npu(m) => write!(f, "NPU: {m}"),
            Error::Input(m) => write!(f, "input: {m}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<taconite_bundle::Error> for Error {
    fn from(e: taconite_bundle::Error) -> Self {
        Error::Bundle(e.to_string())
    }
}

impl From<taconite::Error> for Error {
    fn from(e: taconite::Error) -> Self {
        Error::Npu(e.to_string())
    }
}

/// The model's constants (the manifest's params).
#[derive(Debug, Clone)]
pub struct Config {
    pub d: usize,
    pub i: usize,
    pub layers: usize,
    pub heads: usize,
    pub hd: usize,
    pub kv_heads: Vec<usize>,
    pub regs: usize,
    pub patch: usize,
    /// crop height
    pub h: usize,
    /// crop width
    pub w: usize,
    pub eps: f32,
    pub in_eps: f32,
    /// the transposed convolutions' output channels
    pub up: Vec<usize>,
    /// the 1 x 1 convolutions' output channels
    pub convs: Vec<usize>,
    /// keypoints
    pub k: usize,
    pub box_pad: f32,
    pub blur: usize,
    pub mean: [f32; 3],
    pub std: [f32; 3],
    /// mirrored keypoint pairs (left, right)
    pub flip_pairs: Vec<(usize, usize)>,
}

impl Config {
    fn load(m: &Manifest) -> Result<Self, Error> {
        let p = |k: &str| m.param_as::<usize>(k);
        let f = |k: &str| m.param_as::<f32>(k);
        let three = |k: &str| -> Result<[f32; 3], Error> {
            let v: Vec<f32> = m.list(k)?;
            v.try_into().map_err(|_| Error::Bundle(format!("{k}: 3 values")))
        };
        let flip_pairs = m
            .param("flip_pairs")?
            .split(',')
            .map(|s| {
                let (a, b) = s.split_once(':').ok_or_else(|| Error::Bundle(format!("flip pair {s}")))?;
                Ok((a.parse().map_err(|_| Error::Bundle(s.into()))?, b.parse().map_err(|_| Error::Bundle(s.into()))?))
            })
            .collect::<Result<_, Error>>()?;
        Ok(Config {
            d: p("D")?,
            i: p("I")?,
            layers: p("layers")?,
            heads: p("heads")?,
            hd: p("hd")?,
            kv_heads: m.list("kv_heads")?,
            regs: p("regs")?,
            patch: p("patch")?,
            h: p("H")?,
            w: p("W")?,
            eps: f("eps")?,
            in_eps: f("in_eps")?,
            up: m.list("up")?,
            convs: m.list("convs")?,
            k: p("K")?,
            box_pad: f("box_pad")?,
            blur: p("blur")?,
            mean: three("mean")?,
            std: three("std")?,
            flip_pairs,
        })
    }

    pub fn gh(&self) -> usize {
        self.h / self.patch
    }

    pub fn gw(&self) -> usize {
        self.w / self.patch
    }

    /// CLS + register tokens
    pub fn prefix(&self) -> usize {
        1 + self.regs
    }

    pub fn qkv_width(&self, i: usize) -> usize {
        (self.heads + 2 * self.kv_heads[i]) * self.hd
    }

    /// Heatmap height and width.
    pub fn heatmap(&self) -> (usize, usize) {
        let s = 1 << self.up.len();
        (self.gh() * s, self.gw() * s)
    }
}

pub struct Sapiens2 {
    pub cfg: Config,
    pub npu: Npu,
    pub model: Model,
    pub manifest: Manifest,
    pub store: Store,
    /// where the last call spent its time (`npu:<kernel>`, host stages)
    pub timing: Timing,
}

impl Sapiens2 {
    /// Loads a bundle: opens the NPU and uploads the packed weights
    /// (kernels load on first use; [`Npu::preload`] loads them now).
    pub fn load(dir: &Path) -> Result<Self, Error> {
        let manifest = Manifest::load(dir, VERSION)?;
        let store = Store::load(dir)?;
        let cfg = Config::load(&manifest)?;
        let npu = Npu::open(&manifest)?;
        let model = Model::load(&cfg, &store, &npu)?;
        Ok(Sapiens2 { cfg, npu, model, manifest, store, timing: Timing::default() })
    }

    /// Hardware contexts the bundle's kernels use.
    pub fn contexts(&self) -> usize {
        self.npu.contexts
    }

    /// Frees every NPU hardware context the model holds, for another model
    /// in the process (NPU2 has 16 across every process); the kernels load
    /// again on the next call. The number of contexts freed.
    pub fn release_contexts(&mut self) -> usize {
        self.npu.release()
    }

    /// RGB8 `[h, w, 3]` and a box -> the model's input `[3, H, W]`.
    pub fn preprocess(&self, rgb: &[u8], w: usize, h: usize, b: &BBox) -> Result<Vec<f32>, Error> {
        if rgb.len() != w * h * 3 || w < 2 || h < 2 {
            return Err(Error::Input(format!("{} bytes for a {w} x {h} RGB image", rgb.len())));
        }
        if !(b.w > 0.0 && b.h > 0.0) {
            return Err(Error::Input(format!("an empty box {b:?}")));
        }
        let c = &self.cfg;
        Ok(preprocess::crop(rgb, w, h, b, c.w, c.h, c.box_pad, c.mean, c.std))
    }

    /// pixel_values `[3, H, W]` -> the normalized patch features `[P, D]`.
    pub fn features(&mut self, pixels: &[f32]) -> Result<Vec<f32>, Error> {
        self.model.backbone(&self.cfg, &self.npu, pixels, &mut self.timing)
    }

    /// pixel_values `[3, H, W]` -> heatmaps `[K, h, w]`.
    pub fn heatmaps(&mut self, pixels: &[f32]) -> Result<Vec<f32>, Error> {
        let c = &self.cfg;
        if pixels.len() != 3 * c.h * c.w {
            return Err(Error::Input(format!("{} pixel values (want 3 x {} x {})", pixels.len(), c.h, c.w)));
        }
        self.timing.clear();
        let f = self.model.backbone(&self.cfg, &self.npu, pixels, &mut self.timing)?;
        self.model.head(&self.cfg, &self.npu, &f, &mut self.timing)
    }

    /// Heatmaps of a box's crop -> its keypoints in image coordinates.
    pub fn keypoints(&mut self, heatmaps: &[f32], b: &BBox) -> Vec<Keypoint> {
        let c = &self.cfg;
        let (hh, hw) = c.heatmap();
        let t0 = std::time::Instant::now();
        let p = post::decode(heatmaps, c.k, hh, hw, c.blur);
        let kp = post::to_image(&p, &b.window(c.w, c.h, c.box_pad), hh, hw);
        self.timing.add("host.keypoints", t0.elapsed());
        kp
    }

    /// RGB8 `[h, w, 3]` and a person box -> the 308 keypoints.
    pub fn pose(&mut self, rgb: &[u8], w: usize, h: usize, b: &BBox) -> Result<Vec<Keypoint>, Error> {
        let t0 = std::time::Instant::now();
        let px = self.preprocess(rgb, w, h, b)?;
        let dt = t0.elapsed();
        let hm = self.heatmaps(&px)?;
        self.timing.add("host.crop", dt);
        Ok(self.keypoints(&hm, b))
    }
}

/// Cosine similarity.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let d = |x: &[f32], y: &[f32]| x.iter().zip(y).map(|(p, q)| (*p as f64) * (*q as f64)).sum::<f64>();
    (d(a, b) / (d(a, a) * d(b, b)).sqrt()) as f32
}
