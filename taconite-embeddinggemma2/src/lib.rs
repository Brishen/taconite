// SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
// SPDX-License-Identifier: Apache-2.0

//! EmbeddingGemma 2 (`google/embeddinggemma-2`) image embeddings on an AMD
//! XDNA NPU.
//!
//! The bundle `iron/applications/embeddinggemma2/export_eg2.py` writes
//! holds every compiled IRON kernel and the weights (the NPU ones
//! pre-packed); this crate replays the forward the Python app runs
//! (`eg2_common.py` / `eg2_npu.py`):
//!
//! | | NPU | host (here) |
//! |---|---|---|
//! | vision tower (16 layers, 768 wide) | every projection (`flm.GEMM`s: patch embedding, qkv, o, GeGLU gate+up, down, embed_vision) and the attention (the MHA operator) | resize + patchify (`preprocess.rs`), position embeddings, RMSNorms, q / k / v norms, 2D RoPE, residual adds, 3 x 3 pooling |
//! | text encoder (24 layers, 512 wide) | every projection (per-layer-input projection, qkv, o, GeGLU, down, PLE gate + projection) | norms, RoPE, attention (~270 tokens), per-layer gating, mean pooling, the 512 -> 768 projection |
//!
//! [`EmbeddingGemma2::embed_rgb`] gives the L2-normalized 768-d
//! embedding sentence-transformers computes for an image; it lives in the
//! same space as the model's text embeddings (cosine similarity).

use std::fmt;
use std::path::Path;

use taconite_bundle::{Manifest, Store};

pub use taconite::Timing;

pub mod model;
pub mod npu;
pub mod preprocess;

use model::Model;
use npu::Npu;

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
    pub v_d: usize,
    pub v_i: usize,
    pub v_layers: usize,
    pub v_heads: usize,
    pub v_hd: usize,
    pub patch: usize,
    pub pool: usize,
    pub v_theta: f32,
    pub v_eps: f32,
    /// patches the device buffers hold
    pub v_rows: usize,
    pub d: usize,
    pub i: usize,
    pub layers: usize,
    pub heads: usize,
    pub eps: f32,
    pub hd: Vec<usize>,
    pub kv_heads: Vec<usize>,
    /// per layer: a global (full-attention) layer
    pub global: Vec<bool>,
    pub theta_s: f32,
    pub theta_g: f32,
    pub ple: usize,
    pub out: usize,
    pub window: usize,
    pub t_rows: usize,
    pub o_width: usize,
    pub max_soft_tokens: usize,
    pub no_window: u32,
}

impl Config {
    fn load(m: &Manifest) -> Result<Self, Error> {
        let p = |k: &str| m.param_as::<usize>(k);
        let f = |k: &str| m.param_as::<f32>(k);
        let global = m
            .param("layer_types")?
            .split(',')
            .map(|t| match t {
                "g" => Ok(true),
                "s" => Ok(false),
                _ => Err(Error::Bundle(format!("layer type {t}"))),
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Config {
            v_d: p("v.D")?,
            v_i: p("v.I")?,
            v_layers: p("v.layers")?,
            v_heads: p("v.heads")?,
            v_hd: p("v.hd")?,
            patch: p("v.patch")?,
            pool: p("v.pool")?,
            v_theta: f("v.theta")?,
            v_eps: f("v.eps")?,
            v_rows: p("v.rows")?,
            d: p("D")?,
            i: p("I")?,
            layers: p("layers")?,
            heads: p("heads")?,
            eps: f("eps")?,
            hd: m.list("hd")?,
            kv_heads: m.list("kv_heads")?,
            global,
            theta_s: f("theta_s")?,
            theta_g: f("theta_g")?,
            ple: p("ple")?,
            out: p("out")?,
            window: p("window")?,
            t_rows: p("t.rows")?,
            o_width: p("o_width")?,
            max_soft_tokens: p("max_soft_tokens")?,
            no_window: m.param_as("no_window")?,
        })
    }

    /// Patches the processor makes at most of an image.
    pub fn max_patches(&self) -> usize {
        self.max_soft_tokens * self.pool * self.pool
    }
}

pub struct EmbeddingGemma2 {
    pub cfg: Config,
    pub npu: Npu,
    pub model: Model,
    pub manifest: Manifest,
    pub store: Store,
    /// where the last call spent its time (`npu:<kernel>`, host stages)
    pub timing: Timing,
}

impl EmbeddingGemma2 {
    /// Loads a bundle: opens the NPU, loads every kernel and uploads the
    /// packed weights.
    pub fn load(dir: &Path) -> Result<Self, Error> {
        let manifest = Manifest::load(dir, VERSION)?;
        let store = Store::load(dir)?;
        let cfg = Config::load(&manifest)?;
        let npu = Npu::open(&manifest)?;
        let model = Model::load(&cfg, &store, &npu)?;
        Ok(EmbeddingGemma2 { cfg, npu, model, manifest, store, timing: Timing::default() })
    }

    /// Hardware contexts the bundle's kernels use.
    pub fn contexts(&self) -> usize {
        self.npu.contexts
    }

    /// RGB8 `[h, w, 3]` -> the image's patches, as the HF processor makes
    /// them.
    pub fn preprocess(&self, rgb: &[u8], w: usize, h: usize) -> Result<preprocess::Patches, Error> {
        let c = &self.cfg;
        preprocess::patches(rgb, w, h, c.patch, c.pool, c.max_soft_tokens)
            .ok_or_else(|| Error::Input(format!("a {w} x {h} image is too thin")))
    }

    /// RGB8 `[h, w, 3]` -> its unit 768-d embedding.
    pub fn embed_rgb(&mut self, rgb: &[u8], w: usize, h: usize) -> Result<Vec<f32>, Error> {
        let p = self.preprocess(rgb, w, h)?;
        self.embed_patches(&p)
    }

    /// An image's patches -> its unit embedding.
    pub fn embed_patches(&mut self, p: &preprocess::Patches) -> Result<Vec<f32>, Error> {
        self.timing.clear();
        let soft = self.model.vision(&self.cfg, &self.npu, p, &mut self.timing)?;
        self.model.text(&self.cfg, &self.npu, &soft, &mut self.timing)
    }

    /// An image's patches -> its soft tokens `[s, 512]` (the text model's
    /// input embeddings of the image), for checks.
    pub fn soft_tokens(&mut self, p: &preprocess::Patches) -> Result<Vec<f32>, Error> {
        self.model.vision(&self.cfg, &self.npu, p, &mut self.timing)
    }
}

/// Cosine similarity of two embeddings (dot product of unit vectors).
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let d = |x: &[f32], y: &[f32]| x.iter().zip(y).map(|(p, q)| (*p as f64) * (*q as f64)).sum::<f64>();
    (d(a, b) / (d(a, a) * d(b, b)).sqrt()) as f32
}

/// An embedding truncated to its first `dim` values (Matryoshka: 768, 512,
/// 256 or 128) and re-normalized.
pub fn truncate(e: &[f32], dim: usize) -> Vec<f32> {
    let v = &e[..dim.min(e.len())];
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    v.iter().map(|x| x / n).collect()
}
