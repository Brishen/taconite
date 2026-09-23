// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CLIP ViT-H/14 (`laion/CLIP-ViT-H-14-laion2B-s32B-b79K`) image and text
//! embeddings on an AMD XDNA NPU, with no Python at run time.
//!
//! The bundle `iron/applications/clip_vit_h14/export_clip.py` writes holds
//! every compiled IRON kernel and the model's weights (the NPU ones
//! pre-packed); this crate replays the forward the Python app runs:
//!
//! | | NPU | host (here) |
//! |---|---|---|
//! | image | all 32 layers (`tower.rs`) | resize / crop / normalise (`preprocess.rs`), the patch embedding (f32), CLS + position embedding, pre- and post-LayerNorm, the projection |
//! | text | all 24 layers | BPE tokenizer, token + position embedding, final LayerNorm at the end token, the projection |
//!
//! [`Clip::encode_images`] / [`Clip::encode_texts`] give the projected
//! embeddings, [`logits`] CLIP's scaled cosine similarities.

use std::fmt;
use std::path::Path;
use std::time::Instant;

use taconite_bundle::{Manifest, Store};
use npu::Buffer;
pub use taconite_sam3::Timing;
use taconite_sam3::cpu::{dot, ln_row, par_rows};
use taconite_sam3::tokenizer::Tokenizer;

pub mod npu;
pub mod preprocess;
pub mod tower;

use npu::Npu;
use preprocess::Preprocess;
use tower::Tower;

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

/// Model constants (the manifest's `param` records).
#[derive(Debug, Clone)]
pub struct Config {
    pub v_layers: usize,
    pub v_dim: usize,
    pub v_tokens: usize,
    /// rows between images (tokens rounded up to 8; see the exporter)
    pub v_stride: usize,
    pub v_batch: usize,
    pub v_rows: usize,
    pub v_mha_rows: usize,
    pub v_eps: f32,
    pub patch: usize,
    pub image_size: usize,
    pub t_layers: usize,
    pub t_dim: usize,
    pub t_batch: usize,
    pub t_seq_pad: usize,
    pub t_rows: usize,
    pub t_eps: f32,
    pub context: usize,
    pub embed_dim: usize,
    pub logit_scale: f32,
    pub eos_argmax: bool,
    pub prompt_template: String,
}

impl Config {
    fn load(m: &Manifest) -> Result<Self, Error> {
        let u = |k: &str| -> Result<usize, Error> { Ok(m.param_as(k)?) };
        let f = |k: &str| -> Result<f32, Error> { Ok(m.param_as(k)?) };
        Ok(Config {
            v_layers: u("v_layers")?,
            v_dim: u("v_dim")?,
            v_tokens: u("v_tokens")?,
            v_stride: u("v_stride")?,
            v_batch: u("v_batch")?,
            v_rows: u("v_rows")?,
            v_mha_rows: u("v_mha_rows")?,
            v_eps: f("v_eps")?,
            patch: u("patch")?,
            image_size: u("image_size")?,
            t_layers: u("t_layers")?,
            t_dim: u("t_dim")?,
            t_batch: u("t_batch")?,
            t_seq_pad: u("t_seq_pad")?,
            t_rows: u("t_rows")?,
            t_eps: f("t_eps")?,
            context: u("context")?,
            embed_dim: u("embed_dim")?,
            logit_scale: f("logit_scale")?,
            eos_argmax: u("eos_argmax")? != 0,
            prompt_template: m.param("prompt_template")?.to_string(),
        })
    }

    /// Patches a side (16 for 224 / 14).
    pub fn grid(&self) -> usize {
        self.image_size / self.patch
    }
}

/// Copies `src` (bf16 bits) into the start of `b` and syncs it.
pub(crate) fn push(src: &[u16], b: &mut Buffer) -> Result<(), Error> {
    const COPY: usize = 1 << 16;
    let dst = &mut b.as_mut_slice::<u16>()[..src.len()];
    par_rows(dst, COPY, |r0, piece| piece.copy_from_slice(&src[r0 * COPY..][..piece.len()]));
    Ok(b.sync_to_device()?)
}

/// The first `n` bf16 of `b`, synced from the device, in cached memory.
pub(crate) fn pull(b: &Buffer, n: usize) -> Result<Vec<u16>, Error> {
    const COPY: usize = 1 << 16;
    b.sync_from_device()?;
    let src = &b.as_slice::<u16>()[..n];
    let mut dst = vec![0u16; n];
    par_rows(&mut dst, COPY, |r0, piece| piece.copy_from_slice(&src[r0 * COPY..][..piece.len()]));
    Ok(dst)
}

pub struct Clip {
    pub cfg: Config,
    pub manifest: Manifest,
    pub store: Store,
    pub tokenizer: Tokenizer,
    pub preprocess: Preprocess,
    pub timing: Timing,
    npu: Npu,
    vision: Tower,
    text: Tower,
}

impl Clip {
    /// Loads a bundle: every kernel onto the NPU, every weight into device
    /// buffers.
    pub fn load(dir: &Path) -> Result<Self, Error> {
        let manifest = Manifest::load(dir, VERSION)?;
        let cfg = Config::load(&manifest)?;
        let store = Store::load(dir)?;
        let u = |k: &str| -> Result<u32, Error> { Ok(manifest.param_as(k)?) };
        let tokenizer = Tokenizer::load(dir, u("bos")?, u("eos")?, u("pad")?, cfg.context)
            .map_err(|e| Error::Bundle(e.to_string()))?;
        let three = |k: &str| -> Result<[f32; 3], Error> {
            let v: Vec<f32> = manifest.list(k)?;
            v.try_into().map_err(|_| Error::Bundle(format!("param {k}: expected 3 values")))
        };
        let preprocess = Preprocess {
            shortest: manifest.param_as("resize_shortest")?,
            crop: manifest.param_as("crop")?,
            mean: three("mean")?,
            std: three("std")?,
        };
        if preprocess.crop != cfg.image_size {
            return Err(Error::Bundle("crop size != image size".into()));
        }

        let npu = Npu::open(&manifest)?;
        // one MHA dispatch per image: its tokens, padded (masked) to the
        // kernel's rows, which run into the next image's -- rewritten by
        // that image's dispatch, issued after
        let v_mha: Vec<(usize, usize)> = (0..cfg.v_batch).map(|i| (i * cfg.v_stride, cfg.v_mha_rows)).collect();
        let vision = Tower::new(&npu, &store, "v", cfg.v_layers, cfg.v_dim, cfg.v_rows, cfg.v_eps, &v_mha)?;
        let text = Tower::new(
            &npu,
            &store,
            "t",
            cfg.t_layers,
            cfg.t_dim,
            cfg.t_rows,
            cfg.t_eps,
            &[(0, cfg.t_batch * cfg.t_seq_pad)],
        )?;
        Ok(Clip { cfg, manifest, store, tokenizer, preprocess, timing: Timing::default(), npu, vision, text })
    }

    /// Hardware contexts the bundle holds (of NPU2's 16).
    pub fn contexts(&self) -> usize {
        self.npu.contexts
    }

    /// RGB8 `[h, w, 3]` -> the model input `[3, 224, 224]`.
    pub fn preprocess(&self, rgb: &[u8], w: usize, h: usize) -> Vec<f32> {
        self.preprocess.run(rgb, w, h)
    }

    /// `n` images `[n, 3, 224, 224]` -> their embeddings `[n, embed_dim]`
    /// (projected, not normalised: `get_image_features`).
    pub fn encode_images(&mut self, pixels: &[f32]) -> Result<Vec<f32>, Error> {
        let c = self.cfg.clone();
        let (s, ps, g, d, t, b) = (c.image_size, c.patch, c.grid(), c.v_dim, c.v_tokens, c.v_batch);
        let st = c.v_stride;
        let img = 3 * s * s;
        if pixels.is_empty() || pixels.len() % img != 0 {
            return Err(Error::Input(format!("pixels must be [n, 3, {s}, {s}]")));
        }
        let n = pixels.len() / img;
        let (cls, pos) = (self.store.f32("v.cls")?, self.store.f32("v.pos")?);
        let (pre_w, pre_b) = (self.store.f32("v.ln_pre.w")?, self.store.f32("v.ln_pre.b")?);
        let mut out = Vec::with_capacity(n * c.embed_dim);
        for i0 in (0..n).step_by(b) {
            let ng = (n - i0).min(b);
            let t0 = Instant::now();
            // patches, row (image, py, px), columns (channel, ky, kx)
            let kp = 3 * ps * ps;
            let mut patches = vec![0f32; ng * g * g * kp];
            par_rows(&mut patches, kp, |r0, piece| {
                for (ri, row) in piece.chunks_mut(kp).enumerate() {
                    let r = r0 + ri;
                    let (im, p) = (i0 + r / (g * g), r % (g * g));
                    let (py, px) = (p / g, p % g);
                    let src = &pixels[im * img..][..img];
                    for ch in 0..3 {
                        for ky in 0..ps {
                            let base = ch * s * s + (py * ps + ky) * s + px * ps;
                            row[(ch * ps + ky) * ps..][..ps].copy_from_slice(&src[base..][..ps]);
                        }
                    }
                }
            });
            // f32, as the model's conv (an NPU GEMM here, its output bf16
            // before the position embedding and pre-LayerNorm, cost the
            // image embedding ~0.0025 cosine for ~25 ms)
            let wp = self.store.f32("v.patch_w")?;
            let mut emb = vec![0f32; ng * g * g * d];
            par_rows(&mut emb, d, |r0, piece| {
                for (ri, row) in piece.chunks_mut(d).enumerate() {
                    let x = &patches[(r0 + ri) * kp..][..kp];
                    for (j, o) in row.iter_mut().enumerate() {
                        *o = dot(&wp[j * kp..][..kp], x);
                    }
                }
            });
            self.timing.add("v_patches", t0.elapsed());
            let t1 = Instant::now();
            // [CLS | patches] + position embedding, pre-LayerNorm
            // image im's tokens at rows [im * stride, im * stride + t)
            let mut x = vec![0f32; self.vision.rows * d];
            par_rows(&mut x[..ng * st * d], d, |r0, piece| {
                let mut tmp = vec![0f32; d];
                for (ri, row) in piece.chunks_mut(d).enumerate() {
                    let r = r0 + ri;
                    let (im, tok) = (r / st, r % st);
                    if tok >= t {
                        continue;
                    }
                    for j in 0..d {
                        let e = if tok == 0 { cls[j] } else { emb[(im * g * g + tok - 1) * d + j] };
                        tmp[j] = e + pos[tok * d + j];
                    }
                    ln_row(&tmp, row, pre_w, pre_b, c.v_eps);
                }
            });
            self.timing.add("v_embed_host", t1.elapsed());
            let y = self.vision.run(&self.npu, &x, &mut self.timing)?;
            let t2 = Instant::now();
            let (post_w, post_b) = (self.store.f32("v.ln_post.w")?, self.store.f32("v.ln_post.b")?);
            for im in 0..ng {
                let mut pooled = vec![0f32; d];
                ln_row(&y[im * st * d..][..d], &mut pooled, post_w, post_b, c.v_eps);
                out.extend(project(&pooled, self.store.f32("v.proj")?, c.embed_dim));
            }
            self.timing.add("v_head", t2.elapsed());
        }
        Ok(out)
    }

    /// A prompt's token ids (`context` long, padded).
    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        self.tokenizer.encode(text).0
    }

    /// Prompts -> their text embeddings `[n, embed_dim]`.
    pub fn encode_texts(&mut self, texts: &[&str]) -> Result<Vec<f32>, Error> {
        let t0 = Instant::now();
        let ids: Vec<u32> = texts.iter().flat_map(|t| self.tokenize(t)).collect();
        self.timing.add("t_tokenize", t0.elapsed());
        self.encode_token_ids(&ids)
    }

    /// `n` token-id rows `[n, context]` -> the text embeddings
    /// `[n, embed_dim]` (projected, not normalised: `get_text_features`).
    pub fn encode_token_ids(&mut self, ids: &[u32]) -> Result<Vec<f32>, Error> {
        let c = self.cfg.clone();
        let (l, d, sp, b) = (c.context, c.t_dim, c.t_seq_pad, c.t_batch);
        if ids.is_empty() || ids.len() % l != 0 {
            return Err(Error::Input(format!("token ids must be [n, {l}]")));
        }
        let n = ids.len() / l;
        let vocab = self.store.shape("t.tok_emb")?[0];
        if let Some(&bad) = ids.iter().find(|&&i| i as usize >= vocab) {
            return Err(Error::Input(format!("token id {bad} >= vocabulary size {vocab}")));
        }
        let eos = self.tokenizer.eos;
        let mut out = Vec::with_capacity(n * c.embed_dim);
        for p0 in (0..n).step_by(b) {
            let ng = (n - p0).min(b);
            let t0 = Instant::now();
            let (tok, pos) = (self.store.f32("t.tok_emb")?, self.store.f32("t.pos")?);
            // prompt p's tokens at rows [p * seq_pad, p * seq_pad + context)
            let mut x = vec![0f32; self.text.rows * d];
            par_rows(&mut x[..ng * sp * d], d, |r0, piece| {
                for (ri, row) in piece.chunks_mut(d).enumerate() {
                    let (p, s) = ((r0 + ri) / sp, (r0 + ri) % sp);
                    if s < l {
                        let id = ids[(p0 + p) * l + s] as usize;
                        for j in 0..d {
                            row[j] = tok[id * d + j] + pos[s * d + j];
                        }
                    }
                }
            });
            self.timing.add("t_embed_host", t0.elapsed());
            let y = self.text.run(&self.npu, &x, &mut self.timing)?;
            let t1 = Instant::now();
            let (fw, fb) = (self.store.f32("t.ln_final.w")?, self.store.f32("t.ln_final.b")?);
            for p in 0..ng {
                let row = &ids[(p0 + p) * l..][..l];
                // the end token: the first eos (or, for legacy configs, the
                // highest id -- the same token for CLIP's vocabulary)
                let e = if c.eos_argmax {
                    let m = *row.iter().max().unwrap();
                    row.iter().position(|&i| i == m).unwrap()
                } else {
                    row.iter().position(|&i| i == eos).unwrap_or(0)
                };
                let mut pooled = vec![0f32; d];
                ln_row(&y[(p * sp + e) * d..][..d], &mut pooled, fw, fb, c.t_eps);
                out.extend(project(&pooled, self.store.f32("t.proj")?, c.embed_dim));
            }
            self.timing.add("t_head", t1.elapsed());
        }
        Ok(out)
    }

    /// A class label's prompt (`a photo of a {}`).
    pub fn prompt(&self, label: &str) -> String {
        self.cfg.prompt_template.replace("{}", label)
    }
}

/// `w [out, in] x` (f32).
fn project(x: &[f32], w: &[f32], out: usize) -> Vec<f32> {
    let n = x.len();
    (0..out).map(|o| dot(&w[o * n..][..n], x)).collect()
}

/// Rows of `x` scaled to unit length.
pub fn normalize(x: &[f32], dim: usize) -> Vec<f32> {
    x.chunks(dim)
        .flat_map(|r| {
            let s = 1.0 / dot(r, r).sqrt().max(1e-12);
            r.iter().map(move |v| v * s)
        })
        .collect()
}

/// CLIP's logits `[n_images, n_texts]`: `logit_scale * cos(image, text)`.
pub fn logits(images: &[f32], texts: &[f32], dim: usize, scale: f32) -> Vec<f32> {
    let (a, b) = (normalize(images, dim), normalize(texts, dim));
    a.chunks(dim).flat_map(|i| b.chunks(dim).map(move |t| scale * dot(i, t))).collect()
}

/// Row-wise softmax.
pub fn softmax(x: &[f32], n: usize) -> Vec<f32> {
    x.chunks(n)
        .flat_map(|r| {
            let m = r.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let e: Vec<f32> = r.iter().map(|v| (v - m).exp()).collect();
            let s: f32 = e.iter().sum();
            e.into_iter().map(move |v| v / s)
        })
        .collect()
}
