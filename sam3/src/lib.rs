// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SAM3 (Segment Anything 3) text-prompted instance segmentation on an AMD
//! XDNA NPU, with no Python at run time.
//!
//! The bundle `iron/applications/sam3/export_sam3.py` writes holds every
//! compiled IRON kernel and the model's weights (NPU ones pre-packed); this
//! crate replays the forward the Python app (`iron/applications/sam3`)
//! runs, stage for stage:
//!
//! | stage | NPU | host (here) |
//! |---|---|---|
//! | CLIP text encoder | | all of it (`text.rs`) |
//! | ViT backbone, 32 layers | all of it: patch embed, Linears, RoPE, attention, GELU, residual adds + LayerNorms | the first LayerNorm, once (`vit.rs`) |
//! | FPN neck | ConvTs, 1x1s, 3x3s | GELU, pixel shuffles (`neck.rs`) |
//! | DETR encoder, 6 layers | projections, self-attention, folded prompt cross-attention, MLP | LayerNorms, prompt softmax (`detr.rs`) |
//! | DETR decoder, 6 layers | all six layers' vision keys/values (one GEMM) | the 201-query layers, box refinement, scoring (`detr.rs`) |
//! | mask decoder | pixel-decoder 3x3s, folded mask head, prompt cross-attention | GroupNorms, upsampling (`mask.rs`) |
//!
//! [`Sam3::segment`] is the whole thing; the stage methods are public so
//! `sam3 check` can test each on the bundle's reference inputs.

use std::collections::HashMap;
use std::fmt;
use std::path::Path;
use std::time::{Duration, Instant};

use npu::Buffer;

pub mod bundle;
pub mod cpu;
mod detr;
mod mask;
mod neck;
pub mod npu;
pub mod pack;
pub mod post;
mod text;
pub mod tokenizer;
mod vit;

pub use detr::Decoded;
pub use post::{Instance, instances, preprocess};
pub use text::Text;

use bundle::{Manifest, Store};
use npu::{Io, MhaIo, Npu};
use tokenizer::Tokenizer;

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

impl From<iron_bundle::Error> for Error {
    fn from(e: iron_bundle::Error) -> Self {
        Error::Bundle(e.to_string())
    }
}

impl From<iron_xrt::Error> for Error {
    fn from(e: iron_xrt::Error) -> Self {
        Error::Npu(e.to_string())
    }
}

/// Wall time per stage, in first-seen order; NPU dispatch time is kept
/// under `npu:<kernel>`.
#[derive(Default, Debug, Clone)]
pub struct Timing {
    entries: Vec<(String, Duration)>,
}

impl Timing {
    pub fn add(&mut self, key: &str, d: Duration) {
        match self.entries.iter_mut().find(|(k, _)| k == key) {
            Some((_, t)) => *t += d,
            None => self.entries.push((key.to_string(), d)),
        }
    }

    pub fn get(&self, key: &str) -> Duration {
        self.entries.iter().find(|(k, _)| k == key).map_or(Duration::ZERO, |e| e.1)
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    pub fn npu_total(&self) -> Duration {
        self.entries.iter().filter(|(k, _)| k.starts_with("npu:")).map(|e| e.1).sum()
    }
}

impl fmt::Display for Timing {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ms = |d: &Duration| d.as_secs_f64() * 1e3;
        let (npu, host): (Vec<_>, Vec<_>) = self.entries.iter().partition(|(k, _)| k.starts_with("npu:"));
        write!(f, "stages:")?;
        for (k, d) in &host {
            write!(f, " {k} {:.0}", ms(d))?;
        }
        write!(f, " ms; npu {:.0} ms:", ms(&self.npu_total()))?;
        for (k, d) in &npu {
            write!(f, " {} {:.0}", &k[4..], ms(d))?;
        }
        Ok(())
    }
}

/// Model constants (from the manifest's `param` lines).
#[derive(Debug, Clone)]
pub struct Config {
    pub grid: usize,
    pub window: usize,
    pub vit_dim: usize,
    pub vit_heads: usize,
    pub vit_layers: usize,
    pub vit_ffn_pad: usize,
    pub vit_global: Vec<usize>,
    pub patch: usize,
    pub image_size: usize,
    pub d_model: usize,
    pub d_heads: usize,
    pub d_layers: usize,
    pub d_ffn: usize,
    pub text_len: usize,
    pub text_dim: usize,
    pub text_heads: usize,
    pub text_layers: usize,
    pub text_ffn: usize,
    pub text_eps: f32,
    pub vit_eps: f32,
    pub queries: usize,
    pub dec_layers: usize,
    pub neck_splits: Vec<usize>,
    pub mask_size: usize,
    /// every ViT layer on the device (RoPE, LayerNorms, residual adds);
    /// bundles from before it host that glue
    pub vit_device: bool,
}

impl Config {
    fn from(m: &Manifest) -> Result<Self, Error> {
        Ok(Config {
            grid: m.usize("grid")?,
            window: m.usize("window")?,
            vit_dim: m.usize("vit_dim")?,
            vit_heads: m.usize("vit_heads")?,
            vit_layers: m.usize("vit_layers")?,
            vit_ffn_pad: m.usize("vit_ffn_pad")?,
            vit_global: m.list("vit_global")?,
            patch: m.usize("patch")?,
            image_size: m.usize("image_size")?,
            d_model: m.usize("d_model")?,
            d_heads: m.usize("d_heads")?,
            d_layers: m.usize("d_layers")?,
            d_ffn: m.usize("d_ffn")?,
            text_len: m.usize("text_len")?,
            text_dim: m.usize("text_dim")?,
            text_heads: m.usize("text_heads")?,
            text_layers: m.usize("text_layers")?,
            text_ffn: m.usize("text_ffn")?,
            text_eps: m.f32("text_eps")?,
            vit_eps: m.f32("vit_eps")?,
            queries: m.usize("queries")?,
            dec_layers: m.usize("dec_layers")?,
            neck_splits: m.list("neck_splits")?,
            mask_size: m.usize("mask_size")?,
            vit_device: m.param("vit_device").is_ok_and(|v| v == "1"),
        })
    }

    /// ViT tokens (72 x 72).
    pub fn tokens(&self) -> usize {
        self.grid * self.grid
    }
}

/// The model's raw outputs, as `Sam3Model` returns them (batch of one).
pub struct Output {
    /// `[Q]` classification logits.
    pub logits: Vec<f32>,
    /// `[Q, 4]` boxes, normalised xyxy.
    pub boxes: Vec<f32>,
    pub presence: f32,
    /// `[Q, S, S]` mask logits, `S` = `mask_size` (288).
    pub masks: Vec<f32>,
    /// `[S, S]` semantic segmentation logits.
    pub semantic: Vec<f32>,
}

/// NPU operands, allocated once per session.
struct Ios {
    v_embed: Io,
    v_qkv: Io,
    v_o: Io,
    v_fc1: Io,
    v_fc2: Io,
    mha_win: MhaIo,
    mha_glob: MhaIo,
    n_in: Io,
    n_up: Io,
    conv: HashMap<usize, Io>, // by image side
    d_qkv: Io,
    d_o: Io,
    d_s: Io,
    d_c: Io,
    d_fc1: Io,
    d_fc2: Io,
    mha_d: MhaIo,
    dec_kv: Io,
    m_head: Io,
    /// device-resident ViT: RoPE output `[T, 2 D]`, the residual stream's
    /// two ping-pong buffers `[T, D]`
    rope_out: Option<Buffer>,
    xres: Vec<Buffer>,
}

pub struct Sam3 {
    pub manifest: Manifest,
    pub store: Store,
    pub cfg: Config,
    pub tokenizer: Tokenizer,
    npu: Npu,
    /// packed NPU weights, by tensor name
    w: HashMap<String, Buffer>,
    /// per-prompt packed weights (folded cross-attentions, mask head)
    slots: HashMap<String, Buffer>,
    io: Ios,
    pub timing: Timing,
}

impl Sam3 {
    /// Loads the bundle, opens the NPU, loads every kernel and uploads the
    /// packed weights.
    pub fn load(dir: &Path) -> Result<Self, Error> {
        let manifest = Manifest::load(dir)?;
        let store = Store::load(dir)?;
        let cfg = Config::from(&manifest)?;
        let tokenizer = Tokenizer::load(
            dir,
            manifest.usize("bos")? as u32,
            manifest.usize("eos")? as u32,
            manifest.usize("pad")? as u32,
            cfg.text_len,
        )?;
        let npu = Npu::open(&manifest)?;
        let mut w = HashMap::new();
        let mut names = vec!["v.embed".to_string(), "n.in".into(), "n.up".into(), "dec.kv".into()];
        for i in 0..cfg.vit_layers {
            for k in ["qkv", "o", "fc1", "fc2"] {
                names.push(format!("v.{i}.{k}"));
            }
        }
        for i in 0..cfg.d_layers {
            for k in ["qkv", "o", "fc1", "fc2"] {
                names.push(format!("d.{i}.{k}"));
            }
        }
        for i in 0..3 {
            names.push(format!("n.conv{i}"));
        }
        for i in 0..2 {
            names.push(format!("m.conv{i}"));
        }
        if cfg.vit_device {
            names.push("v.rope_tab.win".into());
            names.push("v.rope_tab.glob".into());
        }
        for n in names {
            let b = npu.upload(store.bytes(&n)?)?;
            w.insert(n, b);
        }
        let mut slots = HashMap::new();
        for i in 0..cfg.d_layers {
            slots.insert(format!("d.{i}.s"), npu.weight_slot("d_s")?);
            slots.insert(format!("d.{i}.c"), npu.weight_slot("d_c")?);
        }
        slots.insert("m.s".into(), npu.weight_slot("d_s")?);
        slots.insert("m.c".into(), npu.weight_slot("d_c")?);
        slots.insert("m.head".into(), npu.weight_slot("m_head")?);

        let t = cfg.tokens();
        let v_fc1 = npu.io("v_fc1", t)?;
        let v_fc2 = npu.io_chained("v_fc2", &v_fc1)?;
        let d_fc1 = npu.io("d_fc1", t)?;
        let d_fc2 = npu.io_chained("d_fc2", &d_fc1)?;
        let mut conv = HashMap::new();
        for s in [cfg.grid, 2 * cfg.grid, 4 * cfg.grid] {
            conv.insert(s, npu.conv_io("conv", s, s)?);
        }
        let s = cfg.mask_size;
        let io = Ios {
            v_embed: npu.io("v_embed", t)?,
            v_qkv: npu.io("v_qkv", t)?,
            v_o: npu.io("v_o", t)?,
            v_fc1,
            v_fc2,
            mha_win: npu.mha_io("mha_win")?,
            mha_glob: npu.mha_io("mha_glob")?,
            n_in: npu.io("n_in", t)?,
            n_up: npu.io("n_up", 4 * t)?,
            conv,
            d_qkv: npu.io("d_qkv", t)?,
            d_o: npu.io("d_o", t)?,
            d_s: npu.io("d_s", t)?,
            d_c: npu.io("d_c", t)?,
            d_fc1,
            d_fc2,
            mha_d: npu.mha_io("mha_d")?,
            dec_kv: npu.io("dec_kv", t)?,
            m_head: npu.io("m_head", s * s)?,
            rope_out: if cfg.vit_device { Some(npu.session.alloc(t * 2 * cfg.vit_dim * 2)?) } else { None },
            xres: if cfg.vit_device {
                vec![npu.session.alloc(t * cfg.vit_dim * 2)?, npu.session.alloc(t * cfg.vit_dim * 2)?]
            } else {
                vec![]
            },
        };
        Ok(Sam3 { manifest, store, cfg, tokenizer, npu, w, slots, io, timing: Timing::default() })
    }

    /// Hardware contexts created and evicted so far (evictions happen when
    /// another process holds some of the NPU's contexts).
    pub fn contexts(&self) -> (usize, usize) {
        (self.npu.loads, self.npu.evictions)
    }

    fn time<T>(&mut self, key: &str, f: impl FnOnce(&mut Self) -> Result<T, Error>) -> Result<T, Error> {
        let t0 = Instant::now();
        let r = f(self)?;
        self.timing.add(key, t0.elapsed());
        Ok(r)
    }

    /// Everything for one image and prompt: `pixels` is the preprocessed
    /// `[3, 1008, 1008]` image ([`preprocess`]).
    pub fn segment(&mut self, pixels: &[f32], prompt: &str) -> Result<Output, Error> {
        let text = self.time("text", |s| {
            let (ids, mask) = s.tokenizer.encode(prompt);
            s.text(&ids, &mask)
        })?;
        self.forward(pixels, &text)
    }

    /// The forward from preprocessed pixels and an encoded prompt.
    pub fn forward(&mut self, pixels: &[f32], text: &Text) -> Result<Output, Error> {
        let vit = self.time("vit", |s| s.vit(pixels))?;
        let fpn = self.time("neck", |s| s.neck(&vit))?;
        let enc = self.time("detr_enc", |s| s.detr_encoder(&fpn[2], text))?;
        let dec = self.time("detr_dec", |s| s.detr_decoder(&enc, text))?;
        let (masks, semantic) = self.time("mask_dec", |s| s.mask_decoder(&dec.hidden, &fpn, &enc, text))?;
        Ok(Output { logits: dec.logits, boxes: dec.boxes, presence: dec.presence, masks, semantic })
    }
}

/// Runs `io` (A synced first unless `io` is chained) with weights `w`,
/// accounting the dispatch time under `npu:<kernel>`.
fn gemm(npu: &mut Npu, io: &Io, w: &Buffer, timing: &mut Timing) -> Result<(), Error> {
    let d = npu.run_synced(io, w)?;
    timing.add(&format!("npu:{}", io.key), d);
    Ok(())
}

/// Runs GEMM `io` whose A another kernel wrote (no host sync).
fn gemm_dev(npu: &mut Npu, io: &Io, w: &Buffer, timing: &mut Timing) -> Result<(), Error> {
    let d = npu.run(io, w)?;
    timing.add(&format!("npu:{}", io.key), d);
    Ok(())
}

/// Runs kernel `key` over device-resident `args`.
fn op(npu: &mut Npu, key: &str, args: &[&Buffer], timing: &mut Timing) -> Result<(), Error> {
    let d = npu.run_args(key, args)?;
    timing.add(&format!("npu:{key}"), d);
    Ok(())
}

fn mha(npu: &mut Npu, io: &MhaIo, timing: &mut Timing) -> Result<(), Error> {
    let d = npu.run_mha(io)?;
    timing.add(&format!("npu:{}", io.key), d);
    Ok(())
}

/// f32 -> bf16 bits (round to nearest even), over the threads.
pub(crate) fn narrow(src: &[f32], dst: &mut [u16]) {
    let n = src.len();
    cpu::par_rows(&mut dst[..n], 1 << 14, |i0, out| {
        for (i, o) in out.iter_mut().enumerate() {
            *o = iron_xrt::f32_to_bf16(src[i0 * (1 << 14) + i]);
        }
    });
}
