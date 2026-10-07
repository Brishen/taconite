// SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
// SPDX-License-Identifier: Apache-2.0

//! SAM3 (Segment Anything 3) text-prompted instance segmentation on an AMD
//! XDNA NPU.
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
//! That is [`Mode::Npu`]. In [`Mode::NpuGpu`] (feature `gpu`) the host
//! column's glue and the decoder's query layers run on the iGPU instead,
//! over the NPU's buffers in place (`gpu/`).
//!
//! [`Sam3::segment`] is the whole thing; the stage methods are public so
//! `sam3 check` can test each on the bundle's reference inputs.
//!
//! Point and box prompts (SAM's clicks, bundles with the tracker head):
//! [`Sam3::embed_image`] runs the ViT and the tracker's FPN neck on the NPU
//! once per image, then [`Sam3::predict_points`] answers each
//! [`PointPrompt`] on the host in tens of milliseconds (`tracker.rs`).

use std::collections::HashMap;
use std::fmt;
use std::path::Path;
use std::time::{Duration, Instant};

use npu::Buffer;

pub mod bundle;
mod detr;
#[cfg(feature = "gpu")]
pub mod gpu;
mod mask;
mod neck;
pub mod npu;
pub mod pack;
pub mod post;
mod text;
pub mod tokenizer;
mod tracker;
mod vit;

/// The threaded host math and [`Timing`], from `taconite` (re-exported:
/// they used to live here).
pub use taconite::{Timing, cpu};

pub use detr::Decoded;
pub use post::{Instance, instances, preprocess, upsample_threshold};
pub use text::Text;
pub use tracker::{ImageEmbedding, PointOutput, PointPrompt, point_mask};

use bundle::{Manifest, Store};
use npu::{Io, MhaIo, Npu};
use tokenizer::Tokenizer;

#[derive(Debug)]
pub enum Error {
    Bundle(String),
    Npu(String),
    Input(String),
    /// Vulkan or a shader (feature `gpu`).
    Gpu(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Bundle(m) => write!(f, "bundle: {m}"),
            Error::Npu(m) => write!(f, "NPU: {m}"),
            Error::Input(m) => write!(f, "input: {m}"),
            Error::Gpu(m) => write!(f, "GPU: {m}"),
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
        match e {
            taconite::Error::Gpu(m) => Error::Gpu(m),
            e => Error::Npu(e.to_string()),
        }
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
    /// the device ViT's residual stream in f32 (`AddLayerNorm(f32_residual)`);
    /// bundles from before it keep it bf16
    pub vit_res_f32: bool,
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
            vit_res_f32: m.param("vit_res_f32").is_ok_and(|v| v == "1"),
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
pub(crate) struct Ios {
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
    pub(crate) rope_out: Option<Buffer>,
    pub(crate) xres: Vec<Buffer>,
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
    mode: Mode,
    /// NPU+GPU mode's device, plans and GPU-resident activations
    #[cfg(feature = "gpu")]
    gpu: Option<gpu::Gpu>,
    /// in NPU+GPU mode, the DETR decoder's query layers on the GPU (see
    /// `use_gpu`)
    #[cfg(feature = "gpu")]
    gpu_dec: bool,
}

/// Where the work between the NPU's kernels runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// On the CPU: the NPU-only path (the default).
    Npu,
    /// On the iGPU (feature `gpu`), reading and writing the NPU's buffers in
    /// place: the glue between kernels, and the DETR decoder's query layers.
    NpuGpu,
}

impl Mode {
    /// `npu` or `npu+gpu`.
    pub fn parse(s: &str) -> Result<Self, Error> {
        match s {
            "npu" => Ok(Mode::Npu),
            "npu+gpu" | "npu-gpu" => Ok(Mode::NpuGpu),
            _ => Err(Error::Input(format!("mode {s:?}: expected npu or npu+gpu"))),
        }
    }

    /// `SAM3_MODE`, or [`Mode::Npu`] when it is unset.
    pub fn from_env() -> Result<Self, Error> {
        match std::env::var("SAM3_MODE") {
            Ok(v) if !v.is_empty() => Mode::parse(&v),
            _ => Ok(Mode::Npu),
        }
    }
}

impl fmt::Display for Mode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Mode::Npu => "npu",
            Mode::NpuGpu => "npu+gpu",
        })
    }
}

impl Sam3 {
    /// Loads the bundle, opens the NPU, loads every kernel and uploads the
    /// packed weights, in the mode `SAM3_MODE` names ([`Mode::Npu`] if
    /// unset).
    pub fn load(dir: &Path) -> Result<Self, Error> {
        Self::load_mode(dir, Mode::from_env()?)
    }

    /// [`load`](Self::load) in `mode`. [`Mode::NpuGpu`] needs the `gpu`
    /// feature and a usable GPU; [`Mode::Npu`] never touches the GPU.
    pub fn load_mode(dir: &Path, mode: Mode) -> Result<Self, Error> {
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
        // mutable for NPU+GPU mode's instruction patches
        #[cfg_attr(not(feature = "gpu"), allow(unused_mut))]
        let mut npu = Npu::open(&manifest)?;
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
        // the point prompt path's neck (tracker.rs), on the same kernels
        if manifest.param("tracker").is_ok_and(|v| v == "1") {
            names.extend(["tn.in", "tn.up"].map(String::from));
            names.extend((0..3).map(|i| format!("tn.conv{i}")));
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
                let bytes = t * cfg.vit_dim * if cfg.vit_res_f32 { 4 } else { 2 };
                vec![npu.session.alloc(bytes)?, npu.session.alloc(bytes)?]
            } else {
                vec![]
            },
        };
        #[cfg(feature = "gpu")]
        let gpu = match mode {
            Mode::NpuGpu => Some(gpu::Gpu::new(&store, &cfg, &mut npu, &io)?),
            Mode::Npu => None,
        };
        #[cfg(not(feature = "gpu"))]
        if mode == Mode::NpuGpu {
            return Err(Error::Input("npu+gpu mode needs the `gpu` feature".into()));
        }
        Ok(Sam3 {
            manifest,
            store,
            cfg,
            tokenizer,
            npu,
            w,
            slots,
            io,
            timing: Timing::default(),
            mode,
            #[cfg(feature = "gpu")]
            gpu,
            #[cfg(feature = "gpu")]
            gpu_dec: true,
        })
    }

    pub fn mode(&self) -> Mode {
        self.mode
    }

    /// The hardware the mode runs on: `NPU`, or `NPU + GPU (<device>)`.
    pub fn devices(&self) -> String {
        #[cfg(feature = "gpu")]
        if let Some(g) = &self.gpu {
            return format!("NPU + GPU ({})", g.vk.name);
        }
        "NPU".to_string()
    }

    /// In NPU+GPU mode, runs the DETR decoder's query layers on the GPU
    /// (`on`, the default) or on the CPU (to compare the two); true if they
    /// are now on the GPU.
    pub fn use_gpu(&mut self, on: bool) -> bool {
        #[cfg(feature = "gpu")]
        {
            self.gpu_dec = on && self.gpu.is_some();
            self.gpu_dec
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = on;
            false
        }
    }

    /// Hardware contexts created and evicted so far (evictions happen when
    /// another process holds some of the NPU's contexts).
    pub fn contexts(&self) -> (usize, usize) {
        (self.npu.loads, self.npu.evictions)
    }

    /// Frees every NPU hardware context the model holds, for another model
    /// in the process (NPU2 has 16 across every process, and SAM 3 alone
    /// can take 15); the kernels load again on the model's next run. The
    /// number of contexts freed.
    pub fn release_contexts(&mut self) -> usize {
        self.npu.release()
    }

    fn time<T>(&mut self, key: &str, f: impl FnOnce(&mut Self) -> Result<T, Error>) -> Result<T, Error> {
        let t0 = Instant::now();
        let r = f(self)?;
        self.timing.add(key, t0.elapsed());
        Ok(r)
    }

    /// Everything for one image and prompt: `pixels` is the preprocessed
    /// `[3, 1008, 1008]` image ([`preprocess`]). The text encoder (host)
    /// runs on a side thread while the ViT has the NPU.
    pub fn segment(&mut self, pixels: &[f32], prompt: &str) -> Result<Output, Error> {
        let (ids, mask) = self.tokenizer.encode(prompt);
        // NPU+GPU mode: the ViT's output stays on the NPU, the neck reads it
        let readback = self.mode == Mode::Npu;
        if !self.cfg.vit_device {
            let text = self.time("text", |s| s.text(&ids, &mask))?;
            return self.forward(pixels, &text);
        }
        // NPU+GPU mode also folds the prompt's cross-attention weights on the
        // text thread, while the ViT has the NPU
        let fold = !readback;
        let specs = (self.npu.spec("d_s")?.clone(), self.npu.spec("d_c")?.clone());
        let Sam3 { store, cfg, npu, io, w, timing, slots, .. } = self;
        #[cfg(feature = "gpu")]
        let gpu_gelu = self.gpu.as_ref().and_then(|g| g.glue.vit_gelu.as_ref().map(|p| (g, p)));
        #[cfg(not(feature = "gpu"))]
        let gpu_gelu: Option<((), ())> = None;
        let (text, vit) = std::thread::scope(|s| {
            let encoder = s.spawn(|| -> (Result<Text, Error>, Duration, Duration) {
                let t0 = Instant::now();
                let r = cpu::run_inline(|| text::encode(store, cfg, &ids, &mask));
                let dt = t0.elapsed();
                let t1 = Instant::now();
                let r = r.and_then(|text| {
                    if fold {
                        cpu::run_inline(|| -> Result<(), Error> {
                            for i in 0..cfg.d_layers {
                                let (p, sl) = (format!("d.{i}.ca"), format!("d.{i}"));
                                detr::fold_cross_into(store, cfg, (&specs.0, &specs.1), slots, &p, &sl, &text)?;
                            }
                            detr::fold_cross_into(store, cfg, (&specs.0, &specs.1), slots, "m.ca", "m", &text)
                        })?;
                    }
                    Ok(text)
                });
                (r, dt, t1.elapsed())
            });
            let t0 = Instant::now();
            let mut after_fc1 = |t: &mut Timing| -> Result<(), Error> {
                #[cfg(feature = "gpu")]
                if let Some((g, plan)) = gpu_gelu {
                    return g.run(plan, "vit_gelu", t);
                }
                let _ = (&gpu_gelu, t);
                Ok(())
            };
            let vit = vit::vit_device(npu, io, w, store, cfg, timing, pixels, readback, &mut after_fc1);
            timing.add("vit", t0.elapsed());
            let (text, dt, fold_dt) = encoder.join().expect("the text encoder thread");
            timing.add("text", dt);
            if fold {
                timing.add("prompt_fold (overlapped)", fold_dt);
            }
            (text, vit)
        });
        let (text, vit) = (text?, vit?);
        #[cfg(feature = "gpu")]
        if !readback {
            return self.finish_gpu(None, &text, true);
        }
        self.finish(&vit, &text)
    }

    /// The forward from preprocessed pixels and an encoded prompt.
    pub fn forward(&mut self, pixels: &[f32], text: &Text) -> Result<Output, Error> {
        let vit = self.time("vit", |s| s.vit(pixels))?;
        self.finish(&vit, text)
    }

    /// The forward after the backbone.
    fn finish(&mut self, vit: &[f32], text: &Text) -> Result<Output, Error> {
        #[cfg(feature = "gpu")]
        if self.gpu.is_some() {
            return self.finish_gpu(Some(vit), text, false);
        }
        let fpn = self.time("neck", |s| s.neck(vit))?;
        let enc = self.time("detr_enc", |s| s.detr_encoder(&fpn[2], text))?;
        let dec = self.time("detr_dec", |s| s.detr_decoder(&enc, text))?;
        let (masks, semantic) = self.time("mask_dec", |s| s.mask_decoder(&dec.hidden, &fpn, &enc, text))?;
        Ok(Output { logits: dec.logits, boxes: dec.boxes, presence: dec.presence, masks, semantic })
    }
}

impl Sam3 {
    /// NPU+GPU mode's forward after the backbone: every stage's input and
    /// output stays on the devices; `vit` is the backbone's output if the
    /// host has it (otherwise the neck reads the NPU's).
    #[cfg(feature = "gpu")]
    fn finish_gpu(&mut self, vit: Option<&[f32]>, text: &Text, folded: bool) -> Result<Output, Error> {
        self.time("neck", |s| s.neck_gpu(vit))?;
        self.time("detr_enc", |s| s.detr_encoder_gpu(text, folded))?;
        let dec = self.time("detr_dec", |s| s.detr_decoder_gpu(text))?;
        let (masks, semantic) = self.time("mask_dec", |s| s.mask_decoder_gpu(&dec.hidden, text, folded))?;
        Ok(Output { logits: dec.logits, boxes: dec.boxes, presence: dec.presence, masks, semantic })
    }

    /// The NPU+GPU mode's state (only called in that mode).
    #[cfg(feature = "gpu")]
    fn gpu(&self) -> &gpu::Gpu {
        self.gpu.as_ref().expect("NPU+GPU mode")
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
            *o = taconite::f32_to_bf16(src[i0 * (1 << 14) + i]);
        }
    });
}

// A model loads on one thread and runs on a worker (see `taconite`'s
// types): it must stay `Send`, with the GPU decoder too.
const _: () = {
    const fn send<T: Send>() {}
    send::<Sam3>();
};
