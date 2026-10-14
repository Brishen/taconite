// SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
// SPDX-License-Identifier: Apache-2.0

//! NVIDIA RE-USE (`nvidia/RE-USE`, universal speech enhancement: SEMamba,
//! 30 bidirectional time-frequency Mamba blocks between a dense
//! convolutional encoder and two decoders) on an AMD XDNA NPU: noisy,
//! reverberant, clipped, band-limited or codec-damaged speech at 8 to 48
//! kHz in, enhanced speech at the same rate out.
//!
//! The bundle `iron/applications/reuse/export_reuse.py` writes holds every
//! compiled IRON kernel and the weights (the NPU ones pre-packed); this
//! crate replays the forward the Python app runs (`reuse_common.py` /
//! `reuse_npu.py`):
//!
//! | | NPU | host (here) |
//! |---|---|---|
//! | STFT (n_fft 320 at 8 kHz, scaled with the rate; hop 1/8 of it) | | `stft.rs` (and its inverse) |
//! | encoder (dense block of 4 dilated 3 x 3 convs, a strided conv) | every conv as an `flm.GEMM` | the 1 x 1 input conv, the map layouts, the dilated taps' sums, InstanceNorm + PReLU |
//! | 30 x (time Mamba, frequency Mamba), both directions | in_proj, causal conv1d + x_proj, the selective scan, the output projection (the `MambaScan` operator's phases) | token layouts, both directions summed, LayerNorm, residual |
//! | decoders (dense block, two sub-pixel upsampling convs) | every conv as an `flm.GEMM` | layouts, InstanceNorm + PReLU, the 1 x 1 heads, atan2 |
//!
//! [`Reuse::enhance`] takes any length: up to the bundle's longest Mamba
//! grid (`t_max`, ~5 s) in one pass, as the checkpoint's `inference.py`;
//! longer audio in 5 s chunks at a 2.5 s hop, overlap-added under a Hann
//! window, as its `inference_chunk.py`.

use std::collections::HashMap;
use std::fmt;
use std::path::Path;
use std::time::Instant;

use taconite_bundle::{Manifest, Store};

pub use taconite::Timing;

#[cfg(feature = "embed")]
pub mod embed;
#[cfg(feature = "encode")]
pub mod encode;
#[cfg(feature = "decode")]
pub mod media;
pub mod model;
pub mod npu;
pub mod stft;
pub mod wav;

use model::Model;
use npu::Npu;
use stft::Stft;

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

/// The model's constants and the bundle's layout (the manifest's params).
#[derive(Debug, Clone)]
pub struct Config {
    pub blocks: usize,
    /// n_fft, hop, window at `base_sr` (scaled to the input's rate)
    pub n_fft: usize,
    pub hop: usize,
    pub win: usize,
    pub base_sr: usize,
    /// the LayerNorms' and the InstanceNorms' epsilon
    pub eps: f32,
    pub in_eps: f32,
    pub pix: usize,
    pub padt: usize,
    pub skip: usize,
    pub rec: usize,
    pub yw: usize,
    /// the sample rates the bundle has kernels for
    pub rates: Vec<u32>,
    /// the Mamba grid's frame buckets: t_step, 2 t_step, ... t_max
    pub t_step: usize,
    pub t_max: usize,
    pub chunk_secs: f32,
    pub chunk_hop: f32,
}

impl Config {
    fn load(m: &Manifest) -> Result<Self, Error> {
        let p = |k: &str| m.param_as::<usize>(k);
        let f = |k: &str| m.param_as::<f32>(k);
        if p("hid")? != 64 || p("d_state")? != 16 {
            return Err(Error::Bundle("not a RE-USE (hid 64, d_state 16) bundle".into()));
        }
        Ok(Config {
            blocks: p("blocks")?,
            n_fft: p("n_fft")?,
            hop: p("hop")?,
            win: p("win")?,
            base_sr: p("base_sr")?,
            eps: f("eps")?,
            in_eps: f("in_eps")?,
            pix: p("pix")?,
            padt: p("padt")?,
            skip: p("skip")?,
            rec: p("rec")?,
            yw: p("yw")?,
            rates: m.list("rates")?,
            t_step: p("t_step")?,
            t_max: p("t_max")?,
            chunk_secs: f("chunk_secs")?,
            chunk_hop: f("chunk_hop")?,
        })
    }

    /// n_fft, hop, window at `sr` (inference.py: v sr // base_sr, rounded
    /// up to even).
    pub fn stft_sizes(&self, sr: u32) -> (usize, usize, usize) {
        let even = |v: usize| v + v % 2;
        let s = |v: usize| even(v * sr as usize / self.base_sr);
        (s(self.n_fft), s(self.hop), s(self.win))
    }

    /// The Mamba grid's frames for `samples` samples at `sr`.
    pub fn grid_frames(&self, sr: u32, samples: usize) -> usize {
        let (_, hop, _) = self.stft_sizes(sr);
        (1 + samples / hop + 2 - 1) / 4 + 1
    }
}

pub struct Reuse {
    pub cfg: Config,
    pub npu: Npu,
    pub model: Model,
    pub manifest: Manifest,
    pub store: Store,
    stfts: HashMap<u32, Stft>,
    /// where the last call spent its time (`npu:<kernel>`, host stages)
    pub timing: Timing,
}

impl Reuse {
    /// Loads a bundle: opens the NPU and uploads the packed weights
    /// (kernels load on first use).
    pub fn load(dir: &Path) -> Result<Self, Error> {
        let manifest = Manifest::load(dir, VERSION)?;
        let store = Store::load(dir)?;
        let cfg = Config::load(&manifest)?;
        let npu = Npu::open(&manifest)?;
        let model = Model::load(&store, &npu)?;
        Ok(Reuse { cfg, npu, model, manifest, store, stfts: HashMap::new(), timing: Timing::default() })
    }

    /// Hardware contexts the bundle's kernels use.
    pub fn contexts(&self) -> usize {
        self.npu.contexts
    }

    /// Frees every NPU hardware context the model holds, for other models
    /// (NPU2 has 16 across every process); the kernels load again on the
    /// next call. The number of contexts freed.
    pub fn release_contexts(&mut self) -> usize {
        self.npu.release()
    }

    /// Enhances one channel `wav` at `sr` (one of the bundle's rates):
    /// in one pass when its Mamba grid fits the bundle (`t_max`), else in
    /// chunks (inference_chunk.py's). Same length out.
    pub fn enhance(&mut self, wav: &[f32], sr: u32) -> Result<Vec<f32>, Error> {
        self.enhance_with(wav, sr, &mut |_| {})
    }

    /// `enhance`, calling `progress` with the fraction done (0 to 1, over
    /// every chunk) as it goes: a few dozen calls a pass, on this thread.
    pub fn enhance_with(&mut self, wav: &[f32], sr: u32, progress: &mut dyn FnMut(f64)) -> Result<Vec<f32>, Error> {
        if !self.cfg.rates.contains(&sr) {
            return Err(Error::Input(format!(
                "{sr} Hz: this bundle has kernels for {:?} Hz (export_reuse.py --rates; or resample)",
                self.cfg.rates
            )));
        }
        self.timing.clear();
        if self.cfg.grid_frames(sr, wav.len()) <= self.cfg.t_max {
            let y = self.enhance_whole_with(wav, sr, progress)?;
            progress(1.0);
            return Ok(y);
        }
        let size = (self.cfg.chunk_secs * sr as f32) as usize;
        let hop = (self.cfg.chunk_hop * size as f32) as usize;
        let s = wav.len();
        let n = (s - size).div_ceil(hop) + 1;
        let win: Vec<f32> =
            (0..size).map(|i| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / size as f64).cos()).max(1e-7) as f32).collect();
        let mut out = vec![0f32; s];
        let mut wsum = vec![0f32; s];
        for i in 0..n {
            let lo = if i < n - 1 { i * hop } else { s - size };
            let y = self.enhance_whole_with(&wav[lo..lo + size], sr, &mut |f| progress((i as f64 + f) / n as f64))?;
            for (k, &v) in y.iter().enumerate() {
                out[lo + k] += v * win[k];
                wsum[lo + k] += win[k];
            }
        }
        progress(1.0);
        Ok(out.iter().zip(&wsum).map(|(a, b)| a / b).collect())
    }

    /// One pass of the network over the whole signal (inference.py): STFT,
    /// the forward, frames mostly at zero magnitude muted, inverse STFT,
    /// the length matched to the input (padded with 1e-8).
    pub fn enhance_whole(&mut self, wav: &[f32], sr: u32) -> Result<Vec<f32>, Error> {
        self.enhance_whole_with(wav, sr, &mut |_| {})
    }

    /// `enhance_whole`, calling `tick` with the fraction of the pass done.
    pub fn enhance_whole_with(&mut self, wav: &[f32], sr: u32, tick: &mut dyn FnMut(f64)) -> Result<Vec<f32>, Error> {
        let (n_fft, hop, win) = self.cfg.stft_sizes(sr);
        if wav.len() <= n_fft / 2 {
            return Err(Error::Input(format!("{} samples: shorter than half an STFT window ({})", wav.len(), n_fft / 2)));
        }
        let st = self.stfts.entry(sr).or_insert_with(|| Stft::new(n_fft, hop, win));
        let t0 = Instant::now();
        let (mag, pha) = st.forward(wav);
        let (fq, tf) = (st.bins(), st.frames(wav.len()));
        self.timing.add("host.stft", t0.elapsed());
        let (em, ep) = self.model.forward_with(&self.cfg, &self.npu, &mag, &pha, fq, tf, &mut self.timing, tick)?;
        let t0 = Instant::now();
        let mut lin = vec![0f32; fq * tf];
        for t in 0..tf {
            let zero = (0..fq).filter(|&f| em[f * tf + t] <= 0.0).count();
            if zero * 2 > fq {
                continue; // muted: inference.py's "strange sweep artifact" guard
            }
            for f in 0..fq {
                lin[f * tf + t] = em[f * tf + t].max(0.0).exp_m1();
            }
        }
        let st = &self.stfts[&sr];
        let mut y = st.inverse(&lin, &ep, tf);
        y.resize(wav.len(), 1e-8);
        self.timing.add("host.istft", t0.elapsed());
        Ok(y)
    }
}

/// SNR of `x` against `reference`, in dB.
pub fn snr(x: &[f32], reference: &[f32]) -> f64 {
    let (mut s, mut e) = (0f64, 0f64);
    for (a, b) in x.iter().zip(reference) {
        s += (*b as f64).powi(2);
        e += (*a as f64 - *b as f64).powi(2);
    }
    10.0 * (s / e).log10()
}
