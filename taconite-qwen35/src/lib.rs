// SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
// SPDX-License-Identifier: Apache-2.0

//! Qwen3.5-2B (`Qwen/Qwen3.5-2B`, its text model) chatting on an AMD XDNA
//! NPU.
//!
//! The model is a hybrid: 18 Gated DeltaNet (linear attention) layers and
//! 6 gated softmax-attention layers, each followed by a SwiGLU MLP. The
//! bundle `iron/applications/qwen3_5/export_qwen35.py` writes holds every
//! compiled IRON kernel, the weights (the NPU's pre-packed) and the
//! tokenizer; this crate replays the forward the Python app runs:
//!
//! | | NPU | host (here) |
//! |---|---|---|
//! | prompt | every projection as an `flm.GEMM` over 256-row chunks, one hardware context | tokenizer, embedding, norms, the DeltaNet's conv + recurrence, RoPE, attention |
//! | a generated token | every projection and the LM head as a `GEMVbfp16`, a second context | the same, one row |
//! | an image (the vision tower) | every projection as an `flm.GEMM` (K = 1024), a third context | preprocessing (`image.rs`), LayerNorms, 2D RoPE, attention |
//!
//! [`Qwen35::chat`] answers one user turn (greedy), with or without
//! images; [`Qwen35::check`] verifies a bundle against the references its
//! exporter recorded.

use std::fmt;
use std::path::Path;
use std::time::Instant;

pub use taconite::Timing;
use taconite_bundle::{Manifest, Store};

pub mod image;
pub mod model;
pub mod npu;
pub mod tokenizer;
pub mod vision;

use model::{ImageInput, Model};
use tokenizer::Tokenizer;

pub const VERSION: u32 = 1;

#[cfg(not(any(feature = "xrt", feature = "direct")))]
compile_error!("taconite-qwen35 needs the `xrt` or the `direct` feature");

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

/// An 8-bit RGB image, rows top to bottom, `[height, width, 3]`.
#[derive(Debug, Clone)]
pub struct RgbImage {
    pub width: usize,
    pub height: usize,
    pub rgb: Vec<u8>,
}

/// How to answer a turn.
#[derive(Debug, Clone)]
pub struct ChatOptions {
    pub system: Option<String>,
    /// let the model reason in a `<think>` block first
    pub thinking: bool,
    pub max_new: usize,
    /// images the turn shows, before its text
    pub images: Vec<RgbImage>,
}

impl Default for ChatOptions {
    fn default() -> Self {
        ChatOptions { system: None, thinking: false, max_new: 512, images: Vec::new() }
    }
}

/// What a generation did.
#[derive(Debug, Clone, Default)]
pub struct Stats {
    /// the vision tower's time over the turn's images
    pub vision_s: f64,
    pub prompt_tokens: usize,
    pub new_tokens: usize,
    pub prefill_s: f64,
    pub decode_s: f64,
}

impl Stats {
    /// Tokens a second after the first (which the prefill produces).
    pub fn decode_tok_s(&self) -> f64 {
        if self.new_tokens > 1 && self.decode_s > 0.0 { (self.new_tokens - 1) as f64 / self.decode_s } else { 0.0 }
    }
}

pub struct Qwen35 {
    pub model: Model,
    pub tok: Tokenizer,
    manifest: Manifest,
    store: Store,
    pub timing: Timing,
}

fn argmax(x: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &v) in x.iter().enumerate() {
        if v > x[best] {
            best = i;
        }
    }
    best as u32
}

fn unhex(s: &str) -> Result<String, Error> {
    if s == "-" {
        return Ok(String::new());
    }
    let bytes: Option<Vec<u8>> =
        (0..s.len()).step_by(2).map(|i| s.get(i..i + 2).and_then(|h| u8::from_str_radix(h, 16).ok())).collect();
    bytes.and_then(|b| String::from_utf8(b).ok()).ok_or_else(|| Error::Bundle(format!("bad hex string {s}")))
}

/// Streams text from token bytes: holds back an incomplete UTF-8 sequence
/// until the token that completes it.
#[derive(Default)]
struct Utf8Stream {
    pending: Vec<u8>,
}

impl Utf8Stream {
    fn push(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        let valid = match std::str::from_utf8(&self.pending) {
            Ok(_) => self.pending.len(),
            // an invalid sequence (not just a truncated one) goes out
            // replaced, as the final decode would show it
            Err(e) if e.error_len().is_some() => self.pending.len(),
            Err(e) => e.valid_up_to(),
        };
        let out = String::from_utf8_lossy(&self.pending[..valid]).into_owned();
        self.pending.drain(..valid);
        out
    }
}

fn greedy(
    model: &mut Model,
    store: &Store,
    timing: &mut Timing,
    ids: &[u32],
    image: Option<&ImageInput>,
    max_new: usize,
    mut on_token: impl FnMut(u32),
) -> Result<(Vec<u32>, Stats), Error> {
    let t0 = Instant::now();
    let mut logits = model.prefill(store, ids, image, timing)?;
    let prefill_s = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    let budget = max_new.min(model.max_ctx.saturating_sub(ids.len()));
    let mut out = Vec::new();
    while out.len() < budget {
        let t = argmax(&logits);
        if model.cfg.stop.contains(&t) {
            break;
        }
        out.push(t);
        on_token(t);
        if out.len() == budget {
            break;
        }
        logits = model.decode(store, t, timing)?;
    }
    let stats = Stats {
        vision_s: 0.0,
        prompt_tokens: ids.len(),
        new_tokens: out.len(),
        prefill_s,
        decode_s: t1.elapsed().as_secs_f64(),
    };
    Ok((out, stats))
}

impl Qwen35 {
    /// Loads a bundle: every kernel into the NPU, the weights into device
    /// buffers. `max_ctx` bounds prompt + answer (the KV cache's length).
    pub fn load(dir: &Path, max_ctx: usize) -> Result<Self, Error> {
        let manifest = Manifest::load(dir, VERSION)?;
        let store = Store::load(dir)?;
        let json = std::fs::read_to_string(dir.join("tokenizer.json"))
            .map_err(|e| Error::Bundle(format!("tokenizer.json: {e}")))?;
        let tok = Tokenizer::from_hf_json(&json).map_err(Error::Bundle)?;
        let model = Model::load(&manifest, &store, max_ctx)?;
        Ok(Qwen35 { model, tok, manifest, store, timing: Timing::default() })
    }

    /// One user turn through the checkpoint's chat template, ready for
    /// the answer. `image_tokens`: each shown image's token count (they
    /// come before the text, `<|image_pad|>` repeated that often).
    pub fn prompt_ids(&self, message: &str, system: Option<&str>, thinking: bool, image_tokens: &[usize]) -> Vec<u32> {
        let mut text = String::new();
        if let Some(s) = system {
            text += &format!("<|im_start|>system\n{}<|im_end|>\n", s.trim());
        }
        let images: String = image_tokens
            .iter()
            .map(|&n| format!("<|vision_start|>{}<|vision_end|>", "<|image_pad|>".repeat(n)))
            .collect();
        let content = format!("{images}{message}");
        text += &format!("<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n", content.trim());
        if !thinking {
            text += "\n</think>\n\n";
        }
        self.tok.encode(&text)
    }

    /// Greedy continuation of `ids` up to a stop token or `max_new`
    /// tokens; `on_token` sees each new token as it is chosen.
    pub fn generate(
        &mut self,
        ids: &[u32],
        max_new: usize,
        on_token: impl FnMut(u32),
    ) -> Result<(Vec<u32>, Stats), Error> {
        greedy(&mut self.model, &self.store, &mut self.timing, ids, None, max_new, on_token)
    }

    /// An image's patches and grid, preprocessed as the checkpoint's
    /// processor does (see [`image`]).
    pub fn preprocess(&self, img: &RgbImage) -> Result<(Vec<f32>, (usize, usize)), Error> {
        let v = self.model.vision.as_ref().ok_or_else(|| Error::Input("the bundle has no vision tower".into()))?;
        let c = &v.cfg;
        if img.rgb.len() != img.width * img.height * 3 {
            return Err(Error::Input("image data is not width x height x 3 bytes".into()));
        }
        Ok(image::preprocess(&img.rgb, img.width, img.height, c.patch, c.merge, c.min_pixels, c.max_pixels))
    }

    /// Images -> (their token embeddings, concatenated; each one's grid).
    fn encode(&mut self, images: &[RgbImage]) -> Result<(Vec<f32>, Vec<(usize, usize)>), Error> {
        let (mut emb, mut grids) = (Vec::new(), Vec::new());
        for img in images {
            let (patches, grid) = self.preprocess(img)?;
            emb.extend(self.model.encode_image(&patches, grid, &mut self.timing)?);
            grids.push(grid);
        }
        Ok((emb, grids))
    }

    /// The prefill inputs of a prompt with images: M-RoPE positions.
    fn image_input<'a>(
        &self,
        ids: &[u32],
        emb: &'a [f32],
        grids: &[(usize, usize)],
        pos: &'a mut Vec<[usize; 3]>,
    ) -> Result<ImageInput<'a>, Error> {
        let v = self.model.vision.as_ref().ok_or_else(|| Error::Input("the bundle has no vision tower".into()))?;
        let (p, next) = model::positions(ids, grids, v.cfg.image_token, v.cfg.merge)?;
        *pos = p;
        Ok(ImageInput { image_token: v.cfg.image_token, embeddings: emb, positions: pos, next })
    }

    /// Answers `message`; `stream` gets the answer's text as it grows.
    pub fn chat(
        &mut self,
        message: &str,
        opts: &ChatOptions,
        mut stream: impl FnMut(&str),
    ) -> Result<(String, Stats), Error> {
        let t0 = Instant::now();
        let (emb, grids) = self.encode(&opts.images)?;
        let vision_s = t0.elapsed().as_secs_f64();
        let merge2 = self.model.vision.as_ref().map_or(1, |v| v.cfg.merge * v.cfg.merge);
        let counts: Vec<usize> = grids.iter().map(|g| g.0 * g.1 / merge2).collect();
        let ids = self.prompt_ids(message, opts.system.as_deref(), opts.thinking, &counts);
        let mut pos = Vec::new();
        let image = if grids.is_empty() { None } else { Some(self.image_input(&ids, &emb, &grids, &mut pos)?) };
        let mut utf8 = Utf8Stream::default();
        let tok = &self.tok;
        let (out, mut stats) =
            greedy(&mut self.model, &self.store, &mut self.timing, &ids, image.as_ref(), opts.max_new, |t| {
                let bytes = tok.decode_bytes(&[t], true);
                let s = utf8.push(&bytes);
                if !s.is_empty() {
                    stream(&s);
                }
            })?;
        stats.vision_s = vision_s;
        Ok((self.tok.decode(&out, true), stats))
    }

    /// Checks the bundle against what its exporter recorded: the
    /// tokenizer on its test strings, then each reference run -- its
    /// prompt through the chat template, every step teacher-forced on the
    /// reference's tokens (the next token must agree wherever the float32
    /// reference's top two logits are 0.5 or more apart), and a free
    /// greedy run. Prints a report; returns whether everything passed.
    pub fn check(&mut self, mut say: impl FnMut(&str)) -> Result<bool, Error> {
        let mut ok = true;
        let cases: Vec<(String, Vec<u32>)> = self
            .manifest
            .tagged("tokcase")
            .map(|r| {
                Ok((
                    unhex(r.field(0)?)?,
                    r.field(1)?.split(',').filter(|s| !s.is_empty()).map(|s| s.parse().unwrap_or(u32::MAX)).collect(),
                ))
            })
            .collect::<Result<_, Error>>()?;
        let bad: Vec<&String> = cases.iter().filter(|(t, ids)| &self.tok.encode(t) != ids).map(|(t, _)| t).collect();
        say(&format!("tokenizer: {}/{} strings encode as HF does", cases.len() - bad.len(), cases.len()));
        for t in &bad {
            say(&format!("  MISMATCH {t:?}: {:?}", self.tok.encode(t)));
        }
        ok &= bad.is_empty();

        let refs: Vec<(String, String, bool, bool)> = self
            .manifest
            .tagged("ref")
            .map(|r| {
                let image = r.has("image") && r.get::<u8>("image")? == 1;
                Ok((r.field(0)?.to_string(), unhex(r.str("prompt")?)?, r.get::<u8>("thinking")? == 1, image))
            })
            .collect::<Result<_, Error>>()?;
        for (name, prompt, thinking, has_image) in refs {
            let p = |k: &str| format!("ref.{name}.{k}");
            let ids: Vec<u32> = self.store.i32(&p("ids"))?.iter().map(|&x| x as u32).collect();
            let toks: Vec<u32> = self.store.i32(&p("tokens"))?.iter().map(|&x| x as u32).collect();
            let top_ids = self.store.i32(&p("top_ids"))?.to_vec();
            let top = self.store.f32(&p("top_logits"))?.to_vec();
            let k = top.len() / toks.len();
            self.timing.clear();
            let (mut emb, mut grids, mut counts) = (Vec::new(), Vec::new(), Vec::new());
            if has_image {
                // the exporter's decoded pixels -> our patches (against the
                // app's) -> the vision tower (against the float32 model's)
                let shape = self.store.shape(&p("rgb"))?.to_vec();
                let img = RgbImage { height: shape[0], width: shape[1], rgb: self.store.u8(&p("rgb"))?.to_vec() };
                let (patches, grid) = self.preprocess(&img)?;
                let want = self.store.f32(&p("patches"))?;
                let dp = patches.iter().zip(want).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
                let want_grid = self.store.i32(&p("grid"))?;
                let same_grid = grid == (want_grid[0] as usize, want_grid[1] as usize) && patches.len() == want.len();
                let t0 = Instant::now();
                emb = self.model.encode_image(&patches, grid, &mut self.timing)?;
                let vs = t0.elapsed().as_secs_f64();
                let r = self.store.f32(&p("vision"))?;
                let (mut dot, mut nn, mut rr, mut dd) = (0f64, 0f64, 0f64, 0f64);
                for (&a, &b) in emb.iter().zip(r) {
                    let (a, b) = (a as f64, b as f64);
                    dot += a * b;
                    nn += a * a;
                    rr += b * b;
                    dd += (a - b) * (a - b);
                }
                let cos = dot / (nn.sqrt() * rr.sqrt());
                say(&format!(
                    "{name} image: {}x{} -> grid {grid:?} ({}), patches max |d| vs the app's {dp:.2e}; vision vs float32: cosine {cos:.4}, relative RMS {:.3}; {vs:.2} s",
                    img.width,
                    img.height,
                    if same_grid { "matches" } else { "MISMATCH" },
                    (dd / rr).sqrt(),
                ));
                ok &= same_grid && dp < 1e-6 && cos > 0.99 && emb.len() == r.len();
                counts.push(grid.0 * grid.1 / 4);
                grids.push(grid);
            }
            let same_prompt = self.prompt_ids(&prompt, None, thinking, &counts) == ids;
            ok &= same_prompt;
            let mut pos = Vec::new();
            let image = if has_image { Some(self.image_input(&ids, &emb, &grids, &mut pos)?) } else { None };

            let t0 = Instant::now();
            let mut logits = self.model.prefill(&self.store, &ids, image.as_ref(), &mut self.timing)?;
            let prefill_s = t0.elapsed().as_secs_f64();
            let (mut agree, mut flips, mut real, mut max_d) = (0, Vec::new(), 0, 0f32);
            let t1 = Instant::now();
            for (j, &want) in toks.iter().enumerate() {
                let (ti, tl) = (&top_ids[j * k..(j + 1) * k], &top[j * k..(j + 1) * k]);
                for (&id, &l) in ti.iter().zip(tl) {
                    max_d = max_d.max((logits[id as usize] - l).abs());
                }
                if argmax(&logits) == want {
                    agree += 1;
                } else {
                    let margin = tl[0] - tl[1];
                    flips.push(format!("{j} ({margin:.3})"));
                    if margin > 0.5 {
                        real += 1;
                    }
                }
                if j + 1 < toks.len() {
                    logits = self.model.decode(&self.store, want, &mut self.timing)?;
                }
            }
            let dec_ms = t1.elapsed().as_secs_f64() * 1e3 / toks.len().saturating_sub(1).max(1) as f64;
            say(&format!(
                "{name}: {} prompt tokens (template {}), {} steps: top-1 agrees {agree}/{}, max |dlogit| over the reference's top {k} {max_d:.3}, flips (ref margin) [{}]; prefill {prefill_s:.2} s, {dec_ms:.0} ms/token",
                ids.len(),
                if same_prompt { "matches" } else { "MISMATCH" },
                toks.len(),
                toks.len(),
                flips.join(", "),
            ));
            ok &= real == 0;
            let (out, stats) =
                greedy(&mut self.model, &self.store, &mut self.timing, &ids, image.as_ref(), toks.len(), |_| {})?;
            let same = out.iter().zip(&toks).take_while(|(a, b)| a == b).count();
            say(&format!(
                "  free run: first {same} of {} tokens as the reference's, {:.1} tokens/s: {:?}",
                toks.len(),
                stats.decode_tok_s(),
                self.tok.decode(&out, true)
            ));
        }
        say(if ok { "PASS" } else { "FAIL" });
        Ok(ok)
    }
}
