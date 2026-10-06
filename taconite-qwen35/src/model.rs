// SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
// SPDX-License-Identifier: Apache-2.0

//! The forward, as `iron/applications/qwen3_5/qwen35_npu.py` runs it: the
//! projections on the NPU (prefill: `flm.GEMM`s over the prompt's 256-row
//! chunks, one dispatch a projection; decode: `GEMVbfp16`s), the rest on
//! the host in f32 -- the norms and residual adds, the DeltaNet's causal
//! convolution, gates and recurrence (one step a token; the heads in
//! parallel), partial M-RoPE, the attention layers' KV cache and GQA
//! attention. With a vision tower in the bundle ([`crate::vision`]), an
//! image's embeddings take its tokens' rows and M-RoPE's (t, h, w)
//! positions.

use std::time::Instant;

use taconite::cpu::{self, par_rows};
use taconite::{Timing, bf16_to_f32, f32_to_bf16};
use taconite_bundle::{Manifest, Store};

use crate::Error;
use crate::npu::{Buffer, Npu, gemm_key, gemv_key};
use crate::vision::{Vision, VisionConfig};

const BF16: usize = 2;
/// A paired epilogue's column half: gate_up's GEMM columns are [32 gate |
/// 32 up] a tile of 64 (the decode GEMV returns them in that order).
const PAIR_HALF: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// Gated DeltaNet (linear attention)
    Lin,
    /// gated softmax attention
    Att,
}

/// Model constants (the manifest's `param` records).
#[derive(Debug, Clone)]
pub struct Config {
    pub d: usize,
    pub i: usize,
    pub kinds: Vec<Kind>,
    pub vocab: usize,
    pub eps: f32,
    pub heads: usize,
    pub kv_heads: usize,
    pub hd: usize,
    pub rope_dim: usize,
    pub rope_theta: f64,
    /// M-RoPE: frequency pairs taken from the h and w positions
    /// (sections 1 and 2; the rest from t)
    pub mrope_section: Vec<usize>,
    pub lin_heads: usize,
    pub lin_hk: usize,
    pub lin_hv: usize,
    pub conv_k: usize,
    /// rows of a GEMM chunk; the prompt's rows a device buffer holds
    pub chunk: usize,
    pub max_prompt: usize,
    /// down_proj's K slices (I padded to a multiple of D, over D)
    pub n_down: usize,
    /// rows of a head GEMV, and the head's dispatches
    pub head_rows: usize,
    pub n_head: usize,
    /// tokens that end a generation
    pub stop: Vec<u32>,
}

impl Config {
    pub fn load(m: &Manifest) -> Result<Self, Error> {
        let kinds = m
            .param("layer_types")?
            .split(',')
            .map(|k| match k {
                "lin" => Ok(Kind::Lin),
                "att" => Ok(Kind::Att),
                _ => Err(Error::Bundle(format!("layer type {k}"))),
            })
            .collect::<Result<_, _>>()?;
        Ok(Config {
            d: m.param_as("D")?,
            i: m.param_as("I")?,
            kinds,
            vocab: m.param_as("vocab")?,
            eps: m.param_as("eps")?,
            heads: m.param_as("heads")?,
            kv_heads: m.param_as("kv_heads")?,
            hd: m.param_as("hd")?,
            rope_dim: m.param_as("rope_dim")?,
            rope_theta: m.param_as("rope_theta")?,
            mrope_section: if m.has_param("mrope_section") { m.list("mrope_section")? } else { vec![0, 0, 0] },
            lin_heads: m.param_as("lin_heads")?,
            lin_hk: m.param_as("lin_hk")?,
            lin_hv: m.param_as("lin_hv")?,
            conv_k: m.param_as("conv_k")?,
            chunk: m.param_as("chunk")?,
            max_prompt: m.param_as("max_prompt")?,
            n_down: m.param_as("n_down")?,
            head_rows: m.param_as("head_rows")?,
            n_head: m.param_as("n_head")?,
            stop: m.list("stop")?,
        })
    }

    fn lin_key(&self) -> usize {
        self.lin_heads * self.lin_hk
    }

    fn lin_value(&self) -> usize {
        self.lin_heads * self.lin_hv
    }

    /// channels of the causal convolution: q | k | v
    fn lin_conv(&self) -> usize {
        2 * self.lin_key() + self.lin_value()
    }

    /// the DeltaNet input projection's width: q | k | v | z
    fn n_lin(&self) -> usize {
        self.lin_conv() + self.lin_value()
    }

    /// the attention input projection's width: (q | gate) a head | k | v
    fn n_att(&self) -> usize {
        (2 * self.heads + 2 * self.kv_heads) * self.hd
    }
}

/// One layer's resident weights: the prefill GEMMs' packed B streams
/// (`g_*`, which the decode GEMVs read too), and its host parameters.
struct Layer {
    kind: Kind,
    g_in: Buffer,
    g_o: Buffer,
    g_gu: Buffer,
    g_down: Vec<Buffer>,
    ln1: Vec<f32>,
    ln2: Vec<f32>,
    /// DeltaNet: b | a projections [2 H, D], conv weights [C, k], -exp(A_log),
    /// dt_bias [H], the gated norm's weight [dv]
    ba: Vec<f32>,
    conv: Vec<f32>,
    a_neg: Vec<f32>,
    dt_bias: Vec<f32>,
    norm: Vec<f32>,
    /// attention: the q / k norms' weights [hd]
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
}

/// The per-sequence state.
struct State {
    /// DeltaNet layer i: the last k-1 conv inputs, time-major [k-1, C]
    conv: Vec<Vec<f32>>,
    /// DeltaNet layer i: the recurrent state [H, dk, dv]
    rec: Vec<Vec<f32>>,
    /// attention layer i: K and V a position [max_ctx, KV hd]
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    /// cache rows used
    pos: usize,
    /// the next token's M-RoPE position
    rope_next: usize,
}

pub struct Model {
    pub cfg: Config,
    pub npu: Npu,
    layers: Vec<Layer>,
    /// the embedding [V, D] (bf16) and the final norm's weight
    ln_f: Vec<f32>,
    head: Vec<Buffer>,
    /// prefill buffers (max_prompt rows): the GEMMs' A / C
    a: Buffer,
    lin_c: Buffer,
    att_c: Buffer,
    o_a: Buffer,
    o_c: Buffer,
    gu_c: Buffer,
    p: Vec<Buffer>,
    gv: Gv,
    rope_inv: Vec<f32>,
    /// per RoPE frequency pair: 0 = t, 1 = h, 2 = w
    rope_axis: Vec<usize>,
    pub max_ctx: usize,
    st: State,
    pub vision: Option<Vision>,
    /// the decode GEMVs' weight format, the head's (and so the embeddings')
    /// included: "bfp16" (bfp16ebs8) or "bfp6s" (bfp6s16, 6.5 bits a weight;
    /// IRON's iron/operators/gemv_bfp16/bfp6s.py); manifest `gemv_format`
    gemv_format: String,
}

/// An image-bearing prompt's extra inputs to [`Model::prefill`]: the
/// vision tower's embeddings of its image tokens (in order, `D` a row) and
/// every prompt token's (t, h, w) position ([`positions`]).
pub struct ImageInput<'a> {
    pub image_token: u32,
    pub embeddings: &'a [f32],
    pub positions: &'a [[usize; 3]],
    pub next: usize,
}

/// The (t, h, w) position of every token of `ids`, and the position the
/// next token takes: text advances all three by one; an image's merged
/// tokens (raster order over its `(gh / merge, gw / merge)` grid) sit at
/// t = start, h = start + row, w = start + col, and the text after it
/// starts at start + max(rows, cols). `grids`: each image's patch grid, in
/// prompt order.
pub fn positions(
    ids: &[u32],
    grids: &[(usize, usize)],
    image_token: u32,
    merge: usize,
) -> Result<(Vec<[usize; 3]>, usize), Error> {
    let (mut out, mut pos, mut i) = (Vec::with_capacity(ids.len()), 0, 0);
    let mut grids = grids.iter();
    while i < ids.len() {
        if ids[i] != image_token {
            out.push([pos; 3]);
            pos += 1;
            i += 1;
            continue;
        }
        let &(gh, gw) = grids.next().ok_or_else(|| Error::Input("more image tokens than images".into()))?;
        let (h, w) = (gh / merge, gw / merge);
        for r in 0..h {
            for c in 0..w {
                out.push([pos, pos + r, pos + c]);
            }
        }
        i += h * w;
        pos += h.max(w);
    }
    if grids.next().is_some() || out.len() != ids.len() {
        return Err(Error::Input("images and image tokens do not match".into()));
    }
    Ok((out, pos))
}

/// Decode's buffers: the GEMV input, an output a distinct M, the head's
/// outputs.
struct Gv {
    x: Buffer,
    y: Vec<(usize, Buffer)>,
    head_y: Vec<Buffer>,
}

/// y = W x on the NPU (GEMV `key`) -> y as f32.
fn gemv(npu: &Npu, gv: &mut Gv, key: &str, x: &[f32], w: &Buffer, t: &mut Timing) -> Result<Vec<f32>, Error> {
    write_rows(&mut gv.x, x)?;
    let m = npu.gemv(key)?.m;
    let y = &gv.y.iter().find(|(mm, _)| *mm == m).expect("a buffer for every M").1;
    let t0 = Instant::now();
    npu.run(&gemv_key(key), &[&gv.x, w, y])?;
    t.add(&format!("npu:v.{key}"), t0.elapsed());
    read(y, 0, m)
}

fn f32s(s: &Store, name: &str) -> Result<Vec<f32>, Error> {
    Ok(s.f32(name)?.to_vec())
}

impl Model {
    pub fn load(m: &Manifest, s: &Store, max_ctx: usize) -> Result<Self, Error> {
        let cfg = Config::load(m)?;
        let npu = Npu::open(m)?;
        let up = |name: &str| npu.upload(s.bytes(name)?);
        let mut layers = Vec::with_capacity(cfg.kinds.len());
        for (i, &kind) in cfg.kinds.iter().enumerate() {
            let t = |k: &str| format!("l{i}.{k}");
            let h = |k: &str| f32s(s, &t(&format!("h.{k}")));
            let lin = kind == Kind::Lin;
            let opt = |k: &str, on: bool| if on { h(k) } else { Ok(Vec::new()) };
            layers.push(Layer {
                kind,
                g_in: up(&t("g.in"))?,
                g_o: up(&t("g.o"))?,
                g_gu: up(&t("g.gu"))?,
                g_down: (0..cfg.n_down).map(|j| up(&t(&format!("g.down{j}")))).collect::<Result<_, _>>()?,
                ln1: h("ln1")?,
                ln2: h("ln2")?,
                ba: opt("ba", lin)?,
                conv: opt("conv", lin)?,
                a_neg: opt("A", lin)?,
                dt_bias: opt("dt_bias", lin)?,
                norm: opt("norm", lin)?,
                q_norm: opt("q_norm", !lin)?,
                k_norm: opt("k_norm", !lin)?,
            });
        }
        let ln_f = f32s(s, "ln_f")?;
        let head = (0..cfg.n_head).map(|j| up(&format!("head{j}"))).collect::<Result<_, _>>()?;

        let rows = cfg.max_prompt;
        let gu = npu.gemm("gu")?;
        let a = npu.zeros(rows * cfg.d)?;
        let lin_c = npu.zeros(rows * cfg.n_lin())?;
        let att_c = npu.zeros(rows * cfg.n_att())?;
        let o_a = npu.zeros(rows * cfg.d)?;
        let o_c = npu.zeros(rows * cfg.d)?;
        let gu_c = npu.zeros((rows + 1) * gu.c_stride)?;
        let p = (0..cfg.n_down).map(|_| npu.zeros(rows * cfg.d)).collect::<Result<_, _>>()?;

        let gv_x = npu.zeros(cfg.d)?;
        let mut gv_y: Vec<(usize, Buffer)> = Vec::new();
        for g in npu.gemvs.values() {
            if !gv_y.iter().any(|(m, _)| *m == g.m) {
                gv_y.push((g.m, npu.zeros(g.m)?));
            }
        }
        let head_y = (0..cfg.n_head).map(|_| npu.zeros(cfg.head_rows)).collect::<Result<_, _>>()?;

        let rd = cfg.rope_dim;
        let theta = cfg.rope_theta as f32;
        let rope_inv = (0..rd / 2).map(|j| 1.0 / theta.powf((2 * j) as f32 / rd as f32)).collect();
        let sec = &cfg.mrope_section;
        let rope_axis = (0..rd / 2)
            .map(|j| match j % 3 {
                1 if j < 3 * sec[1] => 1,
                2 if j < 3 * sec[2] => 2,
                _ => 0,
            })
            .collect();
        let vision = match VisionConfig::load(m)? {
            Some(vc) => Some(Vision::load(vc, s, &npu)?),
            None => None,
        };
        let st = State { conv: Vec::new(), rec: Vec::new(), k: Vec::new(), v: Vec::new(), pos: 0, rope_next: 0 };
        let mut model = Model {
            cfg,
            npu,
            layers,
            ln_f,
            head,
            a,
            lin_c,
            att_c,
            o_a,
            o_c,
            gu_c,
            p,
            gv: Gv { x: gv_x, y: gv_y, head_y },
            rope_inv,
            rope_axis,
            max_ctx,
            st,
            vision,
            gemv_format: if m.has_param("gemv_format") { m.param("gemv_format")?.to_string() } else { "bfp16".into() },
        };
        model.reset();
        Ok(model)
    }

    /// Empties the per-sequence state.
    pub fn reset(&mut self) {
        let c = &self.cfg;
        let (mut conv, mut rec, mut k, mut v) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for &kind in &c.kinds {
            let lin = kind == Kind::Lin;
            let n = |on: bool, n: usize| if on { vec![0f32; n] } else { Vec::new() };
            conv.push(n(lin, (c.conv_k - 1) * c.lin_conv()));
            rec.push(n(lin, c.lin_heads * c.lin_hk * c.lin_hv));
            k.push(n(!lin, self.max_ctx * c.kv_heads * c.hd));
            v.push(n(!lin, self.max_ctx * c.kv_heads * c.hd));
        }
        self.st = State { conv, rec, k, v, pos: 0, rope_next: 0 };
    }

    /// Tokens of context so far.
    pub fn pos(&self) -> usize {
        self.st.pos
    }

    // ------------------------------------------------------------ NPU plumbing

    /// An image's patches (see [`crate::image`]) -> its image-token
    /// embeddings, on the vision tower.
    pub fn encode_image(&mut self, patches: &[f32], grid: (usize, usize), t: &mut Timing) -> Result<Vec<f32>, Error> {
        let v = self.vision.as_mut().ok_or_else(|| Error::Input("the bundle has no vision tower".into()))?;
        v.encode(&self.npu, patches, grid, t)
    }

    /// Logits [V] of one final hidden row.
    fn head(&mut self, h: &[f32], t: &mut Timing) -> Result<Vec<f32>, Error> {
        let hn = rms_norm(h, &self.ln_f, self.cfg.eps);
        write_rows(&mut self.gv.x, &hn)?;
        let t0 = Instant::now();
        for (w, y) in self.head.iter().zip(&self.gv.head_y) {
            self.npu.run(&gemv_key("head"), &[&self.gv.x, w, y])?;
        }
        t.add("npu:v.head", t0.elapsed());
        let mut out = Vec::with_capacity(self.cfg.n_head * self.cfg.head_rows);
        for y in &self.gv.head_y {
            out.extend(read(y, 0, self.cfg.head_rows)?);
        }
        out.truncate(self.cfg.vocab);
        Ok(out)
    }

    // ------------------------------------------------------------ forward

    /// Empties the state and runs the prompt; the next token's logits.
    pub fn prefill(
        &mut self,
        store: &Store,
        ids: &[u32],
        image: Option<&ImageInput>,
        t: &mut Timing,
    ) -> Result<Vec<f32>, Error> {
        let (l, d) = (ids.len(), self.cfg.d);
        if l == 0 || l > self.cfg.max_prompt || l >= self.max_ctx {
            return Err(Error::Input(format!(
                "{l} prompt tokens (1..={} fit)",
                self.cfg.max_prompt.min(self.max_ctx - 1)
            )));
        }
        self.reset();
        let mut x = embed(store, &self.cfg, &self.gemv_format, ids)?;
        let text_pos: Vec<[usize; 3]>;
        let (rope_pos, next) = match image {
            Some(im) => {
                let mut rows = im.embeddings.chunks_exact(d);
                for (r, &id) in ids.iter().enumerate() {
                    if id == im.image_token {
                        let e = rows.next().ok_or_else(|| Error::Input("more image tokens than embeddings".into()))?;
                        x[r * d..(r + 1) * d].copy_from_slice(e);
                    }
                }
                if rows.next().is_some() || im.positions.len() != l {
                    return Err(Error::Input("image embeddings or positions do not match the prompt".into()));
                }
                (im.positions, im.next)
            }
            None => {
                text_pos = (0..l).map(|p| [p; 3]).collect();
                (&text_pos[..], l)
            }
        };
        let n = l.div_ceil(self.cfg.chunk);
        let eps = self.cfg.eps;
        for li in 0..self.layers.len() {
            let t0 = Instant::now();
            let lay = &self.layers[li];
            let xn = rms_rows(&x, d, &lay.ln1, eps);
            write_rows(&mut self.a, &xn)?;
            t.add("norm", t0.elapsed());
            let lin = lay.kind == Kind::Lin;
            let (key, width) = if lin { ("lin", self.cfg.n_lin()) } else { ("att", self.cfg.n_att()) };
            let c_in = if lin { &self.lin_c } else { &self.att_c };
            run_gemm(&self.npu, key, &self.a, 0, &lay.g_in, c_in, n, t)?;
            let t0 = Instant::now();
            let y = read(c_in, 0, l * width)?;
            let o = if lin { self.deltanet(li, &y, &xn, l) } else { self.attention(li, &y, l, 0, rope_pos) };
            write_rows(&mut self.o_a, &o)?;
            t.add("mixer", t0.elapsed());
            let lay = &self.layers[li];
            run_gemm(&self.npu, "o", &self.o_a, 0, &lay.g_o, &self.o_c, n, t)?;
            let t0 = Instant::now();
            cpu::add_(&mut x, &read(&self.o_c, 0, l * d)?);
            let xn = rms_rows(&x, d, &lay.ln2, eps);
            write_rows(&mut self.a, &xn)?;
            t.add("norm", t0.elapsed());
            run_gemm(&self.npu, "gu", &self.a, 0, &lay.g_gu, &self.gu_c, n, t)?;
            for s in 0..self.cfg.n_down {
                run_gemm(&self.npu, "down", &self.gu_c, s * d, &lay.g_down[s], &self.p[s], n, t)?;
            }
            let t0 = Instant::now();
            for p in &self.p {
                cpu::add_(&mut x, &read(p, 0, l * d)?);
            }
            t.add("norm", t0.elapsed());
        }
        self.st.pos = l;
        self.st.rope_next = next;
        self.head(&x[(l - 1) * d..], t)
    }

    /// Appends one token at the next position; the next token's logits.
    pub fn decode(&mut self, store: &Store, token: u32, t: &mut Timing) -> Result<Vec<f32>, Error> {
        if self.st.pos >= self.max_ctx {
            return Err(Error::Input(format!("context full ({} tokens)", self.max_ctx)));
        }
        let (d, eps) = (self.cfg.d, self.cfg.eps);
        let mut x = embed(store, &self.cfg, &self.gemv_format, &[token])?;
        for li in 0..self.layers.len() {
            let t0 = Instant::now();
            let xn = rms_norm(&x, &self.layers[li].ln1, eps);
            t.add("norm", t0.elapsed());
            let lay = &self.layers[li];
            let lin = lay.kind == Kind::Lin;
            let y = gemv(&self.npu, &mut self.gv, if lin { "lin" } else { "att" }, &xn, &lay.g_in, t)?;
            let t0 = Instant::now();
            let rp = [[self.st.rope_next; 3]];
            let o = if lin { self.deltanet(li, &y, &xn, 1) } else { self.attention(li, &y, 1, self.st.pos, &rp) };
            t.add("mixer", t0.elapsed());
            let lay = &self.layers[li];
            cpu::add_(&mut x, &gemv(&self.npu, &mut self.gv, "o", &o, &lay.g_o, t)?);
            let t0 = Instant::now();
            let xn = rms_norm(&x, &lay.ln2, eps);
            t.add("norm", t0.elapsed());
            let gu = gemv(&self.npu, &mut self.gv, "gu", &xn, &lay.g_gu, t)?;
            // the GEMM's paired column order: [32 gate | 32 up] a block of 64
            let mut hmid = vec![0f32; self.cfg.n_down * d];
            for (b, pair) in gu.chunks_exact(2 * PAIR_HALF).enumerate() {
                for j in 0..PAIR_HALF {
                    hmid[b * PAIR_HALF + j] = silu(pair[j]) * pair[PAIR_HALF + j];
                }
            }
            for s in 0..self.cfg.n_down {
                let y = gemv(&self.npu, &mut self.gv, "down", &hmid[s * d..(s + 1) * d], &lay.g_down[s], t)?;
                cpu::add_(&mut x, &y);
            }
        }
        self.st.pos += 1;
        self.st.rope_next += 1;
        self.head(&x, t)
    }

    // ------------------------------------------------------------ token mixers

    /// Layer li's linear attention on its input projection y [L, q|k|v|z]
    /// and normed input xn [L, D] -> [L, value_dim]; its states advanced.
    fn deltanet(&mut self, li: usize, y: &[f32], xn: &[f32], l: usize) -> Vec<f32> {
        let c = &self.cfg;
        let lay = &self.layers[li];
        // the host parameters alone (the layer's device buffers are not Sync)
        let (w_ba, w_conv, a_neg, dt_bias, w_norm) =
            (&lay.ba[..], &lay.conv[..], &lay.a_neg[..], &lay.dt_bias[..], &lay.norm[..]);
        let (h, dk, dv, d) = (c.lin_heads, c.lin_hk, c.lin_hv, c.d);
        let (nl, cc, kk) = (c.n_lin(), c.lin_conv(), c.conv_k);
        // gates: b | a = xn W_ba^T, in f32
        let mut ba = vec![0f32; l * 2 * h];
        par_rows(&mut ba, 2 * h, |r0, piece| {
            for (ri, row) in piece.chunks_mut(2 * h).enumerate() {
                let x = &xn[(r0 + ri) * d..][..d];
                for (o, w) in row.iter_mut().zip(w_ba.chunks(d)) {
                    *o = cpu::dot(x, w);
                }
            }
        });
        // causal depthwise convolution + SiLU over q | k | v, the state
        // carrying the previous k-1 inputs
        let conv_state = &mut self.st.conv[li];
        let mut seq = Vec::with_capacity((kk - 1 + l) * cc);
        seq.extend_from_slice(conv_state);
        for r in 0..l {
            seq.extend_from_slice(&y[r * nl..][..cc]);
        }
        let mut qkv = vec![0f32; l * cc];
        par_rows(&mut qkv, cc, |r0, piece| {
            for (ri, row) in piece.chunks_mut(cc).enumerate() {
                let r = r0 + ri;
                for (ch, o) in row.iter_mut().enumerate() {
                    let mut s = 0f32;
                    for j in 0..kk {
                        s += w_conv[ch * kk + j] * seq[(r + j) * cc + ch];
                    }
                    *o = silu(s);
                }
            }
        });
        conv_state.copy_from_slice(&seq[l * cc..]);
        // per head: [state dk dv | outputs L dv], the heads in parallel
        let span = dk * dv + l * dv;
        let mut work = vec![0f32; h * span];
        let rec = &mut self.st.rec[li];
        for hh in 0..h {
            work[hh * span..][..dk * dv].copy_from_slice(&rec[hh * dk * dv..][..dk * dv]);
        }
        let scale = 1.0 / (dk as f32).sqrt();
        par_rows(&mut work, span, |h0, piece| {
            let (mut q, mut k, mut kv) = (vec![0f32; dk], vec![0f32; dk], vec![0f32; dv]);
            for (hi, wrow) in piece.chunks_mut(span).enumerate() {
                let hh = h0 + hi;
                let (s, out) = wrow.split_at_mut(dk * dv);
                for r in 0..l {
                    let row = &qkv[r * cc..][..cc];
                    q.copy_from_slice(&row[hh * dk..][..dk]);
                    k.copy_from_slice(&row[c.lin_key() + hh * dk..][..dk]);
                    let v = &row[2 * c.lin_key() + hh * dv..][..dv];
                    l2norm(&mut q, scale);
                    l2norm(&mut k, 1.0);
                    let b = ba[r * 2 * h + hh];
                    let a = ba[r * 2 * h + h + hh];
                    let beta = cpu::sigmoid(b);
                    let decay = (a_neg[hh] * softplus(a + dt_bias[hh])).exp();
                    // S *= decay; kv = S^T k
                    kv.fill(0.0);
                    for (i, srow) in s.chunks_mut(dv).enumerate() {
                        for (sv, kvv) in srow.iter_mut().zip(kv.iter_mut()) {
                            *sv *= decay;
                            *kvv += *sv * k[i];
                        }
                    }
                    // delta = (v - kv) beta; S += k delta^T; o = S^T q
                    for (kvv, &vv) in kv.iter_mut().zip(v) {
                        *kvv = (vv - *kvv) * beta;
                    }
                    let o = &mut out[r * dv..][..dv];
                    o.fill(0.0);
                    for (i, srow) in s.chunks_mut(dv).enumerate() {
                        for ((sv, &dl), ov) in srow.iter_mut().zip(kv.iter()).zip(o.iter_mut()) {
                            *sv += k[i] * dl;
                            *ov += *sv * q[i];
                        }
                    }
                }
            }
        });
        for hh in 0..h {
            rec[hh * dk * dv..][..dk * dv].copy_from_slice(&work[hh * span..][..dk * dv]);
        }
        // gated RMSNorm per head: norm(o) w * silu(z)
        let vd = c.lin_value();
        let mut out = vec![0f32; l * vd];
        for r in 0..l {
            let z = &y[r * nl + cc..][..vd];
            for hh in 0..h {
                let o = &work[hh * span + dk * dv + r * dv..][..dv];
                let ms = o.iter().map(|v| v * v).sum::<f32>() / dv as f32;
                let inv = 1.0 / (ms + c.eps).sqrt();
                for j in 0..dv {
                    out[r * vd + hh * dv + j] = o[j] * inv * w_norm[j] * silu(z[hh * dv + j]);
                }
            }
        }
        out
    }

    /// Layer li's gated softmax attention on its input projection y [L,
    /// (q | gate) a head | k | v] at cache rows pos0.. and M-RoPE positions
    /// `rope_pos` -> [L, H hd]; its K and V into the cache.
    fn attention(&mut self, li: usize, y: &[f32], l: usize, pos0: usize, rope_pos: &[[usize; 3]]) -> Vec<f32> {
        let c = &self.cfg;
        let lay = &self.layers[li];
        let (h, kvh, hd) = (c.heads, c.kv_heads, c.hd);
        let na = c.n_att();
        let kvw = kvh * hd;
        let (kc, vc) = (&mut self.st.k[li], &mut self.st.v[li]);
        let mut q = vec![0f32; l * h * hd];
        let mut gate = vec![0f32; l * h * hd];
        for r in 0..l {
            let row = &y[r * na..][..na];
            let pos = rope_pos[r].map(|p| p as f32);
            for hh in 0..h {
                let qn = rms_norm(&row[hh * 2 * hd..][..hd], &lay.q_norm, c.eps);
                let dst = &mut q[(r * h + hh) * hd..][..hd];
                dst.copy_from_slice(&qn);
                rope(dst, &self.rope_inv, &self.rope_axis, pos);
                gate[(r * h + hh) * hd..][..hd].copy_from_slice(&row[hh * 2 * hd + hd..][..hd]);
            }
            let p = pos0 + r;
            for g in 0..kvh {
                let kn = rms_norm(&row[2 * h * hd + g * hd..][..hd], &lay.k_norm, c.eps);
                let dst = &mut kc[p * kvw + g * hd..][..hd];
                dst.copy_from_slice(&kn);
                rope(dst, &self.rope_inv, &self.rope_axis, pos);
                vc[p * kvw + g * hd..][..hd].copy_from_slice(&row[(2 * h + kvh) * hd + g * hd..][..hd]);
            }
        }
        let (kc, vc) = (&*kc, &*vc);
        let scale = 1.0 / (hd as f32).sqrt();
        let group = h / kvh;
        let mut out = vec![0f32; l * h * hd];
        // one task a (query row, head)
        par_rows(&mut out, hd, |t0, piece| {
            let mut s = Vec::new();
            for (ti, o) in piece.chunks_mut(hd).enumerate() {
                let (r, hh) = ((t0 + ti) / h, (t0 + ti) % h);
                let g = hh / group;
                let n = pos0 + r + 1;
                let qh = &q[(r * h + hh) * hd..][..hd];
                s.clear();
                s.extend((0..n).map(|j| cpu::dot(qh, &kc[j * kvw + g * hd..][..hd]) * scale));
                cpu::softmax_(&mut s);
                o.fill(0.0);
                for (j, &p) in s.iter().enumerate() {
                    for (ov, &vv) in o.iter_mut().zip(&vc[j * kvw + g * hd..][..hd]) {
                        *ov += p * vv;
                    }
                }
                let gt = &gate[(r * h + hh) * hd..][..hd];
                for (ov, &gv) in o.iter_mut().zip(gt) {
                    *ov *= cpu::sigmoid(gv);
                }
            }
        });
        out
    }
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// torch's softplus (beta 1, threshold 20).
fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

/// x / sqrt(sum x^2 + 1e-6) * scale, in place.
fn l2norm(x: &mut [f32], scale: f32) {
    let inv = 1.0 / (x.iter().map(|v| v * v).sum::<f32>() + 1e-6).sqrt() * scale;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// Qwen3.5's RMSNorm (the weight stored as an offset from one).
fn rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (ms + eps).sqrt();
    x.iter().zip(w).map(|(v, w)| v * inv * (1.0 + w)).collect()
}

fn rms_rows(x: &[f32], d: usize, w: &[f32], eps: f32) -> Vec<f32> {
    let mut out = vec![0f32; x.len()];
    par_rows(&mut out, d, |r0, piece| {
        for (ri, o) in piece.chunks_mut(d).enumerate() {
            o.copy_from_slice(&rms_norm(&x[(r0 + ri) * d..][..d], w, eps));
        }
    });
    out
}

/// M-RoPE on the first 2 len(inv) dims of x (half-split): pair j at the
/// position of axis `axis[j]` (t, h, w).
fn rope(x: &mut [f32], inv: &[f32], axis: &[usize], pos: [f32; 3]) {
    let half = inv.len();
    for (j, &f) in inv.iter().enumerate() {
        let a = pos[axis[j]] * f;
        let (sn, cs) = a.sin_cos();
        let (x1, x2) = (x[j], x[j + half]);
        x[j] = x1 * cs - x2 * sn;
        x[j + half] = x2 * cs + x1 * sn;
    }
}

/// The input embeddings of `ids` as f32, decoded from the tied LM head's
/// packed rows (tensors `head<j>`, `head_rows` rows each, GEMVbfp16's
/// layout: per 8-row group its D / 8 blocks in k order, a block 8 rows x
/// (exponent byte, 8 int8 mantissas); a value is mantissa x 2^(exponent -
/// 133)). The model keeps no separate embedding table.
fn embed(s: &Store, cfg: &Config, format: &str, ids: &[u32]) -> Result<Vec<f32>, Error> {
    match format {
        "bfp16" => {}
        "bfp6s" => return embed_bfp6s(s, cfg, ids),
        f => return Err(Error::Bundle(format!("gemv_format {f}"))),
    }
    let d = cfg.d;
    let group = d / 8 * 72;
    let mut x = Vec::with_capacity(ids.len() * d);
    for &id in ids {
        let id = id as usize;
        if id >= cfg.vocab {
            return Err(Error::Input(format!("token {id} outside the vocabulary")));
        }
        let (j, r) = (id / cfg.head_rows, id % cfg.head_rows);
        let bytes = s.bytes(&format!("head{j}"))?;
        let row = &bytes[(r / 8) * group + (r % 8) * 9..];
        for b in 0..d / 8 {
            let blk = &row[b * 72..b * 72 + 9];
            let scale = (blk[0] as f32 - 133.0).exp2();
            x.extend(blk[1..].iter().map(|&m| m as i8 as f32 * scale));
        }
    }
    Ok(x)
}

/// `embed` from a bfp6s16 head (bfp6s.py; the GEMV kernels unpack it the
/// same way): per 8-row group its segments of 128 k, a segment its 8 tiles'
/// mantissa planes (96 bytes a tile: nib[j] = u0 & 15 | (u1 & 15) << 4 for
/// lane j = row * 8 + k, two[j % 32] holding the high 2 bits of u0 / u1 at
/// 0 / 4 for lanes < 32, 2 / 6 above) then their scale bytes (a tile's 8,
/// one a row: c = 5 + (b & 3), exponent (b >> 2) + 80). q = u - 32 and the
/// value is rne(q c / 2) x 2^(exponent - 133).
fn embed_bfp6s(s: &Store, cfg: &Config, ids: &[u32]) -> Result<Vec<f32>, Error> {
    const SEG: usize = 832;
    let d = cfg.d;
    let group = d / 128 * SEG;
    let mut x = Vec::with_capacity(ids.len() * d);
    for &id in ids {
        let id = id as usize;
        if id >= cfg.vocab {
            return Err(Error::Input(format!("token {id} outside the vocabulary")));
        }
        let (j, r) = (id / cfg.head_rows, id % cfg.head_rows);
        let bytes = s.bytes(&format!("head{j}"))?;
        let (g, n) = (r / 8, r % 8);
        for seg in bytes[g * group..(g + 1) * group].chunks_exact(SEG) {
            let (mant, scales) = seg.split_at(768);
            for (t, tile) in mant.chunks_exact(96).enumerate() {
                let b = scales[t * 8 + n];
                let c = 5 + (b & 3) as i32;
                let scale = (((b >> 2) as i32 + 80 - 133) as f32).exp2();
                for half in 0..2 {
                    for k in 0..8 {
                        let lane = n * 8 + k;
                        let nib = tile[lane];
                        let shift = if half == 0 { 0 } else { 4 } + if lane < 32 { 0 } else { 2 };
                        let lo = if half == 0 { nib & 15 } else { nib >> 4 };
                        let u = lo | ((tile[64 + lane % 32] >> shift) & 3) << 4;
                        let p = (u as i32 - 32) * c;
                        // rne(p / 2): p odd is a tie, to the even neighbour
                        let m = if p % 2 == 0 {
                            p / 2
                        } else {
                            let lo = (p - 1) / 2;
                            if lo % 2 == 0 { lo } else { lo + 1 }
                        };
                        x.push(m as f32 * scale);
                    }
                }
            }
        }
    }
    Ok(x)
}

/// GEMM `key` over the first n chunks: A from column `col` of `a`, into `c`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_gemm(
    npu: &Npu,
    key: &str,
    a: &Buffer,
    col: usize,
    w: &Buffer,
    c: &Buffer,
    n: usize,
    t: &mut Timing,
) -> Result<(), Error> {
    let g = npu.gemm(key)?;
    let (ae, ce) = match (g.a_elems.get(n - 1), g.c_elems.get(n - 1)) {
        (Some(&a), Some(&c)) => (a, c),
        _ => return Err(Error::Input(format!("{key}: {n} chunks (the bundle has streams for {})", g.a_elems.len()))),
    };
    let av = a.sub(col * BF16, ae * BF16)?;
    let cv = c.sub(0, ce * BF16)?;
    let t0 = Instant::now();
    npu.run(&gemm_key(key, n), &[&av, w, &cv])?;
    t.add(&format!("npu:{key}"), t0.elapsed());
    Ok(())
}

/// x as bf16 into the start of `b`, synced to the device.
pub(crate) fn write_rows(b: &mut Buffer, x: &[f32]) -> Result<(), Error> {
    let dst = &mut b.as_mut_slice::<u16>()[..x.len()];
    for (o, &v) in dst.iter_mut().zip(x) {
        *o = f32_to_bf16(v);
    }
    b.sub(0, x.len() * BF16)?.sync_to_device()?;
    Ok(())
}

/// `n` bf16 values of `b` from element `at`, synced from the device, as f32.
pub(crate) fn read(b: &Buffer, at: usize, n: usize) -> Result<Vec<f32>, Error> {
    b.sub(at * BF16, n * BF16)?.sync_from_device()?;
    Ok(b.as_slice::<u16>()[at..at + n].iter().map(|&v| bf16_to_f32(v)).collect())
}
