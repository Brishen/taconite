// SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
// SPDX-License-Identifier: Apache-2.0

//! The DETR encoder (NPU, as `iron/applications/sam3/detr_npu.py`), the
//! DETR decoder (host, as HF `Sam3DetrDecoder`, with all six layers' vision
//! keys and values from one NPU GEMM) and the dot-product scoring.
//!
//! Encoder, per layer: `[q|k|v] = [LN1(x) + pos | LN1(x)] B` (heads of 32
//! zero-padded to the MHA's 64, q pre-scaled by sqrt 2), MHA, o; the prompt
//! cross-attention folded into two GEMMs (`h (W_q,h K_h^T)` for the scores,
//! `p (V_h W_o,h)` for the output; B matrices packed here per prompt) with
//! the softmax here; the ReLU MLP (fc2 reading fc1's output in place).

use taconite::bf16_to_f32;

use crate::cpu::{self, Attn, Rows, W, ln_row, par_rows, sigmoid};
use crate::npu::{pull, push};
use crate::pack::pack_b;
use crate::text::Text;
use crate::vit::{add_bf16, layer_norm_bf16};
use crate::{Error, Sam3, gemm, mha};

const EPS: f32 = 1e-5; // nn.LayerNorm's default, every LayerNorm here

/// The decoder's outputs (and per-layer values for checking).
pub struct Decoded {
    /// `[Q, 256]` the last layer's normalised query states.
    pub hidden: Vec<f32>,
    /// `[Q, 4]` the final boxes, normalised xyxy.
    pub boxes: Vec<f32>,
    /// `[Q]` classification logits.
    pub logits: Vec<f32>,
    pub presence: f32,
    /// `[layers][Q, 4]` cxcywh boxes after each layer.
    pub layer_boxes: Vec<Vec<f32>>,
    pub layer_presence: Vec<f32>,
}

fn inverse_sigmoid(x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    (x.max(1e-3) / (1.0 - x).max(1e-3)).ln()
}

fn xyxy(b: &[f32]) -> [f32; 4] {
    [b[0] - 0.5 * b[2], b[1] - 0.5 * b[3], b[0] + 0.5 * b[2], b[1] + 0.5 * b[3]]
}

impl Sam3 {
    pub(crate) fn lin(&self, x: &[f32], n_in: usize, name: &str) -> Result<Vec<f32>, Error> {
        let st = &self.store;
        let b = if st.has(&format!("{name}.b")) { Some(st.f32(&format!("{name}.b"))?) } else { None };
        Ok(cpu::linear(x, n_in, W::F32(st.f32(&format!("{name}.w"))?), b))
    }

    fn ln(&self, x: &[f32], dim: usize, name: &str) -> Result<Vec<f32>, Error> {
        let st = &self.store;
        Ok(cpu::layer_norm(x, dim, st.f32(&format!("{name}.w"))?, st.f32(&format!("{name}.b"))?, EPS))
    }

    /// `relu(l1) -> [relu(l2) ->] l_last`, HF `Sam3DecoderMLP`.
    fn mlp(&self, x: &[f32], n_in: usize, name: &str, layers: usize) -> Result<Vec<f32>, Error> {
        let mut h = self.lin(x, n_in, &format!("{name}.1"))?;
        cpu::relu_(&mut h);
        let d = h.len() / (x.len() / n_in);
        let mut h = self.lin(&h, d, &format!("{name}.2"))?;
        if layers == 3 {
            cpu::relu_(&mut h);
            h = self.lin(&h, d, &format!("{name}.3"))?;
        }
        Ok(h)
    }

    /// The cross-attention `prefix` (`d.<i>.ca` or `m.ca`) folded over this
    /// prompt and packed into slots `<slot>.s` / `<slot>.c`: the scores
    /// `h Bs + bs` (`N` = heads x prompt tokens) and the output `p Bc + b_o`.
    pub(crate) fn fold_cross(&mut self, prefix: &str, slot: &str, text: &Text) -> Result<(), Error> {
        let c = &self.cfg;
        let (d, nh, l) = (c.d_model, c.d_heads, c.text_len);
        let hd = d / nh;
        let st = &self.store;
        let k = self.lin(&text.feats, d, &format!("{prefix}.k"))?; // [L, D]
        let v = self.lin(&text.feats, d, &format!("{prefix}.v"))?;
        let wq = st.f32(&format!("{prefix}.q.w"))?; // [D, D] out x in
        let bq = st.f32(&format!("{prefix}.q.b"))?;
        let wo = st.f32(&format!("{prefix}.o.w"))?;
        let bo = st.f32(&format!("{prefix}.o.b"))?;
        let inv = 1.0 / (hd as f32).sqrt();
        let mut bs_m = vec![0f32; d * nh * l];
        let mut bs_b = vec![0f32; nh * l];
        let mut bc_m = vec![0f32; nh * l * d];
        par_rows(&mut bs_m, nh * l, |i0, piece| {
            for (ii, row) in piece.chunks_mut(nh * l).enumerate() {
                let i = i0 + ii; // input feature
                for h in 0..nh {
                    for j in 0..l {
                        let mut s = 0.0;
                        for e in 0..hd {
                            s += wq[(h * hd + e) * d + i] * k[j * d + h * hd + e];
                        }
                        row[h * l + j] = s * inv;
                    }
                }
            }
        });
        for h in 0..nh {
            for j in 0..l {
                bs_b[h * l + j] = (0..hd).map(|e| bq[h * hd + e] * k[j * d + h * hd + e]).sum::<f32>() * inv;
            }
        }
        par_rows(&mut bc_m, d, |r0, piece| {
            for (ri, row) in piece.chunks_mut(d).enumerate() {
                let r = r0 + ri;
                let (h, j) = (r / l, r % l);
                for (o, out) in row.iter_mut().enumerate() {
                    *out = (0..hd).map(|e| v[j * d + h * hd + e] * wo[o * d + h * hd + e]).sum();
                }
            }
        });
        let ps = pack_b(self.npu.spec("d_s")?, &bs_m, Some(&bs_b));
        let pc = pack_b(self.npu.spec("d_c")?, &bc_m, Some(bo));
        self.slots.get_mut(&format!("{slot}.s")).unwrap().write(&ps)?;
        self.slots.get_mut(&format!("{slot}.c")).unwrap().write(&pc)?;
        Ok(())
    }

    /// The folded prompt cross-attention on `x [T, D]` (already normalised
    /// as `h`): `x_res += softmax(h Bs + bs) Bc + b_o`, per head over the
    /// valid prompt tokens. Uses the d_s / d_c kernels and `slot`'s weights.
    pub(crate) fn cross_npu(&mut self, h: &[f32], x_res: &mut [f32], slot: &str, text: &Text) -> Result<(), Error> {
        let c = &self.cfg;
        let (d, nh, l) = (c.d_model, c.d_heads, c.text_len);
        let t = h.len() / d;
        let mut hb = vec![0u16; h.len()];
        crate::narrow(h, &mut hb);
        self.io.d_s.set_a(&hb)?;
        gemm(&mut self.npu, &self.io.d_s, &self.slots[&format!("{slot}.s")], &mut self.timing)?;
        let s = self.io.d_s.get_c(t)?;
        let valid = &text.valid;
        let mut pb = vec![0u16; t * nh * l];
        par_rows(&mut pb, nh * l, |r0, piece| {
            let mut row_f = vec![0f32; l];
            for (ri, row) in piece.chunks_mut(nh * l).enumerate() {
                let src = &s[(r0 + ri) * nh * l..][..nh * l];
                for hh in 0..nh {
                    for j in 0..l {
                        row_f[j] = if valid[j] { bf16_to_f32(src[hh * l + j]) } else { f32::NEG_INFINITY };
                    }
                    cpu::softmax_(&mut row_f);
                    for j in 0..l {
                        row[hh * l + j] = taconite::f32_to_bf16(row_f[j]);
                    }
                }
            }
        });
        self.io.d_c.set_a(&pb)?;
        gemm(&mut self.npu, &self.io.d_c, &self.slots[&format!("{slot}.c")], &mut self.timing)?;
        add_bf16(x_res, &self.io.d_c.get_c(t)?, d, None);
        Ok(())
    }

    /// The DETR encoder: `fpn2 [T, 256]` (the 72 x 72 level, raster) and
    /// the prompt -> `[T, 256]`.
    pub fn detr_encoder(&mut self, fpn2: &[f32], text: &Text) -> Result<Vec<f32>, Error> {
        let c = self.cfg.clone();
        let (d, nh, t) = (c.d_model, c.d_heads, c.tokens());
        let pd = self.npu.mhas["mha_d"].d * nh;
        let mut x = fpn2.to_vec();
        for i in 0..c.d_layers {
            let p = |n: &str| format!("d.{i}.{n}");
            self.fold_cross(&p("ca"), &format!("d.{i}"), text)?;
            let st = &self.store;
            let pos = st.f32("d.pos")?;
            let (w1, b1) = (st.f32(&p("ln1.w"))?, st.f32(&p("ln1.b"))?);
            let mut ab = vec![0u16; t * 2 * d];
            par_rows(&mut ab, 2 * d, |r0, piece| {
                let mut h = vec![0f32; d];
                for (ri, row) in piece.chunks_mut(2 * d).enumerate() {
                    let r = r0 + ri;
                    ln_row(&x[r * d..(r + 1) * d], &mut h, w1, b1, EPS);
                    for j in 0..d {
                        row[j] = taconite::f32_to_bf16(h[j] + pos[r * d + j]);
                        row[d + j] = taconite::f32_to_bf16(h[j]);
                    }
                }
            });
            self.io.d_qkv.set_a(&ab)?;
            gemm(&mut self.npu, &self.io.d_qkv, &self.w[&p("qkv")], &mut self.timing)?;
            let qkv = self.io.d_qkv.get_c(t)?;
            let m = &mut self.io.mha_d;
            let mut part = vec![0u16; t * pd];
            for (j, buf) in [&mut m.q, &mut m.k, &mut m.v].into_iter().enumerate() {
                par_rows(&mut part, pd, |r0, piece| {
                    for (ri, row) in piece.chunks_mut(pd).enumerate() {
                        row.copy_from_slice(&qkv[(r0 + ri) * 3 * pd + j * pd..][..pd]);
                    }
                });
                push(&part, buf)?;
            }
            mha(&mut self.npu, &self.io.mha_d, &mut self.timing)?;
            let o = pull(&self.io.mha_d.o, t * pd)?;
            self.io.d_o.set_a(&o)?;
            gemm(&mut self.npu, &self.io.d_o, &self.w[&p("o")], &mut self.timing)?;
            add_bf16(&mut x, &self.io.d_o.get_c(t)?, d, None);

            let h = self.ln(&x, d, &p("ln2"))?;
            self.cross_npu(&h, &mut x, &format!("d.{i}"), text)?;

            let mut hb = vec![0u16; t * d];
            {
                let st = &self.store;
                layer_norm_bf16(&x, d, st.f32(&p("ln3.w"))?, st.f32(&p("ln3.b"))?, EPS, &mut hb);
            }
            self.io.d_fc1.set_a(&hb)?;
            gemm(&mut self.npu, &self.io.d_fc1, &self.w[&p("fc1")], &mut self.timing)?;
            gemm(&mut self.npu, &self.io.d_fc2, &self.w[&p("fc2")], &mut self.timing)?;
            add_bf16(&mut x, &self.io.d_fc2.get_c(t)?, d, Some(self.store.f32(&p("fc2.b"))?));
        }
        Ok(x)
    }

    /// Box sine embedding (HF `Sam3SinePositionEmbedding.encode_boxes`):
    /// `[Q, 4]` cxcywh -> `[Q, 4 * 128]`, ordered (y, x, w, h).
    fn encode_boxes(&self, boxes: &[f32]) -> Vec<f32> {
        let f = self.cfg.d_model / 2;
        let dim_t: Vec<f32> = (0..f).map(|i| 10000f32.powf(2.0 * (i / 2) as f32 / f as f32)).collect();
        let scale = 2.0 * std::f32::consts::PI;
        let mut out = Vec::with_capacity(boxes.len() / 4 * 4 * f);
        for b in boxes.chunks(4) {
            for coord in [b[1], b[0], b[2], b[3]] {
                for (i, &d) in dim_t.iter().enumerate().take(f) {
                    let v = coord * scale / d;
                    out.push(if i % 2 == 0 { v.sin() } else { v.cos() });
                }
            }
        }
        out
    }

    /// Box relative position bias in its separable form, `(by, bx)`, each
    /// `[H, 1 + Q, g]` (row 0, the presence token's, zero): the bias of
    /// query `q` at key `(iy, ix)` is `by[h, q, iy] + bx[h, q, ix]` -- for
    /// each query box and grid row/column, the log-scaled distances to the
    /// box's edges through a small MLP per axis. The `[H, 1 + Q, T]` sum is
    /// never built; `cpu::attention_dec` adds the two terms on the way.
    fn rpb(&self, boxes: &[f32]) -> Result<(Vec<f32>, Vec<f32>), Error> {
        let c = &self.cfg;
        let (g, nh) = (c.grid, c.d_heads);
        let q = boxes.len() / 4;
        let enc = |v: f32| {
            let v = v * 8.0;
            v.signum() * (v.abs() + 1.0).log2() / 3.0
        };
        // per axis: inputs [Q * g, 2] -> [Q * g, H]
        let axis = |lo: usize, name: &str| -> Result<Vec<f32>, Error> {
            let mut inp = Vec::with_capacity(q * g * 2);
            for b in boxes.chunks(4) {
                let e = xyxy(b);
                for i in 0..g {
                    let p = i as f32 / g as f32;
                    inp.push(enc(p - e[lo]));
                    inp.push(enc(p - e[lo + 2]));
                }
            }
            self.mlp(&inp, 2, name, 2)
        };
        let ry = axis(1, "dec.rpb_y")?;
        let rx = axis(0, "dec.rpb_x")?;
        let lq = q + 1;
        let (mut by, mut bx) = (vec![0f32; nh * lq * g], vec![0f32; nh * lq * g]);
        for h in 0..nh {
            for qq in 0..q {
                let dst = (h * lq + qq + 1) * g;
                for i in 0..g {
                    by[dst + i] = ry[(qq * g + i) * nh + h];
                    bx[dst + i] = rx[(qq * g + i) * nh + h];
                }
            }
        }
        Ok((by, bx))
    }

    /// The DETR decoder + scoring: `enc [T, 256]` and the prompt ->
    /// [`Decoded`].
    pub fn detr_decoder(&mut self, enc: &[f32], text: &Text) -> Result<Decoded, Error> {
        let c = self.cfg.clone();
        let (d, nh, t) = (c.d_model, c.d_heads, c.tokens());
        let kvw = self.npu.spec("dec_kv")?.n;
        // every layer's vision keys (from enc + pos) and values (from enc)
        {
            let pos = self.store.f32("d.pos")?;
            let mut ab = vec![0u16; t * 2 * d];
            par_rows(&mut ab, 2 * d, |r0, piece| {
                for (ri, row) in piece.chunks_mut(2 * d).enumerate() {
                    let r = r0 + ri;
                    for j in 0..d {
                        row[j] = taconite::f32_to_bf16(enc[r * d + j] + pos[r * d + j]);
                        row[d + j] = taconite::f32_to_bf16(enc[r * d + j]);
                    }
                }
            });
            self.io.dec_kv.set_a(&ab)?;
        }
        gemm(&mut self.npu, &self.io.dec_kv, &self.w["dec.kv"], &mut self.timing)?;
        let kv = self.io.dec_kv.get_c(t)?; // [T, layers x (K | V)] bf16
        let hd = d / nh;
        // this layer's keys transposed, [H, hd, T], and values, [H, T, hd]
        let (mut kt, mut vh) = (vec![0f32; t * d], vec![0f32; t * d]);

        let st = &self.store;
        let mut refb: Vec<f32> = st.f32("dec.reference_points")?.iter().map(|&v| sigmoid(v)).collect();
        let mut hs = st.f32("dec.presence_token")?.to_vec();
        hs.extend_from_slice(st.f32("dec.query_embed")?);
        let mut out = Decoded {
            hidden: vec![],
            boxes: vec![],
            logits: vec![],
            presence: 0.0,
            layer_boxes: vec![],
            layer_presence: vec![],
        };
        let mut normed = vec![];
        for l in 0..c.dec_layers {
            let p = |s: &str| format!("dec.{l}.{s}");
            let t0 = std::time::Instant::now();
            let sine = self.encode_boxes(&refb);
            let qpos_q = self.mlp(&sine, 2 * d, "dec.ref_point_head", 2)?;
            let mut qpos = vec![0f32; d];
            qpos.extend_from_slice(&qpos_q);
            let with_pos = |hs: &[f32]| -> Vec<f32> { hs.iter().zip(&qpos).map(|(a, b)| a + b).collect() };
            self.timing.add("dec_qpos", t0.elapsed());
            let t0 = std::time::Instant::now();
            let (by, bx) = self.rpb(&refb)?;
            self.timing.add("dec_rpb", t0.elapsed());
            let t0 = std::time::Instant::now();

            // self-attention (q, k with the query positions)
            let qk = with_pos(&hs);
            let q = self.lin(&qk, d, &p("sa.q"))?;
            let k = self.lin(&qk, d, &p("sa.k"))?;
            let v = self.lin(&hs, d, &p("sa.v"))?;
            let a =
                cpu::attention(&q, d, Rows::f32(&k, d), Rows::f32(&v, d), &Attn { heads: nh, ..Default::default() });
            let mut o = self.lin(&a, d, &p("sa.o"))?;
            cpu::add_(&mut o, &hs);
            hs = self.ln(&o, d, &p("sa_ln"))?;
            self.timing.add("dec_sa", t0.elapsed());
            let t0 = std::time::Instant::now();

            // text cross-attention
            let q = self.lin(&with_pos(&hs), d, &p("tca.q"))?;
            let k = self.lin(&text.feats, d, &p("tca.k"))?;
            let v = self.lin(&text.feats, d, &p("tca.v"))?;
            let at = Attn { heads: nh, valid: Some(&text.valid), ..Default::default() };
            let a = cpu::attention(&q, d, Rows::f32(&k, d), Rows::f32(&v, d), &at);
            let mut o = self.lin(&a, d, &p("tca.o"))?;
            cpu::add_(&mut o, &hs);
            hs = self.ln(&o, d, &p("tca_ln"))?;
            self.timing.add("dec_tca", t0.elapsed());

            // vision cross-attention, with the box bias
            let q = self.lin(&with_pos(&hs), d, &p("vca.q"))?;
            let t0 = std::time::Instant::now();
            let (koff, voff) = (l * 2 * d, l * 2 * d + d);
            par_rows(&mut vh, hd, |r0, piece| {
                for (ri, row) in piece.chunks_mut(hd).enumerate() {
                    let (h, j) = ((r0 + ri) / t, (r0 + ri) % t);
                    for (o, &x) in row.iter_mut().zip(&kv[j * kvw + voff + h * hd..][..hd]) {
                        *o = bf16_to_f32(x);
                    }
                }
            });
            // the transpose in blocks of 8 key dimensions: one strided read
            // of 8 bf16 per token, 8 sequential write streams
            const KB: usize = 8;
            par_rows(&mut kt, KB * t, |b0, piece| {
                for (bi, blk) in piece.chunks_mut(KB * t).enumerate() {
                    let (h, d0) = ((b0 + bi) / (hd / KB), ((b0 + bi) % (hd / KB)) * KB);
                    for j in 0..t {
                        let src = &kv[j * kvw + koff + h * hd + d0..][..KB];
                        for (dd, &x) in src.iter().enumerate() {
                            blk[dd * t + j] = bf16_to_f32(x);
                        }
                    }
                }
            });
            self.timing.add("dec_kvconv", t0.elapsed());
            let t0 = std::time::Instant::now();
            let a = cpu::attention_dec(&q, d, nh, &kt, &vh, &by, &bx, c.grid);
            self.timing.add("dec_vattn", t0.elapsed());
            let t0 = std::time::Instant::now();
            let mut o = self.lin(&a, d, &p("vca.o"))?;
            cpu::add_(&mut o, &hs);
            hs = self.ln(&o, d, &p("vca_ln"))?;
            self.timing.add("dec_vo", t0.elapsed());
            let t0 = std::time::Instant::now();

            // MLP (post-norm)
            let mut f = self.lin(&hs, d, &p("fc1"))?;
            cpu::relu_(&mut f);
            let mut f = self.lin(&f, c.d_ffn, &p("fc2"))?;
            cpu::add_(&mut f, &hs);
            hs = self.ln(&f, d, &p("mlp_ln"))?;
            self.timing.add("dec_mlp", t0.elapsed());
            let t0 = std::time::Instant::now();

            // box refinement on the queries, presence from token 0
            normed = self.ln(&hs[d..], d, "dec.out_ln")?;
            let delta = self.mlp(&normed, d, "dec.box_head", 3)?;
            for (b, dl) in refb.iter_mut().zip(&delta) {
                *b = sigmoid(dl + inverse_sigmoid(*b));
            }
            let pres = self.ln(&hs[..d], d, "dec.presence_ln")?;
            let pl = self.mlp(&pres, d, "dec.presence_head", 3)?[0].clamp(-10.0, 10.0);
            out.layer_boxes.push(refb.clone());
            out.layer_presence.push(pl);
            self.timing.add("dec_heads", t0.elapsed());
        }
        out.presence = *out.layer_presence.last().unwrap();
        out.boxes = refb.chunks(4).flat_map(xyxy).collect();
        out.logits = self.score(&normed, text)?;
        out.hidden = normed;
        Ok(out)
    }

    /// HF `Sam3DotProductScoring`: queries against the mean-pooled,
    /// MLP-refined prompt.
    fn score(&self, hidden: &[f32], text: &Text) -> Result<Vec<f32>, Error> {
        let d = self.cfg.d_model;
        let l = self.cfg.text_len;
        let mut t = self.mlp(&text.feats, d, "score.text_mlp", 2)?;
        cpu::add_(&mut t, &text.feats);
        let t = self.ln(&t, d, "score.text_ln")?;
        let nv = text.valid.iter().filter(|&&v| v).count().max(1) as f32;
        let mut pooled = vec![0f32; d];
        for j in 0..l {
            if text.valid[j] {
                cpu::add_(&mut pooled, &t[j * d..(j + 1) * d]);
            }
        }
        pooled.iter_mut().for_each(|v| *v /= nv);
        let pt = self.lin(&pooled, d, "score.text_proj")?;
        let pq = self.lin(hidden, d, "score.query_proj")?;
        let scale = 1.0 / (d as f32).sqrt();
        Ok(pq.chunks(d).map(|q| (cpu::dot(q, &pt) * scale).clamp(-12.0, 12.0)).collect())
    }
}
