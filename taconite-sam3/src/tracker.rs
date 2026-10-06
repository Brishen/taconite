// SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
// SPDX-License-Identifier: Apache-2.0

//! Point and box prompts: SAM3's tracker head (HF `Sam3TrackerModel`), as
//! `iron/applications/sam3/tracker_npu.py`.
//!
//! | stage | NPU | host (here) |
//! |---|---|---|
//! | ViT backbone | as the text path ([`Sam3::vit`]) | |
//! | the tracker's FPN neck | the text path's neck kernels with the tracker's weights (`tn.*`); the mask decoder's `conv_s0` / `conv_s1` 1x1s folded into levels 0 / 1's 3x3s, `no_memory_embedding` into level 2's bias | GELU, pixel shuffles |
//! | prompt encoder, two-way transformer (2 layers over the 72 x 72 tokens), upscaling, hypernetwork masks | | all of it, per prompt |
//!
//! [`Sam3::embed_image`] runs the image side once (~2 s, the NPU);
//! [`Sam3::predict_points`] then answers any number of prompts on the
//! host (tens of ms each) and needs only `&self`, so it never waits for
//! the NPU.

use std::f32::consts::PI;

use crate::bundle::Store;
use crate::cpu::{self, Attn, Rows, W, gelu, par_rows, sigmoid};
use crate::detr::lin;
use crate::{Error, Sam3};

/// The image side of a point prompt, channels-last f32: what
/// [`Sam3::predict_points`] reads.
#[derive(Clone)]
pub struct ImageEmbedding {
    /// `conv_s0(fpn0)`, `[(4g)^2, 32]`
    pub s0: Vec<f32>,
    /// `conv_s1(fpn1)`, `[(2g)^2, 64]`
    pub s1: Vec<f32>,
    /// `fpn2 + no_memory_embedding`, `[g^2, 256]`
    pub emb: Vec<f32>,
}

/// One object's prompt, in the image's own pixel coordinates.
#[derive(Clone, Debug, Default)]
pub struct PointPrompt {
    /// `(x, y)` clicks
    pub points: Vec<[f32; 2]>,
    /// per point: 1 on the object, 0 background
    pub labels: Vec<i32>,
    /// `[x1, y1, x2, y2]`
    pub bbox: Option<[f32; 4]>,
}

impl PointPrompt {
    /// `"x,y[,label];..."` (label 1 if omitted) and an optional
    /// `"x1,y1,x2,y2"` box, as `sam3_reference.py --points / --box`.
    pub fn parse(points: &str, bbox: Option<&str>) -> Result<Self, Error> {
        let nums = |s: &str| -> Result<Vec<f32>, Error> {
            s.split(',').map(|t| t.trim().parse::<f32>().map_err(|e| Error::Input(format!("{s:?}: {e}")))).collect()
        };
        let mut p = PointPrompt::default();
        for spec in points.split(';').filter(|s| !s.trim().is_empty()) {
            let v = nums(spec)?;
            if !(2..=3).contains(&v.len()) {
                return Err(Error::Input(format!("point {spec:?}: expected x,y[,label]")));
            }
            p.points.push([v[0], v[1]]);
            p.labels.push(if v.len() == 3 { v[2] as i32 } else { 1 });
        }
        if let Some(b) = bbox.filter(|b| !b.trim().is_empty()) {
            let v = nums(b)?;
            if v.len() != 4 {
                return Err(Error::Input(format!("box {b:?}: expected x1,y1,x2,y2")));
            }
            p.bbox = Some([v[0], v[1], v[2], v[3]]);
        }
        Ok(p)
    }

    /// The bundle's `pcase` form: `points=<...> box=<...>`.
    pub fn parse_case(s: &str) -> Result<Self, Error> {
        let field = |k: &str| s.split_whitespace().find_map(|w| w.strip_prefix(k)).unwrap_or("");
        Self::parse(field("points="), Some(field("box=")))
    }
}

/// A prompt's masks, before post-processing.
pub struct PointOutput {
    /// `[n, S, S]` mask logits, `S` = `mask_size` (288)
    pub masks: Vec<f32>,
    /// `[n]` predicted mask IoUs
    pub iou: Vec<f32>,
    /// the object-present logit (> 0: an object at the prompt)
    pub object_score: f32,
    pub n: usize,
    pub size: usize,
}

impl PointOutput {
    /// The mask with the highest predicted IoU.
    pub fn best(&self) -> usize {
        (0..self.n).max_by(|&a, &b| self.iou[a].total_cmp(&self.iou[b])).unwrap_or(0)
    }

    /// Mask `i`'s logits `[S, S]`.
    pub fn logits(&self, i: usize) -> &[f32] {
        let px = self.size * self.size;
        &self.masks[i * px..(i + 1) * px]
    }
}

/// Mask `i` of `out` at the image's size: the logits upsampled bilinearly
/// (`align_corners=False`) and `> 0`, as HF's `post_process_masks`.
pub fn point_mask(out: &PointOutput, i: usize, w: usize, h: usize) -> Vec<bool> {
    crate::post::upsample_threshold(out.logits(i), out.size, w, h, 0.0)
}

/// The decoder's weights and shapes (from the manifest).
struct Dims {
    c: usize,
    heads: usize,
    layers: usize,
    mask_tokens: usize,
    out_ch: Vec<usize>,
}

impl Sam3 {
    /// True if the bundle carries the point prompt path.
    pub fn has_points(&self) -> bool {
        self.manifest.param("tracker").is_ok_and(|v| v == "1")
    }

    fn need_points(&self) -> Result<(), Error> {
        if self.has_points() {
            Ok(())
        } else {
            Err(Error::Bundle("this bundle has no point prompt path (re-export it with export_sam3.py)".into()))
        }
    }

    fn dims(&self) -> Result<Dims, Error> {
        let m = &self.manifest;
        Ok(Dims {
            c: self.cfg.d_model,
            heads: m.usize("trk_heads")?,
            layers: m.usize("trk_layers")?,
            mask_tokens: m.usize("trk_mask_tokens")?,
            out_ch: m.list("trk_out_ch")?,
        })
    }

    /// The image side of point prompts, from the preprocessed
    /// `[3, 1008, 1008]` image ([`preprocess`](crate::preprocess)): the
    /// ViT and the tracker's neck, on the NPU.
    pub fn embed_image(&mut self, pixels: &[f32]) -> Result<ImageEmbedding, Error> {
        self.need_points()?;
        let vit = self.time("vit", |s| s.vit(pixels))?;
        self.time("tracker_neck", |s| s.tracker_neck(&vit))
    }

    /// The backbone's `[T, 1024]` -> the tracker's [`ImageEmbedding`].
    pub fn tracker_neck(&mut self, vit: &[f32]) -> Result<ImageEmbedding, Error> {
        self.need_points()?;
        let d = self.dims()?;
        let [s0, s1, emb] = self.neck_host(vit, "tn")?;
        // each level is the 3x3 kernel's width; keep the folded 1x1's channels
        let keep = |l: Vec<f32>, oc: usize| -> Vec<f32> {
            if oc == d.c {
                return l;
            }
            l.chunks(d.c).flat_map(|r| r[..oc].iter().copied()).collect()
        };
        Ok(ImageEmbedding { s0: keep(s0, d.out_ch[0]), s1: keep(s1, d.out_ch[1]), emb: keep(emb, d.out_ch[2]) })
    }

    /// Masks for one object's prompt on an embedded `w x h` image:
    /// `multimask` gives SAM's three candidates, otherwise one mask (the
    /// single-mask output, or the best candidate when it is unstable).
    pub fn predict_points(
        &self,
        emb: &ImageEmbedding,
        prompt: &PointPrompt,
        w: usize,
        h: usize,
        multimask: bool,
    ) -> Result<PointOutput, Error> {
        self.need_points()?;
        if prompt.points.len() != prompt.labels.len() {
            return Err(Error::Input("one label per point".into()));
        }
        // the processor's coordinates: scaled to the model's square input
        let s = self.cfg.image_size as f32;
        let (sx, sy) = (s / w as f32, s / h as f32);
        let pts: Vec<[f32; 2]> = prompt.points.iter().map(|p| [p[0] * sx, p[1] * sy]).collect();
        let bbox = prompt.bbox.map(|b| [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy]);
        let sparse = self.prompt_tokens(&pts, &prompt.labels, bbox)?;
        self.decode_points(emb, &sparse, multimask)
    }

    /// The prompt encoder: points (model-input pixels, `image_size`
    /// square) with labels (1 object, 0 background, -1 not a point, -10
    /// padding) and an optional box -> the sparse tokens `[n, 256]`.
    pub fn prompt_tokens(
        &self,
        points: &[[f32; 2]],
        labels: &[i32],
        bbox: Option<[f32; 4]>,
    ) -> Result<Vec<f32>, Error> {
        let st = &self.store;
        let c = self.cfg.d_model;
        let gauss = st.f32("trk.pe_gauss")?; // [2, c/2]
        let pe_emb = st.f32("trk.point_embed")?; // [4, c]
        let not_a_point = st.f32("trk.not_a_point")?;
        let s = self.cfg.image_size as f32;
        // SAM's random-Fourier encoding of a pixel centre
        let pe = |x: f32, y: f32| -> Vec<f32> {
            let (cx, cy) = (2.0 * (x + 0.5) / s - 1.0, 2.0 * (y + 0.5) / s - 1.0);
            let half = c / 2;
            let mut out = vec![0f32; c];
            for j in 0..half {
                let v = 2.0 * PI * (cx * gauss[j] + cy * gauss[half + j]);
                out[j] = v.sin();
                out[half + j] = v.cos();
            }
            out
        };
        let embed = |p: [f32; 2], label: i32| -> Vec<f32> {
            match label {
                -10 => vec![0.0; c],
                l if l < 0 => not_a_point.to_vec(),
                l => {
                    let mut e = pe(p[0], p[1]);
                    for (v, &a) in e.iter_mut().zip(&pe_emb[l as usize * c..(l as usize + 1) * c]) {
                        *v += a;
                    }
                    e
                }
            }
        };
        let mut tokens = Vec::new();
        // no prompt at all: one "not a point" (as HF's forward)
        let (points, labels): (Vec<[f32; 2]>, Vec<i32>) = if points.is_empty() && bbox.is_none() {
            (vec![[0.0, 0.0]], vec![-1])
        } else {
            (points.to_vec(), labels.to_vec())
        };
        if labels.iter().any(|&l| l > 3 || (l < -1 && l != -10)) {
            return Err(Error::Input(format!("point labels must be 0 or 1, got {labels:?}")));
        }
        for (&p, &l) in points.iter().zip(&labels) {
            tokens.extend(embed(p, l));
        }
        match bbox {
            // the points are padded with a "not a point" when there is no box
            None => tokens.extend_from_slice(not_a_point),
            Some(b) => {
                tokens.extend(embed([b[0], b[1]], 2));
                tokens.extend(embed([b[2], b[3]], 3));
                tokens.extend_from_slice(not_a_point);
            }
        }
        Ok(tokens)
    }

    /// The mask decoder on an embedded image and sparse prompt tokens.
    pub fn decode_points(&self, emb: &ImageEmbedding, sparse: &[f32], multimask: bool) -> Result<PointOutput, Error> {
        let st = &self.store;
        let d = self.dims()?;
        let (c, g) = (d.c, self.cfg.grid);
        let t = g * g;
        if emb.emb.len() != t * c {
            return Err(Error::Input("image embedding of the wrong size".into()));
        }
        // queries: [object score, IoU, mask tokens..., prompt tokens]; the
        // same tokens are every layer's query positional encoding
        let mut queries = st.f32("trk.tokens")?.to_vec();
        queries.extend_from_slice(sparse);
        let qpe = queries.clone();
        // keys: the image plus the dense prompt (no mask: a constant)
        let no_mask = st.f32("trk.no_mask")?;
        let mut keys = emb.emb.clone();
        for row in keys.chunks_mut(c) {
            cpu::add_(row, no_mask);
        }
        let kpe = st.f32("trk.image_pe")?;
        let ln = |x: &[f32], name: &str, eps: f32| -> Result<Vec<f32>, Error> {
            Ok(cpu::layer_norm(x, c, st.f32(&format!("{name}.w"))?, st.f32(&format!("{name}.b"))?, eps))
        };
        let eps = 1e-5; // nn.LayerNorm's default
        for i in 0..d.layers {
            let p = format!("trk.{i}");
            if i == 0 {
                // the first layer's self-attention replaces the queries
                queries = attention(st, &format!("{p}.sa"), &queries, &queries, &queries, c, d.heads)?;
            } else {
                let q = plus(&queries, &qpe);
                let a = attention(st, &format!("{p}.sa"), &q, &q, &queries, c, d.heads)?;
                cpu::add_(&mut queries, &a);
            }
            queries = ln(&queries, &format!("{p}.ln1"), eps)?;
            let q = plus(&queries, &qpe);
            let k = plus(&keys, kpe);
            let a = attention(st, &format!("{p}.t2i"), &q, &k, &keys, c, d.heads)?;
            cpu::add_(&mut queries, &a);
            queries = ln(&queries, &format!("{p}.ln2"), eps)?;
            let m = self.trk_mlp(&format!("{p}.mlp"), &queries, c)?;
            cpu::add_(&mut queries, &m);
            queries = ln(&queries, &format!("{p}.ln3"), eps)?;
            let q = plus(&queries, &qpe);
            let a = attention(st, &format!("{p}.i2t"), &k, &q, &queries, c, d.heads)?;
            cpu::add_(&mut keys, &a);
            keys = ln(&keys, &format!("{p}.ln4"), eps)?;
        }
        let q = plus(&queries, &qpe);
        let k = plus(&keys, kpe);
        let a = attention(st, "trk.final", &q, &k, &keys, c, d.heads)?;
        cpu::add_(&mut queries, &a);
        queries = ln(&queries, "trk.final_ln", eps)?;

        // upscaling: two 2x2 / stride-2 ConvTs, the high-resolution
        // features added in, then one mask per token as a dot product
        let mut up = conv_t(st, "trk.up1", &keys, g)?;
        cpu::add_(&mut up, &emb.s1);
        let c1 = up.len() / (4 * t);
        up = cpu::layer_norm(&up, c1, st.f32("trk.up_ln.w")?, st.f32("trk.up_ln.b")?, 1e-6);
        up.iter_mut().for_each(|v| *v = gelu(*v));
        let mut up = conv_t(st, "trk.up2", &up, 2 * g)?;
        cpu::add_(&mut up, &emb.s0);
        up.iter_mut().for_each(|v| *v = gelu(*v));
        let c0 = up.len() / (16 * t);
        let nm = d.mask_tokens;
        let mut hyper = Vec::with_capacity(nm * c0);
        for i in 0..nm {
            hyper.extend(self.trk_mlp(&format!("trk.hyper{i}"), &queries[(2 + i) * c..(3 + i) * c], c)?);
        }
        let per_px = cpu::linear(&up, c0, W::F32(&hyper), None); // [P, nm]
        let px = 16 * t;
        let mut masks = vec![0f32; nm * px];
        par_rows(&mut masks, px, |i0, piece| {
            for (ii, m) in piece.chunks_mut(px).enumerate() {
                for (p, v) in m.iter_mut().enumerate() {
                    *v = per_px[p * nm + i0 + ii];
                }
            }
        });
        let iou: Vec<f32> = self.trk_mlp("trk.iou_head", &queries[c..2 * c], c)?.into_iter().map(sigmoid).collect();
        let object_score = self.trk_mlp("trk.obj_head", &queries[..c], c)?[0];

        let size = 4 * g;
        let pick = |i: usize| (masks[i * px..(i + 1) * px].to_vec(), vec![iou[i]]);
        let (masks, iou, n) = if multimask {
            (masks[px..].to_vec(), iou[1..].to_vec(), nm - 1)
        } else {
            // SAM 2's dynamic single mask: token 0's mask unless it is
            // unstable under a small threshold shift, else the best candidate
            let delta = self.manifest.f32("trk_stability_delta")?;
            let thresh = self.manifest.f32("trk_stability_thresh")?;
            let m0 = &masks[..px];
            let inter = m0.iter().filter(|&&v| v > delta).count();
            let union = m0.iter().filter(|&&v| v > -delta).count();
            let stability = if union > 0 { inter as f32 / union as f32 } else { 1.0 };
            let (m, s) = if stability >= thresh {
                pick(0)
            } else {
                pick((1..nm).max_by(|&a, &b| iou[a].total_cmp(&iou[b])).unwrap_or(1))
            };
            (m, s, 1)
        };
        Ok(PointOutput { masks, iou, object_score, n, size })
    }

    /// `name`'s MLP (ReLU between its `name.depth` Linears) on rows of `c`.
    fn trk_mlp(&self, name: &str, x: &[f32], c: usize) -> Result<Vec<f32>, Error> {
        let depth = self.manifest.usize(&format!("{name}.depth"))?;
        let mut h = x.to_vec();
        let mut n_in = c;
        for j in 0..depth {
            h = lin(&self.store, &h, n_in, &format!("{name}.{j}"))?;
            n_in = self.store.shape(&format!("{name}.{j}.w"))?[0];
            if j + 1 < depth {
                cpu::relu_(&mut h);
            }
        }
        Ok(h)
    }
}

fn plus(a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut y = a.to_vec();
    cpu::add_(&mut y, b);
    y
}

/// SAM's attention `name` (q/k/v/o, heads side by side at the projections'
/// width): `q_in [lq, c]` attends over `k_in` / `v_in [lk, c]`.
fn attention(
    st: &Store,
    name: &str,
    q_in: &[f32],
    k_in: &[f32],
    v_in: &[f32],
    c: usize,
    heads: usize,
) -> Result<Vec<f32>, Error> {
    let q = lin(st, q_in, c, &format!("{name}.q"))?;
    let k = lin(st, k_in, c, &format!("{name}.k"))?;
    let v = lin(st, v_in, c, &format!("{name}.v"))?;
    let di = st.shape(&format!("{name}.q.w"))?[0];
    let o = cpu::attention(&q, di, Rows::f32(&k, di), Rows::f32(&v, di), &Attn { heads, ..Default::default() });
    lin(st, &o, di, &format!("{name}.o"))
}

/// A 2x2 / stride-2 ConvTranspose2d (`name.w [Cin, Cout, 2, 2]`, `.b`) on
/// a channels-last `[side^2, Cin]` map -> `[(2 side)^2, Cout]`: its taps do
/// not overlap, so it is one matmul and a pixel shuffle.
fn conv_t(st: &Store, name: &str, x: &[f32], side: usize) -> Result<Vec<f32>, Error> {
    let w = st.f32(&format!("{name}.w"))?;
    let b = st.f32(&format!("{name}.b"))?;
    let shape = st.shape(&format!("{name}.w"))?;
    let (cin, cout) = (shape[0], shape[1]);
    // rows (a, b, o) of W^T for cpu::linear
    let mut wt = vec![0f32; 4 * cout * cin];
    for ci in 0..cin {
        for o in 0..cout {
            for ab in 0..4 {
                wt[(ab * cout + o) * cin + ci] = w[(ci * cout + o) * 4 + ab];
            }
        }
    }
    let y = cpu::linear(x, cin, W::F32(&wt), None); // [side^2, 4 cout]
    let s2 = 2 * side;
    let mut out = vec![0f32; s2 * s2 * cout];
    par_rows(&mut out, cout, |r0, piece| {
        for (ri, row) in piece.chunks_mut(cout).enumerate() {
            let r = r0 + ri;
            let (yy, xx) = (r / s2, r % s2);
            let src = &y[((yy / 2) * side + xx / 2) * 4 * cout + ((yy % 2) * 2 + xx % 2) * cout..][..cout];
            for ((o, &v), &bb) in row.iter_mut().zip(src).zip(b) {
                *o = v + bb;
            }
        }
    });
    Ok(out)
}
