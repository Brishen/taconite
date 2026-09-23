// SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
// SPDX-License-Identifier: Apache-2.0

//! The mask decoder, as `iron/applications/sam3/mask_npu.py`:
//!
//! - the DETR encoder output attends to the prompt -- the same folded
//!   cross-attention as the DETR encoder's, on the same two kernels;
//! - the pixel decoder: nearest 2x upsample + skip, 3x3 conv (NPU), GroupNorm
//!   + ReLU, twice (72 -> 144 -> 288);
//! - the mask head folded into one GEMM over the 288 x 288 pixel embedding:
//!   `pred_masks[q] = (W_inst^T m_q) . p + m_q . b_inst` for the 200 query
//!   mask embeddings `m_q`, and the semantic head in column 200.

use taconite::{bf16_to_f32, f32_to_bf16};

use crate::cpu::{self, par_rows};
use crate::pack::pack_b;
use crate::text::Text;
use crate::{Error, Sam3, gemm};

impl Sam3 {
    /// `(pred_masks [Q, S, S], semantic [S, S])` from the decoder's last
    /// normalised query states, the FPN levels (channels-last) and the DETR
    /// encoder output.
    pub fn mask_decoder(
        &mut self,
        hidden: &[f32],
        fpn: &[Vec<f32>; 3],
        enc: &[f32],
        text: &Text,
    ) -> Result<(Vec<f32>, Vec<f32>), Error> {
        let c = self.cfg.clone();
        let (d, g) = (c.d_model, c.grid);
        let q = hidden.len() / d;

        // prompt cross-attention on the encoder output
        self.fold_cross("m.ca", "m", text)?;
        let h = cpu::layer_norm(enc, d, self.store.f32("m.ca_ln.w")?, self.store.f32("m.ca_ln.b")?, 1e-5);
        let mut p = enc.to_vec();
        self.cross_npu(&h, &mut p, "m", text)?;

        // pixel decoder
        let mut side = g;
        for (i, skip) in [&fpn[1], &fpn[0]].into_iter().enumerate() {
            let s2 = 2 * side;
            let mut up = vec![0u16; s2 * s2 * d];
            par_rows(&mut up, d, |r0, piece| {
                for (ri, row) in piece.chunks_mut(d).enumerate() {
                    let r = r0 + ri;
                    let src = ((r / s2) / 2 * side + (r % s2) / 2) * d;
                    for j in 0..d {
                        row[j] = f32_to_bf16(p[src + j] + skip[r * d + j]);
                    }
                }
            });
            let bias = self.store.f32(&format!("m.conv{i}.b"))?.to_vec();
            let mut y = self.conv3x3(&up, s2, &format!("m.conv{i}"), &bias)?;
            group_norm_relu(
                &mut y,
                d,
                8,
                self.store.f32(&format!("m.gn{i}.w"))?,
                self.store.f32(&format!("m.gn{i}.b"))?,
            );
            p = y;
            side = s2;
        }

        // folded mask + semantic head
        let mut m = self.lin(hidden, d, "m.embed.0")?;
        cpu::relu_(&mut m);
        let mut m = self.lin(&m, d, "m.embed.1")?;
        cpu::relu_(&mut m);
        let m = self.lin(&m, d, "m.embed.2")?; // [Q, D]
        let st = &self.store;
        let (wi, bi) = (st.f32("m.inst.w")?, st.f32("m.inst.b")?); // [D out, D in]
        let (ws, bs) = (st.f32("m.sem.w")?, st.f32("m.sem.b")?);
        let spec = self.npu.spec("m_head")?.clone();
        let n = spec.n;
        if q + 1 > n {
            return Err(Error::Input(format!("{q} queries + semantic > {n}")));
        }
        let mut bm = vec![0f32; d * n];
        let mut bb = vec![0f32; n];
        for qi in 0..q {
            let mq = &m[qi * d..(qi + 1) * d];
            for i in 0..d {
                bm[i * n + qi] = (0..d).map(|o| wi[o * d + i] * mq[o]).sum();
            }
            bb[qi] = cpu::dot(mq, bi);
        }
        for i in 0..d {
            bm[i * n + q] = ws[i];
        }
        bb[q] = bs[0];
        let packed = pack_b(&spec, &bm, Some(&bb));
        self.slots.get_mut("m.head").unwrap().write(&packed)?;
        let px = side * side;
        let mut pb = vec![0u16; p.len()];
        crate::narrow(&p, &mut pb);
        self.io.m_head.set_a(&pb)?;
        gemm(&mut self.npu, &self.io.m_head, &self.slots["m.head"], &mut self.timing)?;
        let out = self.io.m_head.get_c(px)?;
        let mut masks = vec![0f32; q * px];
        par_rows(&mut masks, px, |q0, piece| {
            for (qi, plane) in piece.chunks_mut(px).enumerate() {
                for (pi, v) in plane.iter_mut().enumerate() {
                    *v = bf16_to_f32(out[pi * n + q0 + qi]);
                }
            }
        });
        let semantic = (0..px).map(|pi| bf16_to_f32(out[pi * n + q])).collect();
        Ok((masks, semantic))
    }
}

/// `relu(GroupNorm(x))` in place, channels-last `x [H*W, C]`.
fn group_norm_relu(x: &mut [f32], c: usize, groups: usize, w: &[f32], b: &[f32]) {
    let cg = c / groups;
    let px = x.len() / c;
    let mut stats = vec![(0f64, 0f64); groups];
    for row in x.chunks(c) {
        for (gi, s) in stats.iter_mut().enumerate() {
            for &v in &row[gi * cg..(gi + 1) * cg] {
                s.0 += v as f64;
                s.1 += (v as f64) * (v as f64);
            }
        }
    }
    let nn = (px * cg) as f64;
    let norm: Vec<(f32, f32)> = stats
        .iter()
        .map(|&(s, ss)| {
            let mean = s / nn;
            let var = (ss / nn - mean * mean).max(0.0);
            (mean as f32, (1.0 / (var + 1e-5).sqrt()) as f32)
        })
        .collect();
    par_rows(x, c, |_, piece| {
        for row in piece.chunks_mut(c) {
            for (j, v) in row.iter_mut().enumerate() {
                let (mean, inv) = norm[j / cg];
                *v = ((*v - mean) * inv * w[j] + b[j]).max(0.0);
            }
        }
    });
}
