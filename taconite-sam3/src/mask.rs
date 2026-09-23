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
        let t0 = std::time::Instant::now();
        self.fold_cross("m.ca", "m", text)?;
        let h = cpu::layer_norm(enc, d, self.store.f32("m.ca_ln.w")?, self.store.f32("m.ca_ln.b")?, 1e-5);
        let mut p = enc.to_vec();
        self.cross_npu(&h, &mut p, "m", text)?;
        self.timing.add("mask_cross", t0.elapsed());

        // pixel decoder
        let mut side = g;
        for (i, skip) in [&fpn[1], &fpn[0]].into_iter().enumerate() {
            let s2 = 2 * side;
            let t0 = std::time::Instant::now();
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
            self.timing.add("mask_up", t0.elapsed());
            let bias = self.store.f32(&format!("m.conv{i}.b"))?.to_vec();
            let mut y = self.conv3x3(&up, s2, &format!("m.conv{i}"), &bias)?;
            let t0 = std::time::Instant::now();
            group_norm_relu(
                &mut y,
                d,
                8,
                self.store.f32(&format!("m.gn{i}.w"))?,
                self.store.f32(&format!("m.gn{i}.b"))?,
            );
            self.timing.add("mask_gn", t0.elapsed());
            p = y;
            side = s2;
        }
        let t0 = std::time::Instant::now();

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
        self.timing.add("mask_head_prep", t0.elapsed());
        gemm(&mut self.npu, &self.io.m_head, &self.slots["m.head"], &mut self.timing)?;
        let t0 = std::time::Instant::now();
        let out = self.io.m_head.get_c(px)?;
        // [px, n] -> [q, px], in blocks of pixels: contiguous reads, and
        // each plane written a run at a time (read a plane at a time it
        // is a strided gather over the whole output)
        const PB: usize = 64;
        let blocks = px.div_ceil(PB);
        let mut planes = vec![0f32; (q + 1) * blocks * PB]; // [q + 1, blocks * PB]
        let stride = blocks * PB;
        {
            let ptr = planes.as_mut_ptr() as usize;
            let mut tasks = vec![0u8; blocks];
            par_rows(&mut tasks, 1, |b0, piece| {
                for bi in 0..piece.len() {
                    let p0 = (b0 + bi) * PB;
                    let np = (px - p0).min(PB);
                    for qi in 0..=q {
                        // disjoint runs of one plane per task
                        let dst = unsafe { std::slice::from_raw_parts_mut((ptr as *mut f32).add(qi * stride + p0), np) };
                        for (pi, v) in dst.iter_mut().enumerate() {
                            *v = bf16_to_f32(out[(p0 + pi) * n + qi]);
                        }
                    }
                }
            });
        }
        let mut masks = vec![0f32; q * px];
        par_rows(&mut masks, px, |q0, piece| {
            for (qi, plane) in piece.chunks_mut(px).enumerate() {
                plane.copy_from_slice(&planes[(q0 + qi) * stride..][..px]);
            }
        });
        let semantic = planes[q * stride..][..px].to_vec();
        self.timing.add("mask_out", t0.elapsed());
        Ok((masks, semantic))
    }
}

/// `relu(GroupNorm(x))` in place, channels-last `x [H*W, C]`.
fn group_norm_relu(x: &mut [f32], c: usize, groups: usize, w: &[f32], b: &[f32]) {
    let cg = c / groups;
    let px = x.len() / c;
    // per-group sums, partial per task then combined (f64: 21M values)
    let tasks = px.min(4 * cpu::threads()).max(1);
    let per = px.div_ceil(tasks);
    let mut partial = vec![[0f64; 2 * 8]; tasks];
    assert!(groups <= 8, "GroupNorm with more than 8 groups");
    par_rows(&mut partial, 1, |t0, piece| {
        for (ti, st) in piece.iter_mut().enumerate() {
            let r0 = (t0 + ti) * per;
            let r1 = (r0 + per).min(px);
            for row in x[r0 * c..r1 * c].chunks(c) {
                for gi in 0..groups {
                    let (mut s, mut ss) = (0f32, 0f32);
                    for &v in &row[gi * cg..(gi + 1) * cg] {
                        s += v;
                        ss += v * v;
                    }
                    st[2 * gi] += s as f64;
                    st[2 * gi + 1] += ss as f64;
                }
            }
        }
    });
    let mut stats = vec![(0f64, 0f64); groups];
    for st in &partial {
        for (gi, s) in stats.iter_mut().enumerate() {
            s.0 += st[2 * gi];
            s.1 += st[2 * gi + 1];
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
