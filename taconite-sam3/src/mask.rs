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

use crate::bundle::{GemmSpec, Store};
use crate::cpu::{self, par_rows};
use crate::detr::lin;
use crate::pack::pack_b;
use crate::text::Text;
use crate::{Error, Sam3, gemm, gemm_dev};

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
        #[cfg(feature = "gpu")]
        if self.gpu.is_some() {
            let glue = &self.gpu().glue;
            glue.fpn[0].write(&fpn[0]);
            glue.fpn[1].write(&fpn[1]);
            glue.x.write(enc);
            return self.mask_decoder_gpu(hidden, text, false);
        }
        let c = self.cfg.clone();
        let (d, g) = (c.d_model, c.grid);

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
        self.mask_head(hidden, side, Some(&p), None)
    }

    /// The mask decoder in NPU+GPU mode, from `glue.x` and `glue.fpn`: the
    /// pixel decoder's maps stay on the GPU, which writes the mask head's A.
    #[cfg(feature = "gpu")]
    pub(crate) fn mask_decoder_gpu(
        &mut self,
        hidden: &[f32],
        text: &Text,
        folded: bool,
    ) -> Result<(Vec<f32>, Vec<f32>), Error> {
        let g = self.cfg.grid;
        let d = self.cfg.d_model;
        let t0 = std::time::Instant::now();
        if !folded {
            self.fold_cross("m.ca", "m", text)?;
        }
        let valid: Vec<u32> = text.valid.iter().map(|&v| v as u32).collect();
        let head_spec = self.npu.spec("m_head")?.clone();
        let Sam3 { store, npu, io, w, slots, timing: t, gpu, .. } = self;
        let gpu = gpu.as_ref().expect("NPU+GPU mode");
        let gl = &gpu.glue;
        gl.valid.write(&valid);
        let store = &*store;
        // the head's weights (an MLP over the queries, folded and packed)
        // need only `hidden`: built on a side thread beside the pixel decoder
        let packed = std::thread::scope(|sc| -> Result<Vec<u8>, Error> {
            let head = sc.spawn(|| {
                let t0 = std::time::Instant::now();
                mask_head_packed(store, d, &head_spec, hidden).map(|p| (p, t0.elapsed()))
            });
            gpu.run(&gl.mask_ln, "mask_ln", t)?;
            gemm_dev(npu, &io.d_s, &slots["m.s"], t)?;
            gpu.run(&gl.softmax, "prompt_softmax", t)?;
            gemm_dev(npu, &io.d_c, &slots["m.c"], t)?;
            gpu.run(&gl.mask_add, "mask_add", t)?;
            t.add("mask_cross", t0.elapsed());
            for i in 0..2 {
                let s2 = g << (i + 1);
                gpu.run(&gl.mask_up[i], "mask_up", t)?;
                gemm_dev(npu, &io.conv[&s2], &w[&format!("m.conv{i}")], t)?;
                gpu.run(&gl.mask_gn[i], "mask_gn", t)?;
            }
            let (packed, dt) = head.join().expect("the mask head thread")?;
            t.add("mask_head_weights (overlapped)", dt);
            Ok(packed)
        })?;
        self.mask_head(hidden, 4 * g, None, Some(packed))
    }

    /// The folded mask + semantic head over the `side x side` pixel
    /// embedding: `p` (f32, narrowed here into the GEMM's A), or `None` when
    /// the GPU has written that A.
    fn mask_head(
        &mut self,
        hidden: &[f32],
        side: usize,
        p: Option<&[f32]>,
        packed: Option<Vec<u8>>,
    ) -> Result<(Vec<f32>, Vec<f32>), Error> {
        let d = self.cfg.d_model;
        let q = hidden.len() / d;
        let t0 = std::time::Instant::now();

        // folded mask + semantic head
        let packed = match packed {
            Some(p) => p,
            None => mask_head_packed(&self.store, d, self.npu.spec("m_head")?, hidden)?,
        };
        self.slots.get_mut("m.head").unwrap().write(&packed)?;
        let n = self.npu.spec("m_head")?.n;
        let px = side * side;
        match p {
            Some(p) => {
                let mut pb = vec![0u16; p.len()];
                crate::narrow(p, &mut pb);
                self.io.m_head.set_a(&pb)?;
                self.timing.add("mask_head_prep", t0.elapsed());
                gemm(&mut self.npu, &self.io.m_head, &self.slots["m.head"], &mut self.timing)?;
            }
            None => {
                self.timing.add("mask_head_prep", t0.elapsed());
                gemm_dev(&mut self.npu, &self.io.m_head, &self.slots["m.head"], &mut self.timing)?;
            }
        }
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

/// The folded mask + semantic head's B for the `m_head` GEMM: the query
/// mask embeddings `m = MLP(hidden)` folded through the instance head,
/// `B[i, q] = (W_inst^T m_q)[i]` with bias `m_q . b_inst`, the semantic
/// head in column `Q`; packed.
pub(crate) fn mask_head_packed(store: &Store, d: usize, spec: &GemmSpec, hidden: &[f32]) -> Result<Vec<u8>, Error> {
    let q = hidden.len() / d;
    let mut m = lin(store, hidden, d, "m.embed.0")?;
    cpu::relu_(&mut m);
    let mut m = lin(store, &m, d, "m.embed.1")?;
    cpu::relu_(&mut m);
    let m = lin(store, &m, d, "m.embed.2")?; // [Q, D]
    let (wi, bi) = (store.f32("m.inst.w")?, store.f32("m.inst.b")?); // [D out, D in]
    let (ws, bs) = (store.f32("m.sem.w")?, store.f32("m.sem.b")?);
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
    Ok(pack_b(spec, &bm, Some(&bb)))
}
