// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The ViT backbone (32 layers over 72 x 72 tokens), as
//! `iron/applications/sam3/sam3_npu.py`'s `NpuViT`: tokens in window order
//! for the whole backbone, q/k head dims in half-split order (both baked
//! into the exported weights and tables), every Linear and the attention on
//! the NPU, LayerNorms / RoPE / residuals here. The patch embedding (a
//! 14 x 14 stride-14 conv) is a GEMM over the patches too (K = 588 padded
//! to 768).
//!
//! The host works in cached memory throughout and moves whole buffers in
//! and out of the device BOs (`push` / `pull`): their host mappings are
//! uncached.

use iron_xrt::{bf16_to_f32, f32_to_bf16};

use crate::cpu::{ln_row, par_rows};
use crate::npu::{pull, push};
use crate::{Error, Sam3, gemm, mha};

impl Sam3 {
    /// `pixels [3, S, S]` -> the backbone's last hidden state `[T, 1024]`,
    /// raster order.
    pub fn vit(&mut self, pixels: &[f32]) -> Result<Vec<f32>, Error> {
        let c = self.cfg.clone();
        let (t, dim, g, ps, s) = (c.tokens(), c.vit_dim, c.grid, c.patch, c.image_size);
        if pixels.len() != 3 * s * s {
            return Err(Error::Input(format!("pixels must be [3, {s}, {s}]")));
        }
        let perm = self.store.i32("v.perm")?.to_vec();
        let eps = c.vit_eps;

        // patches, window order: row j = patch perm[j], columns (ch, ky, kx)
        let ke = self.npu.spec("v_embed")?.k;
        let mut patches = vec![0u16; t * ke];
        par_rows(&mut patches, ke, |r0, piece| {
            for (ri, row) in piece.chunks_mut(ke).enumerate() {
                let p = perm[r0 + ri] as usize;
                let (py, px) = (p / g, p % g);
                for ch in 0..3 {
                    for ky in 0..ps {
                        let src = ch * s * s + (py * ps + ky) * s + px * ps;
                        for kx in 0..ps {
                            row[(ch * ps + ky) * ps + kx] = f32_to_bf16(pixels[src + kx]);
                        }
                    }
                }
            }
        });
        self.io.v_embed.set_a(&patches)?;
        gemm(&mut self.npu, &self.io.v_embed, &self.w["v.embed"], &mut self.timing)?;
        let emb = self.io.v_embed.get_c(t)?;
        let st = &self.store;
        let pos = st.f32("v.pos")?;
        let (lw, lb) = (st.f32("v.ln_pre.w")?, st.f32("v.ln_pre.b")?);
        let mut x = vec![0f32; t * dim];
        par_rows(&mut x, dim, |r0, piece| {
            let mut tmp = vec![0f32; dim];
            for (ri, row) in piece.chunks_mut(dim).enumerate() {
                let r = r0 + ri;
                for j in 0..dim {
                    tmp[j] = bf16_to_f32(emb[r * dim + j]) + pos[r * dim + j];
                }
                ln_row(&tmp, row, lw, lb, eps);
            }
        });

        let (heads, hd) = (c.vit_heads, dim / c.vit_heads);
        let half = hd / 2;
        let ws2 = c.window * c.window;
        let rope_win = (st.f32("v.rope.win.cos")?.to_vec(), st.f32("v.rope.win.sin")?.to_vec());
        let rope_glob = (st.f32("v.rope.glob.cos")?.to_vec(), st.f32("v.rope.glob.sin")?.to_vec());
        let mut a = vec![0u16; t * dim];
        let mut mq = vec![0u16; t * dim];
        for i in 0..c.vit_layers {
            let p = |n: &str| format!("v.{i}.{n}");
            // LN1 -> qkv
            {
                let st = &self.store;
                layer_norm_bf16(&x, dim, st.f32(&p("ln1.w"))?, st.f32(&p("ln1.b"))?, eps, &mut a);
            }
            self.io.v_qkv.set_a(&a)?;
            gemm(&mut self.npu, &self.io.v_qkv, &self.w[&p("qkv")], &mut self.timing)?;
            let t0 = std::time::Instant::now();
            let qkv = self.io.v_qkv.get_c(t)?;

            // RoPE into the MHA's layout: [n_win * heads, 576, 64] for the
            // windowed layers, [T, heads * 64] for the global ones
            let global = c.vit_global.contains(&i);
            let (cos, sin) = if global { (&rope_glob.0, &rope_glob.1) } else { (&rope_win.0, &rope_win.1) };
            // MHA row r (64 wide) <-> (token, head)
            let at = |r: usize| -> (usize, usize) {
                if global {
                    (r / heads, r % heads)
                } else {
                    let (wh, si) = (r / ws2, r % ws2);
                    ((wh / heads) * ws2 + si, wh % heads)
                }
            };
            let m = if global { &mut self.io.mha_glob } else { &mut self.io.mha_win };
            for (part, buf) in [(0, &mut m.q), (1, &mut m.k), (2, &mut m.v)] {
                par_rows(&mut mq, hd, |r0, piece| {
                    for (ri, row) in piece.chunks_mut(hd).enumerate() {
                        let (tok, h) = at(r0 + ri);
                        let src = &qkv[tok * 3 * dim + part * dim + h * hd..][..hd];
                        if part == 2 {
                            row.copy_from_slice(src);
                            continue;
                        }
                        for d in 0..half {
                            let (x1, x2) = (bf16_to_f32(src[d]), bf16_to_f32(src[half + d]));
                            let (co, si) = (cos[tok * half + d], sin[tok * half + d]);
                            row[d] = f32_to_bf16(x1 * co - x2 * si);
                            row[half + d] = f32_to_bf16(x2 * co + x1 * si);
                        }
                    }
                });
                push(&mq, buf)?;
            }
            let t1 = std::time::Instant::now();
            mha(&mut self.npu, m, &mut self.timing)?;
            let t2 = std::time::Instant::now();
            // attention output -> o's A, token-major
            let o = pull(&m.o, t * dim)?;
            par_rows(&mut a, dim, |t0, piece| {
                for (ti, row) in piece.chunks_mut(dim).enumerate() {
                    let tok = t0 + ti;
                    for h in 0..heads {
                        let r = if global {
                            tok * heads + h
                        } else {
                            let (w, si) = (tok / ws2, tok % ws2);
                            (w * heads + h) * ws2 + si
                        };
                        row[h * hd..(h + 1) * hd].copy_from_slice(&o[r * hd..(r + 1) * hd]);
                    }
                }
            });
            self.io.v_o.set_a(&a)?;
            self.timing.add("vit_rope_io", t1 - t0 + t2.elapsed());
            gemm(&mut self.npu, &self.io.v_o, &self.w[&p("o")], &mut self.timing)?;
            add_bf16(&mut x, &self.io.v_o.get_c(t)?, dim, None);

            // LN2 -> fc1 (+gelu) -> fc2, the intermediate staying on the device
            {
                let st = &self.store;
                layer_norm_bf16(&x, dim, st.f32(&p("ln2.w"))?, st.f32(&p("ln2.b"))?, eps, &mut a);
            }
            self.io.v_fc1.set_a(&a)?;
            gemm(&mut self.npu, &self.io.v_fc1, &self.w[&p("fc1")], &mut self.timing)?;
            gemm(&mut self.npu, &self.io.v_fc2, &self.w[&p("fc2")], &mut self.timing)?;
            add_bf16(&mut x, &self.io.v_fc2.get_c(t)?, dim, Some(self.store.f32(&p("fc2.b"))?));
        }
        let mut out = vec![0f32; t * dim];
        for (j, row) in x.chunks(dim).enumerate() {
            let r = perm[j] as usize;
            out[r * dim..(r + 1) * dim].copy_from_slice(row);
        }
        Ok(out)
    }
}

/// LayerNorm of `x`'s rows into `dst` (bf16), rows `dim` wide.
pub(crate) fn layer_norm_bf16(x: &[f32], dim: usize, w: &[f32], b: &[f32], eps: f32, dst: &mut [u16]) {
    let n = x.len();
    par_rows(&mut dst[..n], dim, |r0, piece| {
        let mut tmp = vec![0f32; dim];
        for (ri, row) in piece.chunks_mut(dim).enumerate() {
            let r = r0 + ri;
            ln_row(&x[r * dim..(r + 1) * dim], &mut tmp, w, b, eps);
            for (o, v) in row.iter_mut().zip(&tmp) {
                *o = f32_to_bf16(*v);
            }
        }
    });
}

/// `x += c (+ bias)`, `c` bf16 rows `dim` wide (at least as many as `x`'s).
pub(crate) fn add_bf16(x: &mut [f32], c: &[u16], dim: usize, bias: Option<&[f32]>) {
    par_rows(x, dim, |r0, piece| {
        for (ri, row) in piece.chunks_mut(dim).enumerate() {
            let src = &c[(r0 + ri) * dim..(r0 + ri + 1) * dim];
            for j in 0..dim {
                row[j] += bf16_to_f32(src[j]) + bias.map_or(0.0, |b| b[j]);
            }
        }
    });
}
