// SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
// SPDX-License-Identifier: Apache-2.0

//! The FPN neck and the 3x3 convolutions it shares with the mask decoder,
//! as `iron/applications/sam3/neck_npu.py`.
//!
//! A 3x3 (pad 1) conv is one GEMM over an overlapping view: the host writes
//! `Y[p] = [X_pad[p] | X_pad[p + Wp] | X_pad[p + 2 Wp]]` (pixel `p` of the
//! zero-bordered image at row pitch `Wp = W + 2`, next to the pixels one and
//! two rows below), so output pixel `p`'s window is `Y[p..p + 3]`,
//! contiguous; a GEMM row covers two pixels (`K = 4D` at row stride `2D`)
//! against a banded B. The output is pixel-major `[H x Wp, OC]` with two
//! junk columns per image row, dropped here.
//!
//! Levels 0-2 start from the same `[72 x 72, 1024]` map, so their ConvT /
//! 1x1 first steps are one GEMM (columns `[level 0 | level 1 | level 2]`);
//! level 0's GELU and pixel shuffle happen here, then its second ConvT
//! (1x1 folded in) on the NPU; every level ends with its 3x3.

use taconite::{bf16_to_f32, f32_to_bf16};

use crate::cpu::{gelu, par_rows};
use crate::npu::pull;
use crate::{Error, Sam3, gemm, narrow};

impl Sam3 {
    /// 3x3 conv, pad 1: `x [side, side, C]` (bf16) -> `conv(x) + bias`
    /// `[side, side, OC]` (f32), weights `self.w[wkey]`.
    pub(crate) fn conv3x3(&mut self, x: &[u16], side: usize, wkey: &str, bias: &[f32]) -> Result<Vec<f32>, Error> {
        let g = self.npu.spec("conv")?;
        let (k, lda) = (g.k, g.lda);
        let d = (k - lda) / 2;
        let c = d / 3;
        let oc = bias.len();
        let wp = side + 2;
        let n = side * wp;
        let mut ybuf = vec![0u16; n * d];
        par_rows(&mut ybuf, d, |p0, piece| {
            for (pi, row) in piece.chunks_mut(d).enumerate() {
                let p = p0 + pi;
                for dy in 0..3 {
                    let r = p + dy * wp;
                    let (yy, xx) = (r / wp, r % wp);
                    let dst = &mut row[dy * c..(dy + 1) * c];
                    if (1..=side).contains(&yy) && (1..=side).contains(&xx) {
                        let s = ((yy - 1) * side + xx - 1) * c;
                        dst.copy_from_slice(&x[s..s + c]);
                    } else {
                        dst.fill(0);
                    }
                }
            }
        });
        let io = self.io.conv.get_mut(&side).ok_or_else(|| Error::Input(format!("no conv buffers for {side} px")))?;
        io.set_a(&ybuf)?;
        let io = &self.io.conv[&side];
        gemm(&mut self.npu, io, &self.w[wkey], &mut self.timing)?;
        let out_bf = pull(&io.c, n * oc)?;
        let mut out = vec![0f32; side * side * oc];
        par_rows(&mut out, side * oc, |y0, piece| {
            for (yi, row) in piece.chunks_mut(side * oc).enumerate() {
                let src = &out_bf[(y0 + yi) * wp * oc..][..side * oc];
                for (j, o) in row.iter_mut().enumerate() {
                    *o = bf16_to_f32(src[j]) + bias[j % oc];
                }
            }
        });
        Ok(out)
    }

    /// The backbone's `[T, 1024]` (raster) -> FPN levels 0-2, channels-last
    /// f32: `[288^2, 256]`, `[144^2, 256]`, `[72^2, 256]`.
    pub fn neck(&mut self, vit: &[f32]) -> Result<[Vec<f32>; 3], Error> {
        let cfg = self.cfg.clone();
        let (g, t) = (cfg.grid, cfg.tokens());
        let (s0, s1, s2) = (cfg.neck_splits[0], cfg.neck_splits[1], cfg.neck_splits[2]);
        let width = s0 + s1 + s2;
        let fpn = cfg.d_model;
        let mut vb = vec![0u16; vit.len()];
        narrow(vit, &mut vb);
        self.io.n_in.set_a(&vb)?;
        gemm(&mut self.npu, &self.io.n_in, &self.w["n.in"], &mut self.timing)?;
        let y = self.io.n_in.get_c(t)?;
        let y = &y[..];

        // level 0: gelu, then shuffle the 2x2 taps onto the 144 grid
        let mid = s0 / 4;
        let g2 = 2 * g;
        let mut a0 = vec![0u16; 4 * t * mid];
        par_rows(&mut a0, mid, |r0, piece| {
            for (ri, row) in piece.chunks_mut(mid).enumerate() {
                let r = r0 + ri;
                let (yy, xx) = (r / g2, r % g2);
                let src = &y[((yy / 2) * g + xx / 2) * width + ((yy % 2) * 2 + xx % 2) * mid..][..mid];
                for (o, &v) in row.iter_mut().zip(src) {
                    *o = f32_to_bf16(gelu(bf16_to_f32(v)));
                }
            }
        });
        // levels 1 and 2 straight from the shared GEMM
        let shuffle = |src: &[u16], h: usize, off: usize| -> Vec<u16> {
            let mut out = vec![0u16; 4 * h * h * fpn];
            par_rows(&mut out, fpn, |r0, piece| {
                for (ri, row) in piece.chunks_mut(fpn).enumerate() {
                    let r = r0 + ri;
                    let (yy, xx) = (r / (2 * h), r % (2 * h));
                    let s = ((yy / 2) * h + xx / 2) * width + off + ((yy % 2) * 2 + xx % 2) * fpn;
                    row.copy_from_slice(&src[s..s + fpn]);
                }
            });
            out
        };
        let u1 = shuffle(y, g, s0);
        let mut u2 = vec![0u16; t * fpn];
        for (p, row) in u2.chunks_mut(fpn).enumerate() {
            row.copy_from_slice(&y[p * width + s0 + s1..][..fpn]);
        }
        self.io.n_up.set_a(&a0)?;
        gemm(&mut self.npu, &self.io.n_up, &self.w["n.up"], &mut self.timing)?;
        let up = self.io.n_up.get_c(4 * t)?;
        let n_up = self.npu.spec("n_up")?.n;
        let mut u0 = vec![0u16; 16 * t * fpn];
        {
            let g4 = 4 * g;
            par_rows(&mut u0, fpn, |r0, piece| {
                for (ri, row) in piece.chunks_mut(fpn).enumerate() {
                    let r = r0 + ri;
                    let (yy, xx) = (r / g4, r % g4);
                    let s = ((yy / 2) * g2 + xx / 2) * n_up + ((yy % 2) * 2 + xx % 2) * fpn;
                    row.copy_from_slice(&up[s..s + fpn]);
                }
            });
        }
        let b: Vec<Vec<f32>> =
            (0..3).map(|i| self.store.f32(&format!("n.conv{i}.b")).map(<[f32]>::to_vec)).collect::<Result<_, _>>()?;
        let f0 = self.conv3x3(&u0, 4 * g, "n.conv0", &b[0])?;
        let f1 = self.conv3x3(&u1, 2 * g, "n.conv1", &b[1])?;
        let f2 = self.conv3x3(&u2, g, "n.conv2", &b[2])?;
        Ok([f0, f1, f2])
    }
}
