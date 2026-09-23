// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One transformer tower on the device, as
//! `iron/applications/clip_vit_h14/clip_npu.py`'s `NpuTower`: per layer
//!
//!   qkv GEMM -> MHA -> o GEMM -> AddLN(+o) -> fc1 (GELU) -> fc2 ->
//!   AddLN(+fc2)
//!
//! each kernel reading the previous one's output buffer in place. The
//! LayerNorm affines are folded into the packed qkv / fc1 weights, fc2's
//! bias rides fc1's extra column; the residual stream is bf16 between
//! layers, ping-ponging between two device buffers.

use std::time::Instant;

use taconite_bundle::Store;
use taconite::{bf16_to_f32, f32_to_bf16};
use taconite_sam3::Timing;
use taconite_sam3::cpu::{ln_row, par_rows};

use crate::npu::{BF16, Buffer, Io, Npu};
use crate::{Error, pull, push};

pub struct Tower {
    /// `v` (vision) or `t` (text): kernel and tensor name prefix
    pub p: &'static str,
    pub layers: usize,
    pub dim: usize,
    /// device rows (the GEMMs' padded height)
    pub rows: usize,
    pub eps: f32,
    qkv: Io,
    o: Io,
    fc1: Io,
    fc2: Io,
    xres: [Buffer; 2],
    /// per layer: packed qkv, o, fc1, fc2
    w: Vec<[Buffer; 4]>,
    /// per MHA dispatch: q, k, v (views of the qkv GEMM's output), o (of
    /// the o GEMM's input)
    mha: Vec<[Buffer; 4]>,
}

impl Tower {
    /// `mha_rows`: the row offset of each MHA dispatch and the rows it
    /// spans (one per image for the vision tower, one over every prompt for
    /// the text tower).
    pub fn new(
        npu: &Npu,
        store: &Store,
        p: &'static str,
        layers: usize,
        dim: usize,
        rows: usize,
        eps: f32,
        mha_rows: &[(usize, usize)],
    ) -> Result<Self, Error> {
        let k = |n: &str| format!("{p}_{n}");
        let qkv = npu.io(&k("qkv"), rows)?;
        let o = npu.io(&k("o"), rows)?;
        let fc1 = npu.io(&k("fc1"), rows)?;
        let fc2 = npu.io_chained(&k("fc2"), &fc1)?;
        for (io, k, n) in [(&qkv, dim, 3 * dim), (&o, dim, dim), (&fc1, dim, fc1.n), (&fc2, fc1.n, dim)] {
            if (io.k, io.n) != (k, n) {
                return Err(Error::Bundle(format!("{}: K x N = {} x {}, expected {k} x {n}", io.key, io.k, io.n)));
            }
        }
        let xres = [npu.zeros(rows * dim)?, npu.zeros(rows * dim)?];
        let mut w = Vec::with_capacity(layers);
        for i in 0..layers {
            let up = |n: &str| npu.upload(store.u8(&format!("{p}.{i}.{n}"))?);
            w.push([up("qkv")?, up("o")?, up("fc1")?, up("fc2")?]);
        }
        let mut mha = Vec::new();
        for &(r0, n) in mha_rows {
            if r0 + n > rows {
                return Err(Error::Bundle(format!("{p}: an MHA dispatch reaches row {} of {rows}", r0 + n)));
            }
            // q, k and v are column ranges of the same rows (the kernel's
            // row strides and columns are compiled in)
            let view = |b: &Buffer, width: usize| b.sub(r0 * width * BF16, n * width * BF16);
            mha.push([view(&qkv.c, 3 * dim)?, view(&qkv.c, 3 * dim)?, view(&qkv.c, 3 * dim)?, view(&o.a, dim)?]);
        }
        Ok(Tower { p, layers, dim, rows, eps, qkv, o, fc1, fc2, xres, w, mha })
    }

    /// Every layer over `x` (`rows x dim` f32, the embedded input with zero
    /// padding rows); returns the final residual stream (`rows x dim`).
    pub fn run(&mut self, npu: &Npu, x: &[f32], timing: &mut Timing) -> Result<Vec<f32>, Error> {
        let (dim, rows) = (self.dim, self.rows);
        assert_eq!(x.len(), rows * dim);
        let t0 = Instant::now();
        // the residual stream in bf16, and its LayerNorm (no affine: folded
        // into qkv) as the first qkv GEMM's A
        let mut xb = vec![0u16; rows * dim];
        par_rows(&mut xb, dim, |r0, piece| {
            for (i, v) in piece.iter_mut().enumerate() {
                *v = f32_to_bf16(x[r0 * dim + i]);
            }
        });
        let mut h = vec![0u16; rows * dim];
        let (ones, zeros) = (vec![1f32; dim], vec![0f32; dim]);
        par_rows(&mut h, dim, |r0, piece| {
            let (mut xr, mut out) = (vec![0f32; dim], vec![0f32; dim]);
            for (ri, row) in piece.chunks_mut(dim).enumerate() {
                let src = &xb[(r0 + ri) * dim..][..dim];
                for (d, s) in xr.iter_mut().zip(src) {
                    *d = bf16_to_f32(*s);
                }
                ln_row(&xr, &mut out, &ones, &zeros, self.eps);
                for (d, v) in row.iter_mut().zip(&out) {
                    *d = f32_to_bf16(*v);
                }
            }
        });
        push(&xb, &mut self.xres[0])?;
        push(&h, &mut self.qkv.a)?;
        timing.add(&format!("{}_in", self.p), t0.elapsed());

        let p = self.p;
        let key = |n: &str| format!("{p}_{n}");
        let (k_qkv, k_o, k_fc1, k_fc2, k_mha, k_ln) =
            (key("qkv"), key("o"), key("fc1"), key("fc2"), key("mha"), key("addln"));
        let npu_t = |t: &mut Timing, k: &str, d| t.add(&format!("npu:{k}"), d);
        let [xa, xb_] = &self.xres;
        for w in &self.w {
            npu_t(timing, &k_qkv, npu.gemm(&self.qkv, &w[0])?);
            for m in &self.mha {
                npu_t(timing, &k_mha, npu.op(&k_mha, &[&m[0], &m[1], &m[2], &m[3]])?);
            }
            npu_t(timing, &k_o, npu.gemm(&self.o, &w[1])?);
            npu_t(timing, &k_ln, npu.op(&k_ln, &[xa, &self.o.c, xb_, &self.fc1.a])?);
            npu_t(timing, &k_fc1, npu.gemm(&self.fc1, &w[2])?);
            npu_t(timing, &k_fc2, npu.gemm(&self.fc2, &w[3])?);
            npu_t(timing, &k_ln, npu.op(&k_ln, &[xb_, &self.fc2.c, xa, &self.qkv.a])?);
        }
        let t1 = Instant::now();
        let out = pull(xa, rows * dim)?;
        let mut y = vec![0f32; rows * dim];
        par_rows(&mut y, dim, |r0, piece| {
            for (i, v) in piece.iter_mut().enumerate() {
                *v = bf16_to_f32(out[r0 * dim + i]);
            }
        });
        timing.add(&format!("{}_out", self.p), t1.elapsed());
        Ok(y)
    }
}
