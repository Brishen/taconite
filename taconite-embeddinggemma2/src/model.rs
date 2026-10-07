// SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
// SPDX-License-Identifier: Apache-2.0

//! The forward, as `eg2_common.py` runs it with `eg2_npu.NpuBackend`:
//! every projection an `flm.GEMM` dispatch (all of an image's rows in
//! one), the vision attention an MHA dispatch, the rest here in f32.
//!
//! Every RMSNorm that feeds a projection has its weight folded into the
//! packed B, so the host only divides by the RMS there.

use std::time::Instant;

use taconite::Timing;
use taconite::cpu::{self, par_rows};
use taconite_bundle::Store;

use crate::npu::{Buffer, Npu, Rows};
use crate::preprocess::Patches;
use crate::{Config, Error};

const LOG2E: f32 = std::f32::consts::LOG2_E;

struct VLayer {
    qkv: Buffer,
    o: Buffer,
    gu: Buffer,
    down: Vec<Buffer>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    post_attn: Vec<f32>,
    post_ff: Vec<f32>,
}

struct TLayer {
    qkv: Buffer,
    o: Vec<Buffer>,
    gu: Buffer,
    down: Vec<Buffer>,
    pleg: Buffer,
    plep: Buffer,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    post_attn: Vec<f32>,
    post_ff: Vec<f32>,
    post_ple: Vec<f32>,
    scalar: f32,
}

/// The device buffers of one tower.
struct Bufs {
    a: Rows,
    c: Rows,
    gu: Rows,
    p: Vec<Rows>,
    att: Rows,
}

pub struct Model {
    v_patch: Buffer,
    v_ev: Buffer,
    /// [2, P, D]: the x and y position tables (bf16)
    pos: Vec<u16>,
    pos_rows: usize,
    vl: Vec<VLayer>,
    vb: Bufs,
    v_qkv: Rows,
    v_ev_c: Rows,
    t_ple: Buffer,
    ple_norm: Vec<f32>,
    norm: Vec<f32>,
    /// bos, boi, eoi, eos input embeddings
    tok: Vec<f32>,
    proj: Vec<f32>,
    tl: Vec<TLayer>,
    tb: Bufs,
    /// qkv's C, by width (C is dense at row stride N)
    t_qkv: Vec<(usize, Rows)>,
    t_ple_c: Rows,
}

fn f32s(s: &Store, name: &str) -> Result<Vec<f32>, Error> {
    Ok(s.f32(name)?.to_vec())
}

/// Each row of x `[rows, dim]` over its RMS (Gemma's: x / sqrt(mean(x^2)
/// + eps)), times w where given.
fn rms(x: &[f32], dim: usize, eps: f32, w: Option<&[f32]>) -> Vec<f32> {
    let mut y = x.to_vec();
    rms_(&mut y, dim, eps, w);
    y
}

fn rms_(x: &mut [f32], dim: usize, eps: f32, w: Option<&[f32]>) {
    par_rows(x, dim, |_, piece| {
        for row in piece.chunks_mut(dim) {
            rms_row(row, eps, w);
        }
    });
}

#[inline]
fn rms_row(row: &mut [f32], eps: f32, w: Option<&[f32]>) {
    let ms = row.iter().map(|v| v * v).sum::<f32>() / row.len() as f32;
    let r = (ms + eps).powf(-0.5);
    match w {
        Some(w) => row.iter_mut().zip(w).for_each(|(v, &k)| *v *= r * k),
        None => row.iter_mut().for_each(|v| *v *= r),
    }
}

/// x += rms(h) * w, rows of `dim`.
fn add_normed(x: &mut [f32], h: &[f32], dim: usize, eps: f32, w: &[f32]) {
    let mut h = h.to_vec();
    rms_(&mut h, dim, eps, Some(w));
    cpu::add_(x, &h);
}

/// Rotates x by cos / sin, half-split (`rotate_half`) over its length.
#[inline]
fn rope_half(x: &mut [f32], cos: &[f32], sin: &[f32]) {
    let h = x.len() / 2;
    for j in 0..h {
        let (a, b) = (x[j], x[j + h]);
        x[j] = a * cos[j] - b * sin[j];
        x[j + h] = b * cos[j + h] + a * sin[j + h];
    }
}

impl Model {
    pub fn load(c: &Config, s: &Store, npu: &Npu) -> Result<Self, Error> {
        let up = |name: &str| npu.upload(s.bytes(name)?);
        let mut vl = Vec::with_capacity(c.v_layers);
        let v_slices = c.v_i / c.v_d;
        for i in 0..c.v_layers {
            let h = |k: &str| f32s(s, &format!("v{i}.{k}"));
            vl.push(VLayer {
                qkv: up(&format!("v{i}.qkv"))?,
                o: up(&format!("v{i}.o"))?,
                gu: up(&format!("v{i}.gu"))?,
                down: (0..v_slices).map(|j| up(&format!("v{i}.down{j}"))).collect::<Result<_, _>>()?,
                q_norm: h("q_norm")?,
                k_norm: h("k_norm")?,
                post_attn: h("post_attn")?,
                post_ff: h("post_ff")?,
            });
        }
        let scalars = f32s(s, "t.layer_scalar")?;
        let t_slices = c.i / c.d;
        let mut tl = Vec::with_capacity(c.layers);
        for i in 0..c.layers {
            let h = |k: &str| f32s(s, &format!("t{i}.{k}"));
            let n_o = c.heads * c.hd[i] / c.d;
            tl.push(TLayer {
                qkv: up(&format!("t{i}.qkv"))?,
                o: (0..n_o).map(|j| up(&format!("t{i}.o{j}"))).collect::<Result<_, _>>()?,
                gu: up(&format!("t{i}.gu"))?,
                down: (0..t_slices).map(|j| up(&format!("t{i}.down{j}"))).collect::<Result<_, _>>()?,
                pleg: up(&format!("t{i}.pleg"))?,
                plep: up(&format!("t{i}.plep"))?,
                q_norm: h("q_norm")?,
                k_norm: h("k_norm")?,
                post_attn: h("post_attn")?,
                post_ff: h("post_ff")?,
                post_ple: h("post_ple")?,
                scalar: scalars[i],
            });
        }
        let (vr, tr) = (c.v_rows, c.t_rows);
        let v_gu = npu.gemm("v_gu")?.c_stride;
        let t_gu = npu.gemm("t_gu")?.c_stride;
        let vb = Bufs {
            a: npu.rows(vr, c.v_d)?,
            c: npu.rows(vr, c.v_d)?,
            gu: npu.rows(vr + 1, v_gu)?,
            p: (0..v_slices).map(|_| npu.rows(vr, c.v_d)).collect::<Result<_, _>>()?,
            att: npu.rows(vr, 3 * c.v_d)?,
        };
        let tb = Bufs {
            a: npu.rows(tr, c.d)?,
            c: npu.rows(tr, c.d)?,
            gu: npu.rows(tr + 1, t_gu)?,
            p: (0..t_slices.max(c.o_width / c.d)).map(|_| npu.rows(tr, c.d)).collect::<Result<_, _>>()?,
            att: npu.rows(tr + 1, c.o_width)?,
        };
        let mut widths: Vec<usize> = (0..c.layers).map(|i| (c.heads + 2 * c.kv_heads[i]) * c.hd[i]).collect();
        widths.sort();
        widths.dedup();
        let pos_shape = s.shape("v.pos")?.to_vec();
        Ok(Model {
            v_patch: up("v.patch")?,
            v_ev: up("v.ev")?,
            pos: s.bf16("v.pos")?.to_vec(),
            pos_rows: pos_shape[1],
            vl,
            vb,
            v_qkv: npu.rows(vr, 3 * c.v_d)?,
            v_ev_c: npu.rows(c.max_soft_tokens.div_ceil(256) * 256, c.d)?,
            t_ple: up("t.ple")?,
            ple_norm: f32s(s, "t.ple_norm")?,
            norm: f32s(s, "t.norm")?,
            tok: f32s(s, "t.tok")?,
            proj: f32s(s, "t.proj")?,
            tl,
            tb,
            t_qkv: widths.into_iter().map(|w| Ok((w, npu.rows(tr, w)?))).collect::<Result<_, Error>>()?,
            t_ple_c: npu.rows(tr, c.layers * c.ple)?,
        })
    }

    /// An image's patches -> its soft tokens `[n / 9, 512]`.
    pub fn vision(&mut self, c: &Config, npu: &Npu, p: &Patches, t: &mut Timing) -> Result<Vec<f32>, Error> {
        let (d, h, hd, eps) = (c.v_d, c.v_heads, c.v_hd, c.v_eps);
        let n = p.positions.len();
        if n > c.v_rows || n > c.max_patches() {
            return Err(Error::Input(format!("{n} patches (at most {})", c.max_patches())));
        }
        if p.gh % c.pool != 0 || p.gw % c.pool != 0 {
            return Err(Error::Input(format!("a {} x {} grid does not pool by {}", p.gh, p.gw, c.pool)));
        }
        let t0 = Instant::now();
        let a: Vec<f32> = p.pixels.iter().map(|&v| 2.0 * (v - 0.5)).collect();
        self.vb.a.put(&a, d)?;
        t.add("v.host", t0.elapsed());
        npu.run_gemm("v_patch", &self.vb.a, 0, &self.v_patch, &self.vb.c, n, t)?;
        let mut x = self.vb.c.get(n, d)?;
        let t0 = Instant::now();
        // position embeddings: the x table's row px plus the y table's py
        let (pos, pr) = (&self.pos, self.pos_rows);
        par_rows(&mut x, d, |r0, piece| {
            for (ri, row) in piece.chunks_mut(d).enumerate() {
                let (px, py) = p.positions[r0 + ri];
                let tx = &pos[px * d..(px + 1) * d];
                let ty = &pos[(pr + py) * d..(pr + py + 1) * d];
                for ((v, &a), &b) in row.iter_mut().zip(tx).zip(ty) {
                    *v += taconite::bf16_to_f32(a) + taconite::bf16_to_f32(b);
                }
            }
        });
        // the axial RoPE: x's frequencies on a head's first half, y's on
        // the second, each half rotated on its own
        let half = hd / 2;
        let inv: Vec<f32> = (0..half / 2).map(|j| 1.0 / c.v_theta.powf((2 * j) as f32 / half as f32)).collect();
        let (mut cos, mut sin) = (vec![0f32; n * hd], vec![0f32; n * hd]);
        for (r, &(px, py)) in p.positions.iter().enumerate() {
            for (j, &f) in inv.iter().enumerate() {
                for (base, pos) in [(0, px), (half, py)] {
                    let (s, co) = (pos as f32 * f).sin_cos();
                    for k in [base + j, base + half / 2 + j] {
                        cos[r * hd + k] = co;
                        sin[r * hd + k] = s;
                    }
                }
            }
        }
        t.add("v.host", t0.elapsed());

        for li in 0..self.vl.len() {
            let t0 = Instant::now();
            let xn = rms(&x, d, eps, None);
            self.vb.a.put(&xn, d)?;
            t.add("v.host", t0.elapsed());
            let l = &self.vl[li];
            npu.run_gemm("v_qkv", &self.vb.a, 0, &l.qkv, &self.v_qkv, n, t)?;
            let mut qkv = self.v_qkv.get(n, 3 * d)?;
            let t0 = Instant::now();
            let (qn, kn) = (&l.q_norm, &l.k_norm);
            let (cos, sin) = (&cos, &sin);
            par_rows(&mut qkv, 3 * d, |r0, piece| {
                for (ri, row) in piece.chunks_mut(3 * d).enumerate() {
                    let r = r0 + ri;
                    let (cs, sn) = (&cos[r * hd..(r + 1) * hd], &sin[r * hd..(r + 1) * hd]);
                    for hh in 0..h {
                        let q = &mut row[hh * hd..(hh + 1) * hd];
                        rms_row(q, eps, Some(qn));
                        rope_half(&mut q[..half], &cs[..half], &sn[..half]);
                        rope_half(&mut q[half..], &cs[half..], &sn[half..]);
                        q.iter_mut().for_each(|v| *v *= LOG2E);
                        let k = &mut row[d + hh * hd..d + (hh + 1) * hd];
                        rms_row(k, eps, Some(kn));
                        rope_half(&mut k[..half], &cs[..half], &sn[..half]);
                        rope_half(&mut k[half..], &cs[half..], &sn[half..]);
                        rms_row(&mut row[2 * d + hh * hd..2 * d + (hh + 1) * hd], eps, None);
                    }
                }
            });
            self.vb.att.put(&qkv, 3 * d)?;
            t.add("v.attn", t0.elapsed());
            // the MHA writes o into A, where the o projection reads it
            npu.run_mha(&self.vb.att, &self.vb.a, n, c.no_window, t)?;
            let l = &self.vl[li];
            npu.run_gemm("v_o", &self.vb.a, 0, &l.o, &self.vb.c, n, t)?;
            let o = self.vb.c.get(n, d)?;
            let t0 = Instant::now();
            add_normed(&mut x, &o, d, eps, &l.post_attn);
            let xn = rms(&x, d, eps, None);
            self.vb.a.put(&xn, d)?;
            t.add("v.host", t0.elapsed());
            let hsum = self.mlp(npu, "v", li, n, d, t)?;
            let t0 = Instant::now();
            add_normed(&mut x, &hsum, d, eps, &self.vl[li].post_ff);
            t.add("v.host", t0.elapsed());
        }

        // 3 x 3 blocks of patches -> soft tokens, times sqrt(D), RMS
        let t0 = Instant::now();
        let k = c.pool;
        let ns = n / (k * k);
        let mut pooled = vec![0f32; ns * d];
        for (r, &(px, py)) in p.positions.iter().enumerate() {
            let s = px / k + (p.gw / k) * (py / k);
            for (o, &v) in pooled[s * d..(s + 1) * d].iter_mut().zip(&x[r * d..(r + 1) * d]) {
                *o += v;
            }
        }
        let scale = (d as f32).sqrt() / (k * k) as f32;
        pooled.iter_mut().for_each(|v| *v *= scale);
        rms_(&mut pooled, d, eps, None);
        self.vb.a.put(&pooled, d)?;
        t.add("v.host", t0.elapsed());
        npu.run_gemm("v_ev", &self.vb.a, 0, &self.v_ev, &self.v_ev_c, ns, t)?;
        self.v_ev_c.get(ns, c.d)
    }

    /// gelu(x Wg) * (x Wu) -> down, x already in the tower's A: the GeGLU
    /// GEMM, then down as K slices of its output read in place, summed.
    fn mlp(&mut self, npu: &Npu, tower: &str, li: usize, n: usize, d: usize, t: &mut Timing) -> Result<Vec<f32>, Error> {
        let (b, gu, down) = if tower == "v" {
            (&self.vb, &self.vl[li].gu, &self.vl[li].down)
        } else {
            (&self.tb, &self.tl[li].gu, &self.tl[li].down)
        };
        npu.run_gemm(&format!("{tower}_gu"), &b.a, 0, gu, &b.gu, n, t)?;
        for (s, w) in down.iter().enumerate() {
            npu.run_gemm(&format!("{tower}_down"), &b.gu, s * d, w, &b.p[s], n, t)?;
        }
        let mut out = b.p[0].get(n, d)?;
        for p in &b.p[1..down.len()] {
            cpu::add_(&mut out, &p.get(n, d)?);
        }
        Ok(out)
    }

    /// Soft tokens `[s, 512]` -> the encoder's input embeddings of
    /// `<bos> <|image> soft... <image|> <eos>`.
    pub fn image_sequence(&self, c: &Config, soft: &[f32]) -> Vec<f32> {
        let d = c.d;
        let tok = |j: usize| &self.tok[j * d..(j + 1) * d];
        let mut x = Vec::with_capacity(soft.len() + 4 * d);
        x.extend_from_slice(tok(0));
        x.extend_from_slice(tok(1));
        x.extend_from_slice(soft);
        x.extend_from_slice(tok(2));
        x.extend_from_slice(tok(3));
        x
    }

    /// The text encoder over input embeddings x `[T, 512]` -> the unit
    /// `[768]` embedding (mean over the tokens, projected, normalized).
    pub fn text(&mut self, c: &Config, npu: &Npu, x: Vec<f32>, t: &mut Timing) -> Result<Vec<f32>, Error> {
        let (d, eps, lp, nl) = (c.d, c.eps, c.ple, c.layers);
        let mut x = x;
        let t0 = Instant::now();
        let n = x.len() / d;
        if n == 0 || n > c.t_rows {
            return Err(Error::Input(format!("{n} tokens (1..={})", c.t_rows)));
        }
        self.tb.a.put(&x, d)?;
        t.add("t.host", t0.elapsed());
        npu.run_gemm("t_ple", &self.tb.a, 0, &self.t_ple, &self.t_ple_c, n, t)?;
        let mut ple = self.t_ple_c.get(n, nl * lp)?;
        let t0 = Instant::now();
        let s = (d as f32).powf(-0.5);
        ple.iter_mut().for_each(|v| *v *= s);
        rms_(&mut ple, lp, eps, Some(&self.ple_norm));
        t.add("t.host", t0.elapsed());

        for li in 0..nl {
            let (h, hd, kvh) = (c.heads, c.hd[li], c.kv_heads[li]);
            let w = (h + 2 * kvh) * hd;
            let t0 = Instant::now();
            let xn = rms(&x, d, eps, None);
            self.tb.a.put(&xn, d)?;
            t.add("t.host", t0.elapsed());
            let qkv_c = &self.t_qkv.iter().find(|(cw, _)| *cw == w).ok_or_else(|| Error::Bundle(format!("no qkv{w}")))?.1;
            npu.run_gemm(&format!("t_qkv{w}"), &self.tb.a, 0, &self.tl[li].qkv, qkv_c, n, t)?;
            let qkv = qkv_c.get(n, w)?;
            let t0 = Instant::now();
            let o = self.text_attention(c, li, &qkv, n);
            self.tb.att.put(&o, h * hd)?;
            t.add("t.attn", t0.elapsed());
            let l = &self.tl[li];
            for (s, wo) in l.o.iter().enumerate() {
                npu.run_gemm("t_o", &self.tb.att, s * d, wo, &self.tb.p[s], n, t)?;
            }
            let mut o = self.tb.p[0].get(n, d)?;
            for p in &self.tb.p[1..l.o.len()] {
                cpu::add_(&mut o, &p.get(n, d)?);
            }
            let t0 = Instant::now();
            add_normed(&mut x, &o, d, eps, &l.post_attn);
            let xn = rms(&x, d, eps, None);
            self.tb.a.put(&xn, d)?;
            t.add("t.host", t0.elapsed());
            let hsum = self.mlp(npu, "t", li, n, d, t)?;
            let t0 = Instant::now();
            add_normed(&mut x, &hsum, d, eps, &self.tl[li].post_ff);
            self.tb.a.put(&x, d)?;
            t.add("t.host", t0.elapsed());
            // the per-layer-input block: gelu(x Wg) (the GEMM's epilogue)
            // times this layer's slice, projected back
            let l = &self.tl[li];
            npu.run_gemm("t_pleg", &self.tb.a, 0, &l.pleg, &self.tb.c, n, t)?;
            let mut g = self.tb.c.get(n, d)?;
            let t0 = Instant::now();
            for (r, row) in g.chunks_mut(d).enumerate() {
                let sl = &ple[(r * nl + li) * lp..(r * nl + li + 1) * lp];
                row.iter_mut().zip(sl).for_each(|(v, &k)| *v *= k);
            }
            self.tb.a.put(&g, d)?;
            t.add("t.host", t0.elapsed());
            npu.run_gemm("t_plep", &self.tb.a, 0, &l.plep, &self.tb.c, n, t)?;
            let g = self.tb.c.get(n, d)?;
            let t0 = Instant::now();
            add_normed(&mut x, &g, d, eps, &l.post_ple);
            if l.scalar != 1.0 {
                x.iter_mut().for_each(|v| *v *= l.scalar);
            }
            t.add("t.host", t0.elapsed());
        }

        // final norm, mean over the tokens, projection, L2 normalization
        let t0 = Instant::now();
        rms_(&mut x, d, eps, Some(&self.norm));
        let mut mean = vec![0f32; d];
        for row in x.chunks(d) {
            cpu::add_(&mut mean, row);
        }
        mean.iter_mut().for_each(|v| *v /= n as f32);
        let mut e: Vec<f32> = self.proj.chunks(d).map(|w| cpu::dot(w, &mean)).collect();
        let norm = e.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
        e.iter_mut().for_each(|v| *v /= norm);
        t.add("t.host", t0.elapsed());
        Ok(e)
    }

    /// Layer `li`'s attention over its qkv `[n, (H + 2 KV) hd]`: q / k / v
    /// norms, 1D RoPE, bidirectional attention with score scale 1, a
    /// sliding layer's keys within |i - j| <= window -> `[n, H hd]`.
    fn text_attention(&self, c: &Config, li: usize, qkv: &[f32], n: usize) -> Vec<f32> {
        let (h, hd, kvh, eps) = (c.heads, c.hd[li], c.kv_heads[li], c.eps);
        let w = (h + 2 * kvh) * hd;
        let l = &self.tl[li];
        let theta = if c.global[li] { c.theta_g } else { c.theta_s };
        let inv: Vec<f32> = (0..hd / 2).map(|j| 1.0 / theta.powf((2 * j) as f32 / hd as f32)).collect();
        let (dim, kdim) = (h * hd, kvh * hd);
        let (mut q, mut k, mut v) = (vec![0f32; n * dim], vec![0f32; n * kdim], vec![0f32; n * kdim]);
        let mut cs = vec![0f32; n * hd];
        let mut sn = vec![0f32; n * hd];
        for r in 0..n {
            for (j, &f) in inv.iter().enumerate() {
                let (s, co) = (r as f32 * f).sin_cos();
                for k in [j, j + hd / 2] {
                    cs[r * hd + k] = co;
                    sn[r * hd + k] = s;
                }
            }
        }
        let qn = &l.q_norm[..];
        par_rows(&mut q, dim, |r0, piece| {
            for (ri, row) in piece.chunks_mut(dim).enumerate() {
                let r = r0 + ri;
                let (c_, s_) = (&cs[r * hd..(r + 1) * hd], &sn[r * hd..(r + 1) * hd]);
                for (hh, dst) in row.chunks_mut(hd).enumerate() {
                    dst.copy_from_slice(&qkv[r * w + hh * hd..r * w + (hh + 1) * hd]);
                    rms_row(dst, eps, Some(qn));
                    rope_half(dst, c_, s_);
                }
            }
        });
        for r in 0..n {
            let (c_, s_) = (&cs[r * hd..(r + 1) * hd], &sn[r * hd..(r + 1) * hd]);
            for kh in 0..kvh {
                let kk = &mut k[r * kdim + kh * hd..r * kdim + (kh + 1) * hd];
                kk.copy_from_slice(&qkv[r * w + (h + kh) * hd..r * w + (h + kh + 1) * hd]);
                rms_row(kk, eps, Some(&l.k_norm));
                rope_half(kk, c_, s_);
                let vv = &mut v[r * kdim + kh * hd..r * kdim + (kh + 1) * hd];
                vv.copy_from_slice(&qkv[r * w + (h + kvh + kh) * hd..r * w + (h + kvh + kh + 1) * hd]);
                rms_row(vv, eps, None);
            }
        }
        // a sliding layer's window, where it cuts anything off
        let win = if !c.global[li] && n > c.window + 1 { c.window } else { n };
        let group = h / kvh;
        let mut out = vec![0f32; n * dim];
        par_rows(&mut out, dim, |r0, piece| {
            let mut s = vec![0f32; n];
            for (ri, orow) in piece.chunks_mut(dim).enumerate() {
                let i = r0 + ri;
                let (lo, hi) = (i.saturating_sub(win), (i + win + 1).min(n));
                for hh in 0..h {
                    let kh = hh / group;
                    let qh = &q[i * dim + hh * hd..i * dim + (hh + 1) * hd];
                    let sc = &mut s[..hi - lo];
                    for (j, sj) in (lo..hi).zip(sc.iter_mut()) {
                        *sj = cpu::dot(qh, &k[j * kdim + kh * hd..j * kdim + (kh + 1) * hd]);
                    }
                    cpu::softmax_(sc);
                    let oh = &mut orow[hh * hd..(hh + 1) * hd];
                    oh.fill(0.0);
                    for (j, &p) in (lo..hi).zip(sc.iter()) {
                        for (o, &x) in oh.iter_mut().zip(&v[j * kdim + kh * hd..j * kdim + (kh + 1) * hd]) {
                            *o += p * x;
                        }
                    }
                }
            }
        });
        out
    }
}
