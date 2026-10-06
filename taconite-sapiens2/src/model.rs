// SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
// SPDX-License-Identifier: Apache-2.0

//! The forward, as `sapiens2_common.py` runs it with
//! `sapiens2_npu.NpuBackend`: every projection and convolution an
//! `flm.GEMM` dispatch (all of its rows in one), the attention an MHA
//! dispatch, the rest here in f32.
//!
//! Every RMSNorm that feeds a projection has its weight folded into the
//! packed B (the final norm's into the first transposed convolution), so
//! the host only divides by the RMS.
//!
//! A transposed convolution (kernel 4, stride 2, padding 1) runs as one
//! GEMM over the 2 x 2 windows of its input padded by 1: the host writes
//! the input as row pairs, `X2[y][x] = [xp[y][x], xp[y + 1][x]]`, and the
//! GEMM reads a window as 4 C contiguous values at row stride 2 C (rows
//! overlap); its N holds the four output phases, which the host scatters
//! back while it applies the InstanceNorm.

use std::time::Instant;

use taconite::cpu::{self, par_rows};
use taconite::{Timing, bf16_to_f32, f32_to_bf16};
use taconite_bundle::Store;

use crate::npu::{Buffer, Flat, Npu};
use crate::{Config, Error};

const LOG2E: f32 = std::f32::consts::LOG2_E;

struct Layer {
    qkv: Buffer,
    o: Buffer,
    gu: Buffer,
    down: Buffer,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
}

pub struct Model {
    patch: Buffer,
    prefix: Vec<f32>,
    layers: Vec<Layer>,
    deconv: Vec<Buffer>,
    convs: Vec<Buffer>,
    pred: Buffer,
    cos: Vec<f32>,
    sin: Vec<f32>,
    /// A of most GEMMs
    a: Flat,
    /// C of most GEMMs
    c: Flat,
    /// the SwiGLU output (down's A)
    gu: Flat,
    /// qkv's C, by width (C is dense at row stride N)
    qkv: Vec<(usize, Flat)>,
    /// q / k / v staged for the attention
    stage: Flat,
    /// the attention output (o's A)
    ao: Flat,
}

fn f32s(s: &Store, name: &str) -> Result<Vec<f32>, Error> {
    Ok(s.f32(name)?.to_vec())
}

#[inline]
fn rms_row(row: &mut [f32], eps: f32, w: Option<&[f32]>) {
    let ms = row.iter().map(|v| v * v).sum::<f32>() / row.len() as f32;
    let r = 1.0 / (ms + eps).sqrt();
    match w {
        Some(w) => row.iter_mut().zip(w).for_each(|(v, &k)| *v = *v * r * k),
        None => row.iter_mut().for_each(|v| *v *= r),
    }
}

/// Each row of x `[rows, dim]` over its RMS.
fn rms(x: &[f32], dim: usize, eps: f32) -> Vec<f32> {
    let mut y = x.to_vec();
    par_rows(&mut y, dim, |_, piece| piece.chunks_mut(dim).for_each(|r| rms_row(r, eps, None)));
    y
}

/// Each row of x `[rows, dim]` over its RMS, as bf16 into `dst`.
fn put_rms(dst: &mut Flat, x: &[f32], dim: usize, eps: f32) -> Result<(), Error> {
    dst.fill_rows(x.len() / dim, dim, |r, out| {
        let row = &x[r * dim..(r + 1) * dim];
        let ms = row.iter().map(|v| v * v).sum::<f32>() / dim as f32;
        let s = 1.0 / (ms + eps).sqrt();
        out.iter_mut().zip(row).for_each(|(o, &v)| *o = f32_to_bf16(v * s));
    })
}

/// x += the first `x.len()` values of `c`.
fn add_from(x: &mut [f32], c: &Flat) -> Result<(), Error> {
    const P: usize = 1 << 14;
    let src = c.bits(x.len())?;
    par_rows(x, P, |r0, piece| {
        piece.iter_mut().zip(&src[r0 * P..]).for_each(|(v, &b)| *v += bf16_to_f32(b));
    });
    Ok(())
}

/// Rotates x by cos / sin, half-split (`rotate_half`).
#[inline]
fn rope(x: &mut [f32], cos: &[f32], sin: &[f32]) {
    let h = x.len() / 2;
    for j in 0..h {
        let (a, b) = (x[j], x[j + h]);
        x[j] = a * cos[j] - b * sin[j];
        x[j + h] = b * cos[j + h] + a * sin[j + h];
    }
}

/// Per-channel mean and 1 / std (biased) of y `[rows, c]`, summed in f64.
fn col_stats(y: &[f32], c: usize, eps: f32) -> (Vec<f32>, Vec<f32>) {
    let rows = y.len() / c;
    let nt = cpu::threads().clamp(1, 32);
    let per = rows.div_ceil(nt);
    let sums = |f: &(dyn Fn(usize, f32) -> f64 + Sync)| -> Vec<f64> {
        let parts: Vec<Vec<f64>> = std::thread::scope(|s| {
            let hs: Vec<_> = (0..nt)
                .map(|t| {
                    s.spawn(move || {
                        let mut acc = vec![0f64; c];
                        for r in (t * per)..((t + 1) * per).min(rows) {
                            for (j, (a, &v)) in acc.iter_mut().zip(&y[r * c..(r + 1) * c]).enumerate() {
                                *a += f(j, v);
                            }
                        }
                        acc
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let mut tot = vec![0f64; c];
        for p in parts {
            tot.iter_mut().zip(p).for_each(|(t, v)| *t += v);
        }
        tot
    };
    let mean: Vec<f64> = sums(&|_, v| v as f64).into_iter().map(|s| s / rows as f64).collect();
    let var: Vec<f64> = sums(&|j, v| (v as f64 - mean[j]).powi(2)).into_iter().map(|s| s / rows as f64).collect();
    let rstd = var.iter().map(|&v| (1.0 / (v + eps as f64).sqrt()) as f32).collect();
    (mean.into_iter().map(|m| m as f32).collect(), rstd)
}

/// y `[rows, c]` -> silu(InstanceNorm(y)), in place.
fn inorm_silu(y: &mut [f32], c: usize, eps: f32) {
    let (mean, rstd) = col_stats(y, c, eps);
    par_rows(y, c, |_, piece| {
        for row in piece.chunks_mut(c) {
            for ((v, &m), &r) in row.iter_mut().zip(&mean).zip(&rstd) {
                let z = (*v - m) * r;
                *v = z / (1.0 + (-z).exp());
            }
        }
    });
}

impl Model {
    pub fn load(c: &Config, s: &Store, npu: &Npu) -> Result<Self, Error> {
        let up = |name: &str| npu.upload(s.bytes(name)?);
        let mut layers = Vec::with_capacity(c.layers);
        for i in 0..c.layers {
            layers.push(Layer {
                qkv: up(&format!("{i}.qkv"))?,
                o: up(&format!("{i}.o"))?,
                gu: up(&format!("{i}.gu"))?,
                down: up(&format!("{i}.down"))?,
                q_norm: f32s(s, &format!("{i}.q_norm"))?,
                k_norm: f32s(s, &format!("{i}.k_norm"))?,
            });
        }
        let g = |k: &str| npu.gemm(k);
        let mut widths: Vec<usize> = (0..c.layers).map(|i| c.qkv_width(i)).collect();
        widths.sort();
        widths.dedup();
        let mut a_keys: Vec<String> = vec!["patch".into(), "gu".into(), "pred".into()];
        a_keys.extend(widths.iter().map(|w| format!("qkv{w}")));
        a_keys.extend((0..c.up.len()).map(|j| format!("d{j}")));
        a_keys.extend((0..c.convs.len()).map(|j| format!("c{j}")));
        let a_elems = a_keys.iter().map(|k| g(k).map(|s| s.a_elems)).collect::<Result<Vec<_>, _>>()?;
        let c_elems = npu.gemms.values().map(|s| s.c_elems).max().unwrap_or(0);
        let [qe, ke, ve, oe] = npu.mha;
        Ok(Model {
            patch: up("patch")?,
            prefix: f32s(s, "prefix")?,
            layers,
            deconv: (0..c.up.len()).map(|j| up(&format!("d{j}"))).collect::<Result<_, _>>()?,
            convs: (0..c.convs.len()).map(|j| up(&format!("c{j}"))).collect::<Result<_, _>>()?,
            pred: up("pred")?,
            cos: f32s(s, "rope.cos")?,
            sin: f32s(s, "rope.sin")?,
            a: npu.flat(a_elems.into_iter().max().unwrap_or(0))?,
            c: npu.flat(c_elems)?,
            gu: npu.flat(g("gu")?.c_elems.max(g("down")?.a_elems))?,
            qkv: widths
                .iter()
                .map(|&w| Ok((w, npu.flat(g(&format!("qkv{w}"))?.c_elems)?)))
                .collect::<Result<_, Error>>()?,
            stage: npu.flat(qe.max(ke).max(ve))?,
            ao: npu.flat(oe.max(g("o")?.a_elems))?,
        })
    }

    /// pixel_values `[3, H, W]` -> the normalized patch features `[P, D]`
    /// (before the final norm's weight, which the head holds).
    pub fn backbone(&mut self, c: &Config, npu: &Npu, pixels: &[f32], t: &mut Timing) -> Result<Vec<f32>, Error> {
        let (d, h, hd, eps, p) = (c.d, c.heads, c.hd, c.eps, c.patch);
        let (gh, gw) = (c.gh(), c.gw());
        let np = gh * gw;
        let pre = c.prefix();
        let tt = pre + np;
        let t0 = Instant::now();
        // patches in the conv weight's (c, ky, kx) order
        let kp = 3 * p * p;
        let (hh, ww) = (c.h, c.w);
        self.a.fill_rows(np, kp, |r, row| {
            let (py, px) = (r / gw, r % gw);
            for ch in 0..3 {
                for ky in 0..p {
                    let src = &pixels[ch * hh * ww + (py * p + ky) * ww + px * p..][..p];
                    for (o, &v) in row[(ch * p + ky) * p..][..p].iter_mut().zip(src) {
                        *o = f32_to_bf16(v);
                    }
                }
            }
        })?;
        t.add("host", t0.elapsed());
        npu.run_gemm("patch", &self.a, &self.patch, &self.c, t)?;
        let mut x = self.prefix.clone();
        x.extend(self.c.get(np * d)?);

        for li in 0..c.layers {
            let t0 = Instant::now();
            put_rms(&mut self.a, &x, d, eps)?;
            t.add("host", t0.elapsed());
            let width = c.qkv_width(li);
            let qb = &self.qkv.iter().find(|(w, _)| *w == width).ok_or_else(|| Error::Bundle(format!("qkv{width}")))?.1;
            npu.run_gemm(&format!("qkv{width}"), &self.a, &self.layers[li].qkv, qb, t)?;
            let qkv = qb.bits(tt * width)?;
            let t0 = Instant::now();
            let (qn, kn) = (&self.layers[li].q_norm[..], &self.layers[li].k_norm[..]);
            let (kv_heads, rep) = (c.kv_heads[li], h / c.kv_heads[li]);
            let kv = kv_heads * hd;
            let (cos, sin) = (&self.cos, &self.sin);
            let qs = LOG2E / (hd as f32).sqrt();
            // a row of [q * log2(e) / sqrt(hd) | k | v], q / k normed and
            // rotated (patch tokens), k / v repeated to every query head
            self.stage.fill_rows(tt, 3 * d, |r, out| {
                let src = &qkv[r * width..(r + 1) * width];
                let cs = (r >= pre).then(|| (&cos[(r - pre) * hd..][..hd], &sin[(r - pre) * hd..][..hd]));
                let mut head = vec![0f32; hd];
                let mut emit = |dst: &mut [u16], from: &[u16], w: Option<&[f32]>, scale: f32| {
                    for (o, &b) in head.iter_mut().zip(from) {
                        *o = bf16_to_f32(b);
                    }
                    if let Some(w) = w {
                        rms_row(&mut head, eps, Some(w));
                        if let Some((cs, sn)) = cs {
                            rope(&mut head, cs, sn);
                        }
                    }
                    for (o, &v) in dst.iter_mut().zip(&head) {
                        *o = f32_to_bf16(v * scale);
                    }
                };
                let (q_out, rest) = out.split_at_mut(d);
                let (k_out, v_out) = rest.split_at_mut(d);
                for hh in 0..h {
                    emit(&mut q_out[hh * hd..(hh + 1) * hd], &src[hh * hd..(hh + 1) * hd], Some(qn), qs);
                }
                for kh in 0..kv_heads {
                    let k0 = kh * rep * hd;
                    emit(&mut k_out[k0..k0 + hd], &src[d + kh * hd..d + (kh + 1) * hd], Some(kn), 1.0);
                    v_out[k0..k0 + hd].copy_from_slice(&src[d + kv + kh * hd..d + kv + (kh + 1) * hd]);
                    for j in 1..rep {
                        k_out.copy_within(k0..k0 + hd, k0 + j * hd);
                        v_out.copy_within(k0..k0 + hd, k0 + j * hd);
                    }
                }
            })?;
            t.add("host.attn", t0.elapsed());
            npu.run_mha(&self.stage, &self.ao, t)?;
            npu.run_gemm("o", &self.ao, &self.layers[li].o, &self.c, t)?;
            let t0 = Instant::now();
            add_from(&mut x, &self.c)?;
            put_rms(&mut self.a, &x, d, eps)?;
            t.add("host", t0.elapsed());
            npu.run_gemm("gu", &self.a, &self.layers[li].gu, &self.gu, t)?;
            npu.run_gemm("down", &self.gu, &self.layers[li].down, &self.c, t)?;
            let t0 = Instant::now();
            add_from(&mut x, &self.c)?;
            t.add("host", t0.elapsed());
        }
        let t0 = Instant::now();
        let f = rms(&x[pre * d..], d, eps);
        t.add("host", t0.elapsed());
        Ok(f)
    }

    /// The normalized patch features `[P, D]` -> heatmaps `[K, h * w]`.
    pub fn head(&mut self, c: &Config, npu: &Npu, f: &[f32], t: &mut Timing) -> Result<Vec<f32>, Error> {
        let (mut hh, mut ww) = (c.gh(), c.gw());
        let mut x = f.to_vec();
        let mut cin = c.d;
        for (j, &co) in c.up.iter().enumerate() {
            let t0 = Instant::now();
            // X2 [(H + 1), (W + 2), 2 C]: xp[y][x] beside xp[y + 1][x]
            let (xr, h0, w0) = (&x, hh, ww);
            self.a.fill_rows((h0 + 1) * (w0 + 2), 2 * cin, |r, row| {
                let (y, xx) = (r / (w0 + 2), r % (w0 + 2));
                for (part, dst) in row.chunks_mut(cin).enumerate() {
                    let (py, px) = (y + part, xx);
                    if (1..=h0).contains(&py) && (1..=w0).contains(&px) {
                        let src = &xr[((py - 1) * w0 + px - 1) * cin..][..cin];
                        dst.iter_mut().zip(src).for_each(|(o, &v)| *o = f32_to_bf16(v));
                    } else {
                        dst.fill(0);
                    }
                }
            })?;
            t.add("host.head", t0.elapsed());
            let key = format!("d{j}");
            npu.run_gemm(&key, &self.a, &self.deconv[j], &self.c, t)?;
            let g = npu.gemm(&key)?;
            let cb = self.c.bits(g.rows * g.n)?;
            let t0 = Instant::now();
            // phase (a, b) of window (sy, sx) -> output (2 (sy - a) + a, ...)
            let (oh, ow) = (2 * h0, 2 * w0);
            let mut y = vec![0f32; oh * ow * co];
            par_rows(&mut y, co, |r0, piece| {
                for (ri, row) in piece.chunks_mut(co).enumerate() {
                    let (oy, ox) = ((r0 + ri) / ow, (r0 + ri) % ow);
                    let (a, b) = (oy % 2, ox % 2);
                    let (sy, sx) = (oy / 2 + a, ox / 2 + b);
                    let src = &cb[(sy * (w0 + 2) + sx) * g.n + (2 * a + b) * co..][..co];
                    row.iter_mut().zip(src).for_each(|(o, &v)| *o = bf16_to_f32(v));
                }
            });
            inorm_silu(&mut y, co, c.in_eps);
            t.add("host.head", t0.elapsed());
            (hh, ww, x, cin) = (oh, ow, y, co);
        }
        let px = hh * ww;
        for (j, &co) in c.convs.iter().enumerate() {
            let t0 = Instant::now();
            self.a.put(&x)?;
            t.add("host.head", t0.elapsed());
            let key = format!("c{j}");
            npu.run_gemm(&key, &self.a, &self.convs[j], &self.c, t)?;
            let stride = npu.gemm(&key)?.c_stride;
            let mut y = self.c.get(px * stride)?;
            let t0 = Instant::now();
            if stride != co {
                y = y.chunks(stride).flat_map(|r| r[..co].iter().copied()).collect();
            }
            inorm_silu(&mut y, co, c.in_eps);
            x = y;
            t.add("host.head", t0.elapsed());
        }
        let t0 = Instant::now();
        self.a.put(&x)?;
        t.add("host.head", t0.elapsed());
        npu.run_gemm("pred", &self.a, &self.pred, &self.c, t)?;
        let n = npu.gemm("pred")?.c_stride;
        let cb = self.c.bits(px * n)?;
        let t0 = Instant::now();
        let mut hm = vec![0f32; c.k * px];
        par_rows(&mut hm, px, |k0, piece| {
            for (ki, map) in piece.chunks_mut(px).enumerate() {
                let k = k0 + ki;
                for (p, o) in map.iter_mut().enumerate() {
                    *o = bf16_to_f32(cb[p * n + k]);
                }
            }
        });
        t.add("host.head", t0.elapsed());
        Ok(hm)
    }
}
