// SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
// SPDX-License-Identifier: Apache-2.0

//! The forward the Python app runs (`reuse_common.py`), host side here and
//! every convolution, projection and Mamba scan on the NPU:
//!
//! * encoder: the 1 x 1 conv (2 -> 64) on the host; the dense block's four
//!   3 x 3 convs as GEMMs over a 256-channel map buffer (`[x2 | x1 | x0 |
//!   input]`, zero-bordered 8 frames and 1 bin), three output pixels a
//!   row, the three time taps side by side in N -- the host sums each
//!   tap's output shifted by its dilation; the (1, 3) stride-(4, 2) conv
//!   as a GEMM over every 4th frame at a row stride of 2 pixels.
//! * 30 x (a bidirectional Mamba along time, then along frequency): the
//!   host lays the tokens out in MambaScan's stream order (the backward
//!   direction time-reversed), MambaScan's in phase writes the token
//!   records, its conv / scan / proj phases follow, the host sums both
//!   directions back, LayerNorm, residual.
//! * decoders: a dense block, the two upsampling convs as GEMMs (their N
//!   the interleaved outputs), the 1 x 1 heads on the host.
//!
//! InstanceNorm + PReLU after every conv run on the host.

use std::collections::HashMap;
use std::time::Instant;

use taconite::Timing;
use taconite::cpu::par_rows;
use taconite::bf16_to_f32;
use taconite_bundle::{DType, Store};

use crate::npu::{Buffer, Flat, Npu, to_bits};
use crate::{Config, Error};

const H: usize = 64;

#[derive(Default)]
struct Bufs {
    s: Option<Flat>,
    c: Option<Flat>,
    a: Option<Flat>,
    x: Option<Flat>,
    rec: Option<Flat>,
    o: Option<Flat>,
}

// the forward's stages as fractions of its time, for `forward_with`
// (measured on mic44k's first pass: the encoder 22% with the kernels'
// first loads, less on later chunks; the 30 blocks 65%; the two decoders
// 13%)
const ENC: f64 = 0.15;
const BLOCKS: f64 = 0.72;

pub struct Model {
    /// host float32 vectors (norms, PReLU slopes, biases, heads)
    vec: HashMap<String, Vec<f32>>,
    /// packed GEMM weights and MambaScan parameter blocks, on the device
    w: HashMap<String, Buffer>,
    bufs: Bufs,
}

/// Where stream position p of a MambaScan shape (seqs, L) comes from:
/// (direction, sequence, token).
fn stream_pos(p: usize, seqs: usize, l: usize) -> (usize, usize, usize) {
    let (j_n, s_n) = (seqs / 16, l / 4);
    let i = p % 4;
    let q = p / 4;
    let r = q % 4;
    let q = q / 4;
    let s = q % s_n;
    let q = q / s_n;
    let j = q % j_n;
    let c = q / j_n;
    (c / 4, ((c % 4) * 4 + r) * j_n + j, s * 4 + i)
}

/// The stream position of (direction, sequence, token).
fn stream_index(d: usize, seq: usize, t: usize, seqs: usize, l: usize) -> usize {
    let (j_n, s_n) = (seqs / 16, l / 4);
    let (q, j) = (seq / j_n, seq % j_n);
    let c = d * 4 + q / 4;
    let r = q % 4;
    ((((c * j_n + j) * s_n + t / 4) * 4 + r) * 4) + t % 4
}

fn round_up(v: usize, m: usize) -> usize {
    v.div_ceil(m) * m
}

/// Per-channel mean and (biased) variance over the pixels of `y` [P, c].
fn channel_stats(y: &[f32], c: usize) -> (Vec<f64>, Vec<f64>) {
    let px = y.len() / c;
    let pieces = 96.min(px.max(1));
    let mut part = vec![0f64; pieces * c];
    par_rows(&mut part, c, |p0, rows| {
        for (pi, acc) in rows.chunks_mut(c).enumerate() {
            let p = p0 + pi;
            for q in px * p / pieces..px * (p + 1) / pieces {
                for (a, &v) in acc.iter_mut().zip(&y[q * c..(q + 1) * c]) {
                    *a += v as f64;
                }
            }
        }
    });
    let mut mean = vec![0f64; c];
    for acc in part.chunks(c) {
        for (m, a) in mean.iter_mut().zip(acc) {
            *m += a;
        }
    }
    mean.iter_mut().for_each(|m| *m /= px as f64);
    let mut part = vec![0f64; pieces * c];
    par_rows(&mut part, c, |p0, rows| {
        for (pi, acc) in rows.chunks_mut(c).enumerate() {
            let p = p0 + pi;
            for q in px * p / pieces..px * (p + 1) / pieces {
                for ((a, &v), m) in acc.iter_mut().zip(&y[q * c..(q + 1) * c]).zip(&mean) {
                    let d = v as f64 - m;
                    *a += d * d;
                }
            }
        }
    });
    let mut var = vec![0f64; c];
    for acc in part.chunks(c) {
        for (v, a) in var.iter_mut().zip(acc) {
            *v += a;
        }
    }
    var.iter_mut().for_each(|v| *v /= px as f64);
    (mean, var)
}

impl Model {
    /// Uploads the packed weights and MambaScan parameters; keeps the host
    /// vectors.
    pub fn load(store: &Store, npu: &Npu) -> Result<Self, Error> {
        let mut vec = HashMap::new();
        let mut w = HashMap::new();
        for name in store.names() {
            if name.starts_with("ref.") {
                continue;
            }
            match store.entry(name)?.dtype {
                DType::U8 | DType::Bf16 => {
                    w.insert(name.to_string(), npu.upload(store.bytes(name)?)?);
                }
                DType::F32 => {
                    vec.insert(name.to_string(), store.f32(name)?.to_vec());
                }
                _ => {}
            }
        }
        Ok(Model { vec, w, bufs: Bufs::default() })
    }

    fn v(&self, k: &str) -> Result<&[f32], Error> {
        self.vec.get(k).map(Vec::as_slice).ok_or_else(|| Error::Bundle(format!("no tensor {k}")))
    }

    fn w(&self, k: &str) -> Result<&Buffer, Error> {
        self.w.get(k).ok_or_else(|| Error::Bundle(format!("no tensor {k}")))
    }

    /// In place: InstanceNorm over y's pixels (y [P, 64]), affine, PReLU.
    fn inorm_prelu(&self, y: &mut [f32], key: &str, eps: f32) -> Result<(), Error> {
        let (mean, var) = channel_stats(y, H);
        let (g, b, a) = (self.v(&format!("{key}.g"))?, self.v(&format!("{key}.b"))?, self.v(&format!("{key}.a"))?);
        let scale: Vec<f32> = (0..H).map(|o| (g[o] as f64 / (var[o] + eps as f64).sqrt()) as f32).collect();
        let shift: Vec<f32> = (0..H).map(|o| b[o] - mean[o] as f32 * scale[o]).collect();
        par_rows(y, H, |_, rows| {
            for px in rows.chunks_mut(H) {
                for o in 0..H {
                    let v = px[o] * scale[o] + shift[o];
                    px[o] = if v >= 0.0 { v } else { v * a[o] };
                }
            }
        });
        Ok(())
    }

    /// A dense block on x [h, w, 64] -> [h, w, 64].
    fn dense_block(&mut self, c: &Config, npu: &Npu, key: &str, x: &[f32], h: usize, w: usize, t: &mut Timing) -> Result<Vec<f32>, Error> {
        let (padt, skip, pix) = (c.padt, c.skip, c.pix);
        let (wp, hp) = (w + 2, h + 2 * padt);
        let pixels = hp * wp;
        let rows = pixels.div_ceil(pix);
        let rp = round_up(rows, npu.chunk);
        let gs = npu.gemm_spec("dense")?.clone();
        let s_elems = ((rp - 1) * gs.lda + gs.k).max(pixels * skip);
        let t0 = Instant::now();
        let mut bufs = std::mem::take(&mut self.bufs);
        let s = npu.ensure(&mut bufs.s, s_elems)?;
        s.zero(s_elems);
        // the input at the last 64 channels of every interior pixel
        let write = |s: &mut Flat, c0: usize, y: &[f32]| {
            let bits = s.bits_mut(pixels * skip);
            par_rows(bits, wp * skip, |r0, piece| {
                for (ri, row) in piece.chunks_mut(wp * skip).enumerate() {
                    let r = r0 + ri;
                    if r < padt || r >= padt + h {
                        continue;
                    }
                    let tt = r - padt;
                    for f in 0..w {
                        let dst = &mut row[(1 + f) * skip + c0..(1 + f) * skip + c0 + H];
                        for (d, &v) in dst.iter_mut().zip(&y[(tt * w + f) * H..(tt * w + f + 1) * H]) {
                            *d = to_bits(v);
                        }
                    }
                }
            });
        };
        write(s, skip - H, x);
        s.sync(s_elems)?;
        let cb = npu.ensure(&mut bufs.c, rp * gs.n)?;
        t.add("host.dense", t0.elapsed());
        let mut y = vec![0f32; h * w * H];
        for i in 0..4 {
            let d = 1usize << i;
            let wk = self.w(&format!("{key}.{i}"))?;
            npu.gemm("dense", bufs.s.as_ref().unwrap(), 0, wk, cb, 0, rows, t)?;
            let t0 = Instant::now();
            let out = cb.bits(rows * gs.n)?;
            let bias = self.v(&format!("{key}.{i}.bias"))?;
            // y[t][f] = bias + sum_k C[pixel (padt + t + (k - 1) d, f)][k]
            par_rows(&mut y, w * H, |r0, piece| {
                for (ri, row) in piece.chunks_mut(w * H).enumerate() {
                    let tt = r0 + ri;
                    for f in 0..w {
                        let o = &mut row[f * H..(f + 1) * H];
                        o.copy_from_slice(bias);
                        for k in 0..3 {
                            let p = (padt + tt + k * d - d) * wp + f;
                            let src = &out[p * 3 * H + k * H..p * 3 * H + (k + 1) * H];
                            for (a, &b) in o.iter_mut().zip(src) {
                                *a += bf16_to_f32(b);
                            }
                        }
                    }
                }
            });
            self.inorm_prelu(&mut y, &format!("{key}.{i}"), c.in_eps)?;
            if i < 3 {
                let s = bufs.s.as_mut().unwrap();
                write(s, skip - H * (i + 2), &y);
                s.sync(s_elems)?;
            }
            t.add("host.dense", t0.elapsed());
        }
        self.bufs = bufs;
        Ok(y)
    }

    /// Compressed magnitude, phase [F, T] -> the Mamba grid [T', F', 64].
    fn encoder(&mut self, c: &Config, npu: &Npu, mag: &[f32], pha: &[f32], fq: usize, tf: usize, t: &mut Timing) -> Result<(Vec<f32>, usize, usize), Error> {
        let (te, fe) = (tf + 2, fq + 2);
        let (tq, fq2) = ((te - 1) / 4 + 1, (fe - 3) / 2 + 1);
        let t0 = Instant::now();
        let (wi, bi) = (self.v("enc.in.w")?, self.v("enc.in.bias")?);
        let mut x = vec![0f32; te * fe * H];
        par_rows(&mut x, fe * H, |r0, piece| {
            for (ri, row) in piece.chunks_mut(fe * H).enumerate() {
                let tt = r0 + ri;
                for f in 0..fe {
                    let (m, p) = if tt < tf && f < fq { (mag[f * tf + tt], pha[f * tf + tt]) } else { (0.0, 0.0) };
                    for o in 0..H {
                        row[f * H + o] = wi[2 * o] * m + wi[2 * o + 1] * p + bi[o];
                    }
                }
            }
        });
        self.inorm_prelu(&mut x, "enc.in", c.in_eps)?;
        t.add("host.encoder", t0.elapsed());
        let x3 = self.dense_block(c, npu, "enc", &x, te, fe, t)?;
        // every 4th frame, width padded even, rows of 2 pixels
        let t0 = Instant::now();
        let we = fe + fe % 2;
        let rows = tq * we / 2;
        let gs = npu.gemm_spec("down")?.clone();
        let rp = round_up(rows, npu.chunk);
        let need = ((rp - 1) * gs.lda + gs.k).max(tq * we * H);
        let mut bufs = std::mem::take(&mut self.bufs);
        let a = npu.ensure(&mut bufs.a, need)?;
        a.zero(need);
        a.fill_rows(tq, we * H, need, |tt, row| {
            for f in 0..fe {
                for o in 0..H {
                    row[f * H + o] = to_bits(x3[((4 * tt) * fe + f) * H + o]);
                }
            }
        })?;
        let cb = npu.ensure(&mut bufs.c, rp * gs.n)?;
        t.add("host.encoder", t0.elapsed());
        npu.gemm("down", bufs.a.as_ref().unwrap(), 0, self.w("enc.down")?, cb, 0, rows, t)?;
        let t0 = Instant::now();
        let out = cb.bits(rows * H)?;
        let mut y = vec![0f32; tq * fq2 * H];
        par_rows(&mut y, fq2 * H, |r0, piece| {
            for (ri, row) in piece.chunks_mut(fq2 * H).enumerate() {
                let tt = r0 + ri;
                for (o, &b) in row.iter_mut().zip(&out[tt * (we / 2) * H..(tt * (we / 2) + fq2) * H]) {
                    *o = bf16_to_f32(b);
                }
            }
        });
        self.bufs = bufs;
        self.inorm_prelu(&mut y, "enc.down", c.in_eps)?;
        t.add("host.encoder", t0.elapsed());
        Ok((y, tq, fq2))
    }

    /// The shape (sequences, length) a Mamba over n sequences of l runs at,
    /// the time direction's length (or the frequency direction's sequence
    /// count) rounded up to a bucket.
    pub fn mamba_shape(c: &Config, n: usize, l: usize, time: bool) -> (usize, usize) {
        if time { (round_up(n, 16), round_up(l, c.t_step)) } else { (round_up(n, c.t_step), round_up(l, 16)) }
    }

    /// A bidirectional Mamba `key` over xs [n, l, 64] -> its output
    /// projection (before the LayerNorm) [n, l, 64].
    fn mamba(&mut self, c: &Config, npu: &Npu, key: &str, xs: &[f32], n: usize, l: usize, time: bool, t: &mut Timing) -> Result<Vec<f32>, Error> {
        let (seqs, lp) = Self::mamba_shape(c, n, l, time);
        if npu.mamba_spec("conv", seqs, lp).is_none() {
            return Err(Error::Input(format!(
                "no MambaScan shape {seqs} x {lp} in the bundle (rate not exported, or audio past the bundle's t_max)"
            )));
        }
        let t0 = Instant::now();
        let rows = 2 * seqs * lp;
        let mut bufs = std::mem::take(&mut self.bufs);
        let x = npu.ensure(&mut bufs.x, rows * H)?;
        x.fill_rows(rows, H, rows * H, |p, row| {
            if p < rows {
                let (d, s, tt) = stream_pos(p, seqs, lp);
                if s < n && tt < l {
                    let src = if d == 0 { tt } else { l - 1 - tt };
                    for (o, &v) in row.iter_mut().zip(&xs[(s * l + src) * H..(s * l + src + 1) * H]) {
                        *o = to_bits(v);
                    }
                    return;
                }
            }
            row.fill(0);
        })?;
        let rec = npu.ensure(&mut bufs.rec, rows * c.rec)?;
        let o = npu.ensure(&mut bufs.o, rows * H)?;
        t.add("host.mamba", t0.elapsed());
        npu.mamba("in", seqs, lp, bufs.x.as_ref().unwrap(), self.w(&format!("{key}.in"))?, rec, t)?;
        npu.mamba("conv", seqs, lp, rec, self.w(&format!("{key}.conv"))?, rec, t)?;
        npu.mamba("scan", seqs, lp, rec, self.w(&format!("{key}.scan"))?, o, t)?;
        let t0 = Instant::now();
        let ob = o.bits(rows * H)?;
        let mut out = vec![0f32; n * l * H];
        par_rows(&mut out, H, |r0, piece| {
            for (ri, row) in piece.chunks_mut(H).enumerate() {
                let (s, tt) = ((r0 + ri) / l, (r0 + ri) % l);
                let pf = stream_index(0, s, tt, seqs, lp);
                let pb = stream_index(1, s, l - 1 - tt, seqs, lp);
                for (k, v) in row.iter_mut().enumerate() {
                    *v = bf16_to_f32(ob[pf * H + k]) + bf16_to_f32(ob[pb * H + k]);
                }
            }
        });
        self.bufs = bufs;
        t.add("host.mamba", t0.elapsed());
        Ok(out)
    }

    /// x += LayerNorm(o) (both [P, 64]).
    fn add_ln(&self, x: &mut [f32], o: &[f32], key: &str, eps: f32) -> Result<(), Error> {
        let (g, b) = (self.v(&format!("{key}.ln.g"))?, self.v(&format!("{key}.ln.b"))?);
        par_rows(x, H, |r0, piece| {
            for (ri, row) in piece.chunks_mut(H).enumerate() {
                let src = &o[(r0 + ri) * H..(r0 + ri + 1) * H];
                let mean = src.iter().map(|&v| v as f64).sum::<f64>() / H as f64;
                let var = src.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / H as f64;
                let inv = 1.0 / (var + eps as f64).sqrt();
                for k in 0..H {
                    row[k] += ((src[k] as f64 - mean) * inv) as f32 * g[k] + b[k];
                }
            }
        });
        Ok(())
    }

    /// The 30 time-frequency blocks on x [T', F', 64].
    fn blocks(&mut self, c: &Config, npu: &Npu, x: &mut [f32], tq: usize, fq2: usize, t: &mut Timing, tick: &mut dyn FnMut(f64)) -> Result<(), Error> {
        let mut xt = vec![0f32; x.len()];
        for i in 0..c.blocks {
            if i > 0 {
                tick(ENC + BLOCKS * i as f64 / c.blocks as f64);
            }
            // time: sequences over the bins
            let t0 = Instant::now();
            transpose(x, &mut xt, tq, fq2);
            t.add("host.mamba", t0.elapsed());
            let o = self.mamba(c, npu, &format!("{i}.t"), &xt, fq2, tq, true, t)?;
            let t0 = Instant::now();
            let mut ot = vec![0f32; o.len()];
            transpose(&o, &mut ot, fq2, tq);
            self.add_ln(x, &ot, &format!("{i}.t"), c.eps)?;
            t.add("host.mamba", t0.elapsed());
            // frequency
            let o = self.mamba(c, npu, &format!("{i}.f"), x, tq, fq2, false, t)?;
            let t0 = Instant::now();
            self.add_ln(x, &o, &format!("{i}.f"), c.eps)?;
            t.add("host.mamba", t0.elapsed());
        }
        Ok(())
    }

    /// A decoder on x [T', F', 64] -> its trunk [2 F', 4 T', 64]
    /// (frequency major).
    fn decoder(&mut self, c: &Config, npu: &Npu, key: &str, x: &[f32], tq: usize, fq2: usize, t: &mut Timing) -> Result<Vec<f32>, Error> {
        let x3 = self.dense_block(c, npu, &format!("{key}.db"), x, tq, fq2, t)?;
        // up1: three bins of a padded row, N = the 2 interleaved outputs
        let t0 = Instant::now();
        let wu = fq2 + 2;
        let rows = tq * wu;
        let gs = npu.gemm_spec("up1")?.clone();
        let rp = round_up(rows, npu.chunk);
        let need = ((rp - 1) * gs.lda + gs.k).max(rows * H);
        let mut bufs = std::mem::take(&mut self.bufs);
        let a = npu.ensure(&mut bufs.a, need)?;
        a.zero(need);
        a.fill_rows(tq, wu * H, need, |tt, row| {
            for f in 0..fq2 {
                for o in 0..H {
                    row[(1 + f) * H + o] = to_bits(x3[(tt * fq2 + f) * H + o]);
                }
            }
        })?;
        let cb = npu.ensure(&mut bufs.c, rp * gs.n)?;
        t.add("host.decoder", t0.elapsed());
        npu.gemm("up1", bufs.a.as_ref().unwrap(), 0, self.w(&format!("{key}.up1"))?, cb, 0, rows, t)?;
        let t0 = Instant::now();
        let out = cb.bits(rows * gs.n)?;
        let w2 = 2 * fq2;
        let mut y = vec![0f32; tq * w2 * H];
        par_rows(&mut y, w2 * H, |r0, piece| {
            for (ri, row) in piece.chunks_mut(w2 * H).enumerate() {
                let tt = r0 + ri;
                // C row (t, f) = [j 0 | j 1] -> bins 2 f, 2 f + 1
                for (o, &b) in row.iter_mut().zip(&out[tt * wu * 2 * H..(tt * wu + fq2) * 2 * H]) {
                    *o = bf16_to_f32(b);
                }
            }
        });
        self.inorm_prelu(&mut y, &format!("{key}.up1"), c.in_eps)?;
        // up2: along time, on a transposed copy, N = the 4 interleaved outputs
        let tu = tq + 2;
        let rows = w2 * tu;
        let gs = npu.gemm_spec("up2")?.clone();
        let rp = round_up(rows, npu.chunk);
        let need = ((rp - 1) * gs.lda + gs.k).max(rows * H);
        let a = npu.ensure(&mut bufs.a, need)?;
        a.zero(need);
        a.fill_rows(w2, tu * H, need, |f, row| {
            for tt in 0..tq {
                for o in 0..H {
                    row[(1 + tt) * H + o] = to_bits(y[(tt * w2 + f) * H + o]);
                }
            }
        })?;
        let cb = npu.ensure(&mut bufs.c, rp * gs.n)?;
        t.add("host.decoder", t0.elapsed());
        npu.gemm("up2", bufs.a.as_ref().unwrap(), 0, self.w(&format!("{key}.up2"))?, cb, 0, rows, t)?;
        let t0 = Instant::now();
        let out = cb.bits(rows * gs.n)?;
        let t4 = 4 * tq;
        let mut z = vec![0f32; w2 * t4 * H];
        par_rows(&mut z, t4 * H, |r0, piece| {
            for (ri, row) in piece.chunks_mut(t4 * H).enumerate() {
                let f = r0 + ri;
                // C row (f, t) = [j 0..4] -> frames 4 t .. 4 t + 3
                for (o, &b) in row.iter_mut().zip(&out[f * tu * 4 * H..(f * tu + tq) * 4 * H]) {
                    *o = bf16_to_f32(b);
                }
            }
        });
        self.bufs = bufs;
        self.inorm_prelu(&mut z, &format!("{key}.up2"), c.in_eps)?;
        t.add("host.decoder", t0.elapsed());
        Ok(z)
    }

    /// Compressed magnitude, phase [F, T] -> the enhanced pair [F, T].
    pub fn forward(&mut self, c: &Config, npu: &Npu, mag: &[f32], pha: &[f32], fq: usize, tf: usize, t: &mut Timing) -> Result<(Vec<f32>, Vec<f32>), Error> {
        self.forward_with(c, npu, mag, pha, fq, tf, t, &mut |_| {})
    }

    /// `forward`, calling `tick` with the fraction of it done (roughly, by
    /// stage: the encoder, each of the blocks, the decoders) as it goes.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_with(
        &mut self,
        c: &Config,
        npu: &Npu,
        mag: &[f32],
        pha: &[f32],
        fq: usize,
        tf: usize,
        t: &mut Timing,
        tick: &mut dyn FnMut(f64),
    ) -> Result<(Vec<f32>, Vec<f32>), Error> {
        let (mut x, tq, fq2) = self.encoder(c, npu, mag, pha, fq, tf, t)?;
        tick(ENC);
        self.blocks(c, npu, &mut x, tq, fq2, t, tick)?;
        tick(ENC + BLOCKS);
        let zm = self.decoder(c, npu, "mag", &x, tq, fq2, t)?;
        tick(ENC + BLOCKS + (1.0 - ENC - BLOCKS) / 2.0);
        let zp = self.decoder(c, npu, "pha", &x, tq, fq2, t)?;
        let t0 = Instant::now();
        let t4 = 4 * tq;
        let (wm, bm) = (self.v("mag.w")?, self.v("mag.bias")?[0]);
        let (wp, bp) = (self.v("pha.w")?, self.v("pha.bias")?);
        let dot = |z: &[f32], w: &[f32]| z.iter().zip(w).map(|(a, b)| a * b).sum::<f32>();
        let mut em = vec![0f32; fq * tf];
        let mut ep = vec![0f32; fq * tf];
        par_rows(&mut em, tf, |f0, piece| {
            for (fi, row) in piece.chunks_mut(tf).enumerate() {
                let f = f0 + fi;
                for (tt, v) in row.iter_mut().enumerate() {
                    *v = dot(&zm[(f * t4 + tt) * H..(f * t4 + tt + 1) * H], wm) + bm;
                }
            }
        });
        par_rows(&mut ep, tf, |f0, piece| {
            for (fi, row) in piece.chunks_mut(tf).enumerate() {
                let f = f0 + fi;
                for (tt, v) in row.iter_mut().enumerate() {
                    let z = &zp[(f * t4 + tt) * H..(f * t4 + tt + 1) * H];
                    let (r, i) = (dot(z, &wp[..H]) + bp[0], dot(z, &wp[H..]) + bp[1]);
                    *v = i.atan2(r);
                }
            }
        });
        t.add("host.heads", t0.elapsed());
        Ok((em, ep))
    }
}

/// src [a, b, 64] -> dst [b, a, 64].
fn transpose(src: &[f32], dst: &mut [f32], a: usize, b: usize) {
    par_rows(dst, a * H, |r0, piece| {
        for (ri, row) in piece.chunks_mut(a * H).enumerate() {
            let j = r0 + ri;
            for i in 0..a {
                row[i * H..(i + 1) * H].copy_from_slice(&src[(i * b + j) * H..(i * b + j + 1) * H]);
            }
        }
    });
}
