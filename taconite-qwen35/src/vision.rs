// SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
// SPDX-License-Identifier: Apache-2.0

//! The vision tower, as `iron/applications/qwen3_5/qwen35_vision.py` runs
//! it: an image's patches -> the text model's embeddings of its
//! `<|image_pad|>` tokens.
//!
//! A 24-block ViT (hidden 1024, 16 heads of 64, tanh-GELU MLPs of 4096)
//! and a merger folding each 2 x 2 block of patches into one token.
//! Every projection is an `flm.GEMM` with K = 1024 on the vision's own
//! hardware context: the patch embedding (K padded from 768), qkv, proj,
//! fc1, fc2 as four K slices reading fc1's output in place, and the
//! merger's two linears as four K slices each. On the host, in f32: the
//! LayerNorms, the resampled position table, 2D RoPE, attention, the
//! merger's (erf) GELU and the K slices' sums.
//!
//! Weights (`v.weights`): "w8" -- int8 per output channel with scale f 2^e,
//! B holding the exact bfp16 q 2^e and each weight's factors f (tensors
//! `vis.<name>.f`) multiplied into its output columns here, fc1's before
//! its tanh-GELU (so that runs here too) -- or "bfp16x2", B as bfp16 hi +
//! lo with the tanh-GELU fused into fc1.

use std::time::Instant;

use taconite::Timing;
use taconite::cpu::{self, Attn, Rows};
use taconite_bundle::{Manifest, Store};

use crate::Error;
use crate::model::{read, run_gemm, write_rows};
use crate::npu::{Buffer, Npu};

/// K slices of fc2 and of the merger's linears (4096 / 1024).
const SLICES: usize = 4;

/// The vision tower's constants (the manifest's `v.*` params).
#[derive(Debug, Clone)]
pub struct VisionConfig {
    pub d: usize,
    pub i: usize,
    pub depth: usize,
    pub heads: usize,
    pub patch: usize,
    pub merge: usize,
    /// the merger's output: the text model's hidden size
    pub out: usize,
    /// the learned position table is `pos_side` x `pos_side`
    pub pos_side: usize,
    pub rope_theta: f64,
    pub image_token: u32,
    pub vision_start: u32,
    pub vision_end: u32,
    pub min_pixels: usize,
    pub max_pixels: usize,
    /// patches the device buffers hold
    pub max_patches: usize,
}

impl VisionConfig {
    /// None when the bundle has no vision tower.
    pub fn load(m: &Manifest) -> Result<Option<Self>, Error> {
        if !m.has_param("v.D") {
            return Ok(None);
        }
        let p = |k: &str| format!("v.{k}");
        Ok(Some(VisionConfig {
            d: m.param_as(&p("D"))?,
            i: m.param_as(&p("I"))?,
            depth: m.param_as(&p("depth"))?,
            heads: m.param_as(&p("heads"))?,
            patch: m.param_as(&p("patch"))?,
            merge: m.param_as(&p("merge"))?,
            out: m.param_as(&p("out"))?,
            pos_side: m.param_as(&p("pos_side"))?,
            rope_theta: m.param_as(&p("rope_theta"))?,
            image_token: m.param_as(&p("image_token"))?,
            vision_start: m.param_as(&p("vision_start"))?,
            vision_end: m.param_as(&p("vision_end"))?,
            min_pixels: m.param_as(&p("min_pixels"))?,
            max_pixels: m.param_as(&p("max_pixels"))?,
            max_patches: m.param_as(&p("max_patches"))?,
        }))
    }

    pub fn hd(&self) -> usize {
        self.d / self.heads
    }

    /// Inputs of one patch: 3 channels x patch x patch.
    pub fn patch_dim(&self) -> usize {
        3 * self.patch * self.patch
    }
}

struct Block {
    ln1_w: Vec<f32>,
    ln1_b: Vec<f32>,
    ln2_w: Vec<f32>,
    ln2_b: Vec<f32>,
    qkv: Buffer,
    proj: Buffer,
    fc1: Buffer,
    fc2: Vec<Buffer>,
    /// w8: each weight's per-output-channel factors
    qkv_f: Option<Vec<f32>>,
    proj_f: Option<Vec<f32>>,
    fc1_f: Option<Vec<f32>>,
    fc2_f: Option<Vec<f32>>,
}

pub struct Vision {
    pub cfg: VisionConfig,
    blocks: Vec<Block>,
    patch_w: Buffer,
    pos: Vec<f32>,
    m_ln_w: Vec<f32>,
    m_ln_b: Vec<f32>,
    m1: Vec<Buffer>,
    m2: Vec<Buffer>,
    patch_f: Option<Vec<f32>>,
    m1_f: Option<Vec<f32>>,
    m2_f: Option<Vec<f32>>,
    /// device buffers: the GEMMs' A / C (max_patches rows; the merger's a
    /// quarter of that), fc1's and the merger input's with a spare row for
    /// the K slices' shifted views
    a: Buffer,
    qkv_c: Buffer,
    c: Buffer,
    fc1_c: Buffer,
    p: Vec<Buffer>,
    m_a: Buffer,
    m1_p: Vec<Buffer>,
    m2_p: Vec<Buffer>,
}

fn f32s(s: &Store, name: &str) -> Result<Vec<f32>, Error> {
    Ok(s.f32(name)?.to_vec())
}

/// The w8 factors of weight `vis.<name>`, where the bundle has them.
fn factors(s: &Store, name: &str) -> Result<Option<Vec<f32>>, Error> {
    let t = format!("vis.{name}.f");
    if s.has(&t) { Ok(Some(f32s(s, &t)?)) } else { Ok(None) }
}

/// Rows of `x` times per-column factors `f` (none: unchanged).
fn scale_cols(x: &mut [f32], f: Option<&Vec<f32>>) {
    if let Some(f) = f {
        cpu::par_rows(x, f.len(), |_, piece| {
            for row in piece.chunks_mut(f.len()) {
                for (v, &k) in row.iter_mut().zip(f) {
                    *v *= k;
                }
            }
        });
    }
}

/// GELU, tanh form (torch's `approximate="tanh"`).
fn gelu_tanh(x: f32) -> f32 {
    const C: f32 = 0.797_884_6; // sqrt(2 / pi)
    0.5 * x * (1.0 + (C * (x + 0.044_715 * x * x * x)).tanh())
}

impl Vision {
    pub fn load(cfg: VisionConfig, s: &Store, npu: &Npu) -> Result<Self, Error> {
        let up = |name: &str| npu.upload(s.bytes(&format!("vis.{name}"))?);
        let slices = |name: &str| (0..SLICES).map(|j| up(&format!("{name}.{j}"))).collect::<Result<Vec<_>, _>>();
        let mut blocks = Vec::with_capacity(cfg.depth);
        for i in 0..cfg.depth {
            let h = |k: &str| f32s(s, &format!("vis.b{i}.{k}"));
            blocks.push(Block {
                ln1_w: h("ln1_w")?,
                ln1_b: h("ln1_b")?,
                ln2_w: h("ln2_w")?,
                ln2_b: h("ln2_b")?,
                qkv: up(&format!("b{i}.qkv"))?,
                proj: up(&format!("b{i}.proj"))?,
                fc1: up(&format!("b{i}.fc1"))?,
                fc2: slices(&format!("b{i}.fc2"))?,
                qkv_f: factors(s, &format!("b{i}.qkv"))?,
                proj_f: factors(s, &format!("b{i}.proj"))?,
                fc1_f: factors(s, &format!("b{i}.fc1"))?,
                fc2_f: factors(s, &format!("b{i}.fc2"))?,
            });
        }
        let (rows, d, w4) = (cfg.max_patches, cfg.d, SLICES * cfg.d);
        let mrows = rows / (cfg.merge * cfg.merge);
        let zeros = |n: usize| npu.zeros(n);
        let many = |k: usize, n: usize| (0..k).map(|_| zeros(n)).collect::<Result<Vec<_>, _>>();
        Ok(Vision {
            blocks,
            patch_w: up("patch")?,
            pos: f32s(s, "vis.pos")?,
            m_ln_w: f32s(s, "vis.m_ln_w")?,
            m_ln_b: f32s(s, "vis.m_ln_b")?,
            m1: slices("m1")?,
            m2: slices("m2")?,
            patch_f: factors(s, "patch")?,
            m1_f: factors(s, "m1")?,
            m2_f: factors(s, "m2")?,
            a: zeros(rows * d)?,
            qkv_c: zeros(rows * 3 * d)?,
            c: zeros(rows * d)?,
            fc1_c: zeros((rows + 1) * w4)?,
            p: many(SLICES, rows * d)?,
            m_a: zeros((mrows + 1) * w4)?,
            m1_p: many(SLICES, mrows * w4)?,
            m2_p: many(SLICES, mrows * cfg.out)?,
            cfg,
        })
    }

    /// One image's patches `[n, 3 p p]` (merge-block order) on its
    /// `(gh, gw)` grid -> `[n / merge^2, out]` image-token embeddings.
    pub fn encode(
        &mut self,
        npu: &Npu,
        patches: &[f32],
        grid: (usize, usize),
        t: &mut Timing,
    ) -> Result<Vec<f32>, Error> {
        let c = self.cfg.clone();
        let (d, pd) = (c.d, c.patch_dim());
        let n = patches.len() / pd;
        if n == 0 || n > c.max_patches || n != grid.0 * grid.1 {
            return Err(Error::Input(format!("{n} patches (1..={} fit)", c.max_patches)));
        }
        let t0 = Instant::now();
        let pos = self.pos_embed(grid);
        let (cos, sin) = self.rope(grid);
        // the patch embedding: K padded from 3 p p to D with zero columns
        let mut xp = vec![0f32; n * d];
        for r in 0..n {
            xp[r * d..r * d + pd].copy_from_slice(&patches[r * pd..(r + 1) * pd]);
        }
        write_rows(&mut self.a, &xp)?;
        t.add("v.host", t0.elapsed());
        let chunks = |rows: usize, npu: &Npu| npu.gemm("v_patch").map(|g| rows.div_ceil(g.m));
        let nc = chunks(n, npu)?;
        run_gemm(npu, "v_patch", &self.a, 0, &self.patch_w, &self.c, nc, t)?;
        let mut x = read(&self.c, 0, n * d)?;
        scale_cols(&mut x, self.patch_f.as_ref());
        cpu::add_(&mut x, &pos);

        let (h, hd) = (c.heads, c.hd());
        for b in 0..self.blocks.len() {
            let t0 = Instant::now();
            let blk = &self.blocks[b];
            let xn = cpu::layer_norm(&x, d, &blk.ln1_w, &blk.ln1_b, 1e-6);
            write_rows(&mut self.a, &xn)?;
            t.add("v.host", t0.elapsed());
            let blk = &self.blocks[b];
            run_gemm(npu, "v_qkv", &self.a, 0, &blk.qkv, &self.qkv_c, nc, t)?;
            let t0 = Instant::now();
            let mut qkv = read(&self.qkv_c, 0, n * 3 * d)?;
            scale_cols(&mut qkv, self.blocks[b].qkv_f.as_ref());
            let (mut q, mut k) = (vec![0f32; n * d], vec![0f32; n * d]);
            for r in 0..n {
                q[r * d..(r + 1) * d].copy_from_slice(&qkv[r * 3 * d..r * 3 * d + d]);
                k[r * d..(r + 1) * d].copy_from_slice(&qkv[r * 3 * d + d..r * 3 * d + 2 * d]);
                for hh in 0..h {
                    let (cs, sn) = (&cos[r * hd..(r + 1) * hd], &sin[r * hd..(r + 1) * hd]);
                    rope_half(&mut q[r * d + hh * hd..][..hd], cs, sn);
                    rope_half(&mut k[r * d + hh * hd..][..hd], cs, sn);
                }
            }
            let attn = Attn { heads: h, ..Default::default() };
            let o = cpu::attention(&q, d, Rows::f32(&k, d), Rows::strided(&qkv, n, 3 * d, 2 * d), &attn);
            write_rows(&mut self.a, &o)?;
            t.add("v.attn", t0.elapsed());
            let blk = &self.blocks[b];
            run_gemm(npu, "v_proj", &self.a, 0, &blk.proj, &self.c, nc, t)?;
            let t0 = Instant::now();
            let mut o = read(&self.c, 0, n * d)?;
            scale_cols(&mut o, blk.proj_f.as_ref());
            cpu::add_(&mut x, &o);
            let xn = cpu::layer_norm(&x, d, &blk.ln2_w, &blk.ln2_b, 1e-6);
            write_rows(&mut self.a, &xn)?;
            t.add("v.host", t0.elapsed());
            let blk = &self.blocks[b];
            run_gemm(npu, "v_fc1", &self.a, 0, &blk.fc1, &self.fc1_c, nc, t)?;
            if blk.fc1_f.is_some() {
                // w8: the factors come before the GELU, both here; the
                // result goes back where fc2's slices read it
                let t0 = Instant::now();
                let mut h = read(&self.fc1_c, 0, n * SLICES * d)?;
                scale_cols(&mut h, blk.fc1_f.as_ref());
                cpu::par_rows(&mut h, SLICES * d, |_, piece| piece.iter_mut().for_each(|v| *v = gelu_tanh(*v)));
                write_rows(&mut self.fc1_c, &h)?;
                t.add("v.host", t0.elapsed());
            }
            let blk = &self.blocks[b];
            for s in 0..SLICES {
                run_gemm(npu, "v_fc2", &self.fc1_c, s * d, &blk.fc2[s], &self.p[s], nc, t)?;
            }
            let t0 = Instant::now();
            let mut m2 = vec![0f32; n * d];
            for p in &self.p {
                cpu::add_(&mut m2, &read(p, 0, n * d)?);
            }
            scale_cols(&mut m2, blk.fc2_f.as_ref());
            cpu::add_(&mut x, &m2);
            t.add("v.host", t0.elapsed());
        }

        // the merger: LayerNorm, then [n / 4, 4 D] -> GELU(fc1) -> fc2
        let t0 = Instant::now();
        let m = n / (c.merge * c.merge);
        let w4 = SLICES * d;
        let xm = cpu::layer_norm(&x, d, &self.m_ln_w, &self.m_ln_b, 1e-6);
        write_rows(&mut self.m_a, &xm)?;
        t.add("v.host", t0.elapsed());
        let mc = m.div_ceil(npu.gemm("v_m1")?.m);
        for s in 0..SLICES {
            run_gemm(npu, "v_m1", &self.m_a, s * d, &self.m1[s], &self.m1_p[s], mc, t)?;
        }
        let t0 = Instant::now();
        let mut hm = vec![0f32; m * w4];
        for p in &self.m1_p {
            cpu::add_(&mut hm, &read(p, 0, m * w4)?);
        }
        scale_cols(&mut hm, self.m1_f.as_ref());
        hm.iter_mut().for_each(|v| *v = cpu::gelu(*v));
        write_rows(&mut self.m_a, &hm)?;
        t.add("v.host", t0.elapsed());
        for s in 0..SLICES {
            run_gemm(npu, "v_m2", &self.m_a, s * d, &self.m2[s], &self.m2_p[s], mc, t)?;
        }
        let mut out = vec![0f32; m * c.out];
        for p in &self.m2_p {
            cpu::add_(&mut out, &read(p, 0, m * c.out)?);
        }
        scale_cols(&mut out, self.m2_f.as_ref());
        Ok(out)
    }

    /// (row, col) of each patch in merge-block order.
    fn block_order(&self, (gh, gw): (usize, usize)) -> Vec<(usize, usize)> {
        let mg = self.cfg.merge;
        let mut v = Vec::with_capacity(gh * gw);
        for bh in 0..gh / mg {
            for bw in 0..gw / mg {
                for mh in 0..mg {
                    for mw in 0..mg {
                        v.push((bh * mg + mh, bw * mg + mw));
                    }
                }
            }
        }
        v
    }

    /// The learned position table resampled to the grid (bilinear,
    /// align_corners), a row per patch.
    fn pos_embed(&self, grid: (usize, usize)) -> Vec<f32> {
        let (side, d) = (self.cfg.pos_side, self.cfg.d);
        let taps = |i: usize, n: usize| {
            let src = i as f32 * (side - 1) as f32 / (n.max(2) - 1) as f32;
            let f = src.floor();
            let lo = (f as usize).min(side - 1);
            let hi = (f as usize + 1).min(side - 1);
            let w = src - f;
            (lo, hi, 1.0 - w, w)
        };
        let mut out = vec![0f32; grid.0 * grid.1 * d];
        for (p, (r, c)) in self.block_order(grid).into_iter().enumerate() {
            let (r0, r1, wr0, wr1) = taps(r, grid.0);
            let (c0, c1, wc0, wc1) = taps(c, grid.1);
            let o = &mut out[p * d..(p + 1) * d];
            for (idx, wt) in [
                (r0 * side + c0, wr0 * wc0),
                (r0 * side + c1, wr0 * wc1),
                (r1 * side + c0, wr1 * wc0),
                (r1 * side + c1, wr1 * wc1),
            ] {
                for (ov, &tv) in o.iter_mut().zip(&self.pos[idx * d..(idx + 1) * d]) {
                    *ov += tv * wt;
                }
            }
        }
        out
    }

    /// cos, sin `[n, hd]` of the axial RoPE: a quarter of the head's
    /// frequencies from the row, a quarter from the column, repeated for
    /// the half-split rotation.
    fn rope(&self, grid: (usize, usize)) -> (Vec<f32>, Vec<f32>) {
        let hd = self.cfg.hd();
        let q = hd / 2;
        let theta = self.cfg.rope_theta as f32;
        let inv: Vec<f32> = (0..q / 2).map(|j| 1.0 / theta.powf((2 * j) as f32 / q as f32)).collect();
        let order = self.block_order(grid);
        let (mut cos, mut sin) = (vec![0f32; order.len() * hd], vec![0f32; order.len() * hd]);
        for (p, (r, c)) in order.into_iter().enumerate() {
            for (j, &f) in inv.iter().enumerate() {
                for (off, pos) in [(0, r), (q / 2, c)] {
                    let (s, co) = (pos as f32 * f).sin_cos();
                    for base in [0, q] {
                        cos[p * hd + base + off + j] = co;
                        sin[p * hd + base + off + j] = s;
                    }
                }
            }
        }
        (cos, sin)
    }
}

/// x rotated by cos / sin over its whole length, half-split convention.
fn rope_half(x: &mut [f32], cos: &[f32], sin: &[f32]) {
    let h = x.len() / 2;
    for j in 0..h {
        let (x1, x2) = (x[j], x[j + h]);
        x[j] = x1 * cos[j] - x2 * sin[j];
        x[j + h] = x2 * cos[j + h] + x1 * sin[j + h];
    }
}
