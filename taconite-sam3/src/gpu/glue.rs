// SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
// SPDX-License-Identifier: Apache-2.0

//! NPU+GPU mode: the work between the NPU's kernels on the GPU, reading
//! each kernel's output buffer and writing the next one's input buffer in
//! place (every NPU buffer involved is imported once, as a dma-buf), with
//! the activations in between (the FPN levels, the DETR encoder's residual
//! stream, the pixel decoder's maps) kept in GPU memory.
//!
//! Each step is a [`Plan`] recorded at load, run between two NPU kernels:
//!
//! - neck: the ViT's output (window order) -> `n_in`'s A; level 0's GELU and
//!   2x2 shuffle -> `n_up`'s A; the 3x3 convolutions' overlapping inputs
//!   (`neck.rs`) built straight from the GEMM outputs; bias -> FPN levels;
//! - DETR encoder, per layer: LayerNorm (+ position) -> the qkv GEMM's A; the
//!   q/k/v split into the MHA's buffers; residual adds; the prompt softmax
//!   between the folded cross-attention's two GEMMs;
//! - the DETR decoder's `dec_kv` A (`enc + pos | enc`);
//! - mask decoder: the prompt cross-attention's glue, upsample + skip into
//!   the conv inputs, bias + GroupNorm + ReLU, and the mask head's A.
//!
//! What stays on the CPU: folding + packing the weights built per prompt
//! (`pack_b`, small) and the final `[px, Q] -> [Q, px]` mask transpose.
//! The kernels mirror the CPU code's arithmetic (bf16 rounding, `fast_exp`,
//! the erf GELU, the softmax's summation order), so most bf16 values they
//! write are the CPU path's to the bit.

use super::ops::{Builder, EPS, F32};
use super::{Plan, Recorder, View, Vk};
use crate::npu::Npu;
use crate::{Config, Error, Ios};

/// bf16 packing and unpacking, `fast_exp`, the erf GELU -- as the CPU's.
const COMMON: &str = r#"
fn bf16(f: f32) -> u32 {
    let b = bitcast<u32>(f);
    if ((b & 0x7fffffffu) > 0x7f800000u) {
        return (b >> 16u) | 0x40u;
    }
    return (b + 0x7fffu + ((b >> 16u) & 1u)) >> 16u;
}
fn pack(a: f32, b: f32) -> u32 {
    return bf16(a) | (bf16(b) << 16u);
}
fn lo(w: u32) -> f32 {
    return bitcast<f32>(w << 16u);
}
fn hi(w: u32) -> f32 {
    return bitcast<f32>(w & 0xffff0000u);
}
fn fast_exp(x0: f32) -> f32 {
    let x = clamp(x0, -87.0, 87.0);
    let t = x * 1.442695;
    let n = floor(t);
    let f = t - n;
    let p = 1.0 + f * (6.931472e-1 + f * (2.4022648e-1 + f * (5.5503325e-2 + f * (9.618438e-3
        + f * (1.3398874e-3 + f * 1.5353362e-4)))));
    return bitcast<f32>(bitcast<u32>(p) + (u32(i32(n)) << 23u));
}
fn erf(x: f32) -> f32 {
    let z = abs(x);
    let t = 1.0 / (1.0 + 0.5 * z);
    let r = t * fast_exp(-z * z - 1.2655122 + t * (1.0000237 + t * (0.37409196 + t * (0.09678418
        + t * (-0.18628806 + t * (0.27886807 + t * (-1.135204 + t * (1.4885159
        + t * (-0.82215223 + t * 0.17087277)))))))));
    if (x >= 0.0) {
        return 1.0 - r;
    }
    return r - 1.0;
}
fn gelu(x: f32) -> f32 {
    return 0.5 * x * (1.0 + erf(x * 0.70710677));
}
"#;

/// The head of every element-wise kernel: `idx`, this thread's element, over
/// a grid of `NX` x `ny` workgroups of 256 (see [`grid`]).
const IDX: &str = "let idx = (wg.y * {NX}u + wg.x) * 256u + li;";

/// The ViT's residual stream (window order, f32 or bf16) -> `n_in`'s A,
/// raster order, bf16: row `r` is token `inv[r]`.
const VIT_IN: &str = r#"
@group(0) @binding(0) var<storage, read> xres: array<{XT}>;
@group(0) @binding(1) var<storage, read> inv: array<u32>;
@group(0) @binding(2) var<storage, read_write> y: array<u32>;
const N: u32 = {N}u;
const DIM: u32 = {DIM}u;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    {IDX}
    if (idx >= N) {
        return;
    }
    let r = idx / (DIM / 2u);
    let w = idx % (DIM / 2u);
    let j = inv[r];
    {LOAD}
}
"#;

/// Level 0 of the neck: GELU of the shared GEMM's first `MID` columns,
/// shuffled 2x2 onto the `2G` grid -> `n_up`'s A (bf16).
const NECK_GELU: &str = r#"
@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> y: array<u32>;
const N: u32 = {N}u;
const G: u32 = {G}u;
const WIDTH: u32 = {WIDTH}u;
const MID: u32 = {MID}u;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    {IDX}
    if (idx >= N) {
        return;
    }
    let r = idx / (MID / 2u);
    let wj = idx % (MID / 2u);
    let yy = r / (2u * G);
    let xx = r % (2u * G);
    let w = src[(((yy / 2u) * G + xx / 2u) * WIDTH + ((yy % 2u) * 2u + xx % 2u) * MID) / 2u + wj];
    y[idx] = pack(gelu(lo(w)), gelu(hi(w)));
}
"#;

/// A 3x3 conv's overlapping input (`neck.rs`): row `p` of `Y` is pixels
/// `p`, `p + WP`, `p + 2 WP` of the zero-bordered `SIDE x SIDE` image
/// (row pitch `WP`), `C` channels each; `fetch` reads a channel pair of
/// pixel (y, x) from wherever the image is.
const CONV_IN: &str = r#"
{SRC_DECL}
@group(0) @binding({YB}) var<storage, read_write> ybuf: array<u32>;
const N: u32 = {N}u;
const SIDE: u32 = {SIDE}u;
const WP: u32 = {WP}u;
const D: u32 = {D}u;
const C: u32 = {C}u;

{FETCH}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    {IDX}
    if (idx >= N) {
        return;
    }
    let p = idx / (D / 2u);
    let e = (idx % (D / 2u)) * 2u;
    let r = p + (e / C) * WP;
    let yy = r / WP;
    let xx = r % WP;
    var v = 0u;
    if (yy >= 1u && yy <= SIDE && xx >= 1u && xx <= SIDE) {
        v = fetch(yy - 1u, xx - 1u, e % C);
    }
    ybuf[idx] = v;
}
"#;

/// A 3x3 conv's output (`[SIDE x WP, OC]` bf16, two junk columns an image
/// row) + bias -> `[SIDE^2, OC]` f32.
const CONV_OUT: &str = r#"
@group(0) @binding(0) var<storage, read> c: array<u32>;
@group(0) @binding(1) var<storage, read> bias: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
const N: u32 = {N}u;
const SIDE: u32 = {SIDE}u;
const WP: u32 = {WP}u;
const OC: u32 = {OC}u;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    {IDX}
    if (idx >= N) {
        return;
    }
    let pix = idx / OC;
    let o = idx % OC;
    let e = ((pix / SIDE) * WP + pix % SIDE) * OC + o;
    let w = c[e / 2u];
    var v = lo(w);
    if (e % 2u == 1u) {
        v = hi(w);
    }
    y[idx] = v + bias[o];
}
"#;

/// Rows of `x` (f32, `DIM` wide) -> bf16 rows of an NPU input: LayerNorm'd
/// (`NORM`) or not, and with `POS` the row is `[h + pos | h]`, `2 DIM` wide.
/// One workgroup a row.
const ROWS_BF16: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read> pos: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<u32>;
const DIM: u32 = {DIM}u;
const EPS: f32 = {EPS};
const NORM: bool = {NORM};
const POS: bool = {POS};
var<workgroup> red: array<f32, 256>;
var<workgroup> nv: array<f32, {DIM}>;

fn total(t: u32) -> f32 {
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (t < s) {
            red[t] += red[t + s];
        }
        workgroupBarrier();
    }
    return red[0];
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let r = wg.x;
    let base = r * DIM;
    var mean = 0.0;
    var inv = 1.0;
    if (NORM) {
        var s = 0.0;
        for (var j = t; j < DIM; j += 256u) {
            s += x[base + j];
        }
        red[t] = s;
        workgroupBarrier();
        mean = total(t) / f32(DIM);
        workgroupBarrier();
        var v = 0.0;
        for (var j = t; j < DIM; j += 256u) {
            let dd = x[base + j] - mean;
            v += dd * dd;
        }
        red[t] = v;
        workgroupBarrier();
        inv = 1.0 / sqrt(total(t) / f32(DIM) + EPS);
    }
    for (var j = t; j < DIM; j += 256u) {
        if (NORM) {
            nv[j] = (x[base + j] - mean) * inv * w[j] + b[j];
        } else {
            nv[j] = x[base + j];
        }
    }
    workgroupBarrier();
    for (var k = t; k < DIM / 2u; k += 256u) {
        let a0 = nv[2u * k];
        let a1 = nv[2u * k + 1u];
        if (POS) {
            y[r * DIM + k] = pack(a0 + pos[base + 2u * k], a1 + pos[base + 2u * k + 1u]);
            y[r * DIM + DIM / 2u + k] = pack(a0, a1);
        } else {
            y[r * (DIM / 2u) + k] = pack(a0, a1);
        }
    }
}
"#;

/// The qkv GEMM's output rows `[q | k | v]` (`3 PD` wide) -> the MHA's
/// q, k and v buffers (`PD` wide each).
const SPLIT_QKV: &str = r#"
@group(0) @binding(0) var<storage, read> qkv: array<u32>;
@group(0) @binding(1) var<storage, read_write> q: array<u32>;
@group(0) @binding(2) var<storage, read_write> k: array<u32>;
@group(0) @binding(3) var<storage, read_write> v: array<u32>;
const N: u32 = {N}u;
const PD2: u32 = {PD2}u;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    {IDX}
    if (idx >= N) {
        return;
    }
    let s = (idx / PD2) * 3u * PD2 + idx % PD2;
    q[idx] = qkv[s];
    k[idx] = qkv[s + PD2];
    v[idx] = qkv[s + 2u * PD2];
}
"#;

/// `x += c (+ bias)`, `c` a GEMM's bf16 output with rows as wide as `x`'s.
const ADD_BF16: &str = r#"
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> c: array<u32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
const N: u32 = {N}u;
const DIM: u32 = {DIM}u;
const BIAS: bool = {BIAS};

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    {IDX}
    if (idx >= N) {
        return;
    }
    let w = c[idx / 2u];
    var v = lo(w);
    if (idx % 2u == 1u) {
        v = hi(w);
    }
    if (BIAS) {
        v += bias[idx % DIM];
    }
    x[idx] += v;
}
"#;

/// The folded prompt cross-attention's softmax: per row and head, over the
/// `L` prompt tokens (masked ones out), bf16 in and out -- `cpu::softmax_`'s
/// arithmetic, 8-lane partial sums included.
const PROMPT_SOFTMAX: &str = r#"
@group(0) @binding(0) var<storage, read> s: array<u32>;
@group(0) @binding(1) var<storage, read> valid: array<u32>;
@group(0) @binding(2) var<storage, read_write> y: array<u32>;
const N: u32 = {N}u;
const NH: u32 = {NH}u;
const L: u32 = {L}u;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    {IDX}
    if (idx >= N) {
        return;
    }
    let base = idx * L / 2u; // row-major [rows, NH, L]: (row, head) = idx
    var v: array<f32, {L}>;
    var m = -3.0e38;
    for (var j = 0u; j < L; j++) {
        let w = s[base + j / 2u];
        var x = lo(w);
        if (j % 2u == 1u) {
            x = hi(w);
        }
        if (valid[j] == 0u) {
            x = -3.0e38;
        }
        v[j] = x;
        m = max(m, x);
    }
    if (m <= -3.0e38) {
        for (var j = 0u; j < L / 2u; j++) {
            y[base + j] = 0u;
        }
        return;
    }
    var acc: array<f32, 8>;
    for (var j = 0u; j < L; j++) {
        v[j] = fast_exp(v[j] - m);
    }
    for (var j = 0u; j < (L / 8u) * 8u; j++) {
        acc[j % 8u] += v[j];
    }
    for (var j = (L / 8u) * 8u; j < L; j++) {
        acc[0] += v[j];
    }
    var total = 0.0;
    for (var l = 0u; l < 8u; l++) {
        total += acc[l];
    }
    let inv = 1.0 / total;
    for (var j = 0u; j < L / 2u; j++) {
        y[base + j] = pack(v[2u * j] * inv, v[2u * j + 1u] * inv);
    }
}
"#;

/// GroupNorm statistics, pass `MODE` 0 (sums) or 1 (squared deviations
/// from the means in `stats[0..G]`): one workgroup per 256 pixels, a
/// thread per channel, partial sums per group into `part[wg, G]`.
const GN_PART: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> stats: array<f32>;
@group(0) @binding(2) var<storage, read_write> part: array<f32>;
const PX: u32 = {PX}u;
const C: u32 = {C}u;
const G: u32 = {G}u;
const MODE: u32 = {MODE}u;
var<workgroup> red: array<f32, {C}>;

@compute @workgroup_size({C})
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let g = t / (C / G);
    var mean = 0.0;
    if (MODE == 1u) {
        mean = stats[g];
    }
    let p1 = min(PX, (wg.x + 1u) * 256u);
    var s = 0.0;
    for (var p = wg.x * 256u; p < p1; p++) {
        let v = x[p * C + t];
        if (MODE == 0u) {
            s += v;
        } else {
            s += (v - mean) * (v - mean);
        }
    }
    red[t] = s;
    workgroupBarrier();
    if (t < G) {
        var gs = 0.0;
        for (var k = 0u; k < C / G; k++) {
            gs += red[t * (C / G) + k];
        }
        part[wg.x * G + t] = gs;
    }
}
"#;

/// Reduces `GN_PART`'s partials: pass 0 -> the means `stats[0..G]`, pass
/// 1 -> the inverse deviations `stats[G..2G]`.
const GN_REDUCE: &str = r#"
@group(0) @binding(0) var<storage, read> part: array<f32>;
@group(0) @binding(1) var<storage, read_write> stats: array<f32>;
const PARTS: u32 = {PARTS}u;
const G: u32 = {G}u;
const COUNT: f32 = {COUNT};
const MODE: u32 = {MODE}u;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) t: u32) {
    if (t >= G) {
        return;
    }
    var s = 0.0;
    for (var k = 0u; k < PARTS; k++) {
        s += part[k * G + t];
    }
    if (MODE == 0u) {
        stats[t] = s / COUNT;
    } else {
        stats[G + t] = 1.0 / sqrt(s / COUNT + {EPS});
    }
}
"#;

/// `x = relu(GroupNorm(x))` in place, two channels a thread; with `BF` the
/// result also goes to `y` as bf16 (the mask head's A).
const GN_APPLY: &str = r#"
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> stats: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<u32>;
const N: u32 = {N}u;
const C: u32 = {C}u;
const G: u32 = {G}u;
const BF: bool = {BF};

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    {IDX}
    if (idx >= N) {
        return;
    }
    let e = 2u * idx;
    let ch = e % C;
    let g = ch / (C / G);
    let mean = stats[g];
    let inv = stats[G + g];
    let v0 = max((x[e] - mean) * inv * w[ch] + b[ch], 0.0);
    let v1 = max((x[e + 1u] - mean) * inv * w[ch + 1u] + b[ch + 1u], 0.0);
    x[e] = v0;
    x[e + 1u] = v1;
    if (BF) {
        y[idx] = pack(v0, v1);
    }
}
"#;

/// Workgroups (of 256) for `n` threads, as `[NX, ny]` under the 65535 limit.
fn grid(n: usize) -> ([u32; 3], usize) {
    let groups = n.div_ceil(256);
    let nx = groups.min(32768);
    ([nx as u32, groups.div_ceil(nx) as u32, 1], nx)
}

/// Records one element-wise kernel over `n` threads.
fn elementwise(
    rec: &mut Recorder,
    label: &str,
    src: &str,
    consts: &[(&str, String)],
    views: &[View],
    n: usize,
) -> Result<(), Error> {
    let (g, nx) = grid(n);
    let src = format!("{COMMON}{}", src.replace("{IDX}", IDX));
    let mut all: Vec<(&str, String)> = consts.to_vec();
    all.push(("NX", nx.to_string()));
    all.push(("N", n.to_string()));
    rec.op(label, &src, &all, views, g).map_err(Error::from)
}

/// The NPU+GPU mode's plans and the GPU-resident activations.
pub struct Glue {
    /// the ViT's output -> `n_in`'s A (bundles with a device ViT)
    pub vit_in: Option<Plan>,
    /// `n_in`'s output -> `n_up`'s A and the 144 / 72 px convs' inputs
    pub neck_mid: Plan,
    /// `n_up`'s output -> the 288 px conv's input
    pub neck_up: Plan,
    /// the three convs' outputs + bias -> the FPN levels
    pub neck_out: Plan,
    /// DETR encoder: `x = fpn[2]`
    pub enc_init: Plan,
    /// per layer: LN1 (+ pos) -> the qkv GEMM's A
    pub enc_ln1: Vec<Plan>,
    /// the qkv GEMM's output -> the MHA's q, k, v
    pub split: Plan,
    /// the MHA's output -> the o GEMM's A
    pub copy_o: Plan,
    /// per layer: `x += o`; LN2 -> the scores GEMM's A
    pub enc_o: Vec<Plan>,
    /// the prompt softmax: the scores GEMM's output -> the context GEMM's A
    pub softmax: Plan,
    /// per layer: `x += context`; LN3 -> fc1's A
    pub enc_c: Vec<Plan>,
    /// per layer: `x += fc2 + b`
    pub enc_fc: Vec<Plan>,
    /// `[x + pos | x]` -> the `dec_kv` GEMM's A
    pub dec_kv_in: Plan,
    /// mask decoder: `p = x`; its LN -> the scores GEMM's A
    pub mask_ln: Plan,
    /// `p += context`
    pub mask_add: Plan,
    /// per pixel-decoder stage: upsample(p) + skip -> the conv's input
    pub mask_up: Vec<Plan>,
    /// per stage: conv + bias -> GroupNorm -> ReLU (the last also -> the
    /// mask head's A)
    pub mask_gn: Vec<Plan>,
    /// FPN levels 0-2 `[side^2, D]` f32, the DETR encoder's stream
    /// `[T, D]`, the prompt mask `[L]` (u32, 1 = valid)
    pub fpn: [View; 3],
    pub x: View,
    pub valid: View,
    /// the ViT's GELU on fc1's output, in place, when the NPU's epilogue
    /// is off (see `Gpu::new`)
    pub vit_gelu: Option<Plan>,
}

impl Glue {
    /// Imports the NPU buffers the steps touch and records the steps.
    pub(crate) fn new(vk: &mut Vk, b: &mut Builder, cfg: &Config, npu: &Npu, io: &Ios) -> Result<Self, Error> {
        let (g, t, d, dim) = (cfg.grid, cfg.tokens(), cfg.d_model, cfg.vit_dim);
        let (s0, s1, s2) = (cfg.neck_splits[0], cfg.neck_splits[1], cfg.neck_splits[2]);
        let width = s0 + s1 + s2;
        let (nh, l) = (cfg.d_heads, cfg.text_len);
        let conv = npu.spec("conv")?;
        let dconv = (conv.k - conv.lda) / 2; // a Y row: 3 taps of C channels
        let c = dconv / 3;
        if c != d {
            return Err(Error::Gpu(format!("conv input channels {c} != d_model {d}")));
        }
        let n_up = npu.spec("n_up")?.n;
        let pd = npu.mhas["mha_d"].d * nh;
        // the layouts the kernels assume
        let shape = |ok: bool, what: &str| if ok { Ok(()) } else { Err(Error::Gpu(format!("unexpected {what}"))) };
        shape(io.n_in.n == width && npu.spec("n_in")?.k == dim, "n_in shape")?;
        shape(npu.spec("n_up")?.k == s0 / 4, "n_up K")?;
        shape(io.d_qkv.n == 3 * pd && npu.spec("d_qkv")?.k == 2 * d, "d_qkv shape")?;
        shape(npu.spec("d_o")?.k == pd && io.d_o.n == d, "d_o shape")?;
        shape(io.d_s.n == nh * l && npu.spec("d_s")?.k == d, "d_s shape")?;
        shape(npu.spec("d_c")?.k == nh * l && io.d_c.n == d, "d_c shape")?;
        shape(npu.spec("d_fc1")?.k == d && io.d_fc2.n == d, "d_fc shape")?;
        shape(npu.spec("dec_kv")?.k == 2 * d && npu.spec("m_head")?.k == d, "dec_kv / m_head K")?;
        shape(d % 2 == 0 && l % 2 == 0 && pd % 2 == 0 && dim % 2 == 0, "odd widths")?;
        let sides = [4 * g, 2 * g, g];

        let mut npu_view = |buf: &crate::npu::Buffer| vk.import_npu(buf);
        let nin_a = npu_view(&io.n_in.a)?;
        let nin_c = npu_view(&io.n_in.c)?;
        let nup_a = npu_view(&io.n_up.a)?;
        let nup_c = npu_view(&io.n_up.c)?;
        let mut conv_a = Vec::new();
        let mut conv_c = Vec::new();
        for s in sides {
            let cio = &io.conv[&s];
            conv_a.push(npu_view(&cio.a)?);
            conv_c.push(npu_view(&cio.c)?);
        }
        let xres = if cfg.vit_device { Some(npu_view(&io.xres[0])?) } else { None };
        let (qkv_a, qkv_c) = (npu_view(&io.d_qkv.a)?, npu_view(&io.d_qkv.c)?);
        let (mq, mk, mv, mo) =
            (npu_view(&io.mha_d.q)?, npu_view(&io.mha_d.k)?, npu_view(&io.mha_d.v)?, npu_view(&io.mha_d.o)?);
        let (o_a, o_c) = (npu_view(&io.d_o.a)?, npu_view(&io.d_o.c)?);
        let (s_a, s_c) = (npu_view(&io.d_s.a)?, npu_view(&io.d_s.c)?);
        let (c_a, c_c) = (npu_view(&io.d_c.a)?, npu_view(&io.d_c.c)?);
        let fc1_a = npu_view(&io.d_fc1.a)?;
        let fc2_c = npu_view(&io.d_fc2.c)?;
        let kv_a = npu_view(&io.dec_kv.a)?;
        let head_a = npu_view(&io.m_head.a)?;

        // GPU memory: the FPN levels, the encoder stream, the pixel
        // decoder's maps (72 / 144 / 288 px), GroupNorm scratch
        let px = |s: usize| (s * s * d) as u64 * F32;
        let mut maps = vk.arena(3 * px(4 * g) + 4 * px(g) + 2 * px(2 * g) + (1 << 20))?;
        let fpn = [maps.take(px(4 * g))?, maps.take(px(2 * g))?, maps.take(px(g))?];
        let x = maps.take(px(g))?;
        let p = [maps.take(px(g))?, maps.take(px(2 * g))?, maps.take(px(4 * g))?];
        let valid = maps.take(l as u64 * 4)?;
        let gn_groups = 8;
        let parts = (16 * g * g).div_ceil(256);
        let part = maps.take((parts * gn_groups) as u64 * F32)?;
        let stats = maps.take(2 * gn_groups as u64 * F32)?;
        let zero = b.weights.put(&[0f32; 4])?;
        let pos = b.w("d.pos")?;

        let s = |v: usize| v.to_string();
        let bool_ = |v: bool| v.to_string();

        // -- neck
        let vit_in = match xres {
            Some(xres) => {
                let perm = b.store.i32("v.perm")?;
                let mut inv = vec![0u32; t];
                for (j, &r) in perm.iter().enumerate() {
                    inv[r as usize] = j as u32;
                }
                let inv = b.weights.put(&inv)?;
                let (xt, load) = if cfg.vit_res_f32 {
                    ("f32", "y[idx] = pack(xres[j * DIM + 2u * w], xres[j * DIM + 2u * w + 1u]);")
                } else {
                    ("u32", "y[idx] = xres[j * (DIM / 2u) + w];")
                };
                let mut rec = vk.record()?;
                let consts = [("XT", xt.to_string()), ("LOAD", load.to_string()), ("DIM", s(dim))];
                elementwise(&mut rec, "vit_in", VIT_IN, &consts, &[xres, inv, nin_a], t * dim / 2)?;
                Some(rec.finish()?)
            }
            None => None,
        };

        // Y of a conv of `side` px from `fetch` (bindings from 0; the
        // output binding after them)
        let conv_in = |rec: &mut Recorder,
                       label: &str,
                       side: usize,
                       decl: String,
                       fetch: String,
                       srcs: &[View],
                       y: View|
         -> Result<(), Error> {
            let wp = side + 2;
            let consts = [
                ("SRC_DECL", decl),
                ("FETCH", fetch),
                ("YB", s(srcs.len())),
                ("SIDE", s(side)),
                ("WP", s(wp)),
                ("D", s(dconv)),
                ("C", s(c)),
            ];
            let mut views = srcs.to_vec();
            views.push(y);
            elementwise(rec, label, CONV_IN, &consts, &views, side * wp * dconv / 2)
        };
        let bf16_src = "@group(0) @binding(0) var<storage, read> src: array<u32>;".to_string();

        let mut rec = vk.record()?;
        let consts = [("G", s(g)), ("WIDTH", s(width)), ("MID", s(s0 / 4))];
        elementwise(&mut rec, "neck_gelu", NECK_GELU, &consts, &[nin_c, nup_a], 4 * t * (s0 / 4) / 2)?;
        let fetch1 = format!(
            "fn fetch(y: u32, x: u32, ch: u32) -> u32 {{ return src[(((y / 2u) * {g}u + x / 2u) * {width}u + {s0}u \
             + ((y % 2u) * 2u + x % 2u) * {d}u + ch) / 2u]; }}"
        );
        conv_in(&mut rec, "neck_conv1_in", 2 * g, bf16_src.clone(), fetch1, &[nin_c], conv_a[1])?;
        let fetch2 = format!(
            "fn fetch(y: u32, x: u32, ch: u32) -> u32 {{ return src[((y * {g}u + x) * {width}u + {}u + ch) / 2u]; }}",
            s0 + s1
        );
        conv_in(&mut rec, "neck_conv2_in", g, bf16_src.clone(), fetch2, &[nin_c], conv_a[2])?;
        let neck_mid = rec.finish()?;

        let mut rec = vk.record()?;
        let fetch0 = format!(
            "fn fetch(y: u32, x: u32, ch: u32) -> u32 {{ return src[(((y / 2u) * {}u + x / 2u) * {n_up}u \
             + ((y % 2u) * 2u + x % 2u) * {d}u + ch) / 2u]; }}",
            2 * g
        );
        conv_in(&mut rec, "neck_conv0_in", 4 * g, bf16_src.clone(), fetch0, &[nup_c], conv_a[0])?;
        let neck_up = rec.finish()?;

        let conv_out =
            |b: &mut Builder, rec: &mut Recorder, label: &str, side: usize, cc: View, bias: &str, out: View| {
                let bias = b.w(bias)?;
                let consts = [("SIDE", s(side)), ("WP", s(side + 2)), ("OC", s(d))];
                elementwise(rec, label, CONV_OUT, &consts, &[cc, bias, out], side * side * d)
            };
        let mut rec = vk.record()?;
        for i in 0..3 {
            conv_out(b, &mut rec, "neck_conv_out", sides[i], conv_c[i], &format!("n.conv{i}.b"), fpn[i])?;
        }
        let neck_out = rec.finish()?;

        // -- DETR encoder
        let rows_bf16 = |b: &mut Builder,
                         rec: &mut Recorder,
                         label: &str,
                         xv: View,
                         ln: Option<&str>,
                         with_pos: bool,
                         y: View|
         -> Result<(), Error> {
            let (w, bb) = match ln {
                Some(n) => (b.w(&format!("{n}.w"))?, b.w(&format!("{n}.b"))?),
                None => (zero, zero),
            };
            let consts =
                [("DIM", s(d)), ("EPS", format!("{EPS:e}")), ("NORM", bool_(ln.is_some())), ("POS", bool_(with_pos))];
            let pv = if with_pos { pos } else { zero };
            rec.op(label, &format!("{COMMON}{ROWS_BF16}"), &consts, &[xv, w, bb, pv, y], [t as u32, 1, 1])
                .map_err(Error::from)
        };
        let add_bf16 = |rec: &mut Recorder, label: &str, xv: View, cc: View, bias: Option<View>| {
            let consts = [("DIM", s(d)), ("BIAS", bool_(bias.is_some()))];
            elementwise(rec, label, ADD_BF16, &consts, &[xv, cc, bias.unwrap_or(zero)], t * d)
        };

        let mut rec = vk.record()?;
        rec.copy(fpn[2], x);
        let enc_init = rec.finish()?;
        let mut rec = vk.record()?;
        let consts = [("PD2", s(pd / 2))];
        elementwise(&mut rec, "split_qkv", SPLIT_QKV, &consts, &[qkv_c, mq, mk, mv], t * pd / 2)?;
        let split = rec.finish()?;
        let mut rec = vk.record()?;
        rec.copy(mo.head((t * pd * 2) as u64), o_a);
        let copy_o = rec.finish()?;
        let mut rec = vk.record()?;
        let consts = [("NH", s(nh)), ("L", s(l))];
        elementwise(&mut rec, "prompt_softmax", PROMPT_SOFTMAX, &consts, &[s_c, valid, c_a], t * nh)?;
        let softmax = rec.finish()?;

        let (mut enc_ln1, mut enc_o, mut enc_c, mut enc_fc) = (vec![], vec![], vec![], vec![]);
        for i in 0..cfg.d_layers {
            let p = |n: &str| format!("d.{i}.{n}");
            let mut rec = vk.record()?;
            rows_bf16(b, &mut rec, "enc_ln1", x, Some(&p("ln1")), true, qkv_a)?;
            enc_ln1.push(rec.finish()?);
            let mut rec = vk.record()?;
            add_bf16(&mut rec, "enc_add_o", x, o_c, None)?;
            rows_bf16(b, &mut rec, "enc_ln2", x, Some(&p("ln2")), false, s_a)?;
            enc_o.push(rec.finish()?);
            let mut rec = vk.record()?;
            add_bf16(&mut rec, "enc_add_ctx", x, c_c, None)?;
            rows_bf16(b, &mut rec, "enc_ln3", x, Some(&p("ln3")), false, fc1_a)?;
            enc_c.push(rec.finish()?);
            let mut rec = vk.record()?;
            let bias = b.w(&p("fc2.b"))?;
            add_bf16(&mut rec, "enc_add_fc2", x, fc2_c, Some(bias))?;
            enc_fc.push(rec.finish()?);
        }
        let mut rec = vk.record()?;
        rows_bf16(b, &mut rec, "dec_kv_in", x, None, true, kv_a)?;
        let dec_kv_in = rec.finish()?;

        // -- mask decoder
        let mut rec = vk.record()?;
        rec.copy(x, p[0]);
        rows_bf16(b, &mut rec, "mask_ln", x, Some("m.ca_ln"), false, s_a)?;
        let mask_ln = rec.finish()?;
        let mut rec = vk.record()?;
        add_bf16(&mut rec, "mask_add_ctx", p[0], c_c, None)?;
        let mask_add = rec.finish()?;

        let (mut mask_up, mut mask_gn) = (vec![], vec![]);
        for i in 0..2 {
            let (side, s2) = (g << i, g << (i + 1));
            let conv_i = if i == 0 { 1 } else { 0 }; // the 144 px conv, then the 288 px one
            let skip = fpn[conv_i];
            let decl = "@group(0) @binding(0) var<storage, read> pm: array<f32>;\n\
                        @group(0) @binding(1) var<storage, read> skip: array<f32>;"
                .to_string();
            let fetch = format!(
                "fn fetch(y: u32, x: u32, ch: u32) -> u32 {{ let i = ((y / 2u) * {side}u + x / 2u) * {d}u + ch; \
                 let k = (y * {s2}u + x) * {d}u + ch; return pack(pm[i] + skip[k], pm[i + 1u] + skip[k + 1u]); }}"
            );
            let mut rec = vk.record()?;
            conv_in(&mut rec, "mask_up", s2, decl, fetch, &[p[i], skip], conv_a[conv_i])?;
            mask_up.push(rec.finish()?);

            let mut rec = vk.record()?;
            let out = p[i + 1];
            conv_out(b, &mut rec, "mask_conv_out", s2, conv_c[conv_i], &format!("m.conv{i}.b"), out)?;
            let npx = s2 * s2;
            let nparts = npx.div_ceil(256);
            for mode in 0..2 {
                let consts = [("PX", s(npx)), ("C", s(d)), ("G", s(gn_groups)), ("MODE", s(mode))];
                rec.op("gn_part", GN_PART, &consts, &[out, stats, part], [nparts as u32, 1, 1])?;
                let consts = [
                    ("PARTS", s(nparts)),
                    ("G", s(gn_groups)),
                    ("COUNT", format!("{:.1}", (npx * d / gn_groups) as f64)),
                    ("MODE", s(mode)),
                    ("EPS", format!("{EPS:e}")),
                ];
                rec.op("gn_reduce", GN_REDUCE, &consts, &[part, stats], [1, 1, 1])?;
            }
            let (w, bb) = (b.w(&format!("m.gn{i}.w"))?, b.w(&format!("m.gn{i}.b"))?);
            let last = i == 1;
            let consts = [("C", s(d)), ("G", s(gn_groups)), ("BF", bool_(last))];
            let y = if last { head_a } else { zero };
            elementwise(&mut rec, "gn_apply", GN_APPLY, &consts, &[out, stats, w, bb, y], npx * d / 2)?;
            mask_gn.push(rec.finish()?);
        }

        Ok(Glue {
            vit_in,
            neck_mid,
            neck_up,
            neck_out,
            enc_init,
            enc_ln1,
            split,
            copy_o,
            enc_o,
            softmax,
            enc_c,
            enc_fc,
            dec_kv_in,
            mask_ln,
            mask_add,
            mask_up,
            mask_gn,
            fpn,
            x,
            valid,
            vit_gelu: None,
        })
    }
}

/// tanh GELU of a GEMM's bf16 output, in place (the NPU epilogue's
/// `GELU_TANH`, in f32).
const GELU_TANH_INPLACE: &str = r#"
@group(0) @binding(0) var<storage, read_write> c: array<u32>;
const N: u32 = {N}u;

fn g(x: f32) -> f32 {
    return 0.5 * x * (1.0 + tanh(0.7978846 * (x + 0.044715 * x * x * x)));
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    {IDX}
    if (idx >= N) {
        return;
    }
    let w = c[idx];
    c[idx] = pack(g(lo(w)), g(hi(w)));
}
"#;

/// The plan applying GELU to all of `fc1`'s output (`rows x N` bf16).
pub(crate) fn vit_gelu_plan(vk: &mut Vk, fc1: &crate::npu::Io, spec: &crate::bundle::GemmSpec) -> Result<Plan, Error> {
    let c = vk.import_npu(&fc1.c)?;
    let mut rec = vk.record()?;
    elementwise(&mut rec, "vit_gelu", GELU_TANH_INPLACE, &[], &[c], fc1.rows * spec.n / 2)?;
    rec.finish().map_err(Error::from)
}

/// The words that turn `flm.GEMM`'s epilogue off in an instruction stream
/// that sets it to GELU_TANH (mode 5): the stream writes each core's RTP
/// `[n_work, n_drain, mode]` with 6-word write32 ops, the mode at RTP + 8,
/// 24 words a core after a 4-word header. `None` unless the stream is
/// exactly that shape (32 cores, every mode 5), so another bundle's layout
/// is never patched blind.
pub(crate) fn fc1_epilogue_off(insts: &[u32]) -> Option<Vec<(usize, u32)>> {
    const CORES: usize = 32;
    let mut words = Vec::with_capacity(CORES);
    for i in 0..CORES {
        let (addr, val) = (18 + 24 * i, 20 + 24 * i);
        if *insts.get(val)? != 5 || insts.get(addr)? & 0xfff != 0x408 || insts[val + 1] != 0x18 {
            return None;
        }
        words.push((val, 0));
    }
    Some(words)
}
