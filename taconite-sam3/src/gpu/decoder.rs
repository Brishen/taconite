// SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
// SPDX-License-Identifier: Apache-2.0

//! The DETR decoder's query layers on the GPU: `detr.rs`'s CPU loop
//! (HF `Sam3DetrDecoder`), op for op, in f32, as one [`Plan`] of all six
//! layers.
//!
//! The vision keys and values are the `dec_kv` GEMM's output, read in
//! place from the NPU's buffer (bf16, `[T, layers x (K | V)]`) through a
//! dma-buf; per forward the host writes only the prompt (features and
//! mask) and reads back the boxes, the presence logits and the last
//! layer's normalised queries, which the CPU scores.

use std::time::Duration;

use super::ops::{Builder, F32, groups, label};
use super::{Plan, View, Vk};
use crate::cpu::sigmoid;
use crate::text::Text;
use crate::{Config, Error};

/// Box sine embedding: `[Q, 4]` cxcywh -> `[Q, 4 F]`, coordinates in the
/// order (y, x, w, h), sin on even and cos on odd features.
const BOX_SINE: &str = r#"
@group(0) @binding(0) var<storage, read> refb: array<f32>;
@group(0) @binding(1) var<storage, read> dim_t: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
const Q: u32 = {Q}u;
const F: u32 = {F}u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let idx = g.x;
    if (idx >= Q * 4u * F) {
        return;
    }
    let q = idx / (4u * F);
    let c = (idx % (4u * F)) / F;
    let i = idx % F;
    var ci = c;
    if (c == 0u) {
        ci = 1u;
    } else if (c == 1u) {
        ci = 0u;
    }
    let v = refb[q * 4u + ci] * 6.2831855 / dim_t[i];
    if (i % 2u == 0u) {
        y[idx] = sin(v);
    } else {
        y[idx] = cos(v);
    }
}
"#;

/// The box bias along one axis (`LO` 0: x, 1: y), `Sam3::rpb`'s: for
/// each query box and grid line, the log-scaled distances to the box's
/// two edges through the axis MLP (`2 -> HID`, relu, `-> H`), one thread
/// per (box, line), the weights in workgroup memory.
const RPB: &str = r#"
@group(0) @binding(0) var<storage, read> refb: array<f32>;
@group(0) @binding(1) var<storage, read> w1: array<f32>;
@group(0) @binding(2) var<storage, read> b1: array<f32>;
@group(0) @binding(3) var<storage, read> w2: array<f32>;
@group(0) @binding(4) var<storage, read> b2: array<f32>;
@group(0) @binding(5) var<storage, read_write> y: array<f32>;
@group(0) @binding(6) var<storage, read> dims: array<u32>;
const Q: u32 = {Q}u;
const G: u32 = {G}u;
const LO: u32 = {LO}u;
const HID: u32 = {HID}u;
const H: u32 = {H}u;
var<workgroup> sw1: array<f32, {HID2}>;
var<workgroup> sb1: array<f32, {HID}>;
var<workgroup> sw2: array<f32, {HHID}>;

fn enc(v: f32) -> f32 {
    let s = v * 8.0;
    return sign(s) * log2(abs(s) + 1.0) / 3.0;
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    for (var e = t; e < 2u * HID; e += 64u) {
        sw1[e] = w1[e];
    }
    for (var e = t; e < HID; e += 64u) {
        sb1[e] = b1[e];
    }
    for (var e = t; e < H * HID; e += 64u) {
        sw2[e] = w2[e];
    }
    workgroupBarrier();
    let idx = wg.x * 64u + t;
    if (idx >= Q * G) {
        return;
    }
    let q = idx / G;
    let p = f32(idx % G) / f32(G);
    let c = refb[q * 4u + LO];
    let half = 0.5 * refb[q * 4u + LO + 2u];
    let x0 = enc(p - (c - half));
    let x1 = enc(p - (c + half));
    var out: array<f32, {H}>;
    let hid_n = dims[0];
    for (var hh = 0u; hh < hid_n; hh++) {
        let hid = max(x0 * sw1[2u * hh] + x1 * sw1[2u * hh + 1u] + sb1[hh], 0.0);
        for (var o = 0u; o < H; o++) {
            out[o] += sw2[o * HID + hh] * hid;
        }
    }
    for (var o = 0u; o < H; o++) {
        y[idx * H + o] = out[o] + b2[o];
    }
}
"#;

/// Attention of `[LQ, D]` queries against `[LK, D]` f32 keys and values
/// (heads side by side), keys masked by `mask`; one workgroup per (query,
/// head), each thread an online softmax over its share of the keys.
const ATTN: &str = r#"
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read> mask: array<u32>;
@group(0) @binding(4) var<storage, read_write> o: array<f32>;
@group(0) @binding(5) var<storage, read> dims: array<u32>;
const LK: u32 = {LK}u;
const D: u32 = {D}u;
const HD: u32 = {HD}u;
const SCALE: f32 = {SCALE};
const T: u32 = 64u;
var<workgroup> wm: array<f32, 64>;
var<workgroup> wl: array<f32, 64>;
var<workgroup> wacc: array<f32, {T_HD}>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let i = wg.x;
    let h = wg.y;
    var qv: array<f32, {HD}>;
    for (var e = 0u; e < HD; e++) {
        qv[e] = q[i * D + h * HD + e];
    }
    var m = -3.0e38;
    var l = 0.0;
    var acc: array<f32, {HD}>;
    let lk = dims[0];
    for (var j = t; j < lk; j += T) {
        if (mask[j] == 0u) {
            continue;
        }
        let kb = j * D + h * HD;
        var s = 0.0;
        for (var e = 0u; e < HD; e++) {
            s += qv[e] * k[kb + e];
        }
        s *= SCALE;
        let mn = max(m, s);
        let c = exp(m - mn);
        let p = exp(s - mn);
        l = l * c + p;
        for (var e = 0u; e < HD; e++) {
            acc[e] = acc[e] * c + p * v[kb + e];
        }
        m = mn;
    }
    wm[t] = m;
    workgroupBarrier();
    var mx = -3.0e38;
    for (var u = 0u; u < T; u++) {
        mx = max(mx, wm[u]);
    }
    let f = exp(m - mx);
    wl[t] = l * f;
    for (var e = 0u; e < HD; e++) {
        wacc[t * HD + e] = acc[e] * f;
    }
    workgroupBarrier();
    if (t < HD) {
        var sa = 0.0;
        var sl = 0.0;
        for (var u = 0u; u < T; u++) {
            sa += wacc[u * HD + t];
            sl += wl[u];
        }
        var out = 0.0;
        if (sl > 0.0) {
            out = sa / sl;
        }
        o[i * D + h * HD + t] = out;
    }
}
"#;

/// The vision cross-attention: `[R, D]` queries against one layer's keys
/// and values in the NPU's bf16 `dec_kv` output (rows of `KVW`), plus the
/// separable box bias `ry[q, iy, h] + rx[q, ix, h]` for every query but
/// the presence token (row 0). A workgroup takes `QB` queries of one head
/// through the keys in tiles of `KT` (unpacked to f32 in workgroup memory,
/// so each key is read once per `QB` queries), with an online softmax per
/// query. `kv` is bound from the layer's keys on, so its values are
/// `VOFF2` words further.
const VATTN: &str = r#"
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> kv: array<u32>;
@group(0) @binding(2) var<storage, read> ry: array<f32>;
@group(0) @binding(3) var<storage, read> rx: array<f32>;
@group(0) @binding(4) var<storage, read_write> o: array<f32>;
@group(0) @binding(5) var<storage, read> dims: array<u32>;
const R: u32 = {R}u;
const G: u32 = {G}u;
const D: u32 = {D}u;
const H: u32 = {H}u;
const KVW2: u32 = {KVW2}u;
const VOFF2: u32 = {VOFF2}u;
const SCALE: f32 = {SCALE};
const HD: u32 = 32u;
const QB: u32 = 16u;
const KT: u32 = 64u;
const HDP: u32 = 33u; // padded key rows
var<workgroup> qs: array<f32, 512>;   // [QB][HD]
var<workgroup> ks: array<f32, 2112>;  // [KT][HDP]
var<workgroup> vs: array<f32, 2048>;  // [KT][HD]
var<workgroup> ss: array<f32, 1024>;  // [QB][KT] probabilities

// max / sum over the 16 lanes of a query row (lanes 16-aligned in the subgroup)
fn row_max(v: f32) -> f32 {
    var m = v;
    m = max(m, subgroupShuffleXor(m, 1u));
    m = max(m, subgroupShuffleXor(m, 2u));
    m = max(m, subgroupShuffleXor(m, 4u));
    m = max(m, subgroupShuffleXor(m, 8u));
    return m;
}

fn row_sum(v: f32) -> f32 {
    var s = v;
    s += subgroupShuffleXor(s, 1u);
    s += subgroupShuffleXor(s, 2u);
    s += subgroupShuffleXor(s, 4u);
    s += subgroupShuffleXor(s, 8u);
    return s;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let i0 = wg.x * QB;
    let h = wg.y;
    for (var e = t; e < QB * HD; e += 256u) {
        let row = i0 + e / HD;
        var v = 0.0;
        if (row < R) {
            v = q[row * D + h * HD + e % HD] * SCALE;
        }
        qs[e] = v;
    }
    // this thread: query qi of the block; keys lane + 16 n of each tile;
    // outputs 2 lane, 2 lane + 1. Every lane of a row keeps the row's
    // running max and sum (the same values in all 16).
    let qi = t / 16u;
    let lane = t % 16u;
    let row = i0 + qi;
    let biased = row > 0u && row < R;
    var m_run = -3.0e38;
    var l_run = 0.0;
    var acc0 = 0.0;
    var acc1 = 0.0;
    let lk = dims[0];
    for (var k0 = 0u; k0 < lk; k0 += KT) {
        workgroupBarrier();
        for (var e = t; e < KT * HD / 2u; e += 256u) {
            let key = e / 16u;
            let p = e % 16u;
            let at = (k0 + key) * KVW2 + h * 16u + p;
            let wk = kv[at];
            let wv = kv[at + VOFF2];
            ks[key * HDP + 2u * p] = bitcast<f32>(wk << 16u);
            ks[key * HDP + 2u * p + 1u] = bitcast<f32>(wk & 0xffff0000u);
            vs[key * HD + 2u * p] = bitcast<f32>(wv << 16u);
            vs[key * HD + 2u * p + 1u] = bitcast<f32>(wv & 0xffff0000u);
        }
        workgroupBarrier();
        var sc: array<f32, 4>;
        var m_tile = -3.0e38;
        for (var n = 0u; n < 4u; n++) {
            let kj = lane + 16u * n;
            var s = 0.0;
            for (var d = 0u; d < HD; d++) {
                s += qs[qi * HD + d] * ks[kj * HDP + d];
            }
            if (biased) {
                let j = k0 + kj;
                s += ry[((row - 1u) * G + j / G) * H + h] + rx[((row - 1u) * G + j % G) * H + h];
            }
            sc[n] = s;
            m_tile = max(m_tile, s);
        }
        let m_new = max(m_run, row_max(m_tile));
        let c = exp(m_run - m_new);
        var l_tile = 0.0;
        for (var n = 0u; n < 4u; n++) {
            let pr = exp(sc[n] - m_new);
            ss[qi * KT + lane + 16u * n] = pr;
            l_tile += pr;
        }
        l_run = l_run * c + row_sum(l_tile);
        m_run = m_new;
        workgroupBarrier();
        var a0 = 0.0;
        var a1 = 0.0;
        for (var j = 0u; j < KT; j++) {
            let pr = ss[qi * KT + j];
            a0 += pr * vs[j * HD + 2u * lane];
            a1 += pr * vs[j * HD + 2u * lane + 1u];
        }
        acc0 = acc0 * c + a0;
        acc1 = acc1 * c + a1;
    }
    if (row < R) {
        o[row * D + h * HD + 2u * lane] = acc0 / l_run;
        o[row * D + h * HD + 2u * lane + 1u] = acc1 / l_run;
    }
}
"#;

/// Box refinement: `refb = sigmoid(delta + inverse_sigmoid(refb))`, also
/// kept as layer `LAYER`'s boxes.
const BOX_UPDATE: &str = r#"
@group(0) @binding(0) var<storage, read> delta: array<f32>;
@group(0) @binding(1) var<storage, read_write> refb: array<f32>;
@group(0) @binding(2) var<storage, read_write> boxes: array<f32>;
const N: u32 = {N}u;
const LAYER: u32 = {LAYER}u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = g.x;
    if (i >= N) {
        return;
    }
    let x = clamp(refb[i], 0.0, 1.0);
    let inv = log(max(x, 1e-3) / max(1.0 - x, 1e-3));
    let b = 1.0 / (1.0 + exp(-(delta[i] + inv)));
    refb[i] = b;
    boxes[LAYER * N + i] = b;
}
"#;

const PRESENCE_OUT: &str = r#"
@group(0) @binding(0) var<storage, read> p: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
const LAYER: u32 = {LAYER}u;

@compute @workgroup_size(1)
fn main() {
    out[LAYER] = clamp(p[0], -10.0, 10.0);
}
"#;

/// The recorded decoder and the views the host fills and reads.
pub struct Decoder {
    /// layer 0 up to its vision attention (no keys or values needed)
    pre: Plan,
    /// the rest
    plan: Plan,
    /// in: the prompt features `[L, D]` and mask `[L]` (1 = valid)
    text: View,
    mask: View,
    /// out: `[layers, Q, 4]` cxcywh boxes, `[layers]` presence logits, the
    /// last layer's normalised queries `[Q, D]`
    boxes: View,
    presence: View,
    normed: View,
    layers: usize,
    queries: usize,
    d: usize,
}

/// What a decoder run gives back.
pub struct Run {
    pub layer_boxes: Vec<Vec<f32>>,
    pub layer_presence: Vec<f32>,
    pub normed: Vec<f32>,
    pub gpu_time: Duration,
}

impl Decoder {
    /// Records all six layers over `kvv`, the GPU's view of the `dec_kv`
    /// GEMM's output (rows of `kvw` bf16).
    pub(crate) fn new(vk: &mut Vk, b: &mut Builder, cfg: &Config, kvv: View, kvw: usize) -> Result<Self, Error> {
        let (d, nh, q, l, g) = (cfg.d_model, cfg.d_heads, cfg.queries, cfg.text_len, cfg.grid);
        let (r, hd, ffn, layers) = (q + 1, d / nh, cfg.d_ffn, cfg.dec_layers);
        let store = b.store;
        // the constant starting state: presence token + query embeddings,
        // and the reference boxes through the sigmoid
        let mut hs0 = store.f32("dec.presence_token")?.to_vec();
        hs0.extend_from_slice(store.f32("dec.query_embed")?);
        let hs0 = b.weights.put(&hs0)?;
        let refb0: Vec<f32> = store.f32("dec.reference_points")?.iter().map(|&v| sigmoid(v)).collect();
        let refb0 = b.weights.put(&refb0)?;
        let f = d / 2;
        let dim_t: Vec<f32> = (0..f).map(|i| 10000f32.powf(2.0 * (i / 2) as f32 / f as f32)).collect();
        let dim_t = b.weights.put(&dim_t)?;
        let ones = b.weights.put(&vec![1u32; r])?;

        let text = b.act(l * d)?;
        let mask = b.acts.take(l as u64 * 4)?;
        let boxes = b.act(layers * q * 4)?;
        let presence = b.act(layers)?;
        let normed = b.act(q * d)?;
        let hs = b.act(r * d)?;
        let qpos = b.act(r * d)?; // row 0 (the presence token) stays zero
        let refb = b.act(q * 4)?;
        let sine = b.act(q * 2 * d)?;
        let h1 = b.act(r * ffn.max(d))?;
        let h2 = b.act(r * d)?;
        let qk = b.act(r * d)?;
        let (qq, kk, vv, att, o) = (b.act(r * d)?, b.act(r * d)?, b.act(r * d)?, b.act(r * d)?, b.act(r * d)?);
        let (tk, tv) = (b.act(l * d)?, b.act(l * d)?);
        let (ry, rx) = (b.act(q * g * nh)?, b.act(q * g * nh)?);
        let delta = b.act(q * 4)?;
        let pl = b.act(1)?;
        let row = |v: View, i: usize| v.at((i * d) as u64 * F32, d as u64 * F32);
        let rows_from_1 = |v: View| v.at(d as u64 * F32, (q * d) as u64 * F32);

        let scale = format!("{:e}", 1.0 / (hd as f32).sqrt());
        let t_hd = |t: usize| (t * hd).to_string();
        let mut rec = vk.record()?;
        let mut pre = None;
        rec.copy(hs0, hs);
        rec.copy(refb0, refb);
        for layer in 0..layers {
            let p = |s: &str| format!("dec.{layer}.{s}");
            // query positions from the boxes
            let consts = [("Q", q.to_string()), ("F", f.to_string())];
            rec.op("box_sine", BOX_SINE, &consts, &[refb, dim_t, sine], groups(q * 4 * f, 256))?;
            b.mlp(&mut rec, sine, q, 2 * d, "dec.ref_point_head", 2, [h1, h2], rows_from_1(qpos))?;
            // the box bias, per axis
            for (lo, name, out) in [(1, "dec.rpb_y", ry), (0, "dec.rpb_x", rx)] {
                let (w1, b1) = (b.w(&format!("{name}.1.w"))?, b.w(&format!("{name}.1.b"))?);
                let (w2, b2) = (b.w(&format!("{name}.2.w"))?, b.w(&format!("{name}.2.b"))?);
                let hid = (b1.len / F32) as usize;
                let consts = [
                    ("Q", q.to_string()),
                    ("G", g.to_string()),
                    ("LO", lo.to_string()),
                    ("HID", hid.to_string()),
                    ("H", nh.to_string()),
                    ("HID2", (2 * hid).to_string()),
                    ("HHID", (nh * hid).to_string()),
                ];
                let dims = b.dims(&[hid as u32])?;
                rec.op(&label(name), RPB, &consts, &[refb, w1, b1, w2, b2, out, dims], groups(q * g, 64))?;
            }

            // self-attention
            b.add(&mut rec, hs, qpos, qk, r * d)?;
            b.linear(&mut rec, qk, r, d, &p("sa.q"), qq, false, None)?;
            b.linear(&mut rec, qk, r, d, &p("sa.k"), kk, false, None)?;
            b.linear(&mut rec, hs, r, d, &p("sa.v"), vv, false, None)?;
            let consts = [
                ("LK", r.to_string()),
                ("D", d.to_string()),
                ("HD", hd.to_string()),
                ("SCALE", scale.clone()),
                ("T_HD", t_hd(64)),
            ];
            let dims = b.dims(&[r as u32])?;
            rec.op("sa.attn", ATTN, &consts, &[qq, kk, vv, ones, att, dims], [r as u32, nh as u32, 1])?;
            b.linear(&mut rec, att, r, d, &p("sa.o"), o, false, Some(hs))?;
            b.ln(&mut rec, o, r, d, &p("sa_ln"), hs)?;

            // text cross-attention
            b.add(&mut rec, hs, qpos, qk, r * d)?;
            b.linear(&mut rec, qk, r, d, &p("tca.q"), qq, false, None)?;
            b.linear(&mut rec, text, l, d, &p("tca.k"), tk, false, None)?;
            b.linear(&mut rec, text, l, d, &p("tca.v"), tv, false, None)?;
            let consts = [
                ("LK", l.to_string()),
                ("D", d.to_string()),
                ("HD", hd.to_string()),
                ("SCALE", scale.clone()),
                ("T_HD", t_hd(64)),
            ];
            let dims = b.dims(&[l as u32])?;
            rec.op("tca.attn", ATTN, &consts, &[qq, tk, tv, mask, att, dims], [r as u32, nh as u32, 1])?;
            b.linear(&mut rec, att, r, d, &p("tca.o"), o, false, Some(hs))?;
            b.ln(&mut rec, o, r, d, &p("tca_ln"), hs)?;

            // vision cross-attention with the box bias
            b.add(&mut rec, hs, qpos, qk, r * d)?;
            b.linear(&mut rec, qk, r, d, &p("vca.q"), qq, false, None)?;
            if layer == 0 {
                // everything so far reads only the prompt: it can run
                // while the NPU computes the keys and values
                pre = Some(rec.cut()?);
            }
            assert_eq!(hd, 32, "the vision attention kernel is built for heads of 32");
            assert_eq!(g * g % 64, 0, "the vision attention kernel takes keys 64 at a time");
            let consts = [
                ("R", r.to_string()),
                ("G", g.to_string()),
                ("D", d.to_string()),
                ("H", nh.to_string()),
                ("KVW2", (kvw / 2).to_string()),
                ("VOFF2", (d / 2).to_string()),
                ("SCALE", scale.clone()),
            ];
            // this layer's K | V columns on: 2 D bf16 a layer
            let koff = (layer * 2 * d * 2) as u64;
            let kv_layer = kvv.at(koff, kvv.len - koff);
            let dims = b.dims(&[(g * g) as u32])?;
            rec.op(
                "vca.attn",
                VATTN,
                &consts,
                &[qq, kv_layer, ry, rx, att, dims],
                [r.div_ceil(16) as u32, nh as u32, 1],
            )?;
            b.linear(&mut rec, att, r, d, &p("vca.o"), o, false, Some(hs))?;
            b.ln(&mut rec, o, r, d, &p("vca_ln"), hs)?;

            // MLP (post-norm)
            b.linear(&mut rec, hs, r, d, &p("fc1"), h1, true, None)?;
            b.linear(&mut rec, h1, r, ffn, &p("fc2"), o, false, Some(hs))?;
            b.ln(&mut rec, o, r, d, &p("mlp_ln"), hs)?;

            // box refinement on the queries, presence from token 0
            b.ln(&mut rec, rows_from_1(hs), q, d, "dec.out_ln", normed)?;
            b.mlp(&mut rec, normed, q, d, "dec.box_head", 3, [h1, h2], delta)?;
            let consts = [("N", (q * 4).to_string()), ("LAYER", layer.to_string())];
            rec.op("box_update", BOX_UPDATE, &consts, &[delta, refb, boxes], groups(q * 4, 256))?;
            b.ln(&mut rec, row(hs, 0), 1, d, "dec.presence_ln", row(o, 0))?;
            b.mlp(&mut rec, row(o, 0), 1, d, "dec.presence_head", 3, [h1, h2], pl)?;
            rec.op("presence_out", PRESENCE_OUT, &[("LAYER", layer.to_string())], &[pl, presence], [1, 1, 1])?;
        }
        let plan = rec.finish()?;
        Ok(Decoder {
            pre: pre.expect("the decoder has layers"),
            plan,
            text,
            mask,
            boxes,
            presence,
            normed,
            layers,
            queries: q,
            d,
        })
    }

    /// Dispatches in the recorded plans.
    pub fn dispatches(&self) -> usize {
        self.pre.dispatches + self.plan.dispatches
    }

    /// Starts the layers for `text` on the GPU, up to where they need the
    /// NPU's keys and values: call before the `dec_kv` GEMM, so the two
    /// overlap, then [`finish`](Self::finish) after it.
    pub fn start(&self, vk: &Vk, text: &Text) -> Result<(), Error> {
        self.text.write(&text.feats);
        let mask: Vec<u32> = text.valid.iter().map(|&v| v as u32).collect();
        self.mask.write(&mask);
        self.pre.submit(vk).map_err(Error::from)
    }

    /// The rest of the layers, once the `dec_kv` GEMM has run; `gpu_time`
    /// is from this submit to the results.
    pub fn finish(&self, vk: &Vk) -> Result<Run, Error> {
        self.plan.submit(vk)?;
        self.pre.wait(vk)?;
        let gpu_time = self.plan.wait(vk)?;
        // SAM3_GPU_BENCH=n: the whole decoder n more times (it is
        // idempotent), for timings steadier than one run's
        if let Some(n) = std::env::var("SAM3_GPU_BENCH").ok().and_then(|v| v.parse::<usize>().ok()) {
            let mut ts = Vec::with_capacity(n);
            for _ in 0..n {
                let t = std::time::Instant::now();
                self.pre.run(vk)?;
                self.plan.run(vk)?;
                ts.push(t.elapsed());
            }
            ts.sort();
            if let (Some(min), Some(mid)) = (ts.first(), ts.get(n / 2)) {
                eprintln!(
                    "decoder plan x{n}: min {:.2} ms, median {:.2} ms",
                    min.as_secs_f64() * 1e3,
                    mid.as_secs_f64() * 1e3
                );
            }
        }
        let (q, n) = (self.queries, self.layers);
        let boxes = self.boxes.read::<f32>(n * q * 4);
        Ok(Run {
            layer_boxes: boxes.chunks(q * 4).map(<[f32]>::to_vec).collect(),
            layer_presence: self.presence.read::<f32>(n),
            normed: self.normed.read::<f32>(q * self.d),
            gpu_time,
        })
    }

    /// [`start`](Self::start) and [`finish`](Self::finish) in one: the
    /// `dec_kv` GEMM must have run.
    pub fn run(&self, vk: &Vk, text: &Text) -> Result<Run, Error> {
        self.start(vk, text)?;
        self.finish(vk)
    }
}
