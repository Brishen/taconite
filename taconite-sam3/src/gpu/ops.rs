// SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
// SPDX-License-Identifier: Apache-2.0

//! Ops the GPU stages share: Linear, LayerNorm and add shaders, and the
//! [`Builder`] recording them over the bundle's weights.

use std::collections::HashMap;

use super::{Arena, Recorder, View};
use crate::Error;
use crate::bundle::Store;

pub(crate) const EPS: f32 = 1e-5;
pub(crate) const F32: u64 = 4;

/// `y[r, o] = x[r, :] . w[o, :] + b[o]` (a torch Linear) -- then `+ res[r, o]`
/// or `relu` per the epilogue. A workgroup computes a 32 x 32 tile of `y`,
/// each thread 2 x 2 of it, through 32 x 16 tiles of `x` and `w` in
/// workgroup memory.
pub(crate) const LINEAR: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
{RES_DECL}
{DIMS_DECL}
const ROWS: u32 = {ROWS}u;
const NIN: u32 = {NIN}u;
const NOUT: u32 = {NOUT}u;
// [k][row] and [k][out], rows padded against bank conflicts
var<workgroup> xs: array<array<f32, 33>, 16>;
var<workgroup> ws: array<array<f32, 33>, 16>;

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let r0 = wg.y * 32u;
    let o0 = wg.x * 32u;
    let ty = t / 16u;
    let tx = t % 16u;
    var acc: array<f32, 4>;
    // the bound from a buffer, so the loop stays a loop (a constant one
    // is unrolled whole, and compiles for seconds)
    let nin = dims[0];
    for (var k0 = 0u; k0 < nin; k0 += 16u) {
        // 512 values of each tile, 2 per thread: (row or out) = e / 16, k = e % 16
        for (var n = 0u; n < 2u; n++) {
            let e = t + 256u * n;
            let rr = e / 16u;
            let kk = e % 16u;
            let k = k0 + kk;
            var xv = 0.0;
            if (r0 + rr < ROWS && k < NIN) {
                xv = x[(r0 + rr) * NIN + k];
            }
            xs[kk][rr] = xv;
            var wv = 0.0;
            if (o0 + rr < NOUT && k < NIN) {
                wv = w[(o0 + rr) * NIN + k];
            }
            ws[kk][rr] = wv;
        }
        workgroupBarrier();
        for (var kk = 0u; kk < 16u; kk++) {
            let a0 = xs[kk][2u * ty];
            let a1 = xs[kk][2u * ty + 1u];
            let w0 = ws[kk][2u * tx];
            let w1 = ws[kk][2u * tx + 1u];
            acc[0] += a0 * w0;
            acc[1] += a0 * w1;
            acc[2] += a1 * w0;
            acc[3] += a1 * w1;
        }
        workgroupBarrier();
    }
    for (var n = 0u; n < 4u; n++) {
        let r = r0 + 2u * ty + n / 2u;
        let o = o0 + 2u * tx + n % 2u;
        if (r < ROWS && o < NOUT) {
            var v = acc[n] + b[o];
            {EPILOGUE}
            y[r * NOUT + o] = v;
        }
    }
}
"#;

/// LayerNorm over rows of `DIM`, one workgroup a row.
pub(crate) const LAYERNORM: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
const DIM: u32 = {DIM}u;
const EPS: f32 = {EPS};
var<workgroup> red: array<f32, 256>;

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
    let base = wg.x * DIM;
    var s = 0.0;
    for (var j = t; j < DIM; j += 256u) {
        s += x[base + j];
    }
    red[t] = s;
    workgroupBarrier();
    let mean = total(t) / f32(DIM);
    workgroupBarrier();
    var v = 0.0;
    for (var j = t; j < DIM; j += 256u) {
        let dd = x[base + j] - mean;
        v += dd * dd;
    }
    red[t] = v;
    workgroupBarrier();
    let inv = 1.0 / sqrt(total(t) / f32(DIM) + EPS);
    for (var j = t; j < DIM; j += 256u) {
        y[base + j] = (x[base + j] - mean) * inv * w[j] + b[j];
    }
}
"#;

pub(crate) const ADD: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
const N: u32 = {N}u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    if (g.x < N) {
        y[g.x] = a[g.x] + b[g.x];
    }
}
"#;

/// A tensor name without its layer (`dec.3.sa.q` -> `sa.q`), for the profile.
pub(crate) fn label(name: &str) -> String {
    let rest = name.strip_prefix("dec.").unwrap_or(name);
    match rest.split_once('.') {
        Some((l, tail)) if l.parse::<usize>().is_ok() => tail.to_string(),
        _ => rest.to_string(),
    }
}

pub(crate) fn groups(n: usize, per: usize) -> [u32; 3] {
    [n.div_ceil(per) as u32, 1, 1]
}

/// Records ops over views of two arenas: the bundle's weights (uploaded
/// once, by name) and activations.
pub(crate) struct Builder<'s> {
    pub store: &'s Store,
    pub weights: Arena,
    pub acts: Arena,
    cache: HashMap<String, View>,
}

impl<'s> Builder<'s> {
    pub fn new(store: &'s Store, weights: Arena, acts: Arena) -> Self {
        Builder { store, weights, acts, cache: HashMap::new() }
    }

    /// Tensor `name` of the bundle, uploaded once.
    pub fn w(&mut self, name: &str) -> Result<View, Error> {
        if let Some(v) = self.cache.get(name) {
            return Ok(*v);
        }
        let v = self.weights.put(self.store.f32(name)?)?;
        self.cache.insert(name.to_string(), v);
        Ok(v)
    }

    /// A small buffer of loop bounds (`dims`) for a shader.
    pub fn dims(&mut self, v: &[u32]) -> Result<View, Error> {
        self.weights.put(v).map_err(Error::from)
    }

    pub fn act(&mut self, floats: usize) -> Result<View, Error> {
        self.acts.take(floats as u64 * F32).map_err(Error::from)
    }

    /// `y = x W^T + b` (`relu`'d, or `+ res`) over `rows` rows of `x`.
    #[allow(clippy::too_many_arguments)]
    pub fn linear(
        &mut self,
        rec: &mut Recorder,
        x: View,
        rows: usize,
        n_in: usize,
        name: &str,
        y: View,
        relu: bool,
        res: Option<View>,
    ) -> Result<(), Error> {
        let (w, b) = (self.w(&format!("{name}.w"))?, self.w(&format!("{name}.b"))?);
        let n_out = (w.len / F32) as usize / n_in;
        let (decl, epi) = match (relu, res) {
            (false, None) => ("", ""),
            (true, None) => ("", "v = max(v, 0.0);"),
            (false, Some(_)) => {
                ("@group(0) @binding(4) var<storage, read> res: array<f32>;", "v += res[r * NOUT + o];")
            }
            (true, Some(_)) => unreachable!("no layer here has both"),
        };
        let dims_at = if res.is_some() { 5 } else { 4 };
        let dims = self.dims(&[n_in as u32])?;
        let consts = [
            ("RES_DECL", decl.to_string()),
            ("DIMS_DECL", format!("@group(0) @binding({dims_at}) var<storage, read> dims: array<u32>;")),
            ("EPILOGUE", epi.to_string()),
            ("ROWS", rows.to_string()),
            ("NIN", n_in.to_string()),
            ("NOUT", n_out.to_string()),
        ];
        let x = x.head((rows * n_in) as u64 * F32);
        let y = y.head((rows * n_out) as u64 * F32);
        let g = [n_out.div_ceil(32) as u32, rows.div_ceil(32) as u32, 1];
        match res {
            Some(r) => {
                rec.op(&label(name), LINEAR, &consts, &[x, w, b, y, r.head(y.len), dims], g).map_err(Error::from)
            }
            None => rec.op(&label(name), LINEAR, &consts, &[x, w, b, y, dims], g).map_err(Error::from),
        }
    }

    /// `relu(l1) -> [relu(l2) ->] l_last`, as `Sam3::mlp`; the hidden
    /// activations in `tmp` (two views of `rows x hidden`).
    #[allow(clippy::too_many_arguments)]
    pub fn mlp(
        &mut self,
        rec: &mut Recorder,
        x: View,
        rows: usize,
        n_in: usize,
        name: &str,
        layers: usize,
        tmp: [View; 2],
        y: View,
    ) -> Result<(), Error> {
        let hidden = (self.w(&format!("{name}.1.w"))?.len / F32) as usize / n_in;
        self.linear(rec, x, rows, n_in, &format!("{name}.1"), tmp[0], true, None)?;
        if layers == 3 {
            self.linear(rec, tmp[0], rows, hidden, &format!("{name}.2"), tmp[1], true, None)?;
            self.linear(rec, tmp[1], rows, hidden, &format!("{name}.3"), y, false, None)
        } else {
            self.linear(rec, tmp[0], rows, hidden, &format!("{name}.2"), y, false, None)
        }
    }

    pub fn ln(
        &mut self,
        rec: &mut Recorder,
        x: View,
        rows: usize,
        dim: usize,
        name: &str,
        y: View,
    ) -> Result<(), Error> {
        let (w, b) = (self.w(&format!("{name}.w"))?, self.w(&format!("{name}.b"))?);
        let consts = [("DIM", dim.to_string()), ("EPS", format!("{EPS:e}"))];
        let bytes = (rows * dim) as u64 * F32;
        rec.op(&label(name), LAYERNORM, &consts, &[x.head(bytes), w, b, y.head(bytes)], [rows as u32, 1, 1])
            .map_err(Error::from)
    }

    pub fn add(&mut self, rec: &mut Recorder, a: View, b: View, y: View, n: usize) -> Result<(), Error> {
        let bytes = n as u64 * F32;
        rec.op("add", ADD, &[("N", n.to_string())], &[a.head(bytes), b.head(bytes), y.head(bytes)], groups(n, 256))
            .map_err(Error::from)
    }
}
