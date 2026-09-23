// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! GAIC (Grid Anchor based Image Cropping, VGG16 backbone) on an AMD XDNA
//! NPU, replaying the bundle `iron/applications/gaic/export_gaic.py`
//! writes: no Python, no ONNX, just XRT through [`taconite`].
//!
//! [`Gaic::features`] runs the backbone once per image — the 13 VGG16
//! convs, each one flm.GEMM over an im2col *view* (see
//! `iron/applications/gaic/gaic_npu.py`) — and reduces it to the 32-channel
//! 1/16-scale map every crop is scored from; [`Gaic::score`] scores any
//! number of candidate boxes against it (RoI + RoD align on the host, the
//! 5184 -> 768 FC on the NPU, 768 -> 128 -> 1 on the host). [`anchors`]
//! makes GAIC's candidate sets and [`preprocess`] the network input, both
//! matching GAIC-Pytorch's demo exactly.
//!
//! Host glue per conv, in one threaded pass: the GEMM's output (bf16,
//! pixel-major `[H][W + 2][OC]`, two junk columns a row) -> + bias, ReLU,
//! [2x2 max-pool] -> the next conv's A source written straight into the
//! shared input buffer. The activation buffers are host_only (uncached)
//! BOs, so rows move through them with whole-row copies.
//!
//! One [`Gaic`] per process: its kernels stay resident as 9 of the NPU's 16
//! hardware contexts. `Send`, not `Sync`.

pub mod align;
pub mod anchors;
pub mod bundle;
mod par;
pub mod preprocess;

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fmt;
use std::path::Path;
use std::time::{Duration, Instant};

use taconite::{bf16_to_f32, f32_to_bf16};
// The NPU path: XRT (feature `xrt`, the default), or the driver's ioctls
// with no XRT (feature `direct`, which wins when both are on).
#[cfg(feature = "direct")]
use taconite::direct::{Buffer, Kernel, Run, Session};
#[cfg(all(feature = "xrt", not(feature = "direct")))]
use taconite::{Buffer, Kernel, Run, Session};
#[cfg(not(any(feature = "xrt", feature = "direct")))]
compile_error!("no NPU path: enable feature `xrt` (the default) or `direct`");

use par::par_rows;

use bundle::{Bundle, ConvSpec};

#[derive(Debug)]
pub enum Error {
    /// The bundle is missing, malformed, or does not match this runtime.
    Bundle(String),
    /// XRT: the device, a kernel load, a buffer or a run.
    Npu(taconite::Error),
    /// An input the model cannot take (size, box count).
    Input(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Bundle(m) => write!(f, "GAIC bundle: {m}"),
            Error::Npu(e) => write!(f, "{e}"),
            Error::Input(m) => write!(f, "GAIC input: {m}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<taconite::Error> for Error {
    fn from(e: taconite::Error) -> Self {
        Error::Npu(e)
    }
}

/// Launches kept in flight per layer: enough to hide the host's turnaround
/// between chunks, few enough to stay clear of the driver's command queue.
const IN_FLIGHT: usize = 8;

/// Wall time per stage, accumulated since the last [`Gaic::reset_timing`].
#[derive(Debug, Clone, Default)]
pub struct Timing {
    pub stages: BTreeMap<&'static str, Duration>,
    pub dispatches: usize,
}

impl Timing {
    fn add(&mut self, stage: &'static str, d: Duration) {
        *self.stages.entry(stage).or_default() += d;
    }
    pub fn total(&self) -> Duration {
        self.stages.values().sum()
    }
}

impl fmt::Display for Timing {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (k, v) in &self.stages {
            write!(f, "{k} {:.1} ms, ", v.as_secs_f64() * 1e3)?;
        }
        write!(f, "{} dispatches", self.dispatches)
    }
}

/// The reduced feature map of one image: `[reddim][h][w]` f32 at 1/16 of
/// the network input (`input_w x input_h`), what every box is scored from.
#[derive(Debug, Clone)]
pub struct Features {
    pub input_w: usize,
    pub input_h: usize,
    pub h: usize,
    pub w: usize,
    pub channels: usize,
    pub map: Vec<f32>,
}

struct Layer {
    spec: ConvSpec,
    /// The GEMM's K (the window layer's is padded past P x 9C).
    k: usize,
    a_elems: usize,
    c_elems: usize,
    b: Buffer,
    bias: Vec<f32>,
}

/// Per input size: the shared A-source / output buffers and every chunk's
/// sub-buffer views of them (XRT caches a run per argument tuple, so these
/// live as long as the size does).
struct Plan {
    w: usize,
    h: usize,
    y: Buffer,
    c: Buffer,
    layers: Vec<LayerPlan>,
}

struct LayerPlan {
    /// The conv's input (and output) size, after any pool before it.
    h: usize,
    w: usize,
    /// Output pixels computed: h x (w + 2), padded to whole dispatches.
    pixels: usize,
    chunks: Vec<(Buffer, Buffer)>,
}

pub struct Gaic {
    bundle: Bundle,
    session: Session,
    kernels: HashMap<String, Kernel>,
    layers: Vec<Layer>,
    f3: usize,
    f4: usize,
    dimred_w: Vec<f32>,
    dimred_b: Vec<f32>,
    fc1_a: Buffer,
    fc1_b: Buffer,
    fc1_c: Buffer,
    fc1_bias: Vec<f32>,
    fc2_w: Vec<f32>,
    fc2_b: Vec<f32>,
    fc3_w: Vec<f32>,
    fc3_b: f32,
    plan: Option<Plan>,
    threads: usize,
    pub timing: Timing,
}

fn round_up(x: usize, m: usize) -> usize {
    x.div_ceil(m) * m
}

/// `a . b` with eight partial sums, so the loop vectorises.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let (ca, cb) = (a.chunks_exact(8), b.chunks_exact(8));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    let mut s = acc.iter().sum::<f32>();
    for (x, y) in ra.iter().zip(rb) {
        s += x * y;
    }
    s
}

impl Gaic {
    /// Opens the NPU and loads a bundle: every kernel into a resident
    /// hardware context, every weight into a device buffer.
    pub fn load(bundle: &Path) -> Result<Self, Error> {
        let bundle = Bundle::load(bundle)?;
        let session = Session::open(0)?;
        let mut kernels = HashMap::new();
        for k in &bundle.kernels {
            let ops = 2 * (k.m * k.k * k.n) as u64;
            kernels.insert(k.key.clone(), session.load_kernel(&k.xclbin, &k.insts, Some(&k.name), ops)?);
        }
        // Packed weights; their sizes were checked against the kernels at load.
        let upload = |name: &str| -> Result<Buffer, Error> {
            let data = bundle.store.bytes(name)?;
            let mut b = session.alloc(data.len())?;
            b.write(data)?;
            Ok(b)
        };
        let mut layers = Vec::new();
        for c in &bundle.convs {
            let k = bundle.kernel(&c.kernel)?;
            layers.push(Layer {
                spec: c.clone(),
                k: k.k,
                a_elems: k.a_elems,
                c_elems: k.c_elems,
                b: upload(&c.b)?,
                bias: bundle.f32_vec(&c.bias)?,
            });
        }
        let idx = |name: &str| bundle.convs.iter().position(|c| c.name == name).unwrap();
        let (f3, f4) = (idx(&bundle.f3), idx(&bundle.f4));
        let fk = bundle.kernel(&bundle.fc1.kernel)?;
        let fc1_b = upload(&bundle.fc1.b)?;
        let fc1_a = session.alloc_of::<u16>(fk.a_elems)?;
        let fc1_c = session.alloc_of::<u16>(fk.c_elems)?;
        let m = &bundle;
        let dimred_w = m.f32_vec(&m.dimred_w)?;
        let dimred_b = m.f32_vec(&m.dimred_b)?;
        let fc1_bias = m.f32_vec(&m.fc1.bias)?;
        let fc2_w = m.f32_vec(&m.fc2_w)?;
        let fc2_b = m.f32_vec(&m.fc2_b)?;
        let fc3_w = m.f32_vec(&m.fc3_w)?;
        let fc3_b = m.f32_vec(&m.fc3_b)?[0];
        let threads = par::default_threads();
        Ok(Self {
            bundle,
            session,
            kernels,
            layers,
            f3,
            f4,
            dimred_w,
            dimred_b,
            fc1_a,
            fc1_b,
            fc1_c,
            fc1_bias,
            fc2_w,
            fc2_b,
            fc3_w,
            fc3_b,
            plan: None,
            threads,
            timing: Timing::default(),
        })
    }

    /// The loaded bundle: its layer table, and the tensor store (weights and
    /// the self-check's `ref.*` references).
    pub fn bundle(&self) -> &Bundle {
        &self.bundle
    }

    pub fn reset_timing(&mut self) {
        self.timing = Timing::default();
    }

    /// Host threads the glue uses (default: the machine's, up to 16).
    pub fn set_threads(&mut self, n: usize) {
        self.threads = n.max(1);
    }

    fn plan_for(&mut self, w: usize, h: usize) -> Result<(), Error> {
        if self.plan.as_ref().is_some_and(|p| p.w == w && p.h == h) {
            return Ok(());
        }
        self.plan = None; // free the previous size's buffers first
        let (mut hh, mut ww) = (h, w);
        let mut dims = Vec::new();
        let (mut y_max, mut c_max) = (0, 0);
        for l in &self.layers {
            let s = &l.spec;
            if s.pool_before {
                hh /= 2;
                ww /= 2;
            }
            let pixels = round_up(hh * (ww + 2), s.m_chunk * s.p);
            let a_src = if s.window { pixels / s.p * l.k } else { (pixels + 2) * s.d };
            y_max = y_max.max(a_src);
            c_max = c_max.max(pixels * s.oc);
            dims.push((hh, ww, pixels));
        }
        let y = self.session.alloc_of::<u16>(y_max)?;
        let c = self.session.alloc_of::<u16>(c_max)?;
        let mut layers = Vec::new();
        for (l, &(h, w, pixels)) in self.layers.iter().zip(&dims) {
            let s = &l.spec;
            let chunk_px = s.m_chunk * s.p;
            let step = if s.window { s.m_chunk * l.k } else { chunk_px * s.d };
            let chunks = (0..pixels / chunk_px)
                .map(|i| Ok((y.sub_of::<u16>(i * step, l.a_elems)?, c.sub_of::<u16>(i * chunk_px * s.oc, l.c_elems)?)))
                .collect::<Result<Vec<_>, Error>>()?;
            layers.push(LayerPlan { h, w, pixels, chunks });
        }
        self.plan = Some(Plan { w, h, y, c, layers });
        Ok(())
    }

    /// The backbone and DimRed for one image: `chw` is the normalized
    /// `[3][h][w]` input ([`preprocess::preprocess`]); `w` and `h` must be
    /// multiples of 32 (GAIC's own resize makes them so).
    pub fn features(&mut self, chw: &[f32], w: usize, h: usize) -> Result<Features, Error> {
        if w % 32 != 0 || h % 32 != 0 || w < 64 || h < 64 {
            return Err(Error::Input(format!("{w}x{h}: both sides must be multiples of 32, at least 64")));
        }
        if chw.len() != 3 * w * h {
            return Err(Error::Input(format!("{} values for a 3x{h}x{w} input", chw.len())));
        }
        self.plan_for(w, h)?;
        let threads = self.threads;

        // The stem's input: the image as bf16 [h][w][3].
        let t0 = Instant::now();
        let mut act = vec![0u16; w * h * 3];
        par_rows(&mut act, w * 3, threads, |r0, rows| {
            for (i, px) in rows.chunks_exact_mut(3).enumerate() {
                let p = r0 * w + i;
                for c in 0..3 {
                    px[c] = f32_to_bf16(chw[c * w * h + p]);
                }
            }
        });
        self.timing.add("glue", t0.elapsed());

        let (mut f3, mut f4) = (Vec::new(), Vec::new());
        let n = self.layers.len();
        for i in 0..n {
            let t0 = Instant::now();
            {
                let plan = self.plan.as_mut().unwrap();
                let lp = &plan.layers[i];
                build_a(&mut plan.y, &act, &self.layers[i], lp, threads);
            }
            self.timing.add("build_a", t0.elapsed());

            // Syncs go through each chunk's sub-buffer: the shared buffers
            // are sized for the largest layer, and syncing all of them for
            // a small one was most of the host's time.
            let t0 = Instant::now();
            let plan = self.plan.as_ref().unwrap();
            let lp = &plan.layers[i];
            let layer = &self.layers[i];
            let kernel = &self.kernels[&layer.spec.kernel];
            let mut runs: VecDeque<(Run, &Buffer)> = VecDeque::new();
            for (a, c) in &lp.chunks {
                if runs.len() == IN_FLIGHT {
                    let (r, c) = runs.pop_front().unwrap();
                    r.wait()?;
                    c.sync_from_device()?;
                }
                a.sync_to_device()?;
                runs.push_back((kernel.start(&[a, &layer.b, c])?, c));
            }
            for (r, c) in runs {
                r.wait()?;
                c.sync_from_device()?;
            }
            self.timing.dispatches += lp.chunks.len();
            self.timing.add("npu", t0.elapsed());

            // Read in place: measured as fast as a bulk copy out first.
            let t0 = Instant::now();
            let s = &layer.spec;
            let out = &plan.c.as_slice::<u16>()[..lp.pixels * s.oc];
            let pool = self.layers.get(i + 1).is_some_and(|l| l.spec.pool_before);
            if i == self.f3 || i == self.f4 {
                let f = epilogue_f32(out, lp.h, lp.w, s.oc, &layer.bias, threads);
                if i == self.f3 { f3 = f } else { f4 = f }
            }
            if i + 1 < n {
                act = epilogue_bf16(out, lp.h, lp.w, s.oc, &layer.bias, pool, threads);
            }
            self.timing.add("epilogue", t0.elapsed());
        }

        let t0 = Instant::now();
        let plan = self.plan.as_ref().unwrap();
        let (l3, l4) = (&plan.layers[self.f3], &plan.layers[self.f4]);
        let ch = self.layers[self.f4].spec.oc;
        let (h5, w5) = (l4.h / 2, l4.w / 2);
        let f5 = maxpool_f32(&f4, l4.h, l4.w, ch);
        let r = self.bundle.reddim;
        let cin = self.bundle.dimred_in;
        let proj = |f: &[f32], off: usize| -> Vec<f32> {
            let px = f.len() / ch;
            let mut g = vec![0f32; px * r];
            par_rows(&mut g, r, threads, |p0, rows| {
                for (j, o) in rows.chunks_exact_mut(r).enumerate() {
                    let x = &f[(p0 + j) * ch..(p0 + j + 1) * ch];
                    for (k, v) in o.iter_mut().enumerate() {
                        *v = dot(x, &self.dimred_w[k * cin + off..k * cin + off + ch]);
                    }
                }
            });
            g
        };
        let (h4, w4) = (l4.h, l4.w);
        let g3 = interp_align_corners(&proj(&f3, 0), l3.h, l3.w, r, h4, w4);
        let g4 = proj(&f4, ch);
        let g5 = interp_align_corners(&proj(&f5, 2 * ch), h5, w5, r, h4, w4);
        let mut map = vec![0f32; r * h4 * w4];
        for p in 0..h4 * w4 {
            for k in 0..r {
                let i = p * r + k;
                map[k * h4 * w4 + p] = (g3[i] + g4[i]) + (0.5 * g5[i] + self.dimred_b[k]);
            }
        }
        self.timing.add("dimred", t0.elapsed());
        Ok(Features { input_w: w, input_h: h, h: h4, w: w4, channels: r, map })
    }

    /// Scores `boxes` (`[x1, y1, x2, y2]` in the input's pixels) against an
    /// image's [`Features`]; higher is a better crop.
    pub fn score(&mut self, f: &Features, boxes: &[[f32; 4]]) -> Result<Vec<f32>, Error> {
        let m = &self.bundle;
        let (s, scale) = (m.align_size, m.spatial_scale);
        let (k, k_pad, n1, rows) = (m.fc1.k, m.fc1.k_pad, m.fc1.n, m.fc1.m);
        let threads = self.threads;
        let mut scores = Vec::with_capacity(boxes.len());
        for group in boxes.chunks(rows) {
            let t0 = Instant::now();
            let mut a = vec![0u16; rows * k_pad];
            par_rows(&mut a[..group.len() * k_pad], k_pad, threads, |b0, out| {
                let mut feat = vec![0f32; k];
                for (j, row) in out.chunks_exact_mut(k_pad).enumerate() {
                    align::box_features(&f.map, f.channels, f.h, f.w, group[b0 + j], s, scale, &mut feat);
                    for (d, &v) in row.iter_mut().zip(&feat) {
                        *d = f32_to_bf16(v);
                    }
                }
            });
            self.fc1_a.write(&a)?;
            self.timing.add("align", t0.elapsed());

            let t0 = Instant::now();
            self.kernels[&m.fc1.kernel].run(&[&self.fc1_a, &self.fc1_b, &self.fc1_c])?;
            self.fc1_c.sync_from_device()?;
            self.timing.dispatches += 1;
            self.timing.add("npu", t0.elapsed());

            let t0 = Instant::now();
            let c = self.fc1_c.as_slice::<u16>()[..group.len() * n1].to_vec();
            for row in c.chunks_exact(n1) {
                let h1: Vec<f32> =
                    row.iter().zip(&self.fc1_bias).map(|(&v, &b)| (bf16_to_f32(v) + b).max(0.0)).collect();
                let h2: Vec<f32> = (0..m.fc2_out)
                    .map(|o| (dot(&h1, &self.fc2_w[o * m.fc2_in..(o + 1) * m.fc2_in]) + self.fc2_b[o]).max(0.0))
                    .collect();
                scores.push(dot(&h2, &self.fc3_w) + self.fc3_b);
            }
            self.timing.add("fc", t0.elapsed());
        }
        Ok(scores)
    }
}

/// Writes layer `l`'s A source into `y`, from its input activation `act`
/// (bf16 `[h][w][C]`). Pixel `q` of the zero-bordered input (row pitch
/// `Wp = w + 2`) contributes `Y[q] = [X_pad[q] | X_pad[q + Wp] | X_pad[q + 2 Wp]]`
/// (3C, zero past `q = h Wp`); the view layers store Y itself, D wide,
/// the window layer P whole windows `[Y[p] | Y[p+1] | Y[p+2]]` a row.
fn build_a(y: &mut Buffer, act: &[u16], l: &Layer, lp: &LayerPlan, threads: usize) {
    let s = &l.spec;
    let (h, w, c) = (lp.h, lp.w, s.c);
    let wp = w + 2;
    let n = h * wp;
    // Y[q][dy*C .. (dy+1)*C] into `dst` (3C long).
    let y_row = |q: usize, dst: &mut [u16]| {
        if q >= n {
            dst.fill(0);
            return;
        }
        let (r, x) = (q / wp, q % wp);
        for dy in 0..3 {
            let d = &mut dst[dy * c..(dy + 1) * c];
            let rr = r + dy;
            if rr >= 1 && rr <= h && x >= 1 && x <= w {
                let src = ((rr - 1) * w + (x - 1)) * c;
                d.copy_from_slice(&act[src..src + c]);
            } else {
                d.fill(0);
            }
        }
    };
    const BLOCK: usize = 64;
    let (row_len, rows) = if s.window { (l.k, lp.pixels / s.p) } else { (s.d, lp.pixels + 2) };
    let dst = &mut y.as_mut_slice::<u16>()[..rows * row_len];
    par_rows(dst, row_len, threads, |r0, piece| {
        // Rows are assembled in cached memory and stored a block at a time:
        // the destination is an uncached host_only mapping.
        let mut buf = vec![0u16; BLOCK * row_len];
        for (b, out) in piece.chunks_mut(BLOCK * row_len).enumerate() {
            let tmp = &mut buf[..out.len()];
            for (j, row) in tmp.chunks_exact_mut(row_len).enumerate() {
                let m = r0 + b * BLOCK + j;
                if s.window {
                    for q in 0..s.p {
                        for dx in 0..3 {
                            let o = q * 9 * c + dx * 3 * c;
                            y_row(m * s.p + q + dx, &mut row[o..o + 3 * c]);
                        }
                    }
                    row[s.p * 9 * c..].fill(0);
                } else {
                    y_row(m, &mut row[..3 * c]);
                    row[3 * c..].fill(0);
                }
            }
            out.copy_from_slice(tmp);
        }
    });
}

/// GEMM output (bf16 `[pixels][OC]`, `pixels` = h x (w + 2) + padding) ->
/// the next conv's input, bf16 `relu(out + bias)` `[h][w][OC]`, 2x2
/// max-pooled when `pool` (pool and ReLU commute: both monotone).
fn epilogue_bf16(out: &[u16], h: usize, w: usize, oc: usize, bias: &[f32], pool: bool, threads: usize) -> Vec<u16> {
    let wp = w + 2;
    // Input pixel (y, x)'s OC values.
    let px = |y: usize, x: usize| &out[(y * wp + x) * oc..(y * wp + x + 1) * oc];
    let (ho, wo) = if pool { (h / 2, w / 2) } else { (h, w) };
    let mut act = vec![0u16; ho * wo * oc];
    par_rows(&mut act, wo * oc, threads, |y0, rows| {
        for (j, row) in rows.chunks_exact_mut(wo * oc).enumerate() {
            let y = y0 + j;
            for (x, d) in row.chunks_exact_mut(oc).enumerate() {
                if pool {
                    let (a, b) = (px(2 * y, 2 * x), px(2 * y, 2 * x + 1));
                    let (c, e) = (px(2 * y + 1, 2 * x), px(2 * y + 1, 2 * x + 1));
                    for o in 0..oc {
                        let m = bf16_to_f32(a[o]).max(bf16_to_f32(b[o])).max(bf16_to_f32(c[o])).max(bf16_to_f32(e[o]));
                        d[o] = f32_to_bf16((m + bias[o]).max(0.0));
                    }
                } else {
                    for ((d, &s), &b) in d.iter_mut().zip(px(y, x)).zip(bias) {
                        *d = f32_to_bf16((bf16_to_f32(s) + b).max(0.0));
                    }
                }
            }
        }
    });
    act
}

/// As [`epilogue_bf16`] without the pool, kept in f32: f3 / f4.
fn epilogue_f32(out: &[u16], h: usize, w: usize, oc: usize, bias: &[f32], threads: usize) -> Vec<f32> {
    let wp = w + 2;
    let mut f = vec![0f32; h * w * oc];
    par_rows(&mut f, w * oc, threads, |y0, rows| {
        for (j, row) in rows.chunks_exact_mut(w * oc).enumerate() {
            let src = &out[(y0 + j) * wp * oc..((y0 + j) * wp + w) * oc];
            for (d, s) in row.chunks_exact_mut(oc).zip(src.chunks_exact(oc)) {
                for ((d, &s), &b) in d.iter_mut().zip(s).zip(bias) {
                    *d = (bf16_to_f32(s) + b).max(0.0);
                }
            }
        }
    });
    f
}

fn maxpool_f32(f: &[f32], h: usize, w: usize, c: usize) -> Vec<f32> {
    let (ho, wo) = (h / 2, w / 2);
    let mut out = vec![0f32; ho * wo * c];
    for y in 0..ho {
        for x in 0..wo {
            for o in 0..c {
                let at = |yy: usize, xx: usize| f[(yy * w + xx) * c + o];
                out[(y * wo + x) * c + o] =
                    at(2 * y, 2 * x).max(at(2 * y, 2 * x + 1)).max(at(2 * y + 1, 2 * x)).max(at(2 * y + 1, 2 * x + 1));
            }
        }
    }
    out
}

/// `[h][w][c]` -> `[oh][ow][c]`, bilinear with align_corners=True, the way
/// torch's CPU upsample_bilinear2d computes it (f32 source index, the far
/// neighbour clamped at the edge).
fn interp_align_corners(g: &[f32], h: usize, w: usize, c: usize, oh: usize, ow: usize) -> Vec<f32> {
    let axis = |i: usize, o: usize| -> Vec<(usize, usize, f32, f32)> {
        let scale = if o > 1 { (i as f32 - 1.0) / (o as f32 - 1.0) } else { 0.0 };
        (0..o)
            .map(|d| {
                let src = scale * d as f32;
                let i0 = src as usize;
                let i1 = if i0 < i - 1 { i0 + 1 } else { i0 };
                let l1 = src - i0 as f32;
                (i0, i1, 1.0 - l1, l1)
            })
            .collect()
    };
    let (ys, xs) = (axis(h, oh), axis(w, ow));
    let mut out = vec![0f32; oh * ow * c];
    for (y, &(y0, y1, hy0, hy1)) in ys.iter().enumerate() {
        for (x, &(x0, x1, wx0, wx1)) in xs.iter().enumerate() {
            for k in 0..c {
                let at = |yy: usize, xx: usize| g[(yy * w + xx) * c + k];
                out[(y * ow + x) * c + k] =
                    hy0 * (wx0 * at(y0, x0) + wx1 * at(y0, x1)) + hy1 * (wx0 * at(y1, x0) + wx1 * at(y1, x1));
            }
        }
    }
    out
}
