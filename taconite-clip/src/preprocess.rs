// SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
// SPDX-FileCopyrightText: The PyTorch authors (see LICENSE-PYTORCH)
// SPDX-FileCopyrightText: Copyright © 1997-2011 by Secret Labs AB
// SPDX-FileCopyrightText: Copyright © 1995-2011 by Fredrik Lundh and contributors
// SPDX-FileCopyrightText: Copyright © 2010 by Jeffrey 'Alex' Clark and contributors
// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause AND MIT-CMU
//
// The resampler ports torch's antialiased uint8 resize
// (aten/src/ATen/native/cpu/UpSampleKernel.cpp), BSD-3-Clause, which
// follows Pillow's (libImaging/Resample.c), MIT-CMU; their notices are in
// LICENSE-PYTORCH and LICENSE-PILLOW.

//! Image preprocessing, as HF's `CLIPImageProcessor` (torchvision backend):
//!
//! 1. resize so the shorter side is 224 (the longer `int(224 * long /
//!    short)`), antialiased bicubic on the uint8 image -- torchvision's
//!    `resize(antialias=True)`, whose uint8 kernel is PIL's two-pass
//!    fixed-point resampler with int16 weights, reproduced here;
//! 2. center crop 224 x 224 (offsets `int((size - 224) / 2)`);
//! 3. `(x - 255 mean) / (255 std)` in f32 (the processor fuses the 1/255
//!    rescale into the normalisation), CHW.

use taconite_sam3::cpu::par_rows;

/// Torch's (and PIL's) antialiasing bicubic, a = -0.5.
fn bicubic(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * A
    } else {
        0.0
    }
}

/// One axis of the resampling: per output index, (first input index,
/// int16 weights), and the weights' fixed-point precision.
struct Axis {
    taps: Vec<(usize, Vec<i32>)>,
    precision: u32,
}

/// torch's `_compute_indices_min_size_weights_aa` for the bicubic filter
/// (support 2, widened by the downscale factor), quantised as its uint8
/// kernel does: to int16 at the largest precision (<= 22 bits) at which the
/// axis's largest weight still fits.
fn coeffs(in_size: usize, out_size: usize) -> Axis {
    let scale = in_size as f64 / out_size as f64;
    let filterscale = scale.max(1.0);
    let support = 2.0 * filterscale;
    let float: Vec<(usize, Vec<f64>)> = (0..out_size)
        .map(|xx| {
            let center = (xx as f64 + 0.5) * scale;
            let xmin = ((center - support + 0.5) as i64).max(0) as usize;
            let xmax = ((center + support + 0.5) as i64).min(in_size as i64) as usize;
            let mut k: Vec<f64> = (xmin..xmax).map(|x| bicubic((x as f64 - center + 0.5) / filterscale)).collect();
            let ww: f64 = k.iter().sum();
            if ww != 0.0 {
                k.iter_mut().for_each(|v| *v /= ww);
            }
            (xmin, k)
        })
        .collect();
    let max_w = float.iter().flat_map(|(_, k)| k.iter().copied()).fold(0.0, f64::max);
    let mut precision = 0;
    while precision < 22 {
        if (0.5 + max_w * (1u64 << (precision + 1)) as f64) as i64 >= 1 << 15 {
            break;
        }
        precision += 1;
    }
    let s = (1u64 << precision) as f64;
    let taps = float
        .into_iter()
        .map(|(xmin, k)| (xmin, k.iter().map(|&v| (v * s + if v < 0.0 { -0.5 } else { 0.5 }) as i32).collect()))
        .collect();
    Axis { taps, precision }
}

/// Antialiased bicubic resize of an interleaved RGB8 image, horizontal pass
/// first (PIL's and torch's order), each pass rounding and clamping to u8;
/// a pass whose size does not change is skipped, as torch does.
pub fn resize_rgb8(src: &[u8], w: usize, h: usize, ow: usize, oh: usize) -> Vec<u8> {
    let pass = |ss: i64, p: u32| (ss >> p).clamp(0, 255) as u8;
    let tmp = if ow == w {
        src.to_vec()
    } else {
        let cx = coeffs(w, ow);
        let mut tmp = vec![0u8; h * ow * 3];
        par_rows(&mut tmp, ow * 3, |y0, piece| {
            for (yi, row) in piece.chunks_mut(ow * 3).enumerate() {
                let s = &src[(y0 + yi) * w * 3..][..w * 3];
                for (xx, (xmin, k)) in cx.taps.iter().enumerate() {
                    for ch in 0..3 {
                        let mut ss = 1i64 << (cx.precision - 1);
                        for (i, &kv) in k.iter().enumerate() {
                            ss += s[(xmin + i) * 3 + ch] as i64 * kv as i64;
                        }
                        row[xx * 3 + ch] = pass(ss, cx.precision);
                    }
                }
            }
        });
        tmp
    };
    if oh == h {
        return tmp;
    }
    let cy = coeffs(h, oh);
    let mut out = vec![0u8; oh * ow * 3];
    par_rows(&mut out, ow * 3, |y0, piece| {
        for (yi, row) in piece.chunks_mut(ow * 3).enumerate() {
            let (ymin, k) = &cy.taps[y0 + yi];
            for (j, o) in row.iter_mut().enumerate() {
                let mut ss = 1i64 << (cy.precision - 1);
                for (i, &kv) in k.iter().enumerate() {
                    ss += tmp[(ymin + i) * ow * 3 + j] as i64 * kv as i64;
                }
                *o = pass(ss, cy.precision);
            }
        }
    });
    out
}

/// The processor's settings (bundle params).
#[derive(Debug, Clone)]
pub struct Preprocess {
    pub shortest: usize,
    pub crop: usize,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl Preprocess {
    /// `(height, width)` after the shortest-edge resize.
    pub fn resized_size(&self, w: usize, h: usize) -> (usize, usize) {
        let s = self.shortest;
        let (short, long) = if w <= h { (w, h) } else { (h, w) };
        let long = (s as f64 * long as f64 / short as f64) as usize;
        if w <= h { (long, s) } else { (s, long) }
    }

    /// RGB8 `[h, w, 3]` -> the model input `[3, crop, crop]`.
    pub fn run(&self, rgb: &[u8], w: usize, h: usize) -> Vec<f32> {
        let (rh, rw) = self.resized_size(w, h);
        let r = resize_rgb8(rgb, w, h, rw, rh);
        let c = self.crop;
        // torchvision pads with zeros when the crop is larger (not for a
        // shortest-edge resize to the crop size, but kept exact)
        let (top, left) = ((rh as f64 - c as f64) / 2.0, (rw as f64 - c as f64) / 2.0);
        let (top, left) = (top as i64, left as i64);
        // the fused rescale: mean and std scaled by 1 / (1/255) in f32
        let k = (1.0f64 / (1.0f64 / 255.0)) as f32;
        let mean: Vec<f32> = self.mean.iter().map(|m| m * k).collect();
        let std: Vec<f32> = self.std.iter().map(|s| s * k).collect();
        let mut out = vec![0f32; 3 * c * c];
        for y in 0..c {
            for x in 0..c {
                let (sy, sx) = (y as i64 + top, x as i64 + left);
                for ch in 0..3 {
                    let v = if sy < 0 || sx < 0 || sy >= rh as i64 || sx >= rw as i64 {
                        0.0
                    } else {
                        r[(sy as usize * rw + sx as usize) * 3 + ch] as f32
                    };
                    out[ch * c * c + y * c + x] = (v - mean[ch]) / std[ch];
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_pass_is_exact() {
        let src: Vec<u8> = (0..6 * 5 * 3).map(|i| (i * 7 % 256) as u8).collect();
        assert_eq!(resize_rgb8(&src, 6, 5, 6, 5), src);
    }

    #[test]
    fn sizes_follow_the_shorter_side() {
        let p = Preprocess { shortest: 224, crop: 224, mean: [0.0; 3], std: [1.0; 3] };
        assert_eq!(p.resized_size(640, 480), (224, 298));
        assert_eq!(p.resized_size(480, 640), (298, 224));
    }

    #[test]
    fn weights_sum_to_one() {
        for (i, o) in [(640, 224), (224, 224), (100, 224)] {
            let a = coeffs(i, o);
            for (_, k) in &a.taps {
                let s: i32 = k.iter().sum();
                assert!((s - (1 << a.precision)).abs() <= k.len() as i32, "{i}->{o}: {s}");
            }
        }
    }
}
