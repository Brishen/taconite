// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pre- and post-processing, as HF's `Sam3ImageProcessor`:
//!
//! - [`preprocess`]: RGB8 -> resized to 1008 x 1008 with antialiased
//!   bilinear filtering (torchvision's `resize(antialias=True)` on uint8,
//!   which follows PIL's two-pass fixed-point resampler, reproduced here),
//!   scaled to [0, 1] and normalised with mean = std = 0.5, CHW f32.
//! - [`instances`]: `sigmoid(logit) * sigmoid(presence) > threshold`, boxes
//!   scaled to the image, masks = `sigmoid` upsampled bilinearly
//!   (`align_corners=False`) to the image and thresholded.

use crate::Output;
use crate::cpu::{par_rows, sigmoid};

/// One axis of the resampling: per output index, (first input index,
/// int16 weights), and the weights' fixed-point precision.
struct Axis {
    taps: Vec<(usize, Vec<i32>)>,
    precision: u32,
}

/// The triangle-filter weights of PIL's `precompute_coeffs` /
/// torch's `_compute_indices_min_size_weights_aa`, quantised as torch's
/// uint8 kernel does: to int16, at the largest precision (<= 22 bits) at
/// which the axis's largest weight still fits.
fn coeffs(in_size: usize, out_size: usize) -> Axis {
    let scale = in_size as f64 / out_size as f64;
    let filterscale = scale.max(1.0);
    let support = filterscale; // the triangle's support is 1
    let float: Vec<(usize, Vec<f64>)> = (0..out_size)
        .map(|xx| {
            let center = (xx as f64 + 0.5) * scale;
            let xmin = ((center - support + 0.5) as i64).max(0) as usize;
            let xmax = ((center + support + 0.5) as i64).min(in_size as i64) as usize;
            let mut k: Vec<f64> = (xmin..xmax)
                .map(|x| {
                    let t = ((x as f64 - center + 0.5) / filterscale).abs();
                    if t < 1.0 { 1.0 - t } else { 0.0 }
                })
                .collect();
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
    let taps = float
        .into_iter()
        .map(|(xmin, k)| {
            let s = (1u64 << precision) as f64;
            (xmin, k.iter().map(|&v| (v * s + if v < 0.0 { -0.5 } else { 0.5 }) as i32).collect())
        })
        .collect();
    Axis { taps, precision }
}

/// Antialiased bilinear resize of an interleaved RGB8 image, horizontal
/// pass first (PIL's and torch's order), each pass rounding to u8.
pub fn resize_rgb8(src: &[u8], w: usize, h: usize, ow: usize, oh: usize) -> Vec<u8> {
    let cx = coeffs(w, ow);
    let cy = coeffs(h, oh);
    let pass = |ss: i64, p: u32| (ss >> p).clamp(0, 255) as u8;
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

/// RGB8 `[h, w, 3]` -> the model input `[3, size, size]`.
pub fn preprocess(rgb: &[u8], w: usize, h: usize, size: usize) -> Vec<f32> {
    let r = resize_rgb8(rgb, w, h, size, size);
    let mut out = vec![0f32; 3 * size * size];
    for (p, px) in r.chunks(3).enumerate() {
        for ch in 0..3 {
            out[ch * size * size + p] = (px[ch] as f32 * (1.0 / 255.0) - 0.5) / 0.5;
        }
    }
    out
}

/// One detected instance, in image pixels.
pub struct Instance {
    pub score: f32,
    /// xyxy
    pub bbox: [f32; 4],
    /// `[h, w]`, row-major
    pub mask: Vec<bool>,
}

/// Post-processed instances for an `w x h` image (HF
/// `post_process_instance_segmentation`), in query order.
pub fn instances(out: &Output, size: usize, w: usize, h: usize, threshold: f32, mask_threshold: f32) -> Vec<Instance> {
    let pres = sigmoid(out.presence);
    let px = size * size;
    // source index / weight per destination coordinate (align_corners=False)
    let axis = |n_out: usize| -> Vec<(usize, usize, f32)> {
        let scale = size as f32 / n_out as f32;
        (0..n_out)
            .map(|d| {
                let s = ((d as f32 + 0.5) * scale - 0.5).max(0.0);
                let i0 = (s as usize).min(size - 1);
                let i1 = if i0 < size - 1 { i0 + 1 } else { i0 };
                (i0, i1, s - i0 as f32)
            })
            .collect()
    };
    let (ax, ay) = (axis(w), axis(h));
    let mut res = Vec::new();
    for (q, &logit) in out.logits.iter().enumerate() {
        let score = sigmoid(logit) * pres;
        if score <= threshold {
            continue;
        }
        let b = &out.boxes[q * 4..q * 4 + 4];
        let bbox = [b[0] * w as f32, b[1] * h as f32, b[2] * w as f32, b[3] * h as f32];
        let prob: Vec<f32> = out.masks[q * px..(q + 1) * px].iter().map(|&v| sigmoid(v)).collect();
        let mut mask = vec![false; w * h];
        par_rows(&mut mask, w, |y0, piece| {
            for (yi, row) in piece.chunks_mut(w).enumerate() {
                let (y0s, y1s, ly) = ay[y0 + yi];
                for (x, m) in row.iter_mut().enumerate() {
                    let (x0s, x1s, lx) = ax[x];
                    let top = prob[y0s * size + x0s] * (1.0 - lx) + prob[y0s * size + x1s] * lx;
                    let bot = prob[y1s * size + x0s] * (1.0 - lx) + prob[y1s * size + x1s] * lx;
                    *m = top * (1.0 - ly) + bot * ly > mask_threshold;
                }
            }
        });
        res.push(Instance { score, bbox, mask });
    }
    res
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resize_is_identity_at_the_same_size_and_flat_stays_flat() {
        let src: Vec<u8> = (0..4 * 3 * 3).map(|i| (i * 7 % 256) as u8).collect();
        assert_eq!(resize_rgb8(&src, 4, 3, 4, 3), src);
        let flat = vec![77u8; 10 * 6 * 3];
        assert!(resize_rgb8(&flat, 10, 6, 23, 17).iter().all(|&v| v == 77));
        assert!(resize_rgb8(&flat, 10, 6, 3, 2).iter().all(|&v| v == 77));
    }
}
