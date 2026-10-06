// SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
// SPDX-License-Identifier: Apache-2.0

//! Heatmaps -> keypoints, as HF's `post_process_pose_estimation`: each
//! heatmap's argmax (its value the score), refined by DARK / UDP -- one
//! Newton step on the log of the heatmap blurred by an 11 x 11 Gaussian
//! (sigma 2, zero border, rescaled to keep the heatmap's max) -- then
//! mapped from heatmap pixels back through the crop window.

use taconite::cpu::par_rows;

use crate::preprocess::Window;

/// One keypoint: image coordinates and its heatmap score.
#[derive(Debug, Clone, Copy)]
pub struct Keypoint {
    pub x: f32,
    pub y: f32,
    pub score: f32,
}

/// torchvision's Gaussian kernel of `k` taps (`sigma` as OpenCV derives it
/// from k when given 0).
fn kernel(k: usize) -> (Vec<f32>, f32) {
    let sigma = 0.3 * ((k as f32 - 1.0) * 0.5 - 1.0) + 0.8;
    let half = (k - 1) as f32 * 0.5;
    let pdf: Vec<f32> = (0..k)
        .map(|i| {
            let x = -half + i as f32;
            (-0.5 * (x / sigma) * (x / sigma)).exp()
        })
        .collect();
    let s: f32 = pdf.iter().sum();
    (pdf.into_iter().map(|v| v / s).collect(), sigma)
}

/// Blurs one `[h, w]` heatmap (zeros outside) by the separable kernel.
fn blur(m: &[f32], w: usize, h: usize, k: &[f32], out: &mut [f32], tmp: &mut [f32]) {
    let r = k.len() / 2;
    for y in 0..h {
        let row = &m[y * w..(y + 1) * w];
        for x in 0..w {
            let lo = x.saturating_sub(r);
            let hi = (x + r).min(w - 1);
            let mut s = 0f32;
            for (xx, &v) in row.iter().enumerate().take(hi + 1).skip(lo) {
                s += v * k[xx + r - x];
            }
            tmp[y * w + x] = s;
        }
    }
    for y in 0..h {
        let lo = y.saturating_sub(r);
        let hi = (y + r).min(h - 1);
        for x in 0..w {
            let mut s = 0f32;
            for yy in lo..=hi {
                s += tmp[yy * w + x] * k[yy + r - y];
            }
            out[y * w + x] = s;
        }
    }
}

/// heatmaps `[k, h, w]` -> per keypoint (x, y) in heatmap pixels after
/// DARK, and the score.
pub fn decode(heatmaps: &[f32], k: usize, h: usize, w: usize, blur_k: usize) -> Vec<(f32, f32, f32)> {
    let (ker, _) = kernel(blur_k);
    let mut res = vec![0f32; 3 * k];
    par_rows(&mut res, 3, |k0, piece| {
        let mut b = vec![0f32; h * w];
        let mut tmp = vec![0f32; h * w];
        for (ki, out) in piece.chunks_mut(3).enumerate() {
            let m = &heatmaps[(k0 + ki) * h * w..(k0 + ki + 1) * h * w];
            // the first maximum, as torch's argmax
            let (mut best, mut idx) = (f32::NEG_INFINITY, 0);
            for (i, &v) in m.iter().enumerate() {
                if v > best {
                    best = v;
                    idx = i;
                }
            }
            let (mut x, mut y) = if best > 0.0 { ((idx % w) as f32, (idx / w) as f32) } else { (-1.0, -1.0) };
            blur(m, w, h, &ker, &mut b, &mut tmp);
            let bmax = b.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let scale = if bmax > 0.0 { best / bmax } else { 1.0 };
            // the log heatmap, edge-replicated, around (x, y)
            let lv = |xx: i64, yy: i64| -> f32 {
                let xx = xx.clamp(0, w as i64 - 1) as usize;
                let yy = yy.clamp(0, h as i64 - 1) as usize;
                (b[yy * w + xx] * scale).clamp(1e-3, 50.0).ln()
            };
            // HF indexes the replicate-padded map at (x + 1, y + 1) from the
            // truncated (x, y): -1 lands on the padding's first row/column
            let (cx, cy) = (x as i64, y as i64);
            let v = |dx: i64, dy: i64| lv(cx + dx, cy + dy);
            let gx = 0.5 * (v(1, 0) - v(-1, 0));
            let gy = 0.5 * (v(0, 1) - v(0, -1));
            let eps = f32::EPSILON;
            let hxx = v(1, 0) - 2.0 * v(0, 0) + v(-1, 0) + eps;
            let hyy = v(0, 1) - 2.0 * v(0, 0) + v(0, -1) + eps;
            let hxy = 0.5 * (v(1, 1) - v(1, 0) - v(0, 1) + v(0, 0) + v(0, 0) - v(-1, 0) - v(0, -1) + v(-1, -1));
            let det = hxx * hyy - hxy * hxy;
            x -= (hyy * gx - hxy * gy) / det;
            y -= (-hxy * gx + hxx * gy) / det;
            out.copy_from_slice(&[x, y, best]);
        }
    });
    res.chunks(3).map(|c| (c[0], c[1], c[2])).collect()
}

/// Heatmap pixels -> image coordinates through the crop window.
pub fn to_image(p: &[(f32, f32, f32)], win: &Window, h: usize, w: usize) -> Vec<Keypoint> {
    p.iter()
        .map(|&(x, y, score)| Keypoint {
            x: x / (w - 1) as f32 * win.w + win.cx - 0.5 * win.w,
            y: y / (h - 1) as f32 * win.h + win.cy - 0.5 * win.h,
            score,
        })
        .collect()
}
