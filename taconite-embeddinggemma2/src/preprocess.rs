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

//! Image preprocessing, as HF's `Gemma4ImageProcessor` (torchvision
//! backend):
//!
//! 1. resize, aspect ratio kept, to the largest size whose sides are
//!    multiples of 48 (3 x 3 blocks of 16-pixel patches) with at most
//!    `max_soft_tokens` x 9 patches -- antialiased bicubic on the uint8
//!    image, torchvision's `resize(antialias=True)`, whose uint8 kernel is
//!    PIL's two-pass fixed-point resampler with int16 weights, reproduced
//!    here; skipped when the size is already right;
//! 2. x / 255 in f32, patches of 16 x 16 x 3 (rows, columns, channels) in
//!    row-major patch order, each with its (x, y) grid position.

use taconite::cpu::par_rows;

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


/// `(height, width)` the processor resizes an `h` x `w` image to.
pub fn target_size(h: usize, w: usize, patch: usize, pool: usize, max_soft_tokens: usize) -> (usize, usize) {
    let max_patches = max_soft_tokens * pool * pool;
    let factor = ((max_patches * patch * patch) as f64 / (h * w) as f64).sqrt();
    let side = pool * patch;
    let mut th = (factor * h as f64 / side as f64).floor() as usize * side;
    let mut tw = (factor * w as f64 / side as f64).floor() as usize * side;
    let max_side = (max_patches / (pool * pool)) * side;
    if th == 0 && tw > 0 {
        th = side;
        tw = ((w as f64 / h as f64).floor() as usize * side).min(max_side);
    } else if tw == 0 && th > 0 {
        tw = side;
        th = ((h as f64 / w as f64).floor() as usize * side).min(max_side);
    }
    (th, tw)
}

/// An image's patches: pixel values `[n, patch * patch * 3]` in [0, 1] and
/// grid positions `[n]` (x, y), row-major over a `gh` x `gw` grid.
pub struct Patches {
    pub pixels: Vec<f32>,
    pub positions: Vec<(usize, usize)>,
    pub gh: usize,
    pub gw: usize,
}

/// RGB8 `[h, w, 3]` -> its patches (None: too thin to hold a 3 x 3 block).
pub fn patches(rgb: &[u8], w: usize, h: usize, patch: usize, pool: usize, max_soft_tokens: usize) -> Option<Patches> {
    let (th, tw) = target_size(h, w, patch, pool, max_soft_tokens);
    if th == 0 || tw == 0 {
        return None;
    }
    let img = if (th, tw) == (h, w) { rgb.to_vec() } else { resize_rgb8(rgb, w, h, tw, th) };
    let (gh, gw) = (th / patch, tw / patch);
    let pd = patch * patch * 3;
    let mut pixels = vec![0f32; gh * gw * pd];
    let mut positions = Vec::with_capacity(gh * gw);
    for py in 0..gh {
        for px in 0..gw {
            let o = &mut pixels[(py * gw + px) * pd..][..pd];
            for r in 0..patch {
                for c in 0..patch {
                    let s = ((py * patch + r) * tw + px * patch + c) * 3;
                    for ch in 0..3 {
                        o[(r * patch + c) * 3 + ch] = img[s + ch] as f32 * (1.0f32 / 255.0);
                    }
                }
            }
            positions.push((px, py));
        }
    }
    Some(Patches { pixels, positions, gh, gw })
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
    fn sizes_match_the_processor() {
        // the reference images' grids (HF's processor)
        assert_eq!(target_size(1764, 2646, 16, 3, 280), (39 * 16, 60 * 16));
        assert_eq!(target_size(480, 640, 16, 3, 280), (42 * 16, 57 * 16));
        assert_eq!(target_size(425, 640, 16, 3, 280), (39 * 16, 60 * 16));
    }
}
