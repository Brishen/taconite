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

//! Image preprocessing, as the checkpoint's `Qwen2VLImageProcessorFast`
//! (torchvision backend) does it:
//!
//! 1. [`smart_resize`]: both sides a multiple of 32 (patch 16 x merge 2),
//!    the area within `[min_pixels, max_pixels]`, the aspect kept;
//! 2. an antialiased bicubic resize of the uint8 image -- torchvision's
//!    `resize(antialias=True)`, whose uint8 kernel is PIL's two-pass
//!    fixed-point resampler with int16 weights ([`resize_rgb8`], the same
//!    code as taconite-clip's);
//! 3. `x / 127.5 - 1` in f32 (mean = std = 0.5, the 1/255 rescale fused);
//! 4. [`patchify`]: 16 x 16 patches, (channel, row, column) inside a patch,
//!    in 2 x 2 merge-block order -- one temporal frame of what the
//!    processor emits (it repeats an image as two identical frames; the
//!    patch embedding's temporal kernels are summed at export).

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

/// The processor's target `(height, width)` for an `h` x `w` image.
pub fn smart_resize(h: usize, w: usize, factor: usize, min_pixels: usize, max_pixels: usize) -> (usize, usize) {
    // Python's round(): half to even
    let round = |x: f64| {
        let r = x.round();
        if (x - x.trunc()).abs() == 0.5 && r % 2.0 != 0.0 { r - x.signum() } else { r }
    };
    let f = factor as f64;
    let (hf, wf) = (h as f64, w as f64);
    let mut hb = (round(hf / f) * f) as usize;
    let mut wb = (round(wf / f) * f) as usize;
    if hb * wb > max_pixels {
        let beta = (hf * wf / max_pixels as f64).sqrt();
        hb = factor.max(((hf / beta / f).floor() * f) as usize);
        wb = factor.max(((wf / beta / f).floor() * f) as usize);
    } else if hb * wb < min_pixels {
        let beta = (min_pixels as f64 / (hf * wf)).sqrt();
        hb = ((hf * beta / f).ceil() * f) as usize;
        wb = ((wf * beta / f).ceil() * f) as usize;
    }
    (hb, wb)
}

/// RGB8 `[h, w, 3]` (already resized to multiples of `patch * merge`) ->
/// patches `[gh gw, 3 patch patch]` normalised to `x / 127.5 - 1`, in
/// merge-block order, and the grid `(gh, gw)`.
pub fn patchify(rgb: &[u8], w: usize, h: usize, patch: usize, merge: usize) -> (Vec<f32>, (usize, usize)) {
    let (gh, gw) = (h / patch, w / patch);
    let pd = 3 * patch * patch;
    let mut out = vec![0f32; gh * gw * pd];
    let mut i = 0;
    for bh in 0..gh / merge {
        for bw in 0..gw / merge {
            for mh in 0..merge {
                for mw in 0..merge {
                    let (py, px) = ((bh * merge + mh) * patch, (bw * merge + mw) * patch);
                    let dst = &mut out[i * pd..(i + 1) * pd];
                    for c in 0..3 {
                        for y in 0..patch {
                            for x in 0..patch {
                                let v = rgb[((py + y) * w + px + x) * 3 + c] as f32;
                                dst[(c * patch + y) * patch + x] = (v - 127.5) / 127.5;
                            }
                        }
                    }
                    i += 1;
                }
            }
        }
    }
    (out, (gh, gw))
}

/// An RGB8 image `[h, w, 3]` -> its patches and grid (see the module docs).
pub fn preprocess(
    rgb: &[u8],
    w: usize,
    h: usize,
    patch: usize,
    merge: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> (Vec<f32>, (usize, usize)) {
    let (rh, rw) = smart_resize(h, w, patch * merge, min_pixels, max_pixels);
    let r = resize_rgb8(rgb, w, h, rw, rh);
    patchify(&r, rw, rh, patch, merge)
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
    fn smart_resize_matches_the_processor() {
        // values from transformers' smart_resize (factor 32, 65536..1M px)
        assert_eq!(smart_resize(853, 1280, 32, 65536, 1 << 20), (832, 1248));
        assert_eq!(smart_resize(100, 100, 32, 65536, 1 << 20), (256, 256));
        assert_eq!(smart_resize(4000, 3000, 32, 65536, 1 << 20), (1152, 864));
    }
}
