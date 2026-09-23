// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The network input, as GAIC-Pytorch's demo.py makes it: resize (short
//! side 256, each side rounded to a multiple of 32) with PIL's LANCZOS
//! filter, then ToTensor + ImageNet normalisation.
//!
//! [`resize_rgb8`] reproduces Pillow's `ImagingResample` for 8-bit images
//! bit for bit: the same coefficient windows (f64), the same 22-bit fixed
//! point rounding, horizontal pass first into an 8-bit intermediate.

use crate::par::{default_threads, par_rows};

pub const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
pub const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// demo.py's input size for a `w x h` image: short side to `short`, each
/// side rounded (half to even, as Python's `round`) to a multiple of 32.
pub fn input_size(w: usize, h: usize, short: usize) -> (usize, usize) {
    let scale = short as f64 / w.min(h) as f64;
    let r = |v: usize| ((v as f64 * scale / 32.0).round_ties_even() * 32.0) as usize;
    (r(w), r(h))
}

const PRECISION_BITS: u32 = 32 - 8 - 2;

fn lanczos(x: f64) -> f64 {
    fn sinc(x: f64) -> f64 {
        if x == 0.0 {
            1.0
        } else {
            let x = x * std::f64::consts::PI;
            x.sin() / x
        }
    }
    if (-3.0..3.0).contains(&x) { sinc(x) * sinc(x / 3.0) } else { 0.0 }
}

/// Pillow's precompute_coeffs + normalize_coeffs_8bpc: for every output
/// position, the first input index and the fixed-point weights.
fn coeffs(in_size: usize, out_size: usize) -> (Vec<(usize, usize)>, Vec<i32>, usize) {
    let scale = in_size as f64 / out_size as f64;
    let filterscale = scale.max(1.0);
    let support = 3.0 * filterscale;
    let ksize = support.ceil() as usize * 2 + 1;
    let mut bounds = Vec::with_capacity(out_size);
    let mut kk = vec![0i32; out_size * ksize];
    let mut k = vec![0f64; ksize];
    for xx in 0..out_size {
        let center = (xx as f64 + 0.5) * scale;
        let ss = 1.0 / filterscale;
        let xmin = ((center - support + 0.5) as i64).max(0) as usize;
        let xmax = ((center + support + 0.5) as i64).min(in_size as i64) as usize - xmin;
        let mut ww = 0.0;
        for x in 0..xmax {
            let w = lanczos((x as f64 + xmin as f64 - center + 0.5) * ss);
            k[x] = w;
            ww += w;
        }
        for x in 0..xmax {
            let w = if ww != 0.0 { k[x] / ww } else { k[x] };
            let f = w * (1u32 << PRECISION_BITS) as f64;
            kk[xx * ksize + x] = if w < 0.0 { (-0.5 + f) as i32 } else { (0.5 + f) as i32 };
        }
        bounds.push((xmin, xmax));
    }
    (bounds, kk, ksize)
}

#[inline]
fn clip8(ss: i32) -> u8 {
    (ss >> PRECISION_BITS).clamp(0, 255) as u8
}

/// Resizes an RGB8 image (`[h][w][3]`) with Pillow's LANCZOS, exactly.
pub fn resize_rgb8(src: &[u8], w: usize, h: usize, out_w: usize, out_h: usize) -> Vec<u8> {
    assert_eq!(src.len(), w * h * 3, "RGB8 buffer size");
    let (hb, hk, hks) = coeffs(w, out_w);
    let (vb, vk, vks) = coeffs(h, out_h);
    // Horizontal first, only over the rows the vertical pass reads.
    let (tmp, tmp_w, row0) = if out_w != w {
        let first = vb[0].0;
        let last = vb[out_h - 1].0 + vb[out_h - 1].1;
        let rows = last - first;
        let mut t = vec![0u8; rows * out_w * 3];
        par_rows(&mut t, out_w * 3, default_threads(), |y0, piece| {
            for (j, d) in piece.chunks_exact_mut(out_w * 3).enumerate() {
                let s = &src[(first + y0 + j) * w * 3..(first + y0 + j + 1) * w * 3];
                for xx in 0..out_w {
                    let (xmin, xmax) = hb[xx];
                    let k = &hk[xx * hks..xx * hks + xmax];
                    let mut ss = [1i32 << (PRECISION_BITS - 1); 3];
                    for (x, &kx) in k.iter().enumerate() {
                        let p = &s[(xmin + x) * 3..(xmin + x) * 3 + 3];
                        ss[0] += p[0] as i32 * kx;
                        ss[1] += p[1] as i32 * kx;
                        ss[2] += p[2] as i32 * kx;
                    }
                    d[xx * 3] = clip8(ss[0]);
                    d[xx * 3 + 1] = clip8(ss[1]);
                    d[xx * 3 + 2] = clip8(ss[2]);
                }
            }
        });
        (t, out_w, first)
    } else {
        (src.to_vec(), w, 0)
    };
    if out_h == h {
        return tmp;
    }
    let mut out = vec![0u8; out_w * out_h * 3];
    par_rows(&mut out, out_w * 3, default_threads(), |y0, piece| {
        for (j, d) in piece.chunks_exact_mut(out_w * 3).enumerate() {
            let (ymin, ymax) = vb[y0 + j];
            let ymin = ymin - row0;
            let k = &vk[(y0 + j) * vks..(y0 + j) * vks + ymax];
            for xx in 0..tmp_w * 3 {
                let mut ss = 1i32 << (PRECISION_BITS - 1);
                for (y, &ky) in k.iter().enumerate() {
                    ss += tmp[(ymin + y) * tmp_w * 3 + xx] as i32 * ky;
                }
                d[xx] = clip8(ss);
            }
        }
    });
    out
}

/// RGB8 `[h][w][3]` -> the normalized `[3][h][w]` f32 network input.
pub fn normalize(rgb: &[u8], w: usize, h: usize) -> Vec<f32> {
    let mut out = vec![0f32; 3 * w * h];
    for (i, p) in rgb.chunks_exact(3).enumerate() {
        for c in 0..3 {
            out[c * w * h + i] = (p[c] as f32 / 255.0 - MEAN[c]) / STD[c];
        }
    }
    out
}

/// A decoded RGB8 image -> (the `[3][h][w]` network input, (w, h)).
pub fn preprocess(rgb: &[u8], w: usize, h: usize) -> (Vec<f32>, (usize, usize)) {
    let (iw, ih) = input_size(w, h, 256);
    let r = resize_rgb8(rgb, w, h, iw, ih);
    (normalize(&r, iw, ih), (iw, ih))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_size_matches_demo() {
        assert_eq!(input_size(1024, 683, 256), (384, 256));
        assert_eq!(input_size(1024, 939, 256), (288, 256));
        assert_eq!(input_size(1024, 744, 256), (352, 256));
        assert_eq!(input_size(256, 256, 256), (256, 256));
    }

    #[test]
    fn identity_resize_is_a_copy() {
        let src: Vec<u8> = (0..4 * 5 * 3).map(|i| i as u8).collect();
        assert_eq!(resize_rgb8(&src, 4, 5, 4, 5), src);
    }
}
