// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Candidate crops, as GAIC-Pytorch's dataset/candidate_generation.py makes
//! them (same f64 arithmetic, same truncating `int()`), in the network
//! input's pixels: `[x1, y1, x2, y2]`.

pub type Box4 = [i32; 4];

/// The aspect-ratio-free grid: corners on a `bins x bins` grid (x1, y1 in
/// the first third, x2, y2 in the last), covering at least half the image,
/// aspect ratio within (1/2, 2).
pub fn generate_anchors(im_w: usize, im_h: usize, bins: usize) -> Vec<Box4> {
    let (step_w, step_h) = (im_w as f64 / bins as f64, im_h as f64 / bins as f64);
    let third = (bins as f64 / 3.0) as usize;
    let two_thirds = (bins as f64 / 3.0 * 2.0) as usize;
    let mut out = Vec::new();
    for x1 in 0..third {
        for y1 in 0..third {
            for x2 in two_thirds..bins {
                for y2 in two_thirds..bins {
                    let area = ((x2 - x1) * (y2 - y1)) as f64 / (bins * bins) as f64;
                    let aspect = (y2 - y1) as f64 * step_h / ((x2 - x1) as f64 * step_w);
                    if area > 0.4999 && aspect > 0.5 && aspect < 2.0 {
                        out.push([
                            (step_w * (0.5 + x1 as f64)) as i32,
                            (step_h * (0.5 + y1 as f64)) as i32,
                            (step_w * (0.5 + x2 as f64)) as i32,
                            (step_h * (0.5 + y2 as f64)) as i32,
                        ]);
                    }
                }
            }
        }
    }
    out
}

/// Every crop of aspect ratio `w:h` (both >= 1) on a step grid, from about
/// half the image to the whole, covering over 40% of it; `bins` caps the
/// number of scales.
pub fn generate_anchors_aspect_ratio_specific(
    im_w: usize,
    im_h: usize,
    aspect: (usize, usize),
    bins: usize,
) -> Vec<Box4> {
    let (mut w_step, mut h_step) = (aspect.0, aspect.1);
    let fmin = |a: f64, b: f64| a.min(b);
    let mut max_step = fmin(im_w as f64 / w_step as f64, im_h as f64 / h_step as f64) as usize;
    if max_step > bins {
        let scale = (im_w as f64 / w_step as f64 / bins as f64).max(im_h as f64 / h_step as f64 / bins as f64) as usize;
        h_step *= scale;
        w_step *= scale;
        max_step = fmin(im_w as f64 / w_step as f64, im_h as f64 / h_step as f64) as usize;
    }
    let min_step = (max_step as f64 / 2.0 - 1.0) as i64;
    let mut out = Vec::new();
    for i in min_step.max(0) as usize..max_step {
        let (out_h, out_w) = (h_step * i, w_step * i);
        if out_h < im_h && out_w < im_w && (out_w * out_h) as f64 > 0.4 * (im_w * im_h) as f64 {
            for w_start in (0..im_w - out_w).step_by(w_step) {
                for h_start in (0..im_h - out_h).step_by(h_step) {
                    out.push([
                        w_start as i32,
                        h_start as i32,
                        (w_start + out_w - 1) as i32,
                        (h_start + out_h - 1) as i32,
                    ]);
                }
            }
        }
    }
    out
}

/// The anchor sets GAIC-Pytorch's demo scores: aspect-free, 1:1, 4:3, 16:9.
pub fn demo_sets(w: usize, h: usize) -> Vec<(&'static str, Vec<Box4>)> {
    vec![
        ("best", generate_anchors(w, h, 12)),
        ("1:1", generate_anchors_aspect_ratio_specific(w, h, (1, 1), 30)),
        ("4:3", generate_anchors_aspect_ratio_specific(w, h, (4, 3), 20)),
        ("16:9", generate_anchors_aspect_ratio_specific(w, h, (16, 9), 15)),
    ]
}

/// A box in the network input's pixels (`in_w x in_h`) mapped to the
/// source image's, as demo.py does it (f32, truncated).
pub fn rescale_box(b: Box4, in_w: usize, in_h: usize, src_w: usize, src_h: usize) -> Box4 {
    let sx = |v: i32| (v as f32 / in_w as f32 * src_w as f32) as i32;
    let sy = |v: i32| (v as f32 / in_h as f32 * src_h as f32) as i32;
    [sx(b[0]), sy(b[1]), sx(b[2]), sy(b[3])]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_match_the_python_reference() {
        // gaic_cpu on the three GAIC-Pytorch test images.
        assert_eq!(generate_anchors(384, 256, 12).len(), 83);
        assert_eq!(generate_anchors(288, 256, 12).len(), 90);
        assert_eq!(generate_anchors(352, 256, 12).len(), 87);
        assert_eq!(generate_anchors_aspect_ratio_specific(384, 256, (1, 1), 30).len(), 194);
        assert_eq!(generate_anchors_aspect_ratio_specific(288, 256, (1, 1), 30).len(), 416);
        assert_eq!(generate_anchors_aspect_ratio_specific(384, 256, (4, 3), 20).len(), 193);
        assert_eq!(generate_anchors_aspect_ratio_specific(288, 256, (4, 3), 20).len(), 115);
        assert_eq!(generate_anchors_aspect_ratio_specific(384, 256, (16, 9), 15).len(), 280);
        assert_eq!(generate_anchors_aspect_ratio_specific(288, 256, (16, 9), 15).len(), 80);
    }
}
