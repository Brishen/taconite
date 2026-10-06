// SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
// SPDX-FileCopyrightText: The PyTorch authors (see LICENSE-PYTORCH)
// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

//! A person box -> the model's input crop, as HF's
//! `Sapiens2ImageProcessor(boxes=...)` makes it: the box padded by 1.25 and
//! widened or heightened to the crop's aspect ratio, the region sampled
//! onto the crop with PyTorch's `grid_sample` (align_corners, zero
//! padding; bilinear when the crop shrinks the region, bicubic when it
//! grows it) on the float image, then ImageNet-normalized.

/// A COCO box: top-left x, y, width, height (image pixels).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BBox {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

/// The crop window a box maps to: its center and size (image pixels).
#[derive(Debug, Clone, Copy)]
pub struct Window {
    pub cx: f32,
    pub cy: f32,
    pub w: f32,
    pub h: f32,
}

impl BBox {
    /// HF's `boxes_to_crop_params`: the box padded by `pad`, then the
    /// shorter side (relative to `out_w / out_h`) widened.
    pub fn window(&self, out_w: usize, out_h: usize, pad: f32) -> Window {
        let (cx, cy) = (self.x + 0.5 * self.w, self.y + 0.5 * self.h);
        let (sw, sh) = (self.w * pad, self.h * pad);
        let aspect = out_w as f32 / out_h as f32;
        let (w, h) = if sw > sh * aspect { (sw, sw / aspect) } else { (sh * aspect, sh) };
        Window { cx, cy, w, h }
    }
}

const A: f32 = -0.75; // PyTorch's bicubic coefficient

#[inline]
fn cc1(x: f32) -> f32 {
    ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
}

#[inline]
fn cc2(x: f32) -> f32 {
    ((A * x - 5.0 * A) * x + 8.0 * A) * x - 4.0 * A
}

#[inline]
fn cubic_coeffs(t: f32) -> [f32; 4] {
    [cc2(t + 1.0), cc1(t), cc1(1.0 - t), cc2(2.0 - t)]
}

/// RGB8 `[h, w, 3]` and a box -> pixel values `[3, out_h, out_w]` (planar,
/// normalized by `mean` / `std` after scaling to [0, 1]).
#[allow(clippy::too_many_arguments)]
pub fn crop(
    rgb: &[u8],
    w: usize,
    h: usize,
    b: &BBox,
    out_w: usize,
    out_h: usize,
    pad: f32,
    mean: [f32; 3],
    std: [f32; 3],
) -> Vec<f32> {
    let win = b.window(out_w, out_h, pad);
    let sx = (out_w - 1) as f32 / win.w;
    let sy = (out_h - 1) as f32 / win.h;
    let bilinear = sx.min(sy) < 1.0;
    let at = |c: usize, x: i64, y: i64| -> f32 {
        if x < 0 || y < 0 || x >= w as i64 || y >= h as i64 {
            0.0
        } else {
            rgb[(y as usize * w + x as usize) * 3 + c] as f32
        }
    };
    // the grid as HF builds it ([-1, 1], align_corners), then unnormalized
    // as grid_sample does
    let src = |g: f32, size: usize, scale: f32, c: f32, extent: f32| -> f32 {
        let i = g / scale + c - 0.5 * extent;
        let n = 2.0 * i / (size - 1) as f32 - 1.0;
        (n + 1.0) / 2.0 * (size - 1) as f32
    };
    let xs: Vec<f32> = (0..out_w).map(|i| src(i as f32, w, sx, win.cx, win.w)).collect();
    let mut out = vec![0f32; 3 * out_h * out_w];
    let plane = out_h * out_w;
    for oy in 0..out_h {
        let iy = src(oy as f32, h, sy, win.cy, win.h);
        for (ox, &ix) in xs.iter().enumerate() {
            for c in 0..3 {
                let v = if bilinear {
                    let (x0, y0) = (ix.floor(), iy.floor());
                    let (x1, y1) = (x0 + 1.0, y0 + 1.0);
                    let nw = (x1 - ix) * (y1 - iy);
                    let ne = (ix - x0) * (y1 - iy);
                    let sw = (x1 - ix) * (iy - y0);
                    let se = (ix - x0) * (iy - y0);
                    let (x0, y0) = (x0 as i64, y0 as i64);
                    at(c, x0, y0) * nw + at(c, x0 + 1, y0) * ne + at(c, x0, y0 + 1) * sw + at(c, x0 + 1, y0 + 1) * se
                } else {
                    let (x0, y0) = (ix.floor(), iy.floor());
                    let (cx, cy) = (cubic_coeffs(ix - x0), cubic_coeffs(iy - y0));
                    let (x0, y0) = (x0 as i64, y0 as i64);
                    let mut rows = [0f32; 4];
                    for (j, r) in rows.iter_mut().enumerate() {
                        let y = y0 - 1 + j as i64;
                        *r = at(c, x0 - 1, y) * cx[0]
                            + at(c, x0, y) * cx[1]
                            + at(c, x0 + 1, y) * cx[2]
                            + at(c, x0 + 2, y) * cx[3];
                    }
                    rows[0] * cy[0] + rows[1] * cy[1] + rows[2] * cy[2] + rows[3] * cy[3]
                };
                out[c * plane + oy * out_w + ox] = (v / 255.0 - mean[c]) / std[c];
            }
        }
    }
    out
}
