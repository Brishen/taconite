// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! RoIAlignAvg / RoDAlignAvg: ports of GAIC-Pytorch's CPU kernels
//! (untils/{roi,rod}_align/src/*.cpp, forward) followed by the 2x2 stride-1
//! average pool the `*Avg` modules apply. f32 throughout, like the C code.

/// Bilinear sample of one channel plane with the C code's corner clamp:
/// the top-left corner is `min(floor(h), H - 2)`, so a sample on the last
/// row/column extrapolates from the last cell.
#[inline]
fn sample(plane: &[f32], hh: usize, ww: usize, h: f32, w: f32) -> f32 {
    let hs = (h.floor()).min(hh as f32 - 2.0);
    let ws = (w.floor()).min(ww as f32 - 2.0);
    // The ratios are f32 as in the C code; its `1.` literals make the blend
    // itself f64, rounded once on the store.
    let (hr, wr) = ((h - hs) as f64, (w - ws) as f64);
    let i = hs as usize * ww + ws as usize;
    let p = |j: usize| plane[j] as f64;
    (p(i) * (1.0 - hr) * (1.0 - wr)
        + p(i + 1) * (1.0 - hr) * wr
        + p(i + ww) * hr * (1.0 - wr)
        + p(i + ww + 1) * hr * wr) as f32
}

/// The FC head's input for one box: `[2 * C][s][s]` flattened (RoI
/// channels, then RoD), `s = align_size`, from the `[C][hh][ww]` map.
/// `b` is `[x1, y1, x2, y2]` in input pixels; `scale` maps them onto the map.
pub fn box_features(red: &[f32], c: usize, hh: usize, ww: usize, b: [f32; 4], s: usize, scale: f32, out: &mut [f32]) {
    let n = s + 1; // sampled grid, before the 2x2 average
    assert_eq!(out.len(), 2 * c * s * s);
    let [x1, y1, x2, y2] = b.map(|v| v * scale);
    let mut grid = vec![0f32; n * n];

    // RoI: an n x n grid over the box (the "+1" makes the far edge inclusive).
    let roi_w = (x2 - x1 + 1.0).max(0.0);
    let roi_h = (y2 - y1 + 1.0).max(0.0);
    let bin_h = (roi_h as f64 / (n as f64 - 1.0)) as f32;
    let bin_w = (roi_w as f64 / (n as f64 - 1.0)) as f32;
    // RoD: a fixed grid over the whole map, samples inside the box zeroed.
    let rod_bin_h = ((hh as f32 - 1.001) as f64 / (n as f64 - 1.0)) as f32;
    let rod_bin_w = ((ww as f32 - 1.001) as f64 / (n as f64 - 1.0)) as f32;

    for (part, base) in [(0usize, 0usize), (1, c)] {
        for ch in 0..c {
            let plane = &red[ch * hh * ww..(ch + 1) * hh * ww];
            for ph in 0..n {
                for pw in 0..n {
                    let v = if part == 0 {
                        let h = ph as f32 * bin_h + y1;
                        let w = pw as f32 * bin_w + x1;
                        if h < 0.0 || h >= hh as f32 || w < 0.0 || w >= ww as f32 {
                            0.0
                        } else {
                            sample(plane, hh, ww, h, w)
                        }
                    } else {
                        let h = ph as f32 * rod_bin_h;
                        let w = pw as f32 * rod_bin_w;
                        if h >= y1 && h <= y2 && w >= x1 && w <= x2 { 0.0 } else { sample(plane, hh, ww, h, w) }
                    };
                    grid[ph * n + pw] = v;
                }
            }
            let o = &mut out[(base + ch) * s * s..(base + ch + 1) * s * s];
            for ph in 0..s {
                for pw in 0..s {
                    let i = ph * n + pw;
                    o[ph * s + pw] = (grid[i] + grid[i + 1] + grid[i + n] + grid[i + n + 1]) * 0.25;
                }
            }
        }
    }
}
