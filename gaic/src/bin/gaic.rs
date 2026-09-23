// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `gaic` — GAIC image cropping on the NPU, from an exported bundle.
//!
//!     gaic <bundle> check [--reps N] [--threads N]
//!         Replays the bundle's reference image (ref/) and compares against
//!         what the exporter recorded: the resize against PIL's, the
//!         feature map and the scores against the f32 CPU reference and
//!         the Python NPU path.
//!     gaic <bundle> crop [--out DIR] [--reps N] [--threads N] IMAGE...
//!         GAIC-Pytorch's demo: the best crop overall and at 1:1, 4:3 and
//!         16:9, printed in source-image pixels (and written to DIR).

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use gaic::anchors::{Box4, demo_sets, rescale_box};
use gaic::bundle::read_f32;
use gaic::{Gaic, preprocess};

fn usage() -> ExitCode {
    eprintln!(
        "usage: gaic <bundle> check [--reps N] [--threads N]\n       gaic <bundle> crop [--out DIR] [--reps N] [--threads N] IMAGE..."
    );
    ExitCode::from(2)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        return usage();
    }
    let bundle = PathBuf::from(&args[0]);
    let mut reps = 1usize;
    let mut threads: Option<usize> = None;
    let mut out_dir: Option<PathBuf> = None;
    let mut rest = Vec::new();
    let mut it = args[2..].iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--reps" => reps = it.next().and_then(|v| v.parse().ok()).unwrap_or(1),
            "--out" => out_dir = it.next().map(PathBuf::from),
            "--threads" => threads = it.next().and_then(|v| v.parse().ok()),
            _ => rest.push(a.clone()),
        }
    }
    let r = match args[1].as_str() {
        "check" => check(&bundle, reps, threads),
        "crop" if !rest.is_empty() => crop(&bundle, &rest, out_dir.as_deref(), reps, threads),
        _ => return usage(),
    };
    match r {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("gaic: {e}");
            ExitCode::FAILURE
        }
    }
}

type Res<T> = Result<T, Box<dyn std::error::Error>>;

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (x as f64, y as f64);
        ab += x * y;
        aa += x * x;
        bb += y * y;
    }
    ab / (aa.sqrt() * bb.sqrt())
}

fn ranks(v: &[f32]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[a].total_cmp(&v[b]));
    let mut r = vec![0f64; v.len()];
    for (k, &i) in idx.iter().enumerate() {
        r[i] = k as f64;
    }
    r
}

/// Spearman's rho (ties broken by order; the scores are continuous).
fn spearman(a: &[f32], b: &[f32]) -> f64 {
    let (ra, rb) = (ranks(a), ranks(b));
    let n = ra.len() as f64;
    let m = (n - 1.0) / 2.0;
    let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
    for (x, y) in ra.iter().zip(&rb) {
        ab += (x - m) * (y - m);
        aa += (x - m) * (x - m);
        bb += (y - m) * (y - m);
    }
    ab / (aa * bb).sqrt()
}

fn argmax(v: &[f32]) -> usize {
    (0..v.len()).max_by(|&a, &b| v[a].total_cmp(&v[b])).unwrap_or(0)
}

fn as_f32_boxes(b: &[Box4]) -> Vec<[f32; 4]> {
    b.iter().map(|b| b.map(|v| v as f32)).collect()
}

fn check(bundle: &Path, reps: usize, threads: Option<usize>) -> Res<bool> {
    let t0 = Instant::now();
    let mut g = Gaic::load(bundle)?;
    if let Some(n) = threads {
        g.set_threads(n);
    }
    println!("loaded {} in {:.2} s", bundle.display(), t0.elapsed().as_secs_f64());
    let r = g.manifest().reference.clone().ok_or("the bundle has no ref record")?;
    let dir = bundle.join("ref");
    let mut ok = true;

    // The resize: PIL's decoded source through our Lanczos, against PIL's.
    let src = std::fs::read(dir.join("src_rgb.u8"))?;
    let (x_rs, (w, h)) = preprocess::preprocess(&src, r.src_w, r.src_h);
    let x = read_f32(&dir.join("input_chw.f32"))?;
    let diff = x_rs.iter().zip(&x).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    let exact = x_rs.iter().zip(&x).filter(|(a, b)| a == b).count();
    println!("resize {}x{} -> {w}x{h}: {exact}/{} values exact, max |diff| {diff:e}", r.src_w, r.src_h, x.len());
    ok &= (w, h) == (r.w, r.h) && diff < 1e-5;

    let anchors: Vec<[f32; 4]> = std::fs::read(dir.join("anchors.i32"))?
        .chunks_exact(16)
        .map(|b| {
            let v = |i: usize| i32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap()) as f32;
            [v(0), v(1), v(2), v(3)]
        })
        .collect();
    let (red_cpu, red_npu) = (read_f32(&dir.join("red_cpu.f32"))?, read_f32(&dir.join("red_npu.f32"))?);
    let (s_cpu, s_npu) = (read_f32(&dir.join("scores_cpu.f32"))?, read_f32(&dir.join("scores_npu.f32"))?);
    for rep in 0..reps.max(1) {
        g.reset_timing();
        let t = Instant::now();
        let f = g.features(&x, r.w, r.h)?;
        let s = g.score(&f, &anchors)?;
        let wall = t.elapsed();
        if rep == 0 {
            println!(
                "feature map {}x{}x{}: cosine vs CPU {:.6}, vs Python NPU {:.6}",
                f.channels,
                f.h,
                f.w,
                cosine(&f.map, &red_cpu),
                cosine(&f.map, &red_npu)
            );
            let (a, ac, an) = (argmax(&s), argmax(&s_cpu), argmax(&s_npu));
            println!(
                "{} anchors: score cosine vs CPU {:.6}, spearman {:.5}; vs Python NPU {:.6}; best {a} (CPU {ac}, Python NPU {an})",
                s.len(),
                cosine(&s, &s_cpu),
                spearman(&s, &s_cpu),
                cosine(&s, &s_npu)
            );
            ok &= s.len() == r.n_anchors && a == ac && spearman(&s, &s_cpu) > 0.99;
        }
        println!("run {rep}: {:.1} ms  [{}]", wall.as_secs_f64() * 1e3, g.timing);
    }
    println!("{}", if ok { "CHECK OK" } else { "CHECK FAILED" });
    Ok(ok)
}

fn crop(bundle: &Path, images: &[String], out: Option<&Path>, reps: usize, threads: Option<usize>) -> Res<bool> {
    let mut g = Gaic::load(bundle)?;
    if let Some(n) = threads {
        g.set_threads(n);
    }
    if let Some(d) = out {
        std::fs::create_dir_all(d)?;
    }
    for path in images {
        let img = image::open(path)?.to_rgb8();
        let (sw, sh) = img.dimensions();
        let (sw, sh) = (sw as usize, sh as usize);
        for rep in 0..reps.max(1) {
            g.reset_timing();
            let t = Instant::now();
            let (x, (w, h)) = preprocess::preprocess(img.as_raw(), sw, sh);
            let t_pre = t.elapsed();
            let f = g.features(&x, w, h)?;
            let mut lines = Vec::new();
            for (name, set) in demo_sets(w, h) {
                let s = g.score(&f, &as_f32_boxes(&set))?;
                let i = argmax(&s);
                let b = rescale_box(set[i], w, h, sw, sh);
                lines.push(format!(
                    "  {name:>4}: [{}, {}, {}, {}] score {:.4} ({} candidates)",
                    b[0],
                    b[1],
                    b[2],
                    b[3],
                    s[i],
                    set.len()
                ));
                if rep == 0
                    && let Some(d) = out
                {
                    let stem = Path::new(path).file_stem().unwrap_or_default().to_string_lossy();
                    let (x1, y1) = (b[0].max(0) as u32, b[1].max(0) as u32);
                    let (x2, y2) = ((b[2].max(0) as u32).min(sw as u32), (b[3].max(0) as u32).min(sh as u32));
                    let c = image::imageops::crop_imm(&img, x1, y1, x2.saturating_sub(x1), y2.saturating_sub(y1))
                        .to_image();
                    c.save(d.join(format!("{stem}_{}.jpg", name.replace(':', "x"))))?;
                }
            }
            let wall = t.elapsed();
            if rep == 0 {
                println!("{path} {sw}x{sh} -> {w}x{h}");
                for l in &lines {
                    println!("{l}");
                }
            }
            println!(
                "  run {rep}: {:.1} ms (resize {:.1} ms) [{}]",
                wall.as_secs_f64() * 1e3,
                t_pre.as_secs_f64() * 1e3,
                g.timing
            );
        }
    }
    Ok(true)
}
