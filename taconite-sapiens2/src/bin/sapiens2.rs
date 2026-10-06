// SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
// SPDX-License-Identifier: Apache-2.0

//! `sapiens2`: Sapiens2-Pose keypoints on the NPU.
//!
//! ```text
//! sapiens2 pose <bundle> <image> [--box x,y,w,h]... [--threshold 0.3] [-o overlay.png] [--json out.json] [--reps N]
//! sapiens2 check <bundle>
//! ```
//!
//! `pose` runs every box (COCO x, y, width, height; default: the whole
//! image) and prints each one's keypoints above the threshold; `-o` draws
//! them over the image (body + feet red, hands green, face blue), `--json`
//! writes every keypoint. `check` runs the bundle's reference cases against
//! the float32 HF model's results.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use taconite_sapiens2::{BBox, Keypoint, Sapiens2, cosine};

type R<T> = Result<T, Box<dyn std::error::Error>>;

/// Keypoints [0, 21) are body and feet, [21, 63) the hands, the rest the
/// face (Sociopticon's 308).
const HANDS: usize = 21;
const FACE: usize = 63;

fn load_rgb(path: &Path) -> R<image::RgbImage> {
    Ok(image::open(path).map_err(|e| format!("{}: {e}", path.display()))?.to_rgb8())
}

fn timing_line(m: &Sapiens2) -> String {
    let t = &m.timing;
    let ms = |k: &str| t.get(k).as_secs_f64() * 1e3;
    format!(
        "npu {:.0} ms (mha {:.0}, gemms {:.0}), host {:.0} ms (crop {:.0}, layers {:.0}, attn staging {:.0}, head {:.0}, keypoints {:.0}); {} context swaps so far",
        t.npu_total().as_secs_f64() * 1e3,
        ms("npu:mha"),
        t.npu_total().as_secs_f64() * 1e3 - ms("npu:mha"),
        ms("host.crop") + ms("host") + ms("host.attn") + ms("host.head") + ms("host.keypoints"),
        ms("host.crop"),
        ms("host"),
        ms("host.attn"),
        ms("host.head"),
        ms("host.keypoints"),
        m.npu.swaps.get(),
    )
}

fn parse_box(s: &str) -> R<BBox> {
    let v: Vec<f32> = s.split(',').map(|x| x.trim().parse()).collect::<Result<_, _>>()?;
    match v[..] {
        [x, y, w, h] => Ok(BBox { x, y, w, h }),
        _ => Err(format!("--box {s}: want x,y,w,h").into()),
    }
}

fn draw(img: &mut image::RgbImage, b: &BBox, kps: &[Keypoint], threshold: f32) {
    let (w, h) = (img.width() as i64, img.height() as i64);
    let mut put = |x: i64, y: i64, c: [u8; 3]| {
        if x >= 0 && y >= 0 && x < w && y < h {
            img.put_pixel(x as u32, y as u32, image::Rgb(c));
        }
    };
    let (x0, y0, x1, y1) = (b.x as i64, b.y as i64, (b.x + b.w) as i64, (b.y + b.h) as i64);
    for x in x0..=x1 {
        put(x, y0, [255, 255, 0]);
        put(x, y1, [255, 255, 0]);
    }
    for y in y0..=y1 {
        put(x0, y, [255, 255, 0]);
        put(x1, y, [255, 255, 0]);
    }
    let r = ((b.w.max(b.h) / 150.0).round() as i64).clamp(1, 6);
    for (i, k) in kps.iter().enumerate() {
        if k.score <= threshold {
            continue;
        }
        let c = if i < HANDS {
            [255, 40, 40]
        } else if i < FACE {
            [40, 255, 40]
        } else {
            [60, 120, 255]
        };
        let rr = if i < HANDS { r + 1 } else { (r / 2).max(1) };
        let (cx, cy) = (k.x.round() as i64, k.y.round() as i64);
        for dy in -rr..=rr {
            for dx in -rr..=rr {
                if dx * dx + dy * dy <= rr * rr {
                    put(cx + dx, cy + dy, c);
                }
            }
        }
    }
}

fn pose(args: &[String]) -> R<()> {
    let dir = Path::new(args.first().ok_or("give a bundle")?);
    let img_path = PathBuf::from(args.get(1).ok_or("give an image")?);
    let (mut boxes, mut out, mut json, mut reps, mut threshold) = (Vec::new(), None, None, 1, 0.3f32);
    let mut i = 2;
    while i < args.len() {
        let v = args.get(i + 1).ok_or_else(|| format!("{} needs a value", args[i]))?;
        match args[i].as_str() {
            "--box" => boxes.push(parse_box(v)?),
            "-o" => out = Some(PathBuf::from(v)),
            "--json" => json = Some(PathBuf::from(v)),
            "--reps" => reps = v.parse()?,
            "--threshold" => threshold = v.parse()?,
            "--threads" => taconite::cpu::set_threads(v.parse()?),
            a => return Err(format!("unknown option {a}").into()),
        }
        i += 2;
    }
    let mut img = load_rgb(&img_path)?;
    let (w, h) = (img.width() as usize, img.height() as usize);
    if boxes.is_empty() {
        boxes.push(BBox { x: 0.0, y: 0.0, w: w as f32, h: h as f32 });
    }
    let t0 = Instant::now();
    let mut m = Sapiens2::load(dir)?;
    m.npu.preload()?;
    eprintln!("loaded {} ({} hardware contexts) in {:.1} s", dir.display(), m.contexts(), t0.elapsed().as_secs_f64());
    let mut all = Vec::new();
    for b in &boxes {
        let mut kps = Vec::new();
        for _ in 0..reps {
            let t1 = Instant::now();
            kps = m.pose(img.as_raw(), w, h, b)?;
            eprintln!("box {b:?}: {:.0} ms; {}", t1.elapsed().as_secs_f64() * 1e3, timing_line(&m));
        }
        let shown = kps.iter().filter(|k| k.score > threshold).count();
        let mean = kps.iter().map(|k| k.score).sum::<f32>() / kps.len() as f32;
        println!(
            "box {},{},{},{}: {shown} of {} keypoints above {threshold} (mean score {mean:.3})",
            b.x,
            b.y,
            b.w,
            b.h,
            kps.len()
        );
        for (j, k) in kps.iter().enumerate().take(17) {
            println!("  {j:3}: ({:7.1}, {:7.1}) score {:.3}", k.x, k.y, k.score);
        }
        all.push((*b, kps));
    }
    if let Some(p) = out {
        for (b, kps) in &all {
            draw(&mut img, b, kps, threshold);
        }
        img.save(&p).map_err(|e| format!("{}: {e}", p.display()))?;
        eprintln!("wrote {}", p.display());
    }
    if let Some(p) = json {
        let mut s = String::from("[");
        for (bi, (b, kps)) in all.iter().enumerate() {
            let pts: Vec<String> = kps.iter().map(|k| format!("[{:.2},{:.2},{:.4}]", k.x, k.y, k.score)).collect();
            write!(
                s,
                "{}{{\"box\":[{},{},{},{}],\"keypoints\":[{}]}}",
                if bi > 0 { "," } else { "" },
                b.x,
                b.y,
                b.w,
                b.h,
                pts.join(",")
            )?;
        }
        s.push_str("]\n");
        std::fs::write(&p, s)?;
        eprintln!("wrote {} (keypoints as [x, y, score], image pixels)", p.display());
    }
    Ok(())
}

/// Mean and max distance between two keypoint sets over the reference's
/// keypoints scoring above 0.3.
fn kp_err(a: &[Keypoint], b: &[f32], score: &[f32]) -> (f32, f32) {
    let (mut s, mut mx, mut n) = (0f32, 0f32, 0);
    for (i, k) in a.iter().enumerate() {
        if score[i] > 0.3 {
            let d = ((k.x - b[2 * i]).powi(2) + (k.y - b[2 * i + 1]).powi(2)).sqrt();
            s += d;
            mx = mx.max(d);
            n += 1;
        }
    }
    (s / n.max(1) as f32, mx)
}

fn check(args: &[String]) -> R<bool> {
    let dir = Path::new(args.first().ok_or("give a bundle")?);
    let t0 = Instant::now();
    let mut m = Sapiens2::load(dir)?;
    m.npu.preload()?;
    eprintln!("loaded {} ({} hardware contexts) in {:.1} s", dir.display(), m.contexts(), t0.elapsed().as_secs_f64());
    let refs: Vec<(String, usize, usize)> = m
        .manifest
        .tagged("ref")
        .map(|r| Ok((r.field(0)?.to_string(), r.get("w")?, r.get("h")?)))
        .collect::<Result<_, taconite_bundle::Error>>()?;
    if refs.is_empty() {
        return Err("the bundle has no reference cases (export with --refs)".into());
    }
    let mut ok = true;
    for (name, w, h) in &refs {
        let g = |k: &str| format!("ref.{name}.{k}");
        let rgb = m.store.u8(&g("rgb"))?.to_vec();
        let bx = m.store.f32(&g("box"))?.to_vec();
        let b = BBox { x: bx[0], y: bx[1], w: bx[2], h: bx[3] };
        let px = m.preprocess(&rgb, *w, *h, &b)?;
        let rpx = m.store.f32(&g("pixels"))?.to_vec();
        let px_err = px.iter().zip(&rpx).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        // features and heatmaps from the reference pixels (the crop's own
        // rounding aside), keypoints through the whole path
        let t1 = Instant::now();
        let hm = m.heatmaps(&rpx)?;
        let dt = t1.elapsed();
        let tline = timing_line(&m);
        let f = m.features(&rpx)?;
        let to_f = |v: &[u16]| v.iter().map(|&x| taconite::bf16_to_f32(x)).collect::<Vec<f32>>();
        let c_feat = cosine(&f, &to_f(m.store.bf16(&g("features"))?));
        let c_hm = cosine(&hm, &to_f(m.store.bf16(&g("heatmaps"))?));
        let hm2 = m.heatmaps(&px)?;
        let kps = m.keypoints(&hm2, &b);
        let scores = m.store.f32(&g("scores"))?.to_vec();
        let (e_ref, mx_ref) = kp_err(&kps, m.store.f32(&g("keypoints"))?, &scores);
        let vs_py = match m.store.has(&g("npu_keypoints")) {
            true => {
                let (e, mx) = kp_err(&kps, m.store.f32(&g("npu_keypoints"))?, &scores);
                format!("{e:.2} / {mx:.1}")
            }
            false => "(not in the bundle)".into(),
        };
        let side = b.w.max(b.h);
        let pass = px_err <= 2e-3 && c_hm >= 0.98 && e_ref / side <= 0.01;
        ok &= pass;
        println!(
            "{name} (box {:.0}x{:.0}): pixels max|d| {px_err:.1e}, features cos {c_feat:.5}, heatmaps cos {c_hm:.5}, \
             keypoints vs float32 HF {e_ref:.2} px mean ({:.2}% of the box) / {mx_ref:.1} max, vs the Python NPU app \
             {vs_py}; {:.0} ms -- {}",
            b.w,
            b.h,
            100.0 * e_ref / side,
            dt.as_secs_f64() * 1e3,
            if pass { "ok" } else { "FAIL" }
        );
        println!("  {tline}");
    }
    println!("{}", if ok { "ALL CHECKS PASSED" } else { "CHECKS FAILED" });
    Ok(ok)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let r = match args.first().map(String::as_str) {
        Some("pose") => pose(&args[1..]).map(|_| true),
        Some("check") => check(&args[1..]),
        _ => {
            eprintln!(
                "usage:\n  sapiens2 pose <bundle> <image> [--box x,y,w,h]... [--threshold 0.3] [-o overlay.png] [--json out.json] [--reps N]\n  \
                 sapiens2 check <bundle>"
            );
            return ExitCode::from(2);
        }
    };
    match r {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
