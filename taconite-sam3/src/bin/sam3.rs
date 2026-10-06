// SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
// SPDX-License-Identifier: Apache-2.0

//! `sam3 check <bundle>` -- every stage against the bundle's float32
//! reference (each on the reference's own inputs), then end to end on the
//! bundle's test cases.
//!
//! `sam3 segment <bundle> <image> <prompt> [-o overlay.png] [--threshold T]
//! [--reps N] [--threads N]` -- segment every instance of `prompt`.
//!
//! `sam3 points <bundle> <image> <x,y[,label];...> [--box x1,y1,x2,y2]
//! [--single] [-o overlay.png]` -- segment the object at the clicks (SAM's
//! three candidate masks, or one with `--single`).
//!
//! Both take `--mode npu|npu+gpu` (default: `SAM3_MODE`, else `npu`): the
//! work between the NPU's kernels on the CPU, or on the GPU (feature `gpu`).

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use taconite_sam3::{Instance, Mode, Output, PointPrompt, Sam3, Text, instances, point_mask, preprocess};

type R<T> = Result<T, Box<dyn std::error::Error>>;

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        ab += x as f64 * y as f64;
        aa += x as f64 * x as f64;
        bb += y as f64 * y as f64;
    }
    ab / (aa.sqrt() * bb.sqrt()).max(1e-30)
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

fn load_rgb(path: &Path) -> R<(Vec<u8>, usize, usize)> {
    let im = image::open(path).map_err(|e| format!("{}: {e}", path.display()))?.to_rgb8();
    let (w, h) = im.dimensions();
    Ok((im.into_raw(), w as usize, h as usize))
}

struct Checks {
    failed: Vec<String>,
}

impl Checks {
    fn check(&mut self, name: &str, ok: bool, detail: String) {
        println!("  [{}] {name}: {detail}", if ok { "ok" } else { "FAIL" });
        if !ok {
            self.failed.push(name.to_string());
        }
    }
}

fn iou(a: &[bool], b: &[u8]) -> f64 {
    let (mut i, mut u) = (0usize, 0usize);
    for (&x, &y) in a.iter().zip(b) {
        let y = y != 0;
        i += (x && y) as usize;
        u += (x || y) as usize;
    }
    if u == 0 { 1.0 } else { i as f64 / u as f64 }
}

fn ref_text(m: &Sam3) -> R<Text> {
    let st = &m.store;
    Ok(Text {
        feats: st.f32("ref.text")?.to_vec(),
        valid: st.i32("ref.attention_mask")?.iter().map(|&v| v != 0).collect(),
    })
}

fn check(dir: &Path, mode: Mode) -> R<bool> {
    let t0 = Instant::now();
    let mut m = Sam3::load_mode(dir, mode)?;
    println!("loaded {} ({:.1} s); mode {} on {}", dir.display(), t0.elapsed().as_secs_f64(), m.mode(), m.devices());
    let mut c = Checks { failed: vec![] };
    let cfg = m.cfg.clone();
    let (d, t) = (cfg.d_model, cfg.tokens());

    println!("host pieces:");
    {
        let st = &m.store;
        let spec = m.manifest.gemm("d_s")?;
        let packed = taconite_sam3::pack::pack_b(spec, st.f32("pack_test.B")?, Some(st.f32("pack_test.bias")?));
        let want = st.u8("pack_test.packed")?;
        let diff = packed.iter().zip(want).filter(|(a, b)| a != b).count();
        c.check(
            "bfp16 packing",
            diff == 0 && packed.len() == want.len(),
            format!("{diff} of {} bytes differ", want.len()),
        );
        let mut bad = vec![];
        for (j, prompt) in &m.manifest.tok_tests {
            let (ids, mask) = m.tokenizer.encode(prompt);
            let want_ids: Vec<u32> = st.i32(&format!("tok.{j}.ids"))?.iter().map(|&v| v as u32).collect();
            let want_mask: Vec<u32> = st.i32(&format!("tok.{j}.mask"))?.iter().map(|&v| v as u32).collect();
            if ids != want_ids || mask != want_mask {
                bad.push(format!("{prompt:?}: {ids:?} vs {want_ids:?}"));
            }
        }
        let n = m.manifest.tok_tests.len();
        c.check("tokenizer", bad.is_empty(), format!("{}/{n} prompts identical{}", n - bad.len(), bad.join("; ")));
        let want = st.f32("ref.pixel_values")?;
        // one uint8 level is 2/255 after normalisation
        let level = 2.0 / 255.0;
        let rgb = st.u8("ref.rgb")?;
        let shape = st.shape("ref.rgb")?;
        let px = preprocess(rgb, shape[1], shape[0], cfg.image_size);
        let off = px.iter().zip(want).filter(|(a, b)| (*a - *b).abs() > 1e-4).count();
        c.check("resize + normalise", off == 0, format!("{off} of {} values differ from the processor's", px.len()));
        if let Some(img) = &m.manifest.ref_image {
            let (dec, w, h) = load_rgb(img)?;
            let dd = dec.iter().zip(rgb).filter(|(a, b)| a != b).count();
            let px = preprocess(&dec, w, h, cfg.image_size);
            let md = max_abs(&px, want) / level;
            println!(
                "  [info] image decoding: {dd} of {} bytes differ from PIL's; after resize max {md:.1} levels",
                rgb.len()
            );
        }
    }

    println!("stages, each on the float32 reference's inputs:");
    let st_ids: Vec<u32> = m.store.i32("ref.input_ids")?.iter().map(|&v| v as u32).collect();
    let st_mask: Vec<u32> = m.store.i32("ref.attention_mask")?.iter().map(|&v| v as u32).collect();
    let t1 = Instant::now();
    let text = m.text(&st_ids, &st_mask)?;
    let rtext = ref_text(&m)?;
    let nv = text.valid.iter().filter(|&&v| v).count();
    let cos = cosine(&text.feats[..nv * d], &rtext.feats[..nv * d]);
    c.check("text encoder", cos > 0.9999, format!("cosine {cos:.6} ({:.0} ms)", t1.elapsed().as_secs_f64() * 1e3));

    let pixels = m.store.f32("ref.pixel_values")?.to_vec();
    let t1 = Instant::now();
    let vit = m.vit(&pixels)?;
    let cos = cosine(&vit, m.store.f32("ref.vit_out")?);
    c.check("ViT backbone", cos > 0.98, format!("cosine {cos:.6} ({:.0} ms)", t1.elapsed().as_secs_f64() * 1e3));

    let rvit = m.store.f32("ref.vit_out")?.to_vec();
    let t1 = Instant::now();
    let fpn = m.neck(&rvit)?;
    let ms = t1.elapsed().as_secs_f64() * 1e3;
    for (i, level) in fpn.iter().enumerate().take(3) {
        let cos = cosine(level, m.store.f32(&format!("ref.fpn{i}"))?);
        c.check(&format!("neck level {i}"), cos > 0.995, format!("cosine {cos:.6} ({ms:.0} ms all)"));
    }

    let rfpn: [Vec<f32>; 3] = [0, 1, 2].map(|i| m.store.f32(&format!("ref.fpn{i}")).unwrap().to_vec());
    let t1 = Instant::now();
    let enc = m.detr_encoder(&rfpn[2], &rtext)?;
    let cos = cosine(&enc, m.store.f32("ref.enc_out")?);
    c.check("DETR encoder", cos > 0.99, format!("cosine {cos:.6} ({:.0} ms)", t1.elapsed().as_secs_f64() * 1e3));

    let renc = m.store.f32("ref.enc_out")?.to_vec();
    let t1 = Instant::now();
    let dec = m.detr_decoder(&renc, &rtext)?;
    let ms = t1.elapsed().as_secs_f64() * 1e3;
    let st = &m.store;
    let q = cfg.queries;
    let rh = st.f32("ref.dec_hidden")?[(cfg.dec_layers - 1) * q * d..].to_vec();
    let cos = cosine(&dec.hidden, &rh);
    c.check("DETR decoder hidden", cos > 0.999, format!("cosine {cos:.6} ({ms:.0} ms)"));
    let rl = st.f32("ref.pred_logits")?;
    let conf: Vec<usize> = (0..q).filter(|&i| taconite_sam3::cpu::sigmoid(rl[i]) > 0.1).collect();
    let rb = st.f32("ref.pred_boxes")?;
    let db = &dec.boxes;
    let bd = conf.iter().flat_map(|&i| (0..4).map(move |k| (db[i * 4 + k] - rb[i * 4 + k]).abs())).fold(0.0, f32::max);
    c.check("DETR decoder boxes", bd < 0.01, format!("max |diff| {bd:.5} over {} confident queries", conf.len()));
    let ld = conf.iter().map(|&i| (dec.logits[i] - rl[i]).abs()).fold(0.0, f32::max);
    c.check("DETR decoder logits", ld < 0.1, format!("max |diff| {ld:.4} over {} confident queries", conf.len()));
    let pr = st.f32("ref.presence")?[0];
    c.check("presence", (dec.presence - pr).abs() < 0.05, format!("{:.4} vs {pr:.4}", dec.presence));
    if m.mode() == Mode::NpuGpu && !m.use_gpu(false) {
        // the GPU decoder against the CPU one, on the same inputs
        let t1 = Instant::now();
        let cpu = m.detr_decoder(&renc, &rtext)?;
        let ms = t1.elapsed().as_secs_f64() * 1e3;
        m.use_gpu(true);
        let max_diff = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max);
        let hd = max_diff(&dec.hidden, &cpu.hidden);
        let bd = max_diff(&dec.boxes, &cpu.boxes);
        let ld = max_diff(&dec.logits, &cpu.logits);
        let pd = (dec.presence - cpu.presence).abs();
        c.check(
            "DETR decoder, GPU vs CPU",
            hd < 1e-2 && bd < 1e-3 && ld < 1e-2 && pd < 1e-2,
            format!("max |diff| hidden {hd:.2e}, boxes {bd:.2e}, logits {ld:.2e}, presence {pd:.2e} (CPU {ms:.0} ms)"),
        );
    }

    let t1 = Instant::now();
    let (masks, semantic) = m.mask_decoder(&rh, &rfpn, &renc, &rtext)?;
    let ms = t1.elapsed().as_secs_f64() * 1e3;
    let cos = cosine(&masks, m.store.f32("ref.pred_masks")?);
    c.check("mask decoder masks", cos > 0.995, format!("cosine {cos:.6} ({ms:.0} ms)"));
    let cos = cosine(&semantic, m.store.f32("ref.semantic")?);
    c.check("mask decoder semantic", cos > 0.99, format!("cosine {cos:.6}"));

    println!("end to end (image file + prompt -> instances) against the reference cases:");
    let cases = m.manifest.cases.clone();
    for case in &cases {
        let (rgb, w, h) = load_rgb(&case.image)?;
        let t1 = Instant::now();
        let px = preprocess(&rgb, w, h, cfg.image_size);
        m.timing.clear();
        let out = m.segment(&px, &case.prompt)?;
        let ms = t1.elapsed().as_secs_f64() * 1e3;
        let inst = instances(&out, cfg.mask_size, w, h, 0.5, 0.5);
        let st = &m.store;
        let rs = st.f32(&format!("case.{}.scores", case.idx))?;
        let rm = st.u8(&format!("case.{}.masks", case.idx))?;
        let n = rs.len();
        let mut worst = 1.0f64;
        let mut sd = 0f32;
        let mut used = vec![false; inst.len()];
        for r in 0..n {
            let rmask = &rm[r * w * h..(r + 1) * w * h];
            let best = (0..inst.len())
                .filter(|&j| !used[j])
                .max_by(|&a, &b| iou(&inst[a].mask, rmask).total_cmp(&iou(&inst[b].mask, rmask)));
            match best {
                Some(j) => {
                    used[j] = true;
                    worst = worst.min(iou(&inst[j].mask, rmask));
                    sd = sd.max((inst[j].score - rs[r]).abs());
                }
                None => worst = 0.0,
            }
        }
        let name = format!("{} / {:?}", case.image.file_name().unwrap().to_string_lossy(), case.prompt);
        c.check(
            &name,
            inst.len() == n && worst > 0.97 && sd < 0.03,
            format!(
                "{} instances (ref {n}), min mask IoU {worst:.4}, max |score diff| {sd:.3}; {ms:.0} ms ({})",
                inst.len(),
                m.timing
            ),
        );
    }
    let _ = t;
    if m.has_points() {
        check_points(&mut m, &mut c)?;
    }
    if c.failed.is_empty() {
        println!("ALL CHECKS PASSED");
    } else {
        println!("FAILED: {}", c.failed.join(", "));
    }
    Ok(c.failed.is_empty())
}

/// The point prompt path: its stages on the reference's inputs, then the
/// bundle's point cases end to end.
fn check_points(m: &mut Sam3, c: &mut Checks) -> R<()> {
    println!("point prompts, each stage on the float32 reference's inputs:");
    let rvit = m.store.f32("ref.trk.vit_out")?.to_vec();
    let t1 = Instant::now();
    let emb = m.tracker_neck(&rvit)?;
    let ms = t1.elapsed().as_secs_f64() * 1e3;
    for (i, level) in [&emb.s0, &emb.s1, &emb.emb].into_iter().enumerate() {
        let cos = cosine(level, m.store.f32(&format!("ref.trk.emb{i}"))?);
        c.check(&format!("tracker neck level {i}"), cos > 0.995, format!("cosine {cos:.6} ({ms:.0} ms all)"));
    }
    let st = &m.store;
    let pts: Vec<[f32; 2]> = st.f32("ref.trk.points")?.chunks(2).map(|p| [p[0], p[1]]).collect();
    let labels = st.i32("ref.trk.labels")?.to_vec();
    let sparse = m.prompt_tokens(&pts, &labels, None)?;
    let d = max_abs(&sparse, st.f32("ref.trk.sparse")?);
    c.check("prompt encoder", d < 1e-4, format!("max |diff| {d:.2e}"));
    let remb = taconite_sam3::ImageEmbedding {
        s0: st.f32("ref.trk.emb0")?.to_vec(),
        s1: st.f32("ref.trk.emb1")?.to_vec(),
        emb: st.f32("ref.trk.emb2")?.to_vec(),
    };
    for (key, multi) in [("multi", true), ("single", false)] {
        let t1 = Instant::now();
        let out = m.decode_points(&remb, &sparse, multi)?;
        let ms = t1.elapsed().as_secs_f64() * 1e3;
        let st = &m.store;
        let cos = cosine(&out.masks, st.f32(&format!("ref.trk.{key}.masks"))?);
        let id = max_abs(&out.iou, st.f32(&format!("ref.trk.{key}.iou"))?);
        let od = (out.object_score - st.f32("ref.trk.obj")?[0]).abs();
        c.check(
            &format!("mask decoder, {key}mask"),
            cos > 0.9999 && id < 1e-3 && od < 1e-2,
            format!("masks cosine {cos:.6}, max |IoU diff| {id:.2e}, |object diff| {od:.2e} ({ms:.0} ms)"),
        );
    }

    println!("end to end (image file + points / box -> masks) against the reference cases:");
    let size = m.cfg.image_size;
    for case in m.manifest.point_cases.clone() {
        let prompt = PointPrompt::parse_case(&case.prompt)?;
        let (rgb, w, h) = load_rgb(&case.image)?;
        m.timing.clear();
        let t1 = Instant::now();
        let emb = m.embed_image(&preprocess(&rgb, w, h, size))?;
        let t2 = Instant::now();
        let out = m.predict_points(&emb, &prompt, w, h, true)?;
        let t3 = Instant::now();
        let st = &m.store;
        let rm = st.u8(&format!("pcase.{}.masks", case.idx))?;
        let riou = st.f32(&format!("pcase.{}.iou", case.idx))?;
        let ious: Vec<f64> =
            (0..out.n).map(|i| iou(&point_mask(&out, i, w, h), &rm[i * w * h..(i + 1) * w * h])).collect();
        // the best candidate and any the model rates well, judged strictly;
        // a low-rated one has an ambiguous edge that swings with the
        // image's decoding (image crate vs PIL) and the NPU's rounding
        let best = (0..out.n).max_by(|&a, &b| riou[a].total_cmp(&riou[b])).unwrap_or(0);
        let ok = (0..out.n).all(|i| ious[i] > if i == best || riou[i] >= 0.85 { 0.98 } else { 0.85 });
        let id = max_abs(&out.iou, riou);
        let name = format!("{} / {}", case.image.file_name().unwrap().to_string_lossy(), case.prompt);
        c.check(
            &name,
            ok && id < 0.03,
            format!(
                "mask IoU vs ref {:?}, max |IoU score diff| {id:.3}; image {:.0} ms ({}), prompt {:.0} ms",
                ious.iter().map(|v| (v * 1e4).round() / 1e4).collect::<Vec<_>>(),
                (t2 - t1).as_secs_f64() * 1e3,
                m.timing,
                (t3 - t2).as_secs_f64() * 1e3
            ),
        );
    }
    Ok(())
}

/// `sam3 points <bundle> <image> <x,y[,label];...> [--box x1,y1,x2,y2]`
fn points(args: &[String], mode: Mode) -> R<()> {
    let (dir, img, spec) = (Path::new(&args[0]), Path::new(&args[1]), &args[2]);
    let mut out_path: Option<PathBuf> = None;
    let mut bbox: Option<String> = None;
    let mut reps = 1usize;
    let mut single = false;
    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--single" => {
                single = true;
                i += 1;
                continue;
            }
            "-o" => out_path = Some(PathBuf::from(&args[i + 1])),
            "--box" => bbox = Some(args[i + 1].clone()),
            "--reps" => reps = args[i + 1].parse()?,
            "--threads" => taconite_sam3::cpu::set_threads(args[i + 1].parse()?),
            a => return Err(format!("unknown argument {a}").into()),
        }
        i += 2;
    }
    let prompt = PointPrompt::parse(spec, bbox.as_deref())?;
    let t0 = Instant::now();
    let mut m = Sam3::load_mode(dir, mode)?;
    println!("loaded {} ({:.1} s); mode {} on {}", dir.display(), t0.elapsed().as_secs_f64(), m.mode(), m.devices());
    let (rgb, w, h) = load_rgb(img)?;
    let mut emb = None;
    for r in 0..reps {
        m.timing.clear();
        let t1 = Instant::now();
        emb = Some(m.embed_image(&preprocess(&rgb, w, h, m.cfg.image_size))?);
        let (loads, evictions) = m.contexts();
        println!(
            "image {r}: {:.0} ms ({}; contexts loaded {loads}, evicted {evictions})",
            t1.elapsed().as_secs_f64() * 1e3,
            m.timing
        );
    }
    let emb = emb.unwrap();
    let t1 = Instant::now();
    let out = m.predict_points(&emb, &prompt, w, h, !single)?;
    println!("prompt: {:.0} ms (host)", t1.elapsed().as_secs_f64() * 1e3);
    println!("object score {:.2} ({})", out.object_score, if out.object_score > 0.0 { "object" } else { "no object" });
    let best = out.best();
    let mut masks = Vec::new();
    for i in 0..out.n {
        let mask = point_mask(&out, i, w, h);
        let area = mask.iter().filter(|&&v| v).count();
        println!("  mask {i}: predicted IoU {:.3}, {area} px{}", out.iou[i], if i == best { "  <- best" } else { "" });
        masks.push(mask);
    }
    if let Some(p) = out_path {
        point_overlay(&rgb, w, h, &masks[best], &prompt, &p)?;
        println!("wrote {}", p.display());
    }
    Ok(())
}

fn point_overlay(rgb: &[u8], w: usize, h: usize, mask: &[bool], prompt: &PointPrompt, path: &Path) -> R<()> {
    let mut img = rgb.to_vec();
    let col = [30u16, 144, 255];
    for (p, &on) in mask.iter().enumerate() {
        if on {
            for ch in 0..3 {
                img[p * 3 + ch] = ((img[p * 3 + ch] as u16 + col[ch]) / 2) as u8;
            }
        }
    }
    let r = (w.min(h) / 150).max(4) as i64;
    for (pt, &l) in prompt.points.iter().zip(&prompt.labels) {
        let dot = if l == 1 { [0u8, 255, 0] } else { [255, 0, 0] };
        let (cx, cy) = (pt[0] as i64, pt[1] as i64);
        for y in cy - r..=cy + r {
            for x in cx - r..=cx + r {
                if (x - cx).pow(2) + (y - cy).pow(2) <= r * r
                    && (0..w as i64).contains(&x)
                    && (0..h as i64).contains(&y)
                {
                    img[(y as usize * w + x as usize) * 3..][..3].copy_from_slice(&dot);
                }
            }
        }
    }
    if let Some(b) = prompt.bbox {
        let [x0, y0, x1, y1] = b.map(|v| v.round().max(0.0) as usize);
        let (x1, y1) = (x1.min(w - 1), y1.min(h - 1));
        let green = [0u8, 255, 0];
        for x in x0..=x1 {
            for y in [y0, y1] {
                img[(y * w + x) * 3..][..3].copy_from_slice(&green);
            }
        }
        for y in y0..=y1 {
            for x in [x0, x1] {
                img[(y * w + x) * 3..][..3].copy_from_slice(&green);
            }
        }
    }
    image::RgbImage::from_raw(w as u32, h as u32, img).ok_or("overlay size")?.save(path)?;
    Ok(())
}

fn overlay(rgb: &[u8], w: usize, h: usize, inst: &[Instance], path: &Path) -> R<()> {
    let mut img = rgb.to_vec();
    let colors: [[u8; 3]; 6] =
        [[230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48], [145, 30, 180]];
    for (k, it) in inst.iter().enumerate() {
        let col = colors[k % colors.len()];
        for (p, &on) in it.mask.iter().enumerate() {
            if on {
                for ch in 0..3 {
                    img[p * 3 + ch] = ((img[p * 3 + ch] as u16 + col[ch] as u16) / 2) as u8;
                }
            }
        }
        let [x0, y0, x1, y1] = it.bbox.map(|v| v.round().max(0.0) as usize);
        let (x1, y1) = (x1.min(w - 1), y1.min(h - 1));
        for t in 0..3 {
            for x in x0..=x1 {
                for y in [y0 + t, y1.saturating_sub(t)] {
                    if y < h {
                        img[(y * w + x) * 3..][..3].copy_from_slice(&col);
                    }
                }
            }
            for y in y0..=y1 {
                for x in [x0 + t, x1.saturating_sub(t)] {
                    if x < w {
                        img[(y * w + x) * 3..][..3].copy_from_slice(&col);
                    }
                }
            }
        }
    }
    image::RgbImage::from_raw(w as u32, h as u32, img).ok_or("overlay size")?.save(path)?;
    Ok(())
}

fn segment(args: &[String], mode: Mode) -> R<()> {
    let (dir, img, prompt) = (Path::new(&args[0]), Path::new(&args[1]), &args[2]);
    let mut out_path: Option<PathBuf> = None;
    let mut threshold = 0.5f32;
    let mut reps = 1usize;
    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "-o" => out_path = Some(PathBuf::from(&args[i + 1])),
            "--threshold" => threshold = args[i + 1].parse()?,
            "--reps" => reps = args[i + 1].parse()?,
            "--threads" => taconite_sam3::cpu::set_threads(args[i + 1].parse()?),
            a => return Err(format!("unknown argument {a}").into()),
        }
        i += 2;
    }
    let t0 = Instant::now();
    let mut m = Sam3::load_mode(dir, mode)?;
    println!("loaded {} ({:.1} s); mode {} on {}", dir.display(), t0.elapsed().as_secs_f64(), m.mode(), m.devices());
    let (rgb, w, h) = load_rgb(img)?;
    let mut out: Option<Output> = None;
    for r in 0..reps {
        m.timing.clear();
        let t1 = Instant::now();
        let px = preprocess(&rgb, w, h, m.cfg.image_size);
        out = Some(m.segment(&px, prompt)?);
        let (loads, evictions) = m.contexts();
        println!(
            "forward {r}: {:.0} ms ({}; contexts loaded {loads}, evicted {evictions})",
            t1.elapsed().as_secs_f64() * 1e3,
            m.timing
        );
    }
    let out = out.unwrap();
    let inst = instances(&out, m.cfg.mask_size, w, h, threshold, 0.5);
    println!("{prompt:?}: {} instance(s)", inst.len());
    for it in &inst {
        let area = it.mask.iter().filter(|&&v| v).count();
        println!(
            "  score {:.3}  box [{:.1}, {:.1}, {:.1}, {:.1}]  mask {area} px",
            it.score, it.bbox[0], it.bbox[1], it.bbox[2], it.bbox[3]
        );
    }
    if let Some(p) = out_path {
        overlay(&rgb, w, h, &inst, &p)?;
        println!("wrote {}", p.display());
    }
    Ok(())
}

/// Host <-> device-buffer copy rates (the host mapping of an XRT BO).
fn bench_copy() -> R<()> {
    let s = taconite_sam3::npu::Session::open(0)?;
    let n = 16 << 20; // u16s = 32 MB
    let mut b = s.alloc_of::<u16>(n)?;
    let src = vec![1u16; n];
    for _ in 0..3 {
        let t = Instant::now();
        taconite_sam3::npu::push(&src, &mut b)?;
        let push_s = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let v = taconite_sam3::npu::pull::<u16>(&b, n)?;
        let pull_s = t.elapsed().as_secs_f64();
        let t = Instant::now();
        b.sync_from_device()?;
        let sync_s = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let x: u64 = b.as_slice::<u16>().iter().step_by(64).map(|&v| v as u64).sum();
        let strided_s = t.elapsed().as_secs_f64();
        println!(
            "push {:.1} GB/s, pull {:.1} GB/s, sync_from_device {:.1} ms, strided read {:.1} ms ({} {})",
            64e-3 / push_s * 0.5,
            64e-3 / pull_s * 0.5,
            sync_s * 1e3,
            strided_s * 1e3,
            v[0],
            x
        );
    }
    Ok(())
}

fn main() -> ExitCode {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mode = match args.iter().position(|a| a == "--mode") {
        Some(i) if i + 1 < args.len() => {
            let m = Mode::parse(&args[i + 1]);
            args.drain(i..i + 2);
            m
        }
        _ => Mode::from_env(),
    };
    let mode = match mode {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
    };
    if args.first().map(String::as_str) == Some("bench-copy") {
        return match bench_copy() {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        };
    }
    let r = match args.first().map(String::as_str) {
        Some("check") if args.len() == 2 => check(Path::new(&args[1]), mode).map(|ok| ok as u8),
        Some("segment") if args.len() >= 4 => segment(&args[1..], mode).map(|_| 1),
        Some("points") if args.len() >= 4 => points(&args[1..], mode).map(|_| 1),
        _ => {
            eprintln!(
                "usage: sam3 check <bundle> [--mode npu|npu+gpu]\n       sam3 segment <bundle> <image> <prompt> [-o overlay.png] [--threshold 0.5] [--reps N] [--threads N] [--mode npu|npu+gpu]\n       sam3 points <bundle> <image> <x,y[,label];...> [--box x1,y1,x2,y2] [--single] [-o overlay.png] [--reps N] [--threads N]\n                   (image pixels; label 1 = object, 0 = background; \"\" for a box alone)"
            );
            return ExitCode::from(2);
        }
    };
    match r {
        Ok(1) => ExitCode::SUCCESS,
        Ok(_) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
