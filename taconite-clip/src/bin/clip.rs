// SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
// SPDX-License-Identifier: Apache-2.0

//! `clip`: CLIP ViT-H/14 on the NPU from the command line.
//!
//! ```text
//! clip classify <bundle> <image>... --label <name>... [--reps N]
//! clip embed <bundle> [--image <file>]... [--text <prompt>]... -o <out.f32>
//! clip check <bundle>
//! ```

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use taconite_clip::{Clip, logits, softmax};
use taconite_sam3::bundle::unescape;

type R<T> = Result<T, Box<dyn std::error::Error>>;

fn load_rgb(path: &Path) -> R<(Vec<u8>, usize, usize)> {
    let im = image::open(path).map_err(|e| format!("{}: {e}", path.display()))?.to_rgb8();
    let (w, h) = (im.width() as usize, im.height() as usize);
    Ok((im.into_raw(), w, h))
}

fn pixels(clip: &Clip, images: &[PathBuf]) -> R<Vec<f32>> {
    let mut px = Vec::new();
    for p in images {
        let (rgb, w, h) = load_rgb(p)?;
        px.extend(clip.preprocess(&rgb, w, h));
    }
    Ok(px)
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let d = |x: &[f32], y: &[f32]| x.iter().zip(y).map(|(p, q)| (*p as f64) * (*q as f64)).sum::<f64>();
    (d(a, b) / (d(a, a) * d(b, b)).sqrt()) as f32
}

/// Min and mean row cosine.
fn cosines(a: &[f32], b: &[f32], dim: usize) -> (f32, f32) {
    let c: Vec<f32> = a.chunks(dim).zip(b.chunks(dim)).map(|(x, y)| cosine(x, y)).collect();
    (c.iter().cloned().fold(1.0, f32::min), c.iter().sum::<f32>() / c.len() as f32)
}

fn argmax(r: &[f32]) -> usize {
    r.iter().enumerate().fold(0, |b, (i, v)| if *v > r[b] { i } else { b })
}

fn classify(args: &[String]) -> R<()> {
    let dir = Path::new(&args[0]);
    let (mut images, mut labels, mut reps) = (Vec::new(), Vec::new(), 1);
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--label" => {
                labels.push(args[i + 1].clone());
                i += 1;
            }
            "--reps" => {
                reps = args[i + 1].parse()?;
                i += 1;
            }
            "--threads" => {
                taconite_sam3::cpu::set_threads(args[i + 1].parse()?);
                i += 1;
            }
            a => images.push(PathBuf::from(a)),
        }
        i += 1;
    }
    if images.is_empty() || labels.is_empty() {
        return Err("give at least one image and one --label".into());
    }
    let t0 = Instant::now();
    let mut clip = Clip::load(dir)?;
    eprintln!(
        "loaded {} ({} hardware contexts) in {:.1} s",
        dir.display(),
        clip.contexts(),
        t0.elapsed().as_secs_f64()
    );
    let prompts: Vec<String> = labels.iter().map(|l| clip.prompt(l)).collect();
    let prompts: Vec<&str> = prompts.iter().map(String::as_str).collect();
    let e = clip.cfg.embed_dim;
    let mut probs = Vec::new();
    for rep in 0..reps {
        clip.timing.clear();
        let t = Instant::now();
        let px = pixels(&clip, &images)?;
        let tp = t.elapsed();
        let img = clip.encode_images(&px)?;
        let ti = t.elapsed();
        let txt = clip.encode_texts(&prompts)?;
        let tt = t.elapsed();
        probs = softmax(&logits(&img, &txt, e, clip.cfg.logit_scale), labels.len());
        eprintln!(
            "forward {rep}: {:.0} ms (decode + preprocess {:.0}, images {:.0}, texts {:.0}); {}",
            tt.as_secs_f64() * 1e3,
            tp.as_secs_f64() * 1e3,
            (ti - tp).as_secs_f64() * 1e3,
            (tt - ti).as_secs_f64() * 1e3,
            clip.timing
        );
    }
    for (p, row) in images.iter().zip(probs.chunks(labels.len())) {
        let mut idx: Vec<usize> = (0..labels.len()).collect();
        idx.sort_by(|a, b| row[*b].total_cmp(&row[*a]));
        let top: Vec<String> = idx.iter().take(3).map(|&j| format!("{} {:.3}", labels[j], row[j])).collect();
        println!("{}: {}", p.display(), top.join(", "));
    }
    Ok(())
}

fn embed(args: &[String]) -> R<()> {
    let dir = Path::new(&args[0]);
    let (mut images, mut texts, mut out) = (Vec::new(), Vec::new(), None);
    let mut i = 1;
    while i + 1 < args.len() {
        match args[i].as_str() {
            "--image" => images.push(PathBuf::from(&args[i + 1])),
            "--text" => texts.push(args[i + 1].clone()),
            "-o" => out = Some(PathBuf::from(&args[i + 1])),
            a => return Err(format!("unknown argument {a}").into()),
        }
        i += 2;
    }
    let out = out.ok_or("-o <file> is required")?;
    let mut clip = Clip::load(dir)?;
    let mut emb = Vec::new();
    if !images.is_empty() {
        let px = pixels(&clip, &images)?;
        emb.extend(clip.encode_images(&px)?);
    }
    if !texts.is_empty() {
        let t: Vec<&str> = texts.iter().map(String::as_str).collect();
        emb.extend(clip.encode_texts(&t)?);
    }
    let bytes: Vec<u8> = emb.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(&out, bytes)?;
    eprintln!(
        "wrote {} ({} x {} f32: {} images then {} texts, projected, not normalised)",
        out.display(),
        images.len() + texts.len(),
        clip.cfg.embed_dim,
        images.len(),
        texts.len()
    );
    Ok(())
}

/// Reference tensor `ref.<j>.<name>`.
fn refv(clip: &Clip, j: usize, name: &str) -> R<Vec<f32>> {
    Ok(clip.store.f32(&format!("ref.{j}.{name}"))?.to_vec())
}

struct Checker {
    ok: bool,
}

impl Checker {
    fn check(&mut self, pass: bool, msg: String) {
        println!("  [{}] {msg}", if pass { "ok" } else { "FAIL" });
        self.ok &= pass;
    }

    fn info(&self, msg: String) {
        println!("  [info] {msg}");
    }
}

fn check(dir: &Path) -> R<bool> {
    let mut clip = Clip::load(dir)?;
    let m = clip.manifest.clone();
    let e = clip.cfg.embed_dim;
    let mut c = Checker { ok: true };
    println!("{}: {} hardware contexts", dir.display(), clip.contexts());

    println!("tokenizer:");
    let tests: Vec<_> = m.tagged("tok_test").cloned().collect();
    let same = tests
        .iter()
        .filter(|r| {
            let idx: usize = r.field_as(0).unwrap();
            let text = unescape(&r.rest(1));
            let want = clip.store.i32(&format!("tok.{idx}.ids")).unwrap();
            let got = clip.tokenize(&text);
            let eq = got.iter().zip(want).all(|(a, b)| *a as i32 == *b) && got.len() == want.len();
            if !eq {
                println!("    {text:?}: {got:?}\n      HF: {want:?}");
            }
            eq
        })
        .count();
    c.check(same == tests.len(), format!("{same}/{} texts tokenize as HF's CLIPTokenizer", tests.len()));

    for set in m.tagged("refset").cloned().collect::<Vec<_>>() {
        let j: usize = set.field_as(0)?;
        let (ni, np): (usize, usize) = (set.get("images")?, set.get("prompts")?);
        let images: Vec<PathBuf> = m
            .tagged("ref_image")
            .filter(|r| r.field(0).ok() == Some(&j.to_string()))
            .map(|r| dir.join(r.rest(2)))
            .collect();
        let labels: Vec<String> =
            m.tagged("ref_label").filter(|r| r.field(0).ok() == Some(&j.to_string())).map(|r| r.rest(2)).collect();
        assert_eq!((images.len(), labels.len()), (ni, np));
        println!("reference set {j}: {ni} images x {np} prompts ({})", labels.join(", "));

        // preprocessing, from the image files
        let px_ref = refv(&clip, j, "pixel_values")?;
        let px = pixels(&clip, &images)?;
        let img_len = px.len() / ni;
        for (i, p) in images.iter().enumerate() {
            let (a, b) = (&px[i * img_len..][..img_len], &px_ref[i * img_len..][..img_len]);
            let diff = a.iter().zip(b).filter(|(x, y)| x != y).count();
            let max = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max);
            let name = p.file_name().unwrap().to_string_lossy();
            let msg = format!("preprocess {name}: {diff} of {img_len} values differ from HF's (max {max:.4})");
            if name.ends_with(".png") {
                c.check(diff == 0, msg);
            } else {
                // JPEG decoding (image crate vs PIL's libjpeg) differs by a
                // level here and there
                c.info(msg);
            }
        }

        // the towers, on the reference's own inputs
        let img = clip.encode_images(&px_ref)?;
        let ids: Vec<u32> = clip.store.i32(&format!("ref.{j}.input_ids"))?.iter().map(|&v| v as u32).collect();
        let txt = clip.encode_token_ids(&ids)?;
        for (what, got, key) in [("image", &img, "image_embeds"), ("text", &txt, "text_embeds")] {
            let (min_f, mean_f) = cosines(got, &refv(&clip, j, key)?, e);
            let (min_n, _) = cosines(got, &refv(&clip, j, &format!("npu_{key}"))?, e);
            let bar = if what == "image" { 0.999 } else { 0.998 };
            c.check(
                min_f > bar && min_n > 0.999,
                format!(
                    "{what} embeddings: cosine to float32 min {min_f:.5} (mean {mean_f:.5}); to the Python NPU app min {min_n:.5}"
                ),
            );
        }
        let lg = logits(&img, &txt, e, clip.cfg.logit_scale);
        let lr = refv(&clip, j, "logits")?;
        let dmax = lg.iter().zip(&lr).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        let (p, pr) = (softmax(&lg, np), softmax(&lr, np));
        let pmax = p.iter().zip(&pr).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        let top = p.chunks(np).zip(pr.chunks(np)).filter(|(a, b)| argmax(a) == argmax(b)).count();
        c.check(
            top == ni && pmax < 0.03,
            format!("zero-shot: top-1 agrees {top}/{ni}; logits max |diff| {dmax:.3}, probabilities {pmax:.4}"),
        );

        // end to end: image files + label prompts
        let prompts: Vec<String> = labels.iter().map(|l| clip.prompt(l)).collect();
        let prompts: Vec<&str> = prompts.iter().map(String::as_str).collect();
        let img = clip.encode_images(&px)?;
        let txt = clip.encode_texts(&prompts)?;
        let p = softmax(&logits(&img, &txt, e, clip.cfg.logit_scale), np);
        for (i, (row, rrow)) in p.chunks(np).zip(pr.chunks(np)).enumerate() {
            let (k, kr) = (argmax(row), argmax(rrow));
            c.check(
                k == kr && (row[k] - rrow[k]).abs() < 0.03,
                format!(
                    "{}: {} {:.3} (reference {} {:.3})",
                    images[i].file_name().unwrap().to_string_lossy(),
                    labels[k],
                    row[k],
                    labels[kr],
                    rrow[kr]
                ),
            );
        }
    }
    println!("{}", if c.ok { "ALL CHECKS PASSED" } else { "SOME CHECKS FAILED" });
    Ok(c.ok)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let r = match args.first().map(String::as_str) {
        Some("classify") if args.len() >= 3 => classify(&args[1..]).map(|_| true),
        Some("embed") if args.len() >= 2 => embed(&args[1..]).map(|_| true),
        Some("check") if args.len() == 2 => check(Path::new(&args[1])),
        _ => {
            eprintln!(
                "usage:\n  clip classify <bundle> <image>... --label <name>... [--reps N] [--threads N]\n  \
                 clip embed <bundle> [--image <file>]... [--text <prompt>]... -o <out.f32>\n  clip check <bundle>"
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
