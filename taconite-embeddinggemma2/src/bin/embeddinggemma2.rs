// SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
// SPDX-License-Identifier: Apache-2.0

//! `embeddinggemma2`: EmbeddingGemma 2 image and text embeddings on the NPU.
//!
//! ```text
//! embeddinggemma2 embed <bundle> [<image>...] [--text <text>]... [--prompt <name>]
//!                       [--dim 768|512|256|128] [-o out.f32] [--reps N]
//! embeddinggemma2 prompts <bundle>
//! embeddinggemma2 check <bundle>
//! ```
//!
//! `embed` prints each input's time and leading values and, for several,
//! their cosine similarities; `-o` writes the embeddings as raw
//! little-endian f32, a row an input (images first, then texts). `--prompt`
//! prefixes every text with one of the model's task prompts (`prompts`
//! lists them: SearchQuery for queries, Document for what they search,
//! ...). `check` runs the bundle's reference images and texts against the
//! float32 sentence-transformers results.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use taconite_embeddinggemma2::{EmbeddingGemma2, cosine, truncate};

type R<T> = Result<T, Box<dyn std::error::Error>>;

fn load_rgb(path: &Path) -> R<(Vec<u8>, usize, usize)> {
    let im = image::open(path).map_err(|e| format!("{}: {e}", path.display()))?.to_rgb8();
    let (w, h) = (im.width() as usize, im.height() as usize);
    Ok((im.into_raw(), w, h))
}

fn timing_line(m: &EmbeddingGemma2) -> String {
    let t = &m.timing;
    format!(
        "npu {:.0} ms (mha {:.0}), host vision {:.0} ms, attn staging {:.0} ms, text {:.0} ms; {} context swaps so far",
        t.npu_total().as_secs_f64() * 1e3,
        t.get("npu:mha").as_secs_f64() * 1e3,
        t.get("v.host").as_secs_f64() * 1e3,
        t.get("v.attn").as_secs_f64() * 1e3,
        (t.get("t.host") + t.get("t.attn")).as_secs_f64() * 1e3,
        m.npu.swaps.get(),
    )
}

fn embed(args: &[String]) -> R<()> {
    let dir = Path::new(args.first().ok_or("give a bundle")?);
    let (mut images, mut out, mut dim, mut reps) = (Vec::new(), None, 768, 1);
    let (mut texts, mut prompt): (Vec<String>, Option<String>) = (Vec::new(), None);
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-o" => {
                out = Some(PathBuf::from(&args[i + 1]));
                i += 1;
            }
            "--dim" => {
                dim = args[i + 1].parse()?;
                i += 1;
            }
            "--reps" => {
                reps = args[i + 1].parse()?;
                i += 1;
            }
            "--threads" => {
                taconite::cpu::set_threads(args[i + 1].parse()?);
                i += 1;
            }
            "--text" => {
                texts.push(args.get(i + 1).ok_or("--text needs a text")?.clone());
                i += 1;
            }
            "--prompt" => {
                prompt = Some(args.get(i + 1).ok_or("--prompt needs a name")?.clone());
                i += 1;
            }
            a => images.push(PathBuf::from(a)),
        }
        i += 1;
    }
    if images.is_empty() && texts.is_empty() {
        return Err("give at least one image or --text".into());
    }
    if ![768, 512, 256, 128].contains(&dim) {
        return Err(format!("--dim {dim}: the model is trained for 768, 512, 256 or 128").into());
    }
    let t0 = Instant::now();
    let mut m = EmbeddingGemma2::load(dir)?;
    eprintln!("loaded {} ({} hardware contexts) in {:.1} s", dir.display(), m.contexts(), t0.elapsed().as_secs_f64());
    let mut rows = Vec::new();
    let mut labels: Vec<String> = Vec::new();
    for p in &images {
        let (rgb, w, h) = load_rgb(p)?;
        let mut e = Vec::new();
        for _ in 0..reps {
            let t1 = Instant::now();
            e = m.embed_rgb(&rgb, w, h)?;
            eprintln!("{}: {:.0} ms; {}", p.display(), t1.elapsed().as_secs_f64() * 1e3, timing_line(&m));
        }
        let e = truncate(&e, dim);
        println!("{}: [{}, ...]", p.display(), e[..6].iter().map(|v| format!("{v:.4}")).collect::<Vec<_>>().join(", "));
        rows.push(e);
        labels.push(p.display().to_string());
    }
    for text in &texts {
        let mut e = Vec::new();
        for _ in 0..reps {
            let t1 = Instant::now();
            e = m.embed_text(text, prompt.as_deref())?;
            eprintln!(
                "{text:?}: {} tokens, {:.0} ms; {}",
                m.tokenize(text, prompt.as_deref())?.len(),
                t1.elapsed().as_secs_f64() * 1e3,
                timing_line(&m)
            );
        }
        let e = truncate(&e, dim);
        println!("{text:?}: [{}, ...]", e[..6].iter().map(|v| format!("{v:.4}")).collect::<Vec<_>>().join(", "));
        rows.push(e);
        labels.push(format!("{text:?}"));
    }
    if rows.len() > 1 {
        println!("cosine similarity:");
        for (a, ra) in rows.iter().enumerate() {
            let line: Vec<String> = rows.iter().map(|rb| format!("{:.3}", cosine(ra, rb))).collect();
            println!("  {}: {}", labels[a], line.join(" "));
        }
    }
    if let Some(out) = out {
        let mut f = std::fs::File::create(&out)?;
        for e in &rows {
            for v in e {
                f.write_all(&v.to_le_bytes())?;
            }
        }
        eprintln!("wrote {} x {dim} f32 to {}", rows.len(), out.display());
    }
    Ok(())
}

fn hex_text(h: &str) -> String {
    let b: Vec<u8> = (0..h.len() / 2).filter_map(|i| u8::from_str_radix(&h[2 * i..2 * i + 2], 16).ok()).collect();
    String::from_utf8_lossy(&b).into_owned()
}

fn check(args: &[String]) -> R<bool> {
    let dir = Path::new(args.first().ok_or("give a bundle")?);
    let t0 = Instant::now();
    let mut m = EmbeddingGemma2::load(dir)?;
    eprintln!("loaded {} ({} hardware contexts) in {:.1} s", dir.display(), m.contexts(), t0.elapsed().as_secs_f64());
    let refs: Vec<(String, usize, usize)> = m
        .manifest
        .tagged("ref")
        .map(|r| Ok((r.field(0)?.to_string(), r.get("w")?, r.get("h")?)))
        .collect::<Result<_, taconite_bundle::Error>>()?;
    if refs.is_empty() {
        return Err("the bundle has no reference images (export with --refs)".into());
    }
    let mut ok = true;
    let mut embs = Vec::new();
    let mut ref_embs = Vec::new();
    for (name, w, h) in &refs {
        let g = |k: &str| format!("ref.{name}.{k}");
        let rgb = m.store.u8(&g("rgb"))?.to_vec();
        let p = m.preprocess(&rgb, *w, *h)?;
        let rp = m.store.f32(&g("pixels"))?;
        let rpos = m.store.i32(&g("positions"))?;
        let px_err = if rp.len() == p.pixels.len() {
            p.pixels.iter().zip(rp).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max)
        } else {
            f32::INFINITY
        };
        let pos_ok = rpos.len() == 2 * p.positions.len()
            && p.positions.iter().enumerate().all(|(i, &(x, y))| rpos[2 * i] as usize == x && rpos[2 * i + 1] as usize == y);
        let soft = m.soft_tokens(&p)?;
        let rsoft = m.store.f32(&g("soft"))?.to_vec();
        let t1 = Instant::now();
        let e = m.embed_patches(&p)?;
        let dt = t1.elapsed();
        let re = m.store.f32(&g("embedding"))?.to_vec();
        let pe = m.store.f32(&g("npu_embedding"))?.to_vec();
        let (c_ref, c_py, c_soft) = (cosine(&e, &re), cosine(&e, &pe), cosine(&soft, &rsoft));
        let pass = px_err <= 1e-6 && pos_ok && c_ref >= 0.999;
        ok &= pass;
        println!(
            "{name} ({w} x {h}, grid {} x {}): pixels max|d| {px_err:.1e}, positions {}, soft tokens cos {c_soft:.5}, \
             embedding cos {c_ref:.6} vs float32 HF, {c_py:.6} vs the Python NPU app; {:.0} ms -- {}",
            p.gh,
            p.gw,
            if pos_ok { "equal" } else { "DIFFER" },
            dt.as_secs_f64() * 1e3,
            if pass { "ok" } else { "FAIL" }
        );
        println!("  {}", timing_line(&m));
        embs.push(e);
        ref_embs.push(re);
    }
    if m.store.has("ref.texts") {
        let texts: Vec<String> =
            m.manifest.tagged("text").map(|r| r.field(1).map(hex_text)).collect::<Result<_, _>>()?;
        let temb = m.store.f32("ref.texts")?;
        let d = temb.len() / texts.len();
        println!("image x text cosine (ours / float32 HF):");
        let mut max_d = 0f32;
        for (i, (name, _, _)) in refs.iter().enumerate() {
            let ours: Vec<f32> = temb.chunks(d).map(|t| cosine(&embs[i], t)).collect();
            let theirs: Vec<f32> = temb.chunks(d).map(|t| cosine(&ref_embs[i], t)).collect();
            let am = |v: &[f32]| v.iter().enumerate().fold(0, |b, (j, x)| if *x > v[b] { j } else { b });
            let same = am(&ours) == am(&theirs);
            ok &= same;
            max_d = ours.iter().zip(&theirs).map(|(a, b)| (a - b).abs()).fold(max_d, f32::max);
            println!(
                "  {name}: best \"{}\" ({}) -- {}",
                texts[am(&ours)],
                ours.iter().zip(&theirs).map(|(a, b)| format!("{a:.3}/{b:.3}")).collect::<Vec<_>>().join(" "),
                if same { "ok" } else { "FAIL: the reference ranks another text first" }
            );
        }
        println!("  max |d| {max_d:.4}");
    }
    let trefs: Vec<(usize, String, String)> = m
        .manifest
        .tagged("tref")
        .map(|r| Ok((r.field_as(0)?, r.field(1)?.to_string(), hex_text(r.field(2)?))))
        .collect::<Result<_, taconite_bundle::Error>>()?;
    if !trefs.is_empty() {
        println!("texts (tokens, ids vs HF, embedding cos vs float32 / vs the Python NPU app):");
        let (mut worst, mut ids_ok) = (1f32, 0);
        for (j, prompt, text) in &trefs {
            let prompt = (prompt != "-").then_some(prompt.as_str());
            let ids = m.tokenize(text, prompt)?;
            let rids: Vec<u32> = m.store.i32(&format!("tref.{j}.ids"))?.iter().map(|&v| v as u32).collect();
            let t1 = Instant::now();
            let e = m.embed_text(text, prompt)?;
            let dt = t1.elapsed();
            let re = m.store.f32(&format!("tref.{j}.embedding"))?.to_vec();
            let pe = m.store.f32(&format!("tref.{j}.npu_embedding"))?.to_vec();
            let (c, cp) = (cosine(&e, &re), cosine(&e, &pe));
            let pass = ids == rids && c >= 0.999;
            ok &= pass;
            ids_ok += (ids == rids) as usize;
            worst = worst.min(c);
            let short: String = text.chars().take(40).collect::<String>().replace('\n', " ");
            println!(
                "  {j:2} {:>18} {:5} ids {}, cos {c:.6} / {cp:.6}, {:.0} ms {:?}{} -- {}",
                prompt.unwrap_or("-"),
                ids.len(),
                if ids == rids { "equal" } else { "DIFFER" },
                dt.as_secs_f64() * 1e3,
                short,
                if text.chars().count() > 40 { "..." } else { "" },
                if pass { "ok" } else { "FAIL" }
            );
        }
        println!("  ids equal {ids_ok}/{}, worst cosine {worst:.6}", trefs.len());
    }
    println!("{}", if ok { "ALL CHECKS PASSED" } else { "CHECKS FAILED" });
    Ok(ok)
}

fn prompts(args: &[String]) -> R<()> {
    let dir = Path::new(args.first().ok_or("give a bundle")?);
    let m = taconite_bundle::Manifest::load(dir, taconite_embeddinggemma2::VERSION)?;
    for r in m.tagged("prompt") {
        println!("{:22} {:?}", r.field(0)?, hex_text(r.field(1)?));
    }
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let r = match args.first().map(String::as_str) {
        Some("embed") => embed(&args[1..]).map(|_| true),
        Some("check") => check(&args[1..]),
        Some("prompts") => prompts(&args[1..]).map(|_| true),
        _ => {
            eprintln!(
                "usage:\n  embeddinggemma2 embed <bundle> [<image>...] [--text <text>]... [--prompt <name>]\n  \
                 \x20                     [--dim 768|512|256|128] [-o out.f32] [--reps N]\n  \
                 embeddinggemma2 prompts <bundle>\n  embeddinggemma2 check <bundle>"
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
