// SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
// SPDX-License-Identifier: Apache-2.0

//! `qwen35`: chat with Qwen3.5-2B on the NPU.
//!
//! ```text
//! qwen35 <bundle> [--prompt TEXT | --prompt-file FILE] [--system TEXT]
//!                 [--image FILE]... [--thinking] [--max-new N] [--max-ctx N] [--timing]
//! qwen35 <bundle> --interactive [...]     one prompt a line from stdin
//! qwen35 check <bundle>                   verify against the exporter's references
//! ```

use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use taconite_qwen35::{ChatOptions, Qwen35, RgbImage};

const USAGE: &str = "usage: qwen35 <bundle> [--prompt TEXT | --prompt-file FILE] [--system TEXT] [--image FILE]... [--thinking] \
[--max-new N] [--max-ctx N] [--timing] [--interactive]\n       qwen35 check <bundle> [--max-ctx N]";

const EXAMPLE: &str = "Explain in three sentences why the sky is blue.";

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(e) => {
            eprintln!("qwen35: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<ExitCode, Box<dyn std::error::Error>> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let check = args.first().is_some_and(|a| a == "check");
    if check {
        args.remove(0);
    }
    let mut bundle: Option<PathBuf> = None;
    let mut prompt = EXAMPLE.to_string();
    let mut opts = ChatOptions::default();
    let (mut max_ctx, mut timing, mut interactive) = (8192usize, false, false);
    let mut it = args.into_iter();
    while let Some(a) = it.next() {
        let mut val = || it.next().ok_or_else(|| format!("{a} needs a value\n{USAGE}"));
        match a.as_str() {
            "--prompt" => prompt = val()?,
            "--prompt-file" => prompt = std::fs::read_to_string(val()?)?,
            "--system" => opts.system = Some(val()?),
            "--image" => opts.images.push(load_image(&val()?)?),
            "--thinking" => opts.thinking = true,
            "--max-new" => opts.max_new = val()?.parse()?,
            "--max-ctx" => max_ctx = val()?.parse()?,
            "--timing" => timing = true,
            "--interactive" => interactive = true,
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(ExitCode::SUCCESS);
            }
            _ if bundle.is_none() && !a.starts_with('-') => bundle = Some(a.into()),
            _ => return Err(format!("unknown argument {a}\n{USAGE}").into()),
        }
    }
    let bundle = bundle.or_else(|| std::env::var_os("QWEN35_BUNDLE").map(PathBuf::from)).ok_or(USAGE)?;

    let t0 = Instant::now();
    let mut q = Qwen35::load(&bundle, max_ctx)?;
    eprintln!("setup {:.1} s ({} hardware contexts)", t0.elapsed().as_secs_f64(), q.model.npu.contexts);

    if check {
        let ok = q.check(|s| println!("{s}"))?;
        return Ok(if ok { ExitCode::SUCCESS } else { ExitCode::FAILURE });
    }

    let answer = |q: &mut Qwen35, prompt: &str| -> Result<(), Box<dyn std::error::Error>> {
        q.timing.clear();
        let (_, st) = q.chat(prompt, &opts, |s| {
            print!("{s}");
            let _ = std::io::stdout().flush();
        })?;
        println!();
        if st.vision_s > 0.0 {
            eprintln!("vision {:.2} s", st.vision_s);
        }
        eprintln!(
            "{} prompt tokens, {} new; prefill {:.2} s, decode {:.1} tokens/s",
            st.prompt_tokens,
            st.new_tokens,
            st.prefill_s,
            st.decode_tok_s()
        );
        if timing {
            eprintln!("{}", q.timing);
            if q.model.npu.swaps.get() > 0 {
                eprintln!("{} hardware-context swaps (the device lacked a free slot)", q.model.npu.swaps.get());
            }
        }
        Ok(())
    };
    if interactive {
        for line in std::io::stdin().lock().lines() {
            let line = line?;
            if !line.trim().is_empty() {
                answer(&mut q, &line)?;
            }
        }
    } else {
        answer(&mut q, &prompt)?;
    }
    Ok(ExitCode::SUCCESS)
}

/// A JPEG or PNG file as RGB8.
fn load_image(path: &str) -> Result<RgbImage, Box<dyn std::error::Error>> {
    let img = image::open(path).map_err(|e| format!("{path}: {e}"))?.to_rgb8();
    let (width, height) = (img.width() as usize, img.height() as usize);
    Ok(RgbImage { width, height, rgb: img.into_raw() })
}
