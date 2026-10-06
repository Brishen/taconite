// SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
// SPDX-License-Identifier: Apache-2.0

//! The vision tower alone: `cargo run --release --example vision -- <bundle> <image>`
//! prints the grid, the token count and the time of each of three encodes.

use std::path::Path;
use std::time::Instant;

use taconite_qwen35::{Qwen35, RgbImage};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [bundle, path] = &args[..] else { return Err("usage: vision <bundle> <image>".into()) };
    let img = image::open(path)?.to_rgb8();
    let img = RgbImage { width: img.width() as usize, height: img.height() as usize, rgb: img.into_raw() };
    let mut q = Qwen35::load(Path::new(bundle), 4096)?;
    let (patches, grid) = q.preprocess(&img)?;
    for _ in 0..3 {
        q.timing.clear();
        let t0 = Instant::now();
        let emb = q.model.encode_image(&patches, grid, &mut q.timing)?;
        println!("grid {grid:?}: {} tokens in {:.2} s; {}", emb.len() / 2048, t0.elapsed().as_secs_f64(), q.timing);
    }
    Ok(())
}
