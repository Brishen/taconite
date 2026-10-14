// SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
// SPDX-License-Identifier: Apache-2.0

//! `reuse`: RE-USE speech enhancement on the NPU.
//!
//! ```text
//! reuse enhance <bundle> <in.wav> <out.wav> [--float] [--reps N]
//! reuse check [<bundle>]      (none: the one built in, feature `embed`)
//! ```
//!
//! `enhance` enhances every channel of a WAV file (8, 16, 22.05, 24, 32,
//! 44.1 or 48 kHz -- the rates the bundle was exported for) and writes the
//! result at the same rate, 16-bit PCM (`--float`: 32-bit float). `check`
//! runs the bundle's reference cases against the float32 network's output
//! (and the Python NPU app's, when the bundle has it).

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use taconite_reuse::wav::{self, Audio};
use taconite_reuse::{Reuse, snr};

type R<T> = Result<T, Box<dyn std::error::Error>>;

fn timing_line(m: &Reuse) -> String {
    format!("{}; {} context swaps", m.timing, m.npu.swaps.get())
}

fn enhance(args: &[String]) -> R<()> {
    let mut pos = Vec::new();
    let (mut float, mut reps) = (false, 1);
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--float" => float = true,
            "--reps" => reps = it.next().ok_or("--reps N")?.parse()?,
            _ => pos.push(a),
        }
    }
    let [bundle, inp, out] = pos[..] else {
        return Err("usage: reuse enhance <bundle> <in.wav> <out.wav> [--float] [--reps N]".into());
    };
    let t0 = Instant::now();
    let mut m = Reuse::load(&PathBuf::from(bundle))?;
    eprintln!("loaded in {:.1} s ({} hardware contexts)", t0.elapsed().as_secs_f64(), m.contexts());
    let a = wav::read(&PathBuf::from(inp))?;
    let secs = a.channels[0].len() as f64 / a.sr as f64;
    let mut outc = Vec::new();
    for (ci, ch) in a.channels.iter().enumerate() {
        let mut y = Vec::new();
        for _ in 0..reps {
            let t0 = Instant::now();
            y = m.enhance(ch, a.sr)?;
            let dt = t0.elapsed().as_secs_f64();
            eprintln!(
                "channel {ci}: {secs:.2} s at {} Hz in {:.0} ms (x{:.2} real time); {}",
                a.sr,
                dt * 1e3,
                secs / dt,
                timing_line(&m)
            );
        }
        outc.push(y);
    }
    wav::write(&PathBuf::from(out), &Audio { channels: outc, sr: a.sr }, float)?;
    Ok(())
}

fn check(args: &[String]) -> R<bool> {
    let mut m = match args {
        [bundle] => Reuse::load(&PathBuf::from(bundle))?,
        #[cfg(feature = "embed")]
        [] => Reuse::load_embedded()?,
        _ => return Err("usage: reuse check <bundle>".into()),
    };
    let cases: Vec<(String, u32)> = m
        .manifest
        .tagged("ref")
        .map(|r| Ok((r.field(0)?.to_string(), r.get("sr")?)))
        .collect::<Result<_, taconite_bundle::Error>>()?;
    if cases.is_empty() {
        return Err("the bundle has no reference cases".into());
    }
    let mut ok = true;
    for (name, sr) in cases {
        let noisy = m.store.f32(&format!("ref.{name}.noisy"))?.to_vec();
        let reference = m.store.f32(&format!("ref.{name}.wav"))?.to_vec();
        let t0 = Instant::now();
        let y = m.enhance(&noisy, sr)?;
        let dt = t0.elapsed().as_secs_f64();
        let s = snr(&y, &reference);
        // two float32 runs of this model agree to ~43 dB: the network reads
        // the phase of near-silent bins, which 1e-7 of input noise moves
        let pass = s > 30.0;
        ok &= pass;
        let mut line = format!(
            "{name} ({:.2} s at {sr} Hz): {:.1} dB SNR against the float32 network",
            noisy.len() as f64 / sr as f64,
            s
        );
        let key = format!("ref.{name}.npu_wav");
        if m.store.has(&key) {
            line += &format!(", {:.1} dB against the Python NPU app", snr(&y, m.store.f32(&key)?));
        }
        println!("{line}; {:.0} ms {}", dt * 1e3, if pass { "PASS" } else { "FAIL" });
        println!("  {}", timing_line(&m));
    }
    println!("{}", if ok { "ALL PASSED" } else { "FAILED" });
    Ok(ok)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let r = match args.first().map(String::as_str) {
        Some("enhance") => enhance(&args[1..]).map(|_| true),
        Some("check") => check(&args[1..]),
        _ => Err("usage: reuse enhance <bundle> <in.wav> <out.wav> [--float] [--reps N] | reuse check <bundle>".into()),
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
