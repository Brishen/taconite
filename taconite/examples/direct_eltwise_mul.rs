// SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
// SPDX-License-Identifier: Apache-2.0

//! Runs a built IRON element-wise multiply (see `eltwise_mul`) through XRT
//! or straight through the amdxdna ioctls, and checks it against the CPU.
//!
//! ```text
//! cargo run --release --example direct_eltwise_mul -- <xrt|direct> <xclbin> <insts.bin> <elements>
//! ```

use std::path::Path;
use std::time::{Duration, Instant};

use taconite::{Session, bf16_to_f32, f32_to_bf16};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [mode, xclbin, insts, n] = &args[..] else {
        return Err("usage: direct_eltwise_mul <xrt|direct> <xclbin> <insts.bin> <elements>".into());
    };
    let (xclbin, insts, n) = (Path::new(xclbin), Path::new(insts), n.parse::<usize>()?);
    let (a, b) = (inputs(n, 0x2545_f491), inputs(n, 0x9e37_79b9));

    let t = Instant::now();
    let (c, times) = match mode.as_str() {
        "xrt" => run_xrt(xclbin, insts, &a, &b)?,
        "direct" => run_direct(xclbin, insts, &a, &b)?,
        other => return Err(format!("unknown mode {other}").into()),
    };
    println!("{mode}: setup+runs {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
    report(&times);

    let bad = (0..n)
        .filter(|&i| {
            let want = bf16_to_f32(f32_to_bf16(bf16_to_f32(a[i]) * bf16_to_f32(b[i])));
            (bf16_to_f32(c[i]) - want).abs() > 0.04 * want.abs() + 1e-6
        })
        .count();
    println!("{n} elements, {bad} mismatches");
    if bad > 0 {
        return Err(format!("{bad} of {n} elements differ from the CPU").into());
    }
    Ok(())
}

const RUNS: usize = 20;

fn run_xrt(
    xclbin: &Path,
    insts: &Path,
    a: &[u16],
    b: &[u16],
) -> Result<(Vec<u16>, Vec<Duration>), Box<dyn std::error::Error>> {
    let n = a.len();
    let t = Instant::now();
    let session = Session::open(0)?;
    let opened = t.elapsed();
    let t = Instant::now();
    let kernel = session.load_kernel(xclbin, insts, None, n as u64)?;
    let loaded = t.elapsed();
    let t = Instant::now();
    let (mut ab, mut bb, cb) = (session.alloc_of::<u16>(n)?, session.alloc_of::<u16>(n)?, session.alloc_of::<u16>(n)?);
    println!(
        "xrt: open {:.1} ms, load {:.1} ms, 3 buffers {:.1} ms",
        opened.as_secs_f64() * 1e3,
        loaded.as_secs_f64() * 1e3,
        t.elapsed().as_secs_f64() * 1e3
    );
    ab.write(a)?;
    bb.write(b)?;
    let mut times = Vec::new();
    for _ in 0..RUNS {
        times.push(kernel.run(&[&ab, &bb, &cb])?);
    }
    cb.sync_from_device()?;
    Ok((cb.as_slice::<u16>().to_vec(), times))
}

fn run_direct(
    xclbin: &Path,
    insts: &Path,
    a: &[u16],
    b: &[u16],
) -> Result<(Vec<u16>, Vec<Duration>), Box<dyn std::error::Error>> {
    use taconite::direct::Session;
    let n = a.len();
    let t = Instant::now();
    let device = Session::open(0)?;
    let opened = t.elapsed();
    let (cols, rows) = device.array();
    println!(
        "direct: {cols}x{rows} array, AIE {:?}, firmware {:?}, power {:?}",
        device.aie_version()?,
        device.firmware_version()?,
        device.power_mode()?
    );
    let t = Instant::now();
    let kernel = device.load_kernel(xclbin, insts, None, n as u64)?;
    let loaded = t.elapsed();
    let t = Instant::now();
    let (mut ab, mut bb, cb) = (device.alloc_of::<u16>(n)?, device.alloc_of::<u16>(n)?, device.alloc_of::<u16>(n)?);
    println!(
        "direct: open {:.1} ms, load {:.1} ms, 3 buffers {:.1} ms",
        opened.as_secs_f64() * 1e3,
        loaded.as_secs_f64() * 1e3,
        t.elapsed().as_secs_f64() * 1e3
    );
    ab.write(a)?;
    bb.write(b)?;
    let mut times = Vec::new();
    for _ in 0..RUNS {
        match kernel.run(&[&ab, &bb, &cb]) {
            Ok(t) => times.push(t),
            Err(e) => {
                for c in device.contexts().unwrap_or_default() {
                    eprintln!("  {c:?}");
                }
                return Err(e.into());
            }
        }
    }
    for c in device.contexts()? {
        println!("direct: context {c:?}");
    }
    cb.sync_from_device()?;
    Ok((cb.as_slice::<u16>().to_vec(), times))
}

fn report(times: &[Duration]) {
    let mut us: Vec<f64> = times.iter().map(|t| t.as_secs_f64() * 1e6).collect();
    us.sort_by(f64::total_cmp);
    println!(
        "run: first {:.0} us, median {:.0} us, min {:.0} us ({} runs)",
        times[0].as_secs_f64() * 1e6,
        us[us.len() / 2],
        us[0],
        us.len()
    );
}

/// `n` bf16 values in [-4, 4) from a fixed LCG: reproducible without a rand dependency.
fn inputs(n: usize, mut seed: u32) -> Vec<u16> {
    (0..n)
        .map(|_| {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            f32_to_bf16((seed >> 8) as f32 / (1u32 << 24) as f32 * 8.0 - 4.0)
        })
        .collect()
}
