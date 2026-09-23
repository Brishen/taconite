// SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
// SPDX-License-Identifier: Apache-2.0

//! Builds an IRON element-wise multiply from source and runs it: the kernel
//! (`aie_kernels/generic/mul.cc`) through Peano, the design's MLIR (what
//! `ElementwiseMul`'s design.py generates — `build/ElementwiseMul_*.mlir`
//! after the operator's test ran) through aiecc, then the result on the NPU
//! against the product on the CPU.
//!
//! ```text
//! MLIR_AIE_INSTALL_DIR=…/site-packages/mlir_aie \
//!   cargo run --release --example eltwise_mul -- <design.mlir> <mul.cc> <out dir>
//! ```
//!
//! `$IRON_TOOL_LAUNCHER`, when set, is a command that runs the compilers
//! (see `Toolchain::with_launcher`), e.g. a script that execs its arguments
//! in the IRON container on a host that can't run mlir-aie's binaries.

use std::path::PathBuf;
use std::time::Instant;

use iron_xrt::compile::{Arch, Design, KernelSource, Toolchain};
use iron_xrt::{Session, bf16_to_f32, f32_to_bf16};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [mlir, source, out] = &args[..] else {
        return Err("usage: eltwise_mul <design.mlir> <mul.cc> <out dir>".into());
    };
    let (mlir, out) = (PathBuf::from(mlir), PathBuf::from(out));
    let text = std::fs::read_to_string(&mlir)?;
    let arch = Arch::of_mlir(&text).ok_or("the MLIR names no npu1/npu2 device")?;
    let n = elements(&text).ok_or("no memref<Nxbf16> in the runtime sequence")?;

    let mut tc = Toolchain::from_env()?;
    if let Some(launcher) = std::env::var_os("IRON_TOOL_LAUNCHER") {
        tc = tc.with_launcher([launcher]);
    }
    let obj = out.join(arch.as_str()).join("mul.o");
    let (xclbin, insts) = (out.join("eltwise_mul.xclbin"), out.join("eltwise_mul.insts.bin"));

    let t = Instant::now();
    tc.compile_kernel(&KernelSource::new(source, arch), &obj)?;
    println!("kernel  {} ({:.1} s)", obj.display(), t.elapsed().as_secs_f64());
    let t = Instant::now();
    tc.compile_design(&Design::new(&mlir, out.join("work")).link(&obj), &xclbin, &insts)?;
    println!("design  {} + {} ({:.1} s)", xclbin.display(), insts.display(), t.elapsed().as_secs_f64());

    let session = Session::open(0)?;
    let kernel = session.load_kernel(&xclbin, &insts, None, n as u64)?;
    let (mut a, mut b, c) = (session.alloc_of::<u16>(n)?, session.alloc_of::<u16>(n)?, session.alloc_of::<u16>(n)?);
    // A fixed LCG over [-4, 4): reproducible without a rand dependency.
    let mut seed = 0x2545_f491u32;
    let mut next = || {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        f32_to_bf16((seed >> 8) as f32 / (1u32 << 24) as f32 * 8.0 - 4.0)
    };
    a.as_mut_slice::<u16>().iter_mut().for_each(|x| *x = next());
    b.as_mut_slice::<u16>().iter_mut().for_each(|x| *x = next());
    a.sync_to_device()?;
    b.sync_to_device()?;
    let took = kernel.run(&[&a, &b, &c])?;
    c.sync_from_device()?;

    let (a, b, c) = (a.as_slice::<u16>(), b.as_slice::<u16>(), c.as_slice::<u16>());
    let mut bad = 0;
    for i in 0..n {
        let want = bf16_to_f32(f32_to_bf16(bf16_to_f32(a[i]) * bf16_to_f32(b[i])));
        let got = bf16_to_f32(c[i]);
        if (got - want).abs() > 0.04 * want.abs() + 1e-6 {
            if bad < 5 {
                eprintln!("[{i}] {} * {} = {got}, want {want}", bf16_to_f32(a[i]), bf16_to_f32(b[i]));
            }
            bad += 1;
        }
    }
    println!("npu     {n} elements in {:.3} ms, {bad} mismatches", took.as_secs_f64() * 1e3);
    if bad > 0 {
        return Err(format!("{bad} of {n} elements differ from the CPU").into());
    }
    Ok(())
}

/// The element count of the runtime sequence's first `memref<Nxbf16>`.
fn elements(mlir: &str) -> Option<usize> {
    let seq = &mlir[mlir.find("aie.runtime_sequence(")?..];
    let m = &seq[seq.find("memref<")? + "memref<".len()..];
    m[..m.find("xbf16>")?].parse().ok()
}
