// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The direct path on an NPU, over an element-wise multiply built by the
//! `eltwise_mul` example (three bf16 buffers of `IRON_XRT_TEST_ELEMENTS`):
//!
//! ```text
//! IRON_XRT_TEST_XCLBIN=out/eltwise_mul.xclbin IRON_XRT_TEST_INSTS=out/eltwise_mul.insts.bin \
//!   IRON_XRT_TEST_ELEMENTS=1638400 cargo test --release --test direct_hw -- --ignored
//! ```

#![cfg(all(target_os = "linux", target_arch = "x86_64"))]

use std::path::PathBuf;

use iron_xrt::direct::{Device, Kernel};
use iron_xrt::{bf16_to_f32, f32_to_bf16};

fn artifacts() -> (PathBuf, PathBuf, usize) {
    let var = |k: &str| std::env::var(k).unwrap_or_else(|_| panic!("set {k} (see the file's doc)"));
    (
        var("IRON_XRT_TEST_XCLBIN").into(),
        var("IRON_XRT_TEST_INSTS").into(),
        var("IRON_XRT_TEST_ELEMENTS").parse().unwrap(),
    )
}

/// Runs `kernel` on `a * b` for `scale`-shifted inputs and checks every element.
fn check(device: &Device, kernel: &Kernel, n: usize, scale: f32) {
    let (mut a, mut b, c) =
        (device.alloc_of::<u16>(n).unwrap(), device.alloc_of::<u16>(n).unwrap(), device.alloc_of::<u16>(n).unwrap());
    let xs: Vec<u16> = (0..n).map(|i| f32_to_bf16(((i % 251) as f32 - 125.0) / 32.0 * scale)).collect();
    let ys: Vec<u16> = (0..n).map(|i| f32_to_bf16(((i % 97) as f32 - 48.0) / 16.0)).collect();
    a.write(&xs).unwrap();
    b.write(&ys).unwrap();
    kernel.run(&[&a, &b, &c]).unwrap();
    c.sync_from_device().unwrap();
    for (i, &got) in c.as_slice::<u16>().iter().enumerate() {
        let want = bf16_to_f32(f32_to_bf16(bf16_to_f32(xs[i]) * bf16_to_f32(ys[i])));
        let got = bf16_to_f32(got);
        assert!((got - want).abs() <= 0.04 * want.abs() + 1e-6, "[{i}] {got} != {want}");
    }
}

#[test]
#[ignore = "needs an NPU and a built kernel"]
fn kernels_load_run_and_release_their_contexts() {
    let (xclbin, insts, n) = artifacts();
    let device = Device::open().unwrap();
    let mine = |d: &Device| d.contexts().unwrap().into_iter().filter(|c| c.pid == std::process::id() as i64).count();
    // More cycles of resident kernels than NPU2 has contexts (16): one that
    // is not released runs the device out of them.
    for cycle in 0..20 {
        let kernels: Vec<Kernel> = (0..3).map(|_| device.load_kernel(&xclbin, &insts, 0).unwrap()).collect();
        assert_eq!(mine(&device), 3, "cycle {cycle}");
        for (i, k) in kernels.iter().enumerate() {
            check(&device, k, n, 1.0 + i as f32);
        }
    }
    assert_eq!(mine(&device), 0);
}

#[test]
#[ignore = "needs an NPU and a built kernel"]
fn too_many_arguments_is_an_error_not_a_submission() {
    let (xclbin, insts, _) = artifacts();
    let device = Device::open().unwrap();
    let kernel = device.load_kernel(&xclbin, &insts, 0).unwrap();
    let bufs: Vec<_> = (0..6).map(|_| device.alloc(4096).unwrap()).collect();
    let refs: Vec<_> = bufs.iter().collect();
    assert!(matches!(kernel.run(&refs), Err(iron_xrt::Error::Run(_))));
}
