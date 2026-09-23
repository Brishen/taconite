// SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
// SPDX-License-Identifier: Apache-2.0

//! Replaying [IRON](https://github.com/amd/iron) kernels on an AMD XDNA NPU
//! from Rust, through XRT or directly through the `amdxdna` driver.
//!
//! IRON is an ahead-of-time compiler: a kernel is an `.xclbin` (the array
//! configuration and the cores' programs) plus an instruction stream for the
//! shim DMAs (`*.insts.bin`). Running one is "allocate buffer objects, fill
//! the inputs, submit `(opcode 3, instructions, byte count, buffers…)`,
//! wait" — the XRT C++ API, wrapped here by a small C shim
//! (`iron_xrt_shim.cpp`) compiled by `build.rs`. Nothing else of IRON is
//! needed once the kernels exist.
//!
//! Three types: a [`Session`] (the device), [`Kernel`]s loaded into it (each
//! a *resident* hardware context — NPU2 allows 16 at once, and switching
//! between two costs ~2 ms of NPU time, so a caller wants few distinct
//! kernels called many times), and [`Buffer`]s — host-visible buffer objects
//! the caller writes into *directly* through [`Buffer::as_mut_slice`] and
//! syncs around a run, so activations never take an extra copy on the way
//! to or from the array.
//!
//! The `direct` module runs them with no XRT at all, through the `amdxdna`
//! driver's ioctls. The [`compile`] module builds those kernels, too — Peano
//! and the native `aiecc` as subprocesses — from a design's MLIR.
//!
//! The XRT types need the `xrt` feature (on by default), which compiles the
//! shim and links XRT. With `default-features = false` the crate is plain
//! Rust — `compile`, `direct`, [`Error`] and the bf16 helpers — and builds
//! on a machine with no XRT installed.
//!
//! Everything returns [`Result`]; the shim never prints or aborts. The
//! XRT objects may move between threads but not be shared by them, so a
//! `Session` and what it owns are `Send` and not `Sync`: a model loads on
//! one thread and runs on a worker, behind a `Mutex` if it is shared.

// The crate docs (and compile's, and direct's) link the XRT types, which
// only exist with the feature; docs.rs builds with every feature.
#![cfg_attr(not(feature = "xrt"), allow(rustdoc::broken_intra_doc_links))]

use std::fmt;
use std::path::Path;

pub mod compile;
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
pub mod direct;
#[cfg(feature = "xrt")]
mod xrt;
#[cfg(feature = "xrt")]
pub use xrt::{Buffer, Kernel, Run, Session};

/// What went wrong, in XRT's words where it has any.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// The device could not be opened (no `/dev/accel/accel0`, no driver,
    /// no permission, or the XRT libraries are missing).
    Device(String),
    /// An xclbin / instruction stream could not be loaded into a context.
    Kernel(String),
    /// A buffer object could not be allocated or synced.
    Buffer(String),
    /// A launched kernel failed or did not complete.
    Run(String),
    /// A path contained an interior NUL.
    Path(String),
    /// A kernel or design failed to build (see [`compile`]).
    Compile(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Device(m) => write!(f, "NPU device: {m}"),
            Error::Kernel(m) => write!(f, "NPU kernel: {m}"),
            Error::Buffer(m) => write!(f, "NPU buffer: {m}"),
            Error::Run(m) => write!(f, "NPU run: {m}"),
            Error::Path(m) => write!(f, "path: {m}"),
            Error::Compile(m) => write!(f, "compile: {m}"),
        }
    }
}

impl std::error::Error for Error {}

/// Where the XDNA driver exposes the first NPU; checked before opening so a
/// missing driver is a clean error rather than an XRT exception string.
pub const DEVICE_NODE: &str = "/dev/accel/accel0";

/// Whether an NPU device node exists on this machine.
pub fn npu_present() -> bool {
    Path::new(DEVICE_NODE).exists()
}

/// `bf16` as bits: the storage type of IRON's kernels. Round-to-nearest-even
/// from `f32`, the conversion ml_dtypes / torch use, so a host-side value
/// matches what the exporter wrote.
#[inline]
pub fn f32_to_bf16(x: f32) -> u16 {
    // Branchless so the glue loops vectorise: round-to-nearest-even on the
    // dropped 16 bits; a NaN keeps a set mantissa bit so it can't round to
    // an infinity.
    let bits = x.to_bits();
    let is_nan = x.is_nan() as u32;
    let round = 0x7fff + ((bits >> 16) & 1);
    let rounded = bits.wrapping_add(round) >> 16;
    let nan_val = (bits >> 16) | 0x40;
    (rounded * (1 - is_nan) + nan_val * is_nan) as u16
}

/// `e^x` to ~1e-5 relative error over ±80 (the f32 rounding of `x·log2e`
/// dominates; the polynomial itself is good to 2e-7), clamped to ±87,
/// without libm: `2^(x log2 e)` split into an integer exponent and a
/// degree-6 polynomial on the fraction. For values that end up in bf16
/// (8-bit mantissa, 4e-3) this is indistinguishable from `expf`, and the
/// loop it sits in vectorises; `f32::exp` was the tagger glue's single
/// largest cost.
#[inline]
pub fn fast_exp(x: f32) -> f32 {
    let x = x.clamp(-87.0, 87.0);
    let t = x * std::f32::consts::LOG2_E;
    let n = t.floor();
    let f = t - n;
    // 2^f on [0, 1): Cephes' exp2f polynomial (max rel err ~2e-7).
    let p = 1.0
        + f * (6.931_472e-1
            + f * (2.402_264_8e-1
                + f * (5.550_332_5e-2 + f * (9.618_438e-3 + f * (1.339_887_4e-3 + f * 1.535_336_2e-4)))));
    f32::from_bits(p.to_bits().wrapping_add((n as i32 as u32) << 23))
}

#[inline]
pub fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_round_trips_and_rounds_to_nearest_even() {
        for &x in &[0.0f32, 1.0, -2.5, std::f32::consts::PI, 1e-3, 65504.0, -1e30] {
            let b = f32_to_bf16(x);
            let back = bf16_to_f32(b);
            assert!((back - x).abs() <= x.abs() * (1.0 / 128.0) + 1e-30, "{x} -> {back}");
        }
        // 1 + 2^-8 is exactly halfway between 1.0 and 1.0078125 in bf16: ties to even -> 1.0
        assert_eq!(bf16_to_f32(f32_to_bf16(1.0 + 1.0 / 256.0)), 1.0);
        // 1 + 3*2^-8 is halfway between 1.0078125 and 1.015625: ties to even -> 1.015625
        assert_eq!(bf16_to_f32(f32_to_bf16(1.0 + 3.0 / 256.0)), 1.015625);
        assert!(bf16_to_f32(f32_to_bf16(f32::NAN)).is_nan());
    }

    #[test]
    fn fast_exp_is_exact_at_bf16_precision() {
        let mut worst = 0f32;
        let mut x = -80.0f32;
        while x < 80.0 {
            let rel = ((fast_exp(x) - x.exp()) / x.exp()).abs();
            worst = worst.max(rel);
            x += 0.0137;
        }
        assert!(worst < 2e-5, "worst relative error {worst}");
        assert_eq!(fast_exp(0.0), 1.0);
        assert!(fast_exp(-100.0) < 1e-37);
    }

    #[cfg(feature = "xrt")]
    #[test]
    fn opening_a_missing_device_is_an_error_not_a_crash() {
        if npu_present() {
            return; // covered by the tagger's live test on an NPU box
        }
        assert!(matches!(Session::open(0), Err(Error::Device(_))));
    }
}
