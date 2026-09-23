<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# `iron-xrt` — replay IRON kernels through XRT from Rust

This crate is vendored from image-organizer's `crates/iron`, where
the Taggerine tagger runs on it. Only the package name, doc references and
the C++ formatting (this repo's `.clang-format`) differ. [`gaic`](../gaic) builds on it.

IRON is an ahead-of-time compiler. What it leaves behind for a kernel is an
`.xclbin` plus an instruction stream (`*.insts.bin`). This crate runs those
with no Python, through a small C shim over XRT's C++ API (`iron_xrt_shim.cpp`,
compiled by `build.rs` with no `cc` crate, so it builds offline). It provides
three types:

- **`Session`**: the NPU.
- **`Kernel`**: a resident hardware context, shared by every kernel loaded
  from the same xclbin and released when the last of them is dropped. Runs are cached per argument tuple, with a blocking
  `run` and an async `start`/`wait`.
- **`Buffer`**: a host-visible BO the host writes in place, with `sub`-buffer
  views.

`build.rs` links XRT statically from `$XRT_STATIC_ROOT` (default
`~/npu/xrt-static`) when that tree exists. Otherwise it links dynamically from
`$XRT_ROOT` (default `/opt/xilinx/xrt`). It publishes `DEP_IRONXRT_DYNLIST` /
`DEP_IRONXRT_LIBDIR` for dependents to re-emit; see `gaic/build.rs`. Building
also needs XRT's headers and `uuid/uuid.h`.

## Building kernels: `compile`

`iron_xrt::compile` builds the kernels too, still without Python. It runs
the two tools IRON's Python drives as subprocesses, with the flags IRON and
mlir-aie pass them:

- **`Toolchain::compile_kernel`**: compiles an AIE kernel (`aie_kernels/…/*.cc`)
  to an object with Peano's `clang++`, for `Arch::Aie2` (NPU1) or
  `Arch::Aie2p` (NPU2).
- **`Toolchain::compile_design`**: compiles a design's MLIR (what `design.py`
  generates) to the `.xclbin` + instruction stream that `load_kernel` takes,
  with the native `aiecc` binary. It links the design's `link_with` objects.

```rust
use iron_xrt::compile::{Arch, Design, KernelSource, Toolchain};

let tc = Toolchain::from_env()?; // $MLIR_AIE_INSTALL_DIR (+ $PEANO_INSTALL_DIR, $AIECC_PATH)
tc.compile_kernel(&KernelSource::new("aie_kernels/generic/mul.cc", Arch::Aie2p), "out/mul.o".as_ref())?;
tc.compile_design(&Design::new("mul.mlir", "out/work").link("out/mul.o"),
                  "out/mul.xclbin".as_ref(), "out/mul.insts.bin".as_ref())?;
```

aiecc packages the xclbin with XRT's `xclbinutil`, so that must be on its
`PATH`. mlir-aie's binaries are built for a generic Linux. On NixOS they run
in the IRON container instead: `Toolchain::with_launcher` prefixes every tool
invocation with a command, for example a script that runs
`podman run … --entrypoint "$1" <image> "${@:2}"` with the same paths mounted.

`examples/eltwise_mul.rs` goes from source to NPU. It builds `mul.cc` and an
`ElementwiseMul` design MLIR, runs the result, and checks it against the
CPU. Built this way, the object and the instruction stream are byte-identical
to IRON's. The xclbin differs only in its timestamps and UUIDs.

Adapted from [RLX](https://github.com/MIT-RLX/rlx)'s `rlx-xdna` `compile.rs`
(MIT OR Apache-2.0).
