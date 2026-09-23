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

## Running without XRT: `direct`

`iron_xrt::direct` runs the same kernels with no XRT and no C++. It is
Linux/x86-64 only, and it talks to the `amdxdna` driver's ioctls on
`/dev/accel/accel0` itself. It has the same shape as the XRT API:

- **`Device`**: the accel node, plus the process's 64 MiB device heap.
- **`Kernel`**: a hardware context of its own, loaded from an xclbin and an
  instruction stream.
- **`Buffer`**: shared buffers the host fills in place. `run` takes up to
  five of them, the `MLIR_AIE` kernel's `bo0`…`bo4`.

It issues the same ioctl sequence XRT does. The sequence and the command
packet were captured by tracing this crate's XRT path, and `direct` reproduces
it, so each run is one `EXEC_CMD` plus one fence wait. `Device` can also
query the firmware version and the hardware contexts (with their command
counters and fault state), and set the NPU power mode (this needs root).

On an NPU2 it computes the same results as XRT at the same speed. For
example, `examples/direct_eltwise_mul.rs` multiplies 1,638,400 bf16 values
with a median run of 235 µs through `direct` and 232 µs through XRT, and 0
mismatches either way. `tests/direct_hw.rs` (`--ignored`, needs an NPU) loads,
runs and releases 60 contexts. It does not cover sub-buffers, async launch, or
xclbins with more than one PDI.

Adapted from [RLX](https://github.com/MIT-RLX/rlx)'s `rlx-xdna` `direct.rs`
(MIT OR Apache-2.0). RLX never saw a command complete on NPU1. The likely
cause is its fence wait's timeout: the DRM syncobj ioctls read the timeout as
an absolute deadline, and RLX passed a relative one. That makes the wait
return at once with the command still `NEW`, and it reproduces here.
