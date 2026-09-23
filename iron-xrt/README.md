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
