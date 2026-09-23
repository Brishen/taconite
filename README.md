<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
SPDX-License-Identifier: Apache-2.0
-->

# taconite

Rust runtimes for models compiled with [IRON](https://github.com/amd/IRON)
on AMD XDNA NPUs (Ryzen AI).

IRON is an ahead-of-time compiler. An exporter builds every NPU kernel a
model needs (`.xclbin` + `*.insts.bin`), packs the weights, and writes a
*bundle*. The crates here load that bundle and replay the model on the NPU
from Rust.

## Crates

| Crate | What it does |
| --- | --- |
| [`taconite`](taconite) | The core: loads and runs IRON kernels (`Session` / `Kernel` / `Buffer` / `Run`), and builds them with Peano and `aiecc` (`compile`) |
| [`taconite-bundle`](taconite-bundle) | Reads the bundle format every IRON exporter writes (`manifest.txt`, `tensors.txt` / `tensors.bin`); std-only |
| [`taconite-sam3`](taconite-sam3) | SAM 3 text-prompted instance segmentation (`sam3` binary) |
| [`taconite-clip`](taconite-clip) | CLIP ViT-H/14 image and text embeddings, zero-shot classification (`clip` binary) |
| [`taconite-gaic`](taconite-gaic) | GAIC (VGG16) image cropping (`gaic` binary) |
| [`taconite-adaface`](taconite-adaface) | AdaFace IR-18 / IR-101 face embeddings, an embeddable library |

Each crate is its own package (there is no workspace); each one's README
covers its build, usage, accuracy and speed.

## Two ways to reach the NPU

- **XRT** (the default `xrt` feature): a small C shim over XRT's C++ API,
  compiled by `build.rs`. Building needs XRT's headers and library.
- **Direct** (`direct` feature): the same types over the `amdxdna` kernel
  driver's ioctls on `/dev/accel/accel0`. No XRT is needed to build or run,
  and the build has no C++ step. It gives the same results as XRT at the same
  speed.

`taconite`, `taconite-sam3`, `taconite-clip` and `taconite-gaic` support
both. For an XRT-free build of a model crate, use:

```bash
cargo build --release --no-default-features --features cli,direct
```

`taconite-adaface` runs through XRT only.

## Bundles

Bundles come from IRON's exporters (`iron/applications/<model>/export_*.py`).
Prebuilt NPU2 bundles are on Hugging Face:

- SAM 3: [`brishen/iron-sam3-npu2`](https://huggingface.co/brishen/iron-sam3-npu2)
- AdaFace: [`brishen/iron-adaface-ir18-npu2`](https://huggingface.co/brishen/iron-adaface-ir18-npu2),
  [`brishen/iron-adaface-ir101-npu2`](https://huggingface.co/brishen/iron-adaface-ir101-npu2)

## Requirements

Linux on x86-64 with an XDNA NPU and the `amdxdna` driver. The model crates
are tested on NPU2 (Strix).

## License

Apache-2.0. Some model crates also include ported code under MIT, MIT-CMU
or BSD-3-Clause. Each crate's `Cargo.toml` `license` field and its
`LICENSE-*` files give the exact terms.
