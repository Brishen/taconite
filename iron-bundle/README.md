<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# iron-bundle

Reads the bundles IRON's exporters write for its Rust runtimes. It is
std-only and doesn't use XRT; `iron/common/bundle.py` is the writer side.

```text
manifest.txt   one record a line: `<tag> <field>...`; `k=v` fields are also
               looked up by key; blank and `#` lines are skipped
tensors.txt    <name> <dtype> <d0,d1,...> <offset> <bytes>, one a line, into
tensors.bin    this blob (offsets 64-byte aligned; f32, bf16, u8, i32, LE)
kernels/       xclbins and instruction streams
```

Three records mean the same thing in every bundle, and `Manifest`
interprets them:

```text
version <n>                        format version; must match the runtime's
param <name> <value...>            a model constant
xclbin <key> <file> <kernel name>  a hardware context, referred to by key
```

Every other record belongs to the model. Its runtime reads those records from
`Manifest::records()` in file order, using the `Record` accessors
(`field`, `get::<T>("k")`, `flag`, `rest`). A `Record::error` message names the
manifest line.

```rust
let m = iron_bundle::Manifest::load(dir, MY_VERSION)?;
let store = iron_bundle::Store::load(dir)?;
for r in m.tagged("gemm") {
    let n: usize = r.get("N")?;
    let x = m.xclbin(r.str("ctx")?)?;
    let bias = store.f32(r.str("bias")?)?;
    // ...
}
```

Used by `iron/rust/sam3` and `iron/rust/adaface-ir-runtime`.
