<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
SPDX-License-Identifier: Apache-2.0
-->

# `taconite-embeddinggemma2` — EmbeddingGemma 2 image and text embeddings on the NPU, from Rust

A Rust runtime for [`google/embeddinggemma-2`](https://huggingface.co/google/embeddinggemma-2)'s
image and text embeddings: an image or a text in, the L2-normalized 768-d
embedding sentence-transformers computes for it out, with the model's
transformers on the AMD XDNA NPU (NPU2) through IRON kernels replayed with
[`taconite`](https://crates.io/crates/taconite). It is the forward of
[IRON](https://github.com/amd/IRON)'s Python app
(`iron/applications/embeddinggemma2`), kernel for kernel. Images and texts
share one space: a `SearchQuery` text embedding retrieves images and
`Document` texts by cosine.

A prebuilt NPU2 bundle is on Hugging Face:
[`brishen/iron-embeddinggemma2-npu2`](https://huggingface.co/brishen/iron-embeddinggemma2-npu2)
(`hf download brishen/iron-embeddinggemma2-npu2 --local-dir embeddinggemma2`).

```bash
cargo install taconite-embeddinggemma2   # over XRT
# or, with no XRT at all (straight to the amdxdna driver):
cargo install taconite-embeddinggemma2 --no-default-features --features cli,direct
```

```bash
embeddinggemma2 embed <bundle> cats.jpg car.png --text "two cats sleeping" --prompt SearchQuery
#   cats.jpg: 1200 ms; npu 620 ms (mha 255), ...
#   "two cats sleeping": 12 tokens, 150 ms; ...
#   cosine similarity: ...
embeddinggemma2 prompts <bundle> # the task prompts (SearchQuery, Document, Classification, ...)
embeddinggemma2 check <bundle>   # every stage against the float32 references
```

```rust
let mut m = taconite_embeddinggemma2::EmbeddingGemma2::load(Path::new("bundle"))?;
let img = m.embed_rgb(&rgb, width, height)?;                        // [768], unit length
let query = m.embed_text("two cats sleeping", Some("SearchQuery"))?; // the same space
let doc = m.embed_text("title: none | text: ...", None)?;           // or Some("Document")
let score = taconite_embeddinggemma2::cosine(&query, &img);
let e256 = taconite_embeddinggemma2::truncate(&img, 256);           // Matryoshka, re-normalized
```

The bundle comes from `iron/applications/embeddinggemma2/export_eg2.py`
(~770 MB: three xclbins, their instruction streams, the packed weights).

## What runs where

| | NPU | host (this crate) |
|---|---|---|
| vision tower (16 layers, 768 wide, up to 2520 patches) | every projection as an `flm.GEMM` (patch embedding, qkv, o, the GeGLU gate+up as one `geglu_pair` GEMM, down as four K slices read in place, embed_vision) and the bidirectional attention (the MHA operator, 12 heads of 64, one dispatch an image, a 128-row length bucket's stream with its key count rewritten) | resize + patchify (`preprocess.rs`), position embeddings, RMSNorms, q / k / v norms, 2D RoPE, residual adds, 3 x 3 pooling |
| text encoder (24 layers, 512 wide; an image's ~284 tokens or a text's up to 2048) | every projection: the per-layer-input projection, qkv, o, GeGLU, down, the PLE gate (tanh-GELU epilogue) and projection | Gemma's tokenizer (`tokenizer.rs`), the token embeddings, RMSNorms, RoPE, attention (head dims 256 / 512, a 512-token window on 20 of the 24 layers), the per-layer gating, mean pooling, the 512 -> 768 projection |

Three hardware contexts (NPU2 has 16, shared by every process). A kernel
loads on first use; when the device has no free slot for a context, the
others' kernels are dropped and reloaded when needed, which keeps it
working (~33 swaps an image) beside programs holding most of the slots.

- **Preprocessing is HF's `Gemma4ImageProcessor`, bit for bit**: the
  aspect-ratio-preserving size (sides multiples of 48, at most 280 x 9
  patches), torchvision's antialiased bicubic resize of the uint8 image
  (its fixed-point two-pass resampler), x / 255.
- **Tokenization is HF's, id for id**: Gemma's BPE (262144 tokens, byte
  fallback) with its added tokens, from the bundle's vocabulary and merge
  tables, and sentence-transformers' prompts and `<bos>` / `<eos>`
  handling. Texts are limited to 2048 tokens (sentence-transformers has
  no limit; the host attention grows with the square of the length).
- Every RMSNorm feeding a projection has its weight folded into the packed
  weights; GEMM operands are bfp16 hi + lo pairs (`split_a`, `split_b`).

## Validation

`embeddinggemma2 check` against HF transformers + sentence-transformers
6.1 in float32 on the ROCm iGPU:

- four reference images: pixels and patch positions identical; embedding
  cosine 0.99977-0.99993; each image ranks the same of five text queries
  first, image x text cosines within 0.0014. ~1.1-1.2 s an image on a
  Ryzen AI 9 HX 370 (NPU ~0.6 s, of which the vision attention ~0.25 s).
- 18 reference texts (every prompt kind, several scripts, emoji, code,
  whitespace runs, literal special tokens, one of 1421 tokens): ids equal
  to HF's for all 18, embedding cosine 0.99998-0.999999. ~55-70 ms a short
  text, 1.7 s for the 1421-token one.

Text needs a bundle with the text path (0.2.0's; the 0.1.0 bundle is
image-only and still works for images).

## Features

`xrt` (default) runs the kernels through XRT; `direct` through the amdxdna
driver's ioctls with no XRT (`--no-default-features --features
cli,direct`). `cli` builds the binary (the `image` crate decodes files);
the library takes RGB8 pixels.

## License

Apache-2.0. `preprocess.rs`'s resampler ports torch's antialiased uint8
resize (BSD-3-Clause, `LICENSE-PYTORCH`), which follows Pillow's
(MIT-CMU, `LICENSE-PILLOW`). The model's weights are Google's, under
Apache-2.0.
