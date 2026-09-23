<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
SPDX-License-Identifier: Apache-2.0
-->

# `taconite-sam3` — SAM3 text-prompted segmentation on the NPU, from Rust

A zero-Python runtime for [SAM 3](https://huggingface.co/facebook/sam3):
give it an image and a text prompt ("cat", "person", "laptop") and it
returns every matching instance's score, box and mask. The heavy image
side runs on the AMD XDNA NPU (NPU2) through IRON kernels replayed with
[`taconite`](https://crates.io/crates/taconite); everything else runs here
in plain `std` Rust. It is the forward of [IRON](https://github.com/amd/IRON)'s
Python app (`iron/applications/sam3`), stage for stage.

A prebuilt NPU2 bundle is on Hugging Face:
[`brishen/iron-sam3-npu2`](https://huggingface.co/brishen/iron-sam3-npu2)
(`hf download brishen/iron-sam3-npu2 --local-dir sam3`).

```bash
sam3 segment <bundle> photo.jpg "person" -o overlay.png
#   "person": 2 instance(s)
#     score 0.983  box [387.0, 69.5, 498.9, 346.2]  mask 17592 px
#     score 0.913  box [0.2, 263.9, 60.4, 299.7]  mask 1099 px
sam3 check <bundle>          # every stage against the float32 reference
```

## What runs where

| stage | NPU | host (this crate) |
|---|---|---|
| preprocessing | | resize to 1008 x 1008 (torch's antialiased uint8 bilinear, bit-exact), normalise (`post.rs`) |
| CLIP tokenizer + text encoder (24 layers) | | byte-level BPE (`tokenizer.rs`); the encoder over the valid tokens only (`text.rs`) |
| ViT backbone (32 layers, 5184 tokens) | everything: patch embedding, every Linear, RoPE, windowed + global attention, GELU, residual adds + LayerNorms — each kernel reading the last one's output in place | the first LayerNorm and the readback, once (`vit.rs`) |
| FPN neck | ConvTs, 1x1s and 3x3 convs as GEMMs | GELU, pixel shuffles (`neck.rs`) |
| DETR encoder (6 layers) | projections, self-attention, the prompt cross-attention folded into two GEMMs, the MLP | LayerNorms, the softmax over the prompt, folding + packing its weights per prompt (`detr.rs`) |
| DETR decoder (6 layers, 200 queries) | every layer's vision keys and values, in one GEMM | the query layers, box relative-position bias, box refinement, presence, scoring (`detr.rs`) |
| mask decoder | prompt cross-attention, the pixel decoder's 3x3s, the mask + semantic head folded into one GEMM | GroupNorms, upsampling, folding + packing the head per forward (`mask.rs`) |
| post-processing | | score gating, box scaling, mask upsampling (`post.rs`) |

21 kernels on 15 hardware contexts (NPU2 has 16). Weights the runtime builds per prompt
(the folded cross-attentions, the mask head) are packed here into
`flm.GEMM`'s bfp16ebs8 layout (`pack.rs`, byte-identical to the Python
packer).

## Validation

`sam3 check <bundle>` tests, against the bundle's float32 CPU reference
(`transformers`), each stage on the reference's own inputs, then the whole
pipeline from image files on the bundle's cases (references from the ROCm
GPU run, `sam3_reference.py`). On NPU2:

```
host pieces:
  [ok] bfp16 packing: 0 of 82944 bytes differ
  [ok] tokenizer: 6/6 prompts identical
  [ok] resize + normalise: 0 of 3048192 values differ from the processor's
  [info] image decoding: 12593 of 921600 bytes differ from PIL's; after resize max 3.0 levels
stages, each on the float32 reference's inputs:
  [ok] text encoder: cosine 1.000000
  [ok] ViT backbone: cosine 0.983623
  [ok] neck level 0 / 1 / 2: cosine 0.999961 / 0.999953 / 0.999968
  [ok] DETR encoder: cosine 0.999722
  [ok] DETR decoder hidden: cosine 0.999831
  [ok] DETR decoder boxes: max |diff| 0.00008 over 3 confident queries
  [ok] DETR decoder logits: max |diff| 0.0054 over 3 confident queries
  [ok] presence: 5.3509 vs 5.3441
  [ok] mask decoder masks: cosine 0.999991
end to end (image file + prompt -> instances) against the reference cases:
  [ok] cats.jpg / "cat": 2 instances (ref 2), min mask IoU 0.9989, max |score diff| 0.002
  [ok] car.png / "car": 1 instances (ref 1), min mask IoU 0.9994, max |score diff| 0.007
  [ok] cat_laptop.jpg / "laptop": 1 instances (ref 1), min mask IoU 0.9993, max |score diff| 0.000
  [ok] cat_laptop.jpg / "person": 0 instances (ref 0)
  [ok] kitchen.jpg / "person": 2 instances (ref 2), min mask IoU 0.9982, max |score diff| 0.003
ALL CHECKS PASSED
```

JPEG decoding (the `image` crate against PIL's libjpeg) differs by a
level here and there; it shows up as the cases' small score differences,
not in which instances are found.

## Performance

Warm, one image + prompt, image file to instances, Ryzen AI 9 HX 370
(quiet machine): **~2.5–2.6 s**, of which ~1.8 s is NPU execution:

```
text 72  vit 1493 (prologue 13, readback 3; the rest NPU)  neck 101
detr_enc 297  detr_dec ~400  mask_dec ~170  ms
```

The ViT's layers run entirely on the device (bundles with `vit_device`):
RoPE, the residual adds and the LayerNorms are NPU kernels too, and each
kernel reads the previous one's output buffer in place, so a layer never
touches the host. The ViT is now NPU-bound; the CPU's share is the
DETR decoder's query layers (~0.4 s) and the text encoder. The bundle's
15 contexts stay resident when they fit, and when another process holds
some of NPU2's 16 the runtime evicts its least recently used context to
load the next (`SAM3_MAX_CONTEXTS=n` caps it).

Host buffers of the NPU are touched only through bulk parallel copies
(`npu::push` / `pull`); stages compute in ordinary memory.

## Build and run

```bash
# 1. The bundle (Python, once): compiles every kernel, packs the weights,
#    captures the float32 reference and the test cases
python -m iron.applications.sam3.export_sam3 --out bundle --model /path/to/sam3 \
    --image cats.jpg --prompt cat --case cats.jpg:cat:ref_cats.npz ...

# 2. The runtime (links XRT; see the taconite crate)
cargo install taconite-sam3

# 3. Run
sam3 check bundle
sam3 segment bundle photo.jpg "red car" -o overlay.png [--threshold 0.5] [--reps 3] [--threads 16]
```

Step 1 is IRON's exporter; the prebuilt bundle above replaces it.

The bundle is ~1.5 GB: the NPU weights pre-packed (bfp16), the text
encoder's in bf16, everything else f32, plus the reference tensors.

To run without XRT, build with `--no-default-features --features cli,direct`.
The kernels then go through the `amdxdna` driver's ioctls
(`taconite::direct`); the build needs no XRT headers or library, and the
binary links none. On NPU2, `check`
prints the same numbers as the XRT build, at the same speed (~2.6 s a case).

## Limits

- Image + text prompts. Box prompts (the geometry encoder) and the video /
  tracking model are not ported.
- Prompts are NFC-normalised by HF's tokenizer; this one assumes composed
  input (std has no Unicode normalisation).

## License

Apache-2.0, except `src/post.rs`, whose resampler ports torch's
antialiased uint8 resize (BSD-3-Clause, `LICENSE-PYTORCH`), itself after
Pillow's (MIT-CMU, `LICENSE-PILLOW`). `src/pack.rs` ports IRON's
`flm.GEMM` packer (Apache-2.0, AMD).
