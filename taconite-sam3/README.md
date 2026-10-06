<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
SPDX-License-Identifier: Apache-2.0
-->

# `taconite-sam3` — SAM3 segmentation on the NPU, from Rust

A Rust runtime for [SAM 3](https://huggingface.co/facebook/sam3):
give it an image and a text prompt ("cat", "person", "laptop") and it
returns every matching instance's score, box and mask; or clicks and / or
a box, and it returns the object's mask ([below](#point-and-box-prompts)). The heavy image
side runs on the AMD XDNA NPU (NPU2) through IRON kernels replayed with
[`taconite`](https://crates.io/crates/taconite); everything else runs here
in plain `std` Rust. It is the forward of [IRON](https://github.com/amd/IRON)'s
Python app (`iron/applications/sam3`), stage for stage.

A prebuilt NPU2 bundle is on Hugging Face:
[`brishen/iron-sam3-npu2`](https://huggingface.co/brishen/iron-sam3-npu2)
(`hf download brishen/iron-sam3-npu2 --local-dir sam3`). It is bundle
format 2, which this version (0.1.1) reads along with format 1; 0.1.0
reads format 1 only (`--revision bundle-v1`).

```bash
sam3 segment <bundle> photo.jpg "person" -o overlay.png
#   "person": 2 instance(s)
#     score 0.983  box [387.0, 69.5, 498.9, 346.2]  mask 17592 px
#     score 0.913  box [0.2, 263.9, 60.4, 299.7]  mask 1099 px
sam3 points <bundle> photo.jpg "440,160" -o overlay.png     # a click (x,y in pixels)
#   object score 17.69 (object)
#     mask 0: predicted IoU 0.026, 636 px
#     mask 1: predicted IoU 0.788, 17166 px  <- best
#     mask 2: predicted IoU 0.559, 5178 px
sam3 check <bundle>          # every stage against the float32 reference
```

## Point and box prompts

Bundles exported with the point path (`param tracker 1`; IRON's
`export_sam3.py` since the tracker head) also segment the object at a
click, several clicks (label 0 = background), a box, or a box and clicks
-- SAM's interactive segmentation, through SAM3's tracker head:

```rust
let mut sam = Sam3::load(bundle)?;
let emb = sam.embed_image(&preprocess(&rgb, w, h, sam.cfg.image_size))?; // NPU, ~2.2 s
let prompt = PointPrompt::parse("450,200;120,200,0", None)?;           // or bbox: Some("x1,y1,x2,y2")
let out = sam.predict_points(&emb, &prompt, w, h, true)?;               // host, ~60 ms, &self
let mask = point_mask(&out, out.best(), w, h);                          // [h * w] bool
```

| stage | NPU | host (this crate) |
|---|---|---|
| ViT backbone | as above | |
| the tracker's FPN neck | the text neck's kernels with the tracker's weights; the decoder's `conv_s0` / `conv_s1` folded into its 3x3s, `no_memory_embedding` into a bias | GELU, pixel shuffles (`neck.rs`) |
| prompt encoder, two-way transformer, upscaling, masks | | all of it, per prompt (`tracker.rs`) |

`embed_image` runs once per image; `predict_points` takes `&self`, so any
number of prompts on the same embedding never touch the NPU. It returns
SAM's three candidate masks with their predicted IoUs (or one, with
`multimask = false`: SAM 2's stability-based pick) and an object score.
`sam3 check` adds, on such bundles:

```
point prompts, each stage on the float32 reference's inputs:
  [ok] tracker neck level 0 / 1 / 2: cosine 0.999967 / 0.999983 / 0.999975
  [ok] prompt encoder: max |diff| 5.96e-8
  [ok] mask decoder, multimask: masks cosine 1.000000, max |IoU diff| 1.61e-6, |object diff| 1.91e-6 (68 ms)
  [ok] mask decoder, singlemask: masks cosine 1.000000, max |IoU diff| 1.19e-7, |object diff| 1.91e-6 (57 ms)
end to end (image file + points / box -> masks) against the reference cases:
  [ok] cats.jpg / points=450,200;120,200,0 box=: mask IoU vs ref [0.9854, 0.9995, 0.9995], max |IoU score diff| 0.008
  [ok] kitchen.jpg / points=440,160 box=: mask IoU vs ref [0.9922, 0.998, 0.9975], max |IoU score diff| 0.011
  [ok] cat_laptop.jpg / points= box=0,0,290,425: mask IoU vs ref [0.8932, 0.9997, 0.9989], max |IoU score diff| 0.001
  [ok] cats.jpg / points=200,150 box=10,60,320,475: mask IoU vs ref [0.9879, 0.999, 0.9995], max |IoU score diff| 0.000
```

The best-rated candidate and any rated >= 0.85 must reach mask IoU 0.98;
a low-rated one (the laptop's first, rated 0.82) has an ambiguous edge
that moves with JPEG decoding and the NPU's rounding, and is held to 0.85.
Mask prompts (SAM's refine-from-a-previous-mask input) are not supported.

## What runs where

| stage | NPU | host (this crate) |
|---|---|---|
| preprocessing | | resize to 1008 x 1008 (torch's antialiased uint8 bilinear, bit-exact), normalise (`post.rs`) |
| CLIP tokenizer + text encoder (24 layers) | | byte-level BPE (`tokenizer.rs`); the encoder over the valid tokens only (`text.rs`) |
| ViT backbone (32 layers, 5184 tokens) | everything: patch embedding, every Linear, RoPE, windowed + global attention, GELU, residual adds + LayerNorms — each kernel reading the last one's output in place | the first LayerNorm and the readback, once (`vit.rs`) |
| FPN neck | ConvTs, 1x1s and 3x3 convs as GEMMs | GELU, pixel shuffles (`neck.rs`) |
| DETR encoder (6 layers) | projections, self-attention, the prompt cross-attention folded into two GEMMs, the MLP | LayerNorms, the softmax over the prompt, folding + packing its weights per prompt (`detr.rs`) |
| DETR decoder (6 layers, 200 queries) | every layer's vision keys and values, in one GEMM | the query layers, box relative-position bias, box refinement, presence (`detr.rs`), scoring |
| mask decoder | prompt cross-attention, the pixel decoder's 3x3s, the mask + semantic head folded into one GEMM | GroupNorms, upsampling, folding + packing the head per forward (`mask.rs`) |
| post-processing | | score gating, box scaling, mask upsampling (`post.rs`) |

That is the default `npu` mode. In `npu+gpu` mode (below), most of the
host column runs on the iGPU instead.

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
  [ok] ViT backbone: cosine 0.992325
  [ok] neck level 0 / 1 / 2: cosine 0.999969 / 0.999984 / 0.999980
  [ok] DETR encoder: cosine 0.999722
  [ok] DETR decoder hidden: cosine 0.999831
  [ok] DETR decoder boxes: max |diff| 0.00008 over 3 confident queries
  [ok] DETR decoder logits: max |diff| 0.0054 over 3 confident queries
  [ok] presence: 5.3509 vs 5.3441
  [ok] mask decoder masks: cosine 0.999991
end to end (image file + prompt -> instances) against the reference cases:
  [ok] cats.jpg / "cat": 2 instances (ref 2), min mask IoU 0.9992, max |score diff| 0.002
  [ok] car.png / "car": 1 instances (ref 1), min mask IoU 0.9996, max |score diff| 0.006
  [ok] cat_laptop.jpg / "laptop": 1 instances (ref 1), min mask IoU 0.9996, max |score diff| 0.001
  [ok] cat_laptop.jpg / "person": 0 instances (ref 0)
  [ok] kitchen.jpg / "person": 2 instances (ref 2), min mask IoU 0.9989, max |score diff| 0.001
ALL CHECKS PASSED
```

JPEG decoding (the `image` crate against PIL's libjpeg) differs by a
level here and there; it shows up as the cases' small score differences,
not in which instances are found.

(The ViT's 0.9923 is with the backbone numerics of
[`iron/applications/sam3`](../../applications/sam3/README.md#numerics):
fp32 residual stream, exact softmax scale and exp2, bfp16 hi + lo
activations. Bundles exported before them -- no `vit_res_f32` param --
still run, at their 0.9836.)

## Performance

Warm, one image + prompt, image file to instances, Ryzen AI 9 HX 370
(quiet machine), before the backbone numerics above, which add ~0.4 s of
NPU time: **~2.5–2.6 s**, of which ~1.8 s is NPU execution:

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

# 2. The runtime (links XRT; see below for the XRT-free build)
cargo install taconite-sam3

# 3. Run
sam3 check bundle
sam3 segment bundle photo.jpg "red car" -o overlay.png [--threshold 0.5] [--reps 3] [--threads 16]
sam3 points bundle photo.jpg "x,y[,label];..." [--box x1,y1,x2,y2] [--single] -o overlay.png
```

Step 1 is IRON's exporter; the prebuilt bundle above replaces it.

The bundle is ~1.5 GB: the NPU weights pre-packed (bfp16), the text
encoder's in bf16, everything else f32, plus the reference tensors.

To run without XRT, build with `--no-default-features --features cli,direct`.
The kernels then go through the `amdxdna` driver's ioctls
(`taconite::direct`); the build needs no XRT headers or library, and the
binary links none. On NPU2, `check`
prints the same numbers as the XRT build, at the same speed (~2.6 s a case).

## Modes: `npu` and `npu+gpu`

`--mode` (or `SAM3_MODE`; in code, `Sam3::load_mode`) picks where the work
between the NPU's kernels runs:

- **`npu`** (the default): on the CPU, as described above. In this mode the
  GPU is never touched, even in a build with the `gpu` feature.
- **`npu+gpu`** (needs `--features gpu`, which implies `direct`): on the
  Radeon iGPU through Vulkan (RADV). Every NPU buffer the stages share is
  exported as a dma-buf and imported into Vulkan once, at load. Each step
  between two NPU kernels is a plan recorded at load. It reads the kernel's
  output where the NPU wrote it and writes the next kernel's input in place.
  The activations in between stay in GPU memory: the FPN levels, the DETR
  encoder's residual stream and the pixel decoder's maps.

| stage | on the GPU in `npu+gpu` |
|---|---|
| ViT | the MLPs' GELU, in place between fc1 and fc2 (below) |
| neck | the ViT's output into `n_in` (read from the NPU, no readback); GELU + 2x2 shuffles; each 3x3 conv's overlapping input, built straight from the GEMM outputs; bias -> FPN levels |
| DETR encoder | LayerNorms (+ position) into the GEMMs' inputs, the q/k/v split into the MHA's buffers, residual adds, the prompt softmax |
| DETR decoder | its keys/values GEMM's input, and all six query layers (222 dispatches in f32) |
| mask decoder | the prompt cross-attention's glue, upsample + skip into the convs' inputs, bias + GroupNorm + ReLU, the mask head's input |

**The ViT's GELU.** Every ViT layer's fc1 GEMM ends in a GELU epilogue on
the NPU (`flm.GEMM`'s mode 5, a runtime parameter in its instruction
stream). The epilogue took 61% of fc1's time: fc1 ran 5x slower than fc2
for the same FLOPs. In `npu+gpu` mode the runtime rewrites those 32 mode
words to 0 (a plain GEMM), after checking the stream has the expected
shape; the patch is re-applied whenever the kernel reloads. A GPU pass
then applies tanh GELU to fc1's output in place before fc2 reads it.

- The ViT is slightly more accurate this way (`check`: cosine 0.99248, up
  from 0.99233), since the GELU is f32 on the GPU instead of bf16 on the NPU.
- `SAM3_VIT_GELU=npu` keeps the NPU's epilogue.

**Overlap with the CPU.** The CPU work left is also moved off the critical
path:

- The per-prompt folding + bfp16 packing of the cross-attention weights
  runs on the text thread while the NPU runs the ViT.
- The mask head's weights (an MLP over the queries, folded and packed) are
  built on a side thread while the GPU and NPU run the pixel decoder.
- The final `[pixels, queries] -> [queries, pixels]` mask transpose stays
  on the CPU.

The GPU kernels mirror the CPU code's arithmetic: bf16 rounding,
`fast_exp`, the erf GELU, and the softmax's summation order.

It needs a Vulkan loader at run time. On NixOS that is `vulkan-loader` on
`LD_LIBRARY_PATH`; the repo's `nix-shell` sets it. Asking for `npu+gpu`
without a usable GPU is an error rather than a silent fallback. `sam3`
prints the mode and devices it loaded:

```
loaded bundle (1.5 s); mode npu+gpu on NPU + GPU (AMD Radeon 890M Graphics (RADV STRIX1))
```

**Accuracy.** `check` passes in both modes. In `npu+gpu` it adds a stage
comparing the GPU decoder with the CPU one on the same inputs:

```
[ok] DETR decoder, GPU vs CPU: max |diff| hidden 2.04e-4, boxes 2.93e-5, logits 5.15e-5, presence 1.43e-6
```

**Speed.** Warm forwards, with the two modes interleaved, in ms (median
of 9 each; Ryzen AI 9 HX 370, iGPU idle otherwise, no NPU context
evictions):

| stage | `npu` | `npu+gpu` |
|---|---:|---:|
| ViT (NPU) | 2615 | 2611 |
| neck | 160 | 59 |
| DETR encoder | 388 | 334 |
| DETR decoder | 119 | 36 |
| mask decoder | 227 | 122 |
| **forward** | **3544** | **3176** |

- **The whole forward** is ~370 ms (10%) faster.
- **Everything after the backbone** drops from ~930 ms to ~565 ms.
- **The DETR encoder** is mostly its NPU MHA, so it gains the least.

That table predates the two changes above. Their measured effect, `npu+gpu`
before and after, on the same busy machine:

| ms | before | GELU on the GPU |
|---|---:|---:|
| fc1 (NPU, 32 layers) | 927 | 362 |
| GELU (GPU, 32 passes) | — | 67 |
| ViT | 2265 | 1776 |
| **forward** | **2647–2686** | **2165–2173** |

- **The GELU move saves ~490 ms per image (−18%).**
- **The CPU overlap is worth a further ~45 ms** (mostly the DETR encoder
  and the mask decoder).
- **With another app loading the iGPU**, the gains shrink or vanish: the
  GPU steps queue behind it. That is one reason `npu` stays the default.

**Diagnostics.**

- `SAM3_GPU_PROFILE=1` prints the GPU time per op of every plan run, and
  each pipeline's compile time.
- `SAM3_GPU_BENCH=n` replays the decoder `n` more times per forward.
- `SAM3_GLUE_BENCH=n` replays each glue step `n` more times and prints its
  cold and warm times.
- `SAM3_VIT_GELU=npu` keeps the ViT's GELU on the NPU (for comparison).

The buffer sharing behind it is measured in
[`taconite-gpu-spike`](../taconite-gpu-spike): the GPU's accesses snoop the
CPU's caches and a fence wait orders the devices, so no cache maintenance
is needed between the GPU and the NPU.

## Limits

- Image + text prompts, and point / box prompts through the tracker head
  (above). Box prompts to the text path (its geometry encoder), mask
  prompts and the video / tracking model are not ported.
- Prompts are NFC-normalised by HF's tokenizer; this one assumes composed
  input (std has no Unicode normalisation).

## License

Apache-2.0, except `src/post.rs`, whose resampler ports torch's
antialiased uint8 resize (BSD-3-Clause, `LICENSE-PYTORCH`), itself after
Pillow's (MIT-CMU, `LICENSE-PILLOW`). `src/pack.rs` ports IRON's
`flm.GEMM` packer (Apache-2.0, AMD).
