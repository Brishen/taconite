// SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
// SPDX-License-Identifier: Apache-2.0

//! The iGPU (feature `gpu`): Vulkan compute through RADV, for the host
//! stages the NPU has no kernels for.
//!
//! A stage is recorded once into a [`Plan`] -- one command buffer of
//! compute dispatches over fixed buffers, a barrier between each -- and
//! replayed per forward with one submit; only its inputs change. Buffers
//! are views into a few large allocations ([`Arena`]), and an NPU buffer is
//! imported as a dma-buf ([`Vk::import`]), so the GPU reads what the NPU
//! wrote in place: no copy, and no CPU cache maintenance -- the GPU's
//! accesses to system memory snoop the CPU's caches, and a fence wait on
//! either side orders the two devices (measured in `taconite-gpu-spike`).
//!
//! Shaders are WGSL, compiled in process by naga, with their shape
//! constants substituted into the source (`{NAME}`) before compiling.
//!
//! The Vulkan layer ([`Vk`], [`Plan`], ...) is `taconite::gpu`'s.
//!
//! `SAM3_GPU_PROFILE=1` records a timestamp after every dispatch and
//! prints each run's time per op label; `SAM3_GLUE_BENCH=n` replays each
//! glue step `n` more times and prints its cold vs warm time.

pub mod decoder;
pub mod glue;
pub(crate) mod ops;

pub use taconite::gpu::{Arena, Plan, Recorder, View, Vk};

use crate::Error;

/// The iGPU's share of SAM3 in NPU+GPU mode: the device, the DETR
/// decoder's query layers, and the glue between the NPU's kernels.
pub struct Gpu {
    pub vk: Vk,
    pub dec: decoder::Decoder,
    pub glue: glue::Glue,
}

impl Gpu {
    /// Opens the GPU, imports the NPU buffers the stages share and records
    /// every plan.
    pub(crate) fn new(
        store: &crate::bundle::Store,
        cfg: &crate::Config,
        npu: &mut crate::npu::Npu,
        io: &crate::Ios,
    ) -> Result<Self, Error> {
        let mut vk = Vk::new()?;
        let mut b = ops::Builder::new(store, vk.arena(128 << 20)?, vk.arena(48 << 20)?);
        let kvw = npu.spec("dec_kv")?.n;
        let kvv = vk.import_npu(&io.dec_kv.c)?;
        let dec = decoder::Decoder::new(&mut vk, &mut b, cfg, kvv, kvw)?;
        let mut glue = glue::Glue::new(&mut vk, &mut b, cfg, npu, io)?;
        // The ViT's fc1 GEMM applies GELU in an NPU epilogue that costs
        // more than the GEMM; switch it off (its RTP mode words, 5 ->
        // 0) and apply GELU on the GPU in place between fc1 and fc2.
        // SAM3_VIT_GELU=npu keeps the NPU's.
        if cfg.vit_device && std::env::var("SAM3_VIT_GELU").map_or(true, |v| v != "npu") {
            if let Some(words) = glue::fc1_epilogue_off(&npu.insts("v_fc1")?) {
                glue.vit_gelu = Some(glue::vit_gelu_plan(&mut vk, &io.v_fc1, npu.spec("v_fc1")?)?);
                npu.patch_insts("v_fc1", words)?;
            }
        }
        Ok(Gpu { vk, dec, glue })
    }

    /// Runs `plan` and accounts its time under `gpu:<key>`.
    pub(crate) fn run(&self, plan: &Plan, key: &str, timing: &mut crate::Timing) -> Result<(), Error> {
        let dt = plan.run(&self.vk)?;
        timing.add(&format!("gpu:{key}"), dt);
        if let Some(n) = std::env::var("SAM3_GLUE_BENCH").ok().and_then(|v| v.parse::<usize>().ok()) {
            let mut ts = (0..n).map(|_| plan.run(&self.vk)).collect::<Result<Vec<_>, _>>()?;
            ts.sort();
            eprintln!(
                "glue {key}: cold {:.2} ms, warm min {:.2} ms",
                dt.as_secs_f64() * 1e3,
                ts[0].as_secs_f64() * 1e3
            );
        }
        Ok(())
    }
}
