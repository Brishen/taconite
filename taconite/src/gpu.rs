// SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
// SPDX-License-Identifier: Apache-2.0

//! The iGPU (feature `gpu`): Vulkan compute through RADV, for the host
//! work between a model's NPU kernels.
//!
//! A stage is recorded once into a [`Plan`] -- one command buffer of
//! compute dispatches over fixed buffers, a barrier between each -- and
//! replayed per forward with one submit; only its inputs change. Buffers
//! are views into a few large allocations ([`Arena`]), and an NPU buffer is
//! imported as a dma-buf ([`Vk::import_npu`]), so the GPU reads what the NPU
//! wrote in place: no copy, and no CPU cache maintenance -- the GPU's
//! accesses to system memory snoop the CPU's caches, and a fence wait on
//! either side orders the two devices (measured in `taconite-gpu-spike`).
//!
//! Shaders are WGSL, compiled in process by naga, with their shape
//! constants substituted into the source (`{NAME}`) before compiling.
//!
//! `TACONITE_GPU_PROFILE=1` (or `SAM3_GPU_PROFILE=1`) records a timestamp
//! after every dispatch and prints each run's time per op label.

use std::collections::HashMap;
use std::ffi::CStr;
use std::os::fd::{AsRawFd, IntoRawFd, OwnedFd};
use std::time::{Duration, Instant};

use ash::vk;

use crate::Error;

fn vkerr(what: &'static str) -> impl Fn(vk::Result) -> Error {
    move |e| Error::Gpu(format!("{what}: {e}"))
}

/// `TACONITE_GPU_PROFILE=1` (or sam3's `SAM3_GPU_PROFILE=1`).
fn profiling() -> bool {
    ["TACONITE_GPU_PROFILE", "SAM3_GPU_PROFILE"].iter().any(|k| std::env::var(k).is_ok_and(|v| v == "1"))
}

const DMA_BUF: vk::ExternalMemoryHandleTypeFlags = vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT;
/// View offsets: a multiple of every device's `minStorageBufferOffsetAlignment`.
const ALIGN: u64 = 256;

/// The GPU: one compute queue, its command pool and descriptor pool, and
/// the pipelines compiled so far.
pub struct Vk {
    _entry: ash::Entry,
    _instance: ash::Instance,
    device: ash::Device,
    queue: vk::Queue,
    pool: vk::CommandPool,
    desc_pool: vk::DescriptorPool,
    mem_props: vk::PhysicalDeviceMemoryProperties,
    ext_fd: ash::khr::external_memory_fd::Device,
    pipes: HashMap<String, Pipe>,
    /// NPU buffers imported so far, by host address
    imports: HashMap<usize, View>,
    pub name: String,
    /// nanoseconds per timestamp tick
    ts_period: f32,
    /// cooperative matrices (`enable wgpu_cooperative_matrix;`) of 16 x 16
    /// x 16 f16 x f16 + f32 on 64-wide subgroups: `VK_KHR_cooperative_matrix`
    /// with f16 arithmetic and storage, the Vulkan memory model and a
    /// required subgroup size of 64 all available, and enabled
    pub coop_f16: bool,
}

struct Pipe {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    dsl: vk::DescriptorSetLayout,
}

/// A range of a buffer: what a shader binding sees. `ptr` is its host
/// address when the memory is mapped.
#[derive(Clone, Copy)]
pub struct View {
    buffer: vk::Buffer,
    off: u64,
    pub len: u64,
    ptr: *mut u8,
}

// SAFETY: `ptr` is a mapping of device memory, valid from any thread for as
// long as the memory lives; the views are used one thread at a time (the
// model owning them is `Send`, not `Sync`), as `taconite`'s buffers are.
unsafe impl Send for View {}

impl View {
    /// The first `bytes` bytes of the view.
    pub fn head(&self, bytes: u64) -> View {
        assert!(bytes <= self.len);
        View { len: bytes, ..*self }
    }

    /// `bytes` bytes from `off` (a multiple of 256).
    pub fn at(&self, off: u64, bytes: u64) -> View {
        assert!(off % ALIGN == 0 && off + bytes <= self.len, "view [{off}, +{bytes}) of {}", self.len);
        View { buffer: self.buffer, off: self.off + off, len: bytes, ptr: self.ptr.wrapping_add(off as usize) }
    }

    /// Writes `data` at the start (host-visible memory only).
    pub fn write<T: Copy>(&self, data: &[T]) {
        let bytes = std::mem::size_of_val(data);
        assert!(!self.ptr.is_null() && bytes as u64 <= self.len);
        // SAFETY: inside the mapping; the GPU is idle between plan runs.
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr().cast::<u8>(), self.ptr, bytes) };
    }

    /// Reads `n` `T`s from the start (host-visible memory only).
    pub fn read<T: Copy + Default>(&self, n: usize) -> Vec<T> {
        assert!(!self.ptr.is_null() && (n * size_of::<T>()) as u64 <= self.len);
        let mut out = vec![T::default(); n];
        // SAFETY: inside the mapping; the plan that wrote it has finished.
        unsafe { std::ptr::copy_nonoverlapping(self.ptr.cast::<T>(), out.as_mut_ptr(), n) };
        out
    }
}

/// One allocation handed out in 256-byte aligned views.
pub struct Arena {
    view: View,
    next: u64,
}

impl Arena {
    pub fn take(&mut self, bytes: u64) -> Result<View, Error> {
        let off = self.next;
        if off + bytes > self.view.len {
            return Err(Error::Gpu(format!("arena of {} bytes is full ({off} used, {bytes} more)", self.view.len)));
        }
        self.next = (off + bytes).next_multiple_of(ALIGN);
        Ok(self.view.at(off, bytes))
    }

    /// A view holding `data`.
    pub fn put<T: Copy>(&mut self, data: &[T]) -> Result<View, Error> {
        let v = self.take(std::mem::size_of_val(data) as u64)?;
        v.write(data);
        Ok(v)
    }
}

impl Vk {
    /// The first GPU that is not a CPU implementation (llvmpipe), if there
    /// is a Vulkan loader and such a device.
    pub fn new() -> Result<Self, Error> {
        // SAFETY: loading libvulkan; every handle below is used per the spec.
        unsafe {
            let entry = ash::Entry::load().map_err(|e| Error::Gpu(format!("loading libvulkan: {e}")))?;
            let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);
            let instance = entry
                .create_instance(&vk::InstanceCreateInfo::default().application_info(&app), None)
                .map_err(vkerr("vkCreateInstance"))?;
            let pdev = instance
                .enumerate_physical_devices()
                .map_err(vkerr("vkEnumeratePhysicalDevices"))?
                .into_iter()
                .find(|&p| instance.get_physical_device_properties(p).device_type != vk::PhysicalDeviceType::CPU)
                .ok_or_else(|| Error::Gpu("no Vulkan GPU".into()))?;
            let props = instance.get_physical_device_properties(pdev);
            let name = props.device_name_as_c_str().unwrap_or(c"?").to_string_lossy().into_owned();
            let qfi = instance
                .get_physical_device_queue_family_properties(pdev)
                .iter()
                .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
                .ok_or_else(|| Error::Gpu("no compute queue".into()))? as u32;
            let prio = [1.0];
            let queues = [vk::DeviceQueueCreateInfo::default().queue_family_index(qfi).queue_priorities(&prio)];
            let mut exts =
                vec![ash::khr::external_memory_fd::NAME.as_ptr(), ash::ext::external_memory_dma_buf::NAME.as_ptr()];
            // cooperative matrices, where the device has everything they need
            let has_ext = |n: &CStr| {
                instance
                    .enumerate_device_extension_properties(pdev)
                    .unwrap_or_default()
                    .iter()
                    .any(|e| e.extension_name_as_c_str().is_ok_and(|x| x == n))
            };
            let mut f11 = vk::PhysicalDeviceVulkan11Features::default();
            let mut f12 = vk::PhysicalDeviceVulkan12Features::default();
            let mut f13 = vk::PhysicalDeviceVulkan13Features::default();
            let mut fcm = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default();
            let coop_ext = props.api_version >= vk::API_VERSION_1_3 && has_ext(ash::khr::cooperative_matrix::NAME);
            {
                let mut f2 =
                    vk::PhysicalDeviceFeatures2::default().push_next(&mut f11).push_next(&mut f12).push_next(&mut f13);
                if coop_ext {
                    f2 = f2.push_next(&mut fcm);
                }
                instance.get_physical_device_features2(pdev, &mut f2);
            }
            let mut ssc = vk::PhysicalDeviceSubgroupSizeControlProperties::default();
            instance.get_physical_device_properties2(
                pdev,
                &mut vk::PhysicalDeviceProperties2::default().push_next(&mut ssc),
            );
            let f16_config = coop_ext
                && ash::khr::cooperative_matrix::Instance::new(&entry, &instance)
                    .get_physical_device_cooperative_matrix_properties(pdev)
                    .unwrap_or_default()
                    .iter()
                    .any(|c| {
                        (c.m_size, c.n_size, c.k_size) == (16, 16, 16)
                            && c.a_type == vk::ComponentTypeKHR::FLOAT16
                            && c.b_type == vk::ComponentTypeKHR::FLOAT16
                            && c.c_type == vk::ComponentTypeKHR::FLOAT32
                            && c.result_type == vk::ComponentTypeKHR::FLOAT32
                            && c.scope == vk::ScopeKHR::SUBGROUP
                    });
            let coop_f16 = f16_config
                && fcm.cooperative_matrix == vk::TRUE
                && f11.storage_buffer16_bit_access == vk::TRUE
                && f12.shader_float16 == vk::TRUE
                && f12.vulkan_memory_model == vk::TRUE
                && f12.vulkan_memory_model_device_scope == vk::TRUE
                && f13.subgroup_size_control == vk::TRUE
                && ssc.min_subgroup_size <= 64
                && ssc.max_subgroup_size >= 64
                && ssc.required_subgroup_size_stages.contains(vk::ShaderStageFlags::COMPUTE)
                && std::env::var("TACONITE_GPU_COOP").map_or(true, |v| v != "0");
            let mut e11 = vk::PhysicalDeviceVulkan11Features::default().storage_buffer16_bit_access(true);
            let mut e12 = vk::PhysicalDeviceVulkan12Features::default()
                .shader_float16(true)
                .vulkan_memory_model(true)
                .vulkan_memory_model_device_scope(true);
            let mut e13 = vk::PhysicalDeviceVulkan13Features::default().subgroup_size_control(true);
            let mut ecm = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default().cooperative_matrix(true);
            let mut dci = vk::DeviceCreateInfo::default().queue_create_infos(&queues);
            if coop_f16 {
                exts.push(ash::khr::cooperative_matrix::NAME.as_ptr());
                dci = dci.push_next(&mut e11).push_next(&mut e12).push_next(&mut e13).push_next(&mut ecm);
            }
            let dci = dci.enabled_extension_names(&exts);
            let device = instance.create_device(pdev, &dci, None).map_err(vkerr("vkCreateDevice"))?;
            let queue = device.get_device_queue(qfi, 0);
            let pool = device
                .create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(qfi), None)
                .map_err(vkerr("vkCreateCommandPool"))?;
            let sizes = [vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 4096 }];
            let desc_pool = device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default().max_sets(1024).pool_sizes(&sizes),
                    None,
                )
                .map_err(vkerr("vkCreateDescriptorPool"))?;
            let mem_props = instance.get_physical_device_memory_properties(pdev);
            let ts_period = props.limits.timestamp_period;
            let ext_fd = ash::khr::external_memory_fd::Device::new(&instance, &device);
            Ok(Vk {
                _entry: entry,
                _instance: instance,
                device,
                queue,
                pool,
                desc_pool,
                mem_props,
                ext_fd,
                pipes: HashMap::new(),
                imports: HashMap::new(),
                name,
                ts_period,
                coop_f16,
            })
        }
    }

    fn memory_type(&self, bits: u32, want: vk::MemoryPropertyFlags) -> Option<u32> {
        (0..self.mem_props.memory_type_count)
            .find(|&i| bits & (1 << i) != 0 && self.mem_props.memory_types[i as usize].property_flags.contains(want))
    }

    fn buffer(&self, size: u64, external: bool) -> Result<vk::Buffer, Error> {
        let mut ext = vk::ExternalMemoryBufferCreateInfo::default().handle_types(DMA_BUF);
        let mut ci = vk::BufferCreateInfo::default()
            .size(size)
            .usage(
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::INDIRECT_BUFFER,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        if external {
            ci = ci.push_next(&mut ext);
        }
        // SAFETY: a valid create info.
        unsafe { self.device.create_buffer(&ci, None) }.map_err(vkerr("vkCreateBuffer"))
    }

    fn bind_map(&self, buffer: vk::Buffer, memory: vk::DeviceMemory, size: u64, ti: u32) -> Result<View, Error> {
        let flags = self.mem_props.memory_types[ti as usize].property_flags;
        // SAFETY: fresh buffer and memory; host-visible memory is mapped once, for good.
        unsafe {
            self.device.bind_buffer_memory(buffer, memory, 0).map_err(vkerr("vkBindBufferMemory"))?;
            let ptr = if flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE) {
                self.device
                    .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                    .map_err(vkerr("vkMapMemory"))? as *mut u8
            } else {
                std::ptr::null_mut()
            };
            Ok(View { buffer, off: 0, len: size, ptr })
        }
    }

    /// An arena of `bytes` in memory the GPU reads fast and the host maps
    /// (device-local and host-visible on an APU), zeroed. The host should
    /// write it, not read it: that mapping is uncached.
    pub fn arena(&self, bytes: u64) -> Result<Arena, Error> {
        // device-local and host-visible where there is room (an APU's
        // carve-out can be small), else host-visible system memory
        self.arena_in(
            bytes,
            &[
                vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_VISIBLE,
                vk::MemoryPropertyFlags::HOST_VISIBLE,
            ],
        )
    }

    /// An arena the host reads back: cached, coherent system memory where
    /// there is some (the host reading an [`arena`](Self::arena)'s uncached
    /// mapping is several times slower), zeroed.
    pub fn arena_readback(&self, bytes: u64) -> Result<Arena, Error> {
        self.arena_in(
            bytes,
            &[
                vk::MemoryPropertyFlags::HOST_VISIBLE
                    | vk::MemoryPropertyFlags::HOST_COHERENT
                    | vk::MemoryPropertyFlags::HOST_CACHED,
                vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_VISIBLE,
                vk::MemoryPropertyFlags::HOST_VISIBLE,
            ],
        )
    }

    /// An arena in the first memory type of `prefs` that has room.
    fn arena_in(&self, bytes: u64, prefs: &[vk::MemoryPropertyFlags]) -> Result<Arena, Error> {
        let buffer = self.buffer(bytes, false)?;
        // SAFETY: buffer is live.
        let req = unsafe { self.device.get_buffer_memory_requirements(buffer) };
        let mut types: Vec<u32> = prefs.iter().filter_map(|&f| self.memory_type(req.memory_type_bits, f)).collect();
        types.dedup();
        if types.is_empty() {
            return Err(Error::Gpu("no host-visible memory".into()));
        }
        let mut last = vk::Result::ERROR_OUT_OF_DEVICE_MEMORY;
        let mut got = None;
        for ti in types {
            let ai = vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(ti);
            // SAFETY: a valid allocate info.
            match unsafe { self.device.allocate_memory(&ai, None) } {
                Ok(m) => {
                    got = Some((m, ti));
                    break;
                }
                Err(e) => last = e,
            }
        }
        let (memory, ti) = got.ok_or_else(|| vkerr("vkAllocateMemory")(last))?;
        let view = self.bind_map(buffer, memory, bytes, ti)?;
        // SAFETY: the fresh mapping is `bytes` long.
        unsafe { std::ptr::write_bytes(view.ptr, 0, bytes as usize) };
        Ok(Arena { view, next: 0 })
    }

    /// Another device's memory, handed over as a dma-buf of at least
    /// `bytes`: the GPU's view of it. Vulkan takes the fd.
    pub fn import(&self, fd: OwnedFd, bytes: u64) -> Result<View, Error> {
        let buffer = self.buffer(bytes, true)?;
        // SAFETY: as in arena; on success the fd belongs to the driver.
        unsafe {
            let req = self.device.get_buffer_memory_requirements(buffer);
            let mut fdp = vk::MemoryFdPropertiesKHR::default();
            self.ext_fd
                .get_memory_fd_properties(DMA_BUF, fd.as_raw_fd(), &mut fdp)
                .map_err(vkerr("vkGetMemoryFdPropertiesKHR"))?;
            let ti = self
                .memory_type(req.memory_type_bits & fdp.memory_type_bits, vk::MemoryPropertyFlags::empty())
                .ok_or_else(|| Error::Gpu("no memory type takes the dma-buf".into()))?;
            let mut imp = vk::ImportMemoryFdInfoKHR::default().handle_type(DMA_BUF).fd(fd.as_raw_fd());
            let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().buffer(buffer);
            let ai = vk::MemoryAllocateInfo::default()
                .allocation_size(req.size)
                .memory_type_index(ti)
                .push_next(&mut imp)
                .push_next(&mut dedicated);
            let memory = self.device.allocate_memory(&ai, None).map_err(vkerr("vkAllocateMemory(dma-buf)"))?;
            let _ = fd.into_raw_fd(); // the driver's now
            self.device.bind_buffer_memory(buffer, memory, 0).map_err(vkerr("vkBindBufferMemory"))?;
            Ok(View { buffer, off: 0, len: bytes, ptr: std::ptr::null_mut() })
        }
    }

    /// The GPU's view of NPU buffer `b` (a whole allocation, not a
    /// sub-buffer), exported by the NPU and imported the first time.
    pub fn import_npu(&mut self, b: &crate::direct::Buffer) -> Result<View, Error> {
        let key = b.as_slice::<u8>().as_ptr() as usize;
        if let Some(v) = self.imports.get(&key) {
            return Ok(*v);
        }
        let v = self.import(b.export_dmabuf()?, b.len_bytes() as u64)?;
        self.imports.insert(key, v);
        Ok(v)
    }

    /// The compute pipeline for WGSL `src` (entry `main`, storage buffers at
    /// bindings `0..bindings` of group 0), compiled on first use.
    fn pipe(&mut self, src: &str, bindings: u32) -> Result<&Pipe, Error> {
        if !self.pipes.contains_key(src) {
            let t = Instant::now();
            let p = self.compile(src, bindings)?;
            if profiling() {
                let first = src.lines().find(|l| l.starts_with("const ")).unwrap_or("");
                eprintln!(
                    "compiled a pipeline in {:.0} ms ({} bindings, {first})",
                    t.elapsed().as_secs_f64() * 1e3,
                    bindings
                );
            }
            self.pipes.insert(src.to_string(), p);
        }
        Ok(&self.pipes[src])
    }

    fn compile(&self, src: &str, bindings: u32) -> Result<Pipe, Error> {
        let shader_err = |e: String| Error::Gpu(format!("shader: {e}\n{src}"));
        let module = naga::front::wgsl::parse_str(src).map_err(|e| shader_err(e.emit_to_string(src)))?;
        let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
            .validate(&module)
            .map_err(|e| shader_err(format!("{e:?}")))?;
        // cooperative matrices: the Vulkan memory model (SPIR-V 1.5) and
        // 64-wide subgroups, which the shaders' tiling assumes
        let coop = src.contains("enable wgpu_cooperative_matrix;");
        if coop && !self.coop_f16 {
            return Err(shader_err("this device has no f16 cooperative matrices".into()));
        }
        let opts = naga::back::spv::Options {
            // Every shader here writes its workgroup memory before reading
            // it. The default zeroes it first with one store of the whole
            // array, which RADV expands element by element: a 23 KB tile
            // buffer took 11 s to compile.
            zero_initialize_workgroup_memory: naga::back::spv::ZeroInitializeWorkgroupMemoryMode::None,
            // loop bounds are ours, not an untrusted page's
            force_loop_bounding: false,
            // subgroup operations need SPIR-V 1.3 (Vulkan 1.1)
            lang_version: if coop { (1, 5) } else { (1, 3) },
            ..Default::default()
        };
        let words = naga::back::spv::write_vec(&module, &info, &opts, None).map_err(|e| shader_err(e.to_string()))?;
        // SAFETY: valid SPIR-V; create infos over live handles.
        unsafe {
            let shader = self
                .device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
                .map_err(vkerr("vkCreateShaderModule"))?;
            let binds: Vec<_> = (0..bindings)
                .map(|b| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(b)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                })
                .collect();
            let dsl = self
                .device
                .create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&binds), None)
                .map_err(vkerr("vkCreateDescriptorSetLayout"))?;
            let dsls = [dsl];
            let layout = self
                .device
                .create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&dsls), None)
                .map_err(vkerr("vkCreatePipelineLayout"))?;
            let entry: &CStr = c"main";
            let mut wave64 =
                vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default().required_subgroup_size(64);
            let mut stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(shader)
                .name(entry);
            if coop {
                stage = stage.push_next(&mut wave64);
            }
            let pipeline = self
                .device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout)],
                    None,
                )
                .map_err(|(_, e)| Error::Gpu(format!("vkCreateComputePipelines: {e}")))?[0];
            self.device.destroy_shader_module(shader, None);
            Ok(Pipe { pipeline, layout, dsl })
        }
    }

    /// Starts recording a plan.
    pub fn record(&mut self) -> Result<Recorder<'_>, Error> {
        let (cmd, profile) = self.begin()?;
        let mut r = Recorder { vk: self, cmd, dispatches: 0, profile };
        r.start();
        Ok(r)
    }

    /// A fresh command buffer, begun, with its timestamp pool if profiling.
    #[allow(clippy::type_complexity)]
    fn begin(&mut self) -> Result<(vk::CommandBuffer, Option<(vk::QueryPool, Vec<String>)>), Error> {
        // SAFETY: a primary command buffer from our pool, begun once.
        let cmd = unsafe {
            let cmd = self
                .device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default().command_pool(self.pool).command_buffer_count(1),
                )
                .map_err(vkerr("vkAllocateCommandBuffers"))?[0];
            self.device
                .begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())
                .map_err(vkerr("vkBeginCommandBuffer"))?;
            cmd
        };
        let profile = if profiling() {
            // SAFETY: a fresh query pool, reset in the command buffer before use.
            unsafe {
                let pool = self
                    .device
                    .create_query_pool(
                        &vk::QueryPoolCreateInfo::default()
                            .query_type(vk::QueryType::TIMESTAMP)
                            .query_count(MAX_STAMPS),
                        None,
                    )
                    .map_err(vkerr("vkCreateQueryPool"))?;
                self.device.cmd_reset_query_pool(cmd, pool, 0, MAX_STAMPS);
                self.device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, pool, 0);
                Some((pool, Vec::new()))
            }
        } else {
            None
        };
        Ok((cmd, profile))
    }
}

impl Drop for Vk {
    fn drop(&mut self) {
        // SAFETY: plans wait for their runs, so nothing is in flight. The
        // device and instance are left to process exit: views of it may
        // still be held.
        unsafe {
            let _ = self.device.device_wait_idle();
        }
    }
}

/// A dispatch's workgroup counts: fixed at recording, or read from a
/// buffer when the plan runs.
enum Grid {
    Fixed([u32; 3]),
    Indirect(View),
}

/// Records dispatches into a plan's command buffer, a full compute
/// barrier after each (the plans are chains of dependent small ops).
pub struct Recorder<'a> {
    vk: &'a mut Vk,
    cmd: vk::CommandBuffer,
    dispatches: usize,
    /// `TACONITE_GPU_PROFILE`: the timestamp pool and each stamp's op label
    profile: Option<(vk::QueryPool, Vec<String>)>,
}

const MAX_STAMPS: u32 = 1024;

impl Recorder<'_> {
    /// The head of every command buffer: whatever the host, the NPU and
    /// earlier submissions wrote -> the shaders.
    fn start(&mut self) {
        self.barrier(
            vk::PipelineStageFlags::HOST | vk::PipelineStageFlags::ALL_COMMANDS,
            vk::AccessFlags::HOST_WRITE | vk::AccessFlags::MEMORY_WRITE,
        );
    }

    fn barrier(&mut self, src_stage: vk::PipelineStageFlags, src: vk::AccessFlags) {
        let b = [vk::MemoryBarrier::default().src_access_mask(src).dst_access_mask(
            vk::AccessFlags::SHADER_READ
                | vk::AccessFlags::SHADER_WRITE
                | vk::AccessFlags::TRANSFER_READ
                | vk::AccessFlags::INDIRECT_COMMAND_READ,
        )];
        // SAFETY: recording into our own command buffer.
        unsafe {
            self.vk.device.cmd_pipeline_barrier(
                self.cmd,
                src_stage,
                vk::PipelineStageFlags::COMPUTE_SHADER
                    | vk::PipelineStageFlags::TRANSFER
                    | vk::PipelineStageFlags::DRAW_INDIRECT,
                vk::DependencyFlags::empty(),
                &b,
                &[],
                &[],
            );
        }
    }

    /// Copies `src` over the start of `dst`.
    pub fn copy(&mut self, src: View, dst: View) {
        assert!(src.len <= dst.len);
        let region = [vk::BufferCopy { src_offset: src.off, dst_offset: dst.off, size: src.len }];
        // SAFETY: both ranges lie inside their buffers.
        unsafe { self.vk.device.cmd_copy_buffer(self.cmd, src.buffer, dst.buffer, &region) };
        self.barrier(vk::PipelineStageFlags::TRANSFER, vk::AccessFlags::TRANSFER_WRITE);
    }

    /// One dispatch, labelled `label` (for the profile), of `groups`
    /// workgroups of WGSL `src` over `views` (binding `i` = `views[i]`),
    /// after `consts` are substituted into it.
    pub fn op(
        &mut self,
        label: &str,
        src: &str,
        consts: &[(&str, String)],
        views: &[View],
        groups: [u32; 3],
    ) -> Result<(), Error> {
        self.dispatch(label, src, consts, views, Grid::Fixed(groups))
    }

    /// [`op`](Self::op) with its workgroup counts read from `args` when the
    /// plan runs (three `u32`s, `VkDispatchIndirectCommand`): a plan
    /// recorded once can launch only the workgroups a run needs, the host
    /// writing the counts with the run's inputs.
    pub fn op_indirect(
        &mut self,
        label: &str,
        src: &str,
        consts: &[(&str, String)],
        views: &[View],
        args: View,
    ) -> Result<(), Error> {
        assert!(args.len >= 12 && args.off % 4 == 0, "indirect args: 3 u32s at a 4-byte offset");
        self.dispatch(label, src, consts, views, Grid::Indirect(args))
    }

    fn dispatch(
        &mut self,
        label: &str,
        src: &str,
        consts: &[(&str, String)],
        views: &[View],
        grid: Grid,
    ) -> Result<(), Error> {
        let mut s = src.to_string();
        for (k, v) in consts {
            s = s.replace(&format!("{{{k}}}"), v);
        }
        if let Some(i) = s.find('{').filter(|&i| s[i + 1..].starts_with(|c: char| c.is_ascii_uppercase())) {
            return Err(Error::Gpu(format!("unsubstituted constant at {}", &s[i..(i + 16).min(s.len())])));
        }
        let (pipeline, layout, dsl) = {
            let p = self.vk.pipe(&s, views.len() as u32)?;
            (p.pipeline, p.layout, p.dsl)
        };
        let d = &self.vk.device;
        // SAFETY: live pool, layout and buffers; recording into our own command buffer.
        unsafe {
            let dsls = [dsl];
            let set = d
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default().descriptor_pool(self.vk.desc_pool).set_layouts(&dsls),
                )
                .map_err(vkerr("vkAllocateDescriptorSets"))?[0];
            let infos: Vec<[vk::DescriptorBufferInfo; 1]> = views
                .iter()
                .map(|v| [vk::DescriptorBufferInfo { buffer: v.buffer, offset: v.off, range: v.len }])
                .collect();
            let writes: Vec<_> = infos
                .iter()
                .enumerate()
                .map(|(i, info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(i as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(info)
                })
                .collect();
            d.update_descriptor_sets(&writes, &[]);
            d.cmd_bind_pipeline(self.cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            d.cmd_bind_descriptor_sets(self.cmd, vk::PipelineBindPoint::COMPUTE, layout, 0, &[set], &[]);
            match grid {
                Grid::Fixed([x, y, z]) => d.cmd_dispatch(self.cmd, x, y, z),
                Grid::Indirect(a) => d.cmd_dispatch_indirect(self.cmd, a.buffer, a.off),
            }
        }
        self.dispatches += 1;
        self.barrier(vk::PipelineStageFlags::COMPUTE_SHADER, vk::AccessFlags::SHADER_WRITE);
        if let Some((pool, labels)) = &mut self.profile {
            if labels.len() + 1 < MAX_STAMPS as usize {
                labels.push(label.to_string());
                // SAFETY: a query of our pool, reset at the start of the recording.
                unsafe {
                    self.vk.device.cmd_write_timestamp(
                        self.cmd,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        *pool,
                        labels.len() as u32,
                    )
                };
            }
        }
        Ok(())
    }

    /// Ends the recording.
    pub fn finish(mut self) -> Result<Plan, Error> {
        self.seal()
    }

    /// Ends the plan recorded so far and goes on recording a new one:
    /// submitted after it, the new one sees everything it wrote.
    pub fn cut(&mut self) -> Result<Plan, Error> {
        let plan = self.seal()?;
        let (cmd, profile) = self.vk.begin()?;
        (self.cmd, self.profile, self.dispatches) = (cmd, profile, 0);
        self.start();
        Ok(plan)
    }

    fn seal(&mut self) -> Result<Plan, Error> {
        let d = &self.vk.device;
        // SAFETY: ending our own command buffer; a fresh fence.
        unsafe {
            let b = [vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ | vk::AccessFlags::MEMORY_READ)];
            d.cmd_pipeline_barrier(
                self.cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::HOST | vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &b,
                &[],
                &[],
            );
            d.end_command_buffer(self.cmd).map_err(vkerr("vkEndCommandBuffer"))?;
            let fence = d.create_fence(&vk::FenceCreateInfo::default(), None).map_err(vkerr("vkCreateFence"))?;
            Ok(Plan {
                cmd: self.cmd,
                fence,
                dispatches: self.dispatches,
                profile: self.profile.take(),
                submitted: std::cell::Cell::new(None),
            })
        }
    }
}

/// A recorded chain of dispatches, replayed by [`Plan::run`] (or
/// [`Plan::submit`] now and [`Plan::wait`] later).
pub struct Plan {
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    pub dispatches: usize,
    profile: Option<(vk::QueryPool, Vec<String>)>,
    /// when the run in flight was submitted
    submitted: std::cell::Cell<Option<Instant>>,
}

impl Plan {
    /// Submits the plan and waits for it: its inputs must be written, and
    /// its outputs are readable when this returns.
    pub fn run(&self, vk: &Vk) -> Result<Duration, Error> {
        self.submit(vk)?;
        self.wait(vk)
    }

    /// Submits the plan without waiting; [`wait`](Self::wait) before
    /// touching what it reads or writes. A run still in flight (one whose
    /// wait was skipped, by an error in between) is waited for first.
    pub fn submit(&self, vk: &Vk) -> Result<(), Error> {
        if self.submitted.get().is_some() {
            self.wait(vk)?;
        }
        let cmds = [self.cmd];
        // SAFETY: the plan's command buffer, not in flight (waited for above).
        unsafe {
            vk.device
                .queue_submit(vk.queue, &[vk::SubmitInfo::default().command_buffers(&cmds)], self.fence)
                .map_err(vkerr("vkQueueSubmit"))?;
        }
        self.submitted.set(Some(Instant::now()));
        Ok(())
    }

    /// Waits for the submitted run; returns the time since its submit.
    pub fn wait(&self, vk: &Vk) -> Result<Duration, Error> {
        let t = self.submitted.take().expect("wait without a submit");
        // SAFETY: our fence, signalled by the submitted run.
        unsafe {
            vk.device.wait_for_fences(&[self.fence], true, 10_000_000_000).map_err(vkerr("vkWaitForFences"))?;
            vk.device.reset_fences(&[self.fence]).map_err(vkerr("vkResetFences"))?;
        }
        let dt = t.elapsed();
        if let Some((pool, labels)) = &self.profile {
            let mut stamps = vec![0u64; labels.len() + 1];
            // SAFETY: the run has finished, so every stamp is written.
            unsafe {
                vk.device
                    .get_query_pool_results(*pool, 0, &mut stamps, vk::QueryResultFlags::TYPE_64)
                    .map_err(vkerr("vkGetQueryPoolResults"))?;
            }
            // per label, in first-seen order
            let mut per: Vec<(String, f64, usize)> = Vec::new();
            for (i, l) in labels.iter().enumerate() {
                let ms = (stamps[i + 1] - stamps[i]) as f64 * vk.ts_period as f64 / 1e6;
                match per.iter_mut().find(|p| &p.0 == l) {
                    Some(p) => {
                        p.1 += ms;
                        p.2 += 1;
                    }
                    None => per.push((l.clone(), ms, 1)),
                }
            }
            let gpu_ms = (stamps[labels.len()] - stamps[0]) as f64 * vk.ts_period as f64 / 1e6;
            let parts: Vec<String> = per.iter().map(|(l, ms, n)| format!("{l} {ms:.2}/{n}")).collect();
            eprintln!(
                "GPU plan {:.2} ms on the GPU, {:.2} ms wall: {}",
                gpu_ms,
                dt.as_secs_f64() * 1e3,
                parts.join(" ")
            );
        }
        Ok(dt)
    }
}
