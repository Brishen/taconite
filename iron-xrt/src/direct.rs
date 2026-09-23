// SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
// SPDX-FileCopyrightText: Copyright (c) 2026 Eugene Hauptmann, Nataliya Kosmyna (RLX)
// SPDX-License-Identifier: Apache-2.0

//! Running IRON kernels with no XRT at all: the `amdxdna` driver's ioctls
//! on `/dev/accel/accel0`, issued from Rust.
//!
//! What XRT does for a [`crate::Kernel`] is, underneath, a short and fixed
//! ioctl sequence — traced from this crate's own XRT path on an NPU2:
//!
//! - once per process: a 64 MiB device heap (`CREATE_BO(DEV_HEAP)`) that
//!   device-side buffers are carved from, and a `GET_INFO` query of the
//!   array's shape;
//! - once per kernel: `CREATE_HWCTX` sized to the xclbin's partition, the
//!   partition's PDI (read out of the xclbin by [`axlf`]) into a device BO,
//!   `CONFIG_HWCTX(CONFIG_CU)` on it, the instruction stream into another
//!   device BO, and a command BO;
//! - per run: the command packet written into the command BO, one
//!   `EXEC_CMD`, one `SYNCOBJ_TIMELINE_WAIT` on the context's fence.
//!
//! [`Session`] / [`Kernel`] / [`Buffer`] / [`Run`] do that sequence with
//! the methods of the crate's XRT types (sub-buffers, `start`/`wait`,
//! kernels of one xclbin sharing a context), so a model switches paths by
//! its `use` line. Argument buffers are shared-memory BOs the NPU reaches
//! by their *host* virtual address (shared virtual addressing through the
//! IOMMU), so they are filled in place like [`crate::Buffer`]s, with the
//! CPU caches flushed around a run. The command packet is IRON's
//! `MLIR_AIE` kernel ABI, the one every design aiecc builds has: `opcode,
//! instr, ninstr, bo0…bo4` at fixed offsets (aiecc's `kernels.json`).
//!
//! Not covered: xclbins holding more than one PDI. The layouts follow the
//! kernel's `include/uapi/drm/amdxdna_accel.h` (Linux 7.1) and XRT's
//! `xrt/detail/xclbin.h`.
//!
//! Adapted from RLX's `rlx-xdna/src/direct.rs`
//! (<https://github.com/MIT-RLX/rlx>, MIT OR Apache-2.0), where the same
//! sequence was built for NPU1 but never saw a command complete; its ioctl
//! tracer (`tools/xdna_ioctl_trace.c`) is what pinned the sequence and
//! packet here. The likely reason it never saw one: it passed the fence
//! wait a *relative* timeout, which the DRM syncobj ioctls read as an
//! absolute `CLOCK_MONOTONIC` deadline — long past, so the wait returns
//! `ETIME` at once with the command still `NEW`. That is exactly what this
//! module does on an NPU2 given the same timeout (see [`Kernel::run`]).

use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::{c_int, c_ulong, c_void};
use std::fmt;
use std::fs::{File, OpenOptions};
use std::io;
use std::marker::PhantomData;
use std::os::fd::AsRawFd;
use std::path::Path;
use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, Instant};

use crate::Error;

/// Reading the pieces of an `.xclbin` (an AXLF container) the direct path
/// needs: what XRT's `register_xclbin` extracts.
pub mod axlf {
    use std::io;

    /// `axlf.m_header.m_numSections`, after the magic, signature, key block
    /// and unique id (304 bytes) and the header's fixed fields.
    const NUM_SECTIONS_OFF: usize = 448;
    /// `axlf.m_sections[]`: `{kind u32, name[16], offset u64, size u64}`.
    const SECTIONS_OFF: usize = 456;
    const SECTION_HEADER_SIZE: usize = 40;
    const AIE_PARTITION: u32 = 32;

    /// The AIE partition of an xclbin.
    #[derive(Debug, Clone)]
    pub struct Partition {
        /// Columns the partition spans.
        pub column_width: u16,
        /// The PDI the firmware loads into the partition: the array
        /// configuration and the cores' programs.
        pub pdi: Vec<u8>,
    }

    fn u16_at(b: &[u8], off: usize) -> io::Result<u16> {
        b.get(off..off + 2)
            .map(|s| u16::from_le_bytes(s.try_into().unwrap()))
            .ok_or_else(|| io::Error::other("xclbin truncated"))
    }

    fn u32_at(b: &[u8], off: usize) -> io::Result<u32> {
        b.get(off..off + 4)
            .map(|s| u32::from_le_bytes(s.try_into().unwrap()))
            .ok_or_else(|| io::Error::other("xclbin truncated"))
    }

    fn u64_at(b: &[u8], off: usize) -> io::Result<u64> {
        b.get(off..off + 8)
            .map(|s| u64::from_le_bytes(s.try_into().unwrap()))
            .ok_or_else(|| io::Error::other("xclbin truncated"))
    }

    /// The `AIE_PARTITION` section of `xclbin`: its column width and its PDI
    /// (an IRON xclbin has exactly one).
    pub fn partition(xclbin: &[u8]) -> io::Result<Partition> {
        if xclbin.get(..8) != Some(b"xclbin2\0".as_slice()) {
            return Err(io::Error::other("not an xclbin (no xclbin2 magic)"));
        }
        let sections = u32_at(xclbin, NUM_SECTIONS_OFF)? as usize;
        let mut section = None;
        for i in 0..sections {
            let h = SECTIONS_OFF + i * SECTION_HEADER_SIZE;
            if u32_at(xclbin, h)? == AIE_PARTITION {
                section = Some(u64_at(xclbin, h + 24)? as usize);
                break;
            }
        }
        let s = section.ok_or_else(|| io::Error::other("xclbin has no AIE_PARTITION section"))?;
        // struct aie_partition: info (aie_partition_info, column_width first)
        // at +32, aie_pdi (array_offset {count, offset}) at +120.
        let column_width = u16_at(xclbin, s + 32)?;
        let pdis = u32_at(xclbin, s + 120)?;
        if pdis != 1 {
            return Err(io::Error::other(format!(
                "AIE_PARTITION holds {pdis} PDIs; the direct path loads exactly one"
            )));
        }
        // struct aie_pdi: uuid[16], then pdi_image (array_offset of bytes,
        // offset from the section's start).
        let p = s + u32_at(xclbin, s + 124)? as usize;
        let (size, off) = (u32_at(xclbin, p + 16)? as usize, u32_at(xclbin, p + 20)? as usize);
        let pdi = xclbin.get(s + off..s + off + size).ok_or_else(|| io::Error::other("PDI outside the xclbin"))?;
        Ok(Partition { column_width, pdi: pdi.to_vec() })
    }
}

// ── the ioctl ABI ───────────────────────────────────────────────────────

mod sys {
    use super::*;

    unsafe extern "C" {
        pub fn ioctl(fd: c_int, request: c_ulong, ...) -> c_int;
        pub fn mmap(addr: *mut c_void, len: usize, prot: c_int, flags: c_int, fd: c_int, off: i64) -> *mut c_void;
        pub fn munmap(addr: *mut c_void, len: usize) -> c_int;
    }

    pub const PROT_NONE: c_int = 0;
    pub const PROT_RW: c_int = 0x1 | 0x2;
    pub const MAP_SHARED: c_int = 0x01;
    pub const MAP_PRIVATE: c_int = 0x02;
    pub const MAP_FIXED: c_int = 0x10;
    pub const MAP_ANONYMOUS: c_int = 0x20;
    pub const MAP_NORESERVE: c_int = 0x4000;
    pub const MAP_FAILED: *mut c_void = !0usize as *mut c_void;
    pub const ETIME: i32 = 62;

    /// `DRM_IOWR(nr, size)`: `_IOC(READ|WRITE, 'd', nr, size)`.
    pub const fn drm_iowr(nr: u32, size: usize) -> c_ulong {
        ((3u32 << 30) | ((size as u32) << 16) | (0x64 << 8) | nr) as c_ulong
    }

    const DRM_COMMAND_BASE: u32 = 0x40;
    // enum amdxdna_drm_ioctl_id
    pub const CREATE_HWCTX: u32 = DRM_COMMAND_BASE;
    pub const DESTROY_HWCTX: u32 = DRM_COMMAND_BASE + 1;
    pub const CONFIG_HWCTX: u32 = DRM_COMMAND_BASE + 2;
    pub const CREATE_BO: u32 = DRM_COMMAND_BASE + 3;
    pub const GET_BO_INFO: u32 = DRM_COMMAND_BASE + 4;
    pub const EXEC_CMD: u32 = DRM_COMMAND_BASE + 6;
    pub const GET_INFO: u32 = DRM_COMMAND_BASE + 7;
    pub const SET_STATE: u32 = DRM_COMMAND_BASE + 8;
    pub const GET_ARRAY: u32 = DRM_COMMAND_BASE + 10;
    // drm.h
    pub const GEM_CLOSE: u32 = 0x09;
    pub const SYNCOBJ_TIMELINE_WAIT: u32 = 0xca;

    // enum amdxdna_bo_type
    pub const BO_SHARE: u32 = 1;
    pub const BO_DEV_HEAP: u32 = 2;
    pub const BO_DEV: u32 = 3;
    pub const BO_CMD: u32 = 4;

    // enum amdxdna_drm_get_param
    pub const QUERY_AIE_METADATA: u32 = 1;
    pub const QUERY_AIE_VERSION: u32 = 2;
    pub const QUERY_FIRMWARE_VERSION: u32 = 8;
    pub const GET_POWER_MODE: u32 = 9;
    // amdxdna_drm_get_array params
    pub const HW_CONTEXT_ALL: u32 = 0;
    // enum amdxdna_drm_set_param
    pub const SET_POWER_MODE: u32 = 0;

    #[repr(C)]
    #[derive(Default)]
    pub struct GetInfo {
        pub param: u32,
        pub buffer_size: u32,
        pub buffer: u64,
    }

    /// `amdxdna_drm_set_state` has `get_info`'s layout.
    pub type SetState = GetInfo;

    #[repr(C)]
    #[derive(Default)]
    pub struct QosInfo {
        pub gops: u32,
        pub fps: u32,
        pub dma_bandwidth: u32,
        pub latency: u32,
        pub frame_exec_time: u32,
        pub priority: u32,
    }

    #[repr(C)]
    #[derive(Default)]
    pub struct CreateHwctx {
        pub ext: u64,
        pub ext_flags: u64,
        pub qos_p: u64,
        pub umq_bo: u32,
        pub log_buf_bo: u32,
        pub max_opc: u32,
        pub num_tiles: u32,
        pub mem_size: u32,
        pub umq_doorbell: u32,
        pub handle: u32,
        pub syncobj_handle: u32,
    }

    #[repr(C)]
    #[derive(Default)]
    pub struct HandlePad {
        pub handle: u32,
        pub pad: u32,
    }

    #[repr(C)]
    #[derive(Default)]
    pub struct ConfigHwctx {
        pub handle: u32,
        pub param_type: u32,
        pub param_val: u64,
        pub param_val_size: u32,
        pub pad: u32,
    }

    /// `amdxdna_hwctx_param_config_cu` with one `amdxdna_cu_config`.
    #[repr(C)]
    #[derive(Default)]
    pub struct ConfigCu {
        pub num_cus: u16,
        pub pad: [u16; 3],
        pub cu_bo: u32,
        pub cu_func: u8,
        pub cu_pad: [u8; 3],
    }

    #[repr(C)]
    #[derive(Default)]
    pub struct CreateBo {
        pub flags: u64,
        pub vaddr: u64,
        pub size: u64,
        pub ty: u32,
        pub handle: u32,
    }

    #[repr(C)]
    #[derive(Default)]
    pub struct GetBoInfo {
        pub ext: u64,
        pub ext_flags: u64,
        pub handle: u32,
        pub pad: u32,
        pub map_offset: u64,
        pub vaddr: u64,
        pub xdna_addr: u64,
    }

    #[repr(C)]
    #[derive(Default)]
    pub struct ExecCmd {
        pub ext: u64,
        pub ext_flags: u64,
        pub hwctx: u32,
        pub ty: u32,
        pub cmd_handles: u64,
        pub args: u64,
        pub cmd_count: u32,
        pub arg_count: u32,
        pub seq: u64,
    }

    #[repr(C)]
    #[derive(Default)]
    pub struct SyncobjTimelineWait {
        pub handles: u64,
        pub points: u64,
        pub timeout_nsec: i64,
        pub count_handles: u32,
        pub flags: u32,
        pub first_signaled: u32,
        pub pad: u32,
        pub deadline_nsec: u64,
    }

    #[repr(C)]
    #[derive(Default, Clone, Copy)]
    pub struct TileMetadata {
        pub row_count: u16,
        pub row_start: u16,
        pub dma_channel_count: u16,
        pub lock_count: u16,
        pub event_reg_count: u16,
        pub pad: [u16; 3],
    }

    #[repr(C)]
    #[derive(Default)]
    pub struct AieMetadata {
        pub col_size: u32,
        pub cols: u16,
        pub rows: u16,
        pub version: [u32; 2],
        pub core: TileMetadata,
        pub mem: TileMetadata,
        pub shim: TileMetadata,
    }

    #[repr(C)]
    #[derive(Default)]
    pub struct GetArray {
        pub param: u32,
        pub element_size: u32,
        pub num_element: u32,
        pub pad: u32,
        pub buffer: u64,
    }

    /// `amdxdna_drm_hwctx_entry`.
    #[repr(C)]
    #[derive(Default, Clone, Copy)]
    pub struct HwctxEntry {
        pub context_id: u32,
        pub start_col: u32,
        pub num_col: u32,
        pub hwctx_id: u32,
        pub pid: i64,
        pub command_submissions: u64,
        pub command_completions: u64,
        pub migrations: u64,
        pub preemptions: u64,
        pub errors: u64,
        pub priority: u64,
        pub heap_usage: u64,
        pub suspensions: u64,
        pub state: u32,
        pub pasid: u32,
        pub gops: u32,
        pub fps: u32,
        pub dma_bandwidth: u32,
        pub latency: u32,
        pub frame_exec_time: u32,
        pub txn_op_idx: u32,
        pub ctx_pc: u32,
        pub fatal_error_type: u32,
        pub fatal_error_exception_type: u32,
        pub fatal_error_exception_pc: u32,
        pub fatal_error_app_module: u32,
        pub pad: u32,
    }

    /// Issues `DRM_IOWR(nr, T)` on `arg`.
    pub fn drm<T>(fd: c_int, nr: u32, arg: &mut T) -> io::Result<()> {
        // SAFETY: `arg` is a live `T` whose layout is the ioctl's struct; the
        // pointers inside it (if any) are the caller's to keep valid.
        if unsafe { ioctl(fd, drm_iowr(nr, std::mem::size_of::<T>()), arg as *mut T) } < 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }
}

/// Writes back and invalidates the CPU cache lines over `[ptr, ptr+len)`:
/// the NPU's DMA does not snoop them, so this is XRT's `bo.sync` on this
/// driver (in either direction).
fn flush(ptr: *const u8, len: usize) {
    use std::arch::x86_64::{_mm_clflush, _mm_mfence};
    let mut line = ptr as usize & !63;
    let end = ptr as usize + len;
    // SAFETY: the range lies inside one live mapping; clflush has no other
    // effect than on the caches.
    unsafe {
        _mm_mfence();
        while line < end {
            _mm_clflush(line as *const u8);
            line += 64;
        }
        _mm_mfence();
    }
}

fn os(what: &str) -> impl Fn(io::Error) -> String + '_ {
    move |e| format!("{what}: {e}")
}

/// A mapped buffer object of the driver, unmapped and closed on drop.
struct Bo {
    dev: Arc<DeviceInner>,
    handle: u32,
    /// Host mapping (null for a device BO, reached through the heap's).
    ptr: *mut u8,
    size: usize,
    /// Device address (`u64::MAX` for a shared BO, which the NPU reaches
    /// through its host address).
    xdna_addr: u64,
}

impl Drop for Bo {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: mapped by this Bo, `size` bytes, unmapped once.
            unsafe { sys::munmap(self.ptr.cast(), self.size) };
        }
        self.dev.gem_close(self.handle);
    }
}

struct DeviceInner {
    file: File,
    meta: sys::AieMetadata,
    /// The device heap: handle, host mapping, device address.
    heap: (u32, *mut u8, u64),
    /// The live hardware contexts, by xclbin and kernel name.
    contexts: Mutex<HashMap<String, Weak<Context>>>,
}

const HEAP_SIZE: usize = 64 << 20;

impl DeviceInner {
    fn fd(&self) -> c_int {
        self.file.as_raw_fd()
    }

    fn get_info<T>(&self, param: u32, out: &mut T) -> io::Result<u32> {
        let mut gi = sys::GetInfo { param, buffer_size: std::mem::size_of::<T>() as u32, buffer: out as *mut T as u64 };
        sys::drm(self.fd(), sys::GET_INFO, &mut gi)?;
        Ok(gi.buffer_size)
    }

    /// `CREATE_BO` + `GET_BO_INFO`: (handle, map offset, device address).
    fn create_bo(&self, ty: u32, size: usize) -> io::Result<(u32, u64, u64)> {
        let mut cb = sys::CreateBo { size: size as u64, ty, ..Default::default() };
        sys::drm(self.fd(), sys::CREATE_BO, &mut cb)?;
        let mut info = sys::GetBoInfo { handle: cb.handle, ..Default::default() };
        if let Err(e) = sys::drm(self.fd(), sys::GET_BO_INFO, &mut info) {
            self.gem_close(cb.handle);
            return Err(e);
        }
        Ok((cb.handle, info.map_offset, info.xdna_addr))
    }

    fn mmap(&self, map_offset: u64, size: usize) -> io::Result<*mut u8> {
        // SAFETY: a fresh shared mapping of the BO at its fake offset.
        let p = unsafe {
            sys::mmap(std::ptr::null_mut(), size, sys::PROT_RW, sys::MAP_SHARED, self.fd(), map_offset as i64)
        };
        if p == sys::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        Ok(p.cast())
    }

    fn gem_close(&self, handle: u32) {
        let _ = sys::drm(self.fd(), sys::GEM_CLOSE, &mut sys::HandlePad { handle, pad: 0 });
    }

    /// Host address of the device BO at device address `xdna_addr`.
    fn heap_host(&self, xdna_addr: u64) -> *mut u8 {
        // SAFETY: device BOs are carved from the heap, so the offset is
        // inside its mapping.
        unsafe { self.heap.1.add((xdna_addr - self.heap.2) as usize) }
    }
}

impl Drop for DeviceInner {
    fn drop(&mut self) {
        // SAFETY: the heap mapping is HEAP_SIZE bytes, unmapped once.
        unsafe { sys::munmap(self.heap.1.cast(), HEAP_SIZE) };
        self.gem_close(self.heap.0);
    }
}

/// The NPU power modes (`enum amdxdna_power_mode_type`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowerMode {
    /// The driver's own DPM choice.
    Default,
    Low,
    Medium,
    High,
    /// Maximum clocks.
    Turbo,
}

impl PowerMode {
    fn from_raw(v: u8) -> Option<Self> {
        Some(match v {
            0 => PowerMode::Default,
            1 => PowerMode::Low,
            2 => PowerMode::Medium,
            3 => PowerMode::High,
            4 => PowerMode::Turbo,
            _ => return None,
        })
    }
}

/// One hardware context as the driver reports it.
#[derive(Debug, Clone)]
pub struct ContextInfo {
    pub context_id: u32,
    pub pid: i64,
    pub start_col: u32,
    pub num_col: u32,
    pub submissions: u64,
    pub completions: u64,
    pub errors: u64,
    /// The driver's context state: active (loaded on the array) or idle.
    pub active: bool,
    /// Where its instruction stream is, and why it died if it did (0: it
    /// did not).
    pub txn_op_idx: u32,
    pub fatal_error_type: u32,
}

/// The NPU, opened through its DRM accel node, with the process's device
/// heap mapped: this module's [`crate::Session`]. Kernels and buffers hold
/// a reference to it.
#[derive(Clone)]
pub struct Session {
    inner: Arc<DeviceInner>,
    _not_sync: PhantomData<*const ()>,
}

// SAFETY: as for crate::Session — the fd and mappings may move between
// threads; `!Sync` keeps their use one thread at a time.
unsafe impl Send for Session {}
unsafe impl Send for Kernel {}
unsafe impl Send for Buffer {}

impl Session {
    /// Opens `/dev/accel/accel<device_index>` (0 on a laptop).
    pub fn open(device_index: u32) -> Result<Self, Error> {
        Self::open_path(Path::new(&format!("/dev/accel/accel{device_index}")))
    }

    /// Opens the accel node at `path` and sets up the device heap.
    pub fn open_path(path: &Path) -> Result<Self, Error> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .map_err(|e| Error::Device(format!("{}: {e}", path.display())))?;
        let mut inner = DeviceInner {
            file,
            meta: Default::default(),
            heap: (0, std::ptr::null_mut(), 0),
            contexts: Mutex::new(HashMap::new()),
        };

        // XRT's order: the heap, a query, then (per kernel) the context.
        let (handle, map_offset, xdna_addr) =
            inner.create_bo(sys::BO_DEV_HEAP, HEAP_SIZE).map_err(os("CREATE_BO(DEV_HEAP)")).map_err(Error::Device)?;
        inner.heap.0 = handle;
        // The firmware maps the heap onto its device window and refuses a
        // host address not aligned to the heap's size, so place it on one.
        let host =
            map_aligned(&inner, map_offset, HEAP_SIZE).map_err(os("mapping the device heap")).map_err(Error::Device)?;
        inner.heap = (handle, host, xdna_addr);

        let mut meta = sys::AieMetadata::default();
        inner
            .get_info(sys::QUERY_AIE_METADATA, &mut meta)
            .map_err(os("GET_INFO(AIE_METADATA)"))
            .map_err(Error::Device)?;
        inner.meta = meta;
        Ok(Self { inner: Arc::new(inner), _not_sync: PhantomData })
    }

    /// Columns and rows of the AIE array.
    pub fn array(&self) -> (u16, u16) {
        (self.inner.meta.cols, self.inner.meta.rows)
    }

    /// The AIE array's version, as the driver reports it.
    pub fn aie_version(&self) -> Result<(u32, u32), Error> {
        let mut v = [0u32; 2];
        self.inner
            .get_info(sys::QUERY_AIE_VERSION, &mut v)
            .map_err(|e| Error::Device(format!("GET_INFO(AIE_VERSION): {e}")))?;
        Ok((v[0], v[1]))
    }

    /// The NPU firmware's `(major, minor, patch, build)`.
    pub fn firmware_version(&self) -> Result<(u32, u32, u32, u32), Error> {
        let mut v = [0u32; 4];
        self.inner
            .get_info(sys::QUERY_FIRMWARE_VERSION, &mut v)
            .map_err(|e| Error::Device(format!("GET_INFO(FIRMWARE_VERSION): {e}")))?;
        Ok((v[0], v[1], v[2], v[3]))
    }

    pub fn power_mode(&self) -> Result<PowerMode, Error> {
        let mut m = [0u8; 8];
        self.inner
            .get_info(sys::GET_POWER_MODE, &mut m)
            .map_err(|e| Error::Device(format!("GET_INFO(POWER_MODE): {e}")))?;
        PowerMode::from_raw(m[0]).ok_or_else(|| Error::Device(format!("unknown power mode {}", m[0])))
    }

    /// Sets the NPU's power mode, for the whole device (every process's
    /// contexts, XRT's included) until it is set again. The driver allows
    /// it only with `CAP_SYS_ADMIN`.
    pub fn set_power_mode(&self, mode: PowerMode) -> Result<(), Error> {
        let m = [mode as u8, 0, 0, 0, 0, 0, 0, 0];
        let mut st = sys::SetState { param: sys::SET_POWER_MODE, buffer_size: 8, buffer: m.as_ptr() as u64 };
        sys::drm(self.inner.fd(), sys::SET_STATE, &mut st)
            .map_err(|e| Error::Device(format!("SET_STATE(POWER_MODE): {e}")))
    }

    /// The hardware contexts the driver holds, across every process, with
    /// their command counters: where a hung command stopped.
    pub fn contexts(&self) -> Result<Vec<ContextInfo>, Error> {
        let mut buf = vec![sys::HwctxEntry::default(); 64];
        let mut ga = sys::GetArray {
            param: sys::HW_CONTEXT_ALL,
            element_size: std::mem::size_of::<sys::HwctxEntry>() as u32,
            num_element: buf.len() as u32,
            pad: 0,
            buffer: buf.as_mut_ptr() as u64,
        };
        sys::drm(self.inner.fd(), sys::GET_ARRAY, &mut ga)
            .map_err(|e| Error::Device(format!("GET_ARRAY(HW_CONTEXT_ALL): {e}")))?;
        Ok(buf[..(ga.num_element as usize).min(buf.len())]
            .iter()
            .map(|h| ContextInfo {
                context_id: h.context_id,
                pid: h.pid,
                start_col: h.start_col,
                num_col: h.num_col,
                submissions: h.command_submissions,
                completions: h.command_completions,
                errors: h.errors,
                active: h.state == 1,
                txn_op_idx: h.txn_op_idx,
                fatal_error_type: h.fatal_error_type,
            })
            .collect())
    }

    /// Loads an xclbin + instruction stream, as [`crate::Session::load_kernel`]
    /// does: kernels naming the same xclbin (and kernel name) share one
    /// hardware context, resident until the last of them is dropped. The
    /// context runs the xclbin's one PDI whatever `kernel_name` says; the
    /// name only tells contexts apart. `ops_per_run` is declared to the
    /// driver as the context's QoS `gops`.
    pub fn load_kernel(
        &self,
        xclbin: &Path,
        insts: &Path,
        kernel_name: Option<&str>,
        ops_per_run: u64,
    ) -> Result<Kernel, Error> {
        let insts = std::fs::read(insts).map_err(|e| Error::Kernel(format!("{}: {e}", insts.display())))?;
        let key = format!("{}\n{}", xclbin.display(), kernel_name.unwrap_or(""));
        let ctx = {
            let mut contexts = self.inner.contexts.lock().unwrap();
            match contexts.get(&key).and_then(Weak::upgrade) {
                Some(ctx) => ctx,
                None => {
                    let ctx = Arc::new(self.create_context(xclbin, ops_per_run)?);
                    contexts.retain(|_, c| c.strong_count() > 0);
                    contexts.insert(key, Arc::downgrade(&ctx));
                    ctx
                }
            }
        };
        let instr = self
            .device_bo(&insts)
            .map_err(|e| Error::Kernel(format!("{}: instruction buffer: {e}", xclbin.display())))?;
        Ok(Kernel {
            ctx,
            instr,
            ninstr_bytes: insts.len() as u32,
            idle: RefCell::new(Vec::new()),
            _not_sync: PhantomData,
        })
    }

    /// `CREATE_HWCTX` sized to the xclbin's partition, with its PDI bound as
    /// the compute unit.
    fn create_context(&self, xclbin: &Path, ops_per_run: u64) -> Result<Context, Error> {
        let kerr = |what: &str| {
            let what = format!("{}: {what}", xclbin.display());
            move |e: io::Error| Error::Kernel(format!("{what}: {e}"))
        };
        let bytes = std::fs::read(xclbin).map_err(kerr("reading"))?;
        let part = axlf::partition(&bytes).map_err(kerr("parsing"))?;
        let dev = &self.inner;
        let qos = sys::QosInfo {
            gops: u32::try_from((ops_per_run as f64 / 1e9).round() as u64).unwrap_or(u32::MAX),
            ..Default::default()
        };
        let mut c = sys::CreateHwctx {
            qos_p: &qos as *const sys::QosInfo as u64,
            // What XRT asks for: the partition's core tiles, 2048 ops/cycle.
            max_opc: 2048,
            num_tiles: part.column_width as u32 * dev.meta.core.row_count as u32,
            ..Default::default()
        };
        sys::drm(dev.fd(), sys::CREATE_HWCTX, &mut c).map_err(kerr("CREATE_HWCTX"))?;
        let mut ctx = Context { dev: dev.clone(), handle: c.handle, syncobj: c.syncobj_handle, _pdi: None };

        let pdi = self.device_bo(&part.pdi).map_err(kerr("PDI buffer"))?;
        let mut cu = sys::ConfigCu { num_cus: 1, cu_bo: pdi.handle, ..Default::default() };
        let mut cfg = sys::ConfigHwctx {
            handle: ctx.handle,
            param_type: 0, // DRM_AMDXDNA_HWCTX_CONFIG_CU
            param_val: &mut cu as *mut sys::ConfigCu as u64,
            param_val_size: std::mem::size_of::<sys::ConfigCu>() as u32,
            pad: 0,
        };
        ctx._pdi = Some(pdi);
        sys::drm(dev.fd(), sys::CONFIG_HWCTX, &mut cfg).map_err(kerr("CONFIG_HWCTX(CU)"))?;
        Ok(ctx)
    }

    /// A device BO (carved from the heap) holding `data`.
    fn device_bo(&self, data: &[u8]) -> io::Result<Bo> {
        let dev = &self.inner;
        let (handle, _, xdna_addr) = dev.create_bo(sys::BO_DEV, data.len())?;
        let host = dev.heap_host(xdna_addr);
        // SAFETY: the BO is data.len() bytes of the heap mapping at host.
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), host, data.len()) };
        flush(host, data.len());
        Ok(Bo { dev: dev.clone(), handle, ptr: std::ptr::null_mut(), size: data.len(), xdna_addr })
    }

    /// A zeroed buffer of `bytes` bytes the NPU and the host share, usable
    /// as an argument of any kernel of this session.
    pub fn alloc(&self, bytes: usize) -> Result<Buffer, Error> {
        let dev = &self.inner;
        let berr = |what: &'static str| move |e: io::Error| Error::Buffer(format!("{what} ({bytes} bytes): {e}"));
        let (handle, map_offset, xdna_addr) = dev.create_bo(sys::BO_SHARE, bytes).map_err(berr("CREATE_BO(SHARE)"))?;
        let ptr = dev.mmap(map_offset, bytes).inspect_err(|_| dev.gem_close(handle)).map_err(berr("mapping"))?;
        // SAFETY: a fresh `bytes`-byte mapping.
        unsafe { std::ptr::write_bytes(ptr, 0, bytes) };
        flush(ptr, bytes);
        let bo = Arc::new(Bo { dev: dev.clone(), handle, ptr, size: bytes, xdna_addr });
        Ok(Buffer { bo, offset: 0, bytes, _not_sync: PhantomData })
    }

    /// [`alloc`](Self::alloc) sized for `n` elements of `T`.
    pub fn alloc_of<T: Copy>(&self, n: usize) -> Result<Buffer, Error> {
        self.alloc(n * std::mem::size_of::<T>())
    }
}

/// A command BO, mapped.
fn command_bo(dev: &Arc<DeviceInner>) -> io::Result<Bo> {
    let (handle, map_offset, _) = dev.create_bo(sys::BO_CMD, CMD_BYTES)?;
    let ptr = dev.mmap(map_offset, CMD_BYTES).inspect_err(|_| dev.gem_close(handle))?;
    Ok(Bo { dev: dev.clone(), handle, ptr, size: CMD_BYTES, xdna_addr: u64::MAX })
}

/// Maps `size` bytes of the BO at `map_offset` at a host address aligned to
/// `size`: reserve twice the room, map the BO over the aligned middle
/// (`MAP_FIXED` replaces the reservation atomically), release the rest.
fn map_aligned(dev: &DeviceInner, map_offset: u64, size: usize) -> io::Result<*mut u8> {
    let reserve = 2 * size;
    // SAFETY: an inaccessible anonymous reservation.
    let base = unsafe {
        sys::mmap(
            std::ptr::null_mut(),
            reserve,
            sys::PROT_NONE,
            sys::MAP_PRIVATE | sys::MAP_ANONYMOUS | sys::MAP_NORESERVE,
            -1,
            0,
        )
    };
    if base == sys::MAP_FAILED {
        return Err(io::Error::last_os_error());
    }
    let start = base as usize;
    let aligned = start.next_multiple_of(size);
    // SAFETY: [aligned, aligned+size) lies inside the reservation.
    let p = unsafe {
        sys::mmap(
            aligned as *mut c_void,
            size,
            sys::PROT_RW,
            sys::MAP_SHARED | sys::MAP_FIXED,
            dev.fd(),
            map_offset as i64,
        )
    };
    let err = (p == sys::MAP_FAILED).then(io::Error::last_os_error);
    // SAFETY: unmapping only the parts of the reservation not replaced
    // (all of it on failure).
    unsafe {
        if let Some(e) = err {
            sys::munmap(base, reserve);
            return Err(e);
        }
        if aligned > start {
            sys::munmap(base, aligned - start);
        }
        let end = aligned + size;
        if start + reserve > end {
            sys::munmap(end as *mut c_void, start + reserve - end);
        }
    }
    Ok(aligned as *mut u8)
}

const CMD_BYTES: usize = 4096;
/// `ERT_CMD_STATE_NEW` / `ERT_CMD_STATE_COMPLETED`.
const ERT_NEW: u32 = 1;
const ERT_COMPLETED: u32 = 4;
/// Argument buffers the `MLIR_AIE` kernel takes (`bo0`…`bo4`).
pub const MAX_ARGS: usize = 5;
/// How long a run may take before it is reported as hung.
const RUN_TIMEOUT: Duration = Duration::from_secs(10);

/// A hardware context: an xclbin's partition, configured with its PDI.
/// Destroyed when the last kernel on it is dropped.
struct Context {
    dev: Arc<DeviceInner>,
    handle: u32,
    syncobj: u32,
    /// Dropped after the context is destroyed (fields drop after `drop`).
    _pdi: Option<Bo>,
}

impl Drop for Context {
    fn drop(&mut self) {
        let _ = sys::drm(self.dev.fd(), sys::DESTROY_HWCTX, &mut sys::HandlePad { handle: self.handle, pad: 0 });
    }
}

/// A kernel: an instruction stream on a (possibly shared) hardware context.
pub struct Kernel {
    ctx: Arc<Context>,
    instr: Bo,
    ninstr_bytes: u32,
    /// Command BOs of finished runs, for the next ones: a run in flight
    /// owns its own, so several may be.
    idle: RefCell<Vec<Bo>>,
    _not_sync: PhantomData<*const ()>,
}

impl Kernel {
    /// Launches the kernel over `args` (its buffer arguments in order, at
    /// most [`MAX_ARGS`]) and waits for it; returns the time from
    /// submission to completion. Inputs must have been
    /// [`sync_to_device`](Buffer::sync_to_device)d, outputs need
    /// [`sync_from_device`](Buffer::sync_from_device) before reading.
    pub fn run(&self, args: &[&Buffer]) -> Result<Duration, Error> {
        self.start(args)?.wait()
    }

    /// Launches without waiting. The returned [`Run`] borrows the argument
    /// buffers until [`Run::wait`], as [`crate::Kernel::start`]'s does.
    pub fn start<'a>(&'a self, args: &[&'a Buffer]) -> Result<Run<'a>, Error> {
        if args.len() > MAX_ARGS {
            return Err(Error::Run(format!("{} buffer arguments; the kernel takes at most {MAX_ARGS}", args.len())));
        }
        let dev = &self.ctx.dev;
        let idle = self.idle.borrow_mut().pop();
        let cmd = match idle {
            Some(cmd) => cmd,
            None => command_bo(dev).map_err(|e| Error::Run(format!("command buffer: {e}")))?,
        };
        let packet = packet(self.instr.xdna_addr, self.ninstr_bytes, args.iter().map(|b| b.addr()));
        // SAFETY: the command BO is CMD_BYTES long and mapped; the packet is
        // far shorter.
        unsafe { std::ptr::copy_nonoverlapping(packet.as_ptr().cast::<u8>(), cmd.ptr, packet.len() * 4) };
        flush(cmd.ptr, packet.len() * 4);

        // The BOs the command references, for the driver to pin (a
        // sub-buffer's is its parent's), once each.
        let mut handles = vec![self.instr.handle];
        for b in args {
            if !handles.contains(&b.bo.handle) {
                handles.push(b.bo.handle);
            }
        }
        let mut exec = sys::ExecCmd {
            hwctx: self.ctx.handle,
            ty: 0, // AMDXDNA_CMD_SUBMIT_EXEC_BUF
            cmd_handles: cmd.handle as u64,
            args: handles.as_ptr() as u64,
            cmd_count: 1,
            arg_count: handles.len() as u32,
            ..Default::default()
        };
        let start = Instant::now();
        if let Err(e) = sys::drm(dev.fd(), sys::EXEC_CMD, &mut exec) {
            self.idle.borrow_mut().push(cmd);
            return Err(Error::Run(format!("EXEC_CMD: {e}")));
        }
        Ok(Run { kernel: self, cmd: Some(cmd), seq: exec.seq, start, _borrow: PhantomData })
    }

    /// Waits for fence point `seq` of the context, up to [`RUN_TIMEOUT`].
    fn wait(&self, seq: u64) -> io::Result<bool> {
        let (handle, point) = (self.ctx.syncobj, seq);
        // An absolute CLOCK_MONOTONIC deadline: a relative one is already in
        // the past, and the wait gives up before the NPU has started.
        let mut w = sys::SyncobjTimelineWait {
            handles: &handle as *const u32 as u64,
            points: &point as *const u64 as u64,
            timeout_nsec: monotonic_ns() + RUN_TIMEOUT.as_nanos() as i64,
            count_handles: 1,
            ..Default::default()
        };
        match sys::drm(self.ctx.dev.fd(), sys::SYNCOBJ_TIMELINE_WAIT, &mut w) {
            Ok(()) => Ok(true),
            Err(e) if e.raw_os_error() == Some(sys::ETIME) => Ok(false),
            Err(e) => Err(e),
        }
    }
}

/// An in-flight kernel launch; dropped without [`wait`](Self::wait), it
/// waits anyway (the array must not outlive the borrows).
pub struct Run<'a> {
    kernel: &'a Kernel,
    cmd: Option<Bo>,
    seq: u64,
    start: Instant,
    _borrow: PhantomData<&'a Buffer>,
}

impl Run<'_> {
    /// Blocks until the launch completes; returns its submission-to-completion time.
    pub fn wait(mut self) -> Result<Duration, Error> {
        self.wait_inner()
    }

    fn wait_inner(&mut self) -> Result<Duration, Error> {
        let Some(cmd) = self.cmd.take() else {
            return Ok(Duration::ZERO);
        };
        let signaled = self.kernel.wait(self.seq);
        let elapsed = self.start.elapsed();
        // The firmware writes the command's state into the packet header.
        flush(cmd.ptr, 4);
        // SAFETY: the first word of the mapped command BO.
        let state = unsafe { std::ptr::read_volatile(cmd.ptr as *const u32) } & 0xf;
        match signaled {
            Ok(true) => {
                self.kernel.idle.borrow_mut().push(cmd);
                if state == ERT_COMPLETED {
                    Ok(elapsed)
                } else {
                    Err(Error::Run(format!("the command finished in state {}", ert_state(state))))
                }
            }
            // The firmware may still hold the command: never reuse or free
            // its BO under it.
            Ok(false) => {
                std::mem::forget(cmd);
                Err(Error::Run(format!("the command did not finish within 10 s, in state {}", ert_state(state))))
            }
            Err(e) => {
                std::mem::forget(cmd);
                Err(Error::Run(format!("SYNCOBJ_TIMELINE_WAIT: {e}")))
            }
        }
    }
}

impl Drop for Run<'_> {
    fn drop(&mut self) {
        let _ = self.wait_inner();
    }
}

/// `clock_gettime(CLOCK_MONOTONIC)` in nanoseconds: the clock `Instant`
/// reads, whose raw value it does not expose.
fn monotonic_ns() -> i64 {
    #[repr(C)]
    struct Timespec {
        sec: i64,
        nsec: i64,
    }
    unsafe extern "C" {
        fn clock_gettime(clock: c_int, ts: *mut Timespec) -> c_int;
    }
    let mut ts = Timespec { sec: 0, nsec: 0 };
    // SAFETY: CLOCK_MONOTONIC (1) into a valid timespec.
    unsafe { clock_gettime(1, &mut ts) };
    ts.sec * 1_000_000_000 + ts.nsec
}

/// The `ERT_START_CU` packet for IRON's `MLIR_AIE` kernel: a header, the
/// CU mask, then the kernel's arguments at aiecc's offsets — `opcode` (3,
/// run the instruction stream) @0x00, `instr` @0x08, `ninstr` @0x10 (its
/// size in bytes, as IRON sets it), `bo0`…`bo4` @0x14… as addresses.
fn packet(instr: u64, ninstr_bytes: u32, bufs: impl Iterator<Item = u64>) -> [u32; 17] {
    let mut p = [0u32; 17];
    let count = (p.len() - 1) as u32;
    // ert_packet: state:4, custom:8, count:11, opcode:5 (0: START_CU), type:4 (3: CU).
    p[0] = ERT_NEW | (count << 12) | (3 << 28);
    p[1] = 1; // CU 0
    p[2] = 3;
    p[4] = instr as u32;
    p[5] = (instr >> 32) as u32;
    p[6] = ninstr_bytes;
    for (i, addr) in bufs.take(MAX_ARGS).enumerate() {
        p[7 + 2 * i] = addr as u32;
        p[8 + 2 * i] = (addr >> 32) as u32;
    }
    p
}

fn ert_state(state: u32) -> String {
    let name = match state {
        1 => "NEW",
        2 => "QUEUED",
        3 => "RUNNING",
        4 => "COMPLETED",
        5 => "ERROR",
        6 => "ABORT",
        7 => "SUBMITTED",
        8 => "TIMEOUT",
        9 => "NORESPONSE",
        _ => "?",
    };
    format!("{state} ({name})")
}

/// A buffer the host and the NPU share, filled and read in place — or a
/// view of part of one ([`sub`](Self::sub)).
pub struct Buffer {
    bo: Arc<Bo>,
    offset: usize,
    bytes: usize,
    _not_sync: PhantomData<*const ()>,
}

impl fmt::Debug for Buffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Buffer")
            .field("handle", &self.bo.handle)
            .field("offset", &self.offset)
            .field("bytes", &self.bytes)
            .finish()
    }
}

impl Buffer {
    pub fn len_bytes(&self) -> usize {
        self.bytes
    }

    /// Where the NPU finds it: its host address (shared virtual addressing).
    fn addr(&self) -> u64 {
        self.bo.ptr as u64 + self.offset as u64
    }

    fn host(&self) -> *mut u8 {
        // SAFETY: offset is within the mapping (checked by `sub`).
        unsafe { self.bo.ptr.add(self.offset) }
    }

    /// A view of `bytes` bytes of this buffer from `offset`, a [`Buffer`] in
    /// its own right: a kernel argument (the NPU is handed its address) with
    /// syncs that cover its bytes only. It keeps the allocation alive
    /// itself, as [`crate::Buffer::sub`]'s does.
    pub fn sub(&self, offset: usize, bytes: usize) -> Result<Buffer, Error> {
        if offset.checked_add(bytes).is_none_or(|end| end > self.bytes) {
            return Err(Error::Buffer(format!(
                "sub-buffer [{offset}, +{bytes}) outside a buffer of {} bytes",
                self.bytes
            )));
        }
        Ok(Buffer { bo: self.bo.clone(), offset: self.offset + offset, bytes, _not_sync: PhantomData })
    }

    /// [`sub`](Self::sub) in elements of `T`.
    pub fn sub_of<T: Copy>(&self, offset: usize, n: usize) -> Result<Buffer, Error> {
        let size = std::mem::size_of::<T>();
        self.sub(offset * size, n * size)
    }

    /// The buffer as `T`s (as many as fit). Read after
    /// [`sync_from_device`](Self::sync_from_device).
    pub fn as_slice<T: Copy>(&self) -> &[T] {
        // SAFETY: `bytes` bytes of a live mapping; a sub-buffer's offset is
        // the caller's to keep aligned for T, as with XRT's.
        unsafe { std::slice::from_raw_parts(self.host() as *const T, self.bytes / std::mem::size_of::<T>()) }
    }

    /// The buffer as mutable `T`s. Follow with [`sync_to_device`](Self::sync_to_device).
    pub fn as_mut_slice<T: Copy>(&mut self) -> &mut [T] {
        // SAFETY: as for as_slice; `&mut self` makes this the only view
        // through this Buffer (overlapping sub-buffers alias, as with XRT's).
        unsafe { std::slice::from_raw_parts_mut(self.host() as *mut T, self.bytes / std::mem::size_of::<T>()) }
    }

    pub fn sync_to_device(&self) -> Result<(), Error> {
        flush(self.host(), self.bytes);
        Ok(())
    }

    pub fn sync_from_device(&self) -> Result<(), Error> {
        flush(self.host(), self.bytes);
        Ok(())
    }

    /// Copies `data` in (which must fit) and syncs it to the device.
    pub fn write<T: Copy>(&mut self, data: &[T]) -> Result<(), Error> {
        let dst = self.as_mut_slice::<T>();
        if data.len() > dst.len() {
            return Err(Error::Buffer(format!("{} elements do not fit a buffer of {}", data.len(), dst.len())));
        }
        dst[..data.len()].copy_from_slice(data);
        self.sync_to_device()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_packet_is_what_xrt_sends() {
        // Captured from the XRT path (ElementwiseMul, three buffers).
        let p = packet(0x0402_8000, 0xcb0, [0x7b7d_41cd_f000u64, 0x7b7d_419b_f000, 0x7b7d_4169_f000].into_iter());
        assert_eq!(
            p,
            [
                0x3001_0001,
                1,
                3,
                0,
                0x0402_8000,
                0,
                0xcb0,
                0x41cd_f000,
                0x7b7d,
                0x419b_f000,
                0x7b7d,
                0x4169_f000,
                0x7b7d,
                0,
                0,
                0,
                0
            ]
        );
    }

    #[test]
    fn ioctl_numbers_match_the_uapi_header() {
        // DRM_IOCTL_AMDXDNA_EXEC_CMD and DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT.
        assert_eq!(sys::drm_iowr(sys::EXEC_CMD, std::mem::size_of::<sys::ExecCmd>()), 0xc038_6446);
        assert_eq!(
            sys::drm_iowr(sys::SYNCOBJ_TIMELINE_WAIT, std::mem::size_of::<sys::SyncobjTimelineWait>()),
            0xc030_64ca
        );
        assert_eq!(std::mem::size_of::<sys::AieMetadata>(), 64);
        assert_eq!(std::mem::size_of::<sys::ConfigCu>(), 16);
        assert_eq!(std::mem::size_of::<sys::HwctxEntry>(), 144);
    }

    #[test]
    fn a_non_xclbin_is_rejected() {
        assert!(axlf::partition(b"not an xclbin at all").is_err());
    }
}
