// SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
// SPDX-License-Identifier: Apache-2.0

//! Replaying [IRON](https://github.com/amd/iron) kernels on an AMD XDNA NPU
//! from Rust, through XRT, with no Python at run time.
//!
//! IRON is an ahead-of-time compiler: a kernel is an `.xclbin` (the array
//! configuration and the cores' programs) plus an instruction stream for the
//! shim DMAs (`*.insts.bin`). Running one is "allocate buffer objects, fill
//! the inputs, submit `(opcode 3, instructions, byte count, buffers…)`,
//! wait" — the XRT C++ API, wrapped here by a small C shim
//! (`iron_xrt_shim.cpp`) compiled by `build.rs`. Nothing else of IRON is
//! needed once the kernels exist.
//!
//! Three types: a [`Session`] (the device), [`Kernel`]s loaded into it (each
//! a *resident* hardware context — NPU2 allows 16 at once, and switching
//! between two costs ~2 ms of NPU time, so a caller wants few distinct
//! kernels called many times), and [`Buffer`]s — host-visible buffer objects
//! the caller writes into *directly* through [`Buffer::as_mut_slice`] and
//! syncs around a run, so activations never take an extra copy on the way
//! to or from the array.
//!
//! The `direct` module runs them with no XRT at all, through the `amdxdna`
//! driver's ioctls. The [`compile`] module builds those kernels, too — Peano and the native
//! `aiecc` as subprocesses, still no Python — from a design's MLIR.
//!
//! Everything returns [`Result`]; the shim never prints or aborts. The
//! XRT objects may move between threads but not be shared by them, so a
//! `Session` and what it owns are `Send` and not `Sync`: a model loads on
//! one thread and runs on a worker, behind a `Mutex` if it is shared.

use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::fmt;
use std::marker::PhantomData;
use std::path::Path;
use std::ptr::NonNull;
use std::sync::Arc;
use std::time::Duration;

pub mod compile;
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
pub mod direct;

mod ffi {
    use super::*;

    #[repr(C)]
    pub struct iron_session {
        _p: [u8; 0],
    }
    #[repr(C)]
    pub struct iron_kernel {
        _p: [u8; 0],
    }
    #[repr(C)]
    pub struct iron_buffer {
        _p: [u8; 0],
    }
    #[repr(C)]
    pub struct iron_run {
        _p: [u8; 0],
    }

    unsafe extern "C" {
        pub fn iron_open(device_index: c_int, err: *mut c_char, err_len: usize) -> *mut iron_session;
        pub fn iron_close(s: *mut iron_session);
        pub fn iron_kernel_load(
            s: *mut iron_session,
            xclbin: *const c_char,
            insts: *const c_char,
            kernel_name: *const c_char,
            gops: u32,
            err: *mut c_char,
            err_len: usize,
        ) -> *mut iron_kernel;
        pub fn iron_kernel_free(k: *mut iron_kernel);
        pub fn iron_buffer_alloc(s: *mut iron_session, bytes: usize, err: *mut c_char, err_len: usize) -> *mut iron_buffer;
        pub fn iron_buffer_map(b: *mut iron_buffer) -> *mut c_void;
        pub fn iron_buffer_sync_to_device(b: *mut iron_buffer, err: *mut c_char, err_len: usize) -> c_int;
        pub fn iron_buffer_sync_from_device(b: *mut iron_buffer, err: *mut c_char, err_len: usize) -> c_int;
        pub fn iron_buffer_free(b: *mut iron_buffer);
        pub fn iron_buffer_sub(
            parent: *mut iron_buffer,
            offset: usize,
            bytes: usize,
            err: *mut c_char,
            err_len: usize,
        ) -> *mut iron_buffer;
        pub fn iron_kernel_run(
            k: *mut iron_kernel,
            bufs: *mut *mut iron_buffer,
            n: usize,
            elapsed_ns: *mut u64,
            err: *mut c_char,
            err_len: usize,
        ) -> c_int;
        pub fn iron_kernel_start(
            k: *mut iron_kernel,
            bufs: *mut *mut iron_buffer,
            n: usize,
            err: *mut c_char,
            err_len: usize,
        ) -> *mut iron_run;
        pub fn iron_run_wait(r: *mut iron_run, elapsed_ns: *mut u64, err: *mut c_char, err_len: usize) -> c_int;
    }
}

/// What went wrong, in XRT's words where it has any.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// The device could not be opened (no `/dev/accel/accel0`, no driver,
    /// no permission, or the XRT libraries are missing).
    Device(String),
    /// An xclbin / instruction stream could not be loaded into a context.
    Kernel(String),
    /// A buffer object could not be allocated or synced.
    Buffer(String),
    /// A launched kernel failed or did not complete.
    Run(String),
    /// A path contained an interior NUL.
    Path(String),
    /// A kernel or design failed to build (see [`compile`]).
    Compile(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Device(m) => write!(f, "NPU device: {m}"),
            Error::Kernel(m) => write!(f, "NPU kernel: {m}"),
            Error::Buffer(m) => write!(f, "NPU buffer: {m}"),
            Error::Run(m) => write!(f, "NPU run: {m}"),
            Error::Path(m) => write!(f, "path: {m}"),
            Error::Compile(m) => write!(f, "compile: {m}"),
        }
    }
}

impl std::error::Error for Error {}

const ERR_LEN: usize = 1024;

fn err_buf() -> Vec<c_char> {
    vec![0; ERR_LEN]
}

fn err_string(buf: &[c_char]) -> String {
    // SAFETY: the shim always NUL-terminates within ERR_LEN.
    unsafe { CStr::from_ptr(buf.as_ptr()) }.to_string_lossy().into_owned()
}

fn c_path(p: &Path) -> Result<CString, Error> {
    CString::new(p.to_string_lossy().as_bytes()).map_err(|_| Error::Path(format!("{} contains a NUL byte", p.display())))
}

struct SessionInner {
    raw: NonNull<ffi::iron_session>,
}

impl Drop for SessionInner {
    fn drop(&mut self) {
        // SAFETY: created by iron_open, freed once.
        unsafe { ffi::iron_close(self.raw.as_ptr()) }
    }
}

/// An open NPU. Kernels and buffers hold a reference to it, so it lives as
/// long as anything created from it.
#[derive(Clone)]
pub struct Session {
    inner: Arc<SessionInner>,
    _not_sync: PhantomData<*const ()>,
}

// SAFETY: XRT's device/context/bo/kernel objects are plain handles that may
// be used from any thread, one at a time. `PhantomData<*const ()>` keeps
// every type here `!Sync`, so "one at a time" is enforced by the borrow
// checker; `Arc` (not `Rc`) makes the shared session refcount sound across
// the move.
unsafe impl Send for Session {}
unsafe impl Send for Kernel {}
unsafe impl Send for Buffer {}

impl Session {
    /// Opens NPU `device_index` (0 on a laptop).
    #[expect(
        clippy::arc_with_non_send_sync,
        reason = "the handles are Send (see their impls), so the refcount they share must be atomic"
    )]
    pub fn open(device_index: u32) -> Result<Self, Error> {
        let mut err = err_buf();
        // SAFETY: err is ERR_LEN bytes; the shim writes within it.
        let raw = unsafe { ffi::iron_open(device_index as c_int, err.as_mut_ptr(), ERR_LEN) };
        let raw = NonNull::new(raw).ok_or_else(|| Error::Device(err_string(&err)))?;
        Ok(Self { inner: Arc::new(SessionInner { raw }), _not_sync: PhantomData })
    }

    fn raw(&self) -> *mut ffi::iron_session {
        self.inner.raw.as_ptr()
    }

    /// Loads an xclbin + instruction stream. Kernels naming the same xclbin
    /// (and kernel name) share one hardware context, resident until the last
    /// of them is dropped -- dropping them frees the context for another
    /// (NPU2 holds 16, across every process). `kernel_name` is the kernel
    /// inside the xclbin (`None`: its first — IRON's is `MLIR_AIE`).
    ///
    /// `ops_per_run` is the work one [`Kernel::run`] performs — `2·M·K·N`
    /// for a GEMM, a multiply-add counting two — declared to the driver as
    /// the context's QoS `gops` (rounded to whole GOP, the field's unit; a
    /// kernel under half a GOP declares nothing). It is how `xrt-smi` and
    /// the desktop's NPU ACTIVITY card turn the context's completion count
    /// into operations per second; see `iron_xrt_shim.h` for why it can't
    /// change how the context is scheduled.
    pub fn load_kernel(
        &self,
        xclbin: &Path,
        insts: &Path,
        kernel_name: Option<&str>,
        ops_per_run: u64,
    ) -> Result<Kernel, Error> {
        let gops = u32::try_from((ops_per_run as f64 / 1e9).round() as u64).unwrap_or(u32::MAX);
        let xclbin_c = c_path(xclbin)?;
        let insts_c = c_path(insts)?;
        let name_c = kernel_name.map(|n| CString::new(n).map_err(|_| Error::Path(n.into()))).transpose()?;
        let mut err = err_buf();
        // SAFETY: all pointers are valid for the call; err is ERR_LEN bytes.
        let raw = unsafe {
            ffi::iron_kernel_load(
                self.raw(),
                xclbin_c.as_ptr(),
                insts_c.as_ptr(),
                name_c.as_ref().map_or(std::ptr::null(), |n| n.as_ptr()),
                gops,
                err.as_mut_ptr(),
                ERR_LEN,
            )
        };
        let raw = NonNull::new(raw).ok_or_else(|| Error::Kernel(err_string(&err)))?;
        Ok(Kernel { raw, _session: self.clone(), _not_sync: PhantomData })
    }

    /// A zeroed host-visible buffer object of `bytes` bytes, usable as an
    /// argument of any kernel of this session.
    pub fn alloc(&self, bytes: usize) -> Result<Buffer, Error> {
        let mut err = err_buf();
        // SAFETY: err is ERR_LEN bytes.
        let raw = unsafe { ffi::iron_buffer_alloc(self.raw(), bytes, err.as_mut_ptr(), ERR_LEN) };
        let raw = NonNull::new(raw).ok_or_else(|| Error::Buffer(err_string(&err)))?;
        Ok(Buffer { raw, bytes, _session: self.clone(), _not_sync: PhantomData })
    }

    /// [`alloc`](Self::alloc) sized for `n` elements of `T`.
    pub fn alloc_of<T: Copy>(&self, n: usize) -> Result<Buffer, Error> {
        self.alloc(n * std::mem::size_of::<T>())
    }
}

/// A compiled kernel on its (possibly shared) resident hardware context.
pub struct Kernel {
    raw: NonNull<ffi::iron_kernel>,
    _session: Session,
    _not_sync: PhantomData<*const ()>,
}

impl Drop for Kernel {
    fn drop(&mut self) {
        // SAFETY: created by iron_kernel_load, freed once.
        unsafe { ffi::iron_kernel_free(self.raw.as_ptr()) }
    }
}

impl Kernel {
    /// Launches the kernel over `args` (the kernel's buffer arguments in
    /// order) and waits for it. Inputs must have been
    /// [`sync_to_device`](Buffer::sync_to_device)d; outputs need a
    /// [`sync_from_device`](Buffer::sync_from_device) before reading.
    /// Returns the wall time of launch + wait.
    pub fn run(&self, args: &[&Buffer]) -> Result<Duration, Error> {
        let mut raw: Vec<*mut ffi::iron_buffer> = args.iter().map(|b| b.raw.as_ptr()).collect();
        let mut elapsed_ns: u64 = 0;
        let mut err = err_buf();
        // SAFETY: raw holds valid buffer pointers for the call; err is ERR_LEN bytes.
        let rc = unsafe {
            ffi::iron_kernel_run(self.raw.as_ptr(), raw.as_mut_ptr(), raw.len(), &mut elapsed_ns, err.as_mut_ptr(), ERR_LEN)
        };
        if rc != 0 {
            return Err(Error::Run(err_string(&err)));
        }
        Ok(Duration::from_nanos(elapsed_ns))
    }

    /// Launches without waiting. The returned [`Run`] borrows the argument
    /// buffers until [`Run::wait`], so the host can't touch what the array
    /// is reading or writing meanwhile — but can work on any other buffer.
    pub fn start<'a>(&'a self, args: &[&'a Buffer]) -> Result<Run<'a>, Error> {
        let mut raw: Vec<*mut ffi::iron_buffer> = args.iter().map(|b| b.raw.as_ptr()).collect();
        let mut err = err_buf();
        // SAFETY: raw holds valid buffer pointers for the call; err is ERR_LEN bytes.
        let r = unsafe { ffi::iron_kernel_start(self.raw.as_ptr(), raw.as_mut_ptr(), raw.len(), err.as_mut_ptr(), ERR_LEN) };
        let raw = NonNull::new(r).ok_or_else(|| Error::Run(err_string(&err)))?;
        Ok(Run { raw: Some(raw), _borrow: PhantomData })
    }
}

/// An in-flight kernel launch; dropped without [`wait`](Self::wait), it
/// waits anyway (the array must not outlive the borrows).
pub struct Run<'a> {
    raw: Option<NonNull<ffi::iron_run>>,
    _borrow: PhantomData<&'a Buffer>,
}

impl<'a> Run<'a> {
    /// Blocks until the launch completes; returns its launch-to-completion time.
    pub fn wait(mut self) -> Result<Duration, Error> {
        self.wait_inner()
    }

    fn wait_inner(&mut self) -> Result<Duration, Error> {
        let Some(raw) = self.raw.take() else {
            return Ok(Duration::ZERO);
        };
        let mut elapsed_ns: u64 = 0;
        let mut err = err_buf();
        // SAFETY: raw came from iron_kernel_start and is waited/freed exactly once.
        let rc = unsafe { ffi::iron_run_wait(raw.as_ptr(), &mut elapsed_ns, err.as_mut_ptr(), ERR_LEN) };
        if rc != 0 {
            return Err(Error::Run(err_string(&err)));
        }
        Ok(Duration::from_nanos(elapsed_ns))
    }
}

impl Drop for Run<'_> {
    fn drop(&mut self) {
        let _ = self.wait_inner();
    }
}

/// A host-visible buffer object: the host reads and writes it in place
/// through [`as_slice`](Self::as_slice) / [`as_mut_slice`](Self::as_mut_slice)
/// and syncs the caches around a kernel run.
pub struct Buffer {
    raw: NonNull<ffi::iron_buffer>,
    bytes: usize,
    _session: Session,
    _not_sync: PhantomData<*const ()>,
}

impl Drop for Buffer {
    fn drop(&mut self) {
        // SAFETY: created by iron_buffer_alloc or iron_buffer_sub, freed once.
        unsafe { ffi::iron_buffer_free(self.raw.as_ptr()) }
    }
}

impl Buffer {
    pub fn len_bytes(&self) -> usize {
        self.bytes
    }

    /// A view of `bytes` bytes of this buffer from `offset`, a [`Buffer`]
    /// in its own right: a kernel argument like any other (an XRT
    /// sub-buffer — the same allocation, its device address `offset`
    /// further on), with its own identity for the run cache and syncs that
    /// cover its bytes only. It holds the allocation alive itself, so it
    /// may outlive the buffer it was taken from. What the tagger uses to
    /// hand a kernel one matrix of a GEMM output that holds several.
    pub fn sub(&self, offset: usize, bytes: usize) -> Result<Buffer, Error> {
        let mut err = err_buf();
        // SAFETY: raw is a live buffer; err is ERR_LEN bytes.
        let raw = unsafe { ffi::iron_buffer_sub(self.raw.as_ptr(), offset, bytes, err.as_mut_ptr(), ERR_LEN) };
        let raw = NonNull::new(raw).ok_or_else(|| Error::Buffer(err_string(&err)))?;
        Ok(Buffer { raw, bytes, _session: self._session.clone(), _not_sync: PhantomData })
    }

    /// [`sub`](Self::sub) in elements of `T`.
    pub fn sub_of<T: Copy>(&self, offset: usize, n: usize) -> Result<Buffer, Error> {
        let size = std::mem::size_of::<T>();
        self.sub(offset * size, n * size)
    }

    fn map(&self) -> *mut u8 {
        // SAFETY: the shim returns the BO's host mapping, valid for the buffer's life.
        unsafe { ffi::iron_buffer_map(self.raw.as_ptr()) as *mut u8 }
    }

    /// The buffer as `T`s (as many as fit). Read after
    /// [`sync_from_device`](Self::sync_from_device).
    pub fn as_slice<T: Copy>(&self) -> &[T] {
        let n = self.bytes / std::mem::size_of::<T>();
        // SAFETY: the mapping is `bytes` long and page-aligned (XRT maps BOs
        // with mmap), so it is aligned for any T of that size or smaller;
        // the shim keeps it valid for the buffer's life.
        unsafe { std::slice::from_raw_parts(self.map() as *const T, n) }
    }

    /// The buffer as mutable `T`s. Follow with [`sync_to_device`](Self::sync_to_device).
    pub fn as_mut_slice<T: Copy>(&mut self) -> &mut [T] {
        let n = self.bytes / std::mem::size_of::<T>();
        // SAFETY: as for `as_slice`, plus `&mut self` makes this the only
        // reference into the mapping.
        unsafe { std::slice::from_raw_parts_mut(self.map() as *mut T, n) }
    }

    pub fn sync_to_device(&self) -> Result<(), Error> {
        let mut err = err_buf();
        // SAFETY: err is ERR_LEN bytes.
        if unsafe { ffi::iron_buffer_sync_to_device(self.raw.as_ptr(), err.as_mut_ptr(), ERR_LEN) } != 0 {
            return Err(Error::Buffer(err_string(&err)));
        }
        Ok(())
    }

    pub fn sync_from_device(&self) -> Result<(), Error> {
        let mut err = err_buf();
        // SAFETY: err is ERR_LEN bytes.
        if unsafe { ffi::iron_buffer_sync_from_device(self.raw.as_ptr(), err.as_mut_ptr(), ERR_LEN) } != 0 {
            return Err(Error::Buffer(err_string(&err)));
        }
        Ok(())
    }

    /// Copies `data` in (which must fit) and syncs it to the device.
    pub fn write<T: Copy>(&mut self, data: &[T]) -> Result<(), Error> {
        let dst = self.as_mut_slice::<T>();
        if data.len() > dst.len() {
            return Err(Error::Buffer(format!(
                "{} elements do not fit a buffer of {}",
                data.len(),
                dst.len()
            )));
        }
        dst[..data.len()].copy_from_slice(data);
        self.sync_to_device()
    }
}

/// Where the XDNA driver exposes the first NPU; checked before opening so a
/// missing driver is a clean error rather than an XRT exception string.
pub const DEVICE_NODE: &str = "/dev/accel/accel0";

/// Whether an NPU device node exists on this machine.
pub fn npu_present() -> bool {
    Path::new(DEVICE_NODE).exists()
}

/// `bf16` as bits: the storage type of IRON's kernels. Round-to-nearest-even
/// from `f32`, the conversion ml_dtypes / torch use, so a host-side value
/// matches what the exporter wrote.
#[inline]
pub fn f32_to_bf16(x: f32) -> u16 {
    // Branchless so the glue loops vectorise: round-to-nearest-even on the
    // dropped 16 bits; a NaN keeps a set mantissa bit so it can't round to
    // an infinity.
    let bits = x.to_bits();
    let is_nan = x.is_nan() as u32;
    let round = 0x7fff + ((bits >> 16) & 1);
    let rounded = bits.wrapping_add(round) >> 16;
    let nan_val = (bits >> 16) | 0x40;
    (rounded * (1 - is_nan) + nan_val * is_nan) as u16
}

/// `e^x` to ~1e-5 relative error over ±80 (the f32 rounding of `x·log2e`
/// dominates; the polynomial itself is good to 2e-7), clamped to ±87,
/// without libm: `2^(x log2 e)` split into an integer exponent and a
/// degree-6 polynomial on the fraction. For values that end up in bf16
/// (8-bit mantissa, 4e-3) this is indistinguishable from `expf`, and the
/// loop it sits in vectorises; `f32::exp` was the tagger glue's single
/// largest cost.
#[inline]
pub fn fast_exp(x: f32) -> f32 {
    let x = x.clamp(-87.0, 87.0);
    let t = x * std::f32::consts::LOG2_E;
    let n = t.floor();
    let f = t - n;
    // 2^f on [0, 1): Cephes' exp2f polynomial (max rel err ~2e-7).
    let p = 1.0
        + f * (6.931_472e-1
            + f * (2.402_264_8e-1
                + f * (5.550_332_5e-2 + f * (9.618_438e-3 + f * (1.339_887_4e-3 + f * 1.535_336_2e-4)))));
    f32::from_bits(p.to_bits().wrapping_add((n as i32 as u32) << 23))
}

#[inline]
pub fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_round_trips_and_rounds_to_nearest_even() {
        for &x in &[0.0f32, 1.0, -2.5, std::f32::consts::PI, 1e-3, 65504.0, -1e30] {
            let b = f32_to_bf16(x);
            let back = bf16_to_f32(b);
            assert!((back - x).abs() <= x.abs() * (1.0 / 128.0) + 1e-30, "{x} -> {back}");
        }
        // 1 + 2^-8 is exactly halfway between 1.0 and 1.0078125 in bf16: ties to even -> 1.0
        assert_eq!(bf16_to_f32(f32_to_bf16(1.0 + 1.0 / 256.0)), 1.0);
        // 1 + 3*2^-8 is halfway between 1.0078125 and 1.015625: ties to even -> 1.015625
        assert_eq!(bf16_to_f32(f32_to_bf16(1.0 + 3.0 / 256.0)), 1.015625);
        assert!(bf16_to_f32(f32_to_bf16(f32::NAN)).is_nan());
    }

    #[test]
    fn fast_exp_is_exact_at_bf16_precision() {
        let mut worst = 0f32;
        let mut x = -80.0f32;
        while x < 80.0 {
            let rel = ((fast_exp(x) - x.exp()) / x.exp()).abs();
            worst = worst.max(rel);
            x += 0.0137;
        }
        assert!(worst < 2e-5, "worst relative error {worst}");
        assert_eq!(fast_exp(0.0), 1.0);
        assert!(fast_exp(-100.0) < 1e-37);
    }

    #[test]
    fn opening_a_missing_device_is_an_error_not_a_crash() {
        if npu_present() {
            return; // covered by the tagger's live test on an NPU box
        }
        assert!(matches!(Session::open(0), Err(Error::Device(_))));
    }
}
