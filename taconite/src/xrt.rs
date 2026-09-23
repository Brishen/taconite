// SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
// SPDX-License-Identifier: Apache-2.0

//! The XRT path (feature `xrt`): [`Session`], [`Kernel`], [`Run`] and
//! [`Buffer`] over the C shim `build.rs` compiles against XRT's C++ API.
//! Re-exported from the crate root.

use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::marker::PhantomData;
use std::path::Path;
use std::ptr::NonNull;
use std::sync::Arc;
use std::time::Duration;

use crate::Error;

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
