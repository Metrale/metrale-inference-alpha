// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Minimal CUDA driver-API FFI for metrale-storage: context and
//! stream, device and pinned host buffers, async copies, stream sync.
//!
//! Owner: metrale-storage.
//! Invariants:
//! - A non-zero driver status becomes an `Err` naming the call, except in the
//!   `Drop` impls, which ignore the status of the free or destroy call.

use anyhow::{Result, bail};
use std::ffi::c_void;

// 2026-09-25: Re-exported so callers reach the module and event helpers of
// `cuda_module.rs` through `cuda_min::` as well.
pub use crate::cuda_module::{CudaEvent, CudaModule, launch_kernel};

unsafe extern "C" {
    fn cuInit(flags: u32) -> i32;
    fn cuDeviceGet(device: *mut i32, ordinal: i32) -> i32;
    fn cuCtxCreate_v2(pctx: *mut u64, flags: u32, dev: i32) -> i32;
    fn cuCtxDestroy_v2(ctx: u64) -> i32;
    fn cuMemAlloc_v2(dptr: *mut u64, bytesize: usize) -> i32;
    fn cuMemFree_v2(dptr: u64) -> i32;
    fn cuMemAllocHost_v2(pp: *mut *mut c_void, bytesize: usize) -> i32;
    fn cuMemFreeHost(p: *mut c_void) -> i32;
    fn cuMemHostGetDevicePointer_v2(pdptr: *mut u64, p: *mut c_void, flags: u32) -> i32;
    fn cuMemcpyHtoDAsync_v2(dst: u64, src: *const c_void, bytes: usize, stream: u64) -> i32;
    fn cuMemcpyDtoHAsync_v2(dst: *mut c_void, src: u64, bytes: usize, stream: u64) -> i32;
    fn cuMemGetInfo_v2(free: *mut usize, total: *mut usize) -> i32;
    fn cuStreamCreate(phStream: *mut u64, flags: u32) -> i32;
    fn cuStreamDestroy_v2(stream: u64) -> i32;
    fn cuStreamSynchronize(stream: u64) -> i32;
}

/// 2026-09-25: The current context's free and total device memory in bytes
/// (`cuMemGetInfo`).
pub fn mem_info() -> Result<(usize, usize)> {
    let mut free = 0usize;
    let mut total = 0usize;
    let s = unsafe { cuMemGetInfo_v2(&mut free, &mut total) };
    if s != 0 {
        bail!("cuMemGetInfo_v2 failed: {s}");
    }
    Ok((free, total))
}

pub struct CudaCtx {
    pub ctx: u64,
    pub stream: u64,
}

impl CudaCtx {
    pub fn new(ordinal: i32) -> Result<Self> {
        unsafe {
            let s = cuInit(0);
            if s != 0 {
                bail!("cuInit failed: {s}");
            }
            let mut dev = 0i32;
            let s = cuDeviceGet(&mut dev, ordinal);
            if s != 0 {
                bail!("cuDeviceGet({ordinal}) failed: {s}");
            }
            let mut ctx = 0u64;
            let s = cuCtxCreate_v2(&mut ctx, 0, dev);
            if s != 0 {
                bail!("cuCtxCreate failed: {s}");
            }
            let mut stream = 0u64;
            let s = cuStreamCreate(&mut stream, 0);
            if s != 0 {
                cuCtxDestroy_v2(ctx);
                bail!("cuStreamCreate failed: {s}");
            }
            Ok(Self { ctx, stream })
        }
    }
}

impl Drop for CudaCtx {
    fn drop(&mut self) {
        unsafe {
            let _ = cuStreamDestroy_v2(self.stream);
            let _ = cuCtxDestroy_v2(self.ctx);
        }
    }
}

pub struct DeviceBuffer {
    pub ptr: u64,
    pub bytes: usize,
}

impl DeviceBuffer {
    pub fn new(bytes: usize) -> Result<Self> {
        let mut p = 0u64;
        let s = unsafe { cuMemAlloc_v2(&mut p, bytes) };
        if s != 0 {
            bail!("cuMemAlloc_v2({bytes}) failed: {s}");
        }
        Ok(Self { ptr: p, bytes })
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        unsafe {
            let _ = cuMemFree_v2(self.ptr);
        }
    }
}

pub struct PinnedBuffer {
    pub ptr: *mut c_void,
    pub bytes: usize,
}

// 2026-09-25: SAFETY: `cuMemAllocHost` memory keeps its address for the
// buffer's lifetime, so moving a `PinnedBuffer` moves only a pointer and a
// length. `ptr` is a public field: code that shares the memory across threads
// must order its own accesses.
unsafe impl Send for PinnedBuffer {}
unsafe impl Sync for PinnedBuffer {}

impl PinnedBuffer {
    pub fn new(bytes: usize) -> Result<Self> {
        let mut p: *mut c_void = std::ptr::null_mut();
        let s = unsafe { cuMemAllocHost_v2(&mut p, bytes) };
        if s != 0 {
            bail!("cuMemAllocHost_v2({bytes}) failed: {s}");
        }
        Ok(Self { ptr: p, bytes })
    }

    /// 2026-09-25: The device pointer for this pinned host allocation
    /// (`cuMemHostGetDevicePointer`, flags 0). `expert_arena.rs` uses it as the
    /// arena's device base.
    pub fn device_ptr(&self) -> Result<u64> {
        let mut dptr = 0u64;
        let s = unsafe { cuMemHostGetDevicePointer_v2(&mut dptr, self.ptr, 0) };
        if s != 0 {
            bail!("cuMemHostGetDevicePointer_v2 failed: {s}");
        }
        Ok(dptr)
    }
}

impl Drop for PinnedBuffer {
    fn drop(&mut self) {
        unsafe {
            let _ = cuMemFreeHost(self.ptr);
        }
    }
}

#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn copy_h_to_d_async(dst: u64, src: *const c_void, bytes: usize, stream: u64) -> Result<()> {
    let s = unsafe { cuMemcpyHtoDAsync_v2(dst, src, bytes, stream) };
    if s != 0 {
        bail!("cuMemcpyHtoDAsync_v2 failed: {s}");
    }
    Ok(())
}

#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn copy_d_to_h_async(dst: *mut c_void, src: u64, bytes: usize, stream: u64) -> Result<()> {
    let s = unsafe { cuMemcpyDtoHAsync_v2(dst, src, bytes, stream) };
    if s != 0 {
        bail!("cuMemcpyDtoHAsync_v2 failed: {s}");
    }
    Ok(())
}

pub fn stream_sync(stream: u64) -> Result<()> {
    let s = unsafe { cuStreamSynchronize(stream) };
    if s != 0 {
        bail!("cuStreamSynchronize failed: {s}");
    }
    Ok(())
}
