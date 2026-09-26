// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `MockGpuBackend`, a `GpuBackend` over host memory for tests: it
//! moves real bytes, records launches and lookups, and counts copies and syncs.
//!
//! Owner: gpu-runtime.
//! Invariants:
//! - Every copy and memset resolves its pointers to live allocations, interior
//!   pointers included, and errors when one has none.
//! - `free` takes only an allocation's base pointer.

use super::*;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

#[path = "mock_counters.rs"]
mod mock_counters;

#[derive(Debug)]
pub struct MockAlloc {
    pub bytes: usize,
    pub data: Vec<u8>,
}

/// 2026-09-25: Records kernel launches and memory operations for test assertions.
pub struct MockGpuBackend {
    op_cache: crate::op_cache::OpCache,
    allocs: Mutex<HashMap<u64, MockAlloc>>,
    next_ptr: Mutex<u64>,
    max_allocation_bytes: AtomicUsize,
    launches: Mutex<Vec<MockLaunch>>,
    kernel_lookups: Mutex<Vec<(String, String)>>,
    /// 2026-09-25: Modules a test declares absent; every other module is present.
    absent_modules: Mutex<std::collections::HashSet<String>>,
    /// 2026-09-25: `kernel(module, func)` returns `Err` for these pairs.
    denied_kernels: Mutex<Vec<(String, String)>>,
    /// 2026-09-25: Copy and sync counters, so a test can assert the shape of a
    /// bulk transfer as well as its bytes.
    syncs: AtomicUsize,
    d2h_blocking: AtomicUsize,
    d2h_async: AtomicUsize,
    d2h_async_streams: Mutex<Vec<u64>>,
    /// 2026-09-25: `(stream, copy_d2h_async count so far)` at every
    /// `synchronize` call.
    sync_d2h_async_counts: Mutex<Vec<(u64, usize)>>,
    /// 2026-09-25: `copy_d2d` and `copy_d2d_async` calls, in one counter.
    d2d: AtomicUsize,
    /// 2026-09-25: `copy_d2d_2d_async` calls, which the CUDA backend issues as
    /// one `cudaMemcpy2DAsync` whatever `height` is. Counted apart from `d2d`.
    d2d_2d: AtomicUsize,
    /// 2026-09-25: Streams passed to `copy_d2d_async`, in call order, so a test
    /// can check a copy went to the right stream.
    d2d_async_streams: Mutex<Vec<u64>>,
    d2d_2d_async_streams: Mutex<Vec<u64>>,
    host_pinned_allocs: AtomicUsize,
    /// 2026-09-25: `copy_h2d` calls and their total bytes.
    h2d: AtomicUsize,
    h2d_bytes: AtomicUsize,
}

#[derive(Debug, Clone)]
pub struct MockLaunch {
    pub func: u64,
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub shared_mem: u32,
    pub stream: u64,
    pub args: Vec<MockArg>,
}

/// 2026-09-25: An owned copy of a typed kernel argument, as `launch_typed` got it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MockArg {
    Buffer(DevicePtr),
    Bytes(Vec<u8>),
}

impl Default for MockGpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl MockGpuBackend {
    pub fn new() -> Self {
        Self {
            op_cache: crate::op_cache::OpCache::new(),
            allocs: Mutex::new(HashMap::new()),
            next_ptr: Mutex::new(0x1000_0000),
            max_allocation_bytes: AtomicUsize::new(usize::MAX),
            launches: Mutex::new(Vec::new()),
            kernel_lookups: Mutex::new(Vec::new()),
            absent_modules: Mutex::new(std::collections::HashSet::new()),
            denied_kernels: Mutex::new(Vec::new()),
            syncs: AtomicUsize::new(0),
            d2h_blocking: AtomicUsize::new(0),
            d2h_async: AtomicUsize::new(0),
            d2h_async_streams: Mutex::new(Vec::new()),
            sync_d2h_async_counts: Mutex::new(Vec::new()),
            d2d: AtomicUsize::new(0),
            d2d_2d: AtomicUsize::new(0),
            d2d_async_streams: Mutex::new(Vec::new()),
            d2d_2d_async_streams: Mutex::new(Vec::new()),
            host_pinned_allocs: AtomicUsize::new(0),
            h2d: AtomicUsize::new(0),
            h2d_bytes: AtomicUsize::new(0),
        }
    }

    pub fn read_alloc(&self, ptr: DevicePtr) -> Option<Vec<u8>> {
        self.allocs.lock().get(&ptr.0).map(|a| a.data.clone())
    }

    /// 2026-09-25: Copy `bytes` from `src` to `dst` in the simulated device
    /// memory. Errors when either range is unallocated or runs past its
    /// allocation. The source is staged through a temporary, so `src` and `dst`
    /// may share an allocation.
    fn blit(&self, src: DevicePtr, dst: DevicePtr, bytes: usize) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        let mut allocs = self.allocs.lock();
        let staged = {
            let (offset, alloc) = find_alloc(&allocs, src)
                .ok_or_else(|| anyhow::anyhow!("copy_d2d: src {src} not allocated"))?;
            if offset + bytes > alloc.bytes {
                anyhow::bail!("copy_d2d: src {src} + {bytes} overruns its allocation");
            }
            alloc.data[offset..offset + bytes].to_vec()
        };
        let (offset, alloc) = find_alloc_mut(&mut allocs, dst)
            .ok_or_else(|| anyhow::anyhow!("copy_d2d: dst {dst} not allocated"))?;
        if offset + bytes > alloc.bytes {
            anyhow::bail!("copy_d2d: dst {dst} + {bytes} overruns its allocation");
        }
        alloc.data[offset..offset + bytes].copy_from_slice(&staged);
        Ok(())
    }

    /// 2026-09-25: Every launch recorded so far, in dispatch order. `kernel()`
    /// returns the same handle for every function, so geometry and arguments are
    /// what tell launches apart.
    pub fn launches_snapshot(&self) -> Vec<MockLaunch> {
        self.launches.lock().clone()
    }

    /// 2026-09-25: Declare a module absent: `has_module` then answers false.
    /// `kernel` still resolves lookups against it.
    pub fn mark_module_absent(&self, module: &str) {
        self.absent_modules.lock().insert(module.to_owned());
    }

    /// 2026-09-25: Module/function pairs requested through `kernel`, in lookup
    /// order.
    pub fn kernel_lookups_snapshot(&self) -> Vec<(String, String)> {
        self.kernel_lookups.lock().clone()
    }

    /// 2026-09-25: Every later `kernel(module, func)` for this pair fails; the
    /// lookup is still recorded.
    pub fn deny_kernel(&self, module: &str, func_name: &str) {
        self.denied_kernels
            .lock()
            .push((module.to_owned(), func_name.to_owned()));
    }
}

/// 2026-09-25: The allocation containing `ptr`, and `ptr`'s offset in it.
fn find_alloc(allocs: &HashMap<u64, MockAlloc>, ptr: DevicePtr) -> Option<(usize, &MockAlloc)> {
    for (&base, alloc) in allocs.iter() {
        if ptr.0 >= base && ptr.0 < base + alloc.bytes as u64 {
            return Some(((ptr.0 - base) as usize, alloc));
        }
    }
    None
}

fn find_alloc_mut(
    allocs: &mut HashMap<u64, MockAlloc>,
    ptr: DevicePtr,
) -> Option<(usize, &mut MockAlloc)> {
    for (&base, alloc) in allocs.iter_mut() {
        if ptr.0 >= base && ptr.0 < base + alloc.bytes as u64 {
            return Some(((ptr.0 - base) as usize, alloc));
        }
    }
    None
}

impl GpuBackend for MockGpuBackend {
    fn op_cache(&self) -> &crate::op_cache::OpCache {
        &self.op_cache
    }

    fn alloc(&self, bytes: usize) -> Result<DevicePtr> {
        let limit = self.max_allocation_bytes.load(Ordering::Relaxed);
        if bytes > limit {
            anyhow::bail!("alloc: requested {bytes} bytes exceeds mock limit {limit}");
        }
        let mut next = self.next_ptr.lock();
        let ptr = *next;
        *next += bytes as u64;
        *next = (*next + 255) & !255;
        self.allocs.lock().insert(
            ptr,
            MockAlloc {
                bytes,
                data: vec![0u8; bytes],
            },
        );
        Ok(DevicePtr(ptr))
    }

    fn alloc_managed(&self, bytes: usize) -> Result<DevicePtr> {
        self.alloc(bytes)
    }

    fn free(&self, ptr: DevicePtr) -> Result<()> {
        if ptr.is_null() {
            return Ok(());
        }
        if self.allocs.lock().remove(&ptr.0).is_none() {
            anyhow::bail!("free: ptr {ptr} is not an allocation base or is already free");
        }
        Ok(())
    }

    fn copy_h2d(&self, src: &[u8], dst: DevicePtr) -> Result<()> {
        self.h2d.fetch_add(1, Ordering::Relaxed);
        self.h2d_bytes.fetch_add(src.len(), Ordering::Relaxed);
        let mut allocs = self.allocs.lock();
        let (offset, alloc) = find_alloc_mut(&mut allocs, dst)
            .ok_or_else(|| anyhow::anyhow!("copy_h2d: ptr {dst} not allocated"))?;
        alloc.data[offset..offset + src.len()].copy_from_slice(src);
        Ok(())
    }

    fn copy_d2h(&self, src: DevicePtr, dst: &mut [u8]) -> Result<()> {
        self.d2h_blocking.fetch_add(1, Ordering::Relaxed);
        let allocs = self.allocs.lock();
        let (offset, alloc) = find_alloc(&allocs, src)
            .ok_or_else(|| anyhow::anyhow!("copy_d2h: ptr {src} not allocated"))?;
        dst.copy_from_slice(&alloc.data[offset..offset + dst.len()]);
        Ok(())
    }

    fn copy_d2h_async(&self, src: DevicePtr, dst: &mut [u8], stream: u64) -> Result<()> {
        // 2026-09-25: Counted apart from `copy_d2h`; the trait default forwards
        // to it, which would count this as a blocking copy.
        self.d2h_async.fetch_add(1, Ordering::Relaxed);
        self.d2h_async_streams.lock().push(stream);
        let allocs = self.allocs.lock();
        let (offset, alloc) = find_alloc(&allocs, src)
            .ok_or_else(|| anyhow::anyhow!("copy_d2h_async: ptr {src} not allocated"))?;
        dst.copy_from_slice(&alloc.data[offset..offset + dst.len()]);
        Ok(())
    }

    fn copy_d2d(&self, src: DevicePtr, dst: DevicePtr, bytes: usize) -> Result<()> {
        self.d2d.fetch_add(1, Ordering::Relaxed);
        self.blit(src, dst, bytes)
    }

    fn copy_d2d_async(
        &self,
        src: DevicePtr,
        dst: DevicePtr,
        bytes: usize,
        stream: u64,
    ) -> Result<()> {
        // 2026-09-25: Overridden to record `stream`; counted in `d2d` like
        // `copy_d2d`.
        self.d2d.fetch_add(1, Ordering::Relaxed);
        self.d2d_async_streams.lock().push(stream);
        self.blit(src, dst, bytes)
    }

    fn copy_d2d_2d_async(
        &self,
        src: DevicePtr,
        src_pitch: usize,
        dst: DevicePtr,
        dst_pitch: usize,
        width_bytes: usize,
        height: usize,
        stream: u64,
    ) -> Result<()> {
        // 2026-09-25: One `d2d_2d` tick per call; the row loop below emulates
        // the copy and does not touch `d2d`.
        self.d2d_2d.fetch_add(1, Ordering::Relaxed);
        self.d2d_2d_async_streams.lock().push(stream);
        if width_bytes > src_pitch || width_bytes > dst_pitch {
            anyhow::bail!(
                "copy_d2d_2d_async: width {width_bytes} exceeds pitch \
                 (src {src_pitch}, dst {dst_pitch}) — rows would overlap"
            );
        }
        for r in 0..height {
            self.blit(
                src.offset(r * src_pitch),
                dst.offset(r * dst_pitch),
                width_bytes,
            )?;
        }
        Ok(())
    }

    fn launch(
        &self,
        func: KernelHandle,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem: u32,
        stream: u64,
        _params: &mut [*mut std::ffi::c_void],
    ) -> Result<()> {
        self.launches.lock().push(MockLaunch {
            func: func.0,
            grid,
            block,
            shared_mem,
            stream,
            args: Vec::new(),
        });
        Ok(())
    }

    fn launch_typed(
        &self,
        func: KernelHandle,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem: u32,
        stream: u64,
        args: &[KernelArg<'_>],
    ) -> Result<()> {
        let args = args
            .iter()
            .map(|arg| match arg {
                KernelArg::Buffer(ptr) => MockArg::Buffer(*ptr),
                KernelArg::Bytes(bytes) => MockArg::Bytes(bytes.to_vec()),
            })
            .collect();
        self.launches.lock().push(MockLaunch {
            func: func.0,
            grid,
            block,
            shared_mem,
            stream,
            args,
        });
        Ok(())
    }

    fn synchronize(&self, stream: u64) -> Result<()> {
        self.sync_d2h_async_counts
            .lock()
            .push((stream, self.d2h_async.load(Ordering::Relaxed)));
        self.syncs.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn default_stream(&self) -> u64 {
        0
    }

    fn has_module(&self, module: &str) -> bool {
        !self.absent_modules.lock().contains(module)
    }

    #[track_caller]
    fn kernel(&self, module: &str, func_name: &str) -> Result<KernelHandle> {
        self.kernel_lookups
            .lock()
            .push((module.to_owned(), func_name.to_owned()));
        if self
            .denied_kernels
            .lock()
            .iter()
            .any(|(m, f)| m == module && f == func_name)
        {
            anyhow::bail!("Kernel lookup {module}::{func_name}: missing");
        }
        Ok(KernelHandle(0xDEAD))
    }

    fn memset(&self, ptr: DevicePtr, value: u8, bytes: usize) -> Result<()> {
        let mut allocs = self.allocs.lock();
        let (offset, alloc) = find_alloc_mut(&mut allocs, ptr)
            .ok_or_else(|| anyhow::anyhow!("memset: ptr {ptr} not allocated"))?;
        alloc.data[offset..offset + bytes].fill(value);
        Ok(())
    }

    fn memset_async(&self, ptr: DevicePtr, value: u8, bytes: usize, _stream: u64) -> Result<()> {
        self.memset(ptr, value, bytes)
    }

    fn alloc_host_pinned(&self, bytes: usize) -> Result<*mut u8> {
        // 2026-09-25: The trait default's zeroed `(bytes, 64)` heap layout,
        // overridden only to count calls.
        self.host_pinned_allocs.fetch_add(1, Ordering::Relaxed);
        let layout = std::alloc::Layout::from_size_align(bytes, 64)
            .map_err(|e| anyhow::anyhow!("invalid layout: {e}"))?;
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            anyhow::bail!("host alloc failed: {bytes} bytes");
        }
        Ok(ptr)
    }

    fn total_memory(&self) -> Result<usize> {
        Ok(128 * 1024 * 1024 * 1024)
    }

    fn sm_count(&self) -> Result<u32> {
        // 2026-09-25: Rationale (PCND): the `sm_count` of
        // `kernels/gb10/HARDWARE.toml`, so occupancy-gated dispatch in tests sees
        // a GB10.
        Ok(48)
    }

    fn free_memory(&self) -> Result<usize> {
        Ok(120 * 1024 * 1024 * 1024)
    }

    /// 2026-09-25: The sum of live allocation sizes, so a test can catch a leak
    /// that `live_alloc_count` alone would miss (same count, different sizes).
    fn live_bytes(&self) -> Option<usize> {
        Some(self.allocs.lock().values().map(|a| a.bytes).sum())
    }

    fn live_alloc_count(&self) -> usize {
        self.allocs.lock().len()
    }
}
