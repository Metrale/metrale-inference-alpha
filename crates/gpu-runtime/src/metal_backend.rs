// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `MetalGpuBackend`, the `GpuBackend` for Apple Metal through the
//! `objc2-metal` bindings.
//!
//! Owner: gpu-runtime.
//! Invariants:
//! - Every buffer is allocated with `StorageModeShared`; host copies and
//!   `memset` go through `contents()`.
//! - A `DevicePtr` is a `gpuAddress`. `allocations` maps each buffer's base
//!   address to the buffer, and a pointer resolves to the buffer with the
//!   largest base at or below it.
//! - Stream handle 0 and the default stream are slab slot 0, created in `new`;
//!   `create_stream` returns slot index + 1. `synchronize` commits the stream's
//!   in-flight command buffer and waits for it.
//! - A `KernelHandle` is an index into `pipeline_slab`, and `pipeline_cache`
//!   returns the same handle for a `(module, function)` pair after its first
//!   lookup. The libraries are loaded in `new` and not changed after.

use std::collections::{BTreeMap, HashMap};
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::Arc;

use anyhow::{Result, anyhow, bail};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary,
    MTLResourceOptions, MTLSharedEvent,
};
use parking_lot::Mutex;

use crate::gpu::{DevicePtr, GpuBackend, KernelArg, KernelHandle};

mod events;
mod launch;
mod streams;
mod sysctl;

use sysctl::sysctl_memsize;

type ObjDevice = Retained<ProtocolObject<dyn MTLDevice>>;
type ObjBuffer = Retained<ProtocolObject<dyn MTLBuffer>>;
type ObjQueue = Retained<ProtocolObject<dyn MTLCommandQueue>>;
type ObjCmdBuf = Retained<ProtocolObject<dyn MTLCommandBuffer>>;
type ObjLibrary = Retained<ProtocolObject<dyn MTLLibrary>>;
type ObjPipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
type ObjSharedEvent = Retained<ProtocolObject<dyn MTLSharedEvent>>;

struct MetalStream {
    queue: ObjQueue,
    /// 2026-09-25: The command buffer collecting encoded work. `synchronize`
    /// commits it and waits; the next encode opens a fresh one.
    in_flight: Option<ObjCmdBuf>,
}

/// 2026-09-25: One shared event and its signal counter.
struct EventSlot {
    event: ObjSharedEvent,
    /// 2026-09-25: `record_event` signals `next` and increments it;
    /// `stream_wait_event` waits for `next - 1`, the last recorded value.
    /// Guarded by the `events` mutex.
    next: u64,
}

/// 2026-09-25: `(module, function)` key of the pipeline cache.
type PipelineKey = (String, String);

pub struct MetalGpuBackend {
    op_cache: crate::op_cache::OpCache,
    device: ObjDevice,
    /// 2026-09-25: Each buffer's base `gpuAddress` to the buffer; ordered so
    /// `find_buffer` can take `range(..=ptr).next_back()`.
    allocations: Arc<Mutex<BTreeMap<u64, ObjBuffer>>>,
    /// 2026-09-25: Streams: handle 0 is slot 0, the default stream created in
    /// `new`; any other handle `h` is slot `h - 1`.
    streams: Arc<Mutex<Vec<MetalStream>>>,
    /// 2026-09-25: Loaded metallibs by module name.
    libraries: HashMap<String, ObjLibrary>,
    /// 2026-09-25: `(module, function)` to its `pipeline_slab` index.
    pipeline_cache: Arc<Mutex<HashMap<PipelineKey, KernelHandle>>>,
    pipeline_slab: Arc<Mutex<Vec<ObjPipeline>>>,
    /// 2026-09-25: Shared events; handle `h` is slot `h - 1`.
    events: Arc<Mutex<Vec<EventSlot>>>,
}

unsafe impl Send for MetalGpuBackend {}
unsafe impl Sync for MetalGpuBackend {}

impl MetalGpuBackend {
    /// 2026-09-25: Open the system default device and load `kernel_modules`,
    /// `(module_name, metallib_bytes)` pairs, one `MTLLibrary` each. Creates the
    /// default stream. Only ordinal 0 is accepted.
    pub fn new(ordinal: usize, kernel_modules: &[(&'static str, &'static [u8])]) -> Result<Self> {
        if ordinal != 0 {
            bail!(
                "Metal: only ordinal 0 is supported (Apple Silicon has one \
                 system default device); requested ordinal {ordinal}"
            );
        }
        let device: ObjDevice = MTLCreateSystemDefaultDevice().ok_or_else(|| {
            anyhow!("MTLCreateSystemDefaultDevice returned null — no Metal-capable GPU")
        })?;

        // 2026-09-25: `newLibraryWithData_error` takes a `DispatchData`; the
        // module bytes are `'static`, so `from_static_bytes` wraps them.
        let mut libraries: HashMap<String, ObjLibrary> = HashMap::new();
        for (name, bytes) in kernel_modules {
            let data = dispatch2::DispatchData::from_static_bytes(bytes);
            let lib = device.newLibraryWithData_error(&data).map_err(|e| {
                anyhow!(
                    "newLibraryWithData failed for module '{name}': {}",
                    e.localizedDescription()
                )
            })?;
            libraries.insert((*name).to_string(), lib);
        }

        // 2026-09-25: The default stream, slot 0.
        let default_queue = device
            .newCommandQueue()
            .ok_or_else(|| anyhow!("newCommandQueue returned null on default device"))?;
        let streams = vec![MetalStream {
            queue: default_queue,
            in_flight: None,
        }];

        tracing::info!(
            "MetalGpuBackend initialized on device '{}' with {} metallib modules",
            device.name().to_string(),
            libraries.len()
        );

        Ok(Self {
            op_cache: crate::op_cache::OpCache::new(),
            device,
            allocations: Arc::new(Mutex::new(BTreeMap::new())),
            streams: Arc::new(Mutex::new(streams)),
            libraries,
            pipeline_cache: Arc::new(Mutex::new(HashMap::new())),
            pipeline_slab: Arc::new(Mutex::new(Vec::new())),
            events: Arc::new(Mutex::new(Vec::new())),
        })
    }

    /// 2026-09-25: The underlying `MTLDevice`.
    pub fn raw_device(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.device
    }
}

impl GpuBackend for MetalGpuBackend {
    fn op_cache(&self) -> &crate::op_cache::OpCache {
        &self.op_cache
    }

    fn alloc(&self, bytes: usize) -> Result<DevicePtr> {
        let buf: ObjBuffer = self
            .device
            .newBufferWithLength_options(bytes.max(1), MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| anyhow!("newBufferWithLength failed for {bytes} bytes"))?;
        let addr = buf.gpuAddress();
        if addr == 0 {
            bail!("MTLBuffer::gpuAddress returned 0 — Metal 3 / macOS 13 required");
        }
        self.allocations.lock().insert(addr, buf);
        Ok(DevicePtr(addr))
    }

    fn alloc_managed(&self, bytes: usize) -> Result<DevicePtr> {
        // 2026-09-25: `alloc` already returns shared storage.
        self.alloc(bytes)
    }

    fn free(&self, ptr: DevicePtr) -> Result<()> {
        if ptr.is_null() {
            return Ok(());
        }
        // 2026-09-25: Dropping the table's `Retained` releases the buffer. A
        // pointer that is not a buffer base is ignored.
        self.allocations.lock().remove(&ptr.0);
        Ok(())
    }

    fn copy_h2d(&self, src: &[u8], dst: DevicePtr) -> Result<()> {
        if src.is_empty() {
            return Ok(());
        }
        let allocs = self.allocations.lock();
        let (buf, offset) = Self::find_buffer(&allocs, dst)
            .ok_or_else(|| anyhow!("copy_h2d: ptr {dst} not in any allocation"))?;
        if offset + src.len() > buf.length() {
            bail!(
                "copy_h2d: write overflows buffer ({} + {} > {})",
                offset,
                src.len(),
                buf.length()
            );
        }
        let contents: NonNull<c_void> = buf.contents();
        unsafe {
            let dst_ptr = (contents.as_ptr() as *mut u8).add(offset);
            std::ptr::copy_nonoverlapping(src.as_ptr(), dst_ptr, src.len());
        }
        Ok(())
    }

    fn copy_d2h(&self, src: DevicePtr, dst: &mut [u8]) -> Result<()> {
        if dst.is_empty() {
            return Ok(());
        }
        let allocs = self.allocations.lock();
        let (buf, offset) = Self::find_buffer(&allocs, src)
            .ok_or_else(|| anyhow!("copy_d2h: ptr {src} not in any allocation"))?;
        if offset + dst.len() > buf.length() {
            bail!(
                "copy_d2h: read overflows buffer ({} + {} > {})",
                offset,
                dst.len(),
                buf.length()
            );
        }
        let contents: NonNull<c_void> = buf.contents();
        unsafe {
            let src_ptr = (contents.as_ptr() as *const u8).add(offset);
            std::ptr::copy_nonoverlapping(src_ptr, dst.as_mut_ptr(), dst.len());
        }
        Ok(())
    }

    fn copy_d2h_on_stream(&self, src: DevicePtr, dst: &mut [u8], stream: u64) -> Result<()> {
        // 2026-09-25: Wait for the work queued on `stream`, then copy.
        self.synchronize(stream)?;
        self.copy_d2h(src, dst)
    }

    fn copy_d2d(&self, src: DevicePtr, dst: DevicePtr, bytes: usize) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        let allocs = self.allocations.lock();
        let (src_buf, src_off) = Self::find_buffer(&allocs, src)
            .ok_or_else(|| anyhow!("copy_d2d: src ptr {src} not allocated"))?;
        let (dst_buf, dst_off) = Self::find_buffer(&allocs, dst)
            .ok_or_else(|| anyhow!("copy_d2d: dst ptr {dst} not allocated"))?;
        drop(allocs);

        let cmd_buf = self.current_cmd_buf(0)?;
        let enc = cmd_buf
            .blitCommandEncoder()
            .ok_or_else(|| anyhow!("blitCommandEncoder returned null"))?;
        unsafe {
            enc.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                &src_buf, src_off, &dst_buf, dst_off, bytes,
            );
        }
        enc.endEncoding();
        // 2026-09-25: Wait for the default stream, as the CUDA `copy_d2d` does.
        if let Some(cb) = self.commit_in_flight(0)? {
            cb.waitUntilCompleted();
        }
        Ok(())
    }

    fn copy_d2d_async(
        &self,
        src: DevicePtr,
        dst: DevicePtr,
        bytes: usize,
        stream: u64,
    ) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        let allocs = self.allocations.lock();
        let (src_buf, src_off) = Self::find_buffer(&allocs, src)
            .ok_or_else(|| anyhow!("copy_d2d_async: src {src} not allocated"))?;
        let (dst_buf, dst_off) = Self::find_buffer(&allocs, dst)
            .ok_or_else(|| anyhow!("copy_d2d_async: dst {dst} not allocated"))?;
        drop(allocs);
        let cmd_buf = self.current_cmd_buf(stream)?;
        let enc = cmd_buf
            .blitCommandEncoder()
            .ok_or_else(|| anyhow!("blitCommandEncoder returned null"))?;
        unsafe {
            enc.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                &src_buf, src_off, &dst_buf, dst_off, bytes,
            );
        }
        enc.endEncoding();
        Ok(())
    }

    fn launch(
        &self,
        _func: KernelHandle,
        _grid: [u32; 3],
        _block: [u32; 3],
        _shared_mem: u32,
        _stream: u64,
        _params: &mut [*mut c_void],
    ) -> Result<()> {
        // 2026-09-25: Untyped slots cannot be told apart as buffers or bytes,
        // which Metal binds differently; only `launch_typed` is supported.
        bail!(
            "Metal backend: launch() requires typed args. Use launch_typed() \
             with KernelArg::Buffer / KernelArg::Bytes — see KernelLaunch builder."
        );
    }

    fn launch_typed(
        &self,
        func: KernelHandle,
        grid: [u32; 3],
        block: [u32; 3],
        _shared_mem: u32,
        stream: u64,
        args: &[KernelArg<'_>],
    ) -> Result<()> {
        self.launch_typed_mtl(func, grid, block, stream, args)
    }

    fn synchronize(&self, stream: u64) -> Result<()> {
        if let Some(cb) = self.commit_in_flight(stream)? {
            cb.waitUntilCompleted();
        }
        Ok(())
    }

    fn default_stream(&self) -> u64 {
        0
    }

    #[track_caller]
    fn has_module(&self, module: &str) -> bool {
        self.libraries.contains_key(module)
    }

    fn kernel(&self, module: &str, func_name: &str) -> Result<KernelHandle> {
        let key: PipelineKey = (module.to_string(), func_name.to_string());
        if let Some(handle) = self.pipeline_cache.lock().get(&key) {
            return Ok(*handle);
        }
        let lib = self
            .libraries
            .get(module)
            .ok_or_else(|| anyhow!("Metal: unknown module '{module}'"))?;
        let ns_name = NSString::from_str(func_name);
        let function = lib.newFunctionWithName(&ns_name).ok_or_else(|| {
            anyhow!("Metal: function '{func_name}' not found in module '{module}'")
        })?;
        let pipeline = self
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| {
                anyhow!(
                    "newComputePipelineStateWithFunction failed for '{func_name}': {}",
                    e.localizedDescription()
                )
            })?;
        let mut slab = self.pipeline_slab.lock();
        let handle = KernelHandle(slab.len() as u64);
        slab.push(pipeline);
        drop(slab);
        self.pipeline_cache.lock().insert(key, handle);
        Ok(handle)
    }

    fn memset(&self, ptr: DevicePtr, value: u8, bytes: usize) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        // 2026-09-25: Written from the host through `contents()`.
        let allocs = self.allocations.lock();
        let (buf, offset) = Self::find_buffer(&allocs, ptr)
            .ok_or_else(|| anyhow!("memset: ptr {ptr} not allocated"))?;
        if offset + bytes > buf.length() {
            bail!(
                "memset: range overflows buffer ({} + {} > {})",
                offset,
                bytes,
                buf.length()
            );
        }
        let contents = buf.contents();
        unsafe {
            let dst = (contents.as_ptr() as *mut u8).add(offset);
            std::ptr::write_bytes(dst, value, bytes);
        }
        Ok(())
    }

    fn memset_async(&self, ptr: DevicePtr, value: u8, bytes: usize, _stream: u64) -> Result<()> {
        // 2026-09-25: Written at once from the host, not ordered after the work
        // already queued on the stream.
        self.memset(ptr, value, bytes)
    }

    fn total_memory(&self) -> Result<usize> {
        // 2026-09-25: `hw.memsize` from sysctl, or the device's
        // `recommendedMaxWorkingSetSize` when sysctl fails.
        Ok(sysctl_memsize().unwrap_or_else(|| self.device.recommendedMaxWorkingSetSize() as usize))
    }

    fn free_memory(&self) -> Result<usize> {
        // 2026-09-25: Approximated as `recommendedMaxWorkingSetSize -
        // currentAllocatedSize`, floored at 0.
        let max = self.device.recommendedMaxWorkingSetSize() as usize;
        let used = self.device.currentAllocatedSize();
        Ok(max.saturating_sub(used))
    }

    fn sm_count(&self) -> Result<u32> {
        // 2026-09-25: Refuses rather than invent a count for occupancy-based
        // dispatch.
        anyhow::bail!("MetalGpuBackend does not expose a multiprocessor count")
    }

    fn create_stream(&self) -> Result<u64> {
        let queue = self
            .device
            .newCommandQueue()
            .ok_or_else(|| anyhow!("newCommandQueue returned null"))?;
        let mut slab = self.streams.lock();
        slab.push(MetalStream {
            queue,
            in_flight: None,
        });
        // 2026-09-25: Handle = slab index + 1.
        Ok(slab.len() as u64)
    }

    fn bind_to_thread(&self) -> Result<()> {
        Ok(())
    }

    fn create_event(&self) -> Result<u64> {
        self.create_event_mtl()
    }

    fn record_event(&self, event: u64, stream: u64) -> Result<()> {
        self.record_event_mtl(event, stream)
    }

    fn stream_wait_event(&self, stream: u64, event: u64) -> Result<()> {
        self.stream_wait_event_mtl(stream, event)
    }

    fn destroy_event(&self, event: u64) -> Result<()> {
        self.destroy_event_mtl(event)
    }

    fn alloc_host_pinned(&self, bytes: usize) -> Result<*mut u8> {
        // 2026-09-25: A shared buffer, kept in the allocation table by
        // `gpuAddress`; the caller gets its `contents()` pointer, which
        // `free_host_pinned` looks up to release it.
        let buf = self
            .device
            .newBufferWithLength_options(bytes.max(1), MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| anyhow!("alloc_host_pinned: newBufferWithLength failed"))?;
        let host_ptr = buf.contents().as_ptr() as *mut u8;
        // 2026-09-25: Keyed by `gpuAddress`, so `free` of that address also
        // releases it.
        let addr = buf.gpuAddress();
        if addr == 0 {
            bail!("alloc_host_pinned: gpuAddress returned 0");
        }
        // 2026-09-25: Zeroed, as `GpuBackend::alloc_host_pinned` requires.
        // SAFETY: `host_ptr` is the `contents()` pointer of a live shared buffer
        // of at least `bytes.max(1)` bytes, owned only here until it enters the
        // allocation table.
        unsafe { std::ptr::write_bytes(host_ptr, 0, bytes.max(1)) };
        self.allocations.lock().insert(addr, buf);
        Ok(host_ptr)
    }

    fn free_host_pinned(&self, ptr: *mut u8, _bytes: usize) -> Result<()> {
        if ptr.is_null() {
            return Ok(());
        }
        // 2026-09-25: A pointer that matches no buffer is ignored.
        let mut allocs = self.allocations.lock();
        let target_addr = allocs.iter().find_map(|(addr, buf)| {
            let host = buf.contents().as_ptr() as *mut u8;
            if host == ptr { Some(*addr) } else { None }
        });
        if let Some(addr) = target_addr {
            allocs.remove(&addr);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
