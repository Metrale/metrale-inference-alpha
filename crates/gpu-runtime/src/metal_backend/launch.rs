// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The body of `MetalGpuBackend::launch_typed`: one compute
//! dispatch encoded on the stream's in-flight command buffer.
//!
//! Owner: gpu-runtime (Metal backend).
//! Invariants:
//! - Every live allocation is declared read/write with `useResource`, and
//!   argument `i` binds to buffer index `i`.

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::ptr::NonNull;

use anyhow::{Result, anyhow};
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLResource, MTLSize,
};

use super::{MetalGpuBackend, ObjBuffer};
use crate::gpu::{KernelArg, KernelHandle};

impl MetalGpuBackend {
    /// 2026-09-26: `launch_typed` for `func` on `stream`, `grid` threadgroups of
    /// `block` threads, binding `args` in order.
    pub(super) fn launch_typed_mtl(
        &self,
        func: KernelHandle,
        grid: [u32; 3],
        block: [u32; 3],
        stream: u64,
        args: &[KernelArg<'_>],
    ) -> Result<()> {
        let pipeline = {
            let slab = self.pipeline_slab.lock();
            slab.get(func.0 as usize)
                .cloned()
                .ok_or_else(|| anyhow!("launch_typed: unknown kernel handle {}", func.0))?
        };

        // 2026-09-25: Snapshot the allocation table, so buffer args resolve and
        // every live buffer is declared without holding its lock while encoding.
        let live_buffers: Vec<ObjBuffer> = self.allocations.lock().values().cloned().collect();
        let allocs_snapshot: BTreeMap<u64, ObjBuffer> = self.allocations.lock().clone();

        let cmd_buf = self.current_cmd_buf(stream)?;
        let enc = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| anyhow!("computeCommandEncoder returned null"))?;
        enc.setComputePipelineState(&pipeline);

        // 2026-09-25: Every live allocation is declared read/write with
        // `useResource`, not only the bound arguments.
        for buf in &live_buffers {
            let resource: &ProtocolObject<dyn MTLResource> = ProtocolObject::from_ref(&**buf);
            enc.useResource_usage(
                resource,
                objc2_metal::MTLResourceUsage::Read | objc2_metal::MTLResourceUsage::Write,
            );
        }

        // 2026-09-25: Argument `i` binds to index `i`.
        for (idx, arg) in args.iter().enumerate() {
            match arg {
                KernelArg::Buffer(p) => {
                    let (buf, offset) = Self::find_buffer(&allocs_snapshot, *p)
                        .ok_or_else(|| anyhow!("launch_typed: arg #{idx} ptr {p} not allocated"))?;
                    unsafe {
                        enc.setBuffer_offset_atIndex(Some(&buf), offset, idx);
                    }
                }
                KernelArg::Bytes(b) => {
                    let ptr = NonNull::new(b.as_ptr() as *mut c_void)
                        .ok_or_else(|| anyhow!("launch_typed: arg #{idx} bytes is null"))?;
                    unsafe {
                        enc.setBytes_length_atIndex(ptr, b.len(), idx);
                    }
                }
            }
        }

        let threadgroups = MTLSize {
            width: grid[0] as usize,
            height: grid[1] as usize,
            depth: grid[2] as usize,
        };
        let threads_per_tg = MTLSize {
            width: block[0] as usize,
            height: block[1] as usize,
            depth: block[2] as usize,
        };
        enc.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
        enc.endEncoding();
        Ok(())
    }
}
