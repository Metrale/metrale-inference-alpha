// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The heap allocation behind the default `GpuBackend::alloc_host_pinned`
//! and `free_host_pinned`. The mock allocates the same zeroed `(bytes, 64)` layout
//! itself, to count calls, and frees through the default.
//!
//! Owner: gpu-runtime.
//! Invariants:
//! - A block is zeroed, 64-byte aligned, and freed with the same `(bytes, 64)`
//!   layout it was allocated with, so `free` must get the allocated size.

use anyhow::Result;

pub fn alloc_zeroed(bytes: usize) -> Result<*mut u8> {
    let layout = std::alloc::Layout::from_size_align(bytes, 64)
        .map_err(|e| anyhow::anyhow!("invalid layout: {e}"))?;
    let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
    if ptr.is_null() {
        anyhow::bail!("host alloc failed: {bytes} bytes");
    }
    Ok(ptr)
}

#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn free(ptr: *mut u8, bytes: usize) -> Result<()> {
    if !ptr.is_null() {
        let layout = std::alloc::Layout::from_size_align(bytes, 64)
            .map_err(|e| anyhow::anyhow!("invalid layout: {e}"))?;
        unsafe { std::alloc::dealloc(ptr, layout) };
    }
    Ok(())
}
