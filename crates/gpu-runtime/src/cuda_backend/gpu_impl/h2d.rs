// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The H2D enqueue shared by `copy_h2d_async` and
//! `copy_h2d_async_retained`, and the once-per-process warning for a
//! page-locked source handed to `copy_h2d_async`.
//!
//! Owner: gpu-runtime (CUDA backend).
//! Invariants:
//! - The one `unsafe` block is one call to `cuMemcpyHtoDAsync_v2`, declared in
//!   `cuda_backend.rs`; the caller provides a current context and a `dst` range
//!   inside a live allocation.

use std::ffi::c_void;

use anyhow::{Result, bail};

use super::super::cuMemcpyHtoDAsync_v2;
use crate::gpu::DevicePtr;

/// 2026-09-25: Enqueue an H2D copy on `stream` and return without waiting.
/// Both async H2D entry points use it, so they differ only in the wait they
/// add after it.
pub(super) fn h2d_enqueue(src: &[u8], dst: DevicePtr, stream: u64) -> Result<()> {
    let status =
        unsafe { cuMemcpyHtoDAsync_v2(dst.0, src.as_ptr() as *const c_void, src.len(), stream) };
    if status != 0 {
        bail!("cuMemcpyHtoDAsync_v2 failed: status {status}");
    }
    Ok(())
}

/// 2026-09-25: Warn once per process that `copy_h2d_async` was given a
/// page-locked source (`crate::pinned_hosts`). A warning, not an error:
/// `copy_h2d_async` then synchronises the stream, so the copy is correct, but
/// it waits where `copy_h2d_async_retained` would not.
pub(super) fn warn_pinned_transient_source() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        tracing::warn!(target: "metrale_gpu_runtime::cuda_backend::gpu_impl", "copy_h2d_async was handed a PAGE-LOCKED source. That copy is genuinely \
             asynchronous, so the promise that the caller may drop the buffer on return \
             is now being paid for with a cuStreamSynchronize on every such call. If the \
             source outlives the next sync, switch the call site to \
             copy_h2d_async_retained; if it does not, this sync is what keeps it from \
             being a use-after-free."
        );
    });
}
