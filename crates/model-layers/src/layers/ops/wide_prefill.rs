// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Which kernel serves a head_dim > 256 prefill, and the BR its
//! grid must be built for.
//!
//! Owner: model-layers ops.
//! Invariants: none beyond the types.

use metrale_gpu_runtime::gpu::{GpuBackend, KernelHandle};

/// 2026-09-25: The head_dim > 256 prefill kernel and the BR its grid must be
/// built for. `qwen3_attention::init` takes the handle from here and
/// `prefill_attention` takes the BR from here, so the two cannot disagree; a
/// grid built for the wrong BR would not fail, it would cover the wrong
/// q-tiles.
///
/// The tensor-core `attn_prefill_512tc` (`BR=32`) when the target carries it;
/// otherwise, or under `METRALE_ATTN_512_TC=0`, the scalar `attn_prefill_512`
/// (`BR=16`), whose handle is zero when the target lacks it too.
pub fn wide_prefill_kernel(gpu: &dyn GpuBackend) -> (KernelHandle, u32) {
    // 2026-09-25: Resolve with a fallback, not a fixed name. Not every target
    // ships `attn_prefill_512tc` (deepseek-v4-flash ships only the scalar
    // `attn_prefill_512`). A zero handle would make the callers'
    // `hd > 256 && prefill_attn_512_k != 0` guard false and send the wide heads
    // down the general prefill path without an error.
    if std::env::var("METRALE_ATTN_512_TC").ok().as_deref() != Some("0") {
        let tc = crate::layers::try_kernel(gpu, "attn_prefill_512tc", "attn_prefill_512tc");
        if tc.0 != 0 {
            return (tc, 32);
        }
        tracing::debug!(
            "attn_prefill_512tc absent for this target; using the scalar \
             HDIM=512 reference"
        );
    }
    (
        crate::layers::try_kernel(gpu, "attn_prefill_512", "attn_prefill_512"),
        16,
    )
}
