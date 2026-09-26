// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Which kernel serves the width-8 `w4a16_gemv` batched-GEMV tier.
//!
//! Owner: model-layers (NVFP4 GEMV dispatch).
//! Invariants: none beyond the types.

use super::try_kernel;
use metrale_gpu_runtime::gpu::{GpuBackend, KernelHandle};

/// 2026-09-25: Resolve the M<=8 batched-GEMV kernel (`W4a16BatchmTiers::resolve`, width 8).
/// provenance-id: 526f6e616c6420522e205374657369616b
///
/// Prefers `w4a16_gemv_batch8_rt2`, where each thread computes two adjacent
/// output rows, so one activation load feeds two FMA chains. It takes the
/// same launch as `w4a16_gemv_batch8` (`ops::w4a16_gemv_batchm`: grid
/// ceil(N/4), block 256); the surplus blocks return on `n0 >= N`.
/// `batchm_bench` gate 4 requires its output to equal batch8's bit for bit at
/// every M. Measured 2026-08-19 with `batchm_bench` at M=8: +17-26% GB/s on
/// the verify shapes.
///
/// `METRALE_NO_BATCH8_RT=1` (exactly `"1"`) selects `w4a16_gemv_batch8`, as
/// does a target without `rt2`.
pub(crate) fn batch8_kernel(gpu: &dyn GpuBackend) -> KernelHandle {
    if std::env::var("METRALE_NO_BATCH8_RT").as_deref() != Ok("1") {
        let h = try_kernel(gpu, "w4a16_gemv", "w4a16_gemv_batch8_rt2");
        if h.0 != 0 {
            return h;
        }
    }
    try_kernel(gpu, "w4a16_gemv", "w4a16_gemv_batch8")
}
