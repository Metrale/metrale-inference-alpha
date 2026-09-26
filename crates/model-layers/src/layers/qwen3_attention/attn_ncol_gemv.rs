// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The N-column-blocked W8A16 decode tier for the attention projections,
//! `w8a16_gemv_batch16_ncol{2,4}`, behind the `attn_ncol_gemv` target lever.
//!
//! `w8a16_gemv_batchm_impl<16>` gives each thread one output column, so every column's thread loads
//! and converts the same activations. `w8a16_gemv_ncol_impl` gives one thread `N_COLS` adjacent
//! columns and reuses each loaded, converted activation across them.
//!
//! The tier is bit-exact with the batch16 GEMV it replaces: the lane-to-k16 map, the per-row
//! operand order and the two-warp reduction are the same, and the BF16->FP32 convert is only
//! hoisted out of the column loop (`kernels/hopper/common/w8a16_gemv_ncol.cu` against
//! `w8a16_gemv_batchm_impl` in `kernels/gb10/common/w8a16_gemv_batch4.cu`). That is what separates
//! it from the `METRALE_ATTN_M16_TC` tier, which reassociates the K reduction.
//!
//! The band is 5..=16, the rows the batch16 GEMV serves; 2..=4 rows stay on `w8a16_gemv_batch4`.
//! Above 16 rows the kernel's `MAX_M` is exceeded, and the op wrapper refuses the launch
//! (`ops/w8a16_gemv_ncol.rs`), so [`ncol_plan`] declines first.
//!
//! Owner: model-layers attention decode.
//! Invariants:
//! - The route is a pure function of the row count and of fields fixed at layer construction
//!   (`attn_ncol` and the four kernel handles), so it cannot change between a CUDA-graph capture
//!   and its replays at the same row count.
//! - A width whose entry point is not loaded is declined, never replaced by the other width.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::Qwen3AttentionLayer;
use crate::layers::ops;

/// 2026-09-25: Output columns one thread owns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NcolWidth {
    Two,
    Four,
}

impl NcolWidth {
    /// 2026-09-25: The `N_COLS` template argument of the instantiation this width selects.
    pub(crate) fn cols(self) -> u32 {
        match self {
            NcolWidth::Two => 2,
            NcolWidth::Four => 4,
        }
    }
}

/// 2026-09-25: Whether the attention projections may take the tier: the `attn_ncol_gemv` row of
/// the compiled target's `HARDWARE.toml` `[defaults]` (false where a target declares it, and false
/// when a target has no `[defaults]` table), overridden by `METRALE_ATTN_NCOL_GEMV` (`0`, `false`,
/// `off` or `no` turn it off, any other value on). The presence of `METRALE_NO_ATTN_DECODE_BATCH`
/// turns it off and outranks both. The resolution is computed once per process by
/// `ops::target_defaults::resolved`.
pub fn ncol_gemv_enabled() -> bool {
    crate::layers::ops::target_defaults::resolved()
        .attn_ncol_gemv
        .value
}

/// 2026-09-25: `METRALE_ATTN_NCOL_WIDTH=4` picks the 4-column instantiation; any other value, or
/// none, keeps 2, whose `acc[MAX_M][N_COLS]` accumulator array is half the size. Read once per
/// process.
pub fn ncol_gemv_width() -> NcolWidth {
    static W: std::sync::OnceLock<NcolWidth> = std::sync::OnceLock::new();
    *W.get_or_init(
        || match std::env::var("METRALE_ATTN_NCOL_WIDTH").as_deref() {
            Ok("4") => NcolWidth::Four,
            _ => NcolWidth::Two,
        },
    )
}

/// 2026-09-25: The selection rule, as a pure function of the row count, the resolved width, the two
/// handles' presence and the lever, so CPU tests can cover every edge without a `ForwardContext`.
///
/// Returns the width to launch with, or `None` to leave the caller on the next rung of its ladder.
pub(crate) fn ncol_plan(
    m: usize,
    width: NcolWidth,
    ncol2_loaded: bool,
    ncol4_loaded: bool,
    enabled: bool,
) -> Option<NcolWidth> {
    if !enabled || !(5..=16).contains(&m) {
        return None;
    }
    // 2026-09-25: A missing entry point declines rather than substituting the other width, so the
    // route line always names the kernel that ran.
    let loaded = match width {
        NcolWidth::Two => ncol2_loaded,
        NcolWidth::Four => ncol4_loaded,
    };
    loaded.then_some(width)
}

/// 2026-09-25: One line per process, the first time a projection takes the tier, naming the width
/// and the first call site.
pub(crate) fn log_route_once(width: NcolWidth, site: &str) {
    static LOGGED: std::sync::Once = std::sync::Once::new();
    LOGGED.call_once(|| {
        tracing::info!(
            "METRALE_ATTN_NCOL_GEMV: decode attention projections via \
             w8a16_gemv_batch16_ncol{} (bit-exact; first site: {site})",
            width.cols(),
        );
    });
}

/// 2026-09-25: The signature of `ops::w8a16_gemv_batch{4,16}_strided` and the `_ncol*_strided`
/// entry points, so a tier choice is a function pointer rather than another call site.
pub(crate) type StridedBatchGemv = fn(
    &dyn GpuBackend,
    KernelHandle,
    DevicePtr,
    DevicePtr,
    DevicePtr,
    DevicePtr,
    u32,
    u32,
    u32,
    u32,
    u32,
    u64,
) -> Result<()>;

impl Qwen3AttentionLayer {
    /// 2026-09-25: The strided rung for the multi-seq Q/K/V projections, or `None` to stay on
    /// `w8a16_gemv_batch16_strided`. Logs the route the first time it fires.
    pub(super) fn ncol_strided_route(&self, m: usize) -> Option<(StridedBatchGemv, KernelHandle)> {
        let width = ncol_plan(
            m,
            self.attn_ncol?,
            self.w8a16_gemv_ncol2_strided_k.0 != 0,
            self.w8a16_gemv_ncol4_strided_k.0 != 0,
            true,
        )?;
        log_route_once(width, "multi-seq QKV");
        Some(match width {
            NcolWidth::Two => (
                ops::w8a16_gemv_batch16_ncol2_strided as StridedBatchGemv,
                self.w8a16_gemv_ncol2_strided_k,
            ),
            NcolWidth::Four => (
                ops::w8a16_gemv_batch16_ncol4_strided as StridedBatchGemv,
                self.w8a16_gemv_ncol4_strided_k,
            ),
        })
    }

    /// 2026-09-25: The contiguous rung for the FP8 o_proj, or `None` to stay on
    /// `w8a16_gemv_batch16`. Both rungs walk the batch in 16-row groups, so the caller's `step`
    /// is unchanged.
    pub(super) fn ncol_contiguous_route(
        &self,
        m: usize,
    ) -> Option<(ops::ContiguousBatchGemv, KernelHandle)> {
        let width = ncol_plan(
            m,
            self.attn_ncol?,
            self.w8a16_gemv_ncol2_k.0 != 0,
            self.w8a16_gemv_ncol4_k.0 != 0,
            true,
        )?;
        log_route_once(width, "o_proj");
        Some(match width {
            NcolWidth::Two => (
                ops::w8a16_gemv_batch16_ncol2 as ops::ContiguousBatchGemv,
                self.w8a16_gemv_ncol2_k,
            ),
            NcolWidth::Four => (
                ops::w8a16_gemv_batch16_ncol4 as ops::ContiguousBatchGemv,
                self.w8a16_gemv_ncol4_k,
            ),
        })
    }
}

#[cfg(test)]
#[path = "attn_ncol_gemv_tests.rs"]
mod tests;
