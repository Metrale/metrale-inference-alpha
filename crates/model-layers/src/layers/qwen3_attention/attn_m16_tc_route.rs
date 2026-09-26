// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Log-once route lines for the `METRALE_ATTN_M16_TC` tensor-core tiers of attention
//! decode.
//!
//! The lever arms two tiers, the strided Q/K/V GEMM in `qkv_fp8_batch.rs` and the contiguous o_proj
//! GEMM in `attn/o_proj.rs`, so each gets its own line and key. Both call sites take their wording
//! from this module, so the two lines cannot drift apart.
//!
//! Owner: model-layers attention decode.
//! Invariants:
//! - The two tiers latch on different `ModelStats::once` keys, so firing one never consumes the
//!   other.

use crate::layers::ops::ModelStats;

/// 2026-09-25: `ModelStats::once` key for the strided Q/K/V tier (`qkv_fp8_batch.rs`).
pub(crate) const QKV_M16_TC_ROUTE_KEY: &str = "log:attn_qkv_m16_tc_decode";
/// 2026-09-25: `ModelStats::once` key for the contiguous o_proj tier (`attn/o_proj.rs`).
pub(crate) const O_PROJ_M16_TC_ROUTE_KEY: &str = "log:attn_o_proj_m16_tc_decode";

/// 2026-09-25: The Q/K/V tier's line: the tier, the lever, the kernel, its fixed N_TILE, the row
/// band, the two tiers it is checked ahead of (the N-column tier and the batch16 GEMV), the
/// numeric consequence and the off switch.
///
/// The numerics are stated as the tier's contract, `within_m16_tc_budget`: 2 ordinal BF16 ULP, or
/// the FP32 accumulation floor for outputs that cancelled, rather than a bare ULP bound.
pub(crate) const QKV_M16_TC_ROUTE_MSG: &str = "\
[metrale] attention decode q/k/v: METRALE_ATTN_M16_TC — tensor-core \
w8a16_gemm_m16_strided N_TILE=32 (fixed; the strided tier has no wide n64 \
twin) for 5..=16 rows, checked ahead of the bit-exact N-column tier \
(METRALE_ATTN_NCOL_GEMV) and w8a16_gemv_batch16_strided. One weight pass, \
m16n8k16 MMA, so outputs are REASSOCIATED vs the scalar w8a16_gemv — within \
2 ordinal BF16 ULP, OR the FP32 accumulation floor for outputs that have \
catastrophically cancelled (the contract is \
layers::dense_ffn::m16_tc::within_m16_tc_budget, not a bare 2-ULP bound) — \
unlike the batch16 tier. Unset it to restore the bit-exact tier (#927; H100 \
round 9 cell W: C=16 TPOT 53.42 -> 50.01 ms, +5.26% aggregate).";

/// 2026-09-25: The o_proj tier's line. Same shape as [`QKV_M16_TC_ROUTE_MSG`], naming the
/// contiguous kernel and the two tiers it is checked ahead of.
pub(crate) const O_PROJ_M16_TC_ROUTE_MSG: &str = "\
[metrale] attention decode o_proj: METRALE_ATTN_M16_TC — tensor-core \
w8a16_gemm_m16 N_TILE=32 (fixed; the contiguous tier has no wide n64 twin) \
for 5..=16 rows, checked ahead of the bit-exact N-column tier \
(METRALE_ATTN_NCOL_GEMV) and w8a16_gemv_batch16. One weight pass, m16n8k16 \
MMA, so outputs are REASSOCIATED vs the scalar w8a16_gemv — within 2 ordinal \
BF16 ULP, OR the FP32 accumulation floor for outputs that have \
catastrophically cancelled (the contract is \
layers::dense_ffn::m16_tc::within_m16_tc_budget, not a bare 2-ULP bound) — \
unlike the batch16 tier. Unset it to restore the bit-exact tier (#927; H100 \
round 9 cell W: C=16 TPOT 53.42 -> 50.01 ms, +5.26% aggregate).";

/// 2026-09-25: Fire the Q/K/V route line, once per model. Called from the `tc` arm of
/// `ms_qkv_batchm_fp8_gemv` (`qkv_fp8_batch.rs`), the arm that launches `w8a16_gemm_m16_strided`.
pub(crate) fn log_qkv_m16_tc_route(stats: &ModelStats) {
    if stats.once(QKV_M16_TC_ROUTE_KEY) {
        tracing::info!("{QKV_M16_TC_ROUTE_MSG}");
    }
}

/// 2026-09-25: Fire the o_proj route line, once per model. Called from the `tc` arm of
/// `ms_phase_o_proj` (`attn/o_proj.rs`), the arm that launches `w8a16_gemm_m16`.
pub(crate) fn log_o_proj_m16_tc_route(stats: &ModelStats) {
    if stats.once(O_PROJ_M16_TC_ROUTE_KEY) {
        tracing::info!("{O_PROJ_M16_TC_ROUTE_MSG}");
    }
}

#[cfg(test)]
#[path = "attn_m16_tc_route_tests.rs"]
mod tests;
