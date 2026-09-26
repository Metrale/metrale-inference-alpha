// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Cases for the `m16_tc` arm of the batched native-FP8 Q/K/V tier.
//!
//! Owner: model-layers (qwen3 attention, multi-sequence decode).
//! Invariants: none beyond the types.
//!
//! A child of `qkv_fp8_batch_tests.rs`, sharing its harness.

use super::{
    BATCH4_K, BATCH16_K, Case, Expect, M16TC_STRIDED_K, check_dispatch, qkv_phase_launches,
};
use crate::layers::qwen3_attention::attn_ncol_gemv::NcolWidth;

/// 2026-09-25: With `m16_tc` on, the MMA arm takes the band
/// `w8a16_gemv_batch16_strided` owns: one strided launch per projection, same
/// argument layout, a different kernel.
#[test]
fn native_fp8_qkv_attn_m16_tc_takes_the_batch16_band() {
    for rows in [5, 8, 12, 16] {
        check_dispatch(&Case::m16_tc(rows), Expect::Batched(M16TC_STRIDED_K));
    }
}

/// 2026-09-25: The `m16_tc` arm comes before the bit-exact N-column arm when
/// both are on.
#[test]
fn native_fp8_qkv_attn_m16_tc_outranks_the_ncol_tier() {
    let mut case = Case::m16_tc(16);
    case.ncol = Some(NcolWidth::Four);
    check_dispatch(&case, Expect::Batched(M16TC_STRIDED_K));
}

/// 2026-09-25: Below 5 rows `w8a16_gemv_batch4_strided` keeps the rows even
/// with `m16_tc` on.
#[test]
fn native_fp8_qkv_attn_m16_tc_leaves_small_batches_on_batch4() {
    for rows in [2, 3, 4] {
        check_dispatch(&Case::m16_tc(rows), Expect::Batched(BATCH4_K));
    }
}

/// 2026-09-25: Without the `w8a16_gemm_m16_strided` handle the batch16 GEMV
/// serves, rather than a launch of a zero handle.
#[test]
fn native_fp8_qkv_attn_m16_tc_declines_without_its_entry_point() {
    let mut case = Case::m16_tc(16);
    case.m16_tc_handles = false;
    check_dispatch(&case, Expect::Batched(BATCH16_K));
}

/// 2026-09-25: With `m16_tc` off the batch16 GEMV serves.
#[test]
fn native_fp8_qkv_without_the_attn_lever_stays_on_batch16() {
    for rows in [5, 16] {
        check_dispatch(&Case::new(rows), Expect::Batched(BATCH16_K));
    }
}

/// 2026-09-25: On this route too, the phase costs the same number of launches
/// at 16 rows as at 2.
#[test]
fn native_fp8_qkv_attn_m16_tc_phase_launch_count_is_row_independent() {
    let baseline = qkv_phase_launches(&Case::new(2));
    for rows in [4, 8, 12, 16] {
        assert_eq!(
            qkv_phase_launches(&Case::m16_tc(rows)),
            baseline,
            "tensor-core route, rows={rows}"
        );
    }
}
