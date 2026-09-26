// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Cases for the 17+ row arm of the batched native-FP8 Q/K/V tier, the 32-row M-tile `w8a16_gemm_pipelined_m32_strided`.
//!
//! Owner: model-layers (qwen3 attention, multi-sequence decode).
//! Invariants: none beyond the types.
//!
//! A child of `qkv_fp8_batch_tests.rs`, sharing its harness. Numerics are the
//! GPU oracle's (`examples/native_fp8_qkv_batch_microtest`, M32 leg).

use super::{
    BATCH4_K, BATCH16_K, Case, Expect, M16TC_STRIDED_K, M32_STRIDED_K, check_dispatch,
    qkv_phase_launches,
};
use crate::layers::qwen3_attention::attn_ncol_gemv::NcolWidth;

/// 2026-09-25: Every width above the GEMVs' MAX_M is one strided launch per
/// projection on the M32 tile. The shared assertions pin the argument layout
/// (A pitch = hidden, C pitch = per_seq_qkv) and one launch per projection.
#[test]
fn native_fp8_qkv_takes_the_m32_tile_above_sixteen_rows() {
    for rows in [17, 24, 32, 48, 64] {
        check_dispatch(&Case::new(rows), Expect::Batched(M32_STRIDED_K));
    }
}

/// 2026-09-25: The M32 tile starts above 16 rows: 16 stays on batch16, 4 on
/// batch4.
#[test]
fn native_fp8_qkv_m32_tile_leaves_the_gemv_bands_alone() {
    check_dispatch(&Case::new(16), Expect::Batched(BATCH16_K));
    check_dispatch(&Case::new(4), Expect::Batched(BATCH4_K));
}

/// 2026-09-25: The `m16_tc` and N-column arms serve at most 16 rows; at 17+ the
/// M32 tile serves even when they are on.
#[test]
fn native_fp8_qkv_m32_tile_outranks_the_sixteen_row_levers_above_their_band() {
    for rows in [17, 32] {
        check_dispatch(&Case::m16_tc(rows), Expect::Batched(M32_STRIDED_K));
        check_dispatch(
            &Case::ncol(rows, NcolWidth::Four),
            Expect::Batched(M32_STRIDED_K),
        );
    }
    // 2026-09-25: Inside their band they still win.
    check_dispatch(&Case::m16_tc(16), Expect::Batched(M16TC_STRIDED_K));
}

/// 2026-09-25: The tier's other guards apply to this arm unchanged: the kill
/// switch, per-row scales and unaligned dims all decline at 32 rows as they do
/// at 4.
#[test]
fn native_fp8_qkv_m32_tile_honours_the_shared_guards() {
    let mut off = Case::new(32);
    off.enabled = false;
    check_dispatch(&off, Expect::Scalar);
    let mut per_row = Case::new(32);
    per_row.format = crate::weight_map::WeightQuantFormat::Fp8PerRow;
    check_dispatch(&per_row, Expect::Scalar);
    let mut unaligned = Case::new(32);
    unaligned.width = 64;
    check_dispatch(&unaligned, Expect::Scalar);
}

/// 2026-09-25: The whole Q/K/V phase costs the same number of launches at 64
/// rows as at 2.
#[test]
fn native_fp8_qkv_phase_launch_count_is_row_independent_past_sixteen() {
    let baseline = qkv_phase_launches(&Case::new(2));
    for rows in [17, 24, 32, 64] {
        assert_eq!(
            qkv_phase_launches(&Case::new(rows)),
            baseline,
            "M32 tile route, rows={rows}"
        );
    }
}
