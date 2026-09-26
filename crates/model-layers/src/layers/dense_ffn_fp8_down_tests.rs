// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: CPU tests for `fp8_down_arm`, the native-FP8 decode down-projection arm rule.
//!
//! Owner: model-layers (dense FFN).
//! Invariants: none beyond the types.
//!
//! They pin which arm every handle and lever combination selects, and that
//! the split-SiLU arm is not taken on a target that lacks a kernel it needs.

use super::fp8_down::{Fp8DownArm, fp8_down_arm};

/// 2026-09-25: A SiLU layer with every handle present (dual, fused SiLU,
/// `moe_silu_mul`, plain `w8a16_gemv`) and the given split-SiLU lever.
fn full(lever: bool) -> Fp8DownArm {
    fp8_down_arm(true, true, true, true, true, lever)
}

#[test]
fn the_split_silu_arm_is_the_default_on_a_complete_target() {
    assert_eq!(full(true), Fp8DownArm::SplitSilu);
}

#[test]
fn the_kill_switch_restores_the_fused_kernel() {
    // 2026-09-25: With `decode_split_silu` off (`METRALE_NO_DECODE_SPLIT_SILU`)
    // and the fused kernel present, the arm is `FusedSilu`, not
    // `PerProjection`.
    assert_eq!(full(false), Fp8DownArm::FusedSilu);
}

#[test]
fn a_target_without_the_staging_kernels_cannot_take_the_split_arm() {
    // 2026-09-25: No `moe_silu_mul`: nothing can stage silu(gate)*up.
    assert_eq!(
        fp8_down_arm(true, true, true, false, true, true),
        Fp8DownArm::FusedSilu
    );
    // 2026-09-25: No plain `w8a16_gemv`: nothing can consume a staged activation.
    assert_eq!(
        fp8_down_arm(true, true, true, true, false, true),
        Fp8DownArm::FusedSilu
    );
}

#[test]
fn losing_both_fused_kernels_falls_to_the_per_projection_path() {
    assert_eq!(
        fp8_down_arm(true, true, false, false, true, true),
        Fp8DownArm::PerProjection
    );
    assert_eq!(
        fp8_down_arm(true, true, false, true, false, true),
        Fp8DownArm::PerProjection
    );
}

#[test]
fn the_split_arm_survives_a_missing_fused_kernel() {
    // 2026-09-25: Without the fused `w8a16_gemv_silu_input` the split arm is
    // still taken.
    assert_eq!(
        fp8_down_arm(true, true, false, true, true, true),
        Fp8DownArm::SplitSilu
    );
}

#[test]
fn the_dual_gemv_gates_both_fused_arms() {
    // 2026-09-25: Without `w8a16_gemv_dual` there is no staged gate/up pair for
    // either fused arm to read, whatever else resolved.
    for lever in [true, false] {
        assert_eq!(
            fp8_down_arm(true, false, true, true, true, lever),
            Fp8DownArm::PerProjection
        );
    }
}

#[test]
fn a_non_silu_activation_never_reaches_the_fused_arms() {
    // 2026-09-25: A non-SiLU layer takes the per-projection path.
    assert_eq!(
        fp8_down_arm(false, true, true, true, true, true),
        Fp8DownArm::PerProjection
    );
}
