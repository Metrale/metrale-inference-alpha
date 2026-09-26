// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests for the M16 tensor-core rows of the target table:
//! `ffn_m16_tc`, `attn_m16_tc`, `lm_head_m16_tc`, `attn_ncol_gemv` and the
//! `METRALE_M16_TC` umbrella.
//!
//! A child of `target_defaults_tests`, so `HOPPER`, `GB10`, `with`, `empty` and
//! `format_levers` come from there.
//!
//! Owner: model-layers ops (target serving defaults).
//! Invariants: none beyond the types.

use super::*;

/// 2026-09-25: Hopper and gb10 both declare the dense-FFN `w8a16_gemm_m16` arm
/// off; Hopper's row reports the target as its source.
#[test]
fn hopper_leaves_the_ffn_tensor_core_arm_off_by_declaration() {
    assert!(!empty(&HOPPER).ffn_m16_tc.value);
    assert!(!empty(&HOPPER).ffn_m16_tc.from_env());
    assert!(!empty(&GB10).ffn_m16_tc.value);
}

/// 2026-09-25: `METRALE_FFN_M16_TC` overrides the row both ways and reports the
/// environment as its source.
#[test]
fn the_ffn_tensor_core_arm_is_overridable_in_both_directions() {
    let on = with(&HOPPER, &[("METRALE_FFN_M16_TC", "1")]);
    assert!(on.ffn_m16_tc.value && on.ffn_m16_tc.from_env());
    let armed = TargetDefaults {
        ffn_m16_tc: true,
        ..HOPPER
    };
    let off = with(&armed, &[("METRALE_FFN_M16_TC", "0")]);
    assert!(!off.ffn_m16_tc.value && off.ffn_m16_tc.from_env());
}

/// 2026-09-25: The `METRALE_M16_TC` umbrella arms the FFN row when the row's own
/// variable is absent.
#[test]
fn the_m16_umbrella_arms_the_ffn_arm_too() {
    let on = with(&HOPPER, &[("METRALE_M16_TC", "1")]);
    assert!(on.ffn_m16_tc.value && on.ffn_m16_tc.from_env());
    // 2026-09-25: The row's own variable wins when both are set.
    let narrow_off = with(
        &HOPPER,
        &[("METRALE_FFN_M16_TC", "0"), ("METRALE_M16_TC", "1")],
    );
    assert!(!narrow_off.ffn_m16_tc.value);
}

/// 2026-09-25: The boot line names the row, with ` (env)` when the environment
/// set it.
#[test]
fn the_serve_line_names_the_ffn_tensor_core_row() {
    assert!(format_levers(&empty(&HOPPER)).contains("ffn_m16_tc=off"));
    assert!(
        format_levers(&with(&HOPPER, &[("METRALE_FFN_M16_TC", "1")]))
            .contains("ffn_m16_tc=on (env)")
    );
}

/// 2026-09-25: Hopper declares the attention tiers on, independently of the FFN
/// row; gb10 declares them off.
#[test]
fn hopper_arms_the_attention_tensor_core_tiers_by_declaration() {
    let h = empty(&HOPPER);
    assert!(h.attn_m16_tc.value && !h.attn_m16_tc.from_env());
    assert!(!h.ffn_m16_tc.value, "the two rows are independent");
    assert!(!empty(&GB10).attn_m16_tc.value);
}

/// 2026-09-25: `METRALE_ATTN_M16_TC=0` turns the attention tiers off and the
/// boot line tags it ` (env)`.
#[test]
fn the_attention_tiers_are_disarmable_from_the_environment() {
    let off = with(&HOPPER, &[("METRALE_ATTN_M16_TC", "0")]);
    assert!(!off.attn_m16_tc.value && off.attn_m16_tc.from_env());
    assert!(format_levers(&off).contains("attn_m16_tc=off (env)"));
}

/// 2026-09-25: The umbrella arms both rows on gb10, sourced from the environment.
#[test]
fn the_m16_umbrella_arms_both_halves() {
    let both = with(&GB10, &[("METRALE_M16_TC", "1")]);
    assert!(both.ffn_m16_tc.value && both.attn_m16_tc.value);
    assert!(both.ffn_m16_tc.from_env() && both.attn_m16_tc.from_env());
}

/// 2026-09-25: Hopper declares the BF16 decode head's tensor-core arm on and gb10
/// off; `METRALE_M16_TC` does not reach it, and `METRALE_LM_HEAD_M16_TC=0` turns
/// it off.
#[test]
fn hopper_arms_the_tensor_core_head_and_the_umbrella_does_not() {
    assert!(empty(&HOPPER).lm_head_m16_tc.value);
    assert!(!empty(&GB10).lm_head_m16_tc.value);
    let umbrella = with(&GB10, &[("METRALE_M16_TC", "1")]);
    assert!(
        !umbrella.lm_head_m16_tc.value,
        "METRALE_M16_TC is round 6's, and round 6 did not measure the head"
    );
    let off = with(&HOPPER, &[("METRALE_LM_HEAD_M16_TC", "0")]);
    assert!(!off.lm_head_m16_tc.value && off.lm_head_m16_tc.from_env());
}

/// 2026-09-25: The N-column GEMV row is off on Hopper and gb10, and
/// `METRALE_ATTN_NCOL_GEMV` turns it on.
#[test]
fn the_ncol_gemv_row_is_off_everywhere_and_armable() {
    assert!(!empty(&HOPPER).attn_ncol_gemv.value);
    assert!(!empty(&GB10).attn_ncol_gemv.value);
    let on = with(&HOPPER, &[("METRALE_ATTN_NCOL_GEMV", "1")]);
    assert!(on.attn_ncol_gemv.value && on.attn_ncol_gemv.from_env());
}

/// 2026-09-25: `METRALE_NO_ATTN_DECODE_BATCH` outranks both the declaration and
/// `METRALE_ATTN_NCOL_GEMV`.
#[test]
fn the_attention_decode_batch_kill_switch_outranks_the_row() {
    let armed = TargetDefaults {
        attn_ncol_gemv: true,
        ..HOPPER
    };
    for env in [
        vec![("METRALE_NO_ATTN_DECODE_BATCH", "1")],
        vec![
            ("METRALE_NO_ATTN_DECODE_BATCH", "1"),
            ("METRALE_ATTN_NCOL_GEMV", "1"),
        ],
    ] {
        let l = with(&armed, &env);
        assert!(!l.attn_ncol_gemv.value, "{env:?}");
        assert!(l.attn_ncol_gemv.from_env(), "{env:?}");
    }
}
