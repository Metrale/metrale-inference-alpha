// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests for the `ffn_gateup_fused` row of the target table: its
//! declaration, its override and its boot-line spelling.
//!
//! A child of `target_defaults_tests`, so `HOPPER`, `GB10`, `with`, `empty` and
//! `format_levers` come from there.
//!
//! Owner: model-layers ops (target serving defaults).
//! Invariants: none beyond the types.

use super::*;

/// 2026-09-25: Hopper declares the fused gate+up GEMM on and gb10 off, both
/// sourced from the target.
#[test]
fn hopper_arms_the_fused_gate_up_gemm_and_gb10_does_not() {
    assert!(empty(&HOPPER).ffn_gateup_fused.value);
    assert!(!empty(&GB10).ffn_gateup_fused.value);
    assert!(!empty(&HOPPER).ffn_gateup_fused.from_env());
    assert!(!empty(&GB10).ffn_gateup_fused.from_env());
}

/// 2026-09-25: `METRALE_FFN_GATEUP_FUSED` overrides the row both ways: every
/// off spelling turns it off on Hopper, and any other value, including empty,
/// turns it on on gb10.
#[test]
fn the_environment_overrides_the_row_in_both_directions() {
    for off in ["0", "false", "off", "no", "OFF", " 0 "] {
        assert_eq!(
            with(&HOPPER, &[("METRALE_FFN_GATEUP_FUSED", off)]).ffn_gateup_fused,
            Resolved::env(false),
            "METRALE_FFN_GATEUP_FUSED={off:?} must kill the arm",
        );
    }
    for on in ["1", "true", "on", "yes", ""] {
        assert_eq!(
            with(&GB10, &[("METRALE_FFN_GATEUP_FUSED", on)]).ffn_gateup_fused,
            Resolved::env(true),
            "METRALE_FFN_GATEUP_FUSED={on:?} must arm the A/B on a target that \
             declares it off",
        );
    }
}

/// 2026-09-25: The boot line reports the row in both states, with ` (env)`
/// when the environment set it.
#[test]
fn the_serve_line_names_the_row_and_marks_an_override() {
    let line = format_levers(&empty(&HOPPER));
    assert!(line.contains("ffn_gateup_fused=on"), "{line}");
    assert!(!line.contains("ffn_gateup_fused=on (env)"), "{line}");
    let line = format_levers(&empty(&GB10));
    assert!(line.contains("ffn_gateup_fused=off"), "{line}");
    let line = format_levers(&with(&HOPPER, &[("METRALE_FFN_GATEUP_FUSED", "0")]));
    assert!(line.contains("ffn_gateup_fused=off (env)"), "{line}");
}
