// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests for the `fp8_act_quant_hopper` row of the target table:
//! its declaration, its override and its boot-line spelling. The CTA floor the
//! row arms is tested in `fp8_act_quant_tests.rs`.
//!
//! A child of `target_defaults_tests`, so `HOPPER`, `GB10`, `with`, `empty` and
//! `format_levers` come from there.
//!
//! Owner: model-layers ops (target serving defaults).
//! Invariants: none beyond the types.

use super::*;

/// 2026-09-25: Hopper declares the twin on and gb10 off, both sourced from
/// the target. `fp8_act_quant_hopper.cu` exists only under `kernels/hopper`, so
/// on gb10 `try_target_kernel` returns 0 and the row changes nothing.
#[test]
fn hopper_arms_the_fp8_act_quant_twin_and_gb10_does_not() {
    assert!(empty(&HOPPER).fp8_act_quant_hopper.value);
    assert!(!empty(&GB10).fp8_act_quant_hopper.value);
    assert!(!empty(&HOPPER).fp8_act_quant_hopper.from_env());
    assert!(!empty(&GB10).fp8_act_quant_hopper.from_env());
}

/// 2026-09-25: `METRALE_FP8_ACT_QUANT_HOPPER` overrides the row both ways:
/// every off spelling turns it off on Hopper, and any other value, including
/// empty, turns it on on gb10.
#[test]
fn the_environment_overrides_the_row_in_both_directions() {
    for off in ["0", "false", "off", "no", "OFF", " 0 "] {
        assert_eq!(
            with(&HOPPER, &[("METRALE_FP8_ACT_QUANT_HOPPER", off)]).fp8_act_quant_hopper,
            Resolved::env(false),
            "METRALE_FP8_ACT_QUANT_HOPPER={off:?} must kill the twin",
        );
    }
    for on in ["1", "true", "on", "yes", ""] {
        assert_eq!(
            with(&GB10, &[("METRALE_FP8_ACT_QUANT_HOPPER", on)]).fp8_act_quant_hopper,
            Resolved::env(true),
            "METRALE_FP8_ACT_QUANT_HOPPER={on:?} must arm the A/B on a target \
             that declares it off",
        );
    }
}

/// 2026-09-25: The boot line reports the row in both states, with ` (env)`
/// when the environment set it.
#[test]
fn the_serve_line_names_the_row_and_marks_an_override() {
    let line = format_levers(&empty(&HOPPER));
    assert!(line.contains("fp8_act_quant_hopper=on"), "{line}");
    assert!(!line.contains("fp8_act_quant_hopper=on (env)"), "{line}");
    let line = format_levers(&empty(&GB10));
    assert!(line.contains("fp8_act_quant_hopper=off"), "{line}");
    let line = format_levers(&with(&HOPPER, &[("METRALE_FP8_ACT_QUANT_HOPPER", "0")]));
    assert!(line.contains("fp8_act_quant_hopper=off (env)"), "{line}");
}
