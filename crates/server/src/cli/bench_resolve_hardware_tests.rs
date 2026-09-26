// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for the `--hardware` axis of the gate's resolution,
//! against the box-class registry (`metrale_bench::hardware::ids`): which
//! ids resolve, and which refusal an unresolvable one gets.
//!
//! Owner: server CLI (`met benchmark`).
//! Invariants: none beyond the types.

use super::tests::baseline;
use super::*;

/// 2026-09-26: A class in `KNOWN_HARDWARE_IDS` that the baseline lacks gets
/// the "no record yet" refusal, not the unknown-class one.
#[test]
fn a_registered_box_class_with_no_record_says_to_go_measure_it() {
    let b = baseline(&[("gb10", "m", Some("r"))]);
    for hw in ["h100", "h200", "b200"] {
        let err = resolve(&b, "bfcl-subset", Some(hw), None).expect_err("refused");
        let msg = format!("{err:#}");
        assert!(msg.contains(hw), "names what was asked for: {msg}");
        assert!(msg.contains("gb10"), "lists what it has: {msg}");
        assert!(
            msg.contains("no record yet"),
            "names the state, not a typo: {msg}"
        );
        assert!(
            !msg.contains("not a box class Metrale Engine knows"),
            "a registered id must not read as a typo: {msg}"
        );
    }
}

/// 2026-09-26: A registered class that has a baseline entry resolves normally.
#[test]
fn a_registered_box_class_with_a_record_resolves_normally() {
    let b = baseline(&[
        ("gb10", "a", Some("recipe-a")),
        ("h100", "b", Some("recipe-b")),
    ]);
    let r = resolve(&b, "bfcl-subset", Some("h100"), None).expect("resolved");
    assert_eq!(r.model, "b");
    assert_eq!(r.recipe_id, "recipe-b");
}
