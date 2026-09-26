// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for the MoE gate's descriptor: it shares the dense
//! gates' floor params and driver, and is defined on the Qwen3.6-35B-A3B
//! family only. Listed in `gate::coverage::TEST_ONLY_RUST_MODULES`.
//!
//! Owner: bench (concurrency).
//! Invariants: none beyond the types.

use super::*;

/// 2026-09-26: Same metric names as the dense gate, or a MoE floor on
/// `c8_aggregate_tok_s` would gate nothing.
#[test]
fn the_moe_gate_shares_the_drivers_floor_pairing() {
    assert_eq!(MOE_DESCRIPTOR.id, "concurrency-sweep-moe");
    assert_eq!(
        MOE_DESCRIPTOR.threshold_params, DESCRIPTOR.threshold_params,
        "the MoE ladder must gate on the same metric names as the dense one"
    );
    assert_eq!(
        crate::registry::find("concurrency-sweep-moe").map(|d| d.id),
        Some("concurrency-sweep-moe"),
        "an unregistered gate id can be neither run nor owed"
    );
    // 2026-09-26: A `prompt_mode = "essay"` pin selects `Fixture::Essay`, whose
    // prompts `concurrency_essay_tests.rs` pins to the harness's bytes.
    let mut b = ConcurrencySweep::default();
    let mut v = ParamValues::defaults(&b.parameters());
    v.set("prompt_mode", ParamValue::Text("essay".into()));
    v.set("concurrencies", ParamValue::IntList(vec![1, 2, 4, 8, 16]));
    v.set("isls", ParamValue::IntList(vec![128]));
    v.set("osl", ParamValue::Int(1024));
    b.configure(&v).unwrap();
    assert_eq!(b.fixture, Fixture::Essay);
    assert_eq!(b.osl, 1024);
    assert_eq!(
        b.cells,
        vec![(128, 1), (128, 2), (128, 4), (128, 8), (128, 16)]
    );
    assert!(!b.floors.gating(), "an unmeasured entry fills no floor");
}

/// 2026-09-26: `ModelExpectation::accepts` is a case-insensitive substring
/// match on the family name.
#[test]
fn the_moe_gate_is_defined_on_the_moe_family_only() {
    let expect = MOE_DESCRIPTOR
        .intended_for
        .expect("the MoE ladder names the family it is defined on");
    assert!(expect.accepts("Qwen/Qwen3.6-35B-A3B-FP8"));
    assert!(expect.accepts("nvidia/Qwen3.6-35B-A3B-NVFP4"));
    assert!(!expect.accepts("unsloth/Qwen3.8-27B-NVFP4"));
    assert!(!expect.accepts("unsloth/Qwen3.6-27B-NVFP4"));
    assert!(
        DESCRIPTOR.intended_for.is_none(),
        "the plain gate stays family-agnostic; only the MoE and DFlash2 ids are pinned"
    );
}
