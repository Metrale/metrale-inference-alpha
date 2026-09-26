// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests of `PerSequenceState` arithmetic as preflight uses it,
//! with no GPU, model or checkpoint.
//!
//! Owner: server startup (`met serve`).
//! Invariants: none beyond the types.

use metrale_model_arch::seq_state_reserve::PerSequenceState;

#[test]
fn max_batch_size_is_applied_exactly_once() {
    let s = PerSequenceState {
        target_layers: 739_639_296,
        proposer: 67_574_292,
    };
    assert_eq!(s.total(), 807_213_588);
    assert_eq!(s.for_batch(3), 2_421_640_764);
    assert_ne!(s.for_batch(3), s.total() * 9);
}

#[test]
fn the_two_owners_stay_separate() {
    let s = PerSequenceState {
        target_layers: 739_639_296,
        proposer: 67_574_292,
    };
    // 2026-09-26: 806,879,232 is 12 indexer blocks
    // (`proposer_charge_at_131072_is_exact_and_is_a_separate_owner`); the
    // proposer owns one block plus small buffers, not twelve.
    assert_ne!(s.proposer, 806_879_232);
    assert!(s.proposer < s.target_layers / 10);
}

#[test]
fn a_model_that_owns_no_per_sequence_state_is_charged_nothing() {
    let s = PerSequenceState::default();
    assert_eq!(s.total(), 0);
    assert_eq!(s.for_batch(3), 0);
}
