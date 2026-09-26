// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests for the high-speed-swap window helpers `check_safe_to_evict`
//! and `advance_layer_cursors_after_slide`.
//!
//! Owner: model-engine.
//! Invariants: none beyond the types.

use super::*;

#[test]
fn safe_to_evict_when_all_layers_caught_up() {
    // 2026-09-25: A cursor counts offloaded blocks: 6 means positions 0..=5 are
    // on disk, so position 5 may leave the window.
    let cursors = vec![6, 6, 6];
    assert!(check_safe_to_evict(&cursors, 5).is_ok());
}

#[test]
fn unsafe_to_evict_when_a_layer_lags() {
    // 2026-09-25: Layer 1's cursor 5 covers positions 0..=4, so position 5 is
    // not yet offloaded.
    let cursors = vec![10, 5, 10];
    let err = check_safe_to_evict(&cursors, 5).unwrap_err().to_string();
    assert!(err.contains("attention layer 1"), "got: {err}");
    assert!(err.contains("position 5"), "got: {err}");
}

#[test]
fn unsafe_to_evict_when_a_layer_never_offloaded() {
    // 2026-09-25: Cursor 0: the layer has offloaded nothing, so no position
    // may leave the window.
    let cursors = vec![10, 10, 0];
    let err = check_safe_to_evict(&cursors, 0).unwrap_err().to_string();
    assert!(err.contains("attention layer 2"), "got: {err}");
}

#[test]
fn safe_to_evict_with_empty_cursor_vec_is_vacuously_true() {
    // 2026-09-25: With no layer cursors there is no lagging layer, so every
    // position passes.
    let cursors: Vec<u32> = vec![];
    assert!(check_safe_to_evict(&cursors, 100).is_ok());
}

#[test]
fn advance_after_slide_promotes_lagging_cursors() {
    let mut cursors = vec![10, 5, 8];
    advance_layer_cursors_after_slide(&mut cursors, 9);
    assert_eq!(cursors, vec![10, 9, 9]);
}

#[test]
fn advance_after_slide_never_moves_cursor_backward() {
    let mut cursors = vec![100, 100, 100];
    advance_layer_cursors_after_slide(&mut cursors, 50);
    assert_eq!(cursors, vec![100, 100, 100]);
}
