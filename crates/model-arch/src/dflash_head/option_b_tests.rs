// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests for `option_b_from`: unset and every value except `0` select
//! the paged drafter cache.
//!
//! Owner: model-arch (DFlash drafter).
//! Invariants: none beyond the types.

use super::option_b_from;

#[test]
fn option_b_defaults_on_and_only_zero_turns_it_off() {
    assert!(
        option_b_from(None),
        "unset must be ON — this is the 9x line"
    );
    assert!(option_b_from(Some("1")));
    assert!(!option_b_from(Some("0")));
    assert!(
        option_b_from(Some("true")),
        "only the exact string 0 disables"
    );
    assert!(option_b_from(Some("")), "empty is not a kill switch");
}
