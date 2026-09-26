// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests of `check_rail_count`, without hardware: `RailSet`'s rails
//! need `Verbs::create`, which needs a NIC.
//!
//! Owner: metrale-gpu-sys.
//! Invariants: none beyond the types.

use super::check_rail_count;

#[test]
fn equal_counts_are_ok() {
    assert!(check_rail_count(2, 2, "peer").is_ok());
    assert!(check_rail_count(0, 0, "peer").is_ok());
}

#[test]
fn short_server_slice_is_refused_not_truncated() {
    // 2026-09-26: With one param for two rails, the `zip` in `complete` would
    // connect rail 0 only and leave rail 1 in INIT.
    let e = check_rail_count(1, 2, "test-peer").unwrap_err().to_string();
    assert!(e.contains("test-peer"), "peer named: {e}");
    assert!(
        e.contains("1 rail params for 2 client rails"),
        "counts reported: {e}"
    );
}

#[test]
fn long_server_slice_is_also_refused() {
    assert!(check_rail_count(3, 2, "peer").is_err());
}
