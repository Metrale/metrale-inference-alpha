// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests for the drafter tensor-core GEMV routing and its kill-switch rule.
//!
//! Owner: model-layers ops.
//! Invariants: none beyond the types.

use super::*;

/// 2026-09-25: `(N, K)` of the eight BF16 projections of one Qwen3.6/3.8-27B
/// drafter position, in order fc, q+gate, k, v, o, gate, up, down (hidden
/// 5120; 24 query heads of 256 with an output gate; 4 KV heads of 256;
/// intermediate 17408).
const DRAFTER_27B: [(u32, u32); 8] = [
    (5120, 10240),
    (12288, 5120),
    (1024, 5120),
    (1024, 5120),
    (5120, 6144),
    (17408, 5120),
    (17408, 5120),
    (5120, 17408),
];

const ALL: [bool; 3] = [true, true, true];

#[test]
fn ships_on_and_any_non_empty_kill_value_turns_it_off() {
    use std::ffi::OsStr;
    assert!(mtp_tc_from(None), "unset is ON");
    assert!(mtp_tc_from(Some(OsStr::new(""))), "exported-empty is ON");
    assert!(!mtp_tc_from(Some(OsStr::new("1"))), "1 kills");
    assert!(
        !mtp_tc_from(Some(OsStr::new("0"))),
        "0 kills too (presence rule)"
    );
    assert!(
        !mtp_tc_from(Some(OsStr::new("true"))),
        "any non-empty value kills"
    );
}

#[test]
fn every_drafter_shape_routes_to_the_narrowest_entry() {
    for (n, k) in DRAFTER_27B {
        for m in MIN_M..=32 {
            let want = if m <= 8 {
                DtcKind::M8
            } else if m <= 16 {
                DtcKind::M16
            } else {
                DtcKind::M32
            };
            assert_eq!(route(m, n, k, true, ALL), Some(want), "m={m} n={n} k={k}");
        }
    }
}

#[test]
fn off_or_out_of_range_declines() {
    assert_eq!(route(4, 5120, 5120, false, ALL), None, "switch off");
    assert_eq!(route(0, 5120, 5120, true, ALL), None, "m=0");
    for (n, k) in DRAFTER_27B {
        assert_eq!(route(1, n, k, true, ALL), None, "m=1 n={n} k={k}");
    }
    assert_eq!(
        route(33, 5120, 5120, true, ALL),
        None,
        "m above the widest entry"
    );
    assert_eq!(route(4, 0, 5120, true, ALL), None, "n=0");
    assert_eq!(route(4, 5120, 5120 + 32, true, ALL), None);
    assert_eq!(route(4, 5120, 5120 + 8, true, ALL), None);
    // 2026-09-25: An N that is not a multiple of `ROWS_PER_CTA` routes; the
    // kernel guards the partial weight tile.
    assert_eq!(route(4, 1000, 5120, true, ALL), Some(DtcKind::M8));
}

#[test]
fn missing_entries_fall_back_to_a_wider_one_or_decline() {
    assert_eq!(
        route(4, 5120, 5120, true, [false, true, true]),
        Some(DtcKind::M16)
    );
    assert_eq!(
        route(4, 5120, 5120, true, [false, false, true]),
        Some(DtcKind::M32)
    );
    assert_eq!(route(12, 5120, 5120, true, [true, false, false]), None);
    assert_eq!(route(2, 5120, 5120, true, [false, false, false]), None);
}
