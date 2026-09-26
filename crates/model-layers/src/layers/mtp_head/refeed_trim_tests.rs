// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests of `mtp_rows_to_trim`, the drafter-row trim after a
//! verify, with and without `refeed_accepted`.
//!
//! Owner: model-layers (MTP head).
//! Invariants: none beyond the types.

use super::mtp_rows_to_trim;

#[test]
fn flag_off_is_exactly_the_legacy_behaviour() {
    // 2026-09-25: Without `refeed_accepted`, only the rejected rows go.
    assert_eq!(mtp_rows_to_trim(1, 0, false), 1);
    assert_eq!(mtp_rows_to_trim(1, 1, false), 0);
    assert_eq!(mtp_rows_to_trim(2, 0, false), 2);
    assert_eq!(mtp_rows_to_trim(2, 1, false), 1);
    assert_eq!(mtp_rows_to_trim(2, 2, false), 0);
    assert_eq!(mtp_rows_to_trim(3, 3, false), 0);
}

#[test]
fn flag_on_also_drops_accepted_rows_past_the_first() {
    // 2026-09-25: The first accepted draft's row was built from the target
    // hidden, so it stays.
    assert_eq!(mtp_rows_to_trim(1, 1, true), 0);
    assert_eq!(mtp_rows_to_trim(2, 1, true), 1);
    assert_eq!(mtp_rows_to_trim(2, 2, true), 1);
    assert_eq!(mtp_rows_to_trim(3, 2, true), 2);
    assert_eq!(mtp_rows_to_trim(3, 3, true), 2);
}

#[test]
fn full_reject_is_identical_with_and_without_the_flag() {
    // 2026-09-25: Nothing was accepted, so no accepted row carries a
    // drafter hidden.
    for d in 0..8 {
        assert_eq!(mtp_rows_to_trim(d, 0, true), mtp_rows_to_trim(d, 0, false));
    }
}

#[test]
fn never_trims_more_rows_than_were_drafted() {
    for d in 0..8 {
        for a in 0..=d + 2 {
            assert!(mtp_rows_to_trim(d, a, true) <= d, "d={d} a={a}");
            assert!(mtp_rows_to_trim(d, a, false) <= d, "d={d} a={a}");
        }
    }
}
