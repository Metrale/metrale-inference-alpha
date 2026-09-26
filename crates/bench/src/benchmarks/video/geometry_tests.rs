// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for the proportionality check.
//!
//! Owner: bench, video.
//! Invariants: none beyond the types.

use super::*;

/// 2026-09-26: 224x224 at patch 16 / merge 2: a 14x14 patch grid, 7x7 merged,
/// 49 tokens per temporal group. Every clip is 224x224, so this is the
/// driver's `plane`.
#[test]
fn a_224_square_group_is_49_tokens() {
    assert_eq!(tokens_per_group(224, 224, 16, 2), 49);
}

/// 2026-09-26: The check recovers the group count and overhead for any
/// groups-per-unit count (1 to 12) and overhead, so it needs no sampling rate.
#[test]
fn the_check_holds_at_every_sampling_rate() {
    let plane = 49;
    for unit in 1..=12usize {
        for overhead in [0usize, 7, 21, 64] {
            let r = check_proportional(
                overhead + unit * plane,
                overhead + 2 * unit * plane,
                overhead + 4 * unit * plane,
                plane,
            )
            .unwrap_or_else(|| panic!("unit={unit} overhead={overhead} did not resolve"));
            assert_eq!(r.unit_groups, unit);
            assert_eq!(r.overhead, overhead);
        }
    }
}

/// 2026-09-26: A sampler giving 1, 2 and 3 groups for 1x, 2x and 4x is not
/// proportional and is refused.
#[test]
fn a_non_proportional_sampler_is_caught() {
    let plane = 49;
    let overhead = 21;
    assert!(
        check_proportional(
            overhead + plane,
            overhead + 2 * plane,
            overhead + 3 * plane,
            plane
        )
        .is_none(),
        "a 1:2:3 progression must not read as proportional"
    );
}

/// 2026-09-26: A clip that costs the same as a shorter one (sampling collapsed
/// to a fixed frame count) is refused rather than resolved to zero groups.
#[test]
fn a_fixed_frame_count_regardless_of_duration_is_caught() {
    let plane = 49;
    let same = 21 + 4 * plane;
    assert!(check_proportional(same, same, same, plane).is_none());
}

/// 2026-09-26: A difference that is not a whole number of groups cannot
/// describe whole temporal groups.
#[test]
fn a_non_multiple_difference_is_refused() {
    assert!(check_proportional(21 + 30, 21 + 60, 21 + 120, 49).is_none());
}

#[test]
fn descending_totals_and_a_zero_plane_are_refused() {
    assert!(
        check_proportional(100, 50, 200, 49).is_none(),
        "2x below 1x"
    );
    assert!(check_proportional(50, 100, 90, 49).is_none(), "4x below 2x");
    assert!(check_proportional(50, 100, 200, 0).is_none(), "zero plane");
}

/// 2026-09-26: An implied negative overhead is refused.
#[test]
fn an_impossible_overhead_is_refused() {
    let plane = 49;
    // 2026-09-26: The differences say 2 groups per unit, but the 1x total is
    // smaller than those 2 groups.
    assert!(check_proportional(plane, 3 * plane, 7 * plane, plane).is_none());
}

#[test]
fn arithmetic_overflow_is_refused_instead_of_panicking() {
    let too_large_to_double = usize::MAX / 2 + 1;
    assert!(check_proportional(0, too_large_to_double, usize::MAX, 1).is_none());
}
