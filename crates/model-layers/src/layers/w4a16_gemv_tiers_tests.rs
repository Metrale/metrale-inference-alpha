// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests for the narrow W4A16 GEMV tier decision.
//!
//! Owner: model-layers (NVFP4 GEMV dispatch).
//! Invariants: none beyond the types.
//!
//! The narrow tiers are one template at different `MAX_M`, so the behaviour
//! under test is which tier `m` picks. The tests call the pure `select_tier`
//! rather than reading the process env, so both values of the kill switch
//! run in one process.

use super::{W4A16_BATCHM_WIDTHS, W4a16BatchmTiers, select_tier};
use metrale_gpu_runtime::gpu::mock::MockGpuBackend;

/// 2026-09-25: Every tier resolved.
const ALL: [bool; 5] = [true; 5];
/// 2026-09-25: Tiers 5/6/7 absent.
const LEGACY: [bool; 5] = [true, false, false, false, true];

/// 2026-09-25: Width the decision picks, or `None`, so the tests read as
/// `m -> width` rather than `m -> index`.
fn pick(m: u32, present: [bool; 5], exact_m: bool) -> Option<u32> {
    select_tier(m, present, exact_m).map(|i| W4A16_BATCHM_WIDTHS[i])
}

/// 2026-09-25: With every tier loaded, M=5/6/7 pick their exact tier; the
/// `kill_switch_*` tests below cover the disabled case.
#[test]
fn exact_m_rows_pick_their_own_tier() {
    assert_eq!(pick(5, ALL, true), Some(5));
    assert_eq!(pick(6, ALL, true), Some(6));
    assert_eq!(pick(7, ALL, true), Some(7));
}

/// 2026-09-25: M<=4 stays on batch4 and M=8 on batch8.
#[test]
fn pre_existing_widths_are_untouched() {
    for m in 1..=4 {
        assert_eq!(pick(m, ALL, true), Some(4), "m={m} must stay on batch4");
    }
    assert_eq!(pick(8, ALL, true), Some(8));
}

/// 2026-09-25: Above 8 rows, and at M=0, the narrow family declines. A narrow
/// tier given more rows than its `MAX_M` leaves the extra rows unwritten.
#[test]
fn out_of_family_widths_decline() {
    assert_eq!(pick(0, ALL, true), None);
    assert_eq!(pick(9, ALL, true), None);
    assert_eq!(pick(16, ALL, true), None);
    assert_eq!(pick(32, ALL, true), None);
}

/// 2026-09-25: With the exact-M tiers disabled the decision is `1..=4 => batch4`,
/// `5..=8 => batch8`, although tiers 5/6/7 are present.
#[test]
fn kill_switch_restores_the_shipped_decision() {
    for m in 1..=4 {
        assert_eq!(pick(m, ALL, false), Some(4), "m={m}");
    }
    for m in 5..=8 {
        assert_eq!(pick(m, ALL, false), Some(8), "m={m}");
    }
    assert_eq!(pick(9, ALL, false), None);
}

/// 2026-09-25: The kill switch and a target without tiers 5/6/7 pick the same
/// widths, so an A/B on the switch measures only the tiers.
#[test]
fn kill_switch_matches_a_legacy_target() {
    for m in 0..=10 {
        assert_eq!(pick(m, ALL, false), pick(m, LEGACY, true), "m={m}");
    }
}

/// 2026-09-25: 5/6/7 are chosen only when the loaded target resolved them. A
/// target with a partial set widens to the next resolved tier.
#[test]
fn absent_tiers_widen_to_the_next_resolved_one() {
    let partial = [true, false, false, true, true];
    assert_eq!(pick(5, partial, true), Some(7));
    assert_eq!(pick(6, partial, true), Some(7));
    assert_eq!(pick(7, partial, true), Some(7));
    assert_eq!(pick(8, partial, true), Some(8));
    // 2026-09-25: Only 5/6/7 present: M<=4 picks 5, and M=8 finds nothing.
    let no_legacy = [false, true, true, true, false];
    assert_eq!(pick(4, no_legacy, true), Some(5));
    assert_eq!(pick(8, no_legacy, true), None);
}

/// 2026-09-25: Nothing resolved: the family declines at every width, with the
/// switch on or off.
#[test]
fn empty_family_declines_everywhere() {
    for m in 0..=9 {
        assert_eq!(pick(m, [false; 5], true), None, "m={m}");
        assert_eq!(pick(m, [false; 5], false), None, "m={m}");
    }
}

/// 2026-09-25: A default table has no base tier, and `kernel` and `width` decline
/// at every m in 0..=9.
#[test]
fn default_table_is_empty_and_declines() {
    let t = W4a16BatchmTiers::default();
    assert!(!t.has_base());
    for m in 0..=9 {
        assert_eq!(t.kernel(m).0, 0, "m={m}");
        assert_eq!(t.width(m), None, "m={m}");
    }
}

/// 2026-09-25: The width table is sorted ascending, and `resolve` requests the same
/// family in the same order. The batch8 slot asks for `w4a16_gemv_batch8_rt2`
/// first; on the mock every lookup resolves, so there is one request per width.
#[test]
fn width_table_and_resolver_stay_in_lockstep() {
    assert!(W4A16_BATCHM_WIDTHS.windows(2).all(|w| w[0] < w[1]));
    let gpu = MockGpuBackend::new();
    let tiers = W4a16BatchmTiers::resolve(&gpu);
    assert!(tiers.handles.iter().all(|h| h.0 != 0));
    // 2026-09-25: Then one lookup of `w4a16_gemv_batch16`, which `resolve` always
    // makes (with `--w4a4-downcast` off, nothing else is looked up).
    let mut expected: Vec<(String, String)> = W4A16_BATCHM_WIDTHS
        .map(|w| {
            let func = if w == 8 {
                "w4a16_gemv_batch8_rt2".to_owned()
            } else {
                format!("w4a16_gemv_batch{w}")
            };
            ("w4a16_gemv".to_owned(), func)
        })
        .to_vec();
    expected.push(("w4a16_gemv".to_owned(), "w4a16_gemv_batch16".to_owned()));
    assert_eq!(gpu.kernel_lookups_snapshot(), expected);
    assert_ne!(tiers.wide.0, 0);
}
