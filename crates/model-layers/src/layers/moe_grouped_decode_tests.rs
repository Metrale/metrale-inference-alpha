// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests for `moe_grouped_decode_decide`: the arm runs at and above
//! `moe_grouped_decode_min_rows()`, stays off below it unless forced, and never
//! runs when disabled. They call the pure decider, so none depends on process
//! env or on `OnceLock` latch order.
//!
//! Owner: model-layers (MoE).
//! Invariants: none beyond the types.

use super::{moe_grouped_decode_decide, moe_grouped_decode_min_rows};

/// 2026-09-25: At and above the threshold the arm runs. A width-blind gate
/// (`enabled` alone) also passes this test; the next one catches it.
#[test]
fn the_grouped_arm_engages_at_and_above_the_measured_width() {
    let w = moe_grouped_decode_min_rows();
    assert!(
        moe_grouped_decode_decide(w, true, false),
        "n={w} is the smallest width measured on the winning side (+25%); \
         the arm must engage there"
    );
    assert!(
        moe_grouped_decode_decide(w * 4, true, false),
        "wider is more favourable, not less — the amortisation only improves"
    );
}

/// 2026-09-25: Below the threshold the arm stays off. A width-blind gate
/// (`enabled` alone) fails this test.
#[test]
fn the_grouped_arm_stays_off_below_the_measured_width() {
    for n in [1usize, 2, 4, 8, moe_grouped_decode_min_rows() - 1] {
        assert!(
            !moe_grouped_decode_decide(n, true, false),
            "n={n} is below the measured-winning width; at n=4 this arm is a \
             ~45% LOSS (31 vs 56 tok/s), and n=5..15 is unmeasured — the gate \
             must fall to the per-token loop"
        );
    }
}

#[test]
fn the_kill_switch_wins_over_any_width() {
    for n in [1usize, 4, 16, 64, 1024] {
        assert!(
            !moe_grouped_decode_decide(n, false, false),
            "kill switch set: the arm must not engage at n={n} regardless of width"
        );
        assert!(
            !moe_grouped_decode_decide(n, false, true),
            "kill switch must also beat the diagnostic force at n={n}, or the \
             two knobs contradict each other"
        );
    }
}

/// 2026-09-25: The force reaches below the threshold. Dropping `|| forced` from
/// the decider fails this test.
#[test]
fn the_force_override_reaches_below_the_threshold() {
    assert!(
        moe_grouped_decode_decide(4, true, true),
        "METRALE_MOE_GROUPED_DECODE=1 exists so a below-threshold width can be \
         measured; if it cannot reach n=4 it cannot measure the loss it was \
         used to find"
    );
}

/// 2026-09-25: Pins the threshold at 16, so changing it also means editing this
/// test.
#[test]
fn the_threshold_is_the_measured_width() {
    assert_eq!(
        moe_grouped_decode_min_rows(),
        16,
        "16 is the smallest width measured on the winning side; changing it \
         needs a measurement, not a preference"
    );
}
