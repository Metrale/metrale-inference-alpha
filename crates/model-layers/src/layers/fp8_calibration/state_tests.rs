// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Unit tests for the FP8 KV calibration state machine (`CalibrationState`).
//!
//! Owner: model-layers (FP8 KV cache).
//! Invariants: none beyond the types.

use super::{CalibrationState, CalibrationStep, FP8_E4M3_MAX};

/// 2026-09-25: A 13-token batch counts toward a 256-token window without
/// freezing it, and the freeze uses the larger amax of the batch that
/// completes the window.
#[test]
fn readiness_probe_counts_toward_the_window_but_does_not_freeze_it() {
    let mut st = CalibrationState::new(256, 1.0, 2.0);

    assert_eq!(st.record(1.0, 0.5, 13), CalibrationStep::Stage);
    assert!(
        !st.frozen,
        "a 13-token probe must not end a 256-token window"
    );
    assert_eq!(st.tokens_seen, 13);
    assert_eq!(
        st.k_scale, 2.0,
        "provisional scale stays live while staging"
    );

    let step = st.record(8.0, 4.0, 243);
    assert_eq!(
        step,
        CalibrationStep::Freeze {
            k_scale: 8.0 / FP8_E4M3_MAX,
            v_scale: 4.0 / FP8_E4M3_MAX,
            tokens_seen: 256,
        },
        "the freeze must use the amax over all 256 observed tokens"
    );
    assert_eq!(st.staged_tokens, 13, "only the probe needed staging");
    assert_eq!(st.staged_batches, 1);
}

/// 2026-09-25: A first batch longer than the window freezes on the amax of the
/// whole batch.
#[test]
fn a_single_oversized_first_request_freezes_on_all_of_it() {
    let mut st = CalibrationState::new(256, 1.0, 2.0);
    let step = st.record(6.0, 3.0, 300);
    assert_eq!(
        step,
        CalibrationStep::Freeze {
            k_scale: 6.0 / FP8_E4M3_MAX,
            v_scale: 3.0 / FP8_E4M3_MAX,
            tokens_seen: 300,
        }
    );
    assert_eq!(
        st.staged_tokens, 0,
        "the batch that crosses the window is written at the frozen scale, \
         so it never needs staging or a rewrite"
    );
}

/// 2026-09-25: Small batches accumulate to the window; the freeze fires on the
/// observe that reaches it, and every earlier batch is staged.
#[test]
fn many_small_requests_accumulate_across_the_window() {
    let mut st = CalibrationState::new(64, 1.0, 2.0);
    for i in 0..7 {
        assert_eq!(
            st.record(1.0 + i as f32, 1.0, 9),
            CalibrationStep::Stage,
            "batch {i} is inside the window"
        );
    }
    assert_eq!(st.tokens_seen, 63);
    let step = st.record(2.0, 9.0, 1);
    assert_eq!(
        step,
        CalibrationStep::Freeze {
            k_scale: 7.0 / FP8_E4M3_MAX,
            v_scale: 9.0 / FP8_E4M3_MAX,
            tokens_seen: 64,
        },
        "K keeps the largest earlier max, V picks up the crossing batch's"
    );
    assert_eq!(st.staged_tokens, 63);
    assert_eq!(st.staged_batches, 7);
}

/// 2026-09-25: A 1-token window freezes on the first observe and stages nothing.
#[test]
fn a_one_token_window_freezes_immediately_like_before() {
    let mut st = CalibrationState::new(1, 1.0, 2.0);
    assert!(matches!(
        st.record(3.0, 2.0, 13),
        CalibrationStep::Freeze { .. }
    ));
    assert_eq!(st.staged_tokens, 0);
    assert_eq!(st.staged_batches, 0);
}

/// 2026-09-25: A 0 window is clamped to 1 and still freezes. A calibrator that
/// never froze would keep CUDA graphs suppressed (`graphs_ready_after_fp8_kv_cal`).
#[test]
fn a_zero_window_is_clamped_and_still_freezes() {
    let mut st = CalibrationState::new(0, 1.0, 2.0);
    assert_eq!(st.window_tokens, 1);
    assert!(matches!(
        st.record(3.0, 2.0, 1),
        CalibrationStep::Freeze { .. }
    ));
}

/// 2026-09-25: The frozen scale does not clip the data it was calibrated on:
/// `scale * 448 >= amax` for every observation in the window.
#[test]
fn the_frozen_scale_never_shrinks_below_the_pre_freeze_amax() {
    for headroom in [1.0_f32, 1.25, 2.0] {
        let mut st = CalibrationState::new(32, headroom, 2.0);
        let ks = [0.5_f32, 11.0, 3.0, 7.5];
        let vs = [9.0_f32, 1.0, 2.0, 0.25];
        for i in 0..3 {
            assert_eq!(st.record(ks[i], vs[i], 8), CalibrationStep::Stage);
        }
        let CalibrationStep::Freeze {
            k_scale, v_scale, ..
        } = st.record(ks[3], vs[3], 8)
        else {
            panic!("must freeze at the window");
        };
        let k_amax = ks.iter().copied().fold(0.0_f32, f32::max);
        let v_amax = vs.iter().copied().fold(0.0_f32, f32::max);
        assert!(
            k_scale * FP8_E4M3_MAX >= k_amax,
            "headroom {headroom}: k_scale {k_scale} clips the window's own amax {k_amax}"
        );
        assert!(
            v_scale * FP8_E4M3_MAX >= v_amax,
            "headroom {headroom}: v_scale {v_scale} clips the window's own amax {v_amax}"
        );
    }
}

/// 2026-09-25: `record` calls after the freeze do not move the scale.
#[test]
fn later_observes_never_move_a_frozen_scale() {
    let mut st = CalibrationState::new(16, 1.0, 2.0);
    let CalibrationStep::Freeze {
        k_scale, v_scale, ..
    } = st.record(4.0, 2.0, 16)
    else {
        panic!("must freeze at the window");
    };
    for _ in 0..40 {
        assert_eq!(st.record(400.0, 400.0, 16), CalibrationStep::Frozen);
    }
    assert_eq!((st.k_scale, st.v_scale), (k_scale, v_scale));
}

/// 2026-09-25: Every pre-freeze batch is observed. `observe` returns before
/// staging when `should_observe` is false, so a skipped batch would leave
/// entries the freeze cannot rewrite.
#[test]
fn every_pre_freeze_batch_is_observed() {
    let mut st = CalibrationState::new(1024, 1.0, 2.0);
    for _ in 0..50 {
        assert!(st.should_observe(3));
        st.record(1.0, 1.0, 3);
    }
    assert!(!st.frozen);
}
