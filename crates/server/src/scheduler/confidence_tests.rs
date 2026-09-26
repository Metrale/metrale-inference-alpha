// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Unit tests for `confidence.rs`, included as its child module via
//! `#[path]`.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::*;

// 2026-09-25: Code-fence tracking and the confidence-run accumulator.

const FENCE: u32 = 71093;

#[test]
fn fence_toggles_on_fence_token() {
    assert!(
        toggle_code_fence(false, FENCE, Some(FENCE)),
        "``` opens fence"
    );
    assert!(
        !toggle_code_fence(true, FENCE, Some(FENCE)),
        "``` closes fence"
    );
}

#[test]
fn fence_unchanged_by_non_fence_token() {
    assert!(!toggle_code_fence(false, 42, Some(FENCE)));
    assert!(toggle_code_fence(true, 42, Some(FENCE)));
}

#[test]
fn fence_guard_disabled_when_no_fence_token() {
    // 2026-09-25: No single-token fence: the state never changes.
    assert!(!toggle_code_fence(false, FENCE, None));
}

#[test]
fn f2_arms_after_confidence_run_limit_tokens() {
    // 2026-09-25: CONFIDENCE_RUN_LIMIT consecutive confident tokens arm the stop.
    let mut run = 0;
    let mut fired = false;
    for _ in 0..CONFIDENCE_RUN_LIMIT {
        let (next, fire) = confidence_run_step(true, run, CONFIDENCE_RUN_LIMIT);
        run = next;
        fired |= fire;
    }
    assert_eq!(run, CONFIDENCE_RUN_LIMIT);
    assert!(
        fired,
        "CONFIDENCE_RUN_LIMIT consecutive confident tokens must arm F2"
    );
}

#[test]
fn f2_run_breaks_on_non_confident_token() {
    let (run, fire) = confidence_run_step(false, 25, CONFIDENCE_RUN_LIMIT);
    assert_eq!(run, 0);
    assert!(!fire);
}

#[test]
fn f2_accumulates_inside_code_too() {
    // 2026-09-25: The accumulator takes no fence state, so it counts the same inside
    // a fence; the `defer_*` tests cover the injection gate. One step short of
    // the limit, the run advances without arming.
    let (run, fire) = confidence_run_step(true, CONFIDENCE_RUN_LIMIT - 2, CONFIDENCE_RUN_LIMIT);
    assert_eq!(run, CONFIDENCE_RUN_LIMIT - 1);
    assert!(!fire);
    // 2026-09-25: At the limit it arms.
    let (run, fire) = confidence_run_step(true, CONFIDENCE_RUN_LIMIT - 1, CONFIDENCE_RUN_LIMIT);
    assert_eq!(run, CONFIDENCE_RUN_LIMIT);
    assert!(
        fire,
        "F2 arms even inside a fence; injection is what defers"
    );
}

// 2026-09-25: The injection gate, `should_inject_think_end`.

#[test]
fn defer_injection_while_in_code_fence() {
    assert!(
        !should_inject_think_end(true, true, true, false),
        "armed brake must NOT inject </think> mid-code-fence (would split a statement)"
    );
}

#[test]
fn inject_once_fence_closes_at_sentence_boundary() {
    assert!(
        should_inject_think_end(true, false, true, false),
        "armed brake fires cleanly once the ``` fence has closed AND a sentence boundary is reached"
    );
}

#[test]
fn defer_outside_fence_when_not_at_sentence_boundary() {
    assert!(
        !should_inject_think_end(true, false, false, false),
        "armed brake must NOT inject </think> mid-sentence (would corrupt reasoning)"
    );
}

#[test]
fn hard_override_breaks_unbounded_in_fence_defer() {
    // 2026-09-25: A fence that never closes would defer forever; `hard_override`
    // injects even inside it.
    assert!(
        should_inject_think_end(true, true, false, true),
        "armed + in-fence + budget massively overrun must HARD-inject </think>"
    );
    // 2026-09-25: Not armed: no injection, even with the override.
    assert!(!should_inject_think_end(false, true, false, true));
}

#[test]
fn hard_override_breaks_unbounded_sentence_defer() {
    // 2026-09-25: When `sentence_defer_count` reaches MAX_SENTENCE_DEFER_TOKENS the
    // caller folds it into `hard_override`, so a run without a boundary token
    // cannot defer forever.
    assert!(
        should_inject_think_end(true, false, false, true),
        "armed + outside fence + no boundary + hard_override → force-inject"
    );
}

#[test]
fn no_injection_when_not_armed() {
    for &in_fence in &[false, true] {
        for &at_boundary in &[false, true] {
            for &hard_override in &[false, true] {
                assert!(
                    !should_inject_think_end(false, in_fence, at_boundary, hard_override),
                    "not-armed must not inject: in_fence={in_fence}, at_boundary={at_boundary}, hard_override={hard_override}"
                );
            }
        }
    }
}

#[test]
fn boundary_at_least_one_path_eventually_fires() {
    // 2026-09-25: From each (in_fence, at_boundary) state, some armed input fires.
    assert!(should_inject_think_end(true, false, true, false));
    assert!(should_inject_think_end(true, false, false, true));
    assert!(should_inject_think_end(true, true, false, true));
    assert!(should_inject_think_end(true, true, true, true));
}
