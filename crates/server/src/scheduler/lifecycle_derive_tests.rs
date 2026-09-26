// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The precedence matrix of the pure `derive_finish_reason`.
//!
//! These tests build no `ActiveSeq` and no model. The tests that run
//! `finish_sequence` itself are in `lifecycle_tests.rs`.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::super::lifecycle::derive_finish_reason;
use super::super::types::GUARD_STOP_REQUEST_TIMEOUT;
use super::{EOS, MAX_SEQ_LEN, TOOL_END};
use crate::ir::FINISH_REASON_TIMEOUT;

/// 2026-09-25: Common-case shorthand: mid-context position (no seqlen ceiling), so
/// the budget dimension under test is the `remaining` countdown.
fn derive(guard: Option<&'static str>, last: Option<u32>, remaining: usize) -> &'static str {
    derive_finish_reason(guard, last, EOS, TOOL_END, remaining, 10, MAX_SEQ_LEN)
}

#[test]
fn length_means_budget_exhausted_only() {
    // 2026-09-25: max_tokens countdown exhausted on an ordinary content token.
    assert_eq!(derive(None, Some(42), 0), "length");
    // 2026-09-25: Served context ceiling reached with budget left — the other
    // half of the hard-ceiling stop predicate.
    assert_eq!(
        derive_finish_reason(
            None,
            Some(42),
            EOS,
            TOOL_END,
            500,
            MAX_SEQ_LEN - 1,
            MAX_SEQ_LEN
        ),
        "length"
    );
}

#[test]
fn early_stop_one_token_short_of_budget_is_not_length() {
    // 2026-09-25: One token of budget left is not exhausted, so a non-EOS,
    // non-tool-end stop with remaining=1 must not read as "length".
    let r = derive(None, Some(42), 1);
    assert_ne!(r, "length");
    // 2026-09-25: No guard and budget left reads as "stop": nothing was
    // truncated.
    assert_eq!(r, "stop");
}

#[test]
fn normal_stops_are_unchanged() {
    assert_eq!(derive(None, Some(151645), 100), "stop");
    assert_eq!(derive(None, Some(151658), 100), "tool_calls");
}

#[test]
fn eos_on_the_last_budgeted_token_is_stop_not_length() {
    // 2026-09-25: The model finished naturally on the final budgeted token:
    // a sampled EOS outranks an exhausted budget.
    assert_eq!(derive(None, Some(151645), 0), "stop");
}

#[test]
fn timeout_unchanged_and_still_outranks_everything() {
    // 2026-09-25: A deadline cut (the timeout guard) must be distinguishable
    // from both a natural stop and a max_tokens stop, even when the deadline
    // lands on an EOS, tool-close, or budget-end step.
    for last in [Some(42), Some(151645), Some(151658)] {
        assert_eq!(
            derive(Some(GUARD_STOP_REQUEST_TIMEOUT), last, 100),
            FINISH_REASON_TIMEOUT
        );
    }
    assert_eq!(
        derive(Some(GUARD_STOP_REQUEST_TIMEOUT), Some(42), 0),
        FINISH_REASON_TIMEOUT
    );
}

#[test]
fn guard_cuts_report_length_because_the_model_did_not_finish() {
    // 2026-09-25: A guard cut is a server-side truncation: the model was
    // still mid-output when it fired, so this must report "length", not
    // "stop" — "stop" claims the model finished, which is false for a
    // mid-sentence repetition cut.
    for guard in [
        "fuzzy_repetition",
        "inter_tool_prose_budget",
        "tool_envelope_stuck",
        "simhash_semantic_loop",
        "token_loop_watchdog",
    ] {
        assert_eq!(
            derive(Some(guard), Some(42), 100),
            "length",
            "guard={guard}"
        );
        // 2026-09-25: A guard trip on the exact step the budget also runs
        // out is still "length": the two truncation causes agree.
        assert_eq!(derive(Some(guard), Some(42), 0), "length", "guard={guard}");
    }
}

#[test]
fn non_truncating_stops_are_not_relabelled_as_length() {
    // 2026-09-25: "length" is not a catch-all for "the last token was not
    // EOS": with no guard and budget left nothing was truncated, so this is
    // "stop".
    assert_eq!(derive(None, Some(42), 100), "stop");
    // 2026-09-25: The timeout guard keeps its own distinct reason instead
    // of collapsing into "length".
    assert_eq!(
        derive(Some(GUARD_STOP_REQUEST_TIMEOUT), Some(42), 100),
        FINISH_REASON_TIMEOUT
    );
}

#[test]
fn token_derived_stops_outrank_non_timeout_guards() {
    // 2026-09-25: A guard that fires on the same step the model sampled
    // EOS or closed a tool call reports what the model actually did, not
    // the guard.
    assert_eq!(
        derive(Some("tool_envelope_stuck"), Some(151645), 100),
        "stop"
    );
    assert_eq!(
        derive(Some("fuzzy_repetition"), Some(151658), 100),
        "tool_calls"
    );
}

#[test]
fn empty_output_edges() {
    // 2026-09-25: max_tokens==0 path: empty output with remaining==0 is
    // "length" — the zero budget was exhausted before the first token.
    assert_eq!(derive(None, None, 0), "length");
    // 2026-09-25: Empty output on a model with no tool-call end token
    // configured (both are `None`) must not compare equal and misreport
    // "tool_calls"; it must fall through to "stop".
    assert_eq!(
        derive_finish_reason(None, None, EOS, None, 5, 10, MAX_SEQ_LEN),
        "stop"
    );
}
