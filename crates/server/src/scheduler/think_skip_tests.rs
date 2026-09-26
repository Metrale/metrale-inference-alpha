// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Behavioural tests for the stray-`</think>` watchdog on the MTP / spec-verify path (`emit_step::emit_token`).
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.
//!
//! Outside thinking, a stray `</think>` increments `think_skip_count`, a
//! content token after thinking resets it, and the 50th stray in a row
//! force-stops the turn. `emit_step/token.rs` and the non-MTP
//! `decode_logits_step/per_token.rs` both implement this, so both count
//! consecutive strays. These tests drive the real `emit_token` over a real `ActiveSeq`
//! and pin both halves:
//!  * scattered strays do not stop the turn, however many there are;
//!  * a consecutive run of 50 still stops it.

use super::emit_step::emit_token;
use super::lifecycle::derive_finish_reason;
use super::sched_ctx::SchedCtx;
use super::test_support::{EOS, test_seq};
use super::types::{ActiveSeq, GUARD_STOP_THINK_SKIP};

/// 2026-09-25: A `</think>` id distinct from the fixture's EOS (151645) and tool-call
/// close (151658), so neither of those paths is entered.
const THINK_END: u32 = 151668;

/// 2026-09-26: The force-stop point in both `emit_step/token.rs` and
/// `decode_logits_step/per_token.rs`.
const SKIP_LIMIT: u32 = 50;

/// 2026-09-25: A post-`</think>` sequence: `think_ended` is set, the sequence is outside
/// a thinking span, and `</think>` is a stray token. The budget (5000) is far
/// larger than any test emits, so `finished` never comes from the
/// `remaining == 0` length stop.
fn content_phase_seq() -> ActiveSeq {
    let (mut a, _rx) = test_seq(Vec::new(), 5000, None, 10);
    a.finished = false;
    a.inside_thinking = false;
    a.think_ended = true;
    a.think_end_token = Some(THINK_END);
    // 2026-09-25: `_rx` is dropped: the fixture's sink is Blocking, and
    // `emit_token`'s per-token emit does not send on a Blocking sink.
    a
}

/// 2026-09-25: Distinct, strictly increasing content tokens, so the content-loop repeat
/// detector never fires and `finished` isolates the skip counter.
fn content_token(i: u32) -> u32 {
    1000 + i
}

#[test]
fn scattered_think_strays_do_not_stop_the_turn() {
    // 2026-09-25: 60 strays, past the threshold of 50, each followed by
    // one content token. Goes red without the `if a.think_ended {
    // a.think_skip_count = 0; }` reset in `emit_step/token.rs`: the
    // counter would reach 50 on the 50th stray and force-stop the turn.
    let sched = SchedCtx::for_test();
    let mut a = content_phase_seq();
    let strays = 60;
    for i in 0..strays {
        emit_token(&mut a, THINK_END, None, &sched);
        assert!(
            !a.finished,
            "stray #{} of {strays} force-stopped the turn: scattered `</think>` \
             must not accumulate — the counter is reset by intervening content \
             on the non-MTP path, and `emit_token` must match it",
            i + 1
        );
        emit_token(&mut a, content_token(i), None, &sched);
        assert!(
            !a.finished,
            "content token after stray #{} ended the turn",
            i + 1
        );
    }
    assert_eq!(
        a.think_skip_count, 0,
        "the last content token must have zeroed the stray counter"
    );
    // 2026-09-25: the strays were skipped, not emitted: only the content
    // tokens are in the output.
    assert_eq!(a.output_tokens.len(), strays as usize);
    assert!(!a.finished);
}

#[test]
fn fifty_consecutive_think_strays_still_stop_the_turn() {
    // 2026-09-25: boundary control, independent of the reset: 49
    // consecutive strays are survivable and the 50th is not, so the
    // watchdog cannot be removed to satisfy the test above.
    let sched = SchedCtx::for_test();
    let mut a = content_phase_seq();
    for i in 1..SKIP_LIMIT {
        emit_token(&mut a, THINK_END, None, &sched);
        assert!(
            !a.finished,
            "stray #{i} fired the watchdog early — the threshold is {SKIP_LIMIT}"
        );
        assert_eq!(a.think_skip_count, i, "stray #{i} did not increment");
    }
    emit_token(&mut a, THINK_END, None, &sched);
    assert_eq!(a.think_skip_count, SKIP_LIMIT);
    assert!(
        a.finished,
        "{SKIP_LIMIT} CONSECUTIVE `</think>` strays are the degeneration the \
         watchdog exists for; it must still force-stop the sequence"
    );
    assert!(a.output_tokens.is_empty());
}

#[test]
fn the_watchdog_rearms_after_content_defuses_it() {
    // 2026-09-25: the reset must not be a one-way disarm: once content
    // has zeroed the counter, a fresh consecutive run of 50 must still
    // fire.
    let sched = SchedCtx::for_test();
    let mut a = content_phase_seq();
    for _ in 0..SKIP_LIMIT - 1 {
        emit_token(&mut a, THINK_END, None, &sched);
    }
    emit_token(&mut a, content_token(0), None, &sched);
    assert!(!a.finished, "content must defuse a nearly-tripped counter");
    assert_eq!(a.think_skip_count, 0);
    for _ in 0..SKIP_LIMIT {
        emit_token(&mut a, THINK_END, None, &sched);
    }
    assert!(
        a.finished,
        "the watchdog must re-arm: a fresh run of {SKIP_LIMIT} consecutive \
         strays after a reset must still stop the turn"
    );
}

/// 2026-09-25: Calls `derive_finish_reason` with the fields `finish_sequence` passes, and
/// `max_seq_len` 0 (unlimited) so the context-ceiling check never applies.
fn wire_reason(a: &ActiveSeq) -> &'static str {
    derive_finish_reason(
        a.guard_stop,
        a.output_tokens.last().copied(),
        &a.eos_tokens,
        a.tool_call_end_token,
        a.remaining,
        a.seq.seq_len,
        0,
    )
}

#[test]
fn the_watchdog_cut_names_its_guard_and_wires_length() {
    // 2026-09-25: the watchdog skips the stray tokens (they are never
    // pushed), so the last token is plain content. Without a named guard,
    // `derive_finish_reason` would wire "stop"; the agentic harness's
    // `was_cut_off()` (`crates/bench/src/benchmarks/agentic/agent.rs`)
    // grants a recovery turn only on "length" with no tool calls.
    let sched = SchedCtx::for_test();
    let mut a = content_phase_seq();
    for i in 1..SKIP_LIMIT {
        emit_token(&mut a, THINK_END, None, &sched);
        assert!(
            a.guard_stop.is_none(),
            "guard named before the threshold (stray #{i}) — the name must \
             mark the CUT, not the counting"
        );
    }
    emit_token(&mut a, THINK_END, None, &sched);
    assert!(a.finished);
    assert_eq!(
        a.guard_stop,
        Some(GUARD_STOP_THINK_SKIP),
        "the watchdog cut must name its guard at the call site"
    );
    assert_eq!(
        wire_reason(&a),
        "length",
        "a server-side watchdog cut with budget left is a truncation; \
         \"length\" is what lets an agentic client recover the turn"
    );
}

#[test]
fn a_genuine_eos_stop_is_not_converted_to_length() {
    // 2026-09-25: guards against over-application: a model that finishes
    // naturally must still wire "stop", even after a below-threshold
    // burst of strays on the same turn.
    let sched = SchedCtx::for_test();
    let mut a = content_phase_seq();
    // 2026-09-25: the fixture's min_tokens of 7 would suppress this EOS
    // after five content tokens.
    a.min_tokens = 0;
    for i in 0..5 {
        emit_token(&mut a, content_token(i), None, &sched);
    }
    for _ in 0..3 {
        emit_token(&mut a, THINK_END, None, &sched);
    }
    emit_token(&mut a, EOS[0], None, &sched);
    assert!(a.finished, "EOS must finish the turn");
    assert!(
        a.guard_stop.is_none(),
        "a natural EOS finish must NOT name a guard — that would relabel a \
         real model stop as a server truncation and grant phantom recovery \
         turns"
    );
    assert_eq!(
        wire_reason(&a),
        "stop",
        "the model finished; the wire must say so"
    );
}

#[test]
fn strays_inside_thinking_are_not_counted_as_strays() {
    // 2026-09-25: scope check on the increment's `!inside_thinking`
    // gate: a `</think>` that closes a thinking span is not a stray. It
    // exits thinking and leaves the counter alone.
    let sched = SchedCtx::for_test();
    let mut a = content_phase_seq();
    a.inside_thinking = true;
    a.think_ended = false;
    emit_token(&mut a, THINK_END, None, &sched);
    assert!(
        !a.inside_thinking,
        "`</think>` must close the thinking span"
    );
    assert!(a.think_ended);
    assert_eq!(a.think_skip_count, 0, "a legitimate close is not a stray");
    assert!(!a.finished);
}
