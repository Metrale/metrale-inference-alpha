// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests for thinking-mode EOS suppression in `emit_step::emit_token`.
//!
//! They drive the real `emit_token` over a `test_seq` fixture (the pure
//! predicate `eos_suppressed_by_thinking` is tested in
//! `helpers/hard_limit_tests.rs`) and check three cases:
//!  * inside `<think>`, EOS does not finish the sequence;
//!  * outside `<think>`, the same EOS finishes it;
//!  * inside `<think>`, an EOS that uses up `remaining` finishes it: the
//!    hard-ceiling escape.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::emit_step::emit_token;
use super::sched_ctx::SchedCtx;
use super::test_support::{EOS, test_seq};
use super::types::ActiveSeq;

/// 2026-09-25: A sequence mid-`<think>` with seven tokens already emitted,
/// so the fixture's `min_tokens` (7) never suppresses EOS. `remaining` is
/// 5000 and `SchedCtx::for_test()` has `max_seq_len == 0` (no ceiling), so
/// no hard ceiling is hit unless a test lowers `remaining`.
fn thinking_phase_seq() -> ActiveSeq {
    let (mut a, _rx) = test_seq((1000..1007).collect(), 5000, None, 10);
    a.finished = false;
    a.inside_thinking = true;
    // 2026-09-25: `_rx` can be dropped: `TokioRequestIo::emit` returns true
    // for a Blocking sink without sending.
    a
}

#[test]
fn eos_inside_thinking_does_not_finish_the_sequence() {
    // 2026-09-25: no grammar, no `require_tool_call` and `min_tokens` met, so
    // only the `thinking_suppresses_eos` term can hold this EOS back.
    let sched = SchedCtx::for_test();
    let mut a = thinking_phase_seq();
    emit_token(&mut a, EOS[0], None, &sched);
    assert!(
        !a.finished,
        "a spurious EOS inside <think> ended the turn — only </think> may \
         exit a thinking span while no hard ceiling is hit"
    );
    assert!(
        a.inside_thinking,
        "the discarded EOS must not exit thinking mode"
    );
    emit_token(&mut a, 2000, None, &sched);
    assert!(
        !a.finished,
        "the token after the discarded EOS ended the turn"
    );
}

#[test]
fn same_eos_outside_thinking_finishes_the_sequence() {
    // 2026-09-25: same state and token with thinking off: the suppression
    // above is keyed on `inside_thinking`, not on another term.
    let sched = SchedCtx::for_test();
    let mut a = thinking_phase_seq();
    a.inside_thinking = false;
    emit_token(&mut a, EOS[0], None, &sched);
    assert!(
        a.finished,
        "EOS outside <think> with min_tokens met and no grammar must finish \
         the sequence"
    );
}

#[test]
fn eos_at_hard_ceiling_finishes_even_inside_thinking() {
    // 2026-09-25: `remaining` is 1, so this EOS consumes the last of the
    // budget and the hard ceiling is hit when EOS is decided. Without the
    // `!hard_ceiling` term the suppressed-EOS branch returns before the
    // `remaining == 0` stop and `finished` stays false.
    let sched = SchedCtx::for_test();
    let mut a = thinking_phase_seq();
    a.remaining = 1;
    emit_token(&mut a, EOS[0], None, &sched);
    assert!(
        a.finished,
        "a model-sampled EOS at the exhausted completion budget must be \
         honored even inside <think> — the escape that stops budget overrun"
    );
}
