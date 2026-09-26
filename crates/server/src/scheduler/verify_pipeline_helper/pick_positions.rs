// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: the host-side K-position pick loop of
//! `verify_pick_all_with_pipeline`; it takes host logits rather than a
//! model, so tests drive it directly.
//!
//! Owner: scheduler.
//! Invariants:
//! - On return the grammar matcher is at the history depth it had on entry,
//!   and `inside_thinking`, `think_ended` and `think_just_ended` hold their
//!   entry values. Within the loop, each later position is masked against
//!   the matcher state the earlier picks leave behind, and a `</think>` pick
//!   switches the thinking flags for the remaining positions, so the first
//!   position after it is picked outside thinking.

use super::verify_pick_with_pipeline;
use crate::scheduler::logit_processors::LogitsContext;
use crate::scheduler::types::ActiveSeq;

/// 2026-09-25: run `verify_pick_with_pipeline` over `k` host-resident BF16
/// rows of `vocab` logits and return the pick per position. Stops early,
/// returning fewer than `k` picks, when the matcher refuses a pick.
pub(super) fn pick_positions_from_host(
    buf: &[u8],
    vocab: usize,
    elem_bytes: usize,
    k: usize,
    a: &mut ActiveSeq,
    ctx: &LogitsContext,
) -> Vec<u32> {
    let mut picks: Vec<u32> = Vec::with_capacity(k);
    // 2026-09-25: the history depth, for the rollback at the end; see there
    // for why it is a depth and not a count of `accept_token` calls.
    let grammar_steps_before = a.grammar_state.as_ref().map(|gs| gs.num_history_steps());
    let think_flags_before = (a.inside_thinking, a.think_ended, a.think_just_ended);

    for i in 0..k {
        let slice = &buf[i * vocab * elem_bytes..(i + 1) * vocab * elem_bytes];
        let pick = verify_pick_with_pipeline(slice, false, vocab, a, ctx, i);
        picks.push(pick);

        // 2026-09-25: `</think>` closes thinking for the later positions. It
        // is not fed to the matcher.
        if a.inside_thinking && ctx.think_end_token == Some(pick) {
            a.inside_thinking = false;
            a.think_ended = true;
            a.think_just_ended = true;
            continue;
        }

        // 2026-09-25: advance the matcher with the pick so the next position
        // is masked against the state after it. Not after the last position,
        // and not inside thinking.
        if i + 1 < k
            && let Some(ref mut gs) = a.grammar_state
            && !a.inside_thinking
        {
            // 2026-09-25: a refused advance ends the loop; the picks so far
            // are returned.
            if !gs.accept_token(pick) {
                tracing::debug!(
                    pick,
                    i,
                    "verify_pick: grammar speculative advance refused — pipeline picked a token outside the current bitmask. \
                     This indicates a stale bitmask in the pipeline or a forced-token fastpath that terminated grammar. \
                     Stopping speculation here; the real `accept_token` in emit_token will fail and end the response."
                );
                break;
            }
        }
    }

    // 2026-09-25: roll back by the history delta. Counting `accept_token`
    // calls would over-rewind: `GrammarState::accept_token` returns true for
    // stop tokens and in the terminated state without adding a history step.
    // `emit_token` later advances the matcher for the tokens it emits.
    if let (Some(before), Some(gs)) = (grammar_steps_before, a.grammar_state.as_mut()) {
        let advanced = gs.num_history_steps().saturating_sub(before);
        if advanced > 0 {
            gs.rollback(advanced);
        }
    }
    // 2026-09-25: restore the thinking flags; `emit_token` makes the real
    // `</think>` transition.
    (a.inside_thinking, a.think_ended, a.think_just_ended) = think_flags_before;

    picks
}
