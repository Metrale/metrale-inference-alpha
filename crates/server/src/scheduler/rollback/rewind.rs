// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The token-buffer rewind, the rollback application, the
//! `RomHead` trait and the grammar-rewind rule for `rollback.rs`.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::*;

/// 2026-09-25: Truncate `output_tokens` to `keep_len` and pop as many
/// tokens from `seq_tokens` as were dropped, lowering `seq_len` by one per
/// token popped. Returns the new `seq_len`.
pub fn rewind_buffers(
    output_tokens: &mut Vec<u32>,
    seq_tokens: &mut Vec<u32>,
    seq_len: usize,
    keep_len: usize,
) -> usize {
    let dropped = output_tokens.len().saturating_sub(keep_len);
    output_tokens.truncate(keep_len);
    let mut new_seq_len = seq_len;
    for _ in 0..dropped {
        if seq_tokens.pop().is_some() {
            new_seq_len = new_seq_len.saturating_sub(1);
        }
    }
    new_seq_len
}

/// 2026-09-25: Rewind `a` to `keep_len` output tokens: [`rewind_buffers`],
/// add `dropped` back to `remaining` and take it off `content_tokens`,
/// point `last_token` at the new last output token, rewind the grammar
/// matcher when [`grammar_rewind`] allows it, and zero the inter-tool
/// prose and confident-run counters.
pub(super) fn apply_rollback(a: &mut ActiveSeq, keep_len: usize, dropped: usize) {
    a.seq.seq_len = rewind_buffers(
        &mut a.output_tokens,
        &mut a.seq.tokens,
        a.seq.seq_len,
        keep_len,
    );

    a.remaining = a.remaining.saturating_add(dropped);
    a.content_tokens = a.content_tokens.saturating_sub(dropped as u32);

    if let Some(&last) = a.output_tokens.last() {
        a.last_token = last;
    }

    // 2026-09-25: `dropped` counts tokens, not matcher steps: a stop token,
    // a token after termination or a rejected token records no step. When
    // `dropped` exceeds the recorded steps the rewind is skipped.
    if let Some(ref mut gs) = a.grammar_state {
        match grammar_rewind(dropped, gs.num_history_steps()) {
            Some(n) => gs.rollback(n),
            None => tracing::warn!(
                "grammar rollback skipped: {dropped} tokens dropped but the matcher \
                 recorded only {} steps. Constrained output may drift for this \
                 sequence; it is not a reason to kill the scheduler (#842).",
                gs.num_history_steps()
            ),
        }
    }

    // 2026-09-25: Zeroed so the inter-tool prose budget and the
    // confident-run early stop do not fire again on the tokens just dropped.
    a.prose_tokens_since_last_tool = 0;
    a.consecutive_confident = 0;
}

/// 2026-09-25: A trained repetition-onset head. No implementation ships,
/// and nothing outside tests calls one (`SchedCtx::rom_head` is always
/// `None`).
#[allow(dead_code)]
pub trait RomHead: Send + Sync {
    /// 2026-09-25: The probability, in `[0.0, 1.0]`, that a sequence whose
    /// recent output is `recent_tokens` (most recent last) has started to
    /// repeat.
    fn repetition_onset_score(&self, recent_tokens: &[u32]) -> f32;
}

/// 2026-09-25: How many matcher steps to rewind for `dropped` sequence
/// tokens: `dropped`, or `None` (do not rewind) when it exceeds
/// `history_steps`.
///
/// The speculative paths (`spec_step.rs`, `verify_pipeline_helper/`)
/// rewind by the change in `num_history_steps()` across the span they undo.
/// The watchdog path has only a token count, and `accept_token` returns
/// true without recording a step for stop tokens and once the matcher has
/// terminated. When the counts disagree, the recorded steps may belong to
/// kept tokens, so a partial rewind (`min(dropped, steps)`) is not used.
/// Even when `dropped <= history_steps` the rewind can be off, since some
/// dropped tokens may have recorded no step.
pub(super) fn grammar_rewind(dropped: usize, history_steps: usize) -> Option<usize> {
    (dropped <= history_steps).then_some(dropped)
}
