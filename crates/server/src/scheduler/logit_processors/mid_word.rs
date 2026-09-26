// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Mid-word `</think>` defer.
//!
//! Inside thinking, and unless watchdogs are disabled, masks `</think>` to
//! `-inf` when the previous output token is set in `ctx.mid_word_mask` (its
//! text ends in an alphanumeric character), so thinking does not close in the
//! middle of a word. With no mask, no previous token or no `</think>` id,
//! nothing is masked.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::{LogitsContext, LogitsProcessor, ProcessorOutcome};
use crate::scheduler::ActiveSeq;

pub struct MidWordThinkEndMask;

impl LogitsProcessor for MidWordThinkEndMask {
    fn apply(
        &self,
        logits: &mut [f32],
        a: &mut ActiveSeq,
        ctx: &LogitsContext,
    ) -> ProcessorOutcome {
        if !ctx.sampling.disable_watchdogs
            && a.inside_thinking
            && let Some(end_tok) = ctx.think_end_token
            && let Some(prev_tok) = a.output_tokens.last().copied()
            && let Some(mask) = ctx.mid_word_mask.as_deref()
            && mask.get(prev_tok as usize).copied().unwrap_or(false)
        {
            let end_idx = end_tok as usize;
            if end_idx < logits.len() {
                logits[end_idx] = f32::NEG_INFINITY;
            }
        }
        ProcessorOutcome::Continue
    }

    fn name(&self) -> &'static str {
        "mid_word_think_end_mask"
    }
}
