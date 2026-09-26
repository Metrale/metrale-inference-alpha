// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Forced `</think>` injection once `force_end_thinking` is armed.
//!
//! Inside thinking, the stage asks
//! [`crate::scheduler::confidence::should_inject_think_end`] with the
//! sequence's `force_end_thinking` and `in_code_fence` and two values computed
//! here:
//! - `at_sentence_boundary`: the previous output token is set in
//!   `ctx.boundary_mask`;
//! - `defer_hard_override`: thinking has reached `THINK_DEFER_BUDGET_FACTOR`
//!   times the thinking budget (`THINK_DEFER_ABS_CEILING` with no budget), or
//!   `seq.sentence_defer_count` has reached `MAX_SENTENCE_DEFER_TOKENS`.
//!
//! When the gate says inject, every logit is set to `-inf` and `</think>` to
//! `0.0`, so `</think>` is the only finite logit. When `force_end_thinking` is
//! set but the injection does not happen, `seq.sentence_defer_count` is
//! incremented, which bounds the deferral at `MAX_SENTENCE_DEFER_TOKENS` steps.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::{LogitsContext, LogitsProcessor, ProcessorOutcome};
use crate::scheduler::ActiveSeq;
use crate::scheduler::confidence::{
    MAX_SENTENCE_DEFER_TOKENS, THINK_DEFER_ABS_CEILING, THINK_DEFER_BUDGET_FACTOR,
    should_inject_think_end,
};

pub struct ForcedThinkEndInjector;

impl LogitsProcessor for ForcedThinkEndInjector {
    fn apply(
        &self,
        logits: &mut [f32],
        a: &mut ActiveSeq,
        ctx: &LogitsContext,
    ) -> ProcessorOutcome {
        let at_sentence_boundary = a
            .output_tokens
            .last()
            .copied()
            .and_then(|prev_tok| {
                ctx.boundary_mask
                    .clone()
                    .as_deref()
                    .and_then(|m| m.get(prev_tok as usize).copied())
            })
            .unwrap_or(false);
        let defer_hard_override = match a.thinking_budget {
            Some(b) => a.thinking_tokens >= b.saturating_mul(THINK_DEFER_BUDGET_FACTOR),
            None => a.thinking_tokens >= THINK_DEFER_ABS_CEILING,
        } || a.sentence_defer_count >= MAX_SENTENCE_DEFER_TOKENS;
        if a.inside_thinking
            && should_inject_think_end(
                a.force_end_thinking,
                a.in_code_fence,
                at_sentence_boundary,
                defer_hard_override,
            )
            && let Some(end_tok) = ctx.think_end_token
        {
            let end_idx = end_tok as usize;
            if end_idx < logits.len() {
                for logit in logits.iter_mut() {
                    *logit = f32::NEG_INFINITY;
                }
                logits[end_idx] = 0.0;
            }
        } else if a.inside_thinking && a.force_end_thinking {
            // 2026-09-25: Armed but not injected this step: count it toward
            // the MAX_SENTENCE_DEFER_TOKENS override.
            a.sentence_defer_count = a.sentence_defer_count.saturating_add(1);
        }
        ProcessorOutcome::Continue
    }

    fn name(&self) -> &'static str {
        "forced_think_end_injector"
    }
}
