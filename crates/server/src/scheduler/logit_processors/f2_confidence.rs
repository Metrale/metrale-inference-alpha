// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `F2ConfidenceEarlyStop`: arms the forced `</think>` after a run of confident tokens.
//!
//! The stage acts only when watchdogs are not disabled, the sequence is inside
//! thinking with at least 400 thinking tokens, `force_end_thinking` is not yet
//! set, and the model's `confidence_early_stop` is on. It then counts
//! consecutive tokens whose top-1 softmax probability is >= 0.95
//! ([`crate::scheduler::confidence::confidence_run_step`]). When the count
//! reaches `ctx.watchdog.confidence_run_length` it sets
//! `seq.force_end_thinking`; `forced_think_end` decides when `</think>` is
//! injected.
//!
//! Owner: scheduler.
//! Invariants:
//! - This stage never writes `logits`.

use super::{LogitsContext, LogitsProcessor, ProcessorOutcome};
use crate::scheduler::ActiveSeq;
use crate::scheduler::confidence::confidence_run_step;

pub struct F2ConfidenceEarlyStop;

impl LogitsProcessor for F2ConfidenceEarlyStop {
    fn apply(
        &self,
        logits: &mut [f32],
        a: &mut ActiveSeq,
        ctx: &LogitsContext,
    ) -> ProcessorOutcome {
        if !ctx.sampling.disable_watchdogs
            && a.inside_thinking
            && !a.force_end_thinking
            && a.thinking_tokens >= 400
            && ctx.watchdog.confidence_early_stop
        {
            let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let sum_exp: f32 = logits.iter().map(|&l| (l - max_logit).exp()).sum();
            let confident = sum_exp > 0.0 && 1.0 / sum_exp >= 0.95;
            let (run, force_end) = confidence_run_step(
                confident,
                a.consecutive_confident,
                ctx.watchdog.confidence_run_length,
            );
            a.consecutive_confident = run;
            if force_end {
                a.force_end_thinking = true;
                a.sentence_defer_count = 0;
                tracing::info!(
                    "Confidence early stop armed: top-1 prob >= 0.95 for {} tokens (after {} thinking tokens){}",
                    ctx.watchdog.confidence_run_length,
                    a.thinking_tokens,
                    if a.in_code_fence {
                        " — deferred until ``` fence closes"
                    } else {
                        " — deferring until next sentence boundary"
                    }
                );
            }
        }
        ProcessorOutcome::Continue
    }

    fn name(&self) -> &'static str {
        "f2_confidence_early_stop"
    }

    fn is_argmax_invariant(&self) -> bool {
        true
    }
}
