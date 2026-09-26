// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: One-shot pin-to-`<tool_call>` immediately after `</think>`.
//!
//! Acts when all of these hold: `seq.think_just_ended` (set when thinking
//! closes, cleared when the next content token is emitted),
//! `seq.require_tool_call`, no tool call opened yet, not inside thinking, and
//! `ctx.tool_call_start_token` is known. It then sets every logit to `-inf`
//! and the tool-call start token to `0.0`, so the token after `</think>` opens
//! a tool call.
//!
//! `seq.require_tool_call` is set at prefill only when the request requires a
//! tool call, the sequence has no grammar and the tokenizer has a tool-call
//! start token (`use_legacy_tool_call` in `prefill_a_step.rs` and
//! `phase_promote_prefills.rs`).
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::{LogitsContext, LogitsProcessor, ProcessorOutcome};
use crate::scheduler::ActiveSeq;

pub struct PinToToolCallStart;

impl LogitsProcessor for PinToToolCallStart {
    fn apply(
        &self,
        logits: &mut [f32],
        a: &mut ActiveSeq,
        ctx: &LogitsContext,
    ) -> ProcessorOutcome {
        if a.think_just_ended
            && a.require_tool_call
            && !a.tool_call_opened
            && !a.inside_thinking
            && let Some(start_tok) = ctx.tool_call_start_token
        {
            let idx = start_tok as usize;
            if idx < logits.len() {
                for logit in logits.iter_mut() {
                    *logit = f32::NEG_INFINITY;
                }
                logits[idx] = 0.0;
                tracing::debug!(
                    "Forced tool_call_start_token after </think> (require_tool_call set)"
                );
            }
        }
        ProcessorOutcome::Continue
    }

    fn name(&self) -> &'static str {
        "pin_to_tool_call_start"
    }
}
