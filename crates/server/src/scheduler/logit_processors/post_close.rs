// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Post-`</think>` symmetric mask for `</think>` and `<think>`.
//!
//! While `seq.think_ended` is set, masks the close token
//! (`ctx.think_end_token`) and the open token (`seq.think_start_token`) to
//! `-inf`, so the model can neither repeat `</think>` nor re-open `<think>`.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::{LogitsContext, LogitsProcessor, ProcessorOutcome};
use crate::scheduler::ActiveSeq;

pub struct PostCloseThinkMask;

impl LogitsProcessor for PostCloseThinkMask {
    fn apply(
        &self,
        logits: &mut [f32],
        a: &mut ActiveSeq,
        ctx: &LogitsContext,
    ) -> ProcessorOutcome {
        if a.think_ended {
            if let Some(end_tok) = ctx.think_end_token {
                let end_idx = end_tok as usize;
                if end_idx < logits.len() {
                    logits[end_idx] = f32::NEG_INFINITY;
                }
            }
            if let Some(start_tok) = a.think_start_token {
                let start_idx = start_tok as usize;
                if start_idx < logits.len() {
                    logits[start_idx] = f32::NEG_INFINITY;
                }
            }
        }
        ProcessorOutcome::Continue
    }

    fn name(&self) -> &'static str {
        "post_close_think_mask"
    }
}
