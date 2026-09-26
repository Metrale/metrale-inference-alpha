// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Grammar bitmask application.
//!
//! Outside thinking, when the sequence has a grammar state and
//! `GrammarState::fill_bitmask` returns true, applies the grammar's next-token
//! bitmask to `logits`. Inside thinking the logits are left alone.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::{LogitsContext, LogitsProcessor, ProcessorOutcome};
use crate::scheduler::ActiveSeq;

pub struct GrammarBitmaskApply;

impl LogitsProcessor for GrammarBitmaskApply {
    fn apply(
        &self,
        logits: &mut [f32],
        a: &mut ActiveSeq,
        _ctx: &LogitsContext,
    ) -> ProcessorOutcome {
        if !a.inside_thinking
            && let Some(ref mut gs) = a.grammar_state
            && gs.fill_bitmask()
        {
            gs.apply_bitmask_to_logits(logits);
        }
        ProcessorOutcome::Continue
    }

    fn name(&self) -> &'static str {
        "grammar_bitmask_apply"
    }
}
