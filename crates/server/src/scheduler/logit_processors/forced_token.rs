// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Forced-token fast path (the Coalescence technique): emit the grammar's only legal token.
//!
//! The stage returns [`ProcessorOutcome::EmitToken`] with the token that
//! `GrammarState::forced_token` reports as the only legal next token, which
//! skips the rest of the pipeline and the sample. It does so only when all of
//! these hold:
//!  * the sequence is not inside thinking;
//!  * `top_logprobs` is not requested (logprobs need the distribution);
//!  * `ctx.sampling.forced_token_fastpath` is set;
//!  * the sequence has a grammar state, and it reports exactly one legal token;
//!  * the token is not an EOS id while the `min_tokens` floor is unmet.
//!
//! Owner: scheduler.
//! Invariants:
//! - This is the only stage that returns [`ProcessorOutcome::EmitToken`].

use super::{LogitsContext, LogitsProcessor, ProcessorOutcome};
use crate::scheduler::ActiveSeq;

pub struct ForcedTokenFastPath;

impl LogitsProcessor for ForcedTokenFastPath {
    fn apply(
        &self,
        _logits: &mut [f32],
        a: &mut ActiveSeq,
        ctx: &LogitsContext,
    ) -> ProcessorOutcome {
        if !a.inside_thinking
            && a.top_logprobs.is_none()
            && ctx.sampling.forced_token_fastpath
            && let Some(ref mut gs) = a.grammar_state
            && let Some(forced) = gs.forced_token()
        {
            let forced = forced as u32;
            // 2026-09-25: The fast path skips sampling, so a grammar-forced EOS
            // would be emitted before the `min_tokens` floor. Continue instead,
            // and let the rest of the pipeline handle it.
            let effective_len = a.output_tokens.len().saturating_add(ctx.verify_pos);
            if effective_len < a.min_tokens && a.eos_tokens.contains(&forced) {
                return ProcessorOutcome::Continue;
            }
            return ProcessorOutcome::EmitToken(forced);
        }
        ProcessorOutcome::Continue
    }

    fn name(&self) -> &'static str {
        "forced_token_fastpath"
    }
}
