// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Token-0 grammar policy, applied by `sample_step::sample_first_token`.
//!
//! Later tokens skip the grammar while inside `<think>`
//! (`logit_processors/grammar_bitmask.rs`, `decode_logits_step/per_token.rs`,
//! `emit_step/token.rs`). Token 0 is sampled before
//! `ActiveSeq::inside_thinking` exists, so the same rule is applied here
//! from [`born_inside_thinking`].
//!
//! Owner: scheduler.
//! Invariants:
//! - With a grammar, [`first_token_with`] either hands it to the sampler
//!   and, when sampling succeeds, advances it past token 0, or, when
//!   `policy.grammar_suspended`, leaves it untouched.

use crate::grammar::GrammarState;
use anyhow::Result;

/// 2026-09-25: Whether a new sequence starts inside `<think>`: thinking is
/// enabled and the model has a `</think>` token.
///
/// The prefill steps set `ActiveSeq::inside_thinking` from it (or'd with a
/// spontaneous `<think>` first token), and [`FirstTokenPolicy::for_birth`]
/// uses it.
pub(super) fn born_inside_thinking(enable_thinking: bool, think_end_token: Option<u32>) -> bool {
    enable_thinking && think_end_token.is_some()
}

/// 2026-09-25: What token 0 may do with an armed grammar.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct FirstTokenPolicy {
    /// 2026-09-25: Token 0 is sampled inside `<think>`: no bitmask and no
    /// `accept_token`, so the matcher is untouched for the first
    /// post-think token.
    pub grammar_suspended: bool,
    /// 2026-09-25: The `<tool_call>` id, suppressed at token 0 while a
    /// grammar is armed and suspended. Later tokens inside `<think>` get it
    /// masked by `ToolCallDuringThinkingMask`.
    pub tool_call_start: Option<u32>,
}

impl FirstTokenPolicy {
    /// 2026-09-25: Policy for a sequence about to be born from these
    /// request facts.
    pub(super) fn for_birth(
        enable_thinking: bool,
        think_end_token: Option<u32>,
        tool_call_start: Option<u32>,
    ) -> Self {
        Self {
            grammar_suspended: born_inside_thinking(enable_thinking, think_end_token),
            tool_call_start,
        }
    }
}

/// 2026-09-25: Core of `sample_first_token`, generic over the
/// model-dependent sampler.
///
/// `sample(suppress_ids, grammar)` is called exactly once. It receives the
/// grammar only when the policy lets the grammar act; then the matcher is
/// advanced past the returned token. With no grammar the call is a plain
/// pass-through.
pub(super) fn first_token_with<F>(
    policy: FirstTokenPolicy,
    suppress_ids: &[u32],
    grammar_state: Option<&mut GrammarState>,
    sample: F,
) -> Result<u32>
where
    F: FnOnce(&[u32], Option<&mut GrammarState>) -> Result<u32>,
{
    let Some(gs) = grammar_state else {
        return sample(suppress_ids, None);
    };
    if policy.grammar_suspended {
        // 2026-09-25: grammar armed, sequence born inside `<think>`: the
        // matcher does not see this token, and `<tool_call>` is suppressed.
        let mut ids = suppress_ids.to_vec();
        if let Some(t) = policy.tool_call_start
            && !ids.contains(&t)
        {
            ids.push(t);
        }
        return sample(&ids, None);
    }
    let tok = sample(suppress_ids, Some(&mut *gs))?;
    gs.accept_token(tok);
    Ok(tok)
}
