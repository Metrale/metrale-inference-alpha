// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Pure helpers behind the confidence early stop
//! (`F2ConfidenceEarlyStop`) and the forced-`</think>` injection gate: code-fence
//! tracking, the confidence-run accumulator and the injection decision.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

/// 2026-09-25: Flip `in_fence` when `tok` is the model's single-token ``` code fence.
/// With `fence_tok == None` (the tokenizer has no such token) the state never
/// changes.
pub fn toggle_code_fence(in_fence: bool, tok: u32, fence_tok: Option<u32>) -> bool {
    match fence_tok {
        Some(f) if f == tok => !in_fence,
        _ => in_fence,
    }
}

/// 2026-09-25: The confidence-run length in `WatchdogParams::default()`. A served
/// model reads `confidence_run_length` from its MODEL.toml `[behavior]` table
/// instead (`WatchdogParams::from_behavior`), where an unset key resolves to 30.
pub const CONFIDENCE_RUN_LIMIT: u32 = 60;

/// 2026-09-25: Confidence-run accumulator for `F2ConfidenceEarlyStop`. Given whether
/// this token is confident (the caller's test is top-1 probability ≥ 0.95)
/// and the previous run length, return `(new_run, arm_force_end)`; it arms
/// once the run reaches `run_limit`.
///
/// It counts the same inside a ``` code fence. Keeping the forced `</think>`
/// out of a fence is [`should_inject_think_end`]'s job.
pub fn confidence_run_step(confident: bool, prev_run: u32, run_limit: u32) -> (u32, bool) {
    if confident {
        let run = prev_run + 1;
        (run, run >= run_limit)
    } else {
        (0, false)
    }
}

/// 2026-09-25: A deferred forced `</think>` is injected anyway once thinking reaches
/// this multiple of the sequence's thinking budget (`ForcedThinkEndInjector`).
pub const THINK_DEFER_BUDGET_FACTOR: u32 = 3;
/// 2026-09-25: The same limit, in thinking tokens, for a sequence without a
/// thinking budget.
pub const THINK_DEFER_ABS_CEILING: u32 = 2048;
/// 2026-09-25: A deferred forced `</think>` is also injected anyway after this many
/// armed-but-deferred steps (`ActiveSeq::sentence_defer_count`).
pub const MAX_SENTENCE_DEFER_TOKENS: u32 = 64;

/// 2026-09-25: Whether to inject the forced `</think>` now. `force_end_thinking` can be
/// armed mid-statement, so the injection is deferred:
///
/// 1. inside a ``` code fence, until the fence closes;
/// 2. outside a fence, until the previous token is a boundary token
///    (`VocabMasks::boundary`: text ending in a newline or sentence-ending
///    punctuation);
///
/// unless 3. `hard_override`, which the caller sets when one of the limits
/// above ([`THINK_DEFER_BUDGET_FACTOR`], [`THINK_DEFER_ABS_CEILING`],
/// [`MAX_SENTENCE_DEFER_TOKENS`]) is reached.
pub fn should_inject_think_end(
    force_end_thinking: bool,
    in_code_fence: bool,
    at_sentence_boundary: bool,
    hard_override: bool,
) -> bool {
    if !force_end_thinking {
        return false;
    }
    if hard_override {
        return true;
    }
    if in_code_fence {
        return false;
    }
    at_sentence_boundary
}

#[cfg(test)]
#[path = "confidence_tests.rs"]
mod confidence_tests;
