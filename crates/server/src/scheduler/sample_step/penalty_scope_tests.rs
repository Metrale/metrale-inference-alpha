// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests for `penalty_history_scope` and
//! `strip_in_tool_opener_bias`.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::{penalty_history_scope, strip_in_tool_opener_bias};

const CLOSE: u32 = 248059;

#[test]
fn scope_without_completed_call_is_full_history() {
    let toks = vec![1, 2, 3, 4];
    assert_eq!(penalty_history_scope(&toks, Some(CLOSE)), &toks[..]);
    assert_eq!(penalty_history_scope(&toks, None), &toks[..]);
}

#[test]
fn scope_cuts_after_last_completed_call() {
    let toks = vec![10, 11, 12, CLOSE, 198, 20, 21];
    assert_eq!(penalty_history_scope(&toks, Some(CLOSE)), &[198, 20, 21]);
    let toks = vec![10, CLOSE, 198, 20, CLOSE, 30];
    assert_eq!(penalty_history_scope(&toks, Some(CLOSE)), &[30]);
    let toks = vec![10, 11, CLOSE];
    assert_eq!(penalty_history_scope(&toks, Some(CLOSE)), &[] as &[u32]);
}

/// 2026-09-25: The repetition penalty divides once per occurrence. With
/// the unscoped history (seven occurrences, 10.0 / 1.1^7 < 8.5) the
/// scaffold token falls below its unpenalised variant; the scoped history
/// holds one occurrence (10.0 / 1.1 > 8.5) and does not flip it.
#[test]
fn scoped_history_prevents_cross_call_penalty_compounding() {
    use crate::scheduler::{SamplingParams, apply_penalties_and_bias};
    const NL: u32 = 198;
    const NL_VARIANT: u32 = 695;

    let params = SamplingParams {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        top_n_sigma: 0.0,
        min_p: 0.0,
        logit_bias: Vec::new(),
        repetition_penalty: 1.1,
        repetition_penalty_window: 0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        lz_penalty: 0.0,
        dry_multiplier: 0.0,
        dry_base: 1.75,
        dry_allowed_length: 2,
        dry_sequence_breakers: Vec::new(),
        max_tokens: 0,
        stop_token_ids: Vec::new(),
        seed: None,
    };

    let full_history: Vec<u32> = vec![NL; 6]
        .into_iter()
        .chain([10, 11, CLOSE, NL]) // 2026-09-25: the last NL is the only one after CLOSE
        .collect();

    let mut logits = vec![0.0f32; 1000];
    logits[NL as usize] = 10.0;
    logits[NL_VARIANT as usize] = 8.5;
    apply_penalties_and_bias(&mut logits, &params, &full_history);
    assert!(
        logits[NL_VARIANT as usize] > logits[NL as usize],
        "unscoped: 1.1^7 compounding must flip the scaffold token (the bug): {} vs {}",
        logits[NL as usize],
        logits[NL_VARIANT as usize],
    );

    let mut logits = vec![0.0f32; 1000];
    logits[NL as usize] = 10.0;
    logits[NL_VARIANT as usize] = 8.5;
    let scoped = penalty_history_scope(&full_history, Some(CLOSE));
    assert_eq!(scoped, &[NL], "segment = separator newline only");
    apply_penalties_and_bias(&mut logits, &params, scoped);
    assert!(
        logits[NL as usize] > logits[NL_VARIANT as usize],
        "scoped: one occurrence must NOT flip the scaffold token: {} vs {}",
        logits[NL as usize],
        logits[NL_VARIANT as usize],
    );
}

#[test]
fn opener_bias_stripped_only_inside_tool_body() {
    const OPEN: u32 = 248058;
    let mut bias = vec![(OPEN, 3.0f32), (42, -8.0f32)];
    strip_in_tool_opener_bias(&mut bias, true, Some(OPEN));
    assert_eq!(bias, vec![(42, -8.0f32)]);

    let mut bias = vec![(OPEN, -5.0f32)];
    strip_in_tool_opener_bias(&mut bias, true, Some(OPEN));
    assert_eq!(bias, vec![(OPEN, -5.0f32)], "anti-repeat bias survives");

    let mut bias = vec![(OPEN, 3.0f32)];
    strip_in_tool_opener_bias(&mut bias, false, Some(OPEN));
    assert_eq!(bias, vec![(OPEN, 3.0f32)]);

    let mut bias = vec![(OPEN, 3.0f32)];
    strip_in_tool_opener_bias(&mut bias, true, None);
    assert_eq!(bias, vec![(OPEN, 3.0f32)]);
}
