// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Unit tests for `scheduler::helpers`: the thinking- and
//! content-loop detectors and their overrides, forced-token fast-path
//! parsing, and two `WatchdogParams` defaults. A child of `helpers` via
//! `#[path]`, so `use super::*` brings in that module's items.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::*;

#[test]
fn detects_period_8_triple_repeat() {
    let pat: Vec<u32> = (1..=8).collect();
    let mut tokens: Vec<u32> = (0..40).collect();
    tokens.extend(pat.iter());
    tokens.extend(pat.iter());
    tokens.extend(pat.iter());
    assert!(detect_thinking_token_loop(
        &tokens,
        WatchdogParams::default()
    ));
}

#[test]
fn rejects_two_repeats() {
    // 2026-09-25: 60 tokens clear THINK_LOOP_MIN_TOKENS, so only the repeat
    // count rejects this.
    let pat: Vec<u32> = (100..=104).collect();
    let mut tokens: Vec<u32> = (0u32..50).collect();
    tokens.extend(pat.iter());
    tokens.extend(pat.iter());
    assert!(!detect_thinking_token_loop(
        &tokens,
        WatchdogParams::default()
    ));
}

#[test]
fn rejects_numbered_list_reasoning() {
    let tokens: Vec<u32> = (0u32..80).collect();
    assert!(!detect_thinking_token_loop(
        &tokens,
        WatchdogParams::default()
    ));
}

#[test]
fn detects_short_period_fence_loop() {
    // 2026-09-25: A 10-token pattern repeated 4 times, after 50 distinct
    // tokens that lift the length over THINK_LOOP_MIN_TOKENS.
    let pat: Vec<u32> = vec![7, 6, 5, 4, 3, 2, 1, 0, 9, 8];
    let mut tokens: Vec<u32> = (100u32..150).collect();
    for _ in 0..4 {
        tokens.extend(pat.iter());
    }
    assert!(detect_thinking_token_loop(
        &tokens,
        WatchdogParams::default()
    ));
}

#[test]
fn rejects_fence_body_with_varying_prefixes() {
    // 2026-09-25: The end-anchored detector must not fire here: the varying
    // connective prefixes leave no fixed period at the end of the buffer.
    let fence: Vec<u32> = vec![100, 101, 102, 103, 104, 105, 106, 107, 108, 109];
    let prefixes: [&[u32]; 4] = [
        // 2026-09-25: Stand-ins for connective prefixes of different lengths.
        &[200, 201],
        &[202, 203],
        &[204, 205, 206],
        &[207],
    ];
    let mut tokens: Vec<u32> = (0..30).collect();
    for pre in prefixes.iter() {
        tokens.extend(pre.iter());
        tokens.extend(fence.iter());
    }
    assert!(
        !detect_thinking_token_loop(&tokens, WatchdogParams::default()),
        "end-anchored detector intentionally does not fire on varying-prefix patterns"
    );
}

// 2026-09-25: Content-phase loop detector tests.

#[test]
fn content_loop_detects_sentence_triple_repeat() {
    let sentence: Vec<u32> = (1000..1022).collect();
    let mut tokens: Vec<u32> = (0..100).collect();
    tokens.extend(sentence.iter());
    tokens.extend(sentence.iter());
    tokens.extend(sentence.iter());
    assert!(
        detect_content_token_loop(&tokens),
        "22-token sentence repeating 3× must trigger content-loop watchdog"
    );
}

#[test]
fn content_loop_rejects_short_responses() {
    let pat: Vec<u32> = (1..=10).collect();
    // 2026-09-25: `CONTENT_LOOP_MIN_TOKENS - 4` tokens in total, the three
    // copies included.
    let prior_len = (CONTENT_LOOP_MIN_TOKENS as usize).saturating_sub(3 * pat.len() + 4);
    let mut tokens: Vec<u32> = (50u32..50 + prior_len as u32).collect();
    tokens.extend(pat.iter());
    tokens.extend(pat.iter());
    tokens.extend(pat.iter());
    assert!(
        tokens.len() < CONTENT_LOOP_MIN_TOKENS as usize,
        "test setup error: tokens.len()={} exceeds MIN_TOKENS={}",
        tokens.len(),
        CONTENT_LOOP_MIN_TOKENS,
    );
    assert!(
        !detect_content_token_loop(&tokens),
        "responses under {} tokens must not trigger watchdog",
        CONTENT_LOOP_MIN_TOKENS
    );
}

#[test]
fn content_loop_rejects_legitimate_prose() {
    let tokens: Vec<u32> = (0u32..200).collect();
    assert!(
        !detect_content_token_loop(&tokens),
        "legitimate prose with no repeat must not trigger watchdog"
    );
}

#[test]
fn content_loop_accepts_three_repeats() {
    let sentence: Vec<u32> = (500..530).collect();
    let mut tokens: Vec<u32> = (0..100).collect();
    tokens.extend(sentence.iter());
    tokens.extend(sentence.iter());
    tokens.extend(sentence.iter());
    assert!(
        detect_content_token_loop(&tokens),
        "three byte-exact 30-token repeats must trigger watchdog at MIN_REPEATS={}",
        CONTENT_LOOP_MIN_REPEATS,
    );
}

#[test]
fn content_loop_rejects_two_repeats() {
    let sentence: Vec<u32> = (500..530).collect();
    let mut tokens: Vec<u32> = (0..100).collect();
    tokens.extend(sentence.iter());
    tokens.extend(sentence.iter());
    assert!(
        !detect_content_token_loop(&tokens),
        "two byte-exact 30-token repeats must NOT trigger at MIN_REPEATS={}",
        CONTENT_LOOP_MIN_REPEATS,
    );
}

#[test]
fn content_loop_rejects_single_occurrence() {
    let sentence: Vec<u32> = (500..530).collect();
    let mut tokens: Vec<u32> = (0..100).collect();
    tokens.extend(sentence.iter());
    assert!(
        !detect_content_token_loop(&tokens),
        "single occurrence (no repeat) must not trigger watchdog"
    );
}

// 2026-09-25: Digit-normalized content-loop detector tests. Fixture
// convention: `numeric_mask` (length 1100) marks ids 100..=199 numeric;
// the structural ids below 100 and the prefix noise ids 900..990 are not.

fn numeric_mask() -> Vec<bool> {
    let mut m = vec![false; 1100];
    for (i, slot) in m.iter_mut().enumerate() {
        *slot = (100..=199).contains(&i);
    }
    m
}

/// 2026-09-25: 12-token template `[1..=6, <num>, 7..=11]`; `num` varies each
/// repeat so the exact detector cannot match, but normalization collapses
/// every repeat to an identical period.
fn varying_template_stream(repeats: u32) -> Vec<u32> {
    let mut t: Vec<u32> = (900u32..990).collect();
    for k in 0..repeats {
        t.extend([1, 2, 3, 4, 5, 6]);
        t.push(100 + k);
        t.extend([7, 8, 9, 10, 11]);
    }
    t
}

#[test]
fn norm_fires_on_varying_numeric_template() {
    let t = varying_template_stream(5);
    let mask = numeric_mask();
    assert!(
        !detect_content_token_loop(&t),
        "exact detector must miss: integer tokens differ every repeat"
    );
    assert!(
        detect_content_token_loop_normalized(&t, &mask),
        "normalized detector must catch the fixed template (5 repeats >= 4)"
    );
}

#[test]
fn norm_rejects_3item_list_and_pure_columns() {
    let mask = numeric_mask();

    // 2026-09-25: (a) 3 repeats, below CONTENT_LOOP_NORM_MIN_REPEATS: a
    // 3-item numbered list must not hard-stop.
    let three = varying_template_stream(3);
    assert!(
        !detect_content_token_loop_normalized(&three, &mask),
        "3 repeats < NORM_MIN_REPEATS=4 must not fire"
    );

    // 2026-09-25: (b) Pure-number column: its numeric tokens are all
    // consecutive, so they collapse to one sentinel and no period repeats.
    let mut col: Vec<u32> = (900u32..990).collect();
    for k in 0..6 {
        col.extend([100 + k; 12]);
    }
    assert!(
        !detect_content_token_loop_normalized(&col, &mask),
        "pure-number period (no structural token) is the exact path's job"
    );

    // 2026-09-25: (c) Pure-prose period (no numeric token): early-out on
    // !has_sentinel — left to the exact detector.
    let mut prose: Vec<u32> = (900u32..960).collect();
    for _ in 0..6 {
        prose.extend([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    }
    assert!(
        !detect_content_token_loop_normalized(&prose, &mask),
        "pure-prose period (no numeric) must defer to exact detector"
    );
}

#[test]
fn exact_prose_loop_still_caught_regression() {
    let mut t: Vec<u32> = (900u32..990).collect();
    for _ in 0..4 {
        t.extend([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    }
    assert!(
        detect_content_token_loop(&t),
        "exact byte-identical content loop must still be caught"
    );
}

#[test]
fn norm_fires_on_variable_length_digit_runs() {
    // 2026-09-25: The index and value are runs of numeric tokens whose
    // lengths differ every line, as when a tokenizer emits one token per
    // digit. Run-collapse makes every line normalize to the same period.
    let mask = numeric_mask();
    // 2026-09-25: Template [1,2,3] <idx-run> [4,5] <val-run> [6,7,8]; the
    // collapsed period is 3 + 1 + 2 + 1 + 3 = 10.
    let mut t: Vec<u32> = (900u32..990).collect();
    for k in 0..5u32 {
        t.extend([1, 2, 3]);
        t.extend(std::iter::repeat_n(100 + (k % 10), 2 + (k % 2) as usize));
        t.extend([4, 5]);
        t.extend(std::iter::repeat_n(101 + (k % 9), 9 + k as usize));
        t.extend([6, 7, 8]);
    }
    assert!(
        !detect_content_token_loop(&t),
        "exact detector misses: digit-run lengths differ every line"
    );
    assert!(
        detect_content_token_loop_normalized(&t, &mask),
        "run-collapse must catch the variable-length digit-run template"
    );
}

#[test]
fn norm_inert_with_empty_mask() {
    // 2026-09-25: mask=&[] -> is_numeric always false -> no sentinel ->
    // early-out.
    let t = varying_template_stream(5);
    assert!(
        !detect_content_token_loop_normalized(&t, &[]),
        "empty mask must make the normalized path inert (fail-open)"
    );
}

// 2026-09-25: Per-request RepetitionDetectionParams override tests.

#[test]
fn override_loosens_content_loop_threshold() {
    // 2026-09-25: Three copies of a 22-token sentence: the built-in
    // threshold fires, a `min_count=4` override does not.
    let sentence: Vec<u32> = (1000..1022).collect();
    let mut tokens: Vec<u32> = (0..100).collect();
    tokens.extend(sentence.iter());
    tokens.extend(sentence.iter());
    tokens.extend(sentence.iter());

    assert!(
        detect_content_token_loop_with(&tokens, None),
        "default thresholds must still fire on 22-token × 3 repeat"
    );

    let strict = crate::api::inference_types::RepetitionDetectionParams {
        min_pattern_size: 2,
        max_pattern_size: 64,
        min_count: 4,
    };
    assert!(
        !detect_content_token_loop_with(&tokens, Some(strict)),
        "stricter min_count=4 override must suppress 3-repeat firing"
    );
}

#[test]
fn override_tightens_content_loop_threshold() {
    // 2026-09-25: Five copies of a 5-token block after 50 tokens of padding
    // that clear CONTENT_LOOP_MIN_TOKENS; an override of period 5 and
    // `min_count=3` fires.
    let pat: Vec<u32> = vec![42, 43, 44, 45, 46];
    let mut tokens: Vec<u32> = (0u32..50).collect();
    for _ in 0..5 {
        tokens.extend(pat.iter());
    }
    let permissive = crate::api::inference_types::RepetitionDetectionParams {
        min_pattern_size: 5,
        max_pattern_size: 5,
        min_count: 3,
    };
    assert!(
        detect_content_token_loop_with(&tokens, Some(permissive)),
        "override (period=5, min_count=3) must catch 5×period-5 tail"
    );
}

#[test]
fn override_applies_to_thinking_loop() {
    let pat: Vec<u32> = vec![7, 6, 5, 4, 3, 2, 1, 0, 9, 8];
    let mut tokens: Vec<u32> = (100u32..150).collect();
    for _ in 0..4 {
        tokens.extend(pat.iter());
    }
    assert!(
        detect_thinking_token_loop_with(&tokens, None, WatchdogParams::default()),
        "default thinking-loop thresholds must still fire on 4× period-10"
    );
    let strict = crate::api::inference_types::RepetitionDetectionParams {
        min_pattern_size: 4,
        max_pattern_size: 20,
        min_count: 6,
    };
    assert!(
        !detect_thinking_token_loop_with(&tokens, Some(strict), WatchdogParams::default()),
        "stricter min_count=6 override must suppress 4-repeat firing"
    );
}

// 2026-09-25: Forced-token fast-path kill-switch parsing.

#[test]
fn forced_token_fastpath_default_enabled() {
    assert!(parse_forced_token_fastpath(None));
}

#[test]
fn forced_token_fastpath_disabled_by_truthy() {
    assert!(!parse_forced_token_fastpath(Some("1")));
    assert!(!parse_forced_token_fastpath(Some("true")));
    assert!(!parse_forced_token_fastpath(Some("TRUE")));
    assert!(!parse_forced_token_fastpath(Some("  true  ")));
}

#[test]
fn forced_token_fastpath_enabled_by_falsy_or_junk() {
    assert!(parse_forced_token_fastpath(Some("0")));
    assert!(parse_forced_token_fastpath(Some("false")));
    assert!(parse_forced_token_fastpath(Some("")));
    assert!(parse_forced_token_fastpath(Some("yes")));
}

/// 2026-09-25: Pins two `WatchdogParams::default()` values: a mid-`<think>`
/// EOS is not honored as an implicit close, and the think-loop watchdog is
/// on. Served runs take both from `ModelBehavior` instead.
#[test]
fn watchdog_defaults_keep_mid_think_eos_block_inert() {
    let d = WatchdogParams::default();
    assert!(
        !d.honor_eos_inside_thinking,
        "honor_eos_inside_thinking must default OFF (opt-in per model)"
    );
    assert!(
        d.enable_think_loop_watchdog,
        "think-loop watchdog must default ON (opt-out per model)"
    );
}
