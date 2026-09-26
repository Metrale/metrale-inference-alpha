// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Thinking- and content-loop degeneration detectors:
//! end-anchored periodic-repeat checks over the trailing tokens.
//!
//! Owner: scheduler.
//! Invariants:
//! - Every detector is end-anchored: it fires only when the last
//!   `pattern_len` token ids equal each of the `min_repeats - 1`
//!   `pattern_len`-token windows immediately before them (after digit
//!   normalization, for the normalized detector).

use super::*;

/// 2026-09-25: True when `tokens` holds at least `THINK_LOOP_MIN_TOKENS`
/// tokens and, for some period `p` in `[THINK_LOOP_PERIOD_MIN,
/// THINK_LOOP_PERIOD_MAX]`, its last `p * wp.think_loop_min_repeats` tokens
/// are that many exact copies of one `p`-token window.
/// `wp.think_loop_scan_window` has no effect (see [`detect_token_loop`]).
pub fn detect_thinking_token_loop(tokens: &[u32], wp: WatchdogParams) -> bool {
    detect_thinking_token_loop_with(tokens, None, wp)
}

/// 2026-09-25: Per-sequence override variant of [`detect_thinking_token_loop`].
/// When `override_` is `Some(p)`, uses `p.min_pattern_size`,
/// `p.max_pattern_size`, `p.min_count` as the period and repeat
/// thresholds. When `None`, uses the `THINK_LOOP_PERIOD_*` range and
/// `wp.think_loop_min_repeats`.
pub fn detect_thinking_token_loop_with(
    tokens: &[u32],
    override_: Option<crate::api::inference_types::RepetitionDetectionParams>,
    wp: WatchdogParams,
) -> bool {
    let (period_min, period_max, min_repeats) = match override_ {
        Some(p) => (
            p.min_pattern_size as usize,
            p.max_pattern_size as usize,
            p.min_count as usize,
        ),
        None => (
            THINK_LOOP_PERIOD_MIN,
            THINK_LOOP_PERIOD_MAX,
            wp.think_loop_min_repeats,
        ),
    };
    let scan_window = match override_ {
        Some(_) => 0, // 2026-09-25: `detect_token_loop` ignores the scan window.
        None => wp.think_loop_scan_window,
    };
    detect_token_loop(
        tokens,
        THINK_LOOP_MIN_TOKENS as usize,
        period_min,
        period_max,
        min_repeats,
        scan_window,
    )
}

/// 2026-09-25: Content-phase analogue of [`detect_thinking_token_loop`],
/// with the built-in `CONTENT_LOOP_*` thresholds.
pub fn detect_content_token_loop(tokens: &[u32]) -> bool {
    detect_content_token_loop_with(tokens, None)
}

/// 2026-09-25: Per-sequence override variant of [`detect_content_token_loop`].
/// `Some(p)` uses `p.min_pattern_size`, `p.max_pattern_size`,
/// `p.min_count`; `None` falls back to the built-in content-loop
/// constants.
pub fn detect_content_token_loop_with(
    tokens: &[u32],
    override_: Option<crate::api::inference_types::RepetitionDetectionParams>,
) -> bool {
    let (period_min, period_max, min_repeats) = match override_ {
        Some(p) => (
            p.min_pattern_size as usize,
            p.max_pattern_size as usize,
            p.min_count as usize,
        ),
        None => (
            CONTENT_LOOP_PERIOD_MIN,
            CONTENT_LOOP_PERIOD_MAX,
            CONTENT_LOOP_MIN_REPEATS,
        ),
    };
    detect_token_loop(
        tokens,
        CONTENT_LOOP_MIN_TOKENS as usize,
        period_min,
        period_max,
        min_repeats,
        CONTENT_LOOP_SCAN_WINDOW,
    )
}

/// 2026-09-25: Digit-normalized content-loop detector. Maps each run of
/// numeric tokens (per `mask`) in the last `CONTENT_LOOP_SCAN_WINDOW` tokens
/// to one [`NUMERIC_SENTINEL`], then period-matches. This catches a fixed
/// line template (`- B(46) = N\n`) whose integer payload varies each line,
/// which the exact [`detect_content_token_loop`] never matches.
///
/// Only that tail is copied and normalized. False-positive mitigation: the
/// `None` path requires `CONTENT_LOOP_NORM_MIN_REPEATS` copies, and the
/// matched period must contain both a sentinel (numeric) and a non-sentinel
/// (structural) token, leaving pure-number columns and pure-prose loops to
/// the exact path.
pub fn detect_content_token_loop_normalized(tokens: &[u32], mask: &[bool]) -> bool {
    detect_content_token_loop_normalized_with(tokens, mask, None)
}

/// 2026-09-25: Per-sequence override variant of
/// [`detect_content_token_loop_normalized`]. `Some(p)` substitutes the
/// caller's `(min_pattern_size, max_pattern_size, min_count)` for the
/// built-in constants. `None` uses the `CONTENT_LOOP_PERIOD_*` range and
/// `CONTENT_LOOP_NORM_MIN_REPEATS`.
pub fn detect_content_token_loop_normalized_with(
    tokens: &[u32],
    mask: &[bool],
    override_: Option<crate::api::inference_types::RepetitionDetectionParams>,
) -> bool {
    let n = tokens.len();
    if n < CONTENT_LOOP_MIN_TOKENS as usize {
        return false;
    }
    let tail_start = n.saturating_sub(CONTENT_LOOP_SCAN_WINDOW);
    let is_numeric = |t: u32| (t as usize) < mask.len() && mask[t as usize];
    // 2026-09-25: Map numeric tokens to the sentinel and collapse
    // consecutive sentinels to one. A tokenizer that splits an integer into
    // one token per digit gives a variable number of numeric tokens per
    // integer; a 1:1 map would leave the period varying line to line.
    // Collapsing makes `- B(<digits>) = <digits>\n` identical regardless of
    // digit count.
    let mut norm: Vec<u32> = Vec::with_capacity(CONTENT_LOOP_SCAN_WINDOW);
    for &t in &tokens[tail_start..] {
        if is_numeric(t) {
            if norm.last() != Some(&NUMERIC_SENTINEL) {
                norm.push(NUMERIC_SENTINEL);
            }
        } else {
            norm.push(t);
        }
    }
    // 2026-09-25: No qualifying period can exist without both kinds of
    // token, so return before the period scan.
    let has_sentinel = norm.contains(&NUMERIC_SENTINEL);
    let has_struct = norm.iter().any(|&t| t != NUMERIC_SENTINEL);
    if !has_sentinel || !has_struct {
        return false;
    }
    let (period_min, period_max, min_repeats) = match override_ {
        Some(p) => (
            p.min_pattern_size as usize,
            p.max_pattern_size as usize,
            p.min_count as usize,
        ),
        None => (
            CONTENT_LOOP_PERIOD_MIN,
            CONTENT_LOOP_PERIOD_MAX,
            CONTENT_LOOP_NORM_MIN_REPEATS,
        ),
    };
    detect_token_loop_with_period(
        &norm,
        period_min,
        period_max,
        min_repeats,
        CONTENT_LOOP_SCAN_WINDOW,
    )
}

/// 2026-09-25: Whether the model is looping right now. For each period from
/// `period_min` to `period_max`, the last `pattern_len` tokens must equal
/// each of the `min_repeats - 1` windows before them
/// ([`has_repeating_pattern_anchored`]). Anchoring at the end means a
/// pattern the model has already moved past never matches, as it would for
/// a scan anywhere in the window. False below `min_tokens` or when
/// `min_repeats < 2`. `_scan_window` is unused: the check reads at most
/// `pattern_len * min_repeats` tokens.
pub fn detect_token_loop(
    tokens: &[u32],
    min_tokens: usize,
    period_min: usize,
    period_max: usize,
    min_repeats: usize,
    _scan_window: usize,
) -> bool {
    let n = tokens.len();
    if n < min_tokens {
        return false;
    }
    if min_repeats < 2 {
        return false;
    }
    let period_min = period_min.max(1);
    for pattern_len in period_min..=period_max {
        if pattern_len * min_repeats > n {
            return false;
        }
        if has_repeating_pattern_anchored(tokens, pattern_len, min_repeats) {
            return true;
        }
    }
    false
}

/// 2026-09-25: End-anchored repeat check. For each offset `o` in
/// `1..=pattern_len` from the end, the token there must equal the token
/// `pattern_len * m` positions earlier, for every `m` in `1..min_repeats`.
///
/// Caller must ensure `tokens.len() >= pattern_len * min_repeats`.
#[inline]
fn has_repeating_pattern_anchored(tokens: &[u32], pattern_len: usize, min_repeats: usize) -> bool {
    let n = tokens.len();
    for offset_in_window in 1..=pattern_len {
        let target = tokens[n - offset_in_window];
        for m in 1..min_repeats {
            let idx = n - (pattern_len * m + offset_in_window);
            if tokens[idx] != target {
                return false;
            }
        }
    }
    true
}

/// 2026-09-25: The end-anchored check of [`detect_token_loop`] for a
/// digit-normalized sequence. A period counts only when its window (the last
/// `pattern_len` tokens) holds both a [`NUMERIC_SENTINEL`] and a non-sentinel
/// token; pure-number columns and pure-prose loops are the exact detector's job.
fn detect_token_loop_with_period(
    tokens: &[u32],
    period_min: usize,
    period_max: usize,
    min_repeats: usize,
    _scan_window: usize,
) -> bool {
    let n = tokens.len();
    if min_repeats < 2 {
        return false;
    }
    let period_min = period_min.max(1);
    for pattern_len in period_min..=period_max {
        if pattern_len * min_repeats > n {
            return false;
        }
        let window = &tokens[n - pattern_len..];
        let has_numeric = window.contains(&NUMERIC_SENTINEL);
        let has_structural = window.iter().any(|&t| t != NUMERIC_SENTINEL);
        if !(has_numeric && has_structural) {
            continue;
        }
        if has_repeating_pattern_anchored(tokens, pattern_len, min_repeats) {
            return true;
        }
    }
    false
}
