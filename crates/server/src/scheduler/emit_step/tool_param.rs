// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The tool-call body / parameter-value state machine and its
//! envelope-stuck guard.
//!
//! Owner: scheduler.
//! Invariants:
//! - Tokens of a parameter value (`inside_parameter_body`) are exempt from
//!   the envelope-stuck cap; only envelope tokens count toward it.

use super::*;

/// 2026-09-25: Cap on consecutive envelope tokens (tokens inside
/// `<tool_call>` that are not parameter-value content) since the tool call
/// opened or the last `</parameter>`. Past it, `update_tool_param_state`
/// ends the response with guard `tool_envelope_stuck`.
pub(super) const MAX_TOOL_BODY_TOKENS: u32 = 1024;

/// 2026-09-25: Decision core of the envelope-stuck guard, pure over
/// scalars so it is tested without an `ActiveSeq`. A parameter-value token
/// (`inside_parameter_body`) leaves the streak unchanged, so a large file
/// write never trips the cap; any other token advances it. Returns
/// `(new_streak, exceeded_cap)`.
pub(super) fn advance_envelope_streak(inside_parameter_body: bool, streak: u32) -> (u32, bool) {
    if inside_parameter_body {
        (streak, false)
    } else {
        let s = streak.saturating_add(1);
        (s, s > MAX_TOOL_BODY_TOKENS)
    }
}

/// 2026-09-25: Advance the tool-body / parameter-body state for one emitted
/// token; no-op inside thinking. It works whether or not the caller has
/// already pushed `tok` onto `output_tokens`: `emit_token` calls it before
/// the push, `decode_logits_step` after.
pub fn update_tool_param_state(a: &mut ActiveSeq, tok: u32) {
    if a.inside_thinking {
        return;
    }
    if a.tool_call_start_token == Some(tok) {
        a.inside_tool_body = true;
        a.tool_body_streak_tokens = 0;
        return;
    }
    if a.tool_call_end_token == Some(tok) {
        a.inside_tool_body = false;
        a.tool_body_streak_tokens = 0;
        a.inside_parameter_body = false;
        a.param_body_chars_emitted = 0;
        return;
    }
    if !a.inside_tool_body {
        return;
    }
    let (streak, exceeded) =
        advance_envelope_streak(a.inside_parameter_body, a.tool_body_streak_tokens);
    a.tool_body_streak_tokens = streak;
    if exceeded {
        tracing::warn!(
            streak = a.tool_body_streak_tokens,
            "Stuck in tool-call ENVELOPE for {MAX_TOOL_BODY_TOKENS}+ tokens with no </tool_call> (excludes parameter-value content); ending response (model never closed the envelope — would otherwise burn to max_tokens). Sanitizer will salvage what it can."
        );
        a.guard_stop = Some("tool_envelope_stuck");
        a.finished = true;
    }

    const TOK_LT: u32 = 27;
    const TOK_PARAMETER: u32 = 15704;
    const TOK_EQ: u32 = 28;
    const TOK_GT: u32 = 29;
    const TOK_LT_SLASH: u32 = 510;

    if a.inside_parameter_body {
        // 2026-09-25: provisional close detection. A value can contain
        // other close tags (`</div>`) that start with the same `</` token as
        // `</parameter>`, so the body is left only on the full `</`,
        // `parameter`, `>` sequence. Any other continuation stays in the body
        // and counts the held tokens as body chars. A close tokenized any
        // other way also stays in the body, which cannot trip the cap:
        // value tokens do not advance the streak.
        match a.param_close_pending {
            0 => {
                if tok == TOK_LT_SLASH {
                    a.param_close_pending = 1;
                } else {
                    a.param_body_chars_emitted = a.param_body_chars_emitted.saturating_add(1);
                }
            }
            1 => {
                if tok == TOK_PARAMETER {
                    a.param_close_pending = 2;
                } else {
                    // 2026-09-25: `</` was value content, not a close.
                    a.param_close_pending = 0;
                    a.param_body_chars_emitted = a.param_body_chars_emitted.saturating_add(2);
                }
            }
            _ => {
                a.param_close_pending = 0;
                if tok == TOK_GT {
                    // 2026-09-25: confirmed `</parameter>`: leave the body
                    // and reset the envelope streak.
                    a.inside_parameter_body = false;
                    a.param_body_chars_emitted = 0;
                    a.tool_body_streak_tokens = 0;
                } else {
                    // 2026-09-25: `</parameter` not followed by `>`: value
                    // content; stay in the body.
                    a.param_body_chars_emitted = a.param_body_chars_emitted.saturating_add(3);
                }
            }
        }
        return;
    }

    // 2026-09-25: outside a value: does a `<parameter=KEY>` opener end at
    // this `>`? Look back up to 8 tokens for `<`, `parameter`, `=` with no
    // `</` or `>` after it.
    if tok != TOK_GT {
        return;
    }
    // 2026-09-25: leave `tok` itself out of the lookback, whether or not
    // the caller has pushed it yet.
    let n = a.output_tokens.len();
    let n_for_lookback = if n > 0 && a.output_tokens[n - 1] == tok {
        n - 1
    } else {
        n
    };
    if n_for_lookback < 3 {
        return;
    }
    let start = n_for_lookback.saturating_sub(8);
    let window = &a.output_tokens[start..n_for_lookback];
    let mut sig_idx: Option<usize> = None;
    for i in 0..window.len().saturating_sub(2) {
        if window[i] == TOK_LT && window[i + 1] == TOK_PARAMETER && window[i + 2] == TOK_EQ {
            sig_idx = Some(i + 3);
        }
    }
    let Some(after_eq) = sig_idx else { return };
    let body_segment = &window[after_eq..];
    let intervening_close = body_segment
        .iter()
        .any(|&t| t == TOK_LT_SLASH || t == TOK_GT);
    if !intervening_close {
        a.inside_parameter_body = true;
        a.param_body_chars_emitted = 0;
    }
}
