// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The client stop-sequence hold-back that `handle_token` runs on each
//! content delta. A pure function, so it is tested without a `StreamCtx`.
//!
//! Owner: server streaming API.
//! Invariants: none beyond the types.

/// 2026-09-26: Stop-string accumulator with a held-back tail. Returns the bytes to
/// forward for this delta; up to `buffer_len` bytes stay in `accumulated_content` for
/// the next call or for `handle_done` to flush at stream close.
/// 1. Append `new_chars` to the accumulator.
/// 2. Search the recent window of the accumulator for every stop string; the earliest
///    match wins.
/// 3a. On a match, truncate the accumulator at the match and return the bytes before
///     it that were not yet sent; the stop string itself is never returned.
/// 3b. Otherwise hold back the last `buffer_len` bytes and return everything from the
///     previously emitted offset to there, snapped down to a UTF-8 char boundary.
///
/// `*triggered` must be `false` on entry (a `debug_assert`); it becomes `true` only on a
/// match.
pub(super) fn apply_stop_string_holdback(
    new_chars: &str,
    stop_strings: &[String],
    buffer_len: usize,
    accumulated_content: &mut String,
    emitted_len: &mut usize,
    triggered: &mut bool,
) -> String {
    debug_assert!(!*triggered, "caller must gate on !triggered");
    accumulated_content.push_str(new_chars);

    // 2026-09-26: Earlier calls already searched everything before this window, so only
    // a match that overlaps the new chars can be new. The window reaches back
    // `new_chars.len() + buffer_len + max_stop_len` bytes, keeping each call's cost
    // independent of the total length.
    let max_stop_len = stop_strings.iter().map(String::len).max().unwrap_or(0);
    let search_start = {
        let raw = accumulated_content
            .len()
            .saturating_sub(new_chars.len() + buffer_len + max_stop_len);
        accumulated_content.floor_char_boundary(raw)
    };
    let matched_pos = stop_strings
        .iter()
        .filter_map(|s| accumulated_content[search_start..].find(s.as_str()))
        .min()
        .map(|rel| rel + search_start);

    if let Some(pos) = matched_pos {
        accumulated_content.truncate(pos);
        let emit_start = (*emitted_len).min(pos);
        let out = accumulated_content[emit_start..pos].to_string();
        *emitted_len = pos;
        *triggered = true;
        return out;
    }

    // 2026-09-26: Snap down to a char boundary, so the returned prefix never ends inside
    // a codepoint.
    let acc_len = accumulated_content.len();
    let raw_emit_end = acc_len.saturating_sub(buffer_len);
    let emit_end = accumulated_content.floor_char_boundary(raw_emit_end);
    let emit_start = (*emitted_len).min(emit_end);
    let out = accumulated_content[emit_start..emit_end].to_string();
    *emitted_len = emit_end;
    out
}
