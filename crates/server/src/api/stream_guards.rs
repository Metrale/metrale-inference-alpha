// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Guards on the outbound text of a chat stream; none of them changes the
//! prompt.
//!
//! - [`bump_f12_tool_call_count`]: counts tool calls and sets the stop flag past the cap.
//! - [`check_loop_watchdog`]: reports a repeated line or phrase in streamed content.
//! - [`flush_content_sanitizer`]: drains the content sanitizer's held-back tail, at the
//!   end of the stream and before each tool-call event.
//!
//! Owner: server streaming API.
//! Invariants: none beyond the types.

use crate::tool_parser;

/// 2026-09-26: Add one to `count`; when `count` exceeds `max` and `stop` is not yet set,
/// set it and log a warning. The chat stream passes `max` from
/// `METRALE_MAX_TOOL_CALLS_PER_RESPONSE`, 12 when unset or unparsable
/// (`chat_stream/mod.rs`).
pub fn bump_f12_tool_call_count(count: &mut usize, max: usize, stop: &mut bool) {
    *count += 1;
    if *count > max && !*stop {
        tracing::warn!(
            emitted = *count,
            max,
            "tool-call cap reached; ending response"
        );
        *stop = true;
    }
}

/// 2026-09-26: Append `text` to `loop_scan_buf` and report a repetition loop. Once the
/// buffer passes 10,240 bytes it is cut back to its last 8,192 bytes at most. The needle
/// is the newest line longer than 15 bytes (trimmed) that is not a code fence, lowercased
/// with whitespace runs collapsed. True when 4 or more buffer lines normalise to the
/// needle, or, for a needle of 30 bytes or more, when the lowercased buffer contains it 4
/// or more times. Returns false, without touching the buffer, when `already_triggered`
/// or `text` is empty.
pub fn check_loop_watchdog(
    text: &str,
    loop_scan_buf: &mut String,
    already_triggered: bool,
) -> bool {
    if already_triggered || text.is_empty() {
        return false;
    }
    loop_scan_buf.push_str(text);
    if loop_scan_buf.len() > 10_240 {
        let drop = loop_scan_buf.len() - 8_192;
        let cut = loop_scan_buf
            .char_indices()
            .map(|(i, _)| i)
            .find(|&i| i >= drop)
            .unwrap_or(drop);
        loop_scan_buf.drain(..cut);
    }
    let last_line = loop_scan_buf
        .lines()
        .rev()
        .find(|l| l.trim().len() > 15 && !l.trim_start().starts_with("```"))
        .map(|s| s.to_string());
    let Some(line) = last_line else {
        return false;
    };
    fn norm(s: &str) -> String {
        let lowered = s.trim().to_ascii_lowercase();
        let mut out = String::with_capacity(lowered.len());
        let mut prev_space = false;
        for ch in lowered.chars() {
            if ch.is_ascii_whitespace() {
                if !prev_space && !out.is_empty() {
                    out.push(' ');
                }
                prev_space = true;
            } else {
                out.push(ch);
                prev_space = false;
            }
        }
        if out.ends_with(' ') {
            out.pop();
        }
        out
    }
    let needle = norm(&line);
    if needle.is_empty() {
        return false;
    }
    let exact_occurrences = loop_scan_buf.lines().filter(|l| norm(l) == needle).count();
    if exact_occurrences >= 4 {
        tracing::warn!(
            occurrences = exact_occurrences,
            line_len = needle.len(),
            "loop watchdog fired — repeated line (fuzzy-match) in post-detector content"
        );
        return true;
    }
    if needle.len() >= 30 {
        let lowered_buf = loop_scan_buf.to_ascii_lowercase();
        let mut count = 0usize;
        let mut start = 0usize;
        while let Some(rel) = lowered_buf[start..].find(&needle) {
            count += 1;
            start += rel + needle.len();
            if count >= 4 {
                break;
            }
        }
        if count >= 4 {
            tracing::warn!(
                occurrences = count,
                line_len = needle.len(),
                "loop watchdog fired — repeated phrase (substring) in post-detector content"
            );
            return true;
        }
    }
    false
}

/// 2026-09-26: Take the sanitizer's held-back tail and return what may still be emitted.
/// When suppression is active the tail is dropped and the flag cleared. Otherwise
/// complete tags are scrubbed (unless the markers declare envelopes), a trailing partial
/// close marker is cut, and the rest is returned, unless it is a lone partial tag: it
/// starts with `<`, holds no whitespace after trimming its end, and is shorter than the
/// longest `orphan_open`/`close` marker.
pub fn flush_content_sanitizer(
    tag_scan_buf: &mut String,
    suppressing_param_leak: &mut bool,
    markers: &tool_parser::LeakMarkers,
) -> String {
    if *suppressing_param_leak {
        tag_scan_buf.clear();
        *suppressing_param_leak = false;
        return String::new();
    }
    if tag_scan_buf.is_empty() {
        return String::new();
    }
    let tag_max: usize = markers
        .orphan_open
        .iter()
        .chain(markers.close.iter())
        .map(|t| t.len())
        .max()
        .unwrap_or(0);
    let mut final_text = std::mem::take(tag_scan_buf);
    // 2026-09-26: Remove complete tool-call tags from the tail. Skipped when the markers
    // declare envelopes: `sanitize_content_chunk` passes the tags inside an envelope
    // through as content, and the scrub would remove them.
    if markers.envelope_open.is_empty() {
        final_text = super::scrub::scrub_tool_tags(&final_text, markers);
    }
    // 2026-09-26: Cut a trailing partial close marker: when the text from the last `<` to
    // the end is a strict prefix of a `close` marker, drop it and keep the text before
    // it. The partial-tag check below only catches a tail that starts with `<` and holds
    // no whitespace, so it misses a partial close after other text, such as `\n\n</_call`.
    if let Some(lt) = final_text.rfind('<') {
        let suffix = &final_text[lt..];
        let is_partial_close = markers
            .close
            .iter()
            .any(|t| t.len() > suffix.len() && t.starts_with(suffix));
        if is_partial_close {
            final_text.truncate(lt);
        }
    }
    let looks_like_partial_tag = {
        let t = final_text.trim_end();
        tag_max > 0 && t.starts_with('<') && !t.contains(char::is_whitespace) && t.len() < tag_max
    };
    if looks_like_partial_tag {
        String::new()
    } else {
        final_text
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f12_under_cap_does_not_stop() {
        let (mut count, mut stop) = (0usize, false);
        bump_f12_tool_call_count(&mut count, 12, &mut stop);
        assert_eq!(count, 1);
        assert!(!stop);
    }

    #[test]
    fn f12_at_cap_does_not_stop() {
        let (mut count, mut stop) = (11usize, false);
        bump_f12_tool_call_count(&mut count, 12, &mut stop);
        assert_eq!(count, 12);
        assert!(!stop);
    }

    #[test]
    fn f12_over_cap_trips_stop() {
        let (mut count, mut stop) = (12usize, false);
        bump_f12_tool_call_count(&mut count, 12, &mut stop);
        assert_eq!(count, 13);
        assert!(stop);
    }

    #[test]
    fn watchdog_already_triggered_returns_false() {
        let mut buf = String::new();
        assert!(!check_loop_watchdog("anything", &mut buf, true));
    }

    #[test]
    fn watchdog_empty_text_returns_false() {
        let mut buf = String::new();
        assert!(!check_loop_watchdog("", &mut buf, false));
    }

    #[test]
    fn watchdog_four_identical_lines_fires() {
        let mut buf = String::new();
        let line = "Running cargo test on the project\n";
        assert!(!check_loop_watchdog(line, &mut buf, false));
        assert!(!check_loop_watchdog(line, &mut buf, false));
        assert!(!check_loop_watchdog(line, &mut buf, false));
        assert!(check_loop_watchdog(line, &mut buf, false));
    }

    #[test]
    fn watchdog_buffer_caps_at_10kb() {
        let mut buf = String::new();
        let big = "x".repeat(5000);
        check_loop_watchdog(&big, &mut buf, false);
        check_loop_watchdog(&big, &mut buf, false);
        check_loop_watchdog(&big, &mut buf, false);
        assert!(
            buf.len() <= 10_240,
            "buffer should self-trim, got {}",
            buf.len()
        );
    }
}
