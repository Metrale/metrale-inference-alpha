// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tag strippers for the thinking-phase emit path in `handle_token`.
//! Removing a tag inserts one space when the characters on both sides are
//! non-whitespace, so the words around the tag are not glued together.
//!
//! Owner: server chat streaming.
//! Invariants: none beyond the types.

/// 2026-09-26: Strip the `[start..end_exclusive)` byte range from `s`, inserting
/// a single space at the splice point when removal would otherwise
/// glue two non-whitespace characters together. Panics if the range is
/// out of bounds or not on `char` boundaries.
pub(super) fn strip_preserving_boundary(s: &str, start: usize, end_exclusive: usize) -> String {
    debug_assert!(start <= end_exclusive && end_exclusive <= s.len());
    let before = &s[..start];
    let after = &s[end_exclusive..];
    let needs_space = !before.is_empty()
        && !after.is_empty()
        && !before.ends_with(char::is_whitespace)
        && !after.starts_with(char::is_whitespace);
    if needs_space {
        format!("{before} {after}")
    } else {
        format!("{before}{after}")
    }
}

/// 2026-09-26: Boundary-preserving variant of `s.replace(tag, "")`: strips
/// every occurrence of `tag` through [`strip_preserving_boundary`], repeating
/// until none is left. `tag` must be non-empty; an empty tag never terminates.
pub(super) fn strip_all_preserving_boundary(s: &str, tag: &str) -> String {
    let mut out = s.to_string();
    while let Some(pos) = out.find(tag) {
        out = strip_preserving_boundary(&out, pos, pos + tag.len());
    }
    out
}

/// 2026-09-26: Diagnostic trace for the thinking-phase emit path. When
/// `METRALE_THINKING_DECODE_TRACE` is present (any value), logs a DEBUG
/// `thinking-decode-trace` event with the lengths and first 64 characters of
/// the raw delta and of the stripped text, and whether stripping changed it.
pub(super) fn maybe_log_decode_trace(raw: &str, cleaned: &str, full_len: usize, emitted_in: usize) {
    if std::env::var_os("METRALE_THINKING_DECODE_TRACE").is_none() {
        return;
    }
    let raw_head: String = raw.chars().take(64).collect();
    let cleaned_head: String = cleaned.chars().take(64).collect();
    tracing::debug!(
        full_len,
        emitted_in,
        raw_len = raw.len(),
        cleaned_len = cleaned.len(),
        mutated = raw != cleaned,
        raw = %raw_head,
        cleaned = %cleaned_head,
        "thinking-decode-trace"
    );
}

#[cfg(test)]
mod tests {
    use super::strip_preserving_boundary;

    #[test]
    fn inserts_space_between_glued_words() {
        let s = "the<tool_call>foo</tool_call>project";
        let start = s.find("<tool_call>").unwrap();
        let end = s.find("</tool_call>").unwrap() + "</tool_call>".len();
        assert_eq!(strip_preserving_boundary(s, start, end), "the project");
    }

    #[test]
    fn no_double_space_when_already_separated() {
        let s = "before <tool_call>foo</tool_call> after";
        let start = s.find("<tool_call>").unwrap();
        let end = s.find("</tool_call>").unwrap() + "</tool_call>".len();
        assert_eq!(strip_preserving_boundary(s, start, end), "before  after");
    }

    #[test]
    fn no_space_when_after_starts_with_newline() {
        let s = "context</parameter>\nnext line";
        let start = s.find("</parameter>").unwrap();
        let end = start + "</parameter>".len();
        assert_eq!(
            strip_preserving_boundary(s, start, end),
            "context\nnext line"
        );
    }

    #[test]
    fn empty_before_is_safe() {
        let s = "<think>hello";
        let start = 0;
        let end = "<think>".len();
        assert_eq!(strip_preserving_boundary(s, start, end), "hello");
    }
}
