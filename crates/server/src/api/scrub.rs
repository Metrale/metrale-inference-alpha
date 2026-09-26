// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Complete-tag scrubber for tool-call markup in content. Its only caller is
//! `stream_guards::flush_content_sanitizer`, which runs it on the sanitizer's held-back
//! tail and skips it when the markers declare envelopes.
//!
//! Owner: server API (sanitizer).
//! Invariants: none beyond the types.

use crate::tool_parser;

/// 2026-09-26: Remove every complete tool-call tag in `text`. A `close` or `orphan_open`
/// marker that ends in `>` (for qwen3_coder `</tool_call>`, `<tool_call>`, `</_call>`) is
/// removed as it stands. An `orphan_open` marker without a `>` (`<function=`,
/// `<parameter=`) is removed through the next `>`; when no `>` follows, the rest of
/// `text` is dropped. `text` is returned unchanged when both marker lists are empty.
pub(crate) fn scrub_tool_tags(text: &str, markers: &tool_parser::LeakMarkers) -> String {
    if text.is_empty() || (markers.orphan_open.is_empty() && markers.close.is_empty()) {
        return text.to_string();
    }
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    'outer: while i < text.len() {
        if bytes[i] == b'<' {
            for m in markers.close.iter().chain(markers.orphan_open.iter()) {
                if m.ends_with('>') && text[i..].starts_with(m) {
                    i += m.len();
                    continue 'outer;
                }
            }
            for m in markers.orphan_open.iter() {
                if !m.ends_with('>') && text[i..].starts_with(m) {
                    match text[i..].find('>') {
                        Some(gt) => {
                            i += gt + 1;
                            continue 'outer;
                        }
                        None => return out,
                    }
                }
            }
        }
        let ch_len = text[i..].chars().next().map(|c| c.len_utf8()).unwrap_or(1);
        out.push_str(&text[i..i + ch_len]);
        i += ch_len;
    }
    out
}
