// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Chunk-level sanitizer tests that call `sanitize_content_chunk` and
//! `flush_content_sanitizer` directly on raw buffers. `super::` is the `sanitizer` test
//! module, which imports both functions.
//!
//! Owner: server API.
//! Invariants: none beyond the types.

use super::flush_content_sanitizer;
use crate::tool_parser::{LeakMarkers, Qwen3CoderParser, ToolCallParser};

/// 2026-09-26: `sanitize_content_chunk` with a fresh `inside_envelope = false` on every
/// call. Its callers use markers that declare no envelopes (`LeakMarkers::EMPTY` and
/// `Qwen3CoderParser`), so the flag would stay false anyway.
fn sanitize_content_chunk(
    text: &str,
    tag_scan_buf: &mut String,
    suppressing_param_leak: &mut bool,
    markers: &LeakMarkers,
) -> String {
    let mut inside_envelope = false;
    super::sanitize_content_chunk(
        text,
        tag_scan_buf,
        suppressing_param_leak,
        &mut inside_envelope,
        markers,
    )
}

/// 2026-09-26: Inside each `MinimaxXmlParser` envelope spelling, the inner
/// `<invoke …></invoke>` block and the envelope tags pass through unsuppressed.
#[test]
fn sanitizer_envelope_open_disables_orphan_suppression() {
    let markers = crate::tool_parser::MinimaxXmlParser.leak_markers();

    for envelope_open in &["<minimax:tool_call>", "<minimax:_call>", "<tool_call>"] {
        let envelope_close = match *envelope_open {
            "<minimax:tool_call>" => "</minimax:tool_call>",
            "<minimax:_call>" => "</minimax:_call>",
            _ => "</tool_call>",
        };
        let body = format!(
            "{envelope_open}\n<invoke name=\"bash\">\n<parameter name=\"command\">uname -r</parameter>\n</invoke>\n{envelope_close}"
        );
        let mut buf = String::new();
        let mut suppress = false;
        let mut env = false;
        let out = super::sanitize_content_chunk(&body, &mut buf, &mut suppress, &mut env, &markers);
        assert!(
            out.contains("<invoke name=\"bash\">"),
            "envelope {envelope_open}: <invoke> must survive: out={out:?}"
        );
        assert!(
            out.contains("uname -r"),
            "envelope {envelope_open}: command must survive: out={out:?}"
        );
        assert!(
            out.contains("</invoke>"),
            "envelope {envelope_open}: </invoke> must survive: out={out:?}"
        );
        assert!(
            out.contains(envelope_open),
            "envelope_open bytes must pass through: out={out:?}"
        );
        assert!(
            out.contains(envelope_close),
            "envelope_close bytes must pass through: out={out:?}"
        );
        assert!(!suppress, "envelope path must not enter orphan suppression");
        assert!(!env, "envelope state cleared after close");
    }
}

#[test]
fn sanitizer_orphan_invoke_outside_envelope_still_suppressed() {
    let markers = crate::tool_parser::MinimaxXmlParser.leak_markers();
    let body = "prefix<invoke name=\"bash\">cmd</invoke>tail";
    let mut buf = String::new();
    let mut suppress = false;
    let mut env = false;
    let out = super::sanitize_content_chunk(body, &mut buf, &mut suppress, &mut env, &markers);
    assert!(
        out.starts_with("prefix"),
        "non-orphan prefix emits: {out:?}"
    );
    assert!(
        !out.contains("<invoke"),
        "stray <invoke> must still be suppressed: {out:?}"
    );
    assert!(
        !out.contains("cmd"),
        "suppressed body bytes must not leak: {out:?}"
    );
}

#[test]
fn sanitizer_noop_for_empty_markers() {
    let mut buf = String::new();
    let mut suppress = false;
    let out = sanitize_content_chunk(
        "<parameter=foo>value</parameter>",
        &mut buf,
        &mut suppress,
        &LeakMarkers::EMPTY,
    );
    assert_eq!(out, "<parameter=foo>value</parameter>");
    assert!(buf.is_empty(), "no markers → no tail buffering");
    assert!(!suppress);
}

#[test]
fn sanitizer_suppresses_for_qwen3_markers() {
    let markers = Qwen3CoderParser.leak_markers();
    let mut buf = String::new();
    let mut suppress = false;
    let out = sanitize_content_chunk(
        "prefix<parameter=filePath>/tmp/x.txt</parameter>suffix</function>tail",
        &mut buf,
        &mut suppress,
        &markers,
    );
    assert!(out.starts_with("prefix"), "got: {out:?}");
    assert!(
        !out.contains("<parameter="),
        "orphan open must not leak: {out:?}"
    );
    assert!(
        !out.contains("/tmp/x.txt"),
        "suppressed body must not leak: {out:?}"
    );
    assert!(
        !out.contains("</function>"),
        "stray close must be stripped: {out:?}"
    );
}

#[test]
fn sanitizer_fuses_tag_across_chunks() {
    // 2026-09-26: A tag split across two calls still matches: `<param` is a marker
    // prefix, so only it stays buffered, and the text before it is emitted at once.
    let markers = Qwen3CoderParser.leak_markers();
    let mut buf = String::new();
    let mut suppress = false;
    let out1 = sanitize_content_chunk("abc<param", &mut buf, &mut suppress, &markers);
    assert!(!suppress, "partial tag must not trigger suppression");
    assert_eq!(out1, "abc", "prose before a partial tag emits immediately");
    assert_eq!(
        buf, "<param",
        "the tag prefix alone stays in the tail buffer"
    );
    let out2 = sanitize_content_chunk(
        "eter=x>body</parameter>tail",
        &mut buf,
        &mut suppress,
        &markers,
    );
    assert_eq!(out2, "tail", "only the post-close prose emits: {out2:?}");
    assert!(
        !out2.contains("body"),
        "suppressed body must not leak: {out2:?}"
    );
    assert!(
        !out2.contains("<parameter="),
        "orphan open must not leak: {out2:?}"
    );
    assert!(!suppress, "close tag exits suppression state");
}

#[test]
fn flush_empty_markers_emits_tail_verbatim() {
    let mut buf = String::from("anything");
    let mut suppress = false;
    let out = flush_content_sanitizer(&mut buf, &mut suppress, &LeakMarkers::EMPTY);
    assert_eq!(out, "anything");
    assert!(buf.is_empty());
}

#[test]
fn flush_drops_partial_tag_prefix() {
    let markers = Qwen3CoderParser.leak_markers();
    let mut buf = String::from("<par");
    let mut suppress = false;
    let out = flush_content_sanitizer(&mut buf, &mut suppress, &markers);
    assert_eq!(out, "");
}

/// 2026-09-26: The flush skips `scrub_tool_tags` when the markers declare envelopes, so
/// complete tags in a `MinimaxXmlParser` tail survive.
#[test]
fn flush_envelope_markers_skips_scrub() {
    let markers = crate::tool_parser::MinimaxXmlParser.leak_markers();
    let tail = "</invoke>\n</minimax:tool_call>";
    let mut buf = String::from(tail);
    let mut suppress = false;
    let out = flush_content_sanitizer(&mut buf, &mut suppress, &markers);
    assert_eq!(out, tail, "envelope content must survive flush verbatim");
}

#[test]
fn flush_before_tool_boundary_recovers_from_stuck_suppression() {
    // 2026-09-26: A flush between an orphan opener and later content (the chat stream
    // flushes before each tool-call event) clears suppression, so the later content is
    // emitted.
    let markers = Qwen3CoderParser.leak_markers();
    let mut buf = String::new();
    let mut suppress = false;

    let prose = sanitize_content_chunk(
        "Let me write it: <parameter=content>foo",
        &mut buf,
        &mut suppress,
        &markers,
    );
    assert_eq!(prose, "Let me write it: ", "prefix emits: {prose:?}");
    assert!(suppress, "orphan `<parameter=` enters suppression");

    let pre_tool = flush_content_sanitizer(&mut buf, &mut suppress, &markers);
    assert_eq!(pre_tool, "", "suppressed tail is correctly dropped");
    assert!(!suppress, "flush clears the suppression flag");
    assert!(buf.is_empty(), "flush clears the tail buffer");

    let post_tool = sanitize_content_chunk(
        "Done — here is the result.",
        &mut buf,
        &mut suppress,
        &markers,
    );
    assert!(
        post_tool.starts_with("Done"),
        "post-tool content must reach the client: {post_tool:?}"
    );
    assert!(!suppress, "no new orphan, must stay out of suppression");
}

/// 2026-09-26: With no marker match, only a suffix that is a byte prefix of some marker
/// is held back, so a chunk with no such suffix is emitted whole on the first call.
#[test]
fn marker_incompatible_first_chunk_emits_immediately() {
    let markers = Qwen3CoderParser.leak_markers();
    let mut buf = String::new();
    let mut sup = false;
    for text in ["An", "The", "A", "Certainly, here is"] {
        buf.clear();
        let out = sanitize_content_chunk(text, &mut buf, &mut sup, &markers);
        assert_eq!(out, *text, "marker-incompatible chunk was withheld");
        assert!(
            buf.is_empty(),
            "nothing marker-compatible to hold for {text:?}"
        );
    }
}

#[test]
fn marker_prefix_suffix_is_still_held_for_fusion() {
    let markers = Qwen3CoderParser.leak_markers();
    let mut buf = String::new();
    let mut sup = false;
    let out = sanitize_content_chunk("done.<tool_c", &mut buf, &mut sup, &markers);
    assert_eq!(out, "done.");
    assert_eq!(buf, "<tool_c");
    let out2 = sanitize_content_chunk("all>leaked</tool_call>after", &mut buf, &mut sup, &markers);
    assert!(
        !out2.contains("leaked") && out2.ends_with("after"),
        "straddled marker fusion regressed: {out2:?}"
    );
}
