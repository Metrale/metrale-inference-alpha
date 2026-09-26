// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Content-sanitizer tests: orphan tool-call fragments are dropped and the
//! prose around them survives. Tests that use `super::harness` assert on the whole
//! stream (chunks plus the end-of-stream flush); the flush tests call
//! `flush_content_sanitizer` directly.
//!
//! Owner: server API.
//! Invariants: none beyond the types.

use super::harness::Stream;
use crate::tool_parser::{LeakMarkers, Qwen3CoderParser, ToolCallParser};

use crate::api::sanitizer::sanitize_content_chunk;
use crate::api::stream_guards::flush_content_sanitizer;

#[path = "sanitizer_chunk_tests.rs"]
mod sanitizer_tests;

fn qwen() -> LeakMarkers {
    Qwen3CoderParser.leak_markers()
}

#[test]
fn empty_markers_pass_text_through_untouched() {
    // 2026-09-26: With no markers (`LeakMarkers::EMPTY`, the `leak_markers` default)
    // `sanitize_content_chunk` returns the text as it is and buffers nothing.
    let markers = LeakMarkers::EMPTY;
    let mut s = Stream::new(&markers);
    let first = s.feed("<parameter=foo>value</parameter>");
    assert_eq!(first, "<parameter=foo>value</parameter>");
    assert!(s.buffered().is_empty(), "no markers -> no tail buffering");
    assert!(!s.suppressing());
    assert_eq!(s.finish(), "<parameter=foo>value</parameter>");
}

#[test]
fn orphan_parameter_block_is_dropped_and_prose_survives() {
    // 2026-09-26: Suppression runs from the opener to the first close tag; the stray
    // `</function>` after it is dropped too.
    let markers = qwen();
    let mut s = Stream::new(&markers);
    s.feed("prefix<parameter=filePath>/tmp/x.txt</parameter>suffix</function>tail");
    assert_eq!(s.finish(), "prefixsuffixtail");
}

#[test]
fn a_tag_split_across_chunks_still_matches() {
    // 2026-09-26: `<param` is a prefix of a marker, so it is held back and fuses with
    // `eter=x>` into one opener. The text before it is emitted at once.
    let markers = qwen();
    let mut s = Stream::new(&markers);
    let first = s.feed("abc<param");
    assert_eq!(
        first, "abc",
        "prose before a possible tag prefix emits immediately"
    );
    assert!(!s.suppressing(), "a partial tag is not yet a leak");
    s.feed("eter=x>body</parameter>tail");
    assert_eq!(s.finish(), "abctail");
}

#[test]
fn a_leak_split_across_many_tiny_chunks_never_reaches_the_client() {
    let markers = qwen();
    let mut s = Stream::new(&markers);
    s.feed_chunked("before<function=Bash>rm -rf /</function>after", 1);
    assert_eq!(s.finish(), "beforeafter");
}

#[test]
fn suppression_survives_a_chunk_boundary_inside_the_leak_body() {
    let markers = qwen();
    let mut s = Stream::new(&markers);
    s.feed("ok <tool_use>{\"name\":");
    assert!(s.suppressing(), "opener engages suppression");
    s.feed("\"x\"}");
    assert!(s.suppressing(), "still inside the leak");
    s.feed("</tool_use> done");
    assert!(!s.suppressing(), "close tag ends suppression");
    let out = s.finish();
    assert_eq!(out, "ok  done");
    assert!(!out.contains("name"), "leak body must not survive: {out:?}");
}

#[test]
fn legitimate_rust_prose_is_not_mistaken_for_a_tool_call() {
    let markers = qwen();
    let mut s = Stream::new(&markers);
    let prose = "Use `fn add(a: i32, b: i32) -> i32 { a + b }`; note `a < b` and `Vec<String>`.";
    s.feed(prose);
    assert_eq!(s.finish(), prose);
    let mut s = Stream::new(&markers);
    s.feed_chunked(prose, 1);
    assert_eq!(s.finish(), prose);
}

#[test]
fn flush_emits_a_pending_tail_when_no_markers_are_configured() {
    // 2026-09-26: With no markers nothing is buffered, but the flush still returns
    // whatever the buffer holds.
    let markers = LeakMarkers::EMPTY;
    let mut buf = String::from("anything");
    let mut suppress = false;
    let out = crate::api::stream_guards::flush_content_sanitizer(&mut buf, &mut suppress, &markers);
    assert_eq!(out, "anything");
    assert!(buf.is_empty());
}

#[test]
fn flush_drops_a_dangling_partial_tag() {
    // 2026-09-26: The stream ended on `<par` (a prefix of `<parameter=`). The flush drops
    // a lone partial tag: it starts with `<`, has no whitespace, and is shorter than the
    // longest marker.
    let markers = qwen();
    let mut buf = String::from("<par");
    let mut suppress = false;
    let out = crate::api::stream_guards::flush_content_sanitizer(&mut buf, &mut suppress, &markers);
    assert_eq!(out, "");
}

#[test]
fn flush_clears_stuck_suppression_at_a_tool_boundary() {
    // 2026-09-26: The chat stream flushes the sanitizer before each tool-call event
    // (`chat_stream/tool_handlers.rs`). The flush drops a suppressed tail and clears the
    // flag, so content after the tool call is not suppressed.
    let markers = qwen();
    let mut buf = String::new();
    let mut suppress = false;
    let mut env = false;

    let prose = crate::api::sanitizer::sanitize_content_chunk(
        "Let me write it: <parameter=content>foo",
        &mut buf,
        &mut suppress,
        &mut env,
        &markers,
    );
    assert_eq!(prose, "Let me write it: ");
    assert!(suppress, "orphan `<parameter=` enters suppression");

    let pre_tool =
        crate::api::stream_guards::flush_content_sanitizer(&mut buf, &mut suppress, &markers);
    assert_eq!(pre_tool, "", "the suppressed tail is dropped, not emitted");
    assert!(!suppress, "flush clears the suppression flag");
    assert!(buf.is_empty(), "flush clears the tail buffer");

    let mut s = Stream::new(&markers);
    s.feed("Done — here is the result.");
    assert_eq!(s.finish(), "Done — here is the result.");
}

#[test]
fn hallucinated_tool_response_wrapper_is_suppressed() {
    // 2026-09-26: `<tool_response>` is the wrapper `Qwen3CoderParser` puts around tool
    // results (the `format_tool_response` default); from the model it is an orphan
    // opener.
    let markers = qwen();
    let mut s = Stream::new(&markers);
    s.feed("I read the file. <tool_response>fn add() -> i32 { 41 }</tool_response> It returns 41.");
    let out = s.finish();
    assert_eq!(out, "I read the file.  It returns 41.");
}

#[test]
fn a_leak_that_never_closes_is_dropped_at_end_of_stream() {
    let markers = qwen();
    let mut s = Stream::new(&markers);
    s.feed("here goes <parameter=path>/etc/shadow");
    assert!(s.suppressing());
    assert_eq!(s.finish(), "here goes ");
}

#[test]
fn primary_arg_is_client_case_insensitive() {
    // 2026-09-26: Tool names match ignoring ASCII case, and the path key may be spelled
    // `file_path` or `filePath`.
    use crate::api::sanitizer::primary_arg_for_tool;
    let lower = primary_arg_for_tool("bash", r#"{"command":"cd /tmp && cargo init"}"#);
    let upper = primary_arg_for_tool("Bash", r#"{"command":"cd /tmp && cargo init"}"#);
    assert_eq!(lower, upper);
    assert_eq!(lower.as_deref(), Some("cargo init"));

    let lower = primary_arg_for_tool("write", r#"{"filePath":"/tmp/x.rs"}"#);
    let upper = primary_arg_for_tool("Write", r#"{"file_path":"/tmp/x.rs"}"#);
    assert_eq!(lower, upper);
    assert_eq!(lower.as_deref(), Some("/tmp/x.rs"));
}
