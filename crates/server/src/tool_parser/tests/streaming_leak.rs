// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(unused_imports, dead_code)]

//! 2026-09-26: Streaming leak-suppression tests through detector ->
//! `sanitize_content_chunk` -> `flush_content_sanitizer`: stray close tags
//! after a call (`</_call>`, a doubled `</tool_call>`) and raw call markup
//! must not reach content. Also `scrub_tool_tags` on its own.
//!
//! Owner: server (tool parser) tests.
//! Invariants: none beyond the types.

use super::super::*;
use super::streaming_frag::write_and_bash_tools;

/// 2026-09-26: Feed each chunk to a detector and every `Content` payload
/// through `sanitize_content_chunk` with the qwen3_coder leak markers, then
/// `flush` the detector and the sanitizer. Returns each call's
/// `(name, args_json)`, joined from its events, and all sanitised content.
fn stream_through_pipeline(chunks: &[&str]) -> (Vec<(String, String)>, String) {
    use crate::tool_parser::LeakMarkers;
    let markers: LeakMarkers = Qwen3CoderParser.leak_markers();
    let mut det = StreamingToolDetector::new_with_tools(write_and_bash_tools());

    let mut tag_scan_buf = String::new();
    let mut suppressing = false;
    let mut inside_env = false;

    let mut tc_acc: std::collections::BTreeMap<usize, (String, String)> =
        std::collections::BTreeMap::new();
    let mut content = String::new();

    let run = |out: &[DetectorOutput],
               tc_acc: &mut std::collections::BTreeMap<usize, (String, String)>,
               content: &mut String,
               tag_scan_buf: &mut String,
               suppressing: &mut bool,
               inside_env: &mut bool| {
        for o in out {
            match o {
                DetectorOutput::Content(text) => {
                    // 2026-09-26: Production also flushes the sanitizer
                    // before each tool event (`tool_handlers.rs`); the leaks
                    // tested here are content only.
                    let s = crate::api::sanitizer::sanitize_content_chunk(
                        text,
                        tag_scan_buf,
                        suppressing,
                        inside_env,
                        &markers,
                    );
                    content.push_str(&s);
                }
                DetectorOutput::ToolCallStart { name, idx, .. } => {
                    tc_acc.insert(*idx, (name.clone(), String::new()));
                }
                DetectorOutput::ToolCallArgsFragment { fragment, idx } => {
                    tc_acc.entry(*idx).or_default().1.push_str(fragment);
                }
                DetectorOutput::ToolCallDelta { args, idx } => {
                    tc_acc.entry(*idx).or_default().1.push_str(args);
                }
                DetectorOutput::ToolCall(tc, idx) => {
                    tc_acc.insert(
                        *idx,
                        (tc.function.name.clone(), tc.function.arguments.clone()),
                    );
                }
                DetectorOutput::ToolCallEnd { .. } => {}
            }
        }
    };

    for c in chunks {
        let out = det.process(c);
        run(
            &out,
            &mut tc_acc,
            &mut content,
            &mut tag_scan_buf,
            &mut suppressing,
            &mut inside_env,
        );
    }
    let out = det.flush();
    run(
        &out,
        &mut tc_acc,
        &mut content,
        &mut tag_scan_buf,
        &mut suppressing,
        &mut inside_env,
    );
    // 2026-09-26: The stream-end flush that `handle_done` runs.
    content.push_str(&crate::api::stream_guards::flush_content_sanitizer(
        &mut tag_scan_buf,
        &mut suppressing,
        &markers,
    ));

    (tc_acc.into_values().collect(), content)
}

/// 2026-09-26: A stray `</_call>` after the real `</tool_call>`, fed in
/// pieces, does not reach content, and the call still parses. No suppression
/// is active at that point, so `sanitize_content_chunk` drops it through its
/// orphan-close arm.
#[test]
fn qwen3_coder_spurious_trailing_close_underscore_call_not_leaked() {
    let chunks = [
        "<tool_call>",
        "<function=Bash>",
        "<parameter=command>",
        "ls -la",
        "</parameter>",
        "</function>",
        "</tool_call>",
        "\n\n",
        "</",
        "_",
        "call",
        ">",
    ];
    let (calls, content) = stream_through_pipeline(&chunks);
    assert_eq!(calls.len(), 1, "exactly one tool call: {calls:?}");
    assert_eq!(calls[0].0, "Bash", "tool name preserved");
    let args: serde_json::Value = serde_json::from_str(&calls[0].1)
        .unwrap_or_else(|e| panic!("args not valid JSON: {e}; raw={:?}", calls[0].1));
    assert_eq!(args["command"], "ls -la", "args preserved");
    assert!(
        !content.contains("</_call>") && !content.contains("</tool_call>"),
        "spurious trailing close leaked into content: {content:?}"
    );
}

/// 2026-09-26: A doubled `</tool_call>` does not reach content.
#[test]
fn qwen3_coder_doubled_tool_call_close_not_leaked() {
    let chunks = [
        "<tool_call>",
        "<function=Bash>",
        "<parameter=command>",
        "echo hi",
        "</parameter>",
        "</function>",
        "</tool_call>",
        "</tool_call>",
    ];
    let (calls, content) = stream_through_pipeline(&chunks);
    assert_eq!(calls.len(), 1, "exactly one tool call: {calls:?}");
    assert_eq!(calls[0].0, "Bash", "tool name preserved");
    let args: serde_json::Value = serde_json::from_str(&calls[0].1)
        .unwrap_or_else(|e| panic!("args not valid JSON: {e}; raw={:?}", calls[0].1));
    assert_eq!(args["command"], "echo hi", "args preserved");
    assert!(
        !content.contains("</tool_call>"),
        "spurious doubled close leaked into content: {content:?}"
    );
}

/// 2026-09-26: `flush_content_sanitizer` drops a close tag still held in the
/// tail at stream end, whole or missing its `>`, also after whitespace, and
/// keeps the text before it.
#[test]
fn flush_content_sanitizer_drops_held_back_trailing_close() {
    use crate::tool_parser::LeakMarkers;
    let markers: LeakMarkers = Qwen3CoderParser.leak_markers();
    let cases = [
        ("\n\n</_call>", "\n\n"),
        ("\n\n</tool_call>", "\n\n"),
        ("\n\n</_call", "\n\n"),
        ("ok</tool_call>", "ok"),
        ("plain text", "plain text"),
    ];
    for (tail, want) in cases {
        let mut buf = String::from(tail);
        let mut suppress = false;
        let out =
            crate::api::stream_guards::flush_content_sanitizer(&mut buf, &mut suppress, &markers);
        assert_eq!(
            out, want,
            "flush of {tail:?} should yield {want:?}, got {out:?}"
        );
        assert!(
            !out.contains("</_call") && !out.contains("</tool_call"),
            "close leaked from flush of {tail:?}: {out:?}"
        );
    }
}

/// 2026-09-26: The stream ends before the stray close's `>`, so the partial
/// `</_call` is still in the sanitizer tail and `flush_content_sanitizer`
/// drops it. The call still parses.
#[test]
fn qwen3_coder_spurious_trailing_close_reaches_flush_not_leaked() {
    let chunks = [
        "<tool_call>",
        "<function=Bash>",
        "<parameter=command>",
        "ls -la",
        "</parameter>",
        "</function>",
        "</tool_call>",
        "\n\n",
        "</",
        "_",
        "call",
    ];
    let (calls, content) = stream_through_pipeline(&chunks);
    assert_eq!(calls.len(), 1, "exactly one tool call: {calls:?}");
    assert_eq!(calls[0].0, "Bash", "tool name preserved");
    let args: serde_json::Value = serde_json::from_str(&calls[0].1)
        .unwrap_or_else(|e| panic!("args not valid JSON: {e}; raw={:?}", calls[0].1));
    assert_eq!(args["command"], "ls -la", "args preserved");
    assert!(
        !content.contains("</_call") && !content.contains("</tool_call"),
        "spurious trailing close leaked into content: {content:?}"
    );
}

/// 2026-09-26: `scrub_tool_tags` removes every complete tool-call tag,
/// `</_call>` included, from several raw call blocks, and keeps the argument
/// values.
#[test]
fn scrub_tool_tags_strips_runaway_markup_dump() {
    use crate::tool_parser::LeakMarkers;
    let markers: LeakMarkers = Qwen3CoderParser.leak_markers();
    let dump = "</_call>\n<tool_call>\n<function=alarms>\n<parameter=category>\ngrid_power\n\
                </parameter>\n<parameter=window>\n24h\n</parameter>\n</function>\n</tool_call></_call>\n\
                <tool_call>\n<function=fleet>\n<parameter=limit>\n200\n</parameter>\n</function>\n\
                </tool_call></_call>";
    let scrubbed = crate::api::scrub::scrub_tool_tags(dump, &markers);
    for tag in [
        "<tool_call>",
        "</tool_call>",
        "</_call>",
        "<function=",
        "</function>",
        "<parameter=",
        "</parameter>",
    ] {
        assert!(
            !scrubbed.contains(tag),
            "tag {tag:?} survived scrub: {scrubbed:?}"
        );
    }
    assert!(
        scrubbed.contains("grid_power"),
        "value dropped: {scrubbed:?}"
    );
    assert!(scrubbed.contains("200"), "value dropped: {scrubbed:?}");
}

/// 2026-09-26: `scrub_tool_tags` leaves prose alone: a bare `<` and a
/// non-tool tag such as `<div>`.
#[test]
fn scrub_tool_tags_preserves_non_tool_content() {
    use crate::tool_parser::LeakMarkers;
    let markers: LeakMarkers = Qwen3CoderParser.leak_markers();
    let prose = "if a < b and c > d, use <div> in the template";
    let scrubbed = crate::api::scrub::scrub_tool_tags(prose, &markers);
    assert_eq!(scrubbed, prose, "non-tool content altered");
}

/// 2026-09-26: A stray `</_call>`, then a whole call block in one delta: no
/// tool-call markup reaches content.
#[test]
fn qwen3_coder_runaway_content_dump_not_leaked() {
    let chunks = [
        "</_call>",
        "<tool_call><function=alarms><parameter=category>grid_power</parameter></function></tool_call></_call>",
    ];
    let (_calls, content) = stream_through_pipeline(&chunks);
    for tag in [
        "<tool_call>",
        "</tool_call>",
        "</_call>",
        "<function=",
        "<parameter=",
    ] {
        assert!(
            !content.contains(tag),
            "runaway markup {tag:?} leaked into content: {content:?}"
        );
    }
}
