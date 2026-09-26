// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(unused_imports, dead_code)]

//! 2026-09-26: Tool-parser test: `flush` of an unclosed call whose
//! `ToolCallStart` already went out.
//!
//! Owner: server (tool parser) tests.
//! Invariants: none beyond the types.

use super::super::*;

// 2026-09-26: When the stream ends inside `<tool_call>` after `ToolCallStart`
// was emitted, `flush` finishes the call under that header (arguments and
// `ToolCallEnd`) and emits no whole `ToolCall`, whose new id would open a
// second call at the same index.

#[test]
fn flush_after_incremental_start_does_not_double_emit() {
    let mut det = StreamingToolDetector::new();

    let pre_flush =
        det.process("<tool_call>\n<function=web_search>\n<parameter=query>meeting</parameter>\n");
    let starts: Vec<_> = pre_flush
        .iter()
        .filter(|o| matches!(o, DetectorOutput::ToolCallStart { .. }))
        .collect();
    assert_eq!(starts.len(), 1, "expected 1 incremental ToolCallStart");

    let flushed = det.flush();
    let starts_after: Vec<_> = flushed
        .iter()
        .filter(|o| matches!(o, DetectorOutput::ToolCallStart { .. }))
        .collect();
    let complete_after: Vec<_> = flushed
        .iter()
        .filter(|o| matches!(o, DetectorOutput::ToolCall(_, _)))
        .collect();
    assert!(
        starts_after.is_empty(),
        "flush() must not re-emit ToolCallStart when incremental start already happened"
    );
    assert!(
        complete_after.is_empty(),
        "flush() must not emit ToolCall (which carries a fresh id) — emit Delta+End instead"
    );

    // 2026-09-26: Live streaming sends the arguments as `ToolCallArgsFragment`s
    // across `process` and `flush`; with `buffer_args` they come as one
    // `ToolCallDelta`. Either shape is accepted.
    let has_args = flushed.iter().chain(pre_flush.iter()).any(|o| {
        matches!(
            o,
            DetectorOutput::ToolCallDelta { .. } | DetectorOutput::ToolCallArgsFragment { .. }
        )
    });
    let has_end = flushed
        .iter()
        .any(|o| matches!(o, DetectorOutput::ToolCallEnd { .. }));
    assert!(has_args, "must emit the args (Delta or Fragment)");
    assert!(has_end, "flush() must emit ToolCallEnd to close the call");

    let args: String = flushed
        .iter()
        .chain(pre_flush.iter())
        .filter_map(|o| match o {
            DetectorOutput::ToolCallDelta { args, .. } => Some(args.clone()),
            DetectorOutput::ToolCallArgsFragment { fragment, .. } => Some(fragment.clone()),
            _ => None,
        })
        .collect();
    assert!(
        args.contains("meeting"),
        "args should contain 'meeting'; got: {args}"
    );
}
