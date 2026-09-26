// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Helpers for the parallel tool-call tests: detector-event readers,
//! a chunked driver and Hermes/qwen3_coder call builders.
//!
//! Owner: server (tool parser) tests.
//! Invariants: none beyond the types.

use crate::tool_parser::{DetectorOutput, StreamingToolDetector};

/// 2026-09-26: `(kind, idx)` of the indexed detector events, in emission
/// order, with `kind` one of `"start"`, `"end"`, `"call"`. Arguments are read
/// with [`args_for_idx`].
pub(super) fn indexed_trace(outputs: &[DetectorOutput]) -> Vec<(&'static str, usize)> {
    outputs
        .iter()
        .filter_map(|o| match o {
            DetectorOutput::ToolCallStart { idx, .. } => Some(("start", *idx)),
            DetectorOutput::ToolCallEnd { idx } => Some(("end", *idx)),
            DetectorOutput::ToolCall(_, idx) => Some(("call", *idx)),
            _ => None,
        })
        .collect()
}

/// 2026-09-26: The name in the first `ToolCallStart` or whole `ToolCall` for
/// `want`.
pub(super) fn name_for_idx(outputs: &[DetectorOutput], want: usize) -> Option<String> {
    outputs.iter().find_map(|o| match o {
        DetectorOutput::ToolCallStart { name, idx, .. } if *idx == want => Some(name.clone()),
        DetectorOutput::ToolCall(tc, idx) if *idx == want => Some(tc.function.name.clone()),
        _ => None,
    })
}

/// 2026-09-26: Arguments for `want`: its `ToolCallArgsFragment`s joined, else
/// those of its first `ToolCallDelta` or whole `ToolCall`.
pub(super) fn args_for_idx(outputs: &[DetectorOutput], want: usize) -> String {
    let frags: String = outputs
        .iter()
        .filter_map(|o| match o {
            DetectorOutput::ToolCallArgsFragment { fragment, idx } if *idx == want => {
                Some(fragment.as_str())
            }
            _ => None,
        })
        .collect();
    if !frags.is_empty() {
        return frags;
    }
    outputs
        .iter()
        .find_map(|o| match o {
            DetectorOutput::ToolCallDelta { args, idx } if *idx == want => Some(args.clone()),
            DetectorOutput::ToolCall(tc, idx) if *idx == want => {
                Some(tc.function.arguments.clone())
            }
            _ => None,
        })
        .unwrap_or_default()
}

pub(super) fn assert_json_eq(actual: &str, expected: &str, ctx: &str) {
    let a: serde_json::Value =
        serde_json::from_str(actual).unwrap_or_else(|e| panic!("{ctx}: bad JSON {actual:?}: {e}"));
    let b: serde_json::Value = serde_json::from_str(expected).unwrap();
    assert_eq!(a, b, "{ctx}: args mismatch");
}

/// 2026-09-26: Feed `text` in `chunk`-byte slices (the fixtures are ASCII),
/// then `flush`, so tags are split across deltas.
pub(super) fn drive_chunked(
    det: &mut StreamingToolDetector,
    text: &str,
    chunk: usize,
) -> Vec<DetectorOutput> {
    let mut outputs = Vec::new();
    let mut i = 0;
    while i < text.len() {
        let end = (i + chunk).min(text.len());
        outputs.extend(det.process(&text[i..end]));
        i = end;
    }
    outputs.extend(det.flush());
    outputs
}

pub(super) fn hermes_call(name: &str, args: &str) -> String {
    format!("<tool_call>\n{{\"name\": \"{name}\", \"arguments\": {args}}}\n</tool_call>")
}

pub(super) fn qwen3_coder_call(name: &str, params: &[(&str, &str)]) -> String {
    let mut s = format!("<tool_call>\n<function={name}>\n");
    for (k, v) in params {
        s.push_str(&format!("<parameter={k}>\n{v}\n</parameter>\n"));
    }
    s.push_str("</function>\n</tool_call>");
    s
}
