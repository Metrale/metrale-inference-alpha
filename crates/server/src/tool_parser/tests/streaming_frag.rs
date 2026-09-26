// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(unused_imports, dead_code)]

//! 2026-09-26: Streaming-detector tests for Qwen3-Coder calls whose deltas
//! split tags, values and openers at arbitrary bytes, or bundle several parts
//! at once: the arguments must come out whole, and live fragments must be
//! coerced and closed.
//!
//! Owner: server (tool parser) tests.
//! Invariants: none beyond the types.

use super::super::*;

/// 2026-09-26: Every `ToolCallArgsFragment` payload, for any idx, joined in
/// emission order. Live streaming (`!buffer_args`) emits these instead of one
/// `ToolCallDelta`; joined, they are the complete JSON arguments.
pub(super) fn collect_fragments(outputs: &[DetectorOutput]) -> String {
    let mut s = String::new();
    for o in outputs {
        if let DetectorOutput::ToolCallArgsFragment { fragment, .. } = o {
            s.push_str(fragment);
        }
    }
    s
}

/// 2026-09-26: The joined fragments, else the first `ToolCallDelta`'s
/// arguments, so a test accepts the live and the buffered shape.
pub(super) fn args_from_outputs(outputs: &[DetectorOutput]) -> String {
    let frags = collect_fragments(outputs);
    if !frags.is_empty() {
        return frags;
    }
    for o in outputs {
        if let DetectorOutput::ToolCallDelta { args, .. } = o {
            return args.clone();
        }
    }
    panic!("no args emitted (neither fragments nor delta)");
}

#[test]
fn qwen3_coder_streaming_fragmented_at_path_boundary() {
    // 2026-09-26: A value split across deltas: live streaming emits a
    // parameter only once its `</parameter>` arrives, so the path is whole.
    let mut det = StreamingToolDetector::new();
    let chunks = [
        "<tool_call>",
        "<function=Read>",
        "<parameter=file_path>",
        "/home/nolo",
        "gik/test.rs",
        "</parameter>",
        "</function>",
        "</tool_call>",
    ];
    let mut outputs = Vec::new();
    for c in chunks {
        outputs.extend(det.process(c));
    }
    let args: serde_json::Value = serde_json::from_str(&args_from_outputs(&outputs)).unwrap();
    assert_eq!(args["file_path"], "/home/nologik/test.rs");
}

#[test]
fn qwen3_coder_streaming_fragmented_at_xml_opener() {
    // 2026-09-26: A `<parameter=` opener split across deltas. Inside a call
    // the detector buffers the body and emits only complete parameters.
    let mut det = StreamingToolDetector::new();
    let chunks = [
        "<tool_call><function=Read>",
        "<param",
        "eter=file_path>",
        "/etc/hosts</parameter></function></tool_call>",
    ];
    let mut outputs = Vec::new();
    for c in chunks {
        outputs.extend(det.process(c));
    }
    let args: serde_json::Value = serde_json::from_str(&args_from_outputs(&outputs)).unwrap();
    assert_eq!(args["file_path"], "/etc/hosts");
}

#[test]
fn qwen3_coder_streaming_same_name_tool_calls_no_collision() {
    // 2026-09-26: Calls are indexed by `call_counter`, so two calls with the
    // same name get indices 0 and 1. Fed in one delta, each close is found
    // before a `ToolCallStart` went out, so both come as whole `ToolCall`s.
    let mut det = StreamingToolDetector::new();
    let input = "<tool_call>\
                <function=Read>\
                <parameter=file_path>/a.rs</parameter>\
                </function>\
                </tool_call>\
                <tool_call>\
                <function=Read>\
                <parameter=file_path>/b.rs</parameter>\
                </function>\
                </tool_call>";
    let outputs = det.process(input);
    let calls: Vec<_> = outputs
        .iter()
        .filter_map(|o| match o {
            DetectorOutput::ToolCall(tc, idx) => Some((
                *idx,
                tc.function.name.clone(),
                tc.function.arguments.clone(),
            )),
            _ => None,
        })
        .collect();
    assert_eq!(
        calls.len(),
        2,
        "two ToolCall events for two same-name calls"
    );
    assert_eq!(calls[0].0, 0);
    assert_eq!(calls[1].0, 1);
    assert_eq!(calls[0].1, "Read");
    assert_eq!(calls[1].1, "Read");
    let args0: serde_json::Value = serde_json::from_str(&calls[0].2).unwrap();
    let args1: serde_json::Value = serde_json::from_str(&calls[1].2).unwrap();
    assert_eq!(args0["file_path"], "/a.rs");
    assert_eq!(args1["file_path"], "/b.rs");
}

#[test]
fn qwen3_coder_streaming_close_with_final_value_in_same_chunk() {
    // 2026-09-26: The last value and every close tag arrive in one delta;
    // the value is still in the arguments.
    let mut det = StreamingToolDetector::new();
    let chunks = [
        "<tool_call><function=Write>",
        "<parameter=path>/tmp/x</parameter>",
        "<parameter=content>hello world</parameter></function></tool_call>",
    ];
    let mut outputs = Vec::new();
    for c in chunks {
        outputs.extend(det.process(c));
    }
    let args: serde_json::Value = serde_json::from_str(&args_from_outputs(&outputs)).unwrap();
    assert_eq!(args["path"], "/tmp/x");
    assert_eq!(
        args["content"], "hello world",
        "final-param-with-close burst must preserve the value"
    );
}

/// 2026-09-26: A tool definition with the given name and parameters schema.
pub(super) fn tool_def(name: &str, params: serde_json::Value) -> ToolDefinition {
    ToolDefinition {
        tool_type: "function".into(),
        function: FunctionDefinition {
            name: name.into(),
            description: None,
            parameters: Some(params),
        },
    }
}

pub(super) fn write_and_bash_tools() -> Vec<ToolDefinition> {
    vec![
        tool_def(
            "Write",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["file_path", "content"]
            }),
        ),
        tool_def(
            "Bash",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "integer"}
                },
                "required": ["command"]
            }),
        ),
    ]
}

#[test]
fn qwen3_coder_live_streams_multiple_fragments_before_end() {
    // 2026-09-26: Fed in 5-byte deltas with tool schemas and live streaming,
    // a call emits `ToolCallStart`, then several `ToolCallArgsFragment`s, all
    // before `ToolCallEnd`; the joined fragments equal the expected JSON value.
    let mut det = StreamingToolDetector::new_with_tools(write_and_bash_tools());
    let full = "<tool_call>\n<function=Write>\n\
                <parameter=file_path>\n/tmp/x.rs\n</parameter>\n\
                <parameter=content>\nhello\n</parameter>\n\
                </function>\n</tool_call>";
    let bytes = full.as_bytes();
    let mut outputs = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        // 2026-09-26: The fixture is ASCII, so every 5-byte step is a char
        // boundary.
        let end = (i + 5).min(bytes.len());
        outputs.extend(det.process(&full[i..end]));
        i = end;
    }
    outputs.extend(det.flush());

    let start_count = outputs
        .iter()
        .filter(|o| matches!(o, DetectorOutput::ToolCallStart { .. }))
        .count();
    assert!(start_count >= 1, "expected at least one ToolCallStart");

    let end_pos = outputs
        .iter()
        .position(|o| matches!(o, DetectorOutput::ToolCallEnd { .. }))
        .expect("ToolCallEnd emitted");
    let frag_positions: Vec<usize> = outputs
        .iter()
        .enumerate()
        .filter(|(_, o)| matches!(o, DetectorOutput::ToolCallArgsFragment { .. }))
        .map(|(i, _)| i)
        .collect();
    assert!(
        frag_positions.len() >= 2,
        "expected MULTIPLE ToolCallArgsFragment events, got {}",
        frag_positions.len()
    );
    assert!(
        frag_positions.iter().all(|&p| p < end_pos),
        "all fragments must be emitted BEFORE ToolCallEnd"
    );

    let args: serde_json::Value = serde_json::from_str(&collect_fragments(&outputs)).unwrap();
    let expected = serde_json::json!({"file_path": "/tmp/x.rs", "content": "hello"});
    assert_eq!(args, expected);
}

#[test]
fn qwen3_coder_live_flush_path_emits_closing_brace() {
    // 2026-09-26: No `</tool_call>` arrives, so `flush` emits the rest of the
    // arguments, closing `}` included. The joined fragments must be valid JSON.
    let mut det = StreamingToolDetector::new_with_tools(write_and_bash_tools());
    let full = "<tool_call>\n<function=Bash>\n\
                <parameter=command>\nls -lR /etc\n</parameter>\n\
                <parameter=timeout>\n30\n</parameter>\n\
                </function>\n";
    let bytes = full.as_bytes();
    let mut outputs = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let end = (i + 5).min(bytes.len());
        outputs.extend(det.process(&full[i..end]));
        i = end;
    }
    outputs.extend(det.flush());

    let joined = collect_fragments(&outputs);
    let args: serde_json::Value = serde_json::from_str(&joined)
        .unwrap_or_else(|e| panic!("streamed args not valid JSON: {e}; joined={joined:?}"));
    let expected = serde_json::json!({"command": "ls -lR /etc", "timeout": 30});
    assert_eq!(
        args, expected,
        "flush-path streamed args must match + coerce"
    );
}

#[test]
fn qwen3_coder_live_coerces_integer_param() {
    // 2026-09-26: `timeout` is typed `integer`, so its live fragment carries
    // the number `30`: `coerce_kv` coerces through `coerce_all`.
    let mut det = StreamingToolDetector::new_with_tools(write_and_bash_tools());
    let chunks = [
        "<tool_call>",
        "<function=Bash>",
        "<parameter=command>",
        "ls /tmp",
        "</parameter>",
        "<parameter=timeout>",
        "30",
        "</parameter>",
        "</function>",
        "</tool_call>",
    ];
    let mut outputs = Vec::new();
    for c in chunks {
        outputs.extend(det.process(c));
    }
    let args: serde_json::Value = serde_json::from_str(&collect_fragments(&outputs)).unwrap();
    assert_eq!(args["command"], "ls /tmp");
    assert_eq!(
        args["timeout"],
        serde_json::json!(30),
        "integer schema must coerce \"30\" → 30 (number, not string)"
    );
    assert!(args["timeout"].is_number(), "timeout must be a JSON number");
}
