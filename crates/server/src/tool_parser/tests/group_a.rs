// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(unused_imports, dead_code)]

//! 2026-09-26: Tool-parser tests: the MiniMax XML blank-path drop, path
//! validation, Hermes parsing, and MiniMax envelopes in the streaming detector.
//!
//! Owner: server (tool parser) tests.
//! Invariants: none beyond the types.

use super::super::*;

use super::*;

#[test]
fn parse_minimax_xml_drops_write_with_empty_filepath() {
    // 2026-09-26: `parse_minimax_xml_call` returns `None` for a write-family
    // call whose path parameter is blank, so no call is emitted.
    let body = "<invoke name=\"write\">\n\
            <parameter name=\"content\">some code</parameter>\n\
            <parameter name=\"filePath\"></parameter>\n\
            </invoke>";
    let res = parse_minimax_xml_call(body, 0);
    assert!(res.is_none(), "expected drop, got {res:?}");
}

#[test]
fn parse_minimax_xml_drops_write_with_whitespace_filepath() {
    let body = "<invoke name=\"write\">\n\
            <parameter name=\"content\">x</parameter>\n\
            <parameter name=\"filePath\">   </parameter>\n\
            </invoke>";
    let res = parse_minimax_xml_call(body, 0);
    assert!(
        res.is_none(),
        "expected whitespace-only path drop, got {res:?}"
    );
}

#[test]
fn parse_minimax_xml_keeps_bash_with_empty_path_field() {
    // 2026-09-26: The blank-path drop covers write-family tools only.
    let body = "<invoke name=\"bash\">\n\
            <parameter name=\"command\">ls</parameter>\n\
            </invoke>";
    let res = parse_minimax_xml_call(body, 0);
    assert!(res.is_some(), "bash should pass even without path");
}

#[test]
fn parse_minimax_xml_keeps_write_with_valid_filepath() {
    let body = "<invoke name=\"write\">\n\
            <parameter name=\"content\">hi</parameter>\n\
            <parameter name=\"filePath\">/tmp/x.rs</parameter>\n\
            </invoke>";
    let res = parse_minimax_xml_call(body, 0);
    assert!(res.is_some());
    let args: serde_json::Value = serde_json::from_str(&res.unwrap().function.arguments).unwrap();
    assert_eq!(args["filePath"], "/tmp/x.rs");
}

#[test]
fn validate_rejects_write_with_empty_filepath() {
    // 2026-09-26: A write-family call with an empty path is refused.
    let tool = ToolDefinition {
        tool_type: "function".to_string(),
        function: FunctionDefinition {
            name: "write".to_string(),
            description: None,
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "filePath": {"type": "string"}
                },
                "required": ["content", "filePath"]
            })),
        },
    };
    let call = ToolCall {
        id: "x".to_string(),
        call_type: "function".to_string(),
        function: FunctionCall {
            name: "write".to_string(),
            arguments: r#"{"content":"some code","filePath":""}"#.to_string(),
        },
    };
    let res = validate_single_tool_call(&call, &[tool]);
    assert!(res.is_err(), "expected reject, got {res:?}");
    assert!(
        res.as_ref().unwrap_err().contains("non-empty"),
        "error should mention non-empty: {}",
        res.unwrap_err()
    );
}

#[test]
fn validate_allows_read_with_empty_path() {
    // 2026-09-26: Only `WRITE_FAMILY` tools refuse an empty path.
    let tool = ToolDefinition {
        tool_type: "function".to_string(),
        function: FunctionDefinition {
            name: "read".to_string(),
            description: None,
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"]
            })),
        },
    };
    let call = ToolCall {
        id: "x".to_string(),
        call_type: "function".to_string(),
        function: FunctionCall {
            name: "read".to_string(),
            arguments: r#"{"path":""}"#.to_string(),
        },
    };
    assert!(validate_single_tool_call(&call, &[tool]).is_ok());
}

#[test]
fn parse_hermes_single_call() {
    let (c, calls) =
        parse_tool_calls("<tool_call>\n{\"name\":\"f\",\"arguments\":{\"x\":1}}\n</tool_call>");
    assert!(c.is_none());
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "f");
    assert_eq!(calls[0].function.arguments, "{\"x\":1}");
}

#[test]
fn parse_hermes_with_content_and_multiple() {
    let (c, calls) = parse_tool_calls(
        "Hello\n<tool_call>\n{\"name\":\"a\",\"arguments\":{}}\n</tool_call>\n\
             <tool_call>\n{\"name\":\"b\",\"arguments\":{}}\n</tool_call>",
    );
    assert_eq!(c.unwrap(), "Hello");
    assert_eq!(calls.len(), 2);
    // 2026-09-26: Ids come from a process-wide counter (`call_` + 16 hex
    // digits), so the values depend on test order; only the shape is checked.
    assert!(calls[0].id.starts_with("call_"));
    assert!(calls[1].id.starts_with("call_"));
    assert_ne!(calls[0].id, calls[1].id);
    assert_eq!(calls[0].function.name, "a");
    assert_eq!(calls[1].function.name, "b");
}

#[test]
fn parse_no_calls() {
    let (c, calls) = parse_tool_calls("just text");
    assert_eq!(c.unwrap(), "just text");
    assert!(calls.is_empty());
}

#[test]
fn streaming_detector_hermes() {
    let mut det = StreamingToolDetector::new();
    let out = det.process("Hi <tool_call>\n{\"name\":\"f\",\"arguments\":{}}\n</tool_call>");
    assert!(out.len() >= 2);
    assert!(matches!(&out[0], DetectorOutput::Content(s) if s.contains("Hi")));
    assert!(matches!(&out[1], DetectorOutput::ToolCall(tc, 0) if tc.function.name == "f"));
    assert!(det.has_tool_calls());
}

/// 2026-09-26: The streaming detector finds a call in a
/// `<minimax:tool_call>` envelope.
#[test]
fn streaming_detector_minimax_envelope_canonical() {
    let mut det = StreamingToolDetector::new();
    let body = "<minimax:tool_call>\n\
            <invoke name=\"bash\">\n\
            <parameter name=\"command\">uname -r</parameter>\n\
            </invoke>\n\
            </minimax:tool_call>";
    let out = det.process(body);
    let names: Vec<String> = out
        .iter()
        .filter_map(|o| match o {
            DetectorOutput::ToolCall(tc, _) => Some(tc.function.name.clone()),
            DetectorOutput::ToolCallStart { name, .. } => Some(name.clone()),
            _ => None,
        })
        .collect();
    assert!(
        det.has_tool_calls(),
        "detector must report a tool_call, got names={names:?}"
    );
    assert!(
        names.iter().any(|n| n == "bash"),
        "expected bash tool, got names={names:?}"
    );
}

#[test]
fn streaming_detector_minimax_envelope_bpe_broken() {
    // 2026-09-26: Opened with `<minimax:_call>`, closed with
    // `</minimax:tool_call>`.
    let mut det = StreamingToolDetector::new();
    let body = "<minimax:_call>\n\
            <invoke name=\"bash\">\n\
            <parameter name=\"command\">mkdir -p /tmp/calc-test74/src /tmp/calc-test74/tests</parameter>\n\
            <parameter name=\"description\">Create project directories</parameter>\n\
            </invoke>\n\
            </minimax:tool_call>";
    let out = det.process(body);
    let tc = out.iter().find_map(|o| match o {
        DetectorOutput::ToolCall(tc, _) => Some(tc.clone()),
        _ => None,
    });
    assert!(
        det.has_tool_calls(),
        "broken-envelope detector must still extract a tool_call (tc found: {})",
        tc.is_some()
    );
    let tc = tc.expect("ToolCall output expected");
    assert_eq!(tc.function.name, "bash");
    let args: serde_json::Value =
        serde_json::from_str(&tc.function.arguments).expect("args must be JSON");
    assert!(
        args["command"]
            .as_str()
            .unwrap()
            .contains("/tmp/calc-test74"),
        "command arg lost: {args}"
    );
}

/// 2026-09-26: `<minimax:_call>` split across deltas: `safe_emit_len` holds
/// back the `<minimax` prefix, so the opener is still found and no envelope
/// text reaches content.
#[test]
fn streaming_detector_minimax_bpe_broken_split_chunks() {
    let mut det = StreamingToolDetector::new();
    let chunks = [
        "<minimax",
        ":_call>",
        "\n<invoke name=\"bash\">\n",
        "<parameter name=\"command\">",
        "mkdir -p /tmp/calc-test74/src",
        "</parameter>\n</invoke>\n",
        "</minimax:tool_call>",
    ];
    let mut all_outputs = Vec::new();
    for c in chunks {
        all_outputs.extend(det.process(c));
    }
    let tcs: Vec<_> = all_outputs
        .iter()
        .filter_map(|o| match o {
            DetectorOutput::ToolCall(tc, _) => Some(tc.clone()),
            _ => None,
        })
        .collect();
    assert!(
        det.has_tool_calls(),
        "envelope missed under chunked arrival"
    );
    assert_eq!(tcs.len(), 1, "expected 1 ToolCall");
    assert_eq!(tcs[0].function.name, "bash");
    for o in &all_outputs {
        if let DetectorOutput::Content(s) = o {
            assert!(
                !s.contains("<minimax:"),
                "envelope text leaked to content: {s:?}"
            );
        }
    }
}

/// 2026-09-26: Two `<invoke>` blocks in one envelope give two `ToolCall`s:
/// the close path parses every block (`parse_minimax_xml_calls_all`), where
/// `parse_one_call` would return only the first.
#[test]
fn streaming_detector_minimax_envelope_bpe_broken_two_invokes() {
    let mut det = StreamingToolDetector::new();
    let body = "<minimax:_call>\n\
            <invoke name=\"bash\">\n\
            <parameter name=\"command\">mkdir -p /tmp/calc-test74/src</parameter>\n\
            <parameter name=\"description\">Create project directory structure</parameter>\n\
            </invoke>\n\
            <invoke name=\"bash\">\n\
            <parameter name=\"command\">mkdir -p /tmp/calc-test74/tests</parameter>\n\
            <parameter name=\"description\">Create tests directory</parameter>\n\
            </invoke>\n\
            </minimax:tool_call>";
    let out = det.process(body);
    let tcs: Vec<_> = out
        .iter()
        .filter_map(|o| match o {
            DetectorOutput::ToolCall(tc, _) => Some(tc.clone()),
            _ => None,
        })
        .collect();
    assert!(det.has_tool_calls(), "no tool_calls extracted at all");
    assert_eq!(
        tcs.len(),
        2,
        "expected 2 ToolCall outputs, got {}",
        tcs.len()
    );
    for (i, tc) in tcs.iter().enumerate() {
        assert_eq!(tc.function.name, "bash", "call {i} wrong name");
        let args: serde_json::Value =
            serde_json::from_str(&tc.function.arguments).expect("args must be JSON");
        let cmd = args["command"].as_str().unwrap_or("");
        if i == 0 {
            assert!(
                cmd.contains("src"),
                "first call should be src dir, got {cmd}"
            );
        } else {
            assert!(
                cmd.contains("tests"),
                "second call should be tests dir, got {cmd}"
            );
        }
    }
}
