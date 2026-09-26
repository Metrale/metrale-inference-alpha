// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(unused_imports, dead_code)]

//! 2026-09-26: Tool-parser tests for parameters whose real key leaked into
//! the value (`<parameter=parameter>filePath>…`), which `salvage_echoed_param`
//! re-splits for `backfill_required_params` and for live streaming, and for a
//! close tag that lost its `>` (`</parameter<parameter=`).
//!
//! Owner: server (tool parser) tests.
//! Invariants: none beyond the types.

use super::super::*;

fn write_tool() -> ToolDefinition {
    ToolDefinition {
        tool_type: "function".into(),
        function: FunctionDefinition {
            name: "write".into(),
            description: None,
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "filePath": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["content", "filePath"]
            })),
        },
    }
}

#[test]
fn salvage_helper_resplits_echoed_key() {
    let tools = vec![write_tool()];
    // 2026-09-26: The key slot holds `parameter`; the real key opens the value.
    let got = salvage_echoed_param(&tools, "write", "parameter", "filePath>\n/tmp/x/session.rs");
    assert_eq!(
        got,
        Some(("filePath".into(), "/tmp/x/session.rs".into())),
        "echoed `parameter` key with `filePath>` value prefix must re-split"
    );
    // 2026-09-26: The key slot holds the function name.
    let got = salvage_echoed_param(&tools, "write", "write", "content>use axum::Json;");
    assert_eq!(got, Some(("content".into(), "use axum::Json;".into())));
}

#[test]
fn salvage_helper_never_rewrites_legit_schema_keys() {
    let tools = vec![write_tool()];
    // 2026-09-26: The key is a schema property, so a value that starts with
    // `filePath>` is kept as it is.
    let got = salvage_echoed_param(&tools, "write", "content", "filePath> is the arg name");
    assert_eq!(got, None, "legit schema key must never be salvaged");
    let got = salvage_echoed_param(&tools, "write", "bogus", "just some text");
    assert_eq!(got, None);
    let got = salvage_echoed_param(&tools, "nosuch", "parameter", "filePath>/tmp/a");
    assert_eq!(got, None);
}

#[test]
fn buffered_pipeline_recovers_live_shape_1() {
    // 2026-09-26: The path arrives under `parameter`; the salvage runs before
    // the backfill, so `filePath` gets the path, not `""`.
    let input = "<tool_call>\n\
        <function=write>\n\
        <parameter=content>\nuse axum::Json;\n</parameter>\n\
        <parameter=parameter>filePath>\n/tmp/x/session.rs\n</parameter>\n\
        </function>\n\
        </tool_call>";
    let (_c, mut calls) = parse_tool_calls(input);
    assert_eq!(calls.len(), 1);
    let tool = write_tool();
    backfill_required_params(&mut calls, std::slice::from_ref(&tool));
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(
        args["filePath"], "/tmp/x/session.rs",
        "path must be salvaged from the echoed key"
    );
    assert_eq!(args["content"], "use axum::Json;");
    assert!(
        args.get("parameter").is_none(),
        "echoed key must be consumed by the salvage, not left behind"
    );
}

#[test]
fn buffered_pipeline_recovers_live_shape_2() {
    // 2026-09-26: Both keys leaked into their values.
    let input = "<tool_call>\n\
        <function=write>\n\
        <parameter=write>content>use sqlx::SqlitePool;\n</parameter>\n\
        <parameter=parameter>filePath>\n/tmp/x/session.rs\n</parameter>\n\
        </function>\n\
        </tool_call>";
    let (_c, mut calls) = parse_tool_calls(input);
    assert_eq!(calls.len(), 1);
    let tool = write_tool();
    backfill_required_params(&mut calls, std::slice::from_ref(&tool));
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["content"], "use sqlx::SqlitePool;");
    assert_eq!(args["filePath"], "/tmp/x/session.rs");
    assert!(args.get("write").is_none());
    assert!(args.get("parameter").is_none());
}

#[test]
fn buffered_salvage_does_not_clobber_populated_target() {
    // 2026-09-26: A non-empty real key is not overwritten by the echo.
    let input = "<tool_call>\n\
        <function=write>\n\
        <parameter=filePath>\n/tmp/real.rs\n</parameter>\n\
        <parameter=content>\nhello\n</parameter>\n\
        <parameter=parameter>filePath>\n/tmp/echoed.rs\n</parameter>\n\
        </function>\n\
        </tool_call>";
    let (_c, mut calls) = parse_tool_calls(input);
    let tool = write_tool();
    backfill_required_params(&mut calls, std::slice::from_ref(&tool));
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(
        args["filePath"], "/tmp/real.rs",
        "populated key must win over the echo"
    );
}

#[test]
fn streaming_live_fragments_recover_echoed_key() {
    // 2026-09-26: The same shape through live argument streaming (`coerce_kv`).
    let mut det = StreamingToolDetector::new_with_tools(vec![write_tool()]);
    let full = "<tool_call>\n<function=write>\n\
                <parameter=parameter>filePath>\n/tmp/x/session.rs\n</parameter>\n\
                <parameter=content>\nuse axum::Json;\n</parameter>\n\
                </function>\n</tool_call>";
    let mut outputs = Vec::new();
    for chunk in full.as_bytes().chunks(7) {
        outputs.extend(det.process(std::str::from_utf8(chunk).unwrap()));
    }
    outputs.extend(det.flush());
    let mut frags = String::new();
    for o in &outputs {
        if let DetectorOutput::ToolCallArgsFragment { fragment, .. } = o {
            frags.push_str(fragment);
        }
    }
    let args: serde_json::Value =
        serde_json::from_str(&frags).unwrap_or_else(|e| panic!("bad args json {frags:?}: {e}"));
    assert_eq!(args["filePath"], "/tmp/x/session.rs");
    assert_eq!(args["content"], "use axum::Json;");
    assert!(args.get("parameter").is_none());
}

#[test]
fn buffered_recovers_garbled_close_reopen() {
    // 2026-09-26: The close lost its `>`. `parse_qwen3_coder_call` ends the
    // value at the next `<parameter=` and strips the `</parameter` tail.
    let input = "<tool_call>\n\
        <function=write>\n\
        <parameter=content>use axum::Json;\n</parameter<parameter=filePath>\n/tmp/x/+page.svelte\n</parameter>\n\
        </function>\n\
        </tool_call>";
    let (_c, mut calls) = parse_tool_calls(input);
    assert_eq!(calls.len(), 1);
    let tool = write_tool();
    backfill_required_params(&mut calls, std::slice::from_ref(&tool));
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(
        args["content"], "use axum::Json;",
        "orphan </parameter tail must be stripped"
    );
    assert_eq!(args["filePath"], "/tmp/x/+page.svelte");
}

#[test]
fn streaming_recovers_garbled_close_reopen() {
    let mut det = StreamingToolDetector::new_with_tools(vec![write_tool()]);
    let full = "<tool_call>\n<function=write>\n\
                <parameter=content>use axum::Json;\n</parameter<parameter=filePath>\n/tmp/x/+page.svelte\n</parameter>\n\
                </function>\n</tool_call>";
    let mut outputs = Vec::new();
    for chunk in full.as_bytes().chunks(9) {
        outputs.extend(det.process(std::str::from_utf8(chunk).unwrap()));
    }
    outputs.extend(det.flush());
    let mut frags = String::new();
    for o in &outputs {
        if let DetectorOutput::ToolCallArgsFragment { fragment, .. } = o {
            frags.push_str(fragment);
        }
    }
    let args: serde_json::Value =
        serde_json::from_str(&frags).unwrap_or_else(|e| panic!("bad args json {frags:?}: {e}"));
    assert_eq!(
        args["filePath"], "/tmp/x/+page.svelte",
        "reopened param must be recovered live"
    );
    assert!(
        !args["content"].as_str().unwrap().contains("</parameter"),
        "garble must not leak into content"
    );
}

#[test]
fn legit_close_prefix_content_not_split() {
    // 2026-09-26: A `</parameter` not followed by `<parameter=` is content;
    // the value ends at the real `</parameter>`.
    let input = "<tool_call>\n\
        <function=write>\n\
        <parameter=filePath>/tmp/doc.md</parameter>\n\
        <parameter=content>the close tag is </parameter followed by text</parameter>\n\
        </function>\n\
        </tool_call>";
    let (_c, mut calls) = parse_tool_calls(input);
    let tool = write_tool();
    backfill_required_params(&mut calls, std::slice::from_ref(&tool));
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(
        args["content"],
        "the close tag is </parameter followed by text"
    );
}
