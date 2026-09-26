// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(unused_imports, dead_code)]

//! 2026-09-26: Tool-parser tests: Mistral calls, balanced-JSON scanning,
//! Qwen3-Coder parameter bounds, the `<parameter=NAME>` salvage pass, the
//! qwen3_coder system prompt, leak markers and fuzzy tool-name repair.
//!
//! Owner: server (tool parser) tests.
//! Invariants: none beyond the types.

use super::super::*;

#[test]
fn parse_mistral_nested_json_args() {
    let input =
        r#"[TOOL_CALLS]search[ARGS]{"filters":{"year":2025,"tags":["rust","cuda"]},"limit":10}"#;
    let (_, calls) = parse_tool_calls(input);
    assert_eq!(calls.len(), 1);
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["filters"]["year"], 2025);
    assert_eq!(args["filters"]["tags"][1], "cuda");
    assert_eq!(args["limit"], 10);
}

#[test]
fn parse_mistral_with_whitespace() {
    let input = "[TOOL_CALLS] get_weather [ARGS]\n{\n  \"location\": \"Paris\"\n}";
    let (_, calls) = parse_tool_calls(input);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "get_weather");
}

#[test]
fn streaming_detector_mistral_single() {
    let mut det = StreamingToolDetector::new();
    let out = det.process("Let me check.[TOOL_CALLS]get_weather[ARGS]{\"location\":\"Paris\"}");
    assert!(out.len() >= 4);
    assert!(matches!(&out[0], DetectorOutput::Content(s) if s.contains("Let me check")));
    assert!(matches!(&out[1], DetectorOutput::ToolCallStart { name, .. } if name == "get_weather"));
    assert!(matches!(&out[2], DetectorOutput::ToolCallDelta { .. }));
    assert!(matches!(&out[3], DetectorOutput::ToolCallEnd { .. }));
    assert!(det.has_tool_calls());
}

#[test]
fn streaming_detector_mistral_split_tokens() {
    let mut det = StreamingToolDetector::new();
    let chunks = [
        "[TOOL_",
        "CALLS]",
        "get_weather",
        "[ARGS]",
        "{\"city\":",
        "\"Paris",
        "\"}",
    ];
    let mut outs: Vec<DetectorOutput> = Vec::new();
    for c in chunks {
        outs.extend(det.process(c));
    }
    assert!(
        det.has_tool_calls(),
        "Expected tool call, got {} events",
        outs.len()
    );
    let has_start = outs
        .iter()
        .any(|o| matches!(o, DetectorOutput::ToolCallStart { name, .. } if name == "get_weather"));
    assert!(has_start, "Missing ToolCallStart");
    let has_delta = outs
        .iter()
        .any(|o| matches!(o, DetectorOutput::ToolCallDelta { args, .. } if args.contains("Paris")));
    assert!(has_delta, "Missing ToolCallDelta with Paris");
}

#[test]
fn mistral_native_parser_format_tool_calls() {
    let parser = MistralNativeParser;
    let calls = vec![IncomingToolCall {
        id: Some("call_1".into()),
        function: IncomingFunction {
            name: "get_weather".into(),
            arguments: r#"{"location":"Paris"}"#.into(),
        },
    }];
    let formatted = parser.format_tool_calls(&calls);
    assert!(formatted.contains("[TOOL_CALLS]"));
    assert!(formatted.contains("get_weather"));
    assert!(formatted.contains("[ARGS]"));
    assert!(formatted.contains("Paris"));
}

#[test]
fn mistral_native_parser_name_and_dispatch() {
    let p = ToolCallFormat::Mistral.into_parser();
    assert_eq!(p.name(), "mistral");
    assert!("mistral".parse::<ToolCallFormat>().is_ok());
}

#[test]
fn find_balanced_json_end_simple() {
    assert_eq!(find_balanced_json_end(r#"{"a":1}"#), Some(7));
    assert_eq!(find_balanced_json_end(r#"{"a":{"b":2}}"#), Some(13));
    assert_eq!(find_balanced_json_end(r#"{"a":1}extra"#), Some(7));
    assert_eq!(find_balanced_json_end(r#"{"a":1"#), None);
    assert_eq!(find_balanced_json_end("not json"), None);
}

#[test]
fn find_balanced_json_end_string_braces() {
    // 2026-09-26: A brace inside a string, or an escaped quote, does not end
    // the object.
    assert_eq!(find_balanced_json_end(r#"{"msg":"hi } there"}"#), Some(20));
    assert_eq!(find_balanced_json_end(r#"{"msg":"say \"hi\""}"#), Some(20));
}

#[test]
fn parse_mistral_truncated_args_recovered() {
    // 2026-09-26: The second call's JSON is cut off; the first still parses.
    let input = "[TOOL_CALLS]search[ARGS]{\"q\":\"rust\"}[TOOL_CALLS]broken[ARGS]{\"incomp";
    let (_, calls) = parse_tool_calls(input);
    assert!(!calls.is_empty());
    assert_eq!(calls[0].function.name, "search");
}

#[test]
fn qwen3_tool_call_close_inside_param_value_not_terminating() {
    // 2026-09-26: `parse_tool_calls` accepts a `</tool_call>` only outside a
    // parameter value, so one inside a value does not end the call.
    let input = "<tool_call>\n<function=write>\n<parameter=content>before</tool_call> after</parameter>\n</function>\n</tool_call>";
    let (_, calls) = parse_tool_calls(input);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "write");
    assert!(
        calls[0].function.arguments.contains("</tool_call>"),
        "value should preserve embedded close substring: {}",
        calls[0].function.arguments,
    );
}

#[test]
fn qwen3_missing_param_close_doesnt_swallow_next_param() {
    // 2026-09-26: A value with no `</parameter>` ends at the next
    // `<parameter=`.
    let input = "<tool_call>\n<function=foo>\n<parameter=a>first<parameter=b>second</parameter>\n</function>\n</tool_call>";
    let (_, calls) = parse_tool_calls(input);
    assert_eq!(calls.len(), 1);
    let args: serde_json::Value =
        serde_json::from_str(&calls[0].function.arguments).expect("args parse");
    assert_eq!(args["a"].as_str().unwrap_or(""), "first");
    assert_eq!(args["b"].as_str().unwrap_or(""), "second");
}

#[test]
fn salvage_param_as_function_recovers_write_call() {
    // 2026-09-26: A call opened with `<parameter=write>` instead of
    // `<function=write>`, with no envelope, is rebuilt by
    // `ParamAsFunctionSalvagePass`.
    let input = "<parameter=write>\n\
            <parameter=content>\n\
            [package]\nname = \"calc\"\nversion = \"0.1.0\"\n\
            </parameter>\n\
            <parameter=filePath>\n\
            /tmp/calc-test26/Cargo.toml\n\
            </parameter>\n\
            </function>";
    let pipeline = ToolCallPipeline::bare_function_default();
    let (_content, calls) = pipeline.run(input);
    assert_eq!(calls.len(), 1, "salvage should recover exactly one call");
    assert_eq!(calls[0].function.name, "write");
    let args: serde_json::Value =
        serde_json::from_str(&calls[0].function.arguments).expect("args parse");
    assert!(
        args["content"].as_str().unwrap_or("").contains("[package]"),
        "content arg should carry the Cargo.toml body: {}",
        calls[0].function.arguments,
    );
    assert_eq!(
        args["filePath"].as_str().unwrap_or(""),
        "/tmp/calc-test26/Cargo.toml"
    );
}

#[test]
fn salvage_does_not_fire_when_legitimate_function_opener_present() {
    // 2026-09-26: Salvage passes run only when the primary passes found
    // nothing, so a well-formed block yields exactly one call.
    let input = "prelude\n<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>";
    let pipeline = ToolCallPipeline::bare_function_default();
    let (content, calls) = pipeline.run(input);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "bash");
    assert_eq!(content.as_deref(), Some("prelude"));
}

#[test]
fn salvage_requires_closing_function_tag() {
    // 2026-09-26: Without a `</function>` the salvage pass does not fire:
    // the text may be prose that uses `<parameter=…>`.
    let input = "Some prose with <parameter=key> but no close tag.";
    let pipeline = ToolCallPipeline::bare_function_default();
    let (_content, calls) = pipeline.run(input);
    assert!(
        calls.is_empty(),
        "salvage must not fire without </function>"
    );
}

#[test]
fn salvage_rejects_non_identifier_name() {
    // 2026-09-26: A first `<parameter=…>` name with characters outside a
    // tool name (here `/`) stops the salvage.
    let input = "<parameter=some/thing>\n\
            <parameter=k>v</parameter>\n\
            </function>";
    let pipeline = ToolCallPipeline::bare_function_default();
    let (_content, calls) = pipeline.run(input);
    assert!(
        calls.is_empty(),
        "non-identifier first param must not salvage"
    );
}

#[test]
fn qwen3_coder_prompt_has_antinarration_hardening() {
    // 2026-09-26: Pins the rule sentences of the qwen3_coder system prompt.
    let parser = Qwen3CoderParser;
    let prompt = parser.system_prompt(
        &[],
        &ToolChoice::Mode("auto".into()),
        &crate::tool_parser::PromptLevers::OFF,
    );
    assert!(
        prompt.contains("<IMMEDIATE_TOOL_USE>"),
        "IMMEDIATE_TOOL_USE block missing"
    );
    assert!(
        prompt.contains("ONLY place file content, commands"),
        "declarative parameter-content sentence missing"
    );
    assert!(
        prompt.contains("NEVER simulate a tool response"),
        "anti-simulation rule missing"
    );
    assert!(
        prompt.contains("NEVER emit bare tags"),
        "bare-tag rule missing"
    );
    assert!(
        prompt.contains("EVERY required parameter MUST have a non-empty value"),
        "required-param rule missing"
    );
}

#[test]
fn qwen3_coder_f33_bash_description_has_retry_rule() {
    // 2026-09-26: The Bash tool's description gets the `[metrale-f33]` retry
    // rule appended; a Write tool's does not.
    let parser = Qwen3CoderParser;
    let bash_tool = ToolDefinition {
        tool_type: "function".into(),
        function: FunctionDefinition {
            name: "Bash".into(),
            description: Some("Execute a shell command.".into()),
            parameters: None,
        },
    };
    let prompt = parser.system_prompt(
        std::slice::from_ref(&bash_tool),
        &ToolChoice::Mode("auto".into()),
        &crate::tool_parser::PromptLevers::OFF,
    );
    assert!(
        prompt.contains("[metrale-f33]"),
        "F33 marker missing from Bash description"
    );
    assert!(
        prompt.contains("do NOT retry the same command"),
        "F33 retry-rule body missing from Bash description"
    );
    let write_tool = ToolDefinition {
        tool_type: "function".into(),
        function: FunctionDefinition {
            name: "Write".into(),
            description: Some("Write a file.".into()),
            parameters: None,
        },
    };
    let prompt_write_only = parser.system_prompt(
        std::slice::from_ref(&write_tool),
        &ToolChoice::Mode("auto".into()),
        &crate::tool_parser::PromptLevers::OFF,
    );
    assert!(
        !prompt_write_only.contains("[metrale-f33]"),
        "F33 marker should only attach to Bash, not Write"
    );
}

#[test]
fn leak_markers_default_is_empty() {
    // 2026-09-26: These parsers declare no orphan openers and no close
    // markers.
    let defaults: &[Box<dyn ToolCallParser>] = &[
        Box::new(HermesParser),
        Box::new(Gemma4Parser),
        Box::new(MistralNativeParser),
        Box::new(BareJsonParser),
    ];
    for parser in defaults {
        let m = parser.leak_markers();
        assert!(
            m.orphan_open.is_empty(),
            "{}: orphan_open should be empty by default",
            parser.name()
        );
        assert!(
            m.close.is_empty(),
            "{}: close should be empty by default",
            parser.name()
        );
    }
}

#[test]
fn leak_markers_qwen3_coder_has_parameter_open() {
    let m = Qwen3CoderParser.leak_markers();
    assert!(m.orphan_open.contains(&"<parameter="));
    assert!(m.orphan_open.contains(&"<tool_response>"));
    assert!(m.close.contains(&"</parameter>"));
    assert!(m.close.contains(&"</function>"));
    assert!(m.close.contains(&"</tool_call>"));
    assert!(m.close.contains(&"</tool_response>"));
}

#[test]
fn leak_markers_minimax_has_invoke_open() {
    let m = MinimaxXmlParser.leak_markers();
    assert!(m.orphan_open.contains(&"<invoke name=\""));
    assert!(m.orphan_open.contains(&"<parameter name=\""));
    assert!(m.close.contains(&"</invoke>"));
    assert!(m.close.contains(&"</parameter>"));
    assert!(m.close.contains(&"</minimax:tool_call>"));
    assert!(m.close.contains(&"</tool_call>"));
}

// 2026-09-26: `validate_tool_calls` repairs a tool name that differs from a
// registered one only in `_`/`-` runs or case (`fuzzy_match_tool_name`,
// separator-normalised match).

fn _mcp_tool(name: &str) -> ToolDefinition {
    ToolDefinition {
        tool_type: "function".to_string(),
        function: FunctionDefinition {
            name: name.to_string(),
            description: None,
            parameters: Some(serde_json::json!({})),
        },
    }
}

fn _mcp_call(name: &str) -> ToolCall {
    ToolCall {
        id: "call_test".to_string(),
        call_type: "function".to_string(),
        function: FunctionCall {
            name: name.to_string(),
            arguments: "{}".to_string(),
        },
    }
}

#[test]
fn fuzzy_repair_mcp_double_underscore_dropped() {
    let tools = vec![_mcp_tool("mcp__discord__discord_send")];
    let calls = vec![_mcp_call("mcp_discord__discord_send")];
    let v = validate_tool_calls(calls, &tools);
    assert!(v.errors.is_empty(), "errors: {:?}", v.errors);
    assert_eq!(v.valid.len(), 1);
    assert_eq!(v.valid[0].function.name, "mcp__discord__discord_send");
}

#[test]
fn fuzzy_repair_mcp_double_underscore_added() {
    let tools = vec![_mcp_tool("mcp_discord_send")];
    let calls = vec![_mcp_call("mcp__discord__send")];
    let v = validate_tool_calls(calls, &tools);
    assert!(v.errors.is_empty(), "errors: {:?}", v.errors);
    assert_eq!(v.valid.len(), 1);
    assert_eq!(v.valid[0].function.name, "mcp_discord_send");
}

#[test]
fn fuzzy_repair_dash_vs_underscore() {
    let tools = vec![_mcp_tool("read-file")];
    let calls = vec![_mcp_call("read_file")];
    let v = validate_tool_calls(calls, &tools);
    assert!(v.errors.is_empty(), "errors: {:?}", v.errors);
    assert_eq!(v.valid.len(), 1);
    assert_eq!(v.valid[0].function.name, "read-file");
}

#[test]
fn fuzzy_repair_case_insensitive() {
    let tools = vec![_mcp_tool("get_weather")];
    let calls = vec![_mcp_call("Get_Weather")];
    let v = validate_tool_calls(calls, &tools);
    assert!(v.errors.is_empty(), "errors: {:?}", v.errors);
    assert_eq!(v.valid.len(), 1);
    assert_eq!(v.valid[0].function.name, "get_weather");
}

#[test]
fn fuzzy_repair_no_false_positive_on_distinct_names() {
    let tools = vec![_mcp_tool("get_weather"), _mcp_tool("get_news")];
    let calls = vec![_mcp_call("get_unknown")];
    let v = validate_tool_calls(calls, &tools);
    // 2026-09-26: No strategy of `fuzzy_match_tool_name` matches, so the call
    // is refused as an unknown tool.
    assert_eq!(v.valid.len(), 0);
    assert_eq!(v.errors.len(), 1);
    assert!(v.errors[0].contains("Unknown tool"));
}
