// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests of the qwen3_coder tool grammar
//! (`compile_qwen3_coder_tool_grammar`). Inside
//! `<tool_call>\n<function=NAME>\n … \n</function>\n</tool_call>` the body
//! must be `<parameter=KEY>VALUE</parameter>` blocks
//! (`xml_param_value_body_ebnf`), with a value that holds at least one
//! non-whitespace byte.
//!
//! The empty-value cases assume `METRALE_GRAMMAR_ALLOW_EMPTY_VALUE` is
//! unset; with it set to `1`, an empty value is accepted.
//!
//! Owner: server (grammar).
//! Invariants: none beyond the types.

use super::*;
use metrale_grammar::{CompiledGrammar, GrammarMatcher};

fn exec_tool_def() -> Vec<ToolDefinition> {
    vec![ToolDefinition {
        tool_type: "function".to_string(),
        function: crate::tool_parser::FunctionDefinition {
            name: "exec".to_string(),
            description: Some("Run a shell command".to_string()),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "command": {"type": "string"}
                },
                "required": ["command"]
            })),
        },
    }]
}

/// 2026-09-26: Same as `minimax.rs::grammar_accepts`: a fresh matcher
/// accepts `input` only if every byte is accepted and the matcher ends
/// terminated.
fn grammar_accepts(compiled: &CompiledGrammar, input: &str) -> bool {
    let mut matcher =
        GrammarMatcher::new(compiled, None, true, -1).expect("GrammarMatcher::new failed");
    if !matcher.accept_string(input, false) {
        return false;
    }
    matcher.is_terminated()
}

#[test]
fn qwen3_coder_grammar_accepts_canonical_xml_body() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();
    let tools = exec_tool_def();
    let compiled = engine
        .compile_qwen3_coder_tool_grammar(&tools, true, "</parameter>")
        .expect("compile must succeed");

    let canonical_xml = "<tool_call>\n<function=exec>\n<parameter=command>ls /tmp</parameter>\n</function>\n</tool_call>";
    assert!(
        grammar_accepts(&compiled, canonical_xml),
        "canonical native-XML qwen3_coder body must be accepted; input: {canonical_xml:?}"
    );
}

/// 2026-09-26: Tool names where one is a prefix of another
/// (`mcp_scrapling_get`, `mcp_scrapling_get_prompt`) still compile in auto
/// mode. Each per-tool trigger ends with `>`, so no trigger is a prefix of
/// another tool's tag, which the triggered-tags converter would reject.
#[test]
fn qwen3_coder_grammar_compiles_with_shared_tool_name_prefixes() {
    fn mcp_tool(name: &str) -> ToolDefinition {
        ToolDefinition {
            tool_type: "function".to_string(),
            function: crate::tool_parser::FunctionDefinition {
                name: name.to_string(),
                description: Some(format!("MCP tool {name}")),
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": { "query": {"type": "string"} }
                })),
            },
        }
    }

    let tools = vec![
        mcp_tool("mcp_scrapling_get"),
        mcp_tool("mcp_scrapling_get_prompt"),
        mcp_tool("mcp_scrapling_fetch"),
        mcp_tool("mcp_scrapling_bulk_fetch"),
    ];

    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();

    // 2026-09-26: `use_triggers = true` (auto mode) builds per-tool
    // triggers, unless `METRALE_TOOL_SHORT_TRIGGER=1`.
    let compiled = engine
        .compile_qwen3_coder_tool_grammar(&tools, true, "</parameter>")
        .expect("grammar with shared-prefix tool names must compile (issue #88)");

    let long_call = "<tool_call>\n<function=mcp_scrapling_get_prompt>\n<parameter=query>\nhi\n</parameter>\n</function>\n</tool_call>";
    assert!(
        grammar_accepts(&compiled, long_call),
        "longer prefix-colliding tool must be accepted; input: {long_call:?}"
    );

    let short_call = "<tool_call>\n<function=mcp_scrapling_get>\n<parameter=query>\nhi\n</parameter>\n</function>\n</tool_call>";
    assert!(
        grammar_accepts(&compiled, short_call),
        "shorter tool must remain accepted; input: {short_call:?}"
    );
}

/// 2026-09-26: A JSON body (`<function=exec>{...}</function>`) is rejected;
/// the body must be `<parameter=KEY>VALUE</parameter>` blocks.
#[test]
fn qwen3_coder_grammar_rejects_json_body_enforces_xml() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();
    let tools = exec_tool_def();
    let compiled = engine
        .compile_qwen3_coder_tool_grammar(&tools, true, "</parameter>")
        .expect("compile must succeed");

    let json_body =
        "<tool_call>\n<function=exec>\n{\"command\": \"ls /tmp\"}\n</function>\n</tool_call>";
    assert!(
        !grammar_accepts(&compiled, json_body),
        "grammar must enforce native XML <parameter=>; the JSON body is parser-fallback \
         (grammar-off) only. input: {json_body:?}"
    );
}

#[test]
fn qwen3_coder_grammar_accepts_multi_xml_params() {
    let tools = vec![ToolDefinition {
        tool_type: "function".to_string(),
        function: crate::tool_parser::FunctionDefinition {
            name: "write".to_string(),
            description: Some("Write to a file".to_string()),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "filePath": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["filePath", "content"]
            })),
        },
    }];

    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();
    let compiled = engine
        .compile_qwen3_coder_tool_grammar(&tools, true, "</parameter>")
        .expect("compile must succeed");

    let multi_param = "<tool_call>\n<function=write>\n<parameter=filePath>/tmp/test-rust-axum-v42/Cargo.toml</parameter>\n<parameter=content>[package]\nname = \"test-rust-axum-v42\"</parameter>\n</function>\n</tool_call>";
    assert!(
        grammar_accepts(&compiled, multi_param),
        "multi-param native XML body must be accepted with full byte fidelity \
         (path tokens like `axum-v42` and content tokens with newlines/quotes). \
         Input: {multi_param:?}"
    );
}

/// 2026-09-26: An empty value is rejected: `value ::= leading_ws
/// nonempty_value` needs one byte that `first_content` accepts.
#[test]
fn qwen3_coder_grammar_rejects_empty_parameter_body() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();
    let tools = exec_tool_def();
    let compiled = engine
        .compile_qwen3_coder_tool_grammar(&tools, true, "</parameter>")
        .expect("compile must succeed");

    let empty_body =
        "<tool_call>\n<function=exec>\n<parameter=command></parameter>\n</function>\n</tool_call>";
    assert!(
        !grammar_accepts(&compiled, empty_body),
        "empty parameter body must be REJECTED by Tier-0 regex. Input: {empty_body:?}"
    );
}

#[test]
fn qwen3_coder_grammar_rejects_whitespace_only_parameter_body() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();
    let tools = exec_tool_def();
    let compiled = engine
        .compile_qwen3_coder_tool_grammar(&tools, true, "</parameter>")
        .expect("compile must succeed");

    let whitespace_body = "<tool_call>\n<function=exec>\n<parameter=command>   \n  </parameter>\n</function>\n</tool_call>";
    assert!(
        !grammar_accepts(&compiled, whitespace_body),
        "whitespace-only parameter body must be REJECTED. Input: {whitespace_body:?}"
    );
}

/// 2026-09-26: A value may open with whitespace, a newline included
/// (`leading_ws`), as long as a non-whitespace byte follows.
#[test]
fn qwen3_coder_grammar_accepts_leading_newline_content() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();
    let tools = exec_tool_def();
    let compiled = engine
        .compile_qwen3_coder_tool_grammar(&tools, true, "</parameter>")
        .expect("compile must succeed");

    let leading_nl = "<tool_call>\n<function=exec>\n<parameter=command>\nls /tmp</parameter>\n</function>\n</tool_call>";
    assert!(
        grammar_accepts(&compiled, leading_nl),
        "a leading newline before real content must be ACCEPTED. Input: {leading_nl:?}"
    );
}

/// 2026-09-26: `first_content` excludes `=` and `>`. A tokenizer can merge
/// the key's closing `>` with the value's first byte into one token (`>=`),
/// which would leave a stray `=` as the value's first character; rejecting
/// a leading `=` forces a lone `>` token instead.
#[test]
fn qwen3_coder_grammar_rejects_eq_value_start() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();
    let tools = exec_tool_def();
    let compiled = engine
        .compile_qwen3_coder_tool_grammar(&tools, true, "</parameter>")
        .expect("compile must succeed");

    let eq_start = "<tool_call>\n<function=exec>\n<parameter=command>=ls /tmp</parameter>\n</function>\n</tool_call>";
    assert!(
        !grammar_accepts(&compiled, eq_start),
        "a value starting with `=` (the `>=`-merge artifact) must be REJECTED. Input: {eq_start:?}"
    );
}

/// 2026-09-26: A value may start with `<` unless the `<` begins the exact
/// close `</parameter>`: `first_content`'s `<` arms come from the close
/// ladder (`ladder_lt_arms`).
#[test]
fn qwen3_coder_grammar_accepts_lt_initial_value() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();
    let tools = exec_tool_def();
    let compiled = engine
        .compile_qwen3_coder_tool_grammar(&tools, true, "</parameter>")
        .expect("compile must succeed");

    let svelte = "<tool_call>\n<function=exec>\n<parameter=command><script>\n  let x = 1;\n</script>\n\n<main>\n  <button on:click={inc}>+</button>\n</main></parameter>\n</function>\n</tool_call>";
    assert!(
        grammar_accepts(&compiled, svelte),
        "a `<script>`-initial value (Svelte component) must be ACCEPTED"
    );

    let html = "<tool_call>\n<function=exec>\n<parameter=command><!DOCTYPE html>\n<html><body>hi</body></html></parameter>\n</function>\n</tool_call>";
    assert!(
        grammar_accepts(&compiled, html),
        "a `<!DOCTYPE`-initial value (HTML document) must be ACCEPTED"
    );

    let close_prefix = "<tool_call>\n<function=exec>\n<parameter=command></div> is a stray close tag</parameter>\n</function>\n</tool_call>";
    assert!(
        grammar_accepts(&compiled, close_prefix),
        "a `</div`-initial value must be ACCEPTED (only the exact close is barred)"
    );
}

/// 2026-09-26: The close tag cannot open a value: no `first_content`
/// alternative matches the start of `</parameter>`.
#[test]
fn qwen3_coder_grammar_still_rejects_close_tag_as_first_body_token() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();
    let tools = exec_tool_def();
    let compiled = engine
        .compile_qwen3_coder_tool_grammar(&tools, true, "</parameter>")
        .expect("compile must succeed");

    let empty_body =
        "<tool_call>\n<function=exec>\n<parameter=command></parameter>\n</function>\n</tool_call>";
    assert!(
        !grammar_accepts(&compiled, empty_body),
        "an empty body (immediate close tag) must remain REJECTED"
    );
}
