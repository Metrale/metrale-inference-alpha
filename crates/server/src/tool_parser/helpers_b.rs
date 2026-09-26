// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(unused_imports, dead_code)]

//! 2026-09-26: Mistral and Gemma-4 native call parsing, Gemma-4 value
//! rendering, and the system-prompt helpers the parsers share.
//!
//! Owner: server (tool parsing).
//! Invariants: none beyond the types.

use super::*;

/// 2026-09-26: Parse all Mistral native tool calls from completed output.
/// Returns `(content_before_first_call, tool_calls)`; a segment that does not
/// parse is skipped.
pub(super) fn parse_mistral_native_calls(text: &str) -> (Option<String>, Vec<ToolCall>) {
    let mut calls = Vec::new();
    let mut content: Option<String> = None;
    let first_tag = match text.find(MISTRAL_TOOL_CALLS_TAG) {
        Some(p) => p,
        None => return (None, calls),
    };
    let before = text[..first_tag].trim();
    if !before.is_empty() {
        content = Some(before.to_string());
    }
    // 2026-09-26: `text[first_tag..]` starts with the tag, so the first split
    // element is empty and is skipped.
    let segments = text[first_tag..].split(MISTRAL_TOOL_CALLS_TAG).skip(1);
    for segment in segments {
        if segment.trim().is_empty() {
            continue;
        }
        if let Some(tc) = parse_mistral_native_call(segment) {
            calls.push(tc);
        }
    }
    (content, calls)
}

/// 2026-09-26: Render a JSON value in Gemma-4 notation: strings wrapped in
/// `<|"|>`, object keys unquoted.
pub(super) fn format_gemma4_value(out: &mut String, val: &serde_json::Value) {
    match val {
        serde_json::Value::String(s) => {
            out.push_str("<|\"|>");
            out.push_str(s);
            out.push_str("<|\"|>");
        }
        serde_json::Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        serde_json::Value::Number(n) => out.push_str(&n.to_string()),
        serde_json::Value::Null => out.push_str("null"),
        serde_json::Value::Array(arr) => {
            out.push('[');
            for (i, item) in arr.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                format_gemma4_value(out, item);
            }
            out.push(']');
        }
        serde_json::Value::Object(obj) => {
            out.push('{');
            let mut first = true;
            for (key, v) in obj {
                if !first {
                    out.push(',');
                }
                first = false;
                out.push_str(key);
                out.push(':');
                format_gemma4_value(out, v);
            }
            out.push('}');
        }
    }
}

/// 2026-09-26: Parse a Gemma-4 native call, `call:fn_name{key:val,...}` or
/// `_call:fn_name{...}`. Arguments that do not convert to valid JSON
/// (`gemma4_to_json`) become `{}`.
pub(super) fn parse_gemma4_native_call(text: &str) -> Option<ToolCall> {
    let text = text.trim();
    let rest = text
        .strip_prefix("call:")
        .or_else(|| text.strip_prefix("_call:"))?;
    let brace_pos = rest.find('{')?;
    let name = normalize_tool_name(&rest[..brace_pos]);
    if name.is_empty() {
        return None;
    }

    let args_str = &rest[brace_pos..];
    let json_str = gemma4_to_json(args_str);
    let arguments = if let Ok(_v) = serde_json::from_str::<serde_json::Value>(&json_str) {
        json_str
    } else {
        "{}".to_string()
    };

    Some(ToolCall {
        id: next_tool_call_id(),
        call_type: "function".into(),
        function: FunctionCall { name, arguments },
    })
}

/// 2026-09-26: Convert Gemma-4 native `{key:<|"|>val<|"|>,...}` to JSON
/// `{"key":"val",...}`. A word of alphanumerics and `_` followed by `:` is
/// quoted as a key; any other such word except `true`, `false` and `null` is
/// quoted as a string, digits included.
pub(super) fn gemma4_to_json(native: &str) -> String {
    let s = native.replace("<|\"|>", "\"");
    let mut result = String::with_capacity(s.len() + 32);
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '"' {
            result.push('"');
            i += 1;
            while i < chars.len() && chars[i] != '"' {
                if chars[i] == '\\' && i + 1 < chars.len() {
                    result.push(chars[i]);
                    i += 1;
                }
                result.push(chars[i]);
                i += 1;
            }
            if i < chars.len() {
                result.push('"');
                i += 1;
            }
        } else if chars[i].is_alphanumeric() || chars[i] == '_' {
            let start = i;
            while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            let word = &s[start..i];
            if i < chars.len() && chars[i] == ':' {
                result.push('"');
                result.push_str(word);
                result.push('"');
            } else {
                match word {
                    "true" | "false" | "null" => result.push_str(word),
                    _ => {
                        result.push('"');
                        result.push_str(word);
                        result.push('"');
                    }
                }
            }
        } else {
            result.push(chars[i]);
            i += 1;
        }
    }
    result
}

/// 2026-09-26: Append the `tool_choice` instruction: one for "required", one
/// naming the function for a specific choice, nothing otherwise.
pub(crate) fn append_tool_choice_instruction(prompt: &mut String, tool_choice: &ToolChoice) {
    match tool_choice {
        ToolChoice::Mode(s) if s == "required" => {
            prompt.push_str(
                "\n\n<IMPORTANT>\nYou MUST call at least one function. \
                 Do NOT respond with text — respond ONLY with a <tool_call> block.\n</IMPORTANT>",
            );
        }
        ToolChoice::Specific { function } => {
            prompt.push_str(&format!(
                "\n\n<IMPORTANT>\nYou MUST call the '{}' function. \
                 Do NOT respond with text — respond ONLY with a <tool_call> block \
                 calling '{}'.\n</IMPORTANT>",
                function.name, function.name,
            ));
        }
        _ => {}
    }
}

/// 2026-09-26: The `<tools>` body for a parser's `system_prompt()`: the
/// compact TSCG signatures when `levers.tscg` is set, otherwise the parser's
/// own JSON rendering, `render_json`.
pub(super) fn tool_list_body(
    tools: &[ToolDefinition],
    levers: &super::PromptLevers,
    render_json: impl FnOnce() -> String,
) -> String {
    if levers.tscg {
        crate::tscg::compile_tools(tools)
    } else {
        render_json()
    }
}
