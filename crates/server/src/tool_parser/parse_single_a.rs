// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(unused_imports, dead_code)]

//! 2026-09-26: Parsers for the body of one tool call: format detection
//! (`parse_one_call`), JSON and truncated JSON, and MiniMax `<invoke>`.
//!
//! Owner: server (tool parsing).
//! Invariants:
//! - A bare identifier becomes a call only through `parse_complete_call` with
//!   `promote_bare_names` set.

use super::*;

/// 2026-09-26: Parse a body whose outer `</tool_call>` was seen. A bare
/// identifier is Poolside's zero-argument shape and is only safe to recognize
/// here: the same bytes without an outer close are an incomplete envelope.
pub(super) fn parse_complete_call(
    text: &str,
    idx: u32,
    promote_bare_names: bool,
) -> Option<ToolCall> {
    let body = text.trim();
    if is_tool_name_component(body) {
        if promote_bare_names {
            return parse_poolside_v1_call(body);
        }
        // 2026-09-26: For every other format a bare identifier inside
        // `<tool_call>` is malformed output, not a call; it is logged and
        // dropped.
        tracing::warn!(
            "tool_parser: dropped bare-identifier <tool_call> body {body:?} \
             (parser does not promote bare names; pre-batch4 behaviour)"
        );
        return None;
    }
    parse_one_call(body, idx)
}

/// 2026-09-26: Detect the format of one call body and parse it. Tried in
/// order: a Poolside `<arg_key>` body, Gemma-4 `call:`, complete JSON,
/// truncated JSON, MiniMax `<invoke>`, qwen3_coder `<function=`, and
/// tag-style `<function>NAME</function>`.
pub(super) fn parse_one_call(text: &str, idx: u32) -> Option<ToolCall> {
    if text.contains("<arg_key>")
        && let Some(tc) = parse_poolside_v1_call(text)
    {
        return Some(tc);
    }
    if text.starts_with("call:") || text.starts_with("_call:") {
        return parse_gemma4_native_call(text);
    }
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(text) {
        let name = normalize_tool_name(v.get("name")?.as_str()?);
        let args = v
            .get("arguments")
            .map(|a| {
                // 2026-09-26: A string `arguments` is used as is; any other
                // value is serialized.
                if let Some(s) = a.as_str() {
                    s.to_string()
                } else {
                    serde_json::to_string(a).unwrap_or_else(|_| "{}".into())
                }
            })
            .unwrap_or_else(|| "{}".into());
        return Some(ToolCall {
            id: next_tool_call_id(),
            call_type: "function".into(),
            function: FunctionCall {
                name,
                arguments: args,
            },
        });
    }
    // 2026-09-26: Truncated JSON (the output ended inside the call): the
    // name, plus the longest prefix of `arguments` that ends at `}` or `]`
    // and parses, else `{}`.
    if text.contains("\"name\"") && text.contains("\"arguments\"") {
        let name = extract_json_string(text, "name");
        if let Some(name) = name {
            let name = normalize_tool_name(&name);
            let args = if let Some(args_start) = text.find("\"arguments\"") {
                let after = &text[args_start + "\"arguments\"".len()..];
                let colon = after.find(':').map(|p| p + 1).unwrap_or(0);
                let args_text = after[colon..].trim();
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(args_text) {
                    serde_json::to_string(&v).unwrap_or_else(|_| "{}".into())
                } else {
                    let mut best = "{}".to_string();
                    for (i, ch) in args_text.char_indices().rev() {
                        if (ch == '}' || ch == ']')
                            && serde_json::from_str::<serde_json::Value>(&args_text[..=i]).is_ok()
                        {
                            best = args_text[..=i].to_string();
                            break;
                        }
                    }
                    best
                }
            } else {
                "{}".to_string()
            };
            return Some(ToolCall {
                id: next_tool_call_id(),
                call_type: "function".into(),
                function: FunctionCall {
                    name,
                    arguments: args,
                },
            });
        }
    }
    if let Some(tc) = parse_minimax_xml_call(text, idx) {
        return Some(tc);
    }
    if let Some(tc) = parse_qwen3_coder_call(text, idx) {
        return Some(tc);
    }
    parse_tag_style_call(text, idx)
}

/// 2026-09-26: Parse one MiniMax call body,
/// `<invoke name="NAME"><parameter name="K">V</parameter>...</invoke>`,
/// with the outer envelope already removed. Names may be in `"` or `'`
/// quotes.
pub(super) fn parse_minimax_xml_call(text: &str, idx: u32) -> Option<ToolCall> {
    let _ = idx;
    let invoke_start = text
        .find("<invoke name=\"")
        .or_else(|| text.find("<invoke name='"))?;
    let after = &text[invoke_start + "<invoke name=".len()..];
    let quote = after.as_bytes().first().copied()?;
    if quote != b'"' && quote != b'\'' {
        return None;
    }
    let name_start = 1;
    let name_end = after[name_start..]
        .find(quote as char)
        .map(|p| p + name_start)?;
    let func_name = normalize_tool_name(&after[name_start..name_end]);
    if func_name.is_empty() {
        return None;
    }

    let mut rest = &after[name_end + 1..];
    if let Some(gt) = rest.find('>') {
        rest = &rest[gt + 1..];
    } else {
        return None;
    }

    let mut args = serde_json::Map::new();
    while let Some(p) = rest.find("<parameter name=") {
        rest = &rest[p + "<parameter name=".len()..];
        let q = rest.as_bytes().first().copied()?;
        if q != b'"' && q != b'\'' {
            break;
        }
        let key_start = 1;
        let key_end = rest[key_start..].find(q as char).map(|p| p + key_start)?;
        let param_name = rest[key_start..key_end].trim().to_string();
        rest = &rest[key_end + 1..];
        let gt = rest.find('>')?;
        rest = &rest[gt + 1..];

        let proper = rest.find("</parameter>");
        let next_param = rest.find("<parameter=");
        let func_close = rest.find("</function>");
        let mut val_end = rest.len();
        let mut consumed_close = false;
        if let Some(p) = proper {
            val_end = p;
            consumed_close = true;
        }
        for cand in [next_param, func_close].into_iter().flatten() {
            if cand < val_end {
                val_end = cand;
                consumed_close = false;
            }
        }
        let raw_value = rest[..val_end].trim();
        rest = if consumed_close {
            &rest[val_end + "</parameter>".len()..]
        } else if val_end < rest.len() {
            &rest[val_end..]
        } else {
            ""
        };

        // 2026-09-26: Values stay strings, as in `parse_qwen3_coder_call`.
        args.insert(param_name, serde_json::Value::String(raw_value.to_string()));
    }

    // 2026-09-26: A write or edit tool (the names below) with an empty
    // `file_path`, `filePath` or `path` argument is dropped with a warning:
    // the call returns `None`. Other tools are not checked.
    const PATH_KEYS: &[&str] = &["file_path", "filePath", "path"];
    let is_write_tool = matches!(
        func_name.as_str(),
        "Write" | "write" | "Edit" | "edit" | "MultiEdit" | "multiEdit" | "multi_edit",
    );
    if is_write_tool {
        for key in PATH_KEYS {
            if let Some(serde_json::Value::String(v)) = args.get(*key)
                && v.trim().is_empty()
            {
                tracing::warn!(
                    tool = %func_name,
                    key = key,
                    "F80b: dropping minimax_xml call with empty required path; \
                     model self-truncation"
                );
                return None;
            }
        }
    }

    Some(ToolCall {
        id: next_tool_call_id(),
        call_type: "function".into(),
        function: FunctionCall {
            name: func_name,
            arguments: serde_json::to_string(&serde_json::Value::Object(args))
                .unwrap_or_else(|_| "{}".into()),
        },
    })
}

/// 2026-09-26: Parse every `<invoke name="...">…</invoke>` block in `text`.
/// The streaming detector passes a MiniMax envelope's body, which may hold
/// several blocks; `parse_tool_calls` passes output with bare `<invoke>`
/// blocks.
pub(crate) fn parse_minimax_xml_calls_all(text: &str) -> Vec<ToolCall> {
    let mut out = Vec::new();
    let mut rest = text;
    let mut idx: u32 = 0;
    while let Some(start) = rest.find("<invoke name=") {
        let chunk = &rest[start..];
        // 2026-09-26: Each block is parsed up to its own `</invoke>` (or the
        // end of the text), so one block's parameters never include the
        // next block's.
        let end = match chunk.find("</invoke>") {
            Some(e) => e + "</invoke>".len(),
            None => chunk.len(),
        };
        if let Some(tc) = parse_minimax_xml_call(&chunk[..end], idx) {
            out.push(tc);
            idx += 1;
        }
        rest = &chunk[end..];
    }
    out
}

/// 2026-09-26: The string value of the first `"key"` in possibly truncated
/// JSON, found by substring search. `None` if that value is not a string or
/// has no closing quote. Escapes are kept as written.
fn extract_json_string(text: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{}\"", key);
    let key_pos = text.find(&pattern)?;
    let after_key = &text[key_pos + pattern.len()..];
    let colon = after_key.find(':')?;
    let after_colon = after_key[colon + 1..].trim_start();
    if !after_colon.starts_with('"') {
        return None;
    }
    let val_start = 1;
    let mut i = val_start;
    let bytes = after_colon.as_bytes();
    while i < bytes.len() {
        if bytes[i] == b'\\' {
            i += 2;
        } else if bytes[i] == b'"' {
            return Some(after_colon[val_start..i].to_string());
        } else {
            i += 1;
        }
    }
    None
}
