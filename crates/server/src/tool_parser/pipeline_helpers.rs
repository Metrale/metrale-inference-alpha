// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(unused_imports, dead_code)]

//! 2026-09-26: The bare-function and JSON fallbacks shared by
//! `parse_tool_calls` and the streaming detector's `flush`.
//!
//! Owner: server (tool parsing).
//! Invariants: none beyond the types.

use super::*;

/// 2026-09-26: Tool calls written without a `<tool_call>` wrapper, found by
/// `ToolCallPipeline::bare_function_default`.
pub(super) fn parse_bare_function_calls(text: &str) -> (Option<String>, Vec<ToolCall>) {
    ToolCallPipeline::bare_function_default().run(text)
}

/// 2026-09-26: The fenced code blocks in `text`, each with its fences.
pub(super) fn extract_json_code_blocks(text: &str) -> Vec<String> {
    let mut blocks = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find("```") {
        let after_fence = &rest[start + 3..];
        // 2026-09-26: The block's content starts after the first newline,
        // which skips a language tag such as `json`.
        let content_start = after_fence.find('\n').map_or(0, |p| p + 1);
        if let Some(end) = after_fence[content_start..].find("```") {
            let full_block = &rest[start..start + 3 + content_start + end + 3];
            blocks.push(full_block.to_string());
            rest = &rest[start + 3 + content_start + end + 3..];
        } else {
            break;
        }
    }
    blocks
}

/// 2026-09-26: Tool calls written as JSON: the content of each code block,
/// each line that is a whole `{…}` or `[…]` value, and each balanced
/// `{"name"…}` object in prose that has `"arguments"` and parses.
///
/// Accepted shapes:
/// 1. `{"name": "Write", "arguments": {...}}`
/// 2. `["Write", {...}]`
pub(super) fn parse_json_fallback_calls(text: &str) -> Vec<ToolCall> {
    let mut calls = Vec::new();

    let mut candidates: Vec<String> = Vec::new();
    for block in extract_json_code_blocks(text) {
        let inner = block.trim_start_matches("```");
        let inner = if let Some(pos) = inner.find('\n') {
            &inner[pos + 1..]
        } else {
            inner
        };
        let inner = inner.trim_end_matches("```").trim();
        if !inner.is_empty() {
            candidates.push(inner.to_string());
        }
    }

    for line in text.lines() {
        let trimmed = line.trim();
        if (trimmed.starts_with('{') && trimmed.ends_with('}'))
            || (trimmed.starts_with('[') && trimmed.ends_with(']'))
        {
            candidates.push(trimmed.to_string());
        }
    }

    // 2026-09-26: A call inside a sentence (`Here is the call:
    // {"name": ...}.`) is not a whole line, so it is found by scanning for
    // `{"name"` and taking the balanced object from there.
    {
        let bytes = text.as_bytes();
        let needle = b"{\"name\"";
        let mut search_start = 0;
        while let Some(rel) = text[search_start..].find("{\"name\"") {
            let start = search_start + rel;
            let mut depth = 0i32;
            let mut in_str = false;
            let mut escape = false;
            let mut end = start;
            for i in start..bytes.len() {
                let c = bytes[i];
                if escape {
                    escape = false;
                    end = i;
                    continue;
                }
                if in_str {
                    match c {
                        b'\\' => escape = true,
                        b'"' => in_str = false,
                        _ => {}
                    }
                    end = i;
                    continue;
                }
                match c {
                    b'"' => in_str = true,
                    b'{' => depth += 1,
                    b'}' => {
                        depth -= 1;
                        if depth == 0 {
                            end = i + 1;
                            break;
                        }
                    }
                    _ => {}
                }
                end = i + 1;
            }
            if depth == 0 && end > start {
                let slice = &text[start..end];
                if slice.contains("\"arguments\"")
                    && serde_json::from_str::<serde_json::Value>(slice).is_ok()
                {
                    candidates.push(slice.to_string());
                }
                search_start = end;
            } else {
                search_start = start + needle.len();
            }
        }
    }

    // 2026-09-26: The scans can find the same JSON more than once (a code
    // block line, a whole line, an embedded object); duplicates are removed.
    candidates.sort();
    candidates.dedup();

    for candidate in &candidates {
        if let Ok(obj) = serde_json::from_str::<serde_json::Value>(candidate) {
            if let Some(name) = obj.get("name").and_then(|n| n.as_str()) {
                let args = obj
                    .get("arguments")
                    .map(|a| serde_json::to_string(a).unwrap_or_default())
                    .unwrap_or_else(|| "{}".to_string());
                calls.push(ToolCall {
                    id: next_tool_call_id(),
                    call_type: "function".to_string(),
                    function: FunctionCall {
                        name: normalize_tool_name(name),
                        arguments: args,
                    },
                });
                continue;
            }

            if let Some(arr) = obj.as_array()
                && arr.len() == 2
                && let (Some(name), Some(_args_obj)) = (arr[0].as_str(), arr[1].as_object())
            {
                calls.push(ToolCall {
                    id: next_tool_call_id(),
                    call_type: "function".to_string(),
                    function: FunctionCall {
                        name: normalize_tool_name(name),
                        arguments: serde_json::to_string(&arr[1]).unwrap_or_default(),
                    },
                });
                continue;
            }
        }

        if candidate.contains("\"name\"")
            && candidate.contains("\"arguments\"")
            && let Ok(obj) = serde_json::from_str::<serde_json::Value>(candidate)
            && let Some(name) = obj.get("name").and_then(|n| n.as_str())
        {
            let args = obj
                .get("arguments")
                .map(|a| serde_json::to_string(a).unwrap_or_default())
                .unwrap_or_else(|| "{}".to_string());
            calls.push(ToolCall {
                id: next_tool_call_id(),
                call_type: "function".to_string(),
                function: FunctionCall {
                    name: normalize_tool_name(name),
                    arguments: args,
                },
            });
        }
    }

    calls
}
