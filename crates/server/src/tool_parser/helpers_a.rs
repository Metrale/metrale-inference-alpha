// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(unused_imports, dead_code)]

//! 2026-09-26: Salvage parsers for tool calls outside a `<tool_call>`
//! envelope (Mistral segments, `identifier{json}` in prose), the JSON
//! helpers they share, and `normalize_tool_name`.
//!
//! Owner: server (tool parsing).
//! Invariants: none beyond the types.

use super::*;

/// 2026-09-26: Parse one Mistral-native segment, `name[ARGS]{json}`, with the
/// `[TOOL_CALLS]` prefix already stripped. Without `[ARGS]` the name ends at
/// the first `{`.
///
/// Arguments that do not parse as JSON fall back to the first balanced
/// object at the start of the slice (`largest_balanced_json_prefix`), else
/// `{}`; JSON cut off before its closing brace therefore becomes `{}`.
pub(super) fn parse_mistral_native_call(segment: &str) -> Option<ToolCall> {
    let segment = segment.trim_start();
    let (name, json_slice) = if let Some(args_pos) = segment.find(MISTRAL_ARGS_TAG) {
        let name = normalize_tool_name(&segment[..args_pos]);
        let after_args = &segment[args_pos + MISTRAL_ARGS_TAG.len()..];
        let raw_args = after_args.trim();
        let json_start = raw_args.find('{').unwrap_or(0);
        (name, &raw_args[json_start..])
    } else if let Some(brace_pos) = segment.find('{') {
        let name = normalize_tool_name(&segment[..brace_pos]);
        (name, &segment[brace_pos..])
    } else {
        return None;
    };
    // 2026-09-26: A colon that survives normalization marks prose such as
    // `json:{...}`, not a call.
    if !is_normalized_tool_name(&name) {
        return None;
    }
    // 2026-09-26: Rejects text such as "Hello, world" before the first `{`.
    if !name.chars().all(is_tool_name_or_namespace_char) {
        return None;
    }
    let args = if let Ok(v) = serde_json::from_str::<serde_json::Value>(json_slice) {
        serde_json::to_string(&v).unwrap_or_else(|_| "{}".into())
    } else if let Some(valid) = largest_balanced_json_prefix(json_slice) {
        valid.to_string()
    } else {
        "{}".to_string()
    };
    Some(ToolCall {
        id: next_tool_call_id(),
        call_type: "function".into(),
        function: FunctionCall {
            name,
            arguments: args,
        },
    })
}

/// 2026-09-26: Scan `s` from its opening `{` and return the byte offset just
/// past the matching `}`, skipping braces inside strings. `None` if the object
/// is incomplete or `s` does not start with `{`.
pub(super) fn find_balanced_json_end(s: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    if bytes.first() != Some(&b'{') {
        return None;
    }
    let mut depth: i32 = 0;
    let mut in_string = false;
    let mut escaped = false;
    for (i, &b) in bytes.iter().enumerate() {
        if in_string {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_string = false;
            }
            continue;
        }
        match b {
            b'"' => in_string = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i + 1);
                }
            }
            _ => {}
        }
    }
    None
}

/// 2026-09-26: The first balanced `{…}` object at the start of `s`,
/// re-serialized, if it parses as JSON. Text after that object is ignored;
/// an object that never closes gives `None`.
fn largest_balanced_json_prefix(s: &str) -> Option<String> {
    let bytes = s.as_bytes();
    if bytes.first() != Some(&b'{') {
        return None;
    }
    let mut depth: i32 = 0;
    let mut in_string = false;
    let mut escaped = false;
    let mut last_valid: Option<usize> = None;
    for (i, &b) in bytes.iter().enumerate() {
        if in_string {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_string = false;
            }
            continue;
        }
        match b {
            b'"' => in_string = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    last_valid = Some(i + 1);
                    break;
                }
            }
            _ => {}
        }
    }
    let end = last_valid?;
    let prefix = &s[..end];
    serde_json::from_str::<serde_json::Value>(prefix)
        .ok()
        .and_then(|v| serde_json::to_string(&v).ok())
}

/// 2026-09-26: Promote each `identifier{json}` in `text` to a tool call, as
/// in `I'll check the weather for you.get_weather{"city": "Paris"}`.
///
/// The identifier (tool-name characters, `is_tool_name_or_namespace_char`)
/// must be at least 2 bytes, start with a letter or `_`, and touch the `{`
/// with no space between them, so `"Hello, world" {"foo": 1}` does not match.
/// The balanced `{…}` must parse as a JSON object, directly or after
/// `repair_bare_object_json`. Returns the text between calls as content.
pub(super) fn parse_bare_identifier_json_calls(text: &str) -> (Option<String>, Vec<ToolCall>) {
    let mut calls: Vec<ToolCall> = Vec::new();
    let mut content_parts: Vec<String> = Vec::new();
    let mut last_end = 0usize;
    let bytes = text.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'{' {
            let mut start = i;
            while start > 0 {
                let b = bytes[start - 1];
                if (b as char).is_ascii() && is_tool_name_or_namespace_char(b as char) {
                    start -= 1;
                } else {
                    break;
                }
            }
            let name_bytes = &bytes[start..i];
            if name_bytes.len() >= 2
                && (name_bytes[0].is_ascii_alphabetic() || name_bytes[0] == b'_')
                // 2026-09-26: Words that precede `{` in prose and code
                // rather than name a tool.
                && !matches!(name_bytes,
                    b"function" | b"object" | b"params" | b"return" | b"const" |
                    b"let" | b"var" | b"if" | b"else" | b"for" | b"while" | b"class")
            {
                let suffix = &text[i..];
                if let Some(end_rel) = find_balanced_json_end(suffix) {
                    let json_slice = &suffix[..end_rel];
                    // 2026-09-26: Strict JSON first, then the repair for
                    // `name{key:bareword string}` with unquoted keys/values.
                    let parsed = serde_json::from_str::<serde_json::Value>(json_slice)
                        .ok()
                        .or_else(|| {
                            let repaired = repair_bare_object_json(json_slice);
                            serde_json::from_str::<serde_json::Value>(&repaired).ok()
                        });
                    if let Some(v) = parsed {
                        // 2026-09-26: A name that kept its `:` (`json:{...}`
                        // prose) falls through to `i += 1`, which leaves the
                        // text for later fallbacks such as an embedded
                        // `{"name":...}` object.
                        let raw_name = std::str::from_utf8(name_bytes).unwrap_or("");
                        let name = normalize_tool_name(raw_name);
                        if v.is_object() && is_normalized_tool_name(&name) {
                            let args = serde_json::to_string(&v).unwrap_or_else(|_| "{}".into());
                            if start > last_end {
                                let chunk = text[last_end..start].trim();
                                if !chunk.is_empty() {
                                    content_parts.push(chunk.to_string());
                                }
                            }
                            calls.push(ToolCall {
                                id: next_tool_call_id(),
                                call_type: "function".into(),
                                function: FunctionCall {
                                    name,
                                    arguments: args,
                                },
                            });
                            last_end = i + end_rel;
                            i = last_end;
                            continue;
                        }
                    }
                }
            }
        }
        i += 1;
    }
    if !calls.is_empty() && last_end < text.len() {
        let tail = text[last_end..].trim();
        if !tail.is_empty() {
            content_parts.push(tail.to_string());
        }
    }
    let content = if content_parts.is_empty() {
        None
    } else {
        Some(content_parts.join("\n"))
    };
    (content, calls)
}

/// 2026-09-26: Quote the unquoted keys and bareword values of an object's
/// top-level members, so `{query:current Bitcoin price}` becomes
/// `{"query":"current Bitcoin price"}`.
///
/// Top-level members are split on commas outside strings and brackets. A key
/// made of ASCII alphanumerics, `_` and `-` is quoted; a value is quoted
/// unless it starts with `"`, `{` or `[`, is `true`/`false`/`null`, or parses
/// as a number. If any member cannot be repaired, `s` is returned unchanged.
fn repair_bare_object_json(s: &str) -> String {
    let trimmed = s.trim();
    if !trimmed.starts_with('{') || !trimmed.ends_with('}') {
        return s.to_string();
    }
    let inner = &trimmed[1..trimmed.len() - 1];
    let mut out = String::with_capacity(s.len() + 32);
    out.push('{');
    let mut first = true;
    let mut depth: i32 = 0;
    let mut start = 0usize;
    let bytes = inner.as_bytes();
    let mut in_str = false;
    for (i, &b) in bytes.iter().enumerate() {
        if in_str {
            if b == b'"' && (i == 0 || bytes[i - 1] != b'\\') {
                in_str = false;
            }
            continue;
        }
        if b == b'"' {
            in_str = true;
            continue;
        }
        if b == b'{' || b == b'[' {
            depth += 1;
            continue;
        }
        if b == b'}' || b == b']' {
            depth -= 1;
            continue;
        }
        if b == b',' && depth == 0 {
            if !append_repaired_member(&mut out, &inner[start..i], &mut first) {
                return s.to_string();
            }
            start = i + 1;
        }
    }
    if start < inner.len() && !append_repaired_member(&mut out, &inner[start..], &mut first) {
        return s.to_string();
    }
    out.push('}');
    out
}

fn append_repaired_member(out: &mut String, member: &str, first: &mut bool) -> bool {
    let m = member.trim();
    if m.is_empty() {
        return true;
    }
    let colon = match m.find(':') {
        Some(c) => c,
        None => return false,
    };
    let key = m[..colon].trim();
    let val = m[colon + 1..].trim();
    if key.is_empty() || val.is_empty() {
        return false;
    }
    let key_quoted = if key.starts_with('"') && key.ends_with('"') {
        key.to_string()
    } else if key
        .bytes()
        .all(|c| c.is_ascii_alphanumeric() || c == b'_' || c == b'-')
    {
        format!("\"{key}\"")
    } else {
        return false;
    };
    let val_quoted = if val.starts_with('"')
        || val.starts_with('{')
        || val.starts_with('[')
        || val == "true"
        || val == "false"
        || val == "null"
        || val.parse::<f64>().is_ok()
    {
        val.to_string()
    } else {
        let escaped = val.replace('"', "\\\"");
        format!("\"{escaped}\"")
    };
    if !*first {
        out.push(',');
    }
    *first = false;
    out.push_str(&key_quoted);
    out.push(':');
    out.push_str(&val_quoted);
    true
}

/// 2026-09-26: Normalize a model-emitted function name to the client-visible
/// tool name: trim quotes, drop a `name=` prefix, cut at the first `=`
/// (`Bash=Bash` becomes `Bash`, `name="Write"` becomes `Write`), and strip a
/// namespace before the last `:` (`namespace:tool` becomes `tool`) when the
/// part before it is non-empty namespace text and the part after it is a
/// plain name. `.` is kept; only `:` separates a namespace.
pub(super) fn normalize_tool_name(raw: &str) -> String {
    let mut name = raw.trim().trim_matches('"').trim_matches('\'').to_string();

    if name.starts_with("name=") || name.starts_with("name =") {
        name = name
            .trim_start_matches("name")
            .trim_start_matches('=')
            .trim()
            .trim_matches('"')
            .trim_matches('\'')
            .to_string();
    }

    if let Some(eq_pos) = name.find('=') {
        name = name[..eq_pos].trim().to_string();
    }

    if let Some(colon) = name.rfind(':') {
        let head = &name[..colon];
        let tail = &name[colon + 1..];
        let head_is_namespace =
            !head.is_empty() && head.chars().all(is_tool_name_or_namespace_char);
        if head_is_namespace && is_tool_name_component(tail) {
            name = tail.to_string();
        }
    }

    name
}
