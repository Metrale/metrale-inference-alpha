// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(unused_imports, dead_code)]

//! 2026-09-26: `parse_tool_calls`, the blocking tool-call parser for
//! completed output, and its ordered fallbacks for calls written without a
//! `<tool_call>` envelope.
//!
//! Owner: server (tool parsing).
//! Invariants:
//! - Text up to and including the first `</think>` is never parsed for calls
//!   and is not returned as content.

use super::*;

/// 2026-09-26: Bound the body of a `<tool_call>` that has no `</tool_call>`
/// after it. The body is cut after its last complete `</parameter>`; with
/// none, before its first `<parameter=`, which leaves only the name. A body
/// with neither is returned unchanged; `parse_one_call` handles a truncated
/// JSON body itself.
///
/// Without the bound, an unclosed parameter value would take in everything
/// to the end of the output. The streaming detector's `flush` applies the
/// same bound (streaming_flush.rs).
pub(super) fn contain_unterminated_call_tail(rest: &str) -> &str {
    if let Some(p) = rest.rfind("</parameter>") {
        &rest[..p + "</parameter>".len()]
    } else if let Some(p) = rest.find("<parameter=") {
        &rest[..p]
    } else {
        rest
    }
}

/// 2026-09-26: Parse tool calls from completed model output. Returns
/// `(content, tool_calls)`, where content is the text outside the calls.
///
/// Tries a DSML block, then `<tool_call>` blocks, then the fallbacks in
/// `parse_tool_calls_impl` in order; the first that finds a call decides the
/// result. A bare identifier inside `<tool_call>` is dropped as malformed.
/// For Poolside v1, where a bare name is the zero-argument call, callers use
/// [`parse_tool_calls_promoting_bare_names`], choosing by
/// `ToolCallParser::promotes_bare_call_names`.
pub fn parse_tool_calls(text: &str) -> (Option<String>, Vec<ToolCall>) {
    parse_tool_calls_impl(text, false)
}

/// 2026-09-26: [`parse_tool_calls`] for formats whose zero-argument call is a
/// bare tool name inside the envelope (Poolside v1).
pub fn parse_tool_calls_promoting_bare_names(text: &str) -> (Option<String>, Vec<ToolCall>) {
    parse_tool_calls_impl(text, true)
}

fn parse_tool_calls_impl(text: &str, promote_bare_names: bool) -> (Option<String>, Vec<ToolCall>) {
    // 2026-09-26: Text up to the first `</think>` is reasoning; calls written
    // there are not invocations.
    let text = if let Some(think_end) = text.find("</think>") {
        &text[think_end + 8..]
    } else {
        text
    };
    if text.contains(DSML_OPEN) {
        let parsed = parse_dsml_tool_calls(text);
        if !parsed.1.is_empty() {
            return parsed;
        }
    }
    // 2026-09-26: MiniMax envelopes (`<minimax:tool_call>`, and the
    // `<minimax:_call>` form) are rewritten to `<tool_call>` so the loop
    // below handles every envelope. The copy is made only when one appears.
    let owned_normalized: String;
    let text: &str = if text.contains("<minimax:tool_call>")
        || text.contains("</minimax:tool_call>")
        || text.contains("<minimax:_call>")
        || text.contains("</minimax:_call>")
    {
        owned_normalized = text
            .replace("<minimax:tool_call>", "<tool_call>")
            .replace("</minimax:tool_call>", "</tool_call>")
            .replace("<minimax:_call>", "<tool_call>")
            .replace("</minimax:_call>", "</tool_call>");
        owned_normalized.as_str()
    } else {
        text
    };
    let mut calls = Vec::new();
    let mut content_parts = Vec::new();
    let mut rest = text;
    let mut idx = 0u32;

    // 2026-09-26: Byte offset of the next `</tool_call>` that is not inside a
    // `<parameter=...>...</parameter>` block, so a parameter value that
    // contains the literal `</tool_call>` does not end the call.
    fn find_unescaped_tool_call_close(buf: &str) -> Option<usize> {
        let bytes = buf.as_bytes();
        let mut i = 0;
        let mut depth: i32 = 0;
        while i < bytes.len() {
            if buf[i..].starts_with("</tool_call>") && depth == 0 {
                return Some(i);
            }
            if buf[i..].starts_with("<parameter=") {
                depth += 1;
                i += "<parameter=".len();
                continue;
            }
            if buf[i..].starts_with("</parameter>") {
                if depth > 0 {
                    depth -= 1;
                }
                i += "</parameter>".len();
                continue;
            }
            let step = buf[i..].chars().next().map(|c| c.len_utf8()).unwrap_or(1);
            i += step;
        }
        None
    }

    loop {
        match rest.find("<tool_call>") {
            Some(start) => {
                let before = rest[..start].trim();
                if !before.is_empty() {
                    content_parts.push(before.to_string());
                }
                rest = &rest[start + 11..];
                match find_unescaped_tool_call_close(rest) {
                    Some(end) => {
                        if let Some(tc) = parse_complete_call(&rest[..end], idx, promote_bare_names)
                        {
                            calls.push(tc);
                            idx += 1;
                        }
                        rest = &rest[end + 12..];
                    }
                    None => {
                        // 2026-09-26: No `</tool_call>` follows: the output
                        // ended inside the call. The bounded body
                        // (`contain_unterminated_call_tail`) is parsed; a
                        // Poolside `<arg_key>` body, or one that does not
                        // parse, is kept as content.
                        let contained = contain_unterminated_call_tail(rest);
                        if rest.contains("<arg_key>") {
                            content_parts.push(format!("<tool_call>{rest}"));
                        } else if let Some(tc) = parse_one_call(contained.trim(), idx) {
                            calls.push(tc);
                        } else {
                            content_parts.push(format!("<tool_call>{rest}"));
                        }
                        break;
                    }
                }
            }
            None => {
                let after = rest.trim();
                if !after.is_empty() {
                    content_parts.push(after.to_string());
                }
                break;
            }
        }
    }
    // 2026-09-26: Gemma-4 output that starts with `fn_name{key:val,...}`:
    // with tools active, `jinja-templates/gemma4.jinja` ends the generation
    // prompt with `<|tool_call>call:` (unless the last message is a tool
    // response), so the output continues from there. The object is
    // converted by `gemma4_to_json` before the JSON parse. Only text that
    // starts with an identifier directly followed by `{` matches; text after
    // the balanced `}` is dropped.
    if calls.is_empty() {
        let trimmed = text.trim_start();
        let id_end = trimmed
            .find(|c: char| !is_tool_name_or_namespace_char(c))
            .unwrap_or(trimmed.len());
        if id_end >= 2
            && trimmed.as_bytes().get(id_end) == Some(&b'{')
            && trimmed.as_bytes()[0].is_ascii_alphabetic()
        {
            let name = normalize_tool_name(&trimmed[..id_end]);
            let args_part = &trimmed[id_end..];
            // 2026-09-26: `json:` / `tool_call:` prose keeps its colon
            // through normalization; skipping it leaves the text for the JSON
            // fallback (`parse_json_fallback_calls`) below.
            if is_normalized_tool_name(&name)
                && let Some(end_rel) = find_balanced_json_end(args_part)
            {
                let json_slice = &args_part[..end_rel];
                let converted = gemma4_to_json(json_slice);
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&converted)
                    && v.is_object()
                {
                    let args = serde_json::to_string(&v).unwrap_or_else(|_| "{}".into());
                    calls.push(ToolCall {
                        id: next_tool_call_id(),
                        call_type: "function".into(),
                        function: FunctionCall {
                            name,
                            arguments: args,
                        },
                    });
                    return (None, calls);
                }
            }
        }
    }

    // 2026-09-26: Mistral native `[TOOL_CALLS]name[ARGS]{json}`.
    if calls.is_empty() && text.contains(MISTRAL_TOOL_CALLS_TAG) {
        let (m_content, m_calls) = parse_mistral_native_calls(text);
        if !m_calls.is_empty() {
            return (m_content, m_calls);
        }
    }

    // 2026-09-26: `identifier{json}` anywhere in the text, e.g.
    // `...for you.get_weather{"city": "Paris"}`.
    if calls.is_empty() {
        let (bc_content, bc_calls) = parse_bare_identifier_json_calls(text);
        if !bc_calls.is_empty() {
            return (bc_content, bc_calls);
        }
    }

    // 2026-09-26: Gemma-4 native `<|tool_call>call:fn{...}<tool_call|>`.
    if calls.is_empty() {
        let mut g4_rest = text;
        let mut g4_content = Vec::new();
        loop {
            match g4_rest.find("<|tool_call>") {
                Some(start) => {
                    let before = g4_rest[..start].trim();
                    if !before.is_empty() {
                        g4_content.push(before.to_string());
                    }
                    g4_rest = &g4_rest[start + 12..];
                    let end = g4_rest.find("<tool_call|>").unwrap_or(g4_rest.len());
                    let inner = g4_rest[..end].trim();
                    if let Some(tc) = parse_gemma4_native_call(inner) {
                        calls.push(tc);
                    }
                    g4_rest = if end < g4_rest.len() {
                        &g4_rest[end + 12..]
                    } else {
                        ""
                    };
                }
                None => {
                    let after = g4_rest.trim();
                    if !after.is_empty() {
                        g4_content.push(after.to_string());
                    }
                    break;
                }
            }
        }
        if !calls.is_empty() {
            let content = if g4_content.is_empty() {
                None
            } else {
                Some(g4_content.join("\n"))
            };
            return (content, calls);
        }
    }

    // 2026-09-26: A `<tools>JSON</tools>` wrapper in place of `<tool_call>`.
    if calls.is_empty() {
        let (tools_content, tools_calls) = parse_tools_tag_calls(text);
        if !tools_calls.is_empty() {
            return (tools_content, tools_calls);
        }
    }

    // 2026-09-26: Gemma-4 `call:fn{...}` lines without the `<|tool_call>`
    // wrapper.
    if calls.is_empty() {
        let trimmed = text.trim();
        if trimmed.starts_with("call:")
            || trimmed.starts_with("_call:")
            || trimmed.contains("\ncall:")
            || trimmed.contains("\n_call:")
        {
            let mut bare_content = Vec::new();
            for line in trimmed.split('\n') {
                let line = line.trim();
                if line.starts_with("call:") || line.starts_with("_call:") {
                    if let Some(tc) = parse_gemma4_native_call(line) {
                        calls.push(tc);
                    }
                } else if !line.is_empty() {
                    bare_content.push(line.to_string());
                }
            }
            if !calls.is_empty() {
                let content = if bare_content.is_empty() {
                    None
                } else {
                    Some(bare_content.join("\n"))
                };
                return (content, calls);
            }
        }
    }

    // 2026-09-26: `<function>` or `<function=` blocks without a
    // `<tool_call>` wrapper (`parse_bare_function_calls`).
    if calls.is_empty() {
        let (bare_content, bare_calls) = parse_bare_function_calls(text);
        if !bare_calls.is_empty() {
            return (bare_content, bare_calls);
        }
    }

    // 2026-09-26: MiniMax-style `<invoke name="X">…</invoke>` blocks with no
    // envelope, parsed by `parse_minimax_xml_calls_all`, the function the
    // streaming detector uses for a MiniMax envelope's body.
    if calls.is_empty() && text.contains("<invoke name=") {
        let bare_invoke_calls = super::parse_minimax_xml_calls_all(text);
        if !bare_invoke_calls.is_empty() {
            // 2026-09-26: The parsed blocks are removed from content, so
            // they are not returned twice.
            let mut clean = text.to_string();
            let mut search = 0usize;
            while let Some(rel) = clean[search..].find("<invoke name=") {
                let start = search + rel;
                match clean[start..].find("</invoke>") {
                    Some(e) => {
                        let end = start + e + "</invoke>".len();
                        clean.drain(start..end);
                        search = start;
                    }
                    None => break,
                }
            }
            let trimmed = clean.trim();
            let content = if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            };
            return (content, bare_invoke_calls);
        }
    }

    // 2026-09-26: JSON tool calls in code blocks, on their own line, or
    // embedded in prose (`parse_json_fallback_calls`).
    if calls.is_empty() {
        let json_calls = parse_json_fallback_calls(text);
        if !json_calls.is_empty() {
            // 2026-09-26: Code blocks are removed from content; JSON found
            // outside a code block stays in it.
            let mut clean_content = text.to_string();
            for pattern in extract_json_code_blocks(text) {
                clean_content = clean_content.replace(&pattern, "");
            }
            let clean = clean_content.trim().to_string();
            let content = if clean.is_empty() { None } else { Some(clean) };
            return (content, json_calls);
        }
    }

    let content = if content_parts.is_empty() {
        None
    } else {
        Some(content_parts.join("\n"))
    };
    (content, calls)
}
