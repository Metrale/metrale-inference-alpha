// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(unused_imports, dead_code)]

//! 2026-09-26: Parsers for XML call bodies: qwen3_coder
//! `<function=NAME><parameter=KEY>VALUE</parameter></function>`, and the
//! tag-style `<function>NAME</function><parameters>…</parameters>`.
//!
//! Owner: server (tool parsing).
//! Invariants:
//! - `parse_qwen3_coder_call` never returns a `_think` argument.

use super::*;

/// 2026-09-26: Parse a qwen3_coder call body. The opener may be written
/// `<function=`, `<|function=`, `<function ` or `<|function `;
/// `<parameter=` values are returned as JSON strings.
pub(super) fn parse_qwen3_coder_call(text: &str, _idx: u32) -> Option<ToolCall> {
    let (func_start, prefix_len) = if let Some(pos) = text.find("<function=") {
        (pos, "<function=".len())
    } else if let Some(pos) = text.find("<|function=") {
        (pos, "<|function=".len())
    } else if let Some(pos) = text.find("<function ") {
        (pos, "<function ".len())
    } else if let Some(pos) = text.find("<|function ") {
        (pos, "<|function ".len())
    } else {
        return None;
    };
    let name_start = func_start + prefix_len;
    // 2026-09-26: The name ends at the first `>`, newline or `<`, so an
    // opener missing its `>` (`<function=bash\n<parameter=...>`) still parses.
    let name_end = name_start
        + text[name_start..]
            .find(['>', '\n', '<'])
            .unwrap_or(text[name_start..].len());
    let func_name = normalize_tool_name(&text[name_start..name_end]);
    if func_name.is_empty() {
        return None;
    }

    let mut args = serde_json::Map::new();
    let after_name = if name_end < text.len() {
        &text[name_end + 1..]
    } else {
        ""
    };
    let mut rest = after_name;

    while let Some(p) = rest.find("<parameter=") {
        // 2026-09-26: A `</function>` before the next `<parameter=` ends this
        // call; later parameters belong to a following `<function=...>`
        // block.
        if let Some(fc) = rest.find("</function>")
            && fc < p
        {
            break;
        }
        rest = &rest[p + "<parameter=".len()..];
        let key_end = rest.find('>')?;
        let param_name = rest[..key_end].trim().to_string();
        rest = &rest[key_end + 1..];

        // 2026-09-26: The value ends at the earliest of `</parameter>`, the
        // next `<parameter=` and `</function>`, so a value missing its
        // `</parameter>` does not swallow the parameters after it.
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
        // 2026-09-26: When the value ended at `<parameter=` or `</function>`,
        // a trailing `</parameter` (a close missing its `>`) is removed.
        let raw_value = if !consumed_close {
            raw_value
                .strip_suffix("</parameter")
                .map(str::trim_end)
                .unwrap_or(raw_value)
        } else {
            raw_value
        };
        let advanced_to_func_close =
            !consumed_close && val_end < rest.len() && rest[val_end..].starts_with("</function>");
        rest = if consumed_close {
            &rest[val_end + "</parameter>".len()..]
        } else if val_end < rest.len() {
            // 2026-09-26: The next `<parameter=` or `</function>` stays in
            // `rest` for the loop.
            &rest[val_end..]
        } else {
            ""
        };

        // 2026-09-26: Values stay strings here. `coerce_all` converts them to
        // the schema's types afterwards for parsers that ask for it
        // (`ToolCallParser::wants_typed_arguments`).
        args.insert(param_name, serde_json::Value::String(raw_value.to_string()));

        // 2026-09-26: A value ended by `</function>` is this call's last.
        if advanced_to_func_close {
            break;
        }
    }

    // 2026-09-26: With no `<parameter=` values, a JSON object between the
    // name and `</function>` (`<function=Bash>{"command":"ls"}</function>`)
    // is used as the arguments.
    if args.is_empty() {
        let body = after_name
            .find("</function>")
            .map(|end| after_name[..end].trim())
            .unwrap_or(after_name.trim());
        if body.starts_with('{')
            && let Ok(json_args) =
                serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(body)
        {
            args = json_args;
        }
    }

    // 2026-09-26: `_think` is the rationale field
    // `grammar::augment_schema_with_tafc_think` adds to a schema; it is
    // removed so it never reaches the client's tool.
    args.remove("_think");

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

/// 2026-09-26: Parse a tag-style call, `<function>NAME</function>` with an
/// optional `<parameters>` block. A name holding nested tags is taken from
/// its `<name>` element.
pub(super) fn parse_tag_style_call(text: &str, _idx: u32) -> Option<ToolCall> {
    let func_start = text.find("<function>")?;
    let after_tag = &text[func_start + "<function>".len()..];
    let close = after_tag.find("</function>")?;
    let raw_name = after_tag[..close].trim();

    let func_name = if raw_name.contains('<') {
        normalize_tool_name(&extract_tag_value(raw_name, "name")?)
    } else if raw_name.is_empty() {
        return None;
    } else {
        normalize_tool_name(raw_name)
    };

    let mut args = serde_json::Map::new();
    let rest = &after_tag[close + "</function>".len()..];

    if let Some(ps) = rest.find("<parameters>") {
        let params_inner = &rest[ps + "<parameters>".len()..];
        let pe = params_inner
            .find("</parameters>")
            .unwrap_or(params_inner.len());
        extract_params_from_block(&params_inner[..pe], &mut args);
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

/// 2026-09-26: The trimmed text between the first `<tag>` and the next
/// `</tag>`; `None` when either is missing or the text is empty.
fn extract_tag_value(text: &str, tag: &str) -> Option<String> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let start = text.find(&open)? + open.len();
    let end = start + text[start..].find(&close)?;
    let val = text[start..end].trim();
    if val.is_empty() {
        None
    } else {
        Some(val.to_string())
    }
}

/// 2026-09-26: Add the name/value pairs of a `<parameters>` block to `args`.
/// A value that parses as JSON is stored as that JSON, otherwise as a string.
///
/// Two layouts:
/// - `<parameter><name>N</name><value>V</value></parameter>` children, any
///   number;
/// - when there are none, one `<name>N</name>...<value>V</value>` pair.
fn extract_params_from_block(block: &str, args: &mut serde_json::Map<String, serde_json::Value>) {
    let mut rest = block;
    let mut found_any = false;
    while let Some(ps) = rest.find("<parameter>") {
        rest = &rest[ps + "<parameter>".len()..];
        let pe = rest.find("</parameter>").unwrap_or(rest.len());
        let param_block = &rest[..pe];
        if let (Some(name), Some(value)) = (
            extract_tag_value(param_block, "name"),
            extract_tag_value(param_block, "value"),
        ) {
            let json_val = serde_json::from_str::<serde_json::Value>(&value)
                .unwrap_or(serde_json::Value::String(value));
            args.insert(name, json_val);
            found_any = true;
        }
        rest = if pe < rest.len() {
            &rest[pe + "</parameter>".len()..]
        } else {
            ""
        };
    }
    if found_any {
        return;
    }

    if let (Some(name), Some(value)) = (
        extract_tag_value(block, "name"),
        extract_tag_value(block, "value"),
    ) {
        let json_val = serde_json::from_str::<serde_json::Value>(&value)
            .unwrap_or(serde_json::Value::String(value));
        args.insert(name, json_val);
    }
}

/// 2026-09-26: For the streaming detector: the byte offset just past a bare
/// `<function…>` block at the start of `text`, or `None` while the block is
/// incomplete (and when `text` does not start with a `<function` opener).
pub(super) fn bare_function_end(text: &str) -> Option<usize> {
    if text.starts_with("<function>") {
        // 2026-09-26: Tag-style: the block ends at `</parameters>` when there
        // is one.
        if let Some(p) = text.find("</parameters>") {
            return Some(p + "</parameters>".len());
        }
        // 2026-09-26: `<parameters>` opened but not yet closed: wait. The
        // name's `</function>` comes before `<parameters>`, so ending the
        // block there would cut off its parameters, and the rest would reach
        // the client as content.
        if text.contains("<parameters>") {
            return None;
        }
        if let Some(p) = text.find("</function>") {
            let after = p + "</function>".len();
            // 2026-09-26: A second `</function>` after the name's close is
            // part of the block.
            if let Some(p2) = text[after..].find("</function>") {
                return Some(after + p2 + "</function>".len());
            }
            return Some(after);
        }
    } else if text.starts_with("<function=") || text.starts_with("<function ") {
        // 2026-09-26: Attribute-style `<function=NAME>` or `<function NAME>`:
        // the block ends at the first `</function>`.
        if let Some(p) = text.find("</function>") {
            return Some(p + "</function>".len());
        }
    }
    None
}
