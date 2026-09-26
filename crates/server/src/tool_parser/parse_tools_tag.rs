// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(unused_imports, dead_code)]

//! 2026-09-26: The `<tools>…</tools>` fallback of `parse_tool_calls`.
//!
//! Owner: server (tool parsing).
//! Invariants: none beyond the types.

use super::*;

/// 2026-09-26: Parse tool calls wrapped in `<tools>…</tools>` in place of
/// `<tool_call>`; each body goes through `parse_one_call`, and an unclosed
/// last body is parsed as it stands.
pub(super) fn parse_tools_tag_calls(text: &str) -> (Option<String>, Vec<ToolCall>) {
    let mut calls = Vec::new();
    let mut content_parts = Vec::new();
    let mut rest = text;
    let mut idx = 0u32;
    loop {
        match rest.find("<tools>") {
            Some(start) => {
                let before = rest[..start].trim();
                if !before.is_empty() {
                    content_parts.push(before.to_string());
                }
                rest = &rest[start + 7..];
                match rest.find("</tools>") {
                    Some(end) => {
                        if let Some(tc) = parse_one_call(rest[..end].trim(), idx) {
                            calls.push(tc);
                            idx += 1;
                        }
                        rest = &rest[end + 8..];
                    }
                    None => {
                        if let Some(tc) = parse_one_call(rest.trim(), idx) {
                            calls.push(tc);
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
    let content = if content_parts.is_empty() {
        None
    } else {
        Some(content_parts.join("\n"))
    };
    (content, calls)
}
