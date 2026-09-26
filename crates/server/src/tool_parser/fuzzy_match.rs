// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Fuzzy tool-name repair. `validate_tool_calls`
//! (validation.rs) calls it for a name that matches no tool exactly; with no
//! match, `assess_tool_call` then reports `Unknown tool`.
//!
//! Owner: server (tool parsing).
//! Invariants:
//! - A name is returned only when exactly one tool satisfies the strategy
//!   that returns it; an ambiguous strategy falls through to the next.

use super::ToolDefinition;

/// 2026-09-26: Map a tool name that matches no tool to one that does.
///
/// Strategies, in order:
/// 0. Equal after `normalize_tool_name` (case, and runs of `_`/`-`), so
///    `mcp_discord__discord_send` finds `mcp__discord__discord_send`.
/// 1. The model's name is a case-insensitive substring of one tool name.
/// 2. One tool name is a case-insensitive substring of the model's name.
/// 3. The request has exactly one tool.
pub(super) fn fuzzy_match_tool_name(model_name: &str, tools: &[ToolDefinition]) -> Option<String> {
    if tools.is_empty() || model_name.is_empty() {
        return None;
    }

    let lower = model_name.to_lowercase();

    // 2026-09-26: A dropped or doubled `_` makes neither name a substring of
    // the other, so strategies 1 and 2 cannot find these.
    let norm_model = normalize_tool_name(model_name);
    if !norm_model.is_empty() {
        let exact_norm: Vec<&str> = tools
            .iter()
            .filter(|t| normalize_tool_name(&t.function.name) == norm_model)
            .map(|t| t.function.name.as_str())
            .collect();
        if exact_norm.len() == 1 {
            return Some(exact_norm[0].to_string());
        }
    }

    let matches: Vec<&str> = tools
        .iter()
        .filter(|t| t.function.name.to_lowercase().contains(&lower))
        .map(|t| t.function.name.as_str())
        .collect();
    if matches.len() == 1 {
        return Some(matches[0].to_string());
    }

    let matches: Vec<&str> = tools
        .iter()
        .filter(|t| lower.contains(&t.function.name.to_lowercase()))
        .map(|t| t.function.name.as_str())
        .collect();
    if matches.len() == 1 {
        return Some(matches[0].to_string());
    }

    if tools.len() == 1 {
        return Some(tools[0].function.name.clone());
    }

    None
}

/// 2026-09-26: Trim, lowercase, collapse each run of `_`/`-` to one `_`, and
/// trim leading and trailing `_`.
fn normalize_tool_name(s: &str) -> String {
    let lower = s.trim().to_lowercase();
    let mut out = String::with_capacity(lower.len());
    let mut prev_sep = false;
    for c in lower.chars() {
        if c == '_' || c == '-' {
            if !prev_sep {
                out.push('_');
            }
            prev_sep = true;
        } else {
            out.push(c);
            prev_sep = false;
        }
    }
    out.trim_matches('_').to_string()
}
