// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tool-call descriptors: the tool family of a tool name, the primary argument
//! of a call, and the last command of a shell chain.
//!
//! Owner: server API (sanitizer).
//! Invariants:
//! - Every truncated string is cut on a UTF-8 char boundary.

use super::*;

/// 2026-09-26: The last command of a shell chain. Splits on each `&`, `|`, `;` and
/// newline, drops empty pieces and `cd` pieces, and takes the last one left (the
/// whole `command` when none is left). The result is cut to at most
/// `F7_BASH_COMMAND_PREFIX_LEN` bytes.
pub fn extract_bash_final_action(command: &str) -> String {
    let parts: Vec<&str> = command
        .split(['&', '|', ';', '\n'])
        .map(|s| s.trim())
        .filter(|s| !s.is_empty() && !s.starts_with("cd ") && !s.starts_with("cd\t") && *s != "cd")
        .collect();
    let action = parts.last().copied().unwrap_or(command);
    let n = action.len().min(F7_BASH_COMMAND_PREFIX_LEN);
    let mut cut = n;
    while cut > 0 && !action.is_char_boundary(cut) {
        cut -= 1;
    }
    action[..cut].to_string()
}

/// 2026-09-26: Tool family of a tool name, matched ignoring ASCII case (`bash`, `Bash` and
/// `BASH` are all `Bash`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolKind {
    Bash,
    Write,
    Edit,
    Read,
    MultiEdit,
    Other,
}

pub fn classify_tool(name: &str) -> ToolKind {
    if name.eq_ignore_ascii_case("Bash") {
        ToolKind::Bash
    } else if name.eq_ignore_ascii_case("Write") {
        ToolKind::Write
    } else if name.eq_ignore_ascii_case("Edit") {
        ToolKind::Edit
    } else if name.eq_ignore_ascii_case("Read") {
        ToolKind::Read
    } else if name.eq_ignore_ascii_case("MultiEdit") {
        ToolKind::MultiEdit
    } else {
        ToolKind::Other
    }
}

/// 2026-09-26: The primary argument of a tool call, from its JSON arguments:
/// - `Write`, `Edit`, `Read`, `MultiEdit`: `file_path`, else `filePath`;
/// - `Bash`: [`extract_bash_final_action`] of `command`;
/// - any other tool: `key=value` for the first non-empty string field in key order,
///   with the value cut to at most `F7_OTHER_ARG_FALLBACK_LEN` bytes.
///
/// `None` when the arguments are not a JSON object, or the chosen key is missing or not
/// a string, or (other tools) no field is a non-empty string.
pub fn primary_arg_for_tool(name: &str, args_json: &str) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(args_json).ok()?;
    let obj = v.as_object()?;
    let kind = classify_tool(name);
    let key_for_well_known = match kind {
        ToolKind::Write | ToolKind::Edit | ToolKind::Read | ToolKind::MultiEdit => {
            if obj.get("file_path").and_then(|v| v.as_str()).is_some() {
                Some("file_path")
            } else if obj.get("filePath").and_then(|v| v.as_str()).is_some() {
                Some("filePath")
            } else {
                Some("file_path")
            }
        }
        ToolKind::Bash => Some("command"),
        ToolKind::Other => None,
    };
    if let Some(k) = key_for_well_known {
        let val = obj.get(k).and_then(|v| v.as_str())?;
        if matches!(kind, ToolKind::Bash) {
            return Some(extract_bash_final_action(val));
        }
        return Some(val.to_string());
    }
    let mut keys: Vec<&String> = obj.keys().collect();
    keys.sort();
    for k in keys {
        if let Some(s) = obj.get(k).and_then(|v| v.as_str())
            && !s.is_empty()
        {
            let n = s.len().min(F7_OTHER_ARG_FALLBACK_LEN);
            let mut cut = n;
            while cut > 0 && !s.is_char_boundary(cut) {
                cut -= 1;
            }
            return Some(format!("{}={}", k, &s[..cut]));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::{ToolKind, classify_tool, extract_bash_final_action, primary_arg_for_tool};

    #[test]
    fn bash_final_action_returns_last_segment() {
        let out =
            extract_bash_final_action("mkdir -p /tmp/x/src && cd /tmp/x && cargo init --name a");
        assert!(out.starts_with("cargo init"), "got: {out}");
    }

    #[test]
    fn bash_final_action_no_chain_returns_original() {
        let out = extract_bash_final_action("ls -la /tmp/x");
        assert!(out.starts_with("ls -la"));
    }

    #[test]
    fn bash_final_action_empty_returns_empty() {
        assert_eq!(extract_bash_final_action(""), "");
    }

    #[test]
    fn classify_tool_case_insensitive() {
        assert_eq!(classify_tool("Bash"), ToolKind::Bash);
        assert_eq!(classify_tool("bash"), ToolKind::Bash);
        assert_eq!(classify_tool("BASH"), ToolKind::Bash);
        assert_eq!(classify_tool("Write"), ToolKind::Write);
        assert_eq!(classify_tool("Edit"), ToolKind::Edit);
        assert_eq!(classify_tool("Read"), ToolKind::Read);
        assert_eq!(classify_tool("MultiEdit"), ToolKind::MultiEdit);
        assert_eq!(classify_tool("multiedit"), ToolKind::MultiEdit);
    }

    #[test]
    fn classify_tool_unknown_is_other() {
        assert_eq!(classify_tool("GetWeather"), ToolKind::Other);
        assert_eq!(classify_tool(""), ToolKind::Other);
        assert_eq!(classify_tool("Bashly"), ToolKind::Other);
    }

    #[test]
    fn primary_arg_write_snake_and_camel() {
        let out = primary_arg_for_tool("Write", r#"{"file_path":"/tmp/x.rs"}"#);
        assert_eq!(out.as_deref(), Some("/tmp/x.rs"));
        let out = primary_arg_for_tool("write", r#"{"filePath":"/tmp/y.rs"}"#);
        assert_eq!(out.as_deref(), Some("/tmp/y.rs"));
    }

    #[test]
    fn primary_arg_bash_collapses_chain() {
        let out = primary_arg_for_tool("Bash", r#"{"command":"cd /tmp && cargo build"}"#);
        assert!(out.as_ref().is_some_and(|s| s.starts_with("cargo build")));
    }

    #[test]
    fn primary_arg_unknown_tool_falls_back() {
        let out = primary_arg_for_tool("GetWeather", r#"{"location":"Paris"}"#);
        assert!(
            out.is_some(),
            "fallback path should return some(location=Paris)"
        );
    }

    #[test]
    fn primary_arg_malformed_json_returns_none() {
        assert_eq!(primary_arg_for_tool("Write", "not json"), None);
    }

    #[test]
    fn primary_arg_missing_key_returns_none() {
        let out = primary_arg_for_tool("Write", r#"{"content":"fn main(){}"}"#);
        assert_eq!(out, None);
    }
}
