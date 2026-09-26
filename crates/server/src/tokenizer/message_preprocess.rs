// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Chat-message rewrites applied before any Jinja template renders, whichever
//! template is in use. `preprocess_for_render` in chat_impl.rs calls them:
//! - [`remap_developer_role`]: `developer` messages become `system` messages;
//! - [`autoclose_assistant_think`]: close an open `<think>` ahead of a `<tool_call>` in
//!   assistant history;
//! - [`resolve_think_control`]: strip inline `<|think_on|>` / `<|think_off|>` tokens and
//!   report the thinking setting they request.
//!
//! Owner: server (tokenizer).
//! Invariants: none beyond the types.

use serde_json::Value;

/// 2026-09-26: Inline control tokens a client may put in message content to turn thinking
/// on or off.
const THINK_ON: &str = "<|think_on|>";
const THINK_OFF: &str = "<|think_off|>";

/// 2026-09-26: Close an open `<think>` in one assistant-history content string that also
/// holds a `<tool_call>`.
///
/// The last `<think>` is open when no `</think>` follows it. Then `</think>` is inserted
/// just before the first `<tool_call>` if that call comes after the `<think>`, and appended
/// at the end otherwise. The string is returned borrowed and unchanged when it has no
/// `<tool_call>`, no `<think>`, or the last `<think>` is closed.
pub(super) fn autoclose_think_before_tool_call(content: &str) -> std::borrow::Cow<'_, str> {
    let Some(tool_pos) = content.find("<tool_call>") else {
        return std::borrow::Cow::Borrowed(content);
    };
    let Some(last_think) = content.rfind("<think>") else {
        return std::borrow::Cow::Borrowed(content);
    };
    let last_close = content.rfind("</think>");
    let think_unclosed = match last_close {
        None => true,
        Some(close) => close < last_think,
    };
    if !think_unclosed {
        return std::borrow::Cow::Borrowed(content);
    }
    if tool_pos > last_think {
        let mut out = String::with_capacity(content.len() + "</think>".len());
        out.push_str(&content[..tool_pos]);
        out.push_str("</think>");
        out.push_str(&content[tool_pos..]);
        std::borrow::Cow::Owned(out)
    } else {
        std::borrow::Cow::Owned(format!("{content}</think>"))
    }
}

/// 2026-09-26: Strip inline `<|think_on|>` / `<|think_off|>` control tokens from every
/// message's content and resolve the thinking setting they request.
///
/// Returns the rewritten messages plus `Some(on)` for the last control token across all
/// messages, or `None` when there was none (the caller keeps its own `enable_thinking`).
/// String content is scrubbed; in array content, each item's string `text` field is
/// scrubbed; anything else passes through.
pub(crate) fn resolve_think_control(messages: &[Value]) -> (Vec<Value>, Option<bool>) {
    let mut effective: Option<bool> = None;
    let out = messages
        .iter()
        .map(|msg| {
            let mut msg = msg.clone();
            if let Some(content) = msg.get_mut("content") {
                strip_controls_in_content(content, &mut effective);
            }
            msg
        })
        .collect();
    (out, effective)
}

/// 2026-09-26: Scrub control tokens out of one message's `content` (string or array of
/// parts), updating `effective` to the last toggle seen.
fn strip_controls_in_content(content: &mut Value, effective: &mut Option<bool>) {
    match content {
        Value::String(s) => {
            if let Some(stripped) = strip_controls_in_str(s, effective) {
                *s = stripped;
            }
        }
        Value::Array(items) => {
            for item in items.iter_mut() {
                if let Some(text) = item.get_mut("text").and_then(|t| match t {
                    Value::String(s) => Some(s),
                    _ => None,
                }) && let Some(stripped) = strip_controls_in_str(text, effective)
                {
                    *text = stripped;
                }
            }
        }
        _ => {}
    }
}

/// 2026-09-26: `Some(stripped)` when the string held a control token, `None` when it held
/// none. Sets `effective` from the token at the highest position.
fn strip_controls_in_str(s: &str, effective: &mut Option<bool>) -> Option<String> {
    if !s.contains(THINK_ON) && !s.contains(THINK_OFF) {
        return None;
    }
    let mut positions: Vec<(usize, bool)> = Vec::new();
    for (idx, _) in s.match_indices(THINK_OFF) {
        positions.push((idx, false));
    }
    for (idx, _) in s.match_indices(THINK_ON) {
        positions.push((idx, true));
    }
    positions.sort_by_key(|(idx, _)| *idx);
    if let Some((_, last)) = positions.last() {
        *effective = Some(*last);
    }
    let stripped = s.replace(THINK_OFF, "").replace(THINK_ON, "");
    Some(stripped)
}

/// 2026-09-26: Apply [`autoclose_think_before_tool_call`] to the string content of assistant
/// messages. Other roles and array content are left untouched.
pub(crate) fn autoclose_assistant_think(messages: &mut [Value]) {
    for msg in messages.iter_mut() {
        let is_assistant = msg.get("role").and_then(|r| r.as_str()) == Some("assistant");
        if !is_assistant {
            continue;
        }
        let Some(Value::String(content)) = msg.get_mut("content") else {
            continue;
        };
        if let std::borrow::Cow::Owned(fixed) = autoclose_think_before_tool_call(content) {
            *content = fixed;
        }
    }
}

/// 2026-09-26: Turn `developer` messages into `system` messages before the template renders.
/// The Qwen templates in test_data/chat_templates raise `Unexpected message role.` on
/// `developer` and `System message must be at the beginning.` on a later system message.
///
/// When the developer messages are the only system-level messages, each is renamed in place.
/// Otherwise the string contents of all `developer` and `system` messages are joined with a
/// blank line into one system message at the position of the first; a system-level message
/// with non-string content is renamed and kept as its own message.
pub(crate) fn remap_developer_role(messages: Vec<Value>) -> Vec<Value> {
    let role_of = |m: &Value| -> Option<String> {
        m.get("role").and_then(|r| r.as_str()).map(str::to_string)
    };
    let is_sys_level =
        |m: &Value| matches!(role_of(m).as_deref(), Some("developer") | Some("system"));

    let has_dev = messages
        .iter()
        .any(|m| role_of(m).as_deref() == Some("developer"));
    if !has_dev {
        return messages;
    }

    if messages.iter().filter(|m| is_sys_level(m)).count()
        == messages
            .iter()
            .filter(|m| role_of(m).as_deref() == Some("developer"))
            .count()
    {
        let mut messages = messages;
        for m in messages.iter_mut() {
            if role_of(m).as_deref() == Some("developer")
                && let Some(role) = m.get_mut("role")
            {
                *role = Value::String("system".to_string());
            }
        }
        return messages;
    }

    let mut merged = String::new();
    let mut out: Vec<Value> = Vec::with_capacity(messages.len());
    let mut slot: Option<usize> = None;
    for m in messages {
        if is_sys_level(&m) {
            if let Some(s) = m.get("content").and_then(|c| c.as_str()) {
                if !merged.is_empty() && !s.is_empty() {
                    merged.push_str("\n\n");
                }
                merged.push_str(s);
                if slot.is_none() {
                    slot = Some(out.len());
                    out.push(Value::Null);
                }
            } else {
                let mut m = m;
                if let Some(role) = m.get_mut("role") {
                    *role = Value::String("system".to_string());
                }
                out.push(m);
            }
        } else {
            out.push(m);
        }
    }
    if let Some(i) = slot {
        let mut sys = serde_json::Map::new();
        sys.insert("role".to_string(), Value::String("system".to_string()));
        sys.insert("content".to_string(), Value::String(merged));
        out[i] = Value::Object(sys);
    }
    out
}

#[cfg(test)]
mod tests;
