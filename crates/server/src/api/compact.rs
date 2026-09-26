// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Chat context compaction by context fill, and the OpenAI-shaped
//! error helpers shared by the API handlers.
//!
//! Owner: server API.
//! Invariants: none beyond the types.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};

/// 2026-09-26: Shrink a chat by how full the context is,
/// `ratio = prompt_tokens / max_seq_len`. Message 0 is treated as the system
/// message and always kept. "Middle" means neither message 0 nor one of the
/// last 6.
/// - Below 0.80 (stage 2): middle content over 500 bytes with more than 6
///   lines keeps its first and last 3 lines.
/// - 0.80 to 0.85 (stage 3): middle `tool` or `user` content over 200 bytes
///   becomes `[Tool output truncated — N chars]`.
/// - 0.85 to 0.95 (stage 4): keep message 0 and a tail of the last 6
///   messages, adjusted by the tail rules in the body.
/// - 0.95 and above (stage 5): the same with the last 4; a system
///   content over 4000 bytes keeps its first 2000 and last 1000 bytes.
///
/// `chat/template.rs` calls it only above 70% of `max_seq_len` and with more
/// than 4 messages; message 0 must exist.
pub fn compact_messages(
    msgs: &[serde_json::Value],
    prompt_tokens: usize,
    max_seq_len: usize,
) -> Vec<serde_json::Value> {
    let ratio = prompt_tokens as f32 / max_seq_len as f32;

    let (result, stage) = if ratio < 0.80 {
        let keep_tail = 6.min(msgs.len());
        let tail_start = msgs.len().saturating_sub(keep_tail);
        let mut out = Vec::with_capacity(msgs.len());
        for (i, msg) in msgs.iter().enumerate() {
            if i == 0 || i >= tail_start {
                out.push(msg.clone());
            } else {
                let content = msg["content"].as_str().unwrap_or("");
                if content.len() > 500 {
                    let lines: Vec<&str> = content.lines().collect();
                    let truncated = if lines.len() > 6 {
                        format!(
                            "{}\n... [{} lines truncated] ...\n{}",
                            lines[..3].join("\n"),
                            lines.len() - 6,
                            lines[lines.len() - 3..].join("\n")
                        )
                    } else {
                        content.to_string()
                    };
                    let mut m = msg.clone();
                    m["content"] = serde_json::Value::String(truncated);
                    out.push(m);
                } else {
                    out.push(msg.clone());
                }
            }
        }
        (out, 2)
    } else if ratio < 0.85 {
        let keep_tail = 6.min(msgs.len());
        let tail_start = msgs.len().saturating_sub(keep_tail);
        let mut out = Vec::with_capacity(msgs.len());
        for (i, msg) in msgs.iter().enumerate() {
            if i == 0 || i >= tail_start {
                out.push(msg.clone());
            } else {
                let role = msg["role"].as_str().unwrap_or("");
                let content = msg["content"].as_str().unwrap_or("");
                if (role == "tool" || role == "user") && content.len() > 200 {
                    let mut m = msg.clone();
                    m["content"] = serde_json::Value::String(format!(
                        "[Tool output truncated — {} chars]",
                        content.len()
                    ));
                    out.push(m);
                } else {
                    out.push(msg.clone());
                }
            }
        }
        (out, 3)
    } else if ratio < 0.95 {
        // 2026-09-26: The kept tail must hold a user message that is not a
        // `<tool_response>` (the Qwen templates raise "No user query found in
        // messages." otherwise), and must not open with a `tool` message cut
        // off from its assistant call.
        let keep_tail = 6.min(msgs.len().saturating_sub(1));
        let mut tail_start = msgs.len().saturating_sub(keep_tail);
        let has_user_query = (tail_start..msgs.len()).any(|i| {
            let role = msgs[i]["role"].as_str().unwrap_or("");
            let content = msgs[i]["content"].as_str().unwrap_or("");
            role == "user" && !content.starts_with("<tool_response>")
        });
        if !has_user_query {
            while tail_start > 1 {
                tail_start -= 1;
                let role = msgs[tail_start]["role"].as_str().unwrap_or("");
                let content = msgs[tail_start]["content"].as_str().unwrap_or("");
                if role == "user" && !content.starts_with("<tool_response>") {
                    break;
                }
            }
        }
        while tail_start < msgs.len() && msgs[tail_start]["role"].as_str() == Some("tool") {
            tail_start += 1;
        }
        let mut out = Vec::with_capacity(msgs.len() - tail_start + 1);
        out.push(msgs[0].clone());
        for msg in &msgs[tail_start..] {
            out.push(msg.clone());
        }
        (out, 4)
    } else {
        // 2026-09-26: Same tail rules as stage 4.
        let keep_tail = 4.min(msgs.len().saturating_sub(1));
        let mut tail_start = msgs.len().saturating_sub(keep_tail);
        let has_user_query = (tail_start..msgs.len()).any(|i| {
            let role = msgs[i]["role"].as_str().unwrap_or("");
            let content = msgs[i]["content"].as_str().unwrap_or("");
            role == "user" && !content.starts_with("<tool_response>")
        });
        if !has_user_query {
            while tail_start > 1 {
                tail_start -= 1;
                let role = msgs[tail_start]["role"].as_str().unwrap_or("");
                let content = msgs[tail_start]["content"].as_str().unwrap_or("");
                if role == "user" && !content.starts_with("<tool_response>") {
                    break;
                }
            }
        }
        while tail_start < msgs.len() && msgs[tail_start]["role"].as_str() == Some("tool") {
            tail_start += 1;
        }
        let mut out = Vec::with_capacity(msgs.len() - tail_start + 1);
        // 2026-09-26: The cut points are moved to char boundaries, so slicing
        // cannot panic on multi-byte UTF-8.
        let sys_content = msgs[0]["content"].as_str().unwrap_or("");
        let trimmed_sys = if sys_content.len() > 4000 {
            let head_end = sys_content.floor_char_boundary(2000);
            let tail_start = sys_content.ceil_char_boundary(sys_content.len().saturating_sub(1000));
            format!(
                "{}...\n[System prompt truncated — {} chars removed]\n...{}",
                &sys_content[..head_end],
                sys_content.len() - head_end - (sys_content.len() - tail_start),
                &sys_content[tail_start..]
            )
        } else {
            sys_content.to_string()
        };
        let mut sys = msgs[0].clone();
        sys["content"] = serde_json::Value::String(trimmed_sys);
        out.push(sys);
        for msg in &msgs[tail_start..] {
            out.push(msg.clone());
        }
        (out, 5)
    };

    tracing::info!(
        "Auto-compact stage {}: {} → {} messages (was {:.0}% of {})",
        stage,
        msgs.len(),
        result.len(),
        ratio * 100.0,
        max_seq_len,
    );
    result
}

/// 2026-09-26: The SSE `data:` payload `{"error": "<message>"}` for a
/// mid-stream error on the `/v1/completions` stream. Built with `serde_json`
/// so that a `"`, `\` or newline in `message` is escaped and the frame stays
/// valid JSON.
pub(super) fn completion_error_frame(message: &str) -> String {
    serde_json::json!({ "error": message }).to_string()
}

pub(super) fn openai_error_response(status: StatusCode, message: String) -> Response {
    openai_error_response_with_param(status, message, None, None)
}

/// 2026-09-26: OpenAI-compatible error with optional `param` (the offending
/// request field) and `code`; `error.type` is derived from the status.
pub(super) fn openai_error_response_with_param(
    status: StatusCode,
    message: String,
    param: Option<&str>,
    code: Option<&str>,
) -> Response {
    error_body(status, message, type_for_status(status), param, code)
}

/// 2026-09-26: The same, with an explicit `error.type` instead of one derived
/// from the status code. The derived type maps every 503 to `"server_error"`;
/// a handler that knows the condition (such as `"model_not_loaded"`) names it
/// here and gets the matching hint from `error_hints::hint_for`.
pub(super) fn openai_error_response_typed(
    status: StatusCode,
    message: String,
    error_type: &str,
) -> Response {
    error_body(status, message, error_type, None, None)
}

fn type_for_status(status: StatusCode) -> &'static str {
    match status {
        StatusCode::BAD_REQUEST => "invalid_request_error",
        StatusCode::UNAUTHORIZED => "authentication_error",
        StatusCode::FORBIDDEN => "permission_error",
        StatusCode::NOT_FOUND => "not_found_error",
        StatusCode::TOO_MANY_REQUESTS => "rate_limit_exceeded",
        StatusCode::SERVICE_UNAVAILABLE => "server_error",
        _ => "server_error",
    }
}

/// 2026-09-26: Builds the error body for the helpers above. The hint is looked
/// up here from `error_type`, so a caller of these helpers cannot send a known
/// error type without its hint.
fn error_body(
    status: StatusCode,
    message: String,
    error_type: &str,
    param: Option<&str>,
    code: Option<&str>,
) -> Response {
    let hint = crate::error_hints::hint_for(error_type);
    let body = serde_json::json!({
        "error": {
            "message": crate::error_hints::message_with_hint(&message, error_type),
            "type": error_type,
            "param": param,
            "code": code,
            "hint": hint,
        }
    });
    (status, Json(body)).into_response()
}
