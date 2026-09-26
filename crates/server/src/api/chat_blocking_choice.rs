// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Per-choice assembly for the blocking `/v1/chat/completions` path:
//! tool-call parse, validate and coerce, refusal classification, and the logprobs
//! conversion.
//!
//! Owner: server chat API.
//! Invariants:
//! - A choice with tool calls has no refusal.

#![allow(clippy::too_many_arguments)]

use crate::AppState;
use crate::ir;
use crate::tool_parser;

use super::chat_blocking::{extract_hoisted_tool_calls, merge_hoisted_tool_calls};

/// 2026-09-26: Build the assistant message and finish reason for one choice: tool
/// parsing and validation, content stripping, and the refusal classifier. It awaits
/// nothing, so it is not `async`. `matched_stop` and `logprobs` are left `None` for the
/// caller.
pub(super) fn build_choice_message(
    state: &AppState,
    req: &crate::ir::ChatRequest,
    response: &super::inference_types::InferenceResponse,
    reasoning_content_i: Option<String>,
    output_text_i: String,
    tools_active: bool,
    cwd_hint: Option<&str>,
    choice_idx: usize,
) -> ir::Choice {
    let _ = response;
    let mut reasoning_content = reasoning_content_i;
    let mut msg_content: Option<String> = Some(output_text_i.clone());
    let mut msg_tool_calls: Option<Vec<tool_parser::ToolCall>> = None;
    let mut msg_refusal: Option<String> = None;
    let mut finish_reason_i = response.finish_reason.clone();

    if tools_active {
        if std::env::var("METRALE_LOG_TOOL_RAW").as_deref() == Ok("1") {
            tracing::info!(
                target: "metrale::tool_debug",
                "raw pre-parse output (tools_active, choice {choice_idx}): {output_text_i:?}"
            );
        }
        // 2026-09-26: A tool call emitted inside the think block lands in the reasoning,
        // where the content parser below does not look. Parse the reasoning too; when it
        // holds calls, keep them and replace the reasoning with what is left of it.
        let parser_name = state.tool_call_parser.as_ref().map(|parser| parser.name());
        let (hoisted_reasoning, hoisted_tool_calls) =
            extract_hoisted_tool_calls(reasoning_content.as_deref(), parser_name);
        if !hoisted_tool_calls.is_empty() {
            tracing::info!(
                "F7: hoisted {} tool-call(s) from inside <think> block (would have been silently dropped)",
                hoisted_tool_calls.len()
            );
            reasoning_content = hoisted_reasoning;
        }
        let promote_bare_names = state
            .tool_call_parser
            .as_ref()
            .is_some_and(|p| p.promotes_bare_call_names());
        let (content, parsed_tool_calls) = if promote_bare_names {
            tool_parser::parse_tool_calls_promoting_bare_names(&output_text_i)
        } else {
            tool_parser::parse_tool_calls(&output_text_i)
        };
        let mut tool_calls_i = merge_hoisted_tool_calls(hoisted_tool_calls, parsed_tool_calls);
        if !tool_calls_i.is_empty() {
            let tools_ref = req.tools.clone();
            tool_parser::backfill_required_params(&mut tool_calls_i, &tools_ref);
            if state
                .tool_call_parser
                .as_ref()
                .is_some_and(|p| p.wants_typed_arguments())
            {
                tool_parser::coerce_all(&mut tool_calls_i, &tools_ref);
            }
            if let Some(cwd) = cwd_hint {
                tool_parser::normalize_paths(&mut tool_calls_i, cwd);
            }
            let validated = tool_parser::validate_tool_calls(tool_calls_i, &tools_ref);
            if !validated.errors.is_empty() {
                for err in &validated.errors {
                    tracing::warn!("Tool call validation error: {err}");
                }
            }
            // 2026-09-26: Calls were parsed: remove leftover tool-call tags, `<function=…>`
            // openers and every closed ``` fenced block from the content.
            let content = content.map(|mut c| {
                for tag in &["</parameter>", "</function>", "</tool_call>", "<tool_call>"] {
                    c = c.replace(tag, "");
                }
                while let Some(start) = c.find("<function=") {
                    let end = c[start..]
                        .find('>')
                        .map(|p| start + p + 1)
                        .unwrap_or(c.len());
                    c = format!("{}{}", &c[..start], &c[end..]);
                }
                while let Some(start) = c.find("```") {
                    let after_open = start + 3;
                    let Some(rel_close) = c[after_open..].find("```") else {
                        break;
                    };
                    let close_end = after_open + rel_close + 3;
                    c = format!("{}{}", &c[..start], &c[close_end..]);
                }
                c.trim().to_string()
            });
            msg_content = content;
            if !validated.valid.is_empty() {
                for tc in &validated.valid {
                    let p: String = tc.function.arguments.chars().take(120).collect();
                    let s = ["", "…"][usize::from(tc.function.arguments.len() > p.len())];
                    tracing::info!("Tool call: {}({p}{s})", tc.function.name);
                    crate::metrics::TOOL_CALLS_TOTAL.inc();
                }
                msg_tool_calls = Some(validated.valid);
                // 2026-09-26: A deadline cut outranks "tool_calls": the turn was truncated,
                // so a call parsed out of it may be partial.
                if finish_reason_i != ir::FINISH_REASON_TIMEOUT {
                    finish_reason_i = "tool_calls".to_string();
                }
            }
        }
    }

    // 2026-09-26: No valid tool call (tools off, or nothing valid parsed): cut any
    // tool-call markup, and everything after it, from the content.
    if msg_tool_calls.is_none() {
        msg_content = msg_content.map(|c| super::strip::strip_orphan_tool_markup(&c));
    }

    // 2026-09-26: With no tool call, content that opens with a known refusal pattern
    // (`refusal::detect`) moves to `refusal` and `content` becomes `None`.
    if msg_tool_calls.is_none()
        && let Some(content_text) = msg_content.as_deref()
        && let Some(refusal_sentence) = crate::refusal::detect(content_text)
    {
        msg_refusal = Some(refusal_sentence);
        msg_content = None;
    }

    // 2026-09-26: Arguments that do not parse as JSON become an empty object.
    let tool_calls: Vec<ir::message::ToolCall> = msg_tool_calls
        .unwrap_or_default()
        .into_iter()
        .map(|tc| ir::message::ToolCall {
            id: tc.id,
            name: tc.function.name,
            arguments: serde_json::from_str(&tc.function.arguments)
                .unwrap_or_else(|_| serde_json::Value::Object(Default::default())),
        })
        .collect();

    ir::Choice {
        index: choice_idx,
        content: msg_content,
        reasoning: reasoning_content,
        tool_calls,
        refusal: msg_refusal,
        finish_reason: ir::FinishReason::from(finish_reason_i.as_str()),
        matched_stop: None,
        logprobs: None,
    }
}

/// 2026-09-26: Decode the response's per-token logprobs into `ir::ChoiceLogprobs`;
/// `None` when the response carries none.
pub(super) fn build_logprobs(
    state: &AppState,
    response: &super::inference_types::InferenceResponse,
) -> Option<ir::ChoiceLogprobs> {
    if response.logprobs.is_empty() {
        return None;
    }
    Some(ir::ChoiceLogprobs {
        content: response
            .logprobs
            .iter()
            .map(|lp| {
                let token_str = state.tokenizer.decode(&[lp.token_id]).unwrap_or_default();
                ir::TokenLogprob {
                    token: token_str,
                    logprob: lp.logprob,
                    top: lp
                        .top
                        .iter()
                        .map(|&(tid, lp_val)| {
                            (state.tokenizer.decode(&[tid]).unwrap_or_default(), lp_val)
                        })
                        .collect(),
                }
            })
            .collect(),
    })
}
