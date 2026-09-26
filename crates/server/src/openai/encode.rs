// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Encoder from `ir::ChatResponse` to the blocking OpenAI chat
//! completion JSON: the `chatcmpl-` id, URL-citation annotations, the
//! `service_tier` and `metadata` echoes, `store: true` storage, and the
//! `--dump` response capture.
//!
//! Owner: server (OpenAI adapter).
//! Invariants: none beyond the types.

use axum::response::{IntoResponse, Json, Response};

use crate::AppState;

use super::{
    ChatChoice, ChatCompletionResponse, ChatMessage, ChoiceLogprobs, TokenLogprobInfo, TopLogprob,
    Usage, merged_annotations,
};

pub(crate) fn encode_chat_response(
    state: &AppState,
    ir: crate::ir::ChatResponse,
    echo: &crate::api::ResponseEcho,
    dump_seq: Option<u64>,
) -> Response {
    let usage = Usage::from(&ir.usage);

    let choices: Vec<ChatChoice> = ir
        .choices
        .into_iter()
        .map(|c| {
            let tool_calls = if c.tool_calls.is_empty() {
                None
            } else {
                Some(
                    c.tool_calls
                        .into_iter()
                        .map(|tc| crate::tool_parser::ToolCall {
                            id: tc.id,
                            call_type: "function".to_string(),
                            function: crate::tool_parser::FunctionCall {
                                name: tc.name,
                                arguments: tc.arguments.to_string(),
                            },
                        })
                        .collect(),
                )
            };
            // 2026-09-26: Annotations come from the final content, so their
            // offsets index the text the client receives.
            let annotations = c.content.as_deref().and_then(merged_annotations);
            ChatChoice {
                index: c.index,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    reasoning_content: c.reasoning,
                    content: c.content,
                    tool_calls,
                    annotations,
                    refusal: c.refusal,
                },
                finish_reason: c.finish_reason.as_wire().to_string(),
                logprobs: c.logprobs.map(encode_logprobs),
            }
        })
        .collect();

    let completion_id = format!("chatcmpl-{}", ir.id);
    let completion = ChatCompletionResponse {
        id: completion_id.clone(),
        object: "chat.completion".to_string(),
        created: ir.created,
        model: ir.model.clone(),
        system_fingerprint: Some("fp_metrale".to_string()),
        choices,
        usage,
        service_tier: echo.service_tier.clone(),
        metadata: echo.metadata.clone(),
    };

    // 2026-09-26: `store: true` keeps the serialized body in the response
    // store for `GET /v1/chat/completions/{id}`. A serialization failure
    // skips storage.
    if echo.store
        && let Ok(body) = serde_json::to_value(&completion)
    {
        state
            .response_store
            .insert(crate::response_store::StoredEntry {
                id: completion_id,
                kind: crate::response_store::StoredKind::ChatCompletion,
                model: ir.model,
                created_at: ir.created,
                messages: Vec::new(),
                body,
                last_access: std::time::Instant::now(),
            });
    }

    if let (Some(seq), Some(dump)) = (dump_seq, state.dump_writer.as_ref()) {
        dump.dump_response("/v1/chat/completions", seq, &completion, false);
    }

    Json(completion).into_response()
}

fn encode_logprobs(lp: crate::ir::ChoiceLogprobs) -> ChoiceLogprobs {
    ChoiceLogprobs {
        content: lp
            .content
            .into_iter()
            .map(|t| TokenLogprobInfo {
                token: t.token,
                logprob: t.logprob,
                bytes: None,
                top_logprobs: t
                    .top
                    .into_iter()
                    .map(|(token, logprob)| TopLogprob {
                        token,
                        logprob,
                        bytes: None,
                    })
                    .collect(),
            })
            .collect(),
    }
}
