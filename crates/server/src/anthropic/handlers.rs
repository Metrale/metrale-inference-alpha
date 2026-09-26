// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The Anthropic HTTP handlers, `messages` and `count_tokens`.
//!
//! Owner: server (Anthropic adapter).
//! Invariants: none beyond the types.

use crate::main_modules::model_host::CurrentModel;

use axum::extract::rejection::JsonRejection;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};

use super::handlers_stream::*;
use super::helpers::*;
use super::types::*;

/// 2026-09-26: `POST /v1/messages`. Lowers the body into the chat IR
/// (`From<MessagesRequest> for ir::ChatRequest`), runs
/// `api::chat_completions_inner`, and encodes the outcome in Anthropic's
/// shape. It makes no sampling or prompt decision of its own.
pub async fn messages(CurrentModel(state): CurrentModel, body: axum::body::Bytes) -> Response {
    let req: MessagesRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => {
            return anthropic_error(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                format!("Invalid request JSON: {e}"),
            );
        }
    };

    tracing::info!(
        "Anthropic request: max_tokens={}, thinking={:?}, tools={}, model={}, stream={}",
        req.max_tokens,
        req.thinking
            .as_ref()
            .map(|t| format!("type={} budget={:?}", t.thinking_type, t.budget_tokens)),
        req.tools.as_ref().map_or(0, |t| t.len()),
        req.model,
        req.stream,
    );

    let stream = req.stream;
    let model_echo = req.model.clone();

    // 2026-09-26: `--dump`: this handler takes its own seq so the request is
    // recorded under `/v1/messages`, and passes `dump_seq = None` to
    // `chat_completions_inner` so the pipeline writes no second entry.
    let dump_seq = state.dump_writer.as_ref().and_then(|d| {
        match serde_json::from_slice::<serde_json::Value>(&body) {
            Ok(v) => {
                let seq = d.next_seq();
                d.dump_request("/v1/messages", seq, &v);
                Some(seq)
            }
            Err(_) => None,
        }
    });

    let outcome = crate::api::chat_completions_inner(state.clone(), None, req.into(), None).await;

    let chat_resp = match outcome {
        crate::api::ChatOutcome::Blocking(ir) => {
            let messages_resp = MessagesResponse::from(*ir);
            if let (Some(seq), Some(dump)) = (dump_seq, state.dump_writer.as_ref()) {
                dump.dump_response("/v1/messages", seq, &messages_resp, false);
            }
            return Json(messages_resp).into_response();
        }
        crate::api::ChatOutcome::Streaming(deltas) => {
            // 2026-09-26: `--dump`: the encoder collects the Anthropic events
            // it sends and writes them under the request's seq.
            let dump =
                dump_seq.and_then(|seq| state.dump_writer.as_ref().map(|d| (seq, d.clone())));
            return anthropic_sse_from_deltas(deltas, model_echo, dump);
        }
        crate::api::ChatOutcome::Http(r) => r,
    };

    if !chat_resp.status().is_success() {
        // 2026-09-26: Re-wrap the error in Anthropic's body with the same
        // status. The message is `error.message` from an OpenAI-style body,
        // else the raw body text.
        let (parts, body) = chat_resp.into_parts();
        let body_bytes = match axum::body::to_bytes(body, usize::MAX).await {
            Ok(b) => b,
            Err(e) => {
                return anthropic_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "api_error",
                    format!("Error body collect: {e}"),
                );
            }
        };
        let err_msg = serde_json::from_slice::<serde_json::Value>(&body_bytes)
            .ok()
            .and_then(|v| {
                v.get("error")
                    .and_then(|e| e.get("message"))
                    .and_then(|m| m.as_str())
                    .map(|s| s.to_string())
            })
            .unwrap_or_else(|| String::from_utf8_lossy(&body_bytes).into_owned());
        return anthropic_error(parts.status, "api_error", err_msg);
    }

    let _ = stream;
    chat_resp
}

/// 2026-09-26: `POST /v1/messages/count_tokens`: `{"input_tokens": N}` for
/// the prompt the request would render.
pub async fn count_tokens(
    CurrentModel(state): CurrentModel,
    req: Result<Json<MessagesRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match req {
        Ok(r) => r,
        Err(e) => {
            return anthropic_error(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                format!("Invalid request JSON: {e}"),
            );
        }
    };

    // 2026-09-26: Counts the prompt the serving path renders: the same IR
    // adapter and the same `prepare_chat_prompt`.
    let mut ir_req = crate::ir::ChatRequest::from(req);
    // 2026-09-26: Image parts are removed so that counting needs no vision
    // encoder; the count excludes image tokens.
    for m in &mut ir_req.messages {
        m.content
            .retain(|p| !matches!(p, crate::ir::ContentPart::Image(_)));
    }
    // 2026-09-26: `prepare_chat_prompt` renders and tokenizes on the CPU, so
    // it runs on the blocking pool, as in `chat_completions_inner`.
    let state_for_prepare = state.clone();
    let prepared = match tokio::task::spawn_blocking(move || {
        crate::api::chat::prepare::prepare_chat_prompt(&state_for_prepare, &mut ir_req)
    })
    .await
    {
        Ok(Ok(p)) => p,
        Ok(Err(resp)) => return openai_error_to_anthropic(resp).await,
        Err(join_err) => {
            tracing::error!("count_tokens prepare panicked: {join_err}");
            return anthropic_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "api_error",
                format!("Error preparing prompt: {join_err}"),
            );
        }
    };

    let body = serde_json::json!({
        "input_tokens": prepared.prompt_tokens.len()
    });
    Json(body).into_response()
}

/// 2026-09-26: Re-wrap an OpenAI-style error response in Anthropic's error
/// body with the same status: `invalid_request_error` for a 4xx status,
/// `api_error` otherwise.
async fn openai_error_to_anthropic(resp: Response) -> Response {
    let (parts, body) = resp.into_parts();
    let message = match axum::body::to_bytes(body, usize::MAX).await {
        Ok(bytes) => serde_json::from_slice::<serde_json::Value>(&bytes)
            .ok()
            .and_then(|v| {
                v.get("error")
                    .and_then(|e| e.get("message"))
                    .and_then(|m| m.as_str())
                    .map(|s| s.to_string())
            })
            .unwrap_or_else(|| String::from_utf8_lossy(&bytes).into_owned()),
        Err(e) => format!("Error body collect: {e}"),
    };
    let error_type = if parts.status.is_client_error() {
        "invalid_request_error"
    } else {
        "api_error"
    };
    anthropic_error(parts.status, error_type, message)
}
