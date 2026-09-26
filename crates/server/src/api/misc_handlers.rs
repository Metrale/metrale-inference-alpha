// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Operational endpoints: `/metrics`, `/health`, `/health/live`,
//! `/hardware`, `/serve-config`, `/tokenize`, `/detokenize`, and the
//! Responses cancel refusal.
//!
//! Owner: server API.
//! Invariants: none beyond the types.

#![allow(unused_imports, dead_code)]

use crate::main_modules::model_host::CurrentModel;
use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive};
use axum::response::{IntoResponse, Json, Response, Sse};
use futures::StreamExt;
use std::sync::Arc;
use tokio_stream::wrappers::ReceiverStream;

use super::chat_stream::run_chat_stream;
use super::responses_stream::responses_endpoint_stream;
use super::responses_translate::{
    build_responses_usage, emit, find_frame_end, translate_chat_response_to_responses,
};
use super::stored::assistant_incoming_from_ir;
use crate::AppState;
use crate::openai::{
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, CompletionChunk,
    CompletionRequest, CompletionResponse, ModelInfo, ModelListResponse, Usage,
};
use crate::tool_parser;

use super::chat::chat_completions_inner;
use super::compact::{compact_messages, openai_error_response, openai_error_response_with_param};
use super::completions::not_supported;
use super::inference_impl::{extract_thinking, strip_stop_sequences, tokenize_stop_sequences};
use super::inference_types::{
    GrammarSpec, InferenceRequest, InferenceResponse, StreamEvent, TokenLogprobs,
};
use super::sanitizer::{
    F7_STALL_REFUSE_THRESHOLD, F7_STALL_WARN_THRESHOLD, F7StallBuckets, ToolKind, classify_tool,
    extract_bash_final_action, primary_arg_for_tool, sanitize_content_chunk,
};
use super::strip::strip_thinking_tags;

use super::inference_types::*;
use super::sanitizer::*;

pub async fn cancel_response(axum::extract::Path(id): axum::extract::Path<String>) -> Response {
    openai_error_response_with_param(
        StatusCode::BAD_REQUEST,
        format!(
            "Response '{id}' cannot be cancelled: Metrale Engine completes responses synchronously. Cancel only applies when the request was created with `background: true`, which this server does not support."
        ),
        Some("id"),
        Some("response_not_cancellable"),
    )
}

/// 2026-09-26: GET /metrics in Prometheus text format.
pub async fn metrics_handler() -> impl IntoResponse {
    use prometheus::Encoder;
    use std::fmt::Write;

    let encoder = prometheus::TextEncoder::new();
    let metric_families = prometheus::gather();
    let mut buffer = Vec::new();
    encoder.encode(&metric_families, &mut buffer).unwrap();
    let mut text = String::from_utf8(buffer).unwrap_or_default();

    let hits = metrale_telemetry::prefix_cache::cache_hit_count();
    let misses = metrale_telemetry::prefix_cache::cache_miss_count();
    let hit_tokens = metrale_telemetry::prefix_cache::cache_hit_tokens_total();
    let total = hits + misses;
    let hit_rate = if total > 0 {
        hits as f64 / total as f64
    } else {
        0.0
    };

    let _ = write!(
        text,
        "\
        # HELP metrale_prefix_cache_hits_total Prefix cache lookups that found cached blocks\n\
        # TYPE metrale_prefix_cache_hits_total counter\n\
        metrale_prefix_cache_hits_total {hits}\n\
        # HELP metrale_prefix_cache_misses_total Prefix cache lookups with no match\n\
        # TYPE metrale_prefix_cache_misses_total counter\n\
        metrale_prefix_cache_misses_total {misses}\n\
        # HELP metrale_prefix_cache_hit_tokens_total Tokens reused from prefix cache\n\
        # TYPE metrale_prefix_cache_hit_tokens_total counter\n\
        metrale_prefix_cache_hit_tokens_total {hit_tokens}\n\
        # HELP metrale_prefix_cache_hit_rate Prefix cache hit rate (0-1)\n\
        # TYPE metrale_prefix_cache_hit_rate gauge\n\
        metrale_prefix_cache_hit_rate {hit_rate:.4}\n"
    );

    let entropy = metrale_sampling::last_entropy();
    let low_entropy = metrale_sampling::low_entropy_token_count();
    let total_sampled = metrale_sampling::total_sampled_token_count();
    let low_ratio = if total_sampled > 0 {
        low_entropy as f64 / total_sampled as f64
    } else {
        0.0
    };

    // 2026-09-26: Unresolved kernel lookups for the live model, so a check can
    // assert `== 0` without reading the boot log
    // (`metrale_telemetry::kernel_audit::unresolved_lookups`).
    let unresolved = metrale_telemetry::kernel_audit::unresolved_lookups();
    let _ = write!(
        text,
        "\
        # HELP metrale_kernel_lookups_unresolved Kernel lookups that did not resolve for the live model\n\
        # TYPE metrale_kernel_lookups_unresolved gauge\n\
        metrale_kernel_lookups_unresolved {unresolved}\n"
    );

    let _ = write!(
        text,
        "\
        # HELP metrale_token_entropy_last Most recent per-token entropy (nats)\n\
        # TYPE metrale_token_entropy_last gauge\n\
        metrale_token_entropy_last {entropy:.4}\n\
        # HELP metrale_low_entropy_tokens_total Tokens with entropy below 0.3\n\
        # TYPE metrale_low_entropy_tokens_total counter\n\
        metrale_low_entropy_tokens_total {low_entropy}\n\
        # HELP metrale_low_entropy_ratio Fraction of tokens with entropy below 0.3\n\
        # TYPE metrale_low_entropy_ratio gauge\n\
        metrale_low_entropy_ratio {low_ratio:.4}\n"
    );

    // 2026-09-26: The telemetry section; empty at `--telemetry off`
    // (`metrale_telemetry::export::prometheus::render`).
    text.push_str(&metrale_telemetry::export::prometheus::render(
        metrale_telemetry::global(),
        &metrale_telemetry::launch_trace::kernel_name,
    ));

    (
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        text,
    )
}

/// 2026-09-26: The readiness verdict from the published model name and the GPU
/// fault latch: a fault gives 503 `faulted` even with a model published, since
/// readiness means requests can succeed; otherwise a model gives 200 `ready`,
/// and none gives 503 `loading`. Pure, so `api/tests/health_fault.rs` tests it
/// without a server.
pub(crate) fn readiness(
    model: Option<&str>,
    fault: Option<&str>,
) -> (StatusCode, serde_json::Value) {
    if let Some(reason) = fault {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            serde_json::json!({"status": "faulted", "reason": reason}),
        );
    }
    match model {
        Some(name) => (
            StatusCode::OK,
            serde_json::json!({"status": "ready", "model": name}),
        ),
        None => (
            StatusCode::SERVICE_UNAVAILABLE,
            serde_json::json!({"status": "loading"}),
        ),
    }
}

/// 2026-09-26: GET /health, the readiness probe ([`readiness`]).
pub async fn health(
    State(host): State<Arc<crate::main_modules::model_host::ModelHost>>,
) -> Response {
    // 2026-09-26: Takes the host, not the model, so it can answer while no
    // model is published.
    let state = host.current();
    let (code, body) = readiness(
        state.as_ref().map(|s| s.model_name.as_str()),
        metrale_core::fault::global().fault(),
    );
    (code, Json(body)).into_response()
}

/// 2026-09-26: GET /health/live, the liveness probe: 200 `ok`, or 503 once the
/// GPU fault latch is set, so a supervisor restarts the process. The scheduler
/// also requests shutdown on a latched fault
/// (`scheduler/core/pipeline_lane.rs`); this probe reports the fault while the
/// drain runs or if it hangs.
pub async fn health_live() -> Response {
    match metrale_core::fault::global().fault() {
        None => "ok".into_response(),
        Some(reason) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"status": "faulted", "reason": reason})),
        )
            .into_response(),
    }
}

/// 2026-09-26: GET /hardware: the host's hardware fingerprint for benchmark
/// provenance, probed on each request because it includes the live SM clock.
/// The probe runs vendor tools as blocking subprocesses, hence `spawn_blocking`.
pub async fn hardware() -> Response {
    let hw = tokio::task::spawn_blocking(metrale_bench::hardware::Hardware::probe)
        .await
        .unwrap_or_else(|_| metrale_bench::hardware::Hardware::unknown());
    Json(hw).into_response()
}

/// 2026-09-26: GET /serve-config: SHA-256 digests of this server's binary, argv
/// and `METRALE_*` lever environment (`ServeIdentity`), never the values (an
/// argv can carry `--auth-token`). `cli::bench_lease` compares them before it
/// reuses a running server.
pub async fn serve_config() -> Response {
    let id =
        tokio::task::spawn_blocking(|| metrale_bench::serve_identity::this_process().clone()).await;
    match id {
        Ok(id) => Json(id).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": format!("{e}")})),
        )
            .into_response(),
    }
}

/// 2026-09-26: POST /tokenize: token IDs and count for `prompt` text or for
/// `messages` rendered through the chat template.
pub async fn tokenize(
    CurrentModel(state): CurrentModel,
    req: Result<Json<crate::openai::TokenizeRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match req {
        Ok(r) => r,
        Err(e) => {
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                format!("Invalid request JSON: {e}"),
            );
        }
    };

    let tokens = if let Some(ref prompt) = req.prompt {
        match state.tokenizer.encode(prompt) {
            Ok(t) => t,
            Err(e) => {
                return openai_error_response(
                    StatusCode::BAD_REQUEST,
                    format!("Tokenization error: {e}"),
                );
            }
        }
    } else if let Some(ref messages) = req.messages {
        let json_messages: Vec<serde_json::Value> = messages
            .iter()
            .map(|m| serde_json::json!({"role": m.role, "content": m.content.text}))
            .collect();
        match state.tokenizer.apply_chat_template_jinja_with_effort(
            &json_messages,
            None,
            false,
            state.behavior.disable_tool_steering,
            None,
            // 2026-09-26: The model's `preserve_thinking`, which chat also uses
            // when the request sets none (`chat/prepare.rs`).
            state.behavior.preserve_thinking,
        ) {
            Ok(t) => t,
            Err(e) => {
                return openai_error_response(
                    StatusCode::BAD_REQUEST,
                    format!("Tokenization error: {e}"),
                );
            }
        }
    } else {
        return openai_error_response(
            StatusCode::BAD_REQUEST,
            "Either 'prompt' or 'messages' is required".to_string(),
        );
    };

    let count = tokens.len();
    Json(crate::openai::TokenizeResponse { tokens, count }).into_response()
}

/// 2026-09-26: Request body for POST /detokenize.
#[derive(serde::Deserialize)]
pub struct DetokenizeRequest {
    tokens: Vec<u32>,
}

/// 2026-09-26: POST /detokenize: decode token IDs to text.
pub async fn detokenize(
    CurrentModel(state): CurrentModel,
    req: Result<Json<DetokenizeRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match req {
        Ok(r) => r,
        Err(e) => {
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                format!("Invalid request JSON: {e}"),
            );
        }
    };
    match state.tokenizer.decode(&req.tokens) {
        Ok(text) => Json(serde_json::json!({"text": text})).into_response(),
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": e.to_string()})),
        )
            .into_response(),
    }
}
