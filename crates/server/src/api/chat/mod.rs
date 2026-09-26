// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(unused_imports, dead_code)]

//! 2026-09-26: The `/v1/chat/completions` handler and the shared chat
//! pipeline, `chat_completions_inner`, which the Responses and Anthropic
//! surfaces also call.
//!
//! Owner: server (chat API).
//! Invariants: none beyond the types.

pub(crate) mod echo;
pub(crate) mod levers;
mod loop_detect;
mod msg_entry;
pub(crate) mod prepare;
pub(crate) mod remote_image;
mod sampling_setup;
mod template;
mod thinking;

use crate::main_modules::model_host::CurrentModel;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use std::sync::Arc;

use crate::AppState;

pub(crate) use echo::ResponseEcho;

/// 2026-09-26: Result of the shared chat pipeline: the response IR for a
/// non-streaming success, the delta stream for a streaming success, or a
/// finished HTTP response. Each surface encodes the first two itself.
pub(crate) enum ChatOutcome {
    Blocking(Box<crate::ir::ChatResponse>),
    Streaming(crate::ir::DeltaStream),
    Http(Response),
}

/// 2026-09-26: Test-only accessor: the Anthropic adapter's rendered-prompt
/// golden (`anthropic/tests/ir_carry.rs`) drives IR → `MsgEntry` →
/// template JSON without an `AppState`.
#[cfg(test)]
#[allow(clippy::result_large_err)]
pub(crate) fn test_build_msg_entries(
    input: &[crate::ir::Message],
    tools_active: bool,
) -> Result<Vec<msg_entry::MsgEntry>, axum::response::Response> {
    msg_entry::build_msg_entries(
        None,
        None,
        &remote_image::RemoteImagePolicy::default(),
        &msg_entry::VideoDecode {
            ffmpeg: &metrale_model_layers::video_decode_ffmpeg::FfmpegPolicy {
                enabled: false,
                ..Default::default()
            },
            fps: 2.0,
        },
        input,
        tools_active,
        &levers::ChatLevers::OFF,
        false,
    )
    .map(|o| o.messages)
}

#[cfg(test)]
pub(crate) fn test_build_json_messages(entries: &[msg_entry::MsgEntry]) -> Vec<serde_json::Value> {
    template::build_json_messages(entries)
}

use super::compact::openai_error_response;

pub async fn chat_completions(
    // 2026-09-26: The host, not `CurrentModel`: this handler may load the
    // model the body names, and `CurrentModel` resolves the live model before
    // the body is read.
    axum::extract::State(host): axum::extract::State<
        std::sync::Arc<crate::main_modules::model_host::ModelHost>,
    >,
    req_ctx: Option<axum::extract::Extension<crate::rate_limiter::RequestContext>>,
    body: axum::body::Bytes,
) -> Response {
    // 2026-09-26: Parsed by hand so the same bytes also feed the `--dump`
    // capture below.
    let req: crate::openai::ChatCompletionRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => {
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                format!("Invalid request JSON: {e}"),
            );
        }
    };

    // 2026-09-26: An unknown `reasoning_effort` is a 400 here; lowering keeps
    // only values `ir::parse_wire_effort` accepts, so it would otherwise be
    // dropped without notice.
    if let Err(msg) = req.validate_reasoning_effort() {
        return openai_error_response(StatusCode::BAD_REQUEST, msg);
    }

    // 2026-09-26: With `--auto-swap`, a request naming another model that the
    // recipe catalogue knows loads that model first. An empty, unknown or
    // already-live name is served by the current model.
    if host.auto_swap_enabled() {
        let live = host.live_model().unwrap_or_default();
        // 2026-09-26: The empty and already-live cases are settled here,
        // without reading the recipe index from disk.
        if !req.model.is_empty() && req.model != live {
            let requested = req.model.clone();
            let swap_host = host.clone();
            // 2026-09-26: The catalogue read and the load both block, so both
            // run on the blocking pool.
            let outcome = tokio::task::spawn_blocking(move || {
                let catalogue = metrale_bench::ArtifactStore::discover()
                    .ok()
                    .map(|s| crate::recipe::fetch::cached(s.root()).recipes)
                    .unwrap_or_default();
                match crate::main_modules::auto_swap::decide(&requested, &live, &catalogue) {
                    crate::main_modules::auto_swap::Decision::SwapTo(recipe_id) => {
                        crate::main_modules::auto_swap::ensure_loaded(
                            &swap_host, &recipe_id, &requested, &catalogue,
                        )
                    }
                    crate::main_modules::auto_swap::Decision::ServeCurrent => Ok(()),
                }
            })
            .await;
            match outcome {
                Ok(Ok(())) => {}
                // 2026-09-26: A failed swap is logged and the request is served
                // by whichever model is live; `model_swap::swap` restores the
                // previous model when it can.
                Ok(Err(e)) => tracing::warn!("auto-swap to {:?} failed: {e:#}", req.model),
                Err(e) => tracing::warn!("auto-swap task failed: {e}"),
            }
        }
    }

    // 2026-09-26: Resolved after any swap, so the request is served by the
    // model it asked for.
    let Some(state) = host.current() else {
        // 2026-09-26: `_typed`: the plain form derives `error.type` from the
        // status (`server_error` for 503), while the `CurrentModel` extractor
        // reports `model_not_loaded` for the same condition.
        return crate::api::compact::openai_error_response_typed(
            StatusCode::SERVICE_UNAVAILABLE,
            "no model is loaded".to_string(),
            "model_not_loaded",
        );
    };

    let dump_seq = state.dump_writer.as_ref().and_then(|d| {
        match serde_json::from_slice::<serde_json::Value>(&body) {
            Ok(v) => {
                let seq = d.next_seq();
                d.dump_request("/v1/chat/completions", seq, &v);
                Some(seq)
            }
            Err(_) => None,
        }
    });

    // 2026-09-26: The response-only fields are split off here; everything
    // downstream reads only the IR.
    let echo = ResponseEcho::from(&req);
    match chat_completions_inner(state.clone(), req_ctx, req.into(), dump_seq).await {
        ChatOutcome::Blocking(ir) => {
            crate::openai::encode_chat_response(&state, *ir, &echo, dump_seq)
        }
        ChatOutcome::Streaming(deltas) => {
            crate::openai::encode_sse_response(deltas, state.model_name.clone(), echo.include_usage)
        }
        ChatOutcome::Http(r) => r,
    }
}

/// 2026-09-26: The shared pipeline over an IR request. Called by
/// [`chat_completions`] after it lowers the wire request, and by the
/// Responses and Anthropic handlers, which lower their own wire formats and
/// pass `dump_seq = None`.
pub(crate) async fn chat_completions_inner(
    state: Arc<AppState>,
    req_ctx: Option<axum::extract::Extension<crate::rate_limiter::RequestContext>>,
    mut req: crate::ir::ChatRequest,
    dump_seq: Option<u64>,
) -> ChatOutcome {
    crate::metrics::REQUESTS_TOTAL.inc();
    // 2026-09-26: Decrements on drop, so on every exit path, including this
    // future being dropped when the client disconnects. A streaming request
    // moves it into the stream so it outlives this function.
    let active_guard = crate::metrics::ActiveRequestGuard::new();

    if let Err(resp) = super::chat_phases::validate_input(&req) {
        return ChatOutcome::Http(resp);
    }

    // 2026-09-26: The request's LoRA adapter is resolved to a pool slot once,
    // for both dispatch paths. No adapter gives `-1` (the installed one); an
    // unknown name is a 400; a stageable name is promoted into a slot
    // (`lora_control::resolve_request_adapter_slot`).
    let adapter_slot = match super::lora_control::resolve_request_adapter_slot(
        &state,
        req.adapter.as_deref(),
        &req.model,
    )
    .await
    {
        Ok(slot) => slot,
        Err(resp) => return ChatOutcome::Http(resp),
    };

    // 2026-09-26: Source and target language token names → token ids. An
    // absent name gives 0; an unknown one is a 400.
    let resolve_lang = |name: &Option<String>| -> Result<u32, Response> {
        match name {
            None => Ok(0),
            Some(s) => state.tokenizer.inner().token_to_id(s).ok_or_else(|| {
                openai_error_response(
                    StatusCode::BAD_REQUEST,
                    format!("unknown language token '{s}'"),
                )
            }),
        }
    };
    let src_lang_id = match resolve_lang(&req.src_lang) {
        Ok(v) => v,
        Err(resp) => return ChatOutcome::Http(resp),
    };
    let tgt_lang_id = match resolve_lang(&req.tgt_lang) {
        Ok(v) => v,
        Err(resp) => return ChatOutcome::Http(resp),
    };

    let num_beams = req.num_beams.unwrap_or(1);
    let length_penalty = req.length_penalty.unwrap_or(1.0);
    let early_stopping = req.early_stopping.unwrap_or(false);
    if num_beams > 1 && req.stream {
        return ChatOutcome::Http(openai_error_response(
            StatusCode::BAD_REQUEST,
            "num_beams > 1 is not supported in streaming mode".to_string(),
        ));
    }

    // 2026-09-26: Prompt preparation (shared with `count_tokens`) renders and
    // tokenizes on the CPU, so it runs on the blocking pool instead of holding
    // an async worker. `req` is moved in and handed back, which keeps the tool
    // prompt `prepare_chat_prompt` adds to it.
    let _t_seg = std::time::Instant::now();
    let state_for_prepare = state.clone();
    let (prepared, moved_req) = match tokio::task::spawn_blocking(move || {
        let out = prepare::prepare_chat_prompt(&state_for_prepare, &mut req);
        (out, req)
    })
    .await
    {
        Ok(pair) => pair,
        Err(join_err) => {
            tracing::error!("prepare_chat_prompt panicked: {join_err}");
            return ChatOutcome::Http(openai_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Internal error preparing the chat prompt".to_string(),
            ));
        }
    };
    req = moved_req;
    let prepare::PreparedChat {
        tools_active,
        cwd_hint,
        image_pixels,
        prompt_tokens,
        enable_thinking,
        thinking_budget,
    } = match prepared {
        Ok(p) => p,
        Err(resp) => return ChatOutcome::Http(resp),
    };

    let us_prepare = _t_seg.elapsed().as_micros();

    let loop_detect::LoopDetectOut {
        suppress_tool_call,
        tool_call_repeat_count,
    } = loop_detect::check_loops(&req.messages, tools_active);
    let us_loop_detect = _t_seg.elapsed().as_micros() - us_prepare;

    let session_hash = crate::session_manager::compute_session_hash(&prompt_tokens);
    let us_session_hash = _t_seg.elapsed().as_micros() - us_prepare - us_loop_detect;
    let tools_count = req.tools.len();
    tracing::info!(
        "Session {session_hash:#x}: {prompt_tokens} prompt tokens, tools={tools_active} ({tools_count} defined)",
        prompt_tokens = prompt_tokens.len()
    );
    let prompt_len = prompt_tokens.len();
    if prompt_len >= state.max_seq_len {
        return ChatOutcome::Http(openai_error_response(
            StatusCode::BAD_REQUEST,
            format!(
                "Prompt too long: {prompt_len} tokens exceeds max_seq_len {} (leave room for output tokens)",
                state.max_seq_len
            ),
        ));
    }

    let sampling_setup::SamplingSetup {
        temperature,
        top_k,
        top_p,
        top_n_sigma,
        min_p,
        repetition_penalty,
        presence_penalty,
        frequency_penalty,
        dry_multiplier,
        dry_base,
        dry_allowed_length,
        lz_penalty,
        logit_bias,
        max_tokens,
        stop_tokens,
        tool_choice_required,
        grammar_spec,
        timeout_at,
        top_logprobs,
    } = match sampling_setup::build_sampling(
        &state,
        &req,
        enable_thinking,
        tools_active,
        suppress_tool_call,
        tool_call_repeat_count,
    ) {
        Ok(s) => s,
        Err(resp) => return ChatOutcome::Http(resp),
    };
    if state.chat.phase_timing {
        let us_sampling =
            _t_seg.elapsed().as_micros() - us_prepare - us_loop_detect - us_session_hash;
        tracing::info!(
            "CHAT_PHASE handler: prepare={us_prepare}us loop_detect={us_loop_detect}us \
             session_hash={us_session_hash}us sampling_and_grammar={us_sampling}us \
             total_pre_dispatch={}us",
            _t_seg.elapsed().as_micros()
        );
    }

    if req.stream {
        return super::chat_stream_dispatch::dispatch_streaming(
            state,
            &req,
            req_ctx,
            dump_seq,
            prompt_tokens,
            session_hash,
            adapter_slot,
            src_lang_id,
            tgt_lang_id,
            num_beams,
            length_penalty,
            early_stopping,
            image_pixels,
            max_tokens,
            temperature,
            top_k,
            top_p,
            top_n_sigma,
            min_p,
            repetition_penalty,
            presence_penalty,
            frequency_penalty,
            dry_multiplier,
            dry_base,
            dry_allowed_length,
            lz_penalty,
            logit_bias.clone(),
            enable_thinking,
            thinking_budget,
            tools_active,
            tool_choice_required,
            suppress_tool_call,
            cwd_hint.clone(),
            stop_tokens,
            grammar_spec.clone(),
            top_logprobs,
            timeout_at,
            active_guard,
        )
        .await;
    }

    super::chat_blocking::run_blocking_path(super::chat_blocking::BlockingPathArgs {
        state,
        req,
        req_ctx,
        prompt_tokens,
        session_hash,
        adapter_slot,
        src_lang_id,
        tgt_lang_id,
        num_beams,
        length_penalty,
        early_stopping,
        image_pixels,
        max_tokens,
        temperature,
        top_k,
        top_p,
        top_n_sigma,
        min_p,
        repetition_penalty,
        presence_penalty,
        frequency_penalty,
        dry_multiplier,
        dry_base,
        dry_allowed_length,
        lz_penalty,
        logit_bias,
        stop_tokens,
        enable_thinking,
        thinking_budget,
        tools_active,
        tool_choice_required,
        suppress_tool_call,
        grammar_spec,
        top_logprobs,
        timeout_at,
        cwd_hint,
        prompt_len,
    })
    .await
}
