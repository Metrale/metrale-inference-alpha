// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: LoRA control endpoints (`POST /v1/lora/active`,
//! `POST /v1/lora/load`) and per-request adapter selection.
//!
//! Rotation and load commands go to the scheduler over the rotation channel.
//! The scheduler applies queued commands only on a tick with no active,
//! prefilling, new, spilled or requeued sequence (`scheduler/core/tick.rs`).
//!
//! Owner: server LoRA API.
//! Invariants: none beyond the types.

use crate::main_modules::model_host::CurrentModel;

use axum::extract::rejection::JsonRejection;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use serde::{Deserialize, Serialize};

use crate::AppState;
use crate::api::compact::openai_error_response;

/// 2026-09-26: Resolve a request's adapter to a pool slot for the chat and
/// completions handlers. `-1` selects the active adapter. A stageable name that
/// is not resident is promoted from its peer or from disk
/// (`AppState::ensure_adapter_hot_opt`). `Err` holds the response to return:
/// 400 unknown adapter, 503 pool full, 502 peer error.
pub async fn resolve_request_adapter_slot(
    state: &AppState,
    adapter: Option<&str>,
    model: &str,
) -> Result<i32, Response> {
    // 2026-09-26: The `adapter` field wins. Otherwise `model` selects when it
    // names a resident or stageable adapter; any other `model` (the base model
    // included) selects nothing, which resolves to `-1`, never a 400.
    let selector = match adapter {
        Some(a) => Some(a),
        None if state.adapter_names.iter().any(|n| n == model)
            || state.is_stageable_name(model) =>
        {
            Some(model)
        }
        None => None,
    };
    if let Some(slot) =
        crate::main_modules::app_state::resolve_adapter_slot(&state.adapter_names, selector)
    {
        return Ok(slot);
    }
    // 2026-09-26: Not resident: try to promote a stageable name into a cache
    // slot. A name that cannot be promoted is a 400.
    let name = selector.unwrap_or("");
    match state.ensure_adapter_hot_opt(name).await {
        Ok(Some(slot)) => Ok(slot),
        Ok(None) => Err(openai_error_response(
            StatusCode::BAD_REQUEST,
            format!(
                "unknown adapter '{}'; resident adapters: [{}]",
                name,
                state.adapter_names.join(", ")
            ),
        )),
        Err(crate::main_modules::promotion::PromoteReject::PoolFull(m)) => {
            Err(openai_error_response(StatusCode::SERVICE_UNAVAILABLE, m))
        }
        Err(crate::main_modules::promotion::PromoteReject::Peer(m)) => {
            Err(openai_error_response(StatusCode::BAD_GATEWAY, m))
        }
    }
}

#[derive(Deserialize)]
pub struct SetActiveLoraRequest {
    /// 2026-09-26: Name of the resident adapter to activate.
    pub adapter: String,
}

#[derive(Serialize)]
struct SetActiveLoraResponse {
    object: &'static str,
    active: String,
}

/// 2026-09-26: POST /v1/lora/active `{"adapter": "NAME"}`: make a resident
/// adapter the active one.
pub async fn set_active_lora(
    CurrentModel(state): CurrentModel,
    body: Result<Json<SetActiveLoraRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match body {
        Ok(b) => b,
        Err(e) => {
            return openai_error_response(StatusCode::BAD_REQUEST, format!("invalid body: {e}"));
        }
    };

    let Some(ref tx) = state.rotation_tx else {
        return openai_error_response(
            StatusCode::BAD_REQUEST,
            "no LoRA adapter is loaded (start with --lora-adapter NAME=PATH)".to_string(),
        );
    };

    if !state.adapter_names.iter().any(|n| n == &req.adapter) {
        return openai_error_response(
            StatusCode::NOT_FOUND,
            format!(
                "adapter '{}' is not resident (resident: [{}])",
                req.adapter,
                state.adapter_names.join(", ")
            ),
        );
    }

    let (ack_tx, ack_rx) = tokio::sync::oneshot::channel();
    if tx
        .send((
            crate::scheduler::LoraCommand::Rotate(req.adapter.clone()),
            ack_tx,
        ))
        .await
        .is_err()
    {
        return openai_error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "scheduler rotation channel closed".to_string(),
        );
    }
    match ack_rx.await {
        Ok(Ok(_)) => {
            // 2026-09-26: `active_adapter` is a status mirror; the model holds
            // the active slot.
            if let Ok(mut a) = state.active_adapter.lock() {
                *a = Some(req.adapter.clone());
            }
            Json(SetActiveLoraResponse {
                object: "lora.active",
                active: req.adapter,
            })
            .into_response()
        }
        Ok(Err(reason)) => openai_error_response(StatusCode::BAD_REQUEST, reason),
        Err(_) => openai_error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "scheduler dropped the rotation ack (shutting down?)".to_string(),
        ),
    }
}

#[derive(Deserialize)]
pub struct LoadLoraRequest {
    /// 2026-09-26: Name given to the loaded adapter.
    pub name: String,
    /// 2026-09-26: PEFT adapter directory; the handler requires
    /// `adapter_config.json` in it.
    pub path: String,
    /// 2026-09-26: Pool slot to load into; 0 when absent.
    #[serde(default)]
    pub slot: usize,
}

#[derive(Serialize)]
struct LoadLoraResponse {
    object: &'static str,
    loaded: String,
    slot: usize,
}

/// 2026-09-26: POST /v1/lora/load `{"name": "vega", "path": "/dir", "slot": 0}`:
/// load an adapter from disk into a pool slot. The model refuses unless
/// rotation is armed (`METRALE_LORA_ROTATE` on or `METRALE_LORA_PEER` set;
/// `model-engine/src/model/types.rs` `lora_rotatable`).
pub async fn load_lora_into_slot(
    CurrentModel(state): CurrentModel,
    body: Result<Json<LoadLoraRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match body {
        Ok(b) => b,
        Err(e) => {
            return openai_error_response(StatusCode::BAD_REQUEST, format!("invalid body: {e}"));
        }
    };

    // 2026-09-26: Bound the inputs before any work: the name is stamped on a
    // pool slot, the path is opened and the slot indexes the pool.
    if req.name.is_empty() || req.name.len() > 256 {
        return openai_error_response(
            StatusCode::BAD_REQUEST,
            "adapter name must be 1..=256 chars".to_string(),
        );
    }
    if req.path.len() > 4096 {
        return openai_error_response(
            StatusCode::BAD_REQUEST,
            "adapter path too long (max 4096 chars)".to_string(),
        );
    }
    if req.slot > 4096 {
        return openai_error_response(
            StatusCode::BAD_REQUEST,
            format!("slot {} out of range (max 4096)", req.slot),
        );
    }

    let Some(ref tx) = state.rotation_tx else {
        return openai_error_response(
            StatusCode::BAD_REQUEST,
            "no LoRA adapter pool is loaded (start with --lora-adapter NAME=PATH)".to_string(),
        );
    };

    let dir = std::path::PathBuf::from(&req.path);
    if !dir.join("adapter_config.json").exists() {
        return openai_error_response(
            StatusCode::BAD_REQUEST,
            format!("no adapter_config.json under path '{}'", req.path),
        );
    }

    let (ack_tx, ack_rx) = tokio::sync::oneshot::channel();
    let cmd = crate::scheduler::LoraCommand::LoadIntoSlot {
        name: req.name.clone(),
        dir,
        slot: req.slot,
    };
    if tx.send((cmd, ack_tx)).await.is_err() {
        return openai_error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "scheduler rotation channel closed".to_string(),
        );
    }
    match ack_rx.await {
        Ok(Ok(_)) => {
            if let Ok(mut a) = state.active_adapter.lock() {
                *a = Some(req.name.clone());
            }
            Json(LoadLoraResponse {
                object: "lora.loaded",
                loaded: req.name,
                slot: req.slot,
            })
            .into_response()
        }
        Ok(Err(reason)) => openai_error_response(StatusCode::BAD_REQUEST, reason),
        Err(_) => openai_error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "scheduler dropped the load ack (shutting down?)".to_string(),
        ),
    }
}
