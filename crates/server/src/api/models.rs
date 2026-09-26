// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `/v1/models` list and retrieve, which also advertise LoRA
//! adapters as model ids, and the `/v1/embeddings` 501 stub.
//!
//! Owner: server API.
//! Invariants: none beyond the types.

use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};

use crate::openai::{ModelInfo, ModelListResponse};

use super::compact::openai_error_response;

/// 2026-09-26: GET /v1/models
pub async fn list_models(
    State(host): State<Arc<crate::main_modules::model_host::ModelHost>>,
) -> Json<ModelListResponse> {
    // 2026-09-26: With no model loaded the list is empty rather than a 503: the
    // endpoint works, and there is no model to list.
    let Some(state) = host.current() else {
        return Json(ModelListResponse {
            object: "list".to_string(),
            data: Vec::new(),
        });
    };
    // 2026-09-26: The advertised adapters are capped at `MAX_ADVERTISED_MODELS`,
    // so the pre-sized allocation does not grow with the adapter count.
    const MAX_ADVERTISED_MODELS: usize = 1024;
    let advertised = state.adapter_names.len().min(MAX_ADVERTISED_MODELS);
    let mut data = Vec::with_capacity(advertised.saturating_add(1));
    // 2026-09-26: Resident adapters first, in slot order.
    for adapter in state.adapter_names.iter().take(MAX_ADVERTISED_MODELS) {
        data.push(ModelInfo::advertise(adapter.clone(), state.max_seq_len));
    }
    // 2026-09-26: Then the stageable names (peer- and disk-backed), under the
    // same cap. A request whose `model` names one promotes it on first use
    // (`lora_control::resolve_request_adapter_slot`). They are not added to
    // `adapter_names`.
    for name in state
        .lora_stageable
        .keys()
        .chain(state.lora_disk_stageable.keys())
        .take(MAX_ADVERTISED_MODELS.saturating_sub(data.len()))
    {
        data.push(ModelInfo::advertise(name.clone(), state.max_seq_len));
    }
    data.push(ModelInfo::advertise(
        state.model_name.clone(),
        state.max_seq_len,
    ));
    Json(ModelListResponse {
        object: "list".to_string(),
        data,
    })
}

/// 2026-09-26: GET /v1/models/{model_id}: the base model, a resident adapter or
/// a stageable name; any other id is a 404.
pub async fn get_model(
    State(host): State<Arc<crate::main_modules::model_host::ModelHost>>,
    axum::extract::Path(model_id): axum::extract::Path<String>,
) -> Response {
    // 2026-09-26: With no model loaded: 404 `model_not_loaded`.
    let Some(state) = host.current() else {
        return (
            axum::http::StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": {
                    "message": crate::error_hints::message_with_hint(
                        "no model is loaded",
                        "model_not_loaded",
                    ),
                    "type": "model_not_loaded",
                    "hint": crate::error_hints::hint_for("model_not_loaded"),
                }
            })),
        )
            .into_response();
    };
    // 2026-09-26: Every id that `list_models` advertises is found here.
    let known = model_id == state.model_name
        || state.adapter_names.iter().any(|n| n == &model_id)
        || state.is_stageable_name(&model_id);
    if known {
        Json(serde_json::json!({
            "id": model_id,
            "object": "model",
            "created": crate::ids::unix_timestamp(),
            "owned_by": crate::identity::OWNED_BY,
            "max_model_len": state.max_seq_len,
        }))
        .into_response()
    } else {
        openai_error_response(
            StatusCode::NOT_FOUND,
            format!("The model '{model_id}' does not exist"),
        )
    }
}

/// 2026-09-26: POST /v1/embeddings: always 501.
pub async fn embeddings_stub() -> Response {
    openai_error_response(
        StatusCode::NOT_IMPLEMENTED,
        "Embeddings are not supported by this model. Metrale Engine serves generative (chat/completion) models only.".into(),
    )
}
