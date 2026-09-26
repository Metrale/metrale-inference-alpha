// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The HTTP front end: the router and its middleware, the
//! listener bind, the readiness line, and the accept loop.
//!
//! Owner: server (HTTP layer).
//! Invariants:
//! - The readiness line is logged only after the listener has bound.

use std::sync::Arc;

use anyhow::{Context as _, Result};
use axum::Router;
use axum::routing::{get, post};

use crate::anthropic;
use crate::api;
use crate::main_modules::middleware::{
    gpu_fault_middleware, openai_observability_middleware, rate_limit_middleware,
    require_auth_middleware,
};

pub(crate) async fn build_and_serve(
    host: Arc<crate::main_modules::model_host::ModelHost>,
    bind: &str,
    port: u16,
) -> Result<()> {
    metrale_telemetry::progress::phase(10, "router");
    host.set_bound(bind.to_string(), port);
    let cors = tower_http::cors::CorsLayer::new()
        .allow_origin(tower_http::cors::Any)
        .allow_methods([
            axum::http::Method::GET,
            axum::http::Method::POST,
            axum::http::Method::OPTIONS,
        ])
        .allow_headers(tower_http::cors::Any);

    // 2026-09-26: A handler panic becomes a 500 with tower-http's default
    // plain-text body, `Service panicked`; the panic message is logged, not
    // sent to the client.
    let catch_panic = tower_http::catch_panic::CatchPanicLayer::new();

    let app = Router::new()
        .route("/v1/chat/completions", post(api::chat_completions))
        .route("/v1/chat/completions/{id}", get(api::get_stored_completion))
        .route("/v1/completions", post(api::completions))
        .route("/v1/responses", post(api::responses_endpoint))
        .route(
            "/v1/responses/{id}",
            get(api::get_stored_response).delete(api::delete_stored_response),
        )
        .route(
            "/v1/responses/{id}/input_items",
            get(api::list_response_input_items),
        )
        .route("/v1/responses/{id}/cancel", post(api::cancel_response))
        .route("/v1/conversations", post(api::create_conversation))
        .route(
            "/v1/conversations/{id}",
            get(api::get_conversation)
                .post(api::update_conversation)
                .delete(api::delete_conversation),
        )
        .route(
            "/v1/conversations/{id}/items",
            post(api::add_conversation_items).get(api::list_conversation_items),
        )
        .route(
            "/v1/conversations/{id}/items/{item_id}",
            get(api::get_conversation_item).delete(api::delete_conversation_item),
        )
        .route("/v1/messages", post(anthropic::messages))
        .route("/v1/messages/count_tokens", post(anthropic::count_tokens))
        .route("/v1/lora/active", post(api::set_active_lora))
        .route("/v1/lora/load", post(api::load_lora_into_slot))
        .route("/v1/models", get(api::list_models))
        .route("/v1/models/{*model_id}", get(api::get_model))
        .route("/v1/embeddings", post(api::embeddings_stub))
        // 2026-09-26: Unimplemented OpenAI endpoints answer 501 with an
        // OpenAI-shaped error (`api/stubs.rs`).
        .route(
            "/v1/batches",
            post(api::batches_stub).get(api::batch_list_stub),
        )
        .route(
            "/v1/batches/{id}",
            get(api::batch_get_stub).delete(api::batch_get_stub),
        )
        .route("/v1/batches/{id}/cancel", post(api::batch_get_stub))
        .route("/v1/files", post(api::files_stub).get(api::files_stub))
        .route(
            "/v1/files/{id}",
            get(api::files_stub).delete(api::files_stub),
        )
        .route("/v1/files/{id}/content", get(api::files_stub))
        .route("/v1/audio/transcriptions", post(api::audio_stub))
        .route("/v1/audio/translations", post(api::audio_stub))
        .route("/v1/audio/speech", post(api::audio_stub))
        .route("/v1/images/generations", post(api::images_stub))
        .route("/v1/images/edits", post(api::images_stub))
        .route("/v1/images/variations", post(api::images_stub))
        .route("/v1/moderations", post(api::moderations_stub))
        .route("/tokenize", post(api::tokenize))
        .route("/detokenize", post(api::detokenize))
        .route("/hardware", get(api::hardware))
        .route("/serve-config", get(api::serve_config))
        .route("/health", get(api::health))
        .route("/health/live", get(api::health_live))
        .route("/metrics", get(api::metrics_handler))
        .route("/v1/events", get(api::telemetry_events))
        // 2026-09-26: Request body limit: 32 MiB, or `METRALE_MAX_BODY_BYTES`
        // in bytes; an unparseable value keeps 32 MiB.
        .layer(axum::extract::DefaultBodyLimit::max(
            std::env::var("METRALE_MAX_BODY_BYTES")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(32 * 1024 * 1024),
        ))
        // 2026-09-26: The middleware state is the `ModelHost`, not an
        // `AppState`, whose clone would hold the model's `request_tx` open
        // across a swap. The last layer added runs first, so a request passes
        // `catch_panic`, CORS, byte counting, observability, auth and the rate
        // limiter before `gpu_fault_middleware`.
        .layer(axum::middleware::from_fn(gpu_fault_middleware))
        .layer(axum::middleware::from_fn_with_state(
            host.clone(),
            rate_limit_middleware,
        ))
        .layer(axum::middleware::from_fn_with_state(
            host.clone(),
            require_auth_middleware,
        ))
        .layer(axum::middleware::from_fn(openai_observability_middleware))
        .layer(axum::middleware::from_fn(
            crate::main_modules::byte_count::byte_count_middleware,
        ))
        .layer(cors)
        .layer(catch_panic)
        .with_state(host.clone());

    let addr = format!("{bind}:{port}");
    if bind == "0.0.0.0" {
        tracing::warn!(
            "Metrale Engine is listening on {addr} — reachable from any host on the network. \
             If this machine is on a shared LAN or has a public IP, pass \
             --bind 127.0.0.1 (or set --require-auth and a real firewall) before \
             accepting traffic."
        );
    } else if bind == "127.0.0.1" || bind == "localhost" || bind == "::1" {
        tracing::info!(
            "API reachable only from this machine (loopback). To expose on the \
             LAN pass --bind 0.0.0.0; combine with --require-auth and \
             --auth-tokens-file for non-trusted networks."
        );
    }
    let listener = bind_and_announce(&host, bind, port).await?;
    serve_with_header_timeout(listener, app).await
}

/// 2026-09-26: Bind the listener, then log the readiness line and mark the
/// listening phase; a bind error returns before either. Kept apart from the
/// accept loop so tests can run it without `disarm_startup_escape`, which
/// cannot be undone.
async fn bind_and_announce(
    host: &crate::main_modules::model_host::ModelHost,
    bind: &str,
    port: u16,
) -> Result<tokio::net::TcpListener> {
    let addr = format!("{bind}:{port}");
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .with_context(|| format!("binding {addr}"))?;
    tracing::info!("{}", ready_line(bind, port, host.live_model().as_deref()));
    metrale_telemetry::progress::phase(11, "listening");
    metrale_telemetry::progress::ready(port);
    Ok(listener)
}

/// 2026-09-26: The line that ends a successful startup (`bind_and_announce`)
/// or model swap (`model_swap.rs`). The wildcard binds `0.0.0.0` and `::` are
/// shown as `127.0.0.1`, a destination a client can use, and an IPv6 literal
/// is bracketed so the port stays separate.
pub(crate) fn ready_line(bind: &str, port: u16, model: Option<&str>) -> String {
    let host = match bind {
        "0.0.0.0" | "::" => "127.0.0.1".to_string(),
        v6 if v6.contains(':') => format!("[{v6}]"),
        other => other.to_string(),
    };
    match model {
        Some(model) => format!("Server live and ready at {host}:{port} running {model}"),
        // 2026-09-26: With no model loaded the line says live, not ready.
        None => format!(
            "Server live at {host}:{port} — no model loaded yet, requests get 503 until one \
             is started from the Library"
        ),
    }
}

/// 2026-09-26: Serve `app` with an HTTP/1 header-read timeout, so a client
/// that sends its headers slowly cannot hold a connection open.
///
/// `axum::serve` installs no timer, so hyper applies no header-read timeout,
/// and a `TimeoutLayer` would also cut long generations; connections are
/// therefore served through `hyper_util`'s `auto::Builder`. The make-service
/// still injects `ConnectInfo<SocketAddr>`, which `rate_limit_middleware`
/// reads.
async fn serve_with_header_timeout(
    listener: tokio::net::TcpListener,
    app: Router,
) -> anyhow::Result<()> {
    use hyper_util::rt::{TokioExecutor, TokioIo, TokioTimer};
    use hyper_util::server::conn::auto::Builder;
    use tower::{Service, ServiceExt};

    const HEADER_READ_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

    let mut make_service = app.into_make_service_with_connect_info::<std::net::SocketAddr>();

    // 2026-09-26: From here a shutdown stops accepting and drains in-flight
    // requests, so the startup escape is disarmed.
    crate::tui::shutdown::disarm_startup_escape();

    loop {
        let accepted = tokio::select! {
            conn = listener.accept() => conn,
            _ = crate::tui::shutdown::wait() => {
                // 2026-09-26: Stop accepting, give in-flight requests up to
                // 15 s, flush the tee and return.
                crate::tui::shutdown::drain_in_flight(std::time::Duration::from_secs(15)).await;
                crate::tui::init::flush_tee();
                tracing::info!("Shutdown complete");
                return Ok(());
            }
        };
        let (socket, peer_addr) = match accepted {
            Ok(conn) => conn,
            Err(e) => {
                // 2026-09-26: An accept error is logged and the loop goes on
                // after 10 ms.
                tracing::warn!("accept error: {e}");
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                continue;
            }
        };

        let tower_service = match make_service.call(peer_addr).await {
            Ok(svc) => svc,
            Err(infallible) => match infallible {},
        };

        tokio::spawn(async move {
            let socket = TokioIo::new(socket);
            let hyper_service = hyper::service::service_fn(
                move |request: hyper::Request<hyper::body::Incoming>| {
                    tower_service.clone().oneshot(request)
                },
            );

            let mut builder = Builder::new(TokioExecutor::new());
            // 2026-09-26: hyper applies `header_read_timeout` only with a timer.
            builder
                .http1()
                .timer(TokioTimer::new())
                .header_read_timeout(HEADER_READ_TIMEOUT);
            builder.http2().timer(TokioTimer::new());

            if let Err(err) = builder
                .serve_connection_with_upgrades(socket, hyper_service)
                .await
            {
                tracing::debug!("connection closed: {err}");
            }
        });
    }
}

#[cfg(test)]
#[path = "serve_router_tests.rs"]
mod tests;
