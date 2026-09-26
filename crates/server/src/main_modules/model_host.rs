// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `ModelHost`, the cell every request reads the current model's
//! [`AppState`] through, and the process-scoped state that outlives a swap.
//!
//! A request that took its `Arc<AppState>` before a swap finishes against the
//! model it began with.
//!
//! Owner: server (model hosting).
//! Invariants: `current` holds one whole `AppState` or none; `publish` and
//! `take` replace it whole.

use std::sync::Arc;

use super::app_state::AppState;

pub struct ModelHost {
    current: parking_lot::RwLock<Option<Arc<AppState>>>,
    /// 2026-09-26: The loaded model's scheduler thread. A swap takes and joins
    /// it before the model is torn down (`model_swap::swap`).
    scheduler: parking_lot::Mutex<Option<std::thread::JoinHandle<()>>>,
    /// 2026-09-26: The argv the live model was loaded from; `swap` reads it to
    /// carry the process flags and to restore the model when a load fails.
    args: parking_lot::RwLock<Option<crate::cli::ServeArgs>>,
    /// 2026-09-26: Held for the whole of a swap (`swap_guard`), so concurrent
    /// swaps run one at a time.
    swapping: parking_lot::Mutex<()>,
    /// 2026-09-26: The Tokio runtime in scope at construction. `swap` enters
    /// it: the load spawns Tokio tasks (the OOM watchdog), and the TUI calls
    /// `swap` from a plain thread.
    runtime: parking_lot::RwLock<Option<tokio::runtime::Handle>>,
    /// 2026-09-26: The listener's address, set when it binds. The socket is
    /// fixed for the process lifetime.
    bound: parking_lot::RwLock<Option<(String, u16)>>,
    /// 2026-09-26: The API-key policy. Process-scoped, so it applies while no
    /// model is loaded.
    auth: parking_lot::RwLock<Option<Arc<crate::auth::AuthConfig>>>,
    /// 2026-09-26: The process-scoped stores and rate limiter (`Carried`). Each
    /// load builds its `AppState` from these same `Arc`s, so the limiter the
    /// middleware debits is the one handlers refund.
    process: parking_lot::RwLock<Option<super::serve_load::Carried>>,
    /// 2026-09-26: The dashboard's run-handle channel. `swap` hands it to every
    /// load, so the dashboard follows the model now serving.
    tui_handles: parking_lot::RwLock<Option<std::sync::mpsc::Sender<crate::tui::RunHandles>>>,
}

impl ModelHost {
    /// 2026-09-26: A host with nothing loaded, holding the Tokio runtime in
    /// scope, if any.
    pub fn empty() -> Self {
        Self {
            current: parking_lot::RwLock::new(None),
            scheduler: parking_lot::Mutex::new(None),
            args: parking_lot::RwLock::new(None),
            swapping: parking_lot::Mutex::new(()),
            runtime: parking_lot::RwLock::new(tokio::runtime::Handle::try_current().ok()),
            bound: parking_lot::RwLock::new(None),
            auth: parking_lot::RwLock::new(None),
            process: parking_lot::RwLock::new(None),
            tui_handles: parking_lot::RwLock::new(None),
        }
    }

    /// 2026-09-26: The model serving now, or `None`. An owned `Arc`, so no lock
    /// is held across an await and the caller keeps its model through a swap.
    pub fn current(&self) -> Option<Arc<AppState>> {
        self.current.read().clone()
    }

    /// 2026-09-26: Install the API-key policy. `serve` calls it once, before
    /// any load.
    pub fn set_auth(&self, cfg: Option<Arc<crate::auth::AuthConfig>>) {
        *self.auth.write() = cfg;
    }

    pub fn auth(&self) -> Option<Arc<crate::auth::AuthConfig>> {
        self.auth.read().clone()
    }

    /// 2026-09-26: Install the process-scoped state. `serve` calls it once,
    /// before any load.
    pub(crate) fn set_process(&self, carried: super::serve_load::Carried) {
        *self.process.write() = Some(carried);
    }

    /// 2026-09-26: The process-scoped state, once installed. It outlives every
    /// model, so handlers that read only it need none (`ProcessState`).
    pub(crate) fn process(&self) -> Option<super::serve_load::Carried> {
        self.process.read().clone()
    }

    pub fn rate_limiter(&self) -> Option<Arc<crate::rate_limiter::RateLimiter>> {
        self.process().map(|c| c.rate_limiter)
    }

    /// 2026-09-26: Whether a request naming another model may trigger a load
    /// (`--auto-swap`). Reads under the lock rather than cloning `ServeArgs`
    /// through `args()`, since the chat path asks on every request.
    pub fn auto_swap_enabled(&self) -> bool {
        self.args
            .read()
            .as_ref()
            .is_some_and(super::auto_swap::enabled)
    }

    /// 2026-09-26: Install the dashboard's run-handle channel. `serve` calls it
    /// once, at boot.
    pub(crate) fn set_tui_handles(&self, tx: std::sync::mpsc::Sender<crate::tui::RunHandles>) {
        *self.tui_handles.write() = Some(tx);
    }

    pub(crate) fn tui_handles(&self) -> Option<std::sync::mpsc::Sender<crate::tui::RunHandles>> {
        self.tui_handles.read().clone()
    }

    pub fn set_bound(&self, addr: String, port: u16) {
        *self.bound.write() = Some((addr, port));
    }

    pub fn bound(&self) -> Option<(String, u16)> {
        self.bound.read().clone()
    }

    /// 2026-09-26: The runtime a swap must run inside, if one was in scope at
    /// construction.
    pub fn runtime(&self) -> Option<tokio::runtime::Handle> {
        self.runtime.read().clone()
    }

    /// 2026-09-26: Remove the current model and return it, so the caller can
    /// wait until it holds the last reference before dropping it
    /// (`model_swap::release_state`).
    pub fn take(&self) -> Option<Arc<AppState>> {
        self.current.write().take()
    }

    /// 2026-09-26: Install a newly loaded model. The previous `Arc` stays alive
    /// while any in-flight request still holds it.
    pub fn publish(&self, state: Arc<AppState>) {
        *self.current.write() = Some(state);
    }

    pub fn set_scheduler(&self, handle: std::thread::JoinHandle<()>) {
        *self.scheduler.lock() = Some(handle);
    }

    /// 2026-09-26: Take the current scheduler, for a swap to join.
    pub fn take_scheduler(&self) -> Option<std::thread::JoinHandle<()>> {
        self.scheduler.lock().take()
    }

    /// 2026-09-26: Record what the live model was loaded from, for a restore.
    pub fn set_args(&self, args: crate::cli::ServeArgs) {
        *self.args.write() = Some(args);
    }

    pub fn args(&self) -> Option<crate::cli::ServeArgs> {
        self.args.read().clone()
    }

    /// 2026-09-26: Serialise swaps. `swap` re-checks what is loaded after
    /// acquiring it, since the previous holder may have loaded the same argv.
    pub fn swap_guard(&self) -> parking_lot::MutexGuard<'_, ()> {
        // 2026-09-26: `parking_lot` does not poison, so a panic mid-swap does
        // not block later swaps.
        self.swapping.lock()
    }

    /// 2026-09-26: The served model name (`AppState::model_name`), if a model is
    /// loaded.
    pub fn live_model(&self) -> Option<String> {
        self.current.read().as_ref().map(|s| s.model_name.clone())
    }

    pub fn is_loaded(&self) -> bool {
        self.current.read().is_some()
    }
}

#[cfg(test)]
#[path = "model_host_tests.rs"]
mod tests;

/// 2026-09-26: Extractor for the model serving now. With none loaded, the
/// request is rejected with 503 `model_not_loaded`.
pub struct CurrentModel(pub Arc<AppState>);

impl<S> axum::extract::FromRequestParts<S> for CurrentModel
where
    Arc<ModelHost>: axum::extract::FromRef<S>,
    S: Send + Sync,
{
    type Rejection = axum::response::Response;

    async fn from_request_parts(
        _parts: &mut axum::http::request::Parts,
        state: &S,
    ) -> Result<Self, Self::Rejection> {
        use axum::extract::FromRef;
        use axum::response::IntoResponse;
        let host = Arc::<ModelHost>::from_ref(state);
        match host.current() {
            Some(state) => Ok(Self(state)),
            None => Err((
                axum::http::StatusCode::SERVICE_UNAVAILABLE,
                axum::Json(serde_json::json!({
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
                .into_response()),
        }
    }
}

/// 2026-09-26: Extractor for the process-scoped state, for handlers that need
/// no model, such as those reading the conversation and response stores.
pub(crate) struct ProcessState(pub super::serve_load::Carried);

impl<S> axum::extract::FromRequestParts<S> for ProcessState
where
    Arc<ModelHost>: axum::extract::FromRef<S>,
    S: Send + Sync,
{
    type Rejection = axum::response::Response;

    async fn from_request_parts(
        _parts: &mut axum::http::request::Parts,
        state: &S,
    ) -> Result<Self, Self::Rejection> {
        use axum::extract::FromRef;
        use axum::response::IntoResponse;
        let host = Arc::<ModelHost>::from_ref(state);
        match host.process() {
            Some(carried) => Ok(Self(carried)),
            // 2026-09-26: `serve` installs it before the listener binds; 503
            // rather than a panic if it is ever absent.
            None => Err((
                axum::http::StatusCode::SERVICE_UNAVAILABLE,
                axum::Json(serde_json::json!({
                    "error": {
                        "message": crate::error_hints::message_with_hint(
                            "server is still starting",
                            "not_ready",
                        ),
                        "type": "not_ready",
                        "hint": crate::error_hints::hint_for("not_ready"),
                    }
                })),
            )
                .into_response()),
        }
    }
}
