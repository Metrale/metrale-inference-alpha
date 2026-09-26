// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `Carried`, the state that outlives any one model: it passes from
//! the model being replaced to the next one on a swap.
//!
//! Owner: server startup (`met serve`).
//! Invariants: `load_model` moves each field into `AppState` unchanged.

use anyhow::Result;

use crate::main_modules::AppState;
use crate::{conversation_store, rate_limiter, response_store};

/// 2026-09-26: State that outlives any model: the response store, the rate
/// limiter and the conversation store. `load_model` takes it as a parameter,
/// so a swap passes the previous model's instances in (`from_previous`)
/// instead of building empty ones.
#[derive(Clone)]
pub(crate) struct Carried {
    pub response_store: std::sync::Arc<response_store::ResponseStore>,
    pub rate_limiter: std::sync::Arc<rate_limiter::RateLimiter>,
    pub conversation_store: std::sync::Arc<conversation_store::ConversationStore>,
}

impl Carried {
    /// 2026-09-26: First boot: build each store and the limiter once, from the
    /// environment. `serve` installs the result on the `ModelHost` before
    /// startup and before the listener binds, so handlers that need no model
    /// reach them, and handlers refund through the same limiter the middleware
    /// debits.
    ///
    /// # Errors
    /// When a `METRALE_RATE_LIMIT_*`, `METRALE_STORE_*` or
    /// `METRALE_CONVERSATION_*` variable is set to a value that does not parse.
    pub fn from_env() -> Result<Self, String> {
        Ok(Self {
            response_store: response_store::ResponseStore::from_env()?,
            rate_limiter: rate_limiter::RateLimiter::from_env()?,
            conversation_store: conversation_store::ConversationStore::from_env()?,
        })
    }

    /// 2026-09-26: A swap: take them from the model being replaced.
    pub fn from_previous(previous: &AppState) -> Self {
        Self {
            response_store: previous.response_store.clone(),
            rate_limiter: previous.rate_limiter.clone(),
            conversation_store: previous.conversation_store.clone(),
        }
    }
}
