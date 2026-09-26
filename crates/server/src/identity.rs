// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The names the engine gives itself on the wire.
//!
//! Owner: server.
//! Invariants: none beyond the types.

/// 2026-09-26: The `User-Agent` of Hugging Face downloads, recipe registry
/// fetches and the issue-report service. The remote-image fetch and the OTLP
/// push do not send it.
pub(crate) const USER_AGENT: &str = concat!("metrale/", env!("CARGO_PKG_VERSION"));

/// 2026-09-26: `owned_by` in `/v1/models` list entries and in the
/// `/v1/models/{id}` response.
pub(crate) const OWNED_BY: &str = "metrale";
