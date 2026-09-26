// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Exporters. Each renders a [`crate::snapshot::TelemetrySnapshot`]
//! or the instruments behind it into a `String`; the server does the I/O.
//!
//! * [`prometheus`]: the telemetry section of `GET /metrics`.
//! * [`events`]: one JSON object per line for `GET /v1/events` (SSE).
//! * `otlp` (cargo feature `otlp`): an OTLP/HTTP JSON metrics body.
//!
//! Owner: telemetry.
//! Invariants: no exporter performs I/O.

pub mod events;
#[cfg(feature = "otlp")]
pub mod otlp;
pub mod prometheus;
mod prometheus_layers;
mod prometheus_text;
