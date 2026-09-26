// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Decodes `metrale::tui::progress` events into typed [`ProgressEvent`]s.
//!
//! `init::install_tty_subscriber` attaches the layer with its own filter for
//! that target, so progress flows whatever `RUST_LOG` says. The events are
//! `debug!`, so the log layers' default `info` filter hides them.
//!
//! Owner: server tui.
//! Invariants:
//! - An event with an unknown `ev` is dropped, never sent.

use std::sync::mpsc::Sender;

use tracing::field::{Field, Visit};

/// 2026-09-26: Typed startup-progress event, one variant per `ev` value that
/// `metrale_telemetry::progress` emits.
#[derive(Clone, Debug, PartialEq)]
pub enum ProgressEvent {
    Phase {
        phase: u8,
        name: String,
    },
    Preflight {
        disk_gb: f64,
        free_gb: f64,
    },
    ShardStart {
        shard: u64,
        total: u64,
        name: String,
    },
    ShardDone {
        shard: u64,
        total: u64,
        used_gb: f64,
        free_gb: f64,
    },
    Layer {
        layer: u64,
        total: u64,
    },
    Ready {
        port: u16,
    },
}

/// 2026-09-26: Field bag filled by the visitor; `decode` shapes it by `ev`.
#[derive(Default)]
struct Fields {
    ev: String,
    name: String,
    phase: u64,
    shard: u64,
    total: u64,
    layer: u64,
    port: u64,
    disk_gb: f64,
    free_gb: f64,
    used_gb: f64,
}

impl Visit for Fields {
    fn record_u64(&mut self, field: &Field, value: u64) {
        match field.name() {
            "phase" => self.phase = value,
            "shard" => self.shard = value,
            "total" => self.total = value,
            "layer" => self.layer = value,
            "port" => self.port = value,
            _ => {}
        }
    }
    fn record_i64(&mut self, field: &Field, value: i64) {
        self.record_u64(field, value.max(0) as u64);
    }
    fn record_f64(&mut self, field: &Field, value: f64) {
        match field.name() {
            "disk_gb" => self.disk_gb = value,
            "free_gb" => self.free_gb = value,
            "used_gb" => self.used_gb = value,
            _ => {}
        }
    }
    fn record_str(&mut self, field: &Field, value: &str) {
        match field.name() {
            "ev" => self.ev = value.to_string(),
            "name" => self.name = value.to_string(),
            _ => {}
        }
    }
    fn record_debug(&mut self, _field: &Field, _value: &dyn std::fmt::Debug) {}
}

fn decode(f: Fields) -> Option<ProgressEvent> {
    Some(match f.ev.as_str() {
        "phase" => ProgressEvent::Phase {
            phase: f.phase.min(u8::MAX as u64) as u8,
            name: f.name,
        },
        "preflight" => ProgressEvent::Preflight {
            disk_gb: f.disk_gb,
            free_gb: f.free_gb,
        },
        "shard_start" => ProgressEvent::ShardStart {
            shard: f.shard,
            total: f.total,
            name: f.name,
        },
        "shard_done" => ProgressEvent::ShardDone {
            shard: f.shard,
            total: f.total,
            used_gb: f.used_gb,
            free_gb: f.free_gb,
        },
        "layer" => ProgressEvent::Layer {
            layer: f.layer,
            total: f.total,
        },
        "ready" => ProgressEvent::Ready {
            port: f.port.min(u16::MAX as u64) as u16,
        },
        _ => return None,
    })
}

/// 2026-09-26: The layer: the send end of a channel the TUI event loop drains.
/// Sends never block, and a send to a closed channel is ignored.
pub struct ProgressCaptureLayer {
    tx: Sender<ProgressEvent>,
}

impl ProgressCaptureLayer {
    pub fn new(tx: Sender<ProgressEvent>) -> Self {
        Self { tx }
    }
}

impl<S> tracing_subscriber::Layer<S> for ProgressCaptureLayer
where
    S: tracing::Subscriber,
{
    fn on_event(
        &self,
        event: &tracing::Event<'_>,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        if event.metadata().target() != metrale_telemetry::progress::TARGET {
            return;
        }
        let mut f = Fields::default();
        event.record(&mut f);
        if let Some(ev) = decode(f) {
            let _ = self.tx.send(ev);
        }
    }
}

#[cfg(test)]
#[path = "capture_layer_tests.rs"]
mod tests;
