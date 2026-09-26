// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Log capture for the log pane: a `tracing` layer that stores each event as a [`LogLine`] in a process-global ring.
//!
//! Owner: server tui.
//! Invariants:
//! - The ring holds at most `CAP` lines; at capacity the oldest is dropped.
//! - Events whose target is `metrale_telemetry::progress::TARGET` are not stored.

use std::collections::VecDeque;
use std::io::Write;
use std::time::SystemTime;

use parking_lot::Mutex;
use tracing::Level;
use tracing::field::{Field, Visit};

const CAP: usize = 10_000;

#[derive(Clone, Debug)]
pub struct LogLine {
    pub at: SystemTime,
    pub level: Level,
    pub target: String,
    pub message: String,
}

/// 2026-09-26: The captured lines and the count of lines ever pushed.
struct Captured {
    lines: Mutex<VecDeque<LogLine>>,
    /// 2026-09-26: Lines ever pushed; keeps rising after the ring is full.
    seq: std::sync::atomic::AtomicU64,
}

/// 2026-09-26: A static because it backs a `tracing` layer, installed once per
/// process and fed from every thread, and because the panic hook
/// (`terminal_guard::install_panic_hook`) dumps it with no handle in hand.
static CAPTURED: Captured = Captured {
    lines: Mutex::new(VecDeque::new()),
    seq: std::sync::atomic::AtomicU64::new(0),
};

fn push(line: LogLine) {
    let mut ring = CAPTURED.lines.lock();
    if ring.len() == CAP {
        ring.pop_front();
    }
    ring.push_back(line);
    CAPTURED
        .seq
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// 2026-09-26: Total lines ever captured, including evicted ones.
pub fn seq() -> u64 {
    CAPTURED.seq.load(std::sync::atomic::Ordering::Relaxed)
}

/// 2026-09-26: Clone out the newest `n` lines, oldest first.
pub fn tail(n: usize) -> Vec<LogLine> {
    let ring = CAPTURED.lines.lock();
    let skip = ring.len().saturating_sub(n);
    ring.iter().skip(skip).cloned().collect()
}

/// 2026-09-26: Panic-hook dump: the newest `n` lines as plain text to `w`.
/// Write errors are ignored.
pub fn dump_to(w: &mut dyn Write, n: usize) {
    for l in tail(n) {
        let _ = writeln!(w, "{:>5} {} {}", l.level, l.target, l.message);
    }
}

/// 2026-09-26: Extracts the `message` field of an event.
#[derive(Default)]
struct MessageVisitor(String);

impl Visit for MessageVisitor {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            use std::fmt::Write as _;
            let _ = write!(self.0, "{value:?}");
        }
    }
    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "message" {
            self.0.push_str(value);
        }
    }
}

/// 2026-09-26: The capture layer. `init::install_tty_subscriber` gives it the
/// same `EnvFilter` as the fmt layer.
pub struct LogRingLayer;

impl<S> tracing_subscriber::Layer<S> for LogRingLayer
where
    S: tracing::Subscriber,
{
    fn on_event(
        &self,
        event: &tracing::Event<'_>,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        // 2026-09-26: Progress events have their own layer
        // (`capture_layer`); keep them out of the pane.
        if event.metadata().target() == metrale_telemetry::progress::TARGET {
            return;
        }
        let mut v = MessageVisitor::default();
        event.record(&mut v);
        push(LogLine {
            at: SystemTime::now(),
            level: *event.metadata().level(),
            target: event.metadata().target().to_string(),
            message: v.0,
        });
    }
}

#[cfg(test)]
#[path = "log_ring_tests.rs"]
mod tests;
