// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Structured startup-progress events for the TUI.
//!
//! Each function emits one `debug!` event under [`TARGET`]. The TUI's
//! progress layer accepts that target whatever `RUST_LOG` says
//! (`crates/server/src/tui/init.rs`) and decodes the fields.
//!
//! Discriminated by the `ev` field:
//!   `phase`       — `phase` (0..=11), `name`
//!   `preflight`   — `disk_gb`, `free_gb`
//!   `shard_start` — `shard` (1-based), `total`, `name`
//!   `shard_done`  — `shard`, `total`, `used_gb`, `free_gb`
//!   `layer`       — `layer` (layers loaded so far), `total`
//!   `ready`       — `port`
//!
//! Owner: telemetry.
//! Invariants: every event here is emitted under [`TARGET`].

/// 2026-09-26: The target of every event here, which the TUI's progress layer
/// filters on. The `debug!` calls below repeat it as a literal.
pub const TARGET: &str = "metrale::tui::progress";

/// 2026-09-26: A named startup phase has begun; `phase` is its index.
pub fn phase(phase: u8, name: &str) {
    tracing::debug!(target: "metrale::tui::progress", ev = "phase", phase, name);
}

/// 2026-09-26: Weight-load preflight: estimated on-disk weight GB and free GPU GB.
pub fn preflight(disk_gb: f64, free_gb: f64) {
    tracing::debug!(target: "metrale::tui::progress", ev = "preflight", disk_gb, free_gb);
}

/// 2026-09-26: A safetensors shard load has started.
pub fn shard_start(shard: usize, total: usize, name: &str) {
    tracing::debug!(target: "metrale::tui::progress", ev = "shard_start", shard, total, name);
}

/// 2026-09-26: A shard finished, with GPU memory used and free in GB.
pub fn shard_done(shard: usize, total: usize, used_gb: f64, free_gb: f64) {
    tracing::debug!(
        target: "metrale::tui::progress",
        ev = "shard_done",
        shard,
        total,
        used_gb,
        free_gb
    );
}

/// 2026-09-26: Layer-build progress, sampled by the weight loaders.
pub fn layer(layer: usize, total: usize) {
    tracing::debug!(target: "metrale::tui::progress", ev = "layer", layer, total);
}

/// 2026-09-26: The server is listening.
pub fn ready(port: u16) {
    tracing::debug!(target: "metrale::tui::progress", ev = "ready", port);
}
