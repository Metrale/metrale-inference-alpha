// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Engine telemetry: run metrics, kernel audit, launch trace,
//! startup progress events, and the metal-up instrument set.
//!
//! # Levels
//! [`Level::Off`] (where a new [`Telemetry`] starts), [`Level::Basic`] and
//! [`Level::Kernel`].
//!
//! # Layers
//! | Layer | Module | Content |
//! |---|---|---|
//! | L0 device | [`device`], [`sampler`] | NVML via dlopen, sampled at 10 Hz when serving: power, energy counter, clocks, temperature, clock-event reasons, memory, PCIe, NVLink |
//! | L1 kernel | [`kernel`] | CUDA-event spans per serve lane each step; per kernel one step in N |
//! | L2 scheduler | [`sched`] | queue depths, batch shape, loop phases, blocking syncs/D2H, graph replays/captures |
//! | L3 cache | [`sched`], [`prefix_cache`] | KV blocks, SSM slots, prefix-cache hits/misses |
//! | L4 spec | [`sched::SpecMatrix`] | verify acceptance by draft depth |
//! | L5 request | [`request`] | TTFT, TPOT, end-to-end |
//! | Energy | [`energy`] | live J/token and J/request from NVML counter deltas |
//!
//! Exported by [`export`]: Prometheus text, JSONL events, and OTLP behind a
//! feature.
//!
//! Owner: telemetry.
//! Invariants: at [`Level::Off`], every hot-path entry point of [`Telemetry`]
//! returns after one relaxed atomic load, with no clock read and no allocation
//! (`tests/zero_cost_off.rs`).

pub mod clock;
pub mod device;
pub mod energy;
pub mod export;
pub mod hub;
pub mod instrument;
pub mod kernel;
pub mod kernel_audit;
pub mod launch_trace;
pub mod level;
pub mod prefix_cache;
pub mod progress;
pub mod request;
pub mod run_metrics;
pub mod sampler;
pub mod sched;
pub mod seqcell;
pub mod snapshot;

#[cfg(test)]
mod nvml_script;

pub use hub::{DeviceState, Telemetry, global};
pub use level::{ConfigError, Level, TelemetryConfig};
