// SPDX-License-Identifier: MIT OR Apache-2.0
#![deny(warnings)]
#![deny(clippy::all)]

//! 2026-09-26: Metrale Engine plugins, and the benchmark suite the `met serve`
//! TUI and `met benchmark` drive.
//!
//! [`Plugin`] is the general abstraction: [`Plugin::load`] receives a
//! [`PluginHandle`] (status, log, progress and glow events, the artifact
//! store, the target endpoint and a cancellation flag). [`Benchmark`]
//! specialises it into a drivable state machine: the implementor does one
//! step per [`Benchmark::next`], and [`Benchmark::run`] streams the
//! [`BenchmarkResult`] frames.
//!
//! `impl Stream` and `async fn` in a trait are not dyn-compatible, so
//! [`DynBenchmark`] carries the same contract with boxed futures and is
//! blanket-implemented for every `Benchmark`; [`executor::drive`] consumes
//! it.
//!
//! Owner: bench.
//! Invariants: none beyond the types.

pub mod artifacts;
pub mod benchmark;
pub mod benchmarks;
pub mod coherence;
pub mod dynamic;
pub mod executor;
pub mod gate;
pub mod hardware;
pub mod headless;
pub mod history;
pub mod http;
pub mod metadata;
pub mod param_text;
pub mod params;
pub mod plugin;
pub mod python;
pub mod registry;
pub mod result;
pub mod serve_env;
pub mod serve_identity;

pub use artifacts::ArtifactStore;
pub use benchmark::{Benchmark, BenchmarkDescriptor};
pub use coherence::CoherencePolicy;
pub use dynamic::DynBenchmark;
pub use executor::{BenchmarkExecutor, ExecutorMessage, RunHandle};
pub use hardware::{Hardware, HardwareState, HardwareStateReport, Sensitivity};
pub use history::{RunRecord, RunSource};
pub use metadata::PluginMetadata;
pub use params::{ParamKind, ParamSpec, ParamValue, ParamValues};
pub use plugin::{Plugin, PluginEvent, PluginHandle, TargetEndpoint};
pub use result::{
    Align, BenchmarkResult, Cell, CellStyle, Column, LogLevel, LogLine, ResultTable, RunStatus,
    Stat, Verdict, VerdictKind,
};
