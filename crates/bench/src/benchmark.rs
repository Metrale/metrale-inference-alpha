// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The [`Benchmark`] trait, a [`Plugin`] driven one step at a time,
//! and [`BenchmarkDescriptor`], its static identity.
//!
//! A benchmark owns its phase state and does one step of work per
//! [`Benchmark::next`], returning a frame. It must not block the runtime or
//! loop to completion inside one step: the driver checks for cancellation only
//! between frames. [`Benchmark::run`] delegates to [`crate::executor::drive`],
//! the same loop the executor uses.
//!
//! Owner: bench.
//! Invariants: none beyond the types.

use std::future::Future;

use anyhow::Result;
use futures::Stream;

use crate::dynamic::DynBenchmark;
use crate::hardware::Sensitivity;
use crate::params::{ParamSpec, ParamValues};
use crate::plugin::Plugin;
use crate::result::BenchmarkResult;

/// 2026-09-26: Static identity of a benchmark, and how to construct one. The
/// registry (`registry::ALL`), the TUI list pane and the run directories
/// (`ArtifactStore::runs_dir`) all read it.
pub struct BenchmarkDescriptor {
    /// 2026-09-26: Stable and filename-safe (`[a-z0-9-]`, checked by
    /// `registry::tests::ids_are_unique_and_filename_safe`). Names the run
    /// directory `<home>/runs/<id>/`.
    pub id: &'static str,
    pub name: &'static str,
    /// 2026-09-26: One line for the suite list.
    pub summary: &'static str,
    /// 2026-09-26: A paragraph for the detail pane: what it measures and what
    /// it costs.
    pub detail: &'static str,
    /// 2026-09-26: Rough wall time at default parameters, for display, e.g.
    /// `"~15 min"`.
    pub duration_hint: &'static str,
    /// 2026-09-26: Expected wall time at default parameters, in seconds: the
    /// certification planner's estimate when no completed run of this id has
    /// been measured (`bench_certify::plan::units`). Zero only on descriptors
    /// whose `duration_hint` starts with `unrunnable`
    /// (`registry::tests::every_runnable_benchmark_declares_an_expected_duration`).
    pub expected_secs: u64,
    /// 2026-09-26: When this benchmark's measurement last changed (new
    /// thresholds, prompt set or scoring rule), as `YYYY-MM-DD`; not when its
    /// code was edited. It tells a reader whether two runs are comparable.
    pub updated: &'static str,
    /// 2026-09-26: True when starting has a side effect beyond load on the
    /// endpoint. The TUI (`tui::bench_keys`) and `bench certify` (`--yes`)
    /// require an explicit confirmation for these.
    pub needs_confirmation: bool,
    /// 2026-09-26: The checkpoint families this benchmark is defined on, if
    /// any. The executor's endpoint check (`coherence::probe_for`) warns on a
    /// mismatch and never refuses the run. `None` means the benchmark measures
    /// whatever it is pointed at.
    pub intended_for: Option<ModelExpectation>,
    /// 2026-09-26: Parameters whose run-time value comes from the selected
    /// model variant's committed bound, as `(param key, metric key)` pairs, for
    /// benchmarks that compute their own verdict against a knob that is also a
    /// `BENCH.toml` threshold (the agentic gate's `wall_budget_s` against
    /// `sum_wall_s`). A gate run (`bench_resolve::apply_threshold_params`) and
    /// the TUI derive the value from the variant's bound; an explicit `--param`
    /// wins.
    pub threshold_params: &'static [(&'static str, &'static str)],
    /// 2026-09-26: Whether this benchmark's number is a speed number, and so
    /// corruptible by the state of the box that produced it. A required field
    /// of the descriptor, not a list in the hardware policy, so every new
    /// benchmark has to answer it. See [`crate::hardware::policy`].
    pub sensitivity: Sensitivity,
    pub ctor: fn() -> Box<dyn DynBenchmark>,
}

/// 2026-09-26: Which checkpoints a benchmark's numbers mean something for.
#[derive(Clone, Copy, Debug)]
pub struct ModelExpectation {
    /// 2026-09-26: Lower-case substrings identifying an acceptable checkpoint
    /// family, not an exact id: [`ModelExpectation::accepts`] matches any model
    /// id whose lower-cased form contains one of them, whatever its org prefix
    /// or quantization suffix.
    pub families: &'static [&'static str],
    /// 2026-09-26: Appended to the mismatch warning: which model the benchmark
    /// is defined on, and what running it elsewhere means.
    pub note: &'static str,
}

impl ModelExpectation {
    /// 2026-09-26: Does `model` belong to a family this benchmark is defined on?
    pub fn accepts(&self, model: &str) -> bool {
        let lowered = model.to_lowercase();
        self.families.iter().any(|f| lowered.contains(f))
    }
}

impl BenchmarkDescriptor {
    pub fn build(&self) -> Box<dyn DynBenchmark> {
        (self.ctor)()
    }
}

pub trait Benchmark: Plugin {
    fn descriptor(&self) -> &'static BenchmarkDescriptor;

    /// 2026-09-26: The parameters shown before the run starts, so the user can
    /// change them. Their defaults are in the returned specs.
    fn parameters(&self) -> Vec<ParamSpec>;

    /// 2026-09-26: Receive the edited values. Validate here and return an
    /// error naming the offending field; the executor calls `next()` only after
    /// this returns `Ok`.
    fn configure(&mut self, values: &ParamValues) -> Result<()>;

    /// 2026-09-26: Drive `next()` to completion, streaming every frame. The
    /// stream ends after the first terminal [`crate::RunStatus`], or after an
    /// error. It does not call `cleanup()`.
    fn run(&mut self) -> impl Stream<Item = Result<BenchmarkResult>> + '_
    where
        Self: Sized + Send,
    {
        crate::executor::drive(self)
    }

    /// 2026-09-26: One step of work, returning the frame for that step.
    fn next(&mut self) -> impl Future<Output = Result<BenchmarkResult>> + Send;

    /// 2026-09-26: Release whatever the run acquired. The executor calls it
    /// once on every return path of a run: completion, failure, cancellation,
    /// a failed setup and a refused hardware precheck.
    fn cleanup(&mut self) -> impl Future<Output = Result<()>> + Send {
        async { Ok(()) }
    }
}

#[cfg(test)]
#[path = "benchmark_desc_tests.rs"]
mod desc_tests;
