// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Driving a benchmark without a terminal: start it on the
//! executor, drain its messages until it finishes, build the run record and
//! save it with [`crate::history::save`], as the dashboard also does.
//!
//! Owner: bench.
//! Invariants:
//! - [`run_blocking`] validates the parameters before the executor starts.
//! - A run that ends without a terminal frame is recorded as a failure.

use std::path::PathBuf;
use std::time::Duration;

use anyhow::Result;

use crate::benchmark::BenchmarkDescriptor;
use crate::coherence::CoherencePolicy;
use crate::executor::{BenchmarkExecutor, ExecutorMessage};
use crate::history::{self, RunRecord, RunSource};
use crate::params::ParamValues;
use crate::plugin::{PluginEvent, TargetEndpoint};
use crate::result::{BenchmarkResult, VerdictKind};

/// 2026-09-26: How the driver behaves, as opposed to what it runs.
#[derive(Clone, Debug)]
pub struct HeadlessOptions {
    /// 2026-09-26: Sleep between drains, and so between cancellation checks.
    pub poll: Duration,
    /// 2026-09-26: Save the run into the executor's artifact store, under
    /// `runs/<benchmark>`.
    pub save: bool,
    pub source: RunSource,
    /// 2026-09-26: Recorded on the run so a result can be traced to the build.
    pub metrale_version: String,
    /// 2026-09-26: Whether the executor probes the endpoint's coherence before
    /// measuring.
    pub coherence: CoherencePolicy,
    /// 2026-09-26: The box class's temperature ceilings for the hardware
    /// pre-check (`hardware::limits`), when the caller could read them.
    pub temp_ceilings: Option<crate::hardware::policy::TempCeilings>,
}

impl HeadlessOptions {
    pub fn cli(metrale_version: impl Into<String>) -> Self {
        Self {
            poll: Duration::from_millis(250),
            save: true,
            source: RunSource::Cli,
            metrale_version: metrale_version.into(),
            coherence: CoherencePolicy::Probe,
            temp_ceilings: None,
        }
    }
}

/// 2026-09-26: What to run, and against what.
pub struct RunRequest {
    pub descriptor: &'static BenchmarkDescriptor,
    pub values: ParamValues,
    /// 2026-09-26: Where to measure. Its `serve_overrides` are copied into
    /// the run record.
    pub target: TargetEndpoint,
    pub options: HeadlessOptions,
}

/// 2026-09-26: Callbacks for a run as it happens. Every method defaults to
/// nothing.
pub trait RunReporter {
    fn started(&mut self, _request: &RunRequest) {}
    fn event(&mut self, _event: &PluginEvent) {}
    fn frame(&mut self, _frame: &BenchmarkResult) {}
}

/// 2026-09-26: A reporter that says nothing: the CLI's JSON output format
/// and the tests use it.
pub struct SilentReporter;
impl RunReporter for SilentReporter {}

#[derive(Debug)]
pub struct RunOutcome {
    pub record: RunRecord,
    /// 2026-09-26: Where it was written, when `options.save`.
    pub saved_to: Option<PathBuf>,
    pub cancelled: bool,
}

impl RunOutcome {
    /// 2026-09-26: `1` when the run was cancelled or did not complete, `2` when
    /// it completed with a failing verdict, else `0`.
    pub fn exit_code(&self) -> i32 {
        if self.cancelled
            || !matches!(
                self.record.frame.status,
                crate::result::RunStatus::Completed
            )
        {
            return 1;
        }
        match self.record.verdict_kind() {
            Some(VerdictKind::Fail) => 2,
            _ => 0,
        }
    }
}

/// 2026-09-26: Run to completion, pumping the executor's channels. Blocks the
/// calling thread; from async, use `spawn_blocking`.
///
/// `should_cancel` is polled once per drain; the first true cancels the run.
/// Errors on invalid parameters, before the run starts, and when saving
/// fails.
pub fn run_blocking(
    executor: &BenchmarkExecutor,
    request: RunRequest,
    reporter: &mut dyn RunReporter,
    should_cancel: &dyn Fn() -> bool,
) -> Result<RunOutcome> {
    let specs = request.descriptor.build().parameters();
    request.values.validate_against(&specs)?;

    reporter.started(&request);
    let run = executor.start(
        request.descriptor,
        request.values.clone(),
        request.target.clone(),
        request.options.coherence,
        request.options.temp_ceilings,
    );

    let mut terminal: Option<BenchmarkResult> = None;
    loop {
        for message in run.drain() {
            dispatch(message, reporter, &mut terminal);
        }
        if should_cancel() && !run.is_cancelled() {
            run.cancel();
            reporter.event(&PluginEvent::Status(
                "cancelling — stopping after the request in flight".into(),
            ));
        }
        if run.is_finished() {
            break;
        }
        std::thread::sleep(request.options.poll);
    }

    // 2026-09-26: The executor sets `finished` in `teardown`, after its last
    // send, so drain until a drain comes back empty.
    loop {
        let messages = run.drain();
        if messages.is_empty() {
            break;
        }
        for message in messages {
            dispatch(message, reporter, &mut terminal);
        }
        std::thread::sleep(Duration::from_millis(10));
    }

    let cancelled = run.is_cancelled();
    let frame = terminal.unwrap_or_else(|| {
        BenchmarkResult::failed(
            "run",
            "the run ended without a terminal frame",
            Duration::ZERO,
        )
    });

    let mut record = RunRecord::new(
        request.descriptor,
        &request.values,
        &request.target,
        request.target.serve_overrides.clone(),
        request.options.source,
        &request.options.metrale_version,
        frame,
    );
    let saved_to = if request.options.save {
        Some(history::save(executor.artifacts(), &mut record)?)
    } else {
        None
    };

    Ok(RunOutcome {
        record,
        saved_to,
        cancelled,
    })
}

fn dispatch(
    message: ExecutorMessage,
    reporter: &mut dyn RunReporter,
    terminal: &mut Option<BenchmarkResult>,
) {
    match message {
        ExecutorMessage::Event(e) => reporter.event(&e),
        ExecutorMessage::Frame(f) => {
            reporter.frame(&f);
            if f.status.is_terminal() {
                *terminal = Some(*f);
            }
        }
    }
}

#[cfg(test)]
#[path = "headless_tests.rs"]
mod tests;
