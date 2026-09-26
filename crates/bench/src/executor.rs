// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: [`drive`], the loop that turns a benchmark into a stream of
//! frames, and [`BenchmarkExecutor`], which runs a benchmark's lifecycle as a
//! task on a given tokio runtime.
//!
//! The run's frames and plugin events cross to the caller over
//! `std::sync::mpsc` channels, which [`RunHandle::drain`] reads without
//! blocking, so a render loop never awaits the run.
//!
//! Owner: bench.
//! Invariants:
//! - [`drive`] yields nothing after the first terminal frame or the first
//!   error.
//! - Every return path of a run's task calls the benchmark's `cleanup()` once
//!   and then sets [`RunHandle::is_finished`].

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::time::{Duration, Instant};

use anyhow::Result;
use futures::{Stream, StreamExt};

use crate::artifacts::ArtifactStore;
use crate::benchmark::BenchmarkDescriptor;
use crate::coherence::CoherencePolicy;
use crate::dynamic::DynBenchmark;
use crate::hardware::policy;
use crate::hardware::report::HardwareStateReport;
use crate::hardware::state::HardwareState;
use crate::params::ParamValues;
use crate::plugin::{PluginEvent, PluginHandle, TargetEndpoint};
use crate::result::{BenchmarkResult, LogLine, RunStatus};

/// 2026-09-26: Drive `bench` by calling `next()` until it reports a terminal
/// status or errors. `Benchmark::run`'s default body and the executor both use
/// it.
pub fn drive(bench: &mut dyn DynBenchmark) -> impl Stream<Item = Result<BenchmarkResult>> + '_ {
    futures::stream::unfold(Some(bench), |state| async move {
        let bench = state?;
        match bench.next().await {
            Ok(frame) => {
                let finished = frame.status.is_terminal();
                Some((Ok(frame), if finished { None } else { Some(bench) }))
            }
            // 2026-09-26: An `Err` ends the stream.
            Err(e) => Some((Err(e), None)),
        }
    })
}

/// 2026-09-26: One message drained from a run. `Frame` is boxed so that an
/// `Event` does not carry the size of a [`BenchmarkResult`].
pub enum ExecutorMessage {
    Event(PluginEvent),
    Frame(Box<BenchmarkResult>),
}

/// 2026-09-26: The caller's control surface for one in-flight run.
pub struct RunHandle {
    events: Receiver<PluginEvent>,
    frames: Receiver<BenchmarkResult>,
    cancel: Arc<AtomicBool>,
    finished: Arc<AtomicBool>,
}

impl RunHandle {
    /// 2026-09-26: Non-blocking drain of everything queued: all events first,
    /// then all frames.
    pub fn drain(&self) -> Vec<ExecutorMessage> {
        let mut out: Vec<ExecutorMessage> =
            self.events.try_iter().map(ExecutorMessage::Event).collect();
        out.extend(
            self.frames
                .try_iter()
                .map(|f| ExecutorMessage::Frame(Box::new(f))),
        );
        out
    }

    /// 2026-09-26: Ask the run to stop. The benchmark can observe this through
    /// `PluginHandle::check_cancelled`, and the executor checks it after each
    /// frame and then ends the run with a "cancelled" frame.
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }

    /// 2026-09-26: True once the run's `cleanup()` has returned.
    pub fn is_finished(&self) -> bool {
        self.finished.load(Ordering::Relaxed)
    }
}

/// 2026-09-26: Starts benchmark runs on an existing tokio runtime.
#[derive(Clone)]
pub struct BenchmarkExecutor {
    runtime: tokio::runtime::Handle,
    artifacts: ArtifactStore,
    /// 2026-09-26: The next run id for `PluginHandle`, counted from 1 and
    /// shared by every clone of this executor, so ids are unique among the
    /// runs one executor starts. See `PluginHandle::run_id`.
    next_run_id: Arc<std::sync::atomic::AtomicU64>,
}

impl BenchmarkExecutor {
    pub fn new(runtime: tokio::runtime::Handle, artifacts: ArtifactStore) -> Self {
        Self {
            runtime,
            artifacts,
            next_run_id: Arc::new(std::sync::atomic::AtomicU64::new(1)),
        }
    }

    /// 2026-09-26: The runtime this executor spawns runs on, for callers that
    /// need to run a short async check on the same one.
    pub fn runtime(&self) -> &tokio::runtime::Handle {
        &self.runtime
    }

    pub fn artifacts(&self) -> &ArtifactStore {
        &self.artifacts
    }

    /// 2026-09-26: Spawn the run: hardware precheck, coherence probe, build,
    /// load, configure, drive, clean up. `cleanup()` runs on every return path
    /// of the task, including a refused precheck, a failed `load()` and
    /// cancellation.
    pub fn start(
        &self,
        descriptor: &'static BenchmarkDescriptor,
        values: ParamValues,
        target: TargetEndpoint,
        coherence: CoherencePolicy,
        ceilings: Option<crate::hardware::policy::TempCeilings>,
    ) -> RunHandle {
        let (event_tx, event_rx) = channel();
        let (frame_tx, frame_rx) = channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let finished = Arc::new(AtomicBool::new(false));
        let handle = PluginHandle::new(
            self.next_run_id
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            target,
            self.artifacts.clone(),
            event_tx.clone(),
            cancel.clone(),
        );
        let task = RunTask {
            descriptor,
            values,
            coherence,
            handle,
            events: event_tx,
            frames: frame_tx,
            cancel: cancel.clone(),
            finished: finished.clone(),
            ceilings,
        };
        self.runtime.spawn(task.execute());
        RunHandle {
            events: event_rx,
            frames: frame_rx,
            cancel,
            finished,
        }
    }
}

/// 2026-09-26: The per-request timeout of the coherence probe, whose
/// questions ask for at most 96 tokens (`coherence::ask`).
const COHERENCE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

struct RunTask {
    descriptor: &'static BenchmarkDescriptor,
    values: ParamValues,
    handle: PluginHandle,
    events: Sender<PluginEvent>,
    frames: Sender<BenchmarkResult>,
    cancel: Arc<AtomicBool>,
    finished: Arc<AtomicBool>,
    coherence: CoherencePolicy,
    /// 2026-09-26: The box class's temperature ceilings for the pre-run
    /// capture, when the caller has them.
    ceilings: Option<crate::hardware::policy::TempCeilings>,
}

impl RunTask {
    /// 2026-09-26: Run the coherence probe unless the policy is `Skip`. Any
    /// concern becomes a warning in the run log; nothing here stops the run.
    async fn probe_coherence(&self) {
        if self.coherence == CoherencePolicy::Skip {
            return;
        }
        self.handle.status("checking the endpoint".to_string());
        let report = crate::coherence::probe_for(
            self.handle.target(),
            self.descriptor.intended_for,
            COHERENCE_TIMEOUT,
        )
        .await;
        match report.concern(self.handle.target()) {
            Some(concern) => self.handle.warn(concern),
            None => {
                for a in &report.answers {
                    self.handle
                        .info(format!("endpoint check {}: {:?}", a.label, a.answer.trim()));
                }
            }
        }
    }

    /// 2026-09-26: Capture the box state and open the hardware report, whose
    /// precheck decides whether this run may start.
    ///
    /// `None` when the capture task failed (it panicked or was cancelled). The
    /// capture runs on `spawn_blocking` because `HardwareState::collect` runs
    /// `nvidia-smi` as a synchronous subprocess.
    async fn open_hardware_report(&self) -> Option<HardwareStateReport> {
        let sensitivity = self.descriptor.sensitivity;
        let before = tokio::task::spawn_blocking(HardwareState::collect)
            .await
            .ok()?;
        self.handle
            .info(format!("box state: {}", before.one_line()));
        let report = HardwareStateReport::opened(sensitivity, before, self.ceilings);
        if policy::PolicyOptions::from_env().kill_switch {
            // 2026-09-26: Warned on every run while the kill switch is set, so
            // the run log of every number shows it.
            self.handle.warn(format!(
                "{}=1 — the hardware pre-check CANNOT refuse this run. Whatever it found is \
                 recorded below and travels with the record.",
                policy::KILL_SWITCH_ENV
            ));
        }
        for concern in &report.precheck.concerns {
            self.handle.warn(format!("hardware: {concern}"));
        }
        Some(report)
    }

    /// 2026-09-26: Take the after-capture, close the report and attach it to
    /// the terminal frame. Without an open report the frame is returned as is;
    /// a failed after-capture leaves the report unclosed.
    async fn close_hardware(
        &self,
        mut frame: BenchmarkResult,
        report: Option<HardwareStateReport>,
    ) -> BenchmarkResult {
        let Some(mut report) = report else {
            return frame;
        };
        if let Ok(after) = tokio::task::spawn_blocking(HardwareState::collect).await {
            report.close(after);
        }
        if let Some(post) = &report.postcheck {
            for concern in &post.concerns {
                self.handle.warn(format!("hardware: {concern}"));
            }
        }
        // 2026-09-26: Only a completed run has numbers to invalidate.
        if report.invalidated() && frame.status == RunStatus::Completed {
            self.handle.warn(
                "hardware: this run's SPEED numbers are marked INVALID — the box throttled \
                 while it was measuring. Re-run on a box that does not."
                    .to_string(),
            );
        }
        frame.hardware_state = Some(report);
        frame
    }

    async fn execute(self) {
        let started = Instant::now();
        self.handle.set_glow(true);
        let mut bench = self.descriptor.build();

        // 2026-09-26: The hardware precheck comes first, so a refused run has
        // touched neither the endpoint nor the artifacts.
        let hardware = self.open_hardware_report().await;
        if let Some(report) = hardware.as_ref().filter(|r| r.refuses()) {
            let frame = BenchmarkResult::failed(
                "hardware precheck",
                format!(
                    "this box is not in a state to produce a comparable SPEED number: {}. \
                     Set {}=1 to measure anyway.",
                    report.precheck.concerns.join("; "),
                    policy::KILL_SWITCH_ENV
                ),
                started.elapsed(),
            );
            // 2026-09-26: The report is attached unclosed: nothing ran, so
            // there is no after-capture and no validity to judge.
            let _ = self.frames.send(frame.with_hardware_state(report.clone()));
            self.teardown(bench.as_mut()).await;
            return;
        }

        // 2026-09-26: Probe, then load, then configure; a setup error becomes
        // a terminal "setup" frame. The probe runs before `load()`, which for
        // BFCL provisions a venv and dataset, so its warning is in the log
        // before that wait.
        self.probe_coherence().await;
        let setup = async {
            bench.load(self.handle.clone()).await?;
            bench.configure(&self.values)
        }
        .await;
        if let Err(e) = setup {
            self.emit_failed("setup", e, started.elapsed(), hardware)
                .await;
            self.teardown(bench.as_mut()).await;
            return;
        }

        // 2026-09-26: Drive. The terminal frame is held back until the
        // after-capture is attached to it (`close_hardware`).
        let mut terminal: Option<BenchmarkResult> = None;
        {
            let stream = drive(bench.as_mut());
            futures::pin_mut!(stream);
            while let Some(item) = stream.next().await {
                match item {
                    Ok(frame) => {
                        if frame.status.is_terminal() {
                            terminal = Some(frame);
                            break;
                        }
                        if self.frames.send(frame).is_err() {
                            break;
                        }
                    }
                    // 2026-09-26: `{:#}` prints the whole anyhow context chain.
                    Err(e) => {
                        terminal = Some(BenchmarkResult::failed(
                            "run",
                            format!("{e:#}"),
                            started.elapsed(),
                        ));
                        break;
                    }
                }
                if self.cancel.load(Ordering::Relaxed) {
                    terminal = Some(BenchmarkResult::failed(
                        "cancelled",
                        "cancelled by user",
                        started.elapsed(),
                    ));
                    break;
                }
            }
        }
        if let Some(frame) = terminal {
            let frame = self.close_hardware(frame, hardware).await;
            let _ = self.frames.send(frame);
        }

        self.teardown(bench.as_mut()).await;
    }

    async fn emit_failed(
        &self,
        phase: &str,
        error: anyhow::Error,
        elapsed: Duration,
        hardware: Option<HardwareStateReport>,
    ) {
        let frame = BenchmarkResult::failed(phase, format!("{error:#}"), elapsed);
        let frame = self.close_hardware(frame, hardware).await;
        let _ = self.frames.send(frame);
    }

    async fn teardown(&self, bench: &mut dyn DynBenchmark) {
        if let Err(e) = bench.cleanup().await {
            let _ = self
                .events
                .send(PluginEvent::Log(LogLine::warn(format!("cleanup: {e:#}"))));
        }
        let _ = self.events.send(PluginEvent::Glow(false));
        self.finished.store(true, Ordering::Relaxed);
    }
}

/// 2026-09-26: True when a frame ends the run, by the same rule [`drive`]
/// stops on.
pub fn is_final(frame: &BenchmarkResult) -> bool {
    frame.status.is_terminal()
}

/// 2026-09-26: `Completed` when `ok`, else `Failed`.
pub fn terminal_status(ok: bool) -> RunStatus {
    if ok {
        RunStatus::Completed
    } else {
        RunStatus::Failed
    }
}

#[cfg(test)]
#[path = "executor_tests.rs"]
mod tests;
