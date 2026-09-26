// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The endpoint check that runs between pressing start and the benchmark starting.
//!
//! `coherence::probe_for` reads the endpoint's model list and asks the
//! `coherence::CHECKS` questions; a concern opens a modal. The check cannot
//! veto a run: the user can proceed past any concern.
//!
//! Owner: server tui.
//! Invariants:
//! - `poll` never returns `Some` twice: it drops the receiver on the first answer.

use std::sync::mpsc::{Receiver, TryRecvError, channel};

use metrale_bench::coherence::{self, Report};

/// 2026-09-26: Where the pre-flight has got to.
#[derive(Debug, PartialEq, Eq)]
pub enum Phase {
    /// 2026-09-26: Asking. The modal shows a spinner.
    Checking,
    /// 2026-09-26: Something is worth saying before starting; the user picks.
    Concern(String),
}

pub struct Preflight {
    pub phase: Phase,
    rx: Option<Receiver<Report>>,
}

impl Preflight {
    /// 2026-09-26: Begin checking `target` on `runtime`.
    ///
    /// The probe runs as a task and answers over a channel that the UI drains
    /// on its tick, so the render thread never blocks on it.
    pub fn begin(
        runtime: &tokio::runtime::Handle,
        target: metrale_bench::TargetEndpoint,
        expectation: Option<metrale_bench::benchmark::ModelExpectation>,
        timeout: std::time::Duration,
    ) -> Self {
        let (tx, rx) = channel();
        runtime.spawn(async move {
            // 2026-09-26: A dropped receiver means the user moved on; not an error.
            let _ = tx.send(coherence::probe_for(&target, expectation, timeout).await);
        });
        Self {
            phase: Phase::Checking,
            rx: Some(rx),
        }
    }

    /// 2026-09-26: Drain the check. Returns `Some(true)` when the run should start now,
    /// `Some(false)` when the user must be asked first, `None` while waiting.
    pub fn poll(&mut self, target: &metrale_bench::TargetEndpoint) -> Option<bool> {
        let rx = self.rx.as_ref()?;
        match rx.try_recv() {
            Ok(report) => {
                self.rx = None;
                match report.concern(target) {
                    None => Some(true),
                    Some(concern) => {
                        self.phase = Phase::Concern(concern);
                        Some(false)
                    }
                }
            }
            Err(TryRecvError::Empty) => None,
            // 2026-09-26: The task ended without answering; treat it as nothing to report and start.
            Err(TryRecvError::Disconnected) => {
                self.rx = None;
                Some(true)
            }
        }
    }

    /// 2026-09-26: A pre-flight that stays waiting (its sender is leaked), for tests.
    #[cfg(test)]
    pub fn pending() -> Self {
        let (_tx, rx) = channel();
        std::mem::forget(_tx);
        Self {
            phase: Phase::Checking,
            rx: Some(rx),
        }
    }

    /// 2026-09-26: A pre-flight that has already found something to say, for tests.
    #[cfg(test)]
    pub fn with_concern(text: String) -> Self {
        Self {
            phase: Phase::Concern(text),
            rx: None,
        }
    }

    pub fn is_checking(&self) -> bool {
        self.phase == Phase::Checking
    }
}

#[cfg(test)]
#[path = "bench_preflight_tests.rs"]
mod tests;
