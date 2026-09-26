// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: [`HardwareStateReport`]: the two captures, the delta and both
//! verdicts, as one value that travels with the run into its gate record
//! (`GateRecord::hardware_state`, `.benchmarks/<id>/<date>-<sha>.json`).
//!
//! Owner: bench hardware.
//! Invariants: none beyond the types. [`HardwareStateReport::close`] sets `after`,
//! `delta` and `postcheck` together.

use serde::{Deserialize, Serialize};

use super::policy::{self, Postcheck, Precheck, Sensitivity};
use super::state::{HardwareState, HardwareStateDelta};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HardwareStateReport {
    /// 2026-09-26: Which class of number this was, so a reader knows why the
    /// verdict is what it is without looking the benchmark up.
    pub sensitivity: Sensitivity,
    /// 2026-09-26: Captured by the executor before the coherence probe and
    /// `load()`, so before any request.
    pub before: HardwareState,
    /// 2026-09-26: Captured when the run's terminal frame is ready, including a
    /// failed setup or a cancelled run. `None` when the precheck refused the run,
    /// when the stream ended without a terminal frame, or when the capture task
    /// failed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub after: Option<HardwareState>,
    /// 2026-09-26: `after` minus `before`, set with `after` by [`HardwareStateReport::close`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delta: Option<HardwareStateDelta>,
    pub precheck: Precheck,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub postcheck: Option<Postcheck>,
    /// 2026-09-26: `before.machine.perf_class()`, e.g. `"gb10@dgx1"`, copied out so
    /// a reader need not derive it.
    pub perf_class: String,
}

impl HardwareStateReport {
    /// 2026-09-26: Open a report with the pre-run capture and its verdict. The
    /// switches come from the environment, the ceilings from the caller.
    pub fn opened(
        sensitivity: Sensitivity,
        before: HardwareState,
        ceilings: Option<policy::TempCeilings>,
    ) -> Self {
        let precheck = policy::precheck(
            sensitivity,
            &before,
            policy::PolicyOptions {
                ceilings,
                ..policy::PolicyOptions::from_env()
            },
        );
        Self {
            sensitivity,
            perf_class: before.machine.perf_class(),
            before,
            after: None,
            delta: None,
            precheck,
            postcheck: None,
        }
    }

    /// 2026-09-26: Close it with the post-run capture, the delta and the validity
    /// verdict.
    pub fn close(&mut self, after: HardwareState) {
        let delta = HardwareStateDelta::between(&self.before, &after);
        self.postcheck = Some(policy::postcheck(
            self.sensitivity,
            &delta,
            policy::PolicyOptions::from_env(),
        ));
        self.delta = Some(delta);
        self.after = Some(after);
    }

    /// 2026-09-26: True when the precheck refused the run.
    pub fn refuses(&self) -> bool {
        self.precheck.decision == policy::Decision::Refuse
    }

    /// 2026-09-26: True when the postcheck judged the run invalid.
    ///
    /// A report that was never closed is not invalid: it is unmeasured, and
    /// saying otherwise would blame the box for a harness failure.
    pub fn invalidated(&self) -> bool {
        self.postcheck
            .as_ref()
            .is_some_and(|p| p.validity == policy::Validity::Invalid)
    }

    /// 2026-09-26: Every concern from both phases, precheck first, for the run log.
    pub fn concerns(&self) -> Vec<&str> {
        self.precheck
            .concerns
            .iter()
            .chain(self.postcheck.iter().flat_map(|p| p.concerns.iter()))
            .map(String::as_str)
            .collect()
    }
}

#[cfg(test)]
#[path = "report_tests.rs"]
mod tests;
