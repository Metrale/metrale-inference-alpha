// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Process-wide latch for a destroyed CUDA context, the probe-based verdict that sets it, and the exit status it maps to.
//!
//! [`classify`] decides from a probe result, not from the failing call's error
//! code or text: the gpu-runtime fault probe (`cuda_backend/fault_probe.rs`)
//! passes the result of `cuStreamSynchronize` on the default stream, issued
//! after the failure. The latch is read by the request middleware
//! (`gpu_fault_middleware`), `/health` and `/health/live`, the scheduler, and
//! `main`'s exit path.
//!
//! Owner: core.
//! Invariants:
//! - A [`FaultLatch`] is set at most once, and keeps the first reason.
//! - `is_faulted()` is true exactly when `fault()` is `Some`: both read the
//!   same `OnceLock<String>`, so no reader sees one without the other.

use std::sync::OnceLock;

/// 2026-09-25: The verdict for one failed GPU operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Fatality {
    /// 2026-09-25: The probe failed too: the context is unusable. The payload
    /// is the operator-facing reason.
    ContextLost(String),
    /// 2026-09-25: The probe succeeded: the context still works.
    Isolated,
}

/// 2026-09-25: Decide whether a failed GPU operation destroyed the context.
///
/// `probe` is the result of a call issued after the failure that succeeds on a
/// healthy context. Only `probe` decides the verdict; `op` and `err` appear
/// only in the reason text.
pub fn classify(op: &str, err: &str, probe: Result<(), String>) -> Fatality {
    match probe {
        Ok(()) => Fatality::Isolated,
        Err(probe_err) => Fatality::ContextLost(format!(
            "{op} failed ({err}), and a no-op synchronize issued afterwards \
             also failed ({probe_err}) — the CUDA context is destroyed. Errors \
             of this class are sticky: every later driver call in this process \
             returns the same status, so no request can be served."
        )),
    }
}

/// 2026-09-25: A one-shot, first-writer-wins fault flag.
///
/// Constructible on its own so tests need not latch the process-wide instance,
/// which cannot be reset.
#[derive(Debug, Default)]
pub struct FaultLatch {
    reason: OnceLock<String>,
}

impl FaultLatch {
    pub const fn new() -> Self {
        Self {
            reason: OnceLock::new(),
        }
    }

    /// 2026-09-25: Record a fatal fault. Returns `true` only for the call that
    /// set the latch; later calls leave the first reason in place.
    pub fn latch(&self, reason: impl Into<String>) -> bool {
        self.reason.set(reason.into()).is_ok()
    }

    /// 2026-09-25: The reason for the fault, or `None` if healthy.
    pub fn fault(&self) -> Option<&str> {
        self.reason.get().map(String::as_str)
    }

    pub fn is_faulted(&self) -> bool {
        self.reason.get().is_some()
    }
}

static GLOBAL: FaultLatch = FaultLatch::new();

/// 2026-09-25: The process-wide latch that the fault probe sets and the
/// server reads.
pub fn global() -> &'static FaultLatch {
    &GLOBAL
}

/// 2026-09-25: Exit status of a process whose CUDA context was lost: nonzero,
/// and distinct from the status 1 of an ordinary failure ([`exit_code`]).
pub const EXIT_GPU_FAULT: i32 = 70;

/// 2026-09-25: The process exit status for a run that is ending: a latched
/// fault gives [`EXIT_GPU_FAULT`] whatever the run's own result, otherwise 0
/// for success and 1 for failure. `main` exits through this, so a run that
/// ended after a fault does not exit 0.
pub fn exit_code(run_succeeded: bool, fault: Option<&str>) -> i32 {
    match (run_succeeded, fault) {
        (_, Some(_)) => EXIT_GPU_FAULT,
        (true, None) => 0,
        (false, None) => 1,
    }
}

#[cfg(test)]
#[path = "fault_tests.rs"]
mod tests;
