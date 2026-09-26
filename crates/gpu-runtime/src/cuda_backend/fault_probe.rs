// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: After a failed GPU operation, probe whether the CUDA context
//! still works and latch the process-wide fault ([`metrale_core::fault`]) if
//! it does not.
//!
//! The probe is `cuStreamSynchronize` on stream 0; the verdict is
//! [`classify`](metrale_core::fault::classify), which decides from the probe's
//! result alone, not from the failed call's status.
//!
//! Owner: gpu-runtime (CUDA backend).
//! Invariants:
//! - `note_failure` never changes the caller's control flow; it returns `()`.
//! - The probe runs only on a failure path, and not at all once the latch
//!   is set.

use metrale_core::fault::{self, Fatality};

use super::cuStreamSynchronize;

/// 2026-09-25: Stream 0, the NULL stream.
const DEFAULT_STREAM: u64 = 0;

/// 2026-09-25: Synchronize stream 0. `Err` carries the driver's error text.
fn probe() -> Result<(), String> {
    let status = unsafe { cuStreamSynchronize(DEFAULT_STREAM) };
    if status == 0 {
        Ok(())
    } else {
        Err(crate::registry::cuda_error_text(status))
    }
}

/// 2026-09-25: Record a failed GPU operation `op` with error text `err`:
/// unless the fault latch is already set, probe the context, and if the probe
/// fails, latch the reason and log it once at ERROR.
///
/// Callers are the kernel-launch failure path (`gpu_impl.rs` `launch`) and
/// the memset failure paths (`gpu_impl_graph.rs` `memset_cu`,
/// `memset_async_cu`). The caller still returns its own error.
pub(super) fn note_failure(op: &str, err: &str) {
    // 2026-09-25: Already latched: the latch keeps the first reason
    // (`FaultLatch::latch`), so a probe here could change nothing.
    if fault::global().is_faulted() {
        return;
    }
    if let Fatality::ContextLost(reason) = fault::classify(op, err, probe())
        && fault::global().latch(reason.clone())
    {
        // 2026-09-25: Logged only by the call that set the latch.
        tracing::error!(target: "metrale::fault", "{reason}");
    }
}
