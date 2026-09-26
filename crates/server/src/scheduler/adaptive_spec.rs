// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Adaptive DFlash speculation, on when `METRALE_DFLASH_ADAPTIVE=1`.
//!
//! Policy: each sequence keeps the accept counts of its last [`WINDOW`] K=γ
//! verify steps. When the window is full and its mean is below
//! `METRALE_DFLASH_ADAPTIVE_MIN` (default 2.0), speculation is suspended for
//! that sequence. After `METRALE_DFLASH_ADAPTIVE_REPROBE` (default 256)
//! serially decoded tokens, [`spec_allowed`] lifts the suspension with an
//! empty window, so it can only re-trigger after [`WINDOW`] more verify steps.
//! The state is reset when a sequence is requeued or restored from spill.
//!
//! Owner: scheduler.
//! Invariants:
//! - `window` never holds more than [`WINDOW`] entries.

use crate::scheduler::ActiveSeq;

/// 2026-09-25: Rolling accept window + suspend state, embedded in [`ActiveSeq`].
#[derive(Default)]
pub(crate) struct AdaptState {
    window: Vec<u32>,
    suspended: bool,
    serial_tokens: u32,
}

const WINDOW: usize = 12;

/// 2026-09-25: Record one K=γ verify step's accept count; may trip suspension.
pub(crate) fn record_verify(
    a: &mut ActiveSeq,
    num_accepted: usize,
    sched: &crate::scheduler::sched_ctx::SchedCtx,
) {
    if !sched.levers.dflash_adaptive {
        return;
    }
    let st = &mut a.spec_adapt;
    st.window.push(num_accepted as u32);
    if st.window.len() > WINDOW {
        st.window.remove(0);
    }
    if st.window.len() == WINDOW {
        let mean = st.window.iter().sum::<u32>() as f32 / WINDOW as f32;
        if mean < sched.levers.dflash_adaptive_min {
            st.suspended = true;
            st.serial_tokens = 0;
            st.window.clear();
            tracing::info!(
                "adaptive spec: SUSPENDED (mean accepted {mean:.2} < {} over {WINDOW} steps) — \
                 serial decode until re-probe",
                sched.levers.dflash_adaptive_min,
            );
        }
    }
}

/// 2026-09-25: May this sequence propose/speculate right now? Un-suspends (re-probe)
/// once enough serial tokens have passed.
pub(crate) fn spec_allowed(
    a: &mut ActiveSeq,
    sched: &crate::scheduler::sched_ctx::SchedCtx,
) -> bool {
    if !sched.levers.dflash_adaptive {
        return true;
    }
    let st = &mut a.spec_adapt;
    if !st.suspended {
        return true;
    }
    if st.serial_tokens >= sched.levers.dflash_adaptive_reprobe {
        st.suspended = false;
        st.serial_tokens = 0;
        st.window.clear();
        tracing::info!(
            "adaptive spec: RE-PROBING after {} serial tokens",
            sched.levers.dflash_adaptive_reprobe
        );
        return true;
    }
    false
}

/// 2026-09-25: Is this sequence currently in the adaptive-suspended (serial) regime?
/// Read-only peek — unlike `spec_allowed`, never mutates re-probe state.
pub(crate) fn is_suspended(a: &ActiveSeq, sched: &crate::scheduler::sched_ctx::SchedCtx) -> bool {
    sched.levers.dflash_adaptive && a.spec_adapt.suspended
}

/// 2026-09-25: Count a serially-decoded token toward the re-probe interval.
pub(crate) fn tick_serial(a: &mut ActiveSeq, sched: &crate::scheduler::sched_ctx::SchedCtx) {
    if sched.levers.dflash_adaptive && a.spec_adapt.suspended {
        a.spec_adapt.serial_tokens = a.spec_adapt.serial_tokens.saturating_add(1);
    }
}
