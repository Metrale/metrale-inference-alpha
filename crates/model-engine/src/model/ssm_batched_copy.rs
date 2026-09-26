// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Batched execution of the SSM verify-state copy sets.
//!
//! The speculative-verify state moves (pre-verify checkpoint, full-reject rollback,
//! partial-accept commit) each copy one h blob and one conv blob per SSM layer between
//! two pool regions: `2 × num_ssm_layers` `copy_d2d_async` launches per sequence as a
//! loop.
//!
//! h and conv live in different pool families, but when [`super::ssm_pool::SsmStatePool`]
//! got one contiguous block for a family (`alloc_layer_pools`), that family's per-layer
//! regions sit at `base + layer * stride`. The h copies then collapse to one pitched 2-D
//! copy, and likewise the conv copies: two launches per sequence.
//!
//! [`copy_plan_as_strided_run`] collapses only a plan whose rows it reproduces exactly
//! (one non-zero width, one forward pitch per side at least that width), and
//! `copy_d2d_2d_async` writes row `r` as `width_bytes` from `src + r*src_pitch` to
//! `dst + r*dst_pitch` (the trait default does it row by row). Anything else (per-layer
//! pool allocations, a ragged plan, a single row) runs the per-copy loop.
//!
//! Kill switch: `METRALE_NO_BATCHED_SSM_ROLLBACK`, set to any value, forces the loop
//! everywhere.
//!
//! Owner: model-engine.
//! Invariants:
//! - A plan is issued as a 2-D copy only when that copy's rows are exactly the plan's rows.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

/// 2026-09-25: One device-to-device state blob move: `bytes` from `src` to `dst`.
///
/// A `Vec<StateCopy>` is the plan both executors below consume, so the batched
/// and looped forms cannot disagree on which bytes land where.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct StateCopy {
    pub src: DevicePtr,
    pub dst: DevicePtr,
    pub bytes: usize,
}

/// 2026-09-25: A copy plan that collapses to one pitched 2-D transfer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct StridedRun {
    pub src: DevicePtr,
    pub src_pitch: usize,
    pub dst: DevicePtr,
    pub dst_pitch: usize,
    pub width_bytes: usize,
    pub height: usize,
}

/// 2026-09-25: Can `plan` be issued as one pitched 2-D copy instead of
/// `plan.len()` separate ones? Pure, so the equivalence is testable without a
/// GPU.
///
/// Requires, and checks, every condition for a 2-D copy to reproduce the plan
/// row for row:
/// - at least two rows (one row is already one launch, and a pitch cannot be
///   derived from a single row);
/// - every row the same non-zero `width_bytes`;
/// - one forward `src_pitch` and `dst_pitch` for every row (row `r` at
///   `base + r*pitch`);
/// - `pitch >= width_bytes` on both sides, so consecutive rows never overlap
///   and row order does not matter.
///
/// Returns `None` otherwise; the caller then runs the plan as-is.
pub(crate) fn copy_plan_as_strided_run(plan: &[StateCopy]) -> Option<StridedRun> {
    if plan.len() < 2 {
        return None;
    }
    let width_bytes = plan[0].bytes;
    if width_bytes == 0 || plan.iter().any(|c| c.bytes != width_bytes) {
        return None;
    }
    // 2026-09-25: Pitches are derived from the first pair and then checked
    // against every row; `checked_sub` on u64 rejects a descending family (a
    // negative pitch has no 2-D copy form).
    let src_pitch = plan[1].src.0.checked_sub(plan[0].src.0)?;
    let dst_pitch = plan[1].dst.0.checked_sub(plan[0].dst.0)?;
    if src_pitch < width_bytes as u64 || dst_pitch < width_bytes as u64 {
        return None;
    }
    for (r, c) in plan.iter().enumerate() {
        let r = r as u64;
        if c.src.0 != plan[0].src.0 + r * src_pitch || c.dst.0 != plan[0].dst.0 + r * dst_pitch {
            return None;
        }
    }
    Some(StridedRun {
        src: plan[0].src,
        src_pitch: src_pitch as usize,
        dst: plan[0].dst,
        dst_pitch: dst_pitch as usize,
        width_bytes,
        height: plan.len(),
    })
}

/// 2026-09-25: False when `METRALE_NO_BATCHED_SSM_ROLLBACK` is set, to any value
/// including `0`; the per-copy loop then runs everywhere. Read once per
/// process, since it is on the verify path.
pub(crate) fn batched_ssm_copy_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("METRALE_NO_BATCHED_SSM_ROLLBACK").is_none())
}

/// 2026-09-25: Issue `plan` on `stream`, batched when `batched` is set and the
/// plan collapses.
///
/// `batched` is a parameter so both settings are testable without the process
/// environment or the latching `OnceLock` above. Production callers pass
/// [`batched_ssm_copy_enabled`].
pub(crate) fn run_state_copies_with(
    gpu: &dyn GpuBackend,
    plan: &[StateCopy],
    batched: bool,
    stream: u64,
) -> Result<()> {
    if batched && let Some(run) = copy_plan_as_strided_run(plan) {
        return gpu.copy_d2d_2d_async(
            run.src,
            run.src_pitch,
            run.dst,
            run.dst_pitch,
            run.width_bytes,
            run.height,
            stream,
        );
    }
    for c in plan {
        gpu.copy_d2d_async(c.src, c.dst, c.bytes, stream)?;
    }
    Ok(())
}

/// 2026-09-25: [`run_state_copies_with`] at the process-wide kill-switch setting.
pub(crate) fn run_state_copies(
    gpu: &dyn GpuBackend,
    plan: &[StateCopy],
    stream: u64,
) -> Result<()> {
    run_state_copies_with(gpu, plan, batched_ssm_copy_enabled(), stream)
}

/// 2026-09-25: Issue an h plan and a conv plan back to back, on one stream.
///
/// The families run separately because a 2-D copy has one width and the h and
/// conv blob widths differ in general. Issuing every h copy before every conv
/// copy is sound: [`super::ssm_pool::SsmStatePool`] allocates the h and conv
/// pools separately (`alloc_layer_pools` per family), so no h copy touches a
/// byte a conv copy touches.
pub(crate) fn run_ssm_state_copies(
    gpu: &dyn GpuBackend,
    h_plan: &[StateCopy],
    conv_plan: &[StateCopy],
    stream: u64,
) -> Result<()> {
    let batched = batched_ssm_copy_enabled();
    run_state_copies_with(gpu, h_plan, batched, stream)?;
    run_state_copies_with(gpu, conv_plan, batched, stream)
}

#[cfg(test)]
#[path = "ssm_batched_copy_tests.rs"]
mod tests;
