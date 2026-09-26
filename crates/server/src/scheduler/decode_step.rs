// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: One plain (non-speculative) decode step for every active sequence.
//!
//! Owner: scheduler.
//! Invariants:
//! - With more than one active sequence, `active` is sorted ascending by SSM
//!   slot (KV slot when the sequence has none) before the decode is launched.

use super::*;

/// 2026-09-25: Decode one token for every sequence in `active`, then sample
/// and emit through `process_decode_logits`. KV exhaustion can move
/// sequences out of `active` into `swapped` or `preempted`.
pub fn step_decode_only(
    active: &mut Vec<ActiveSeq>,
    think_end_token: Option<u32>,
    think_start_token: Option<u32>,
    code_fence_token: Option<u32>,
    tool_call_start_token: Option<u32>,
    tool_call_end_token: Option<u32>,
    adaptive_sampling: bool,
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    spill: Option<&dyn crate::scheduler::io::SpillIo>,
    swapped: &mut Vec<SwappedSeq>,
    preempted: &mut Vec<PreemptedSeq>,
) {
    let t0 = sched.io.clock.now();
    let n = active.len();
    // 2026-09-25: the batched-recurrent SSM path runs only when row i's state
    // sits at slot base + i * stride (`ssm_batched_recurrent.rs`); otherwise
    // it falls back to the per-sequence loop. Sorting by slot gives it that
    // order whenever the slots are contiguous. Whole `ActiveSeq`s move, so
    // row i of the result still belongs to `active[i]`.
    if n > 1 {
        active.sort_by_key(|a| a.seq.ssm_slot_idx().unwrap_or(a.seq.slot_idx));
    }

    // 2026-09-25: per-step batch-state line, at debug because it is logged on
    // every decode step.
    if n > 1 && tracing::enabled!(tracing::Level::DEBUG) {
        let diag: Vec<String> = active
            .iter()
            .enumerate()
            .map(|(i, a)| {
                let bt0 = a.seq.block_table.first().copied().unwrap_or(u32::MAX);
                let btn = a.seq.block_table.len();
                format!(
                    "[{i}: slot={} seq_len={} bt={}/{} last={} out_n={}]",
                    a.seq.slot_idx,
                    a.seq.seq_len,
                    bt0,
                    btn,
                    a.last_token,
                    a.output_tokens.len(),
                )
            })
            .collect();
        tracing::debug!("CONC_DIAG n={n}: {}", diag.join(" "));
    }

    // 2026-09-25: this step sends no EP broadcasts. The model's
    // `decode_batch_dispatch` (model-engine `trait_impl/decode_a2.rs`) sends
    // them in the comm-stream order both ranks must share.

    // 2026-09-25: on "KV cache exhausted" with more than one row, the launch
    // loop (`decode_launch::launch_with_preemption`) removes one victim per
    // retry and spills it (when `spill` is set) or requeues it for
    // re-prefill, so the rest of the batch continues. `None` means every row
    // left in `active` has been sent the error and `active` is empty.
    //
    // The readback lands in the run's staging buffer. `process_decode_logits`
    // puts it back in `sched.scratch.host_bytes`; the early returns here drop
    // it, which loses only its capacity.
    let mut staging = sched.scratch.host_bytes.borrow_mut().split_off(0);
    let Some(step) = super::preempt::decode_batch_with_preemption(
        sched,
        active,
        spill,
        swapped,
        preempted,
        &mut staging,
    ) else {
        return;
    };
    let n = active.len();
    if n == 0 {
        return;
    }

    process_decode_logits(
        active,
        step,
        &mut staging,
        t0,
        think_end_token,
        think_start_token,
        code_fence_token,
        tool_call_start_token,
        tool_call_end_token,
        adaptive_sampling,
        sched,
    );
}
