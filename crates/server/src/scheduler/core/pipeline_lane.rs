// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The pipelined decode lane: launch ahead, settle, validate, drain. The
//! state and the depth policy live in `pipeline.rs`.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::pipeline::{CommitTokens, Inflight, Pipeline, Settled, ahead_allowed, row_mask};
use super::*;
use crate::scheduler::io::{DecodeRows, Effect, EffectOutcome, StepOutcome, StepPlan};
use metrale_scheduler::{FeedSource, RowMask};

/// 2026-09-25: The plain decode step of a tick when the router can run ahead.
pub(super) fn step_decode_pipelined(
    active: &mut Vec<ActiveSeq>,
    pipeline: &mut Pipeline,
    toks: CommitTokens,
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    spill: Option<&dyn crate::scheduler::io::SpillIo>,
    swapped: &mut Vec<SwappedSeq>,
    preempted: &mut Vec<PreemptedSeq>,
) {
    // 2026-09-25: Slot order, as the synchronous step establishes it (`step_decode_only`).
    if active.len() > 1 {
        active.sort_by_key(|a| a.seq.ssm_slot_idx().unwrap_or(a.seq.slot_idx));
    }
    if let Some(inf) = pipeline.inflight.as_ref() {
        let can_go_ahead = !inf.host_logits
            && !inf.discard_all
            && rows_match(inf, active)
            && ahead_allowed(active, sched, toks.adaptive_sampling)
            && reserve_all(active, sched);
        if can_go_ahead && launch_ahead(active, pipeline, &toks, sched) {
            // 2026-09-25: t+1 is in flight; t is committed against the state it was
            // launched from, and t+1 is validated against the result.
            settle(active, pipeline, &toks, sched);
            return;
        }
        match settle(active, pipeline, &toks, sched) {
            Settled::Done | Settled::Failed | Settled::Idle => return,
            Settled::Redo => {}
        }
    }
    // 2026-09-25: No step in flight, or the last one was discarded: launch now, and
    // keep the step in flight only when the next tick may launch ahead of it.
    let t0 = sched.io.clock.now();
    let ahead = ahead_allowed(active, sched, toks.adaptive_sampling);
    let feed = if ahead {
        super::super::preempt::FeedReadback::Masked {
            think_end_token: toks.think_end_token,
        }
    } else {
        super::super::preempt::FeedReadback::Plain
    };
    let mut staging = sched.scratch.host_bytes.borrow_mut().split_off(0);
    let launched = super::super::preempt::launch_with_preemption(
        sched,
        active,
        spill,
        swapped,
        preempted,
        &mut staging,
        feed,
    );
    *sched.scratch.host_bytes.borrow_mut() = staging;
    let Some(ticket) = launched else {
        return;
    };
    let n = active.len();
    if n == 0 {
        return;
    }
    // 2026-09-25: Preemption may have changed the batch: re-ask before holding the step.
    let hold = ahead && ahead_allowed(active, sched, toks.adaptive_sampling);
    pipeline.inflight = Some(Inflight {
        ticket,
        t0,
        // 2026-09-25: The plain launch has already pushed: the position written is
        // `seq_len - 1`.
        rows: active
            .iter()
            .map(|a| (a.seq.slot_idx, a.seq.seq_len.saturating_sub(1)))
            .collect(),
        masks: active
            .iter()
            .map(|a| row_mask(a, toks.think_end_token))
            .collect(),
        inputs: None,
        discard: vec![false; n],
        discard_all: false,
        host_logits: false,
    });
    if !hold {
        settle(active, pipeline, &toks, sched);
    }
}

fn rows_match(inf: &Inflight, active: &[ActiveSeq]) -> bool {
    inf.rows.len() == active.len()
        && inf
            .rows
            .iter()
            .zip(active)
            .all(|((slot, _), a)| *slot == a.seq.slot_idx)
}

/// 2026-09-25: Reserve every row's next KV block before a launch that runs ahead of
/// the host. `false` (a dry pool or an error) means no launch ahead this
/// tick; blocks already reserved for earlier rows stay allocated.
fn reserve_all(active: &mut [ActiveSeq], sched: &crate::scheduler::sched_ctx::SchedCtx) -> bool {
    for a in active.iter_mut() {
        match sched.io.dev.apply(Effect::ReserveKv { seq: &mut a.seq }) {
            Ok(EffectOutcome::Reserved { .. }) => {}
            _ => return false,
        }
    }
    true
}

/// 2026-09-25: Launch `t+1` fed from `t`'s cells. `false` when the launch failed; the
/// caller then only settles `t` this tick.
fn launch_ahead(
    active: &mut [ActiveSeq],
    pipeline: &mut Pipeline,
    toks: &CommitTokens,
    sched: &crate::scheduler::sched_ctx::SchedCtx,
) -> bool {
    let t0 = sched.io.clock.now();
    let n = active.len();
    let sources: Vec<FeedSource> = (0..n as u32)
        .map(|from_row| FeedSource::Feed { from_row })
        .collect();
    let masks: Vec<RowMask> = active
        .iter()
        .map(|a| row_mask(a, toks.think_end_token))
        .collect();
    let rows: Vec<(usize, usize)> = active
        .iter()
        .map(|a| (a.seq.slot_idx, a.seq.seq_len))
        .collect();
    let mut refs: Vec<&mut SequenceState> = active.iter_mut().map(|a| &mut a.seq).collect();
    let launched = sched.io.dev.launch(
        StepPlan::DecodeFed {
            sources: sources.clone(),
            masks: masks.clone(),
            ctx: super::super::preempt::decode_ctx_commit(n, &sched.levers),
        },
        &mut refs,
    );
    match launched {
        Ok(ticket) => {
            pipeline.next = Some(Inflight {
                ticket,
                t0,
                rows,
                masks,
                inputs: Some(sources),
                discard: vec![false; n],
                discard_all: false,
                host_logits: false,
            });
            true
        }
        Err(e) => {
            tracing::warn!("ahead decode launch declined, settling first: {e}");
            false
        }
    }
}

/// 2026-09-25: Await and commit the step in flight, then validate the one launched
/// ahead of it (if any) against the state the commit produced.
pub(super) fn settle(
    active: &mut Vec<ActiveSeq>,
    pipeline: &mut Pipeline,
    toks: &CommitTokens,
    sched: &crate::scheduler::sched_ctx::SchedCtx,
) -> Settled {
    let Some(inf) = pipeline.inflight.take() else {
        return Settled::Idle;
    };
    assert!(
        rows_match(&inf, active),
        "pipelined step settled against a different batch"
    );
    let n = active.len();
    let result = match sched.io.dev.await_result(inf.ticket) {
        Ok(r) => r,
        Err(e) => {
            // 2026-09-25: The forward or its readback failed: every row is told, as the
            // synchronous step does. A step launched ahead is awaited so the
            // router's slot is released, and dropped with the rows.
            if let Some(next) = pipeline.next.take() {
                let _ = sched.io.dev.await_result(next.ticket);
            }
            let e = e.into_inner();
            tracing::error!("decode_batch error: {e:#}");
            for mut a in active.drain(..) {
                send_error(&sched.io, &mut a, &format!("{e:#}"));
            }
            if let Some(reason) = metrale_core::fault::global().fault() {
                crate::tui::shutdown::request(reason);
            }
            return Settled::Failed;
        }
    };
    let StepOutcome::Decode { logits, rows } = result.outcome;
    let device_tokens: Vec<u32> = match &rows {
        DecodeRows::Tokens(t) => t.clone(),
        DecodeRows::HostLogits { .. } => Vec::new(),
    };
    // 2026-09-25: Over-runs hand their reserved block back before anything else runs.
    if inf.inputs.is_some() {
        for (i, a) in active.iter_mut().enumerate() {
            if (inf.discard_all || inf.discard[i])
                && let Err(e) = sched.io.dev.apply(Effect::Rollback { seq: &mut a.seq })
            {
                tracing::warn!("over-run rollback: {e}");
            }
        }
    }
    if inf.discard_all {
        debug_assert!(
            pipeline.next.is_none(),
            "no step is launched ahead of a redo"
        );
        return Settled::Redo;
    }
    let mut staging = sched.scratch.host_bytes.borrow_mut().split_off(0);
    let step = if inf.host_logits {
        match sched.io.dev.apply(Effect::ReadLogits {
            logits,
            rows: n,
            into: &mut staging,
        }) {
            Ok(EffectOutcome::HostLogits { elem_bytes }) => StepOutcome::Decode {
                logits,
                rows: DecodeRows::HostLogits { elem_bytes },
            },
            Ok(_) => unreachable!("ReadLogits answers HostLogits"),
            Err(e) => {
                tracing::error!("copy_logits_to_host error: {e}");
                for mut a in active.drain(..) {
                    send_error(&sched.io, &mut a, &format!("{e}"));
                }
                return Settled::Failed;
            }
        }
    } else {
        StepOutcome::Decode { logits, rows }
    };
    super::super::decode_logits_step::process_decode_logits_skipping(
        active,
        step,
        &mut staging,
        inf.t0,
        toks.think_end_token,
        toks.think_start_token,
        toks.code_fence_token,
        toks.tool_call_start_token,
        toks.tool_call_end_token,
        toks.adaptive_sampling,
        sched,
        &inf.discard,
    );
    if let Some(next) = pipeline.next.take() {
        pipeline.inflight = Some(validate_next(
            next,
            active,
            &device_tokens,
            sched,
            pipeline.faults,
            toks,
        ));
    }
    Settled::Done
}

/// 2026-09-25: The step launched ahead was fed from the step just committed. Decide
/// which of its rows are over-runs, apply the launch bookkeeping to the
/// rest, and ask whether its commit can still take the device tokens.
fn validate_next(
    mut next: Inflight,
    active: &mut [ActiveSeq],
    prev_tokens: &[u32],
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    faults: PipelineFaults,
    toks: &CommitTokens,
) -> Inflight {
    let inputs = next.inputs.as_ref().expect("a step launched ahead is fed");
    assert!(
        rows_match(&next, active),
        "pipelined step validated against a different batch"
    );
    let mut expected = Vec::with_capacity(active.len());
    for (i, a) in active.iter().enumerate() {
        let input = match inputs[i] {
            FeedSource::Feed { from_row } => prev_tokens[from_row as usize],
            FeedSource::Host(id) => id,
        };
        let over_run = a.finished || a.last_token != input || a.seq.seq_len != next.rows[i].1;
        next.discard[i] = over_run && !faults.keep_overrun;
        expected.push(input);
    }
    // 2026-09-25: The bookkeeping `decode_batch` performs at launch, for the rows whose
    // launch was valid: the commit then runs over the state the synchronous
    // loop would have launched them from.
    for (i, a) in active.iter_mut().enumerate() {
        if !next.discard[i] {
            a.seq.tokens.push(expected[i]);
            a.seq.seq_len += 1;
        }
    }
    let survivors = active
        .iter()
        .enumerate()
        .filter(|(i, _)| !next.discard[*i])
        .map(|(_, a)| a);
    next.host_logits =
        !super::super::decode_logits_step::argmax_readback_eligible(survivors, sched);
    if !next.host_logits {
        // 2026-09-25: The masks were fixed at launch; if a survivor's mask has
        // changed, the step is redone.
        next.discard_all = active
            .iter()
            .enumerate()
            .any(|(i, a)| !next.discard[i] && row_mask(a, toks.think_end_token) != next.masks[i]);
    }
    next
}

/// 2026-09-25: Settle everything in flight: what a tick that needs the device quiet
/// (arrivals, retirements, shutdown) calls before it goes on.
pub(super) fn drain(
    active: &mut Vec<ActiveSeq>,
    pipeline: &mut Pipeline,
    toks: &CommitTokens,
    sched: &crate::scheduler::sched_ctx::SchedCtx,
) {
    while pipeline.has_inflight() {
        if settle(active, pipeline, toks, sched) != Settled::Done {
            break;
        }
    }
}
