// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: K=3 acceptance telemetry: the positional and outcome
//! recorders `step_verify_k3` calls, and the summary line each logs every
//! `K3_SUMMARY_PERIOD` steps. Nothing here influences a pick.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

// 2026-09-25: the counters are fields of the run's `SpecStats`
// (`sched.io.tel.stats()`). A summary line is logged every
// `K3_SUMMARY_PERIOD` recorded steps.
const K3_SUMMARY_PERIOD: u64 = 100;

// 2026-09-25: why an unconditional position-2 counter. The accept chain
// stops at the first mismatch, so it scores draft 2 only on steps where
// draft 1 matched: a conditional rate on a biased sample. The verify forward
// yields a pick at every row, so `drafts[1] == v1` can be scored on every
// step. When draft 1 was wrong, `v1` is the target's pick after the wrong
// draft, so that sample measures the drafter on a context the target did
// not choose.

#[inline]
pub(super) fn k3_record_positional(
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    d1_match: bool,
    d2_match: bool,
    seq_len: usize,
) {
    sched
        .io
        .tel
        .stats()
        .k3_steps
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if d1_match {
        sched
            .io
            .tel
            .stats()
            .k3_d1_match
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if d2_match {
            sched
                .io
                .tel
                .stats()
                .k3_d2_match_cond
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }
    if d2_match {
        sched
            .io
            .tel
            .stats()
            .k3_d2_match_uncond
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    if sched
        .io
        .tel
        .stats()
        .k3_steps
        .load(std::sync::atomic::Ordering::Relaxed)
        >= K3_SUMMARY_PERIOD
    {
        let steps = sched
            .io
            .tel
            .stats()
            .k3_steps
            .swap(0, std::sync::atomic::Ordering::Relaxed)
            .max(1);
        let d1 = sched
            .io
            .tel
            .stats()
            .k3_d1_match
            .swap(0, std::sync::atomic::Ordering::Relaxed);
        let d2u = sched
            .io
            .tel
            .stats()
            .k3_d2_match_uncond
            .swap(0, std::sync::atomic::Ordering::Relaxed);
        let d2c = sched
            .io
            .tel
            .stats()
            .k3_d2_match_cond
            .swap(0, std::sync::atomic::Ordering::Relaxed);
        let p1 = (d1 as f64) / (steps as f64);
        let p2_uncond = (d2u as f64) / (steps as f64);
        let p2_cond = if d1 > 0 {
            (d2c as f64) / (d1 as f64)
        } else {
            f64::NAN
        };
        tracing::info!(
            "K3 positional: steps={steps} p1={p1:.3} p2_uncond={p2_uncond:.3} \
             p2_cond={p2_cond:.3} (d1={d1} d2u={d2u} d2c={d2c}) seq_len={seq_len} \
             [p2_uncond ~= p2_cond => position 2 genuinely worse; \
              p2_uncond ~= p1 => p2_cond is survivorship]"
        );
    }
}

#[inline]
pub(super) fn k3_record_outcome(
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    num_accepted: usize,
    seq_len: usize,
) {
    let counter = match num_accepted {
        2 => &sched.io.tel.stats().k3_accept[2],
        1 => &sched.io.tel.stats().k3_accept[1],
        _ => &sched.io.tel.stats().k3_accept[0],
    };
    counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    sched.io.tel.spec_verified(2, num_accepted);
    let total = sched.io.tel.stats().k3_accept[2].load(std::sync::atomic::Ordering::Relaxed)
        + sched.io.tel.stats().k3_accept[1].load(std::sync::atomic::Ordering::Relaxed)
        + sched.io.tel.stats().k3_accept[0].load(std::sync::atomic::Ordering::Relaxed);
    if total >= K3_SUMMARY_PERIOD {
        let a2 = sched.io.tel.stats().k3_accept[2].swap(0, std::sync::atomic::Ordering::Relaxed);
        let a1 = sched.io.tel.stats().k3_accept[1].swap(0, std::sync::atomic::Ordering::Relaxed);
        let a0 = sched.io.tel.stats().k3_accept[0].swap(0, std::sync::atomic::Ordering::Relaxed);
        let total = (a2 + a1 + a0).max(1);
        let mean = (2 * a2 + a1) as f64 / total as f64;
        tracing::info!(
            "K3 summary: {a2} accept-2 / {a1} accept-1 / {a0} reject in last {total} steps (mean accepted={mean:.2}) seq_len={seq_len}"
        );
    }
}
