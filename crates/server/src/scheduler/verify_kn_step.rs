// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: Single-sequence verify of four or more drafts on an MTP serve:
//! a long prompt-lookup copy (`prompt_lookup_step`). One forward over
//! `[last_token, drafts..]` through the K-generic verify, then the K4 verdict,
//! which is K-generic too (commit, drafter trim, hidden save and propose).
//!
//! Owner: scheduler.
//! Invariants:
//! - Only `spec_capacity::SerialArm::KN` reaches this step, so it runs for an
//!   MTP serve without expert parallelism (`prompt_lookup_step` keeps copies
//!   under EP to 3 drafts), never for a DFlash drafter.

use super::*;

/// 2026-10-02: Verify `drafts` (4 or more) for `a` and apply the verdict.
/// `num_drafts` is the drafter's chain length for the propose that follows.
pub fn step_verify_kn(
    model: &dyn Model,
    a: &mut ActiveSeq,
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    drafts: &[u32],
    num_drafts: usize,
    verify_ctx: &crate::scheduler::logit_processors::LogitsContext,
    _dflash_verify_raw_argmax: bool,
) {
    let _step_timer = sched.io.tel.step_timer(a.seq.seq_len);
    if let Err(e) = model.sync_secondary() {
        tracing::error!("sync_secondary: {e:#}");
        super::lifecycle::fail_sequence(a, format!("sync_secondary: {e:#}"));
        return;
    }
    let mut tokens = Vec::with_capacity(drafts.len() + 1);
    tokens.push(a.last_token);
    tokens.extend_from_slice(drafts);

    let t_verify = sched.io.clock.now();
    let raw = match model.decode_verify_graphed_kgamma(&tokens, &mut a.seq, 0) {
        Ok(r) => r,
        Err(e) => {
            tracing::error!("decode_verify_graphed_kgamma (K={}): {e:#}", tokens.len());
            super::lifecycle::fail_sequence(
                a,
                format!("decode_verify_graphed_kgamma (K={}): {e:#}", tokens.len()),
            );
            return;
        }
    };
    let verify_us = sched
        .io
        .clock
        .now()
        .saturating_duration_since(t_verify)
        .as_micros();
    a.last_token_time = sched.io.clock.now();

    // 2026-10-02: The same pick function as the K2/K3/K4 steps; a position it
    // returns no pick for keeps the GPU argmax.
    let processed = crate::scheduler::verify_pipeline_helper::verify_pick_all_with_pipeline(
        model, &raw, a, verify_ctx, 0,
    );
    let v: Vec<u32> = (0..raw.len())
        .map(|j| processed.get(j).copied().unwrap_or(raw[j]))
        .collect();
    let mut num_accepted = 0usize;
    while num_accepted < drafts.len()
        && num_accepted + 1 < v.len()
        && drafts[num_accepted] == v[num_accepted]
    {
        num_accepted += 1;
    }
    let verify_lps = if let Some(top_logprobs) = a.top_logprobs {
        extract_verify_logprobs(model, &v, top_logprobs, 0)
    } else {
        Vec::new()
    };
    tracing::debug!(
        "KN verify: K={} accepted={num_accepted} seq_len={}",
        tokens.len(),
        a.seq.seq_len
    );
    k4_apply_verdict(
        model,
        a,
        sched,
        drafts,
        &v,
        verify_lps,
        num_drafts,
        num_accepted,
        K4Hidden::VerifyRow,
        verify_us,
    );
}
