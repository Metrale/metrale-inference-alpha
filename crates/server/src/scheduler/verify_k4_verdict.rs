// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: applies a K-row verify verdict to one sequence; shared by
//! `step_verify_k4` and `step_verify_k4_batched`.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::*;

/// 2026-09-25: where the next propose's accepted-position hidden comes from.
#[derive(Clone, Copy)]
pub(super) enum K4Hidden {
    /// 2026-09-25: the live verify forward's row `num_accepted`
    /// (`save_hidden_for_mtp`); used by `step_verify_k4`.
    VerifyRow,
    /// 2026-09-25: stash slot `i` written by `stash_verify_hidden_rows`
    /// (`save_hidden_for_mtp_from_stash`). No caller constructs it.
    Stash(usize),
    /// 2026-09-25: apply the verdict but skip the hidden save and the
    /// propose; the caller proposes afterwards. Used by
    /// `step_verify_k4_batched`.
    DeferPropose,
}

#[inline]
fn save_hidden(model: &dyn Model, hidden: K4Hidden, na: usize) -> anyhow::Result<()> {
    match hidden {
        K4Hidden::VerifyRow => model.save_hidden_for_mtp(na, 0),
        K4Hidden::Stash(i) => model.save_hidden_for_mtp_from_stash(i, 0),
        K4Hidden::DeferPropose => Ok(()),
    }
}

/// 2026-09-25: apply a verify verdict to one sequence.
///
/// `v` holds one pick per verified row, the last being the bonus row.
/// `nd = min(drafts.len(), v.len() - 1)` drafts are judged and any surplus is
/// dropped; `num_accepted` is clamped to `nd`. Accepted drafts and then
/// `v[na]` are emitted, `seq_len`/`tokens` are rewound by the rejected
/// count, and the proposer and SSM state are trimmed and committed to the
/// accepted prefix. Unless `hidden` is `DeferPropose`, the accepted hidden
/// is saved and `num_drafts` new drafts are proposed.
///
/// Order: on full accept, emit, commit, save, trim, propose; otherwise
/// rewind, trim, commit, emit, save, propose. The function returns early,
/// skipping the later stages, when an emit finishes the sequence or the
/// commit or the hidden save fails.
#[allow(clippy::too_many_arguments)]
pub(super) fn k4_apply_verdict(
    model: &dyn Model,
    a: &mut ActiveSeq,
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    drafts: &[u32],
    v: &[u32],
    verify_lps: Vec<crate::api::TokenLogprobs>,
    num_drafts: usize,
    num_accepted: usize,
    hidden: K4Hidden,
    verify_us: u128,
) {
    let defer = matches!(hidden, K4Hidden::DeferPropose);
    // 2026-09-25: `nd` is sized by `v` (the rows the forward added), not by
    // `drafts`: the DFlash bootstrap in `step_mtp` passes all its drafts to
    // `step_verify_k4`, which verifies three. Rewinding by `drafts.len()`
    // would erase emitted tokens. The clamp holds in release builds, where
    // the `debug_assert!` does not.
    let k_rows = v.len();
    debug_assert!(k_rows >= 1, "verdict needs at least the bonus row");
    let nd = drafts.len().min(k_rows.saturating_sub(1));
    if drafts.len() > nd {
        // 2026-09-25: the unverified surplus is dropped. The caller has
        // already taken it out of `pending_drafts`, and
        // `trim_proposer_state` trims by the drafter's own count, not this
        // slice.
        tracing::debug!(
            "k4 verdict: {} drafts carried into a {}-row verify — verifying the \
             first {nd}, dropping the surplus",
            drafts.len(),
            k_rows,
        );
    }
    let drafts = &drafts[..nd];
    let na = num_accepted.min(nd);

    if na == nd {
        // 2026-09-25: full accept: every draft matched; `v[nd]` is the bonus
        // token.
        for j in 0..nd {
            emit_token(a, drafts[j], verify_lps.get(j).cloned(), sched);
            if a.finished {
                return;
            }
        }
        emit_token(a, v[nd], verify_lps.get(nd).cloned(), sched);
        if a.finished {
            return;
        }
        a.last_token = v[nd];

        // 2026-09-25: commit all `k_rows` rows; `commit_accepted_prefix`
        // treats a full accept as a no-op.
        if let Err(e) = model.commit_accepted_prefix(&mut a.seq, k_rows, k_rows) {
            // 2026-09-25: the SSM state cannot be trusted after a failed
            // commit, so the sequence ends.
            tracing::error!("commit_accepted_prefix (K={k_rows} accept-{k_rows}): {e:#}");
            a.finished = true;
            return;
        }
    } else {
        // 2026-09-25: partial accept / reject: rewind the rejected tail.
        a.seq.seq_len -= nd - na;
        for _ in 0..(nd - na) {
            a.seq.tokens.pop();
        }
        if let Err(e) = model.trim_proposer_state(&mut a.seq, na, 0) {
            tracing::error!("trim_proposer_state: {e:#}");
        }
        // 2026-09-25: commit rows `0..=na` (the last verified token plus
        // the `na` accepted drafts); row `na`'s pick `v[na]` is the
        // correction token.
        if let Err(e) = model.commit_accepted_prefix(&mut a.seq, na + 1, k_rows) {
            tracing::error!(
                "commit_accepted_prefix (K={k_rows} accept-{}): {e:#}",
                na + 1
            );
            a.finished = true;
            return;
        }
        for j in 0..na {
            emit_token(a, drafts[j], verify_lps.get(j).cloned(), sched);
            if a.finished {
                return;
            }
        }
        emit_token(a, v[na], verify_lps.get(na).cloned(), sched);
        if a.finished {
            return;
        }
        a.last_token = v[na];
    }

    if !defer && let Err(e) = save_hidden(model, hidden, na) {
        tracing::error!("save_hidden_for_mtp({na}): {e:#}");
        return;
    }
    if na == nd {
        // 2026-09-25: On a full accept the proposer is trimmed only after the hidden state above is saved.
        if let Err(e) = model.trim_proposer_state(&mut a.seq, na, 0) {
            tracing::error!("trim_proposer_state: {e:#}");
        }
    }
    if !defer {
        let t_propose = sched.io.clock.now();
        let _mtp_grammar_mask = mtp_grammar_mask_for(a);
        match model.run_mtp_propose_multi(
            a.last_token,
            a.seq.seq_len,
            num_drafts,
            &mut a.seq,
            0,
            _mtp_grammar_mask.as_deref(),
        ) {
            Ok(d) if !d.is_empty() => a.pending_drafts = d,
            Ok(_) => {}
            Err(e) => {
                tracing::error!("run_mtp_propose_multi: {e:#}");
            }
        }
        let propose_us = sched
            .io
            .clock
            .now()
            .saturating_duration_since(t_propose)
            .as_micros();
        tracing::debug!(
            "K{k_rows} ACCEPT-{na}: verify={verify_us}μs propose={propose_us}μs seq_len={}",
            a.seq.seq_len
        );
    }
    crate::scheduler::verify_k4_step::stats::k4_record_outcome(sched, na, a.seq.seq_len);
    sched.io.tel.spec_verified(k_rows - 1, na);
}
