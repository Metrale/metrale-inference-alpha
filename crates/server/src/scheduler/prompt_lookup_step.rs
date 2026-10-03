// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: Prompt-lookup decoding inside the MTP step: before a verify, a
//! sequence whose history repeats its last n-gram swaps the drafter's chain
//! for a copy of what followed the earlier occurrence; after the verify the
//! copy's window learns from the outcome.
//!
//! Owner: scheduler.
//! Invariants:
//! - A copy replaces `pending_drafts` only for a grammarless sequence that
//!   already has drafts, and is never longer than its verify slot's draft
//!   capacity or `--prompt-lookup-max-drafts`; in a batch or under expert
//!   parallelism, never longer than 3.
//! - Every copy marked in flight by [`take_rounds_with_copies`] is settled by
//!   [`settle_copies`] in the same step, so no mark outlives its verify.
//! - While a copy is in flight the drafter is trimmed as if none of its own
//!   drafts were accepted ([`drafter_accepted`]): the verified tokens were not
//!   the ones it drafted.

use super::*;

/// 2026-10-02: Length of the copy `a` is verifying this step; 0 when its
/// pending drafts are the drafter's.
pub(super) fn copy_in_flight(a: &ActiveSeq) -> usize {
    a.prompt_lookup.as_ref().map_or(0, |pl| pl.in_flight())
}

/// 2026-10-02: The accepted count to trim the MTP drafter's state by: `na`
/// for its own drafts, 0 when the verified drafts were a copy.
pub(super) fn drafter_accepted(a: &ActiveSeq, na: usize) -> usize {
    if copy_in_flight(a) > 0 { 0 } else { na }
}

/// 2026-10-02: For each of `verify_idxs` (sequences holding drafts), proposes a
/// copy and, on a match at least as long as the drafter's chain this step
/// (`min(pending drafts, ladder_nd)`), makes it the sequence's drafts. A
/// shorter copy would verify fewer tokens than the chain it displaces, so it
/// is dropped; the window is never asked for less than that chain. Returns
/// `(active index, seq_len before the verify)` of every sequence that took a
/// copy, for [`settle_copies`]. A no-op when prompt lookup is off or the batch
/// is wider than `--prompt-lookup-max-seqs`.
pub(super) fn take_rounds_with_copies(
    model: &dyn Model,
    active: &mut [ActiveSeq],
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    verify_idxs: &[usize],
    ladder_nd: usize,
) -> Vec<(usize, usize)> {
    let Some(cfg) = sched.prompt_lookup else {
        return Vec::new();
    };
    if active.len() > cfg.max_seqs {
        return Vec::new();
    }
    // 2026-10-02: A lone sequence verifies a copy of any length in one pass
    // (`step_verify_kn`); in a batch, or under expert parallelism (whose
    // workers follow only the K<=4 verify commands), a copy is cut to the
    // 4-row verify.
    let path_cap = if active.len() == 1 && !model.is_ep() {
        usize::MAX
    } else {
        BATCHED_COPY_MAX
    };
    let mut took = Vec::new();
    for &i in verify_idxs {
        let a = &mut active[i];
        if a.grammar_state.is_some() {
            continue;
        }
        let chain = a.pending_drafts.len().min(ladder_nd);
        let slot_cap = cfg
            .max_drafts
            .min(model.mtp_slot_draft_capacity(a.seq.slot_idx))
            .min(path_cap);
        if slot_cap < chain.max(1) {
            continue;
        }
        let pl = a.prompt_lookup.get_or_insert_with(|| {
            Box::new(metrale_speculative::prompt_lookup::PromptLookupSeq::new(
                &cfg,
            ))
        });
        let cap = slot_cap.min(pl.window().max(chain));
        // 2026-10-02: The lookup history is everything the sequence holds plus
        // `last_token`, which the verify feeds as its first row and which is
        // not in `seq.tokens` yet.
        a.seq.tokens.push(a.last_token);
        let copy = pl.propose(&a.seq.tokens, cap);
        a.seq.tokens.pop();
        match copy.map(bucket_copy) {
            Some(copy) if copy.len() >= chain => {
                a.pending_drafts = copy;
                a.pending_draft_conf.clear();
                took.push((i, a.seq.seq_len));
            }
            Some(_) => pl.abandon(),
            None => {}
        }
    }
    took
}

/// 2026-10-02: Drafts in the widest copy a batched verify takes (4 rows).
const BATCHED_COPY_MAX: usize = 3;

/// 2026-10-02: Cuts a copy of 4 or more tokens to the largest power of two that fits, so a
/// lone sequence's long verifies use few distinct widths (each width captures its own graph
/// per slot). Shorter copies are kept whole.
fn bucket_copy(mut copy: Vec<u32>) -> Vec<u32> {
    if copy.len() >= 4 {
        let keep = 1usize << (usize::BITS - 1 - copy.len().leading_zeros());
        copy.truncate(keep);
    }
    copy
}

/// 2026-10-02: Settles the copies [`take_rounds_with_copies`] put in flight:
/// a verdict advances `seq_len` by the accepted drafts plus one, so the
/// accepted count is read from that. Updates the serve-wide counters and logs
/// them every `LOG_EVERY` copies.
pub(super) fn settle_copies(
    active: &mut [ActiveSeq],
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    took: &[(usize, usize)],
) {
    const LOG_EVERY: u64 = 1024;
    for &(i, len_before) in took {
        let a = &mut active[i];
        let advanced = a.seq.seq_len.saturating_sub(len_before);
        let Some(pl) = a.prompt_lookup.as_mut() else {
            continue;
        };
        let accepted = advanced.saturating_sub(1);
        let proposed = pl.settle(accepted);
        let [n, p, acc] = sched.prompt_lookup_stats.get();
        let stats = [
            n + 1,
            p + proposed as u64,
            acc + accepted.min(proposed) as u64,
        ];
        sched.prompt_lookup_stats.set(stats);
        if stats[0] % LOG_EVERY == 1 {
            tracing::info!(
                copies = stats[0],
                proposed = stats[1],
                accepted = stats[2],
                window = pl.window(),
                "prompt-lookup: copies verified"
            );
        }
    }
}

#[cfg(test)]
#[path = "prompt_lookup_step_tests.rs"]
mod tests;
