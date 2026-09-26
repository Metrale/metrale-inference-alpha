// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: D-Cut: adaptive verification-depth pruning (arXiv 2607.14647).
//!
//! Owner: scheduler.
//! Invariants:
//! - When `plan` prunes, no sequence loses its first draft: every row count
//!   it returns is in `2..=ladder_nd + 1`.
//! - `plan` prunes with `SchedLevers::dcut_ratio`, which is always one of
//!   `BUCKETS` (`snap_ratio`).
//! - Pruning only truncates drafts that were already proposed, so it saves
//!   verify rows, not drafter work.
//!
//! Each draft costs one verify row, and a batch's rows `R = Σ rows_i` are
//! capped by `VERIFY_ROW_BUDGET`. D-Cut spends rows on the drafts most
//! likely to be accepted, and only in batches of at most `dcut_width_cap`
//! sequences.
//!
//! The batched propose reports each draft's top-1 log-probability
//! `ln c_{i,t}` (`argmax_bf16_batch_lp`). A draft at depth `j` is only
//! reachable if every draft before it was accepted, so its survival score is
//! the prefix product `s_{i,j} = Π_{t<=j} c_{i,t}`, which in log space is the
//! prefix sum. Every prunable position across the batch is ranked by that
//! score and the top `ratio` fraction is retained.
//!
//! Because log-probabilities are <= 0, `s_{i,j}` is non-increasing in `j`,
//! so the retained set is a contiguous prefix of each sequence's drafts and
//! the result is one draft count per sequence.

use super::types::ActiveSeq;

/// 2026-09-25: The retention ratios `snap_ratio` rounds to.
const BUCKETS: [f32; 4] = [0.25, 0.5, 0.75, 1.0];

/// 2026-09-25: Verify row-buffer capacity. It must equal the model's
/// `VERIFY_ROW_CAP` (verify_e2.rs), which `can_batch_verify` enforces as
/// `Σ ks <= VERIFY_ROW_CAP`.
pub(super) const VERIFY_ROW_BUDGET: usize = 160;

/// 2026-09-25: Widest verify batch, in sequences, that the model accepts:
/// `can_batch_verify` requires `(2..=VERIFY_WY_TABLE_SEQS).contains(&n)`.
pub(super) const WIDTH_CAP: usize = metrale_model_layers::layer::VERIFY_WY_TABLE_SEQS;

/// 2026-09-25: Expected default of `SchedLevers::dcut_width_cap`, the widest
/// verify batch (in sequences) that D-Cut prunes. The run's value comes from
/// `METRALE_MTP_DCUT_MAX_SEQS` in `SchedLevers::from_env`, which repeats the
/// 8 as a literal; `mtp_dcut_tests` pins this constant to it. 0 disables
/// pruning. Above the cap, [`plan`] returns the uniform ladder shape.
pub(super) const DCUT_WIDTH_CAP_DEFAULT: usize = 8;

/// 2026-09-25: Retention ratio from `METRALE_MTP_DCUT_RATIO` (default 0.75),
/// snapped to the nearest [`BUCKETS`] entry. `SchedLevers::from_env` reads it.
pub(crate) fn dcut_ratio_from_env() -> f32 {
    snap_ratio(
        std::env::var("METRALE_MTP_DCUT_RATIO")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(0.75),
    )
}

/// 2026-09-25: The nearest [`BUCKETS`] entry to `raw` (clamped to `0..=1`).
pub(super) fn snap_ratio(raw: f32) -> f32 {
    let raw = raw.clamp(0.0, 1.0);
    {
        *BUCKETS
            .iter()
            .min_by(|a, b| {
                (*a - raw)
                    .abs()
                    .partial_cmp(&(*b - raw).abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("BUCKETS is non-empty")
    }
}

/// 2026-09-25: Retained draft count per sequence for one verify batch.
///
/// `confs[i]` is sequence i's per-draft top-1 log-probability, in draft order.
/// A short or empty row means "not measured": those positions score 0.0
/// (certainty), which ranks them first.
///
/// Returns `retained[i]` in `1..=k_drafts` (all 0 when `k_drafts` is 0).
/// `Σ (retained + 1)` stays within `row_budget` whenever `row_budget >= 2n`,
/// the rows the mandatory first drafts already use.
pub(super) fn select(
    confs: &[&[f32]],
    k_drafts: usize,
    row_budget: usize,
    ratio: f32,
) -> Vec<usize> {
    let n = confs.len();
    // 2026-09-25: Depth 1 is mandatory (see module docs), so only depths
    // 2..=k_drafts are rankable. Nothing to do at k_drafts <= 1.
    let mut retained = vec![1usize.min(k_drafts); n];
    if k_drafts <= 1 || n == 0 {
        return retained;
    }
    let prunable = n * (k_drafts - 1);

    // 2026-09-25: Score every prunable position by its log survival (prefix sum).
    let mut ranked: Vec<(f32, usize, usize)> = Vec::with_capacity(prunable);
    for (i, c) in confs.iter().enumerate() {
        let mut acc = 0.0f32;
        for j in 0..k_drafts {
            // 2026-09-25: Missing measurement -> 0.0 (certain), which sorts
            // to the top.
            acc += c.get(j).copied().unwrap_or(0.0);
            if j >= 1 {
                ranked.push((acc, i, j));
            }
        }
    }
    // 2026-09-25: Descending by score; ties break on (sequence, depth), so the
    // selection is a deterministic function of the batch and, among equal
    // scores, a sequence's shallower draft ranks before its deeper one.
    ranked.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
            .then(a.2.cmp(&b.2))
    });

    let by_ratio = ((prunable as f32) * ratio).round() as usize;
    // 2026-09-25: Rows already committed: one base row + one mandatory draft
    // per sequence.
    let committed = 2 * n;
    let by_budget = row_budget.saturating_sub(committed);
    let keep = by_ratio.min(by_budget).min(prunable);

    for &(_, i, j) in ranked.iter().take(keep) {
        // 2026-09-25: Scores are non-increasing in depth, so the top-`keep` set
        // is already prefix-closed; `max` records the deepest retained position.
        retained[i] = retained[i].max(j + 1);
    }
    retained
}

/// 2026-09-25: Plan one verify batch. Chooses how many drafts each sequence
/// keeps from the drafter's confidences (`select`), orders the batch and
/// assigns the depths with `verify_key::verify_batch_order`, truncates each
/// sequence's drafts to its assigned depth, rewrites `batchable` into
/// dispatch order, and returns the row counts (`drafts + 1`) in that order.
///
/// When `verify_key::canonical_assignment(batchable.len())` holds (width at
/// least `CANONICAL_KEY_MIN_WIDTH`, overridable by
/// `METRALE_CANONICAL_KEY_MIN_WIDTH`, and no `METRALE_NO_CANONICAL_VERIFY_KEY`),
/// the depths are re-paired in descending order onto ssm slots in ascending
/// order, so the batched-verify graph key depends on the depth multiset, not
/// on which sequence got which depth. The multiset stays confidence-chosen.
/// Otherwise each sequence keeps its own depth, sorted deepest first, then
/// by slot.
///
/// When D-Cut is off (`METRALE_NO_MTP_DCUT`), `ladder_nd < 2`, or the batch is
/// wider than `dcut_width_cap`, this returns `rows` for every sequence and
/// leaves `batchable` and the drafts untouched.
pub(super) fn plan(
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    active: &mut [ActiveSeq],
    batchable: &mut Vec<usize>,
    ladder_nd: usize,
    rows: usize,
) -> Vec<usize> {
    let mut ks: Vec<usize> = vec![rows; batchable.len()];
    if !sched.levers.dcut_enabled
        || ladder_nd < 2
        || batchable.is_empty()
        || batchable.len() > sched.levers.dcut_width_cap
    {
        return ks;
    }
    let confs: Vec<&[f32]> = batchable
        .iter()
        .map(|&i| {
            let a = &active[i];
            // 2026-09-25: A confidence vector whose length differs from the
            // drafts is passed as empty ("not measured"), never as scores.
            if a.pending_draft_conf.len() == a.pending_drafts.len() {
                a.pending_draft_conf.as_slice()
            } else {
                &[]
            }
        })
        .collect();
    let retained = select(
        &confs,
        ladder_nd,
        VERIFY_ROW_BUDGET,
        sched.levers.dcut_ratio,
    );
    for (pos, r) in retained.iter().enumerate() {
        ks[pos] = (*r).clamp(1, ladder_nd) + 1;
    }
    // 2026-09-25: Dispatch order and depth assignment, from the ordering rule
    // the graph key also uses. Canonical: `ks_out[p]` is the multiset's p-th
    // deepest row count, not necessarily the confidence-chosen depth of the
    // sequence placed there. This is the only place depths are assigned;
    // `mtp_step` re-applies only the permutation, asking the gate with the
    // same `batchable.len()`.
    let slots: Vec<usize> = batchable
        .iter()
        .map(|&idx| active[idx].seq.ssm_slot_idx().unwrap_or(usize::MAX))
        .collect();
    let (order, ks_out) = metrale_model_layers::speculative::verify_key::verify_batch_order(
        &slots,
        &ks,
        metrale_model_layers::speculative::verify_key::canonical_assignment(batchable.len()),
    );
    // 2026-09-25: Truncate to the assigned depth. Every batchable sequence
    // enters with exactly `ladder_nd` drafts (`mtp_step` truncates the
    // surplus) and `ks_out[p] - 1` is in `1..=ladder_nd`, so this always
    // keeps a prefix of the proposed drafts.
    let reordered: Vec<usize> = order.iter().map(|&p| batchable[p]).collect();
    for (idx, &k) in reordered.iter().zip(&ks_out) {
        let a = &mut active[*idx];
        debug_assert!(
            k >= 2 && k - 1 <= a.pending_drafts.len(),
            "assigned depth {k} exceeds the {} drafts proposed",
            a.pending_drafts.len()
        );
        a.pending_drafts.truncate(k - 1);
        a.pending_draft_conf.truncate(k - 1);
    }
    sched.dcut.record(
        sched.levers.mtp_accept_debug,
        sched.levers.dcut_ratio,
        batchable.len() * rows,
        ks_out.iter().sum(),
        &ks_out,
    );
    *batchable = reordered;
    ks_out
}

/// 2026-09-25: Split a batch into verify chunks: `[lo, hi)` index ranges over
/// `ks`.
///
/// Each chunk holds at most `WIDTH_CAP` sequences and at most
/// `VERIFY_ROW_BUDGET` rows (unless one sequence alone exceeds it), the two
/// bounds `can_batch_verify` checks on the width and on `Σ ks`. No range is
/// empty. `plan` returns `ks` deepest first, so a chunk's first row count is
/// its widest and `VERIFY_ROW_BUDGET / ks[lo]` caps its sequence count.
pub(super) fn chunk_ranges(ks: &[usize]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut lo = 0usize;
    while lo < ks.len() {
        // 2026-09-25: Sequence cap for this chunk: the row budget over the
        // first (widest) row count, clamped to the verify stash width.
        let seq_cap = (VERIFY_ROW_BUDGET / ks[lo].max(1)).min(WIDTH_CAP);
        let mut hi = lo;
        let mut r = 0usize;
        while hi < ks.len() && hi - lo < seq_cap && r + ks[hi] <= VERIFY_ROW_BUDGET {
            r += ks[hi];
            hi += 1;
        }
        // 2026-09-25: Never emit an empty range, even when one sequence's
        // rows exceed the budget.
        if hi == lo {
            hi = lo + 1;
        }
        out.push((lo, hi));
        lo = hi;
    }
    out
}

/// 2026-09-25: D-Cut retained-rows telemetry, active only when
/// `METRALE_MTP_ACCEPT_DEBUG` is set: counters, logged as one line every
/// `PERIOD` recorded steps.
#[derive(Debug, Default)]
pub struct DcutTelemetry {
    steps: std::cell::Cell<u64>,
    full: std::cell::Cell<u64>,
    kept: std::cell::Cell<u64>,
}

impl DcutTelemetry {
    /// 2026-09-25: `debug` is the `METRALE_MTP_ACCEPT_DEBUG` lever; `ratio`
    /// the run's D-Cut ratio, named in the line.
    fn record(&self, debug: bool, ratio: f32, rows_full: usize, rows_kept: usize, ks: &[usize]) {
        const PERIOD: u64 = 200;
        if !debug {
            return;
        }
        self.full.set(self.full.get() + rows_full as u64);
        self.kept.set(self.kept.get() + rows_kept as u64);
        let steps = self.steps.get() + 1;
        self.steps.set(steps);
        if steps >= PERIOD {
            let steps = self.steps.replace(0).max(1);
            let full = self.full.replace(0).max(1);
            let kept = self.kept.replace(0);
            tracing::info!(
                "MTP D-Cut ratio={ratio:.2} steps={steps} rows_full={full} rows_kept={kept} \
                 kept_frac={:.3} last_ks={ks:?}",
                kept as f64 / full as f64,
            );
        }
    }
}
#[cfg(test)]
#[path = "mtp_dcut_tests.rs"]
mod tests;
