// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests for [`super`] (`mtp_dcut`), included by `mtp_dcut.rs`
//! with `#[path]`.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.
use super::*;

#[test]
fn ratio_one_retains_full_depth() {
    let c: Vec<Vec<f32>> = vec![vec![-0.1, -2.0, -3.0]; 4];
    let refs: Vec<&[f32]> = c.iter().map(|v| v.as_slice()).collect();
    assert_eq!(select(&refs, 3, 32, 1.0), vec![3, 3, 3, 3]);
}

#[test]
fn zero_ratio_keeps_the_mandatory_first_draft() {
    let c: Vec<Vec<f32>> = vec![vec![-0.1, -0.2, -0.3]; 4];
    let refs: Vec<&[f32]> = c.iter().map(|v| v.as_slice()).collect();
    assert_eq!(select(&refs, 3, 32, 0.0), vec![1, 1, 1, 1]);
}

#[test]
fn budget_spent_on_the_confident_sequence() {
    // 2026-09-25: seq 0 is confident at every depth, seq 1 at none.
    let c = [vec![-0.01f32, -0.02, -0.03], vec![-3.0f32, -4.0, -5.0]];
    let refs: Vec<&[f32]> = c.iter().map(|v| v.as_slice()).collect();
    // 2026-09-25: 2 sequences x 2 prunable depths = 4 candidates; ratio 0.5 keeps 2,
    // both of which belong to seq 0.
    assert_eq!(select(&refs, 3, 32, 0.5), vec![3, 1]);
}

#[test]
fn retained_set_is_always_a_prefix() {
    let c = [vec![-0.1f32, -9.0, -0.001], vec![-0.2f32, -0.2, -0.2]];
    let refs: Vec<&[f32]> = c.iter().map(|v| v.as_slice()).collect();
    let r = select(&refs, 3, 32, 0.5);
    // 2026-09-25: Depth-3 of seq 0 has a worse survival score than depth-2
    // despite its own high confidence, because survival is the prefix
    // product.
    assert!(r[0] <= 2, "prefix product must dominate the local value");
    assert!(r.iter().all(|&k| (1..=3).contains(&k)));
}

#[test]
fn row_budget_is_never_exceeded() {
    let c: Vec<Vec<f32>> = vec![vec![-0.001, -0.001, -0.001]; 8];
    let refs: Vec<&[f32]> = c.iter().map(|v| v.as_slice()).collect();
    // 2026-09-25: 8 sequences, budget 24 rows: 16 committed, 8 spare -> 8
    // extra depths.
    let r = select(&refs, 3, 24, 1.0);
    let rows: usize = r.iter().map(|k| k + 1).sum();
    assert!(rows <= 24, "rows={rows}");
}

#[test]
fn missing_confidences_are_never_pruned() {
    let empty: Vec<f32> = Vec::new();
    let c = [vec![-5.0f32, -5.0, -5.0], empty];
    let refs: Vec<&[f32]> = c.iter().map(|v| v.as_slice()).collect();
    assert_eq!(select(&refs, 3, 32, 0.5), vec![1, 3]);
}

#[test]
fn chunk_ranges_reproduce_the_uniform_caps() {
    // 2026-09-25: Every shape the default ladder and the adaptive n=16 rung
    // produce is a single chunk.
    assert_eq!(chunk_ranges(&[4; 8]), vec![(0, 8)]);
    assert_eq!(chunk_ranges(&[3; 8]), vec![(0, 8)]);
    // 2026-09-25: n=16 at 2 drafts (the adaptive rung's raised value):
    // [3; 16] = 48 rows, one chunk.
    assert_eq!(chunk_ranges(&[3; 16]), vec![(0, 16)]);
    assert_eq!(chunk_ranges(&[2; 16]), vec![(0, 16)]);
    // 2026-09-25: The 32:1 rung: one chunk up to n=32 (R = 64).
    assert_eq!(chunk_ranges(&[2; 17]), vec![(0, 17)]);
    assert_eq!(chunk_ranges(&[2; 32]), vec![(0, 32)]);
}

#[test]
fn chunk_ranges_seq_cap_derives_from_the_row_budget() {
    // 2026-09-29: Two bounds apply, and the chunk takes the smaller: the row
    // budget (`VERIFY_ROW_BUDGET` = 256, giving 128 sequences at rows=2, 85 at
    // rows=3, 64 at rows=4, 32 at rows=8) and the verify stash width `W`
    // (`VERIFY_WY_TABLE_SEQS` = 128).
    const W: usize = metrale_model_layers::layer::VERIFY_WY_TABLE_SEQS;
    assert_eq!((VERIFY_ROW_BUDGET, W), (256, 128));
    // 2026-09-29: rows=2: both bounds are 128, so a 128-sequence k=1 verify is ONE chunk
    // (one forward, one read of the expert weights).
    assert_eq!(chunk_ranges(&[2; 128]), vec![(0, 128)]);
    assert_eq!(chunk_ranges(&[2; 129]), vec![(0, W), (W, 129)]);
    // 2026-09-29: rows=3: the row budget (85) is the tighter bound.
    assert_eq!(chunk_ranges(&[3; 21]), vec![(0, 21)]);
    assert_eq!(chunk_ranges(&[3; 85]), vec![(0, 85)]);
    assert_eq!(chunk_ranges(&[3; 86]), vec![(0, 85), (85, 86)]);
    // 2026-09-29: rows=4: row budget 64.
    assert_eq!(chunk_ranges(&[4; 9]), vec![(0, 9)]);
    assert_eq!(chunk_ranges(&[4; 64]), vec![(0, 64)]);
    assert_eq!(chunk_ranges(&[4; 80]), vec![(0, 64), (64, 80)]);
    assert_eq!(chunk_ranges(&[3; 10]), vec![(0, 10)]);
    // 2026-09-29: rows=8: 256/8 = 32.
    assert_eq!(chunk_ranges(&[8; 32]), vec![(0, 32)]);
    assert_eq!(chunk_ranges(&[8; 33]), vec![(0, 32), (32, 33)]);
}

#[test]
fn chunk_ranges_respect_the_row_budget_when_ragged() {
    // 2026-09-25: Deepest-first, mixed depths: rows must never exceed the
    // budget per chunk.
    let ks = vec![4, 4, 4, 4, 4, 3, 3, 2, 2, 2];
    for (lo, hi) in chunk_ranges(&ks) {
        let rows: usize = ks[lo..hi].iter().sum();
        assert!(rows <= VERIFY_ROW_BUDGET, "rows={rows}");
        assert!(hi > lo);
    }
}

#[test]
fn dcut_width_cap_default_is_the_measured_win_regime() {
    assert_eq!(DCUT_WIDTH_CAP_DEFAULT, 8);
}

/// 2026-09-25: The canonical assignment re-pairs exactly the multiset
/// `select` produced: Σ rows is unchanged, slots come out ascending and
/// depths descending. `plan` takes a `SchedCtx` and `ActiveSeq`s, so this
/// test composes the functions it calls instead.
#[test]
fn canonical_assignment_preserves_the_selected_row_total() {
    use metrale_model_layers::speculative::verify_key::verify_batch_order;
    // 2026-09-25: A confidence spread that prunes raggedly: seq 0 collapses, seq 3 is
    // confident throughout.
    let c = [
        vec![-0.01f32, -6.0, -7.0],
        vec![-0.01f32, -0.02, -4.0],
        vec![-0.01f32, -5.0, -6.0],
        vec![-0.01f32, -0.02, -0.03],
    ];
    let refs: Vec<&[f32]> = c.iter().map(|v| v.as_slice()).collect();
    let retained = select(&refs, 3, VERIFY_ROW_BUDGET, 0.5);
    let ks: Vec<usize> = retained.iter().map(|r| r + 1).collect();
    assert!(ks.iter().any(|&k| k != ks[0]), "the case must be ragged");
    // 2026-09-25: Pool slots out of batch order.
    let slots = [5usize, 0, 7, 2];
    let (order, depths) = verify_batch_order(&slots, &ks, true);
    assert_eq!(
        depths.iter().sum::<usize>(),
        ks.iter().sum::<usize>(),
        "Σ rows must survive the re-pairing — the row budget depends on it"
    );
    let placed: Vec<usize> = order.iter().map(|&i| slots[i]).collect();
    assert!(placed.windows(2).all(|w| w[0] < w[1]), "{placed:?}");
    assert!(depths.windows(2).all(|w| w[0] >= w[1]), "{depths:?}");
    // 2026-09-25: Every assigned depth is in 2..=4, the MTP row range
    // `can_batch_verify` accepts.
    assert!(depths.iter().all(|k| (2..=4).contains(k)));
    assert_eq!(chunk_ranges(&depths), vec![(0, 4)]);
}

#[test]
fn ratio_snaps_to_a_bucket() {
    // 2026-09-25: Pure snapping arithmetic, no env: 0.6 is closest to 0.5.
    let nearest = |raw: f32| {
        *BUCKETS
            .iter()
            .min_by(|a, b| (*a - raw).abs().partial_cmp(&(*b - raw).abs()).unwrap())
            .unwrap()
    };
    assert_eq!(nearest(0.6), 0.5);
    assert_eq!(nearest(0.9), 1.0);
    assert_eq!(nearest(0.1), 0.25);
}

/// 2026-09-25: D-Cut prunes at n=2, so skipping the planner at small widths
/// would change behaviour.
///
/// With `--num-drafts` 3 or more, the default ladder gives 3 drafts at n=2
/// (rows = 4), inside D-Cut's range (`ladder_nd >= 2`, n <= `dcut_width_cap`).
/// With the default ratio 0.75, 3 of the 2x2 = 4 prunable positions are
/// kept, so one sequence loses its deepest draft: retained `{3, 2}`, verify
/// rows `{4, 3}`, R = 7 instead of the uniform 8.
#[test]
fn width_two_is_inside_the_pruning_envelope_and_is_not_uniform() {
    // 2026-09-25: Both sequences confident, seq 1 slightly less so at its deepest draft.
    let c = [vec![-0.01f32, -0.02, -0.03], vec![-0.01f32, -0.02, -0.40]];
    let refs: Vec<&[f32]> = c.iter().map(|v| v.as_slice()).collect();
    let retained = select(&refs, 3, VERIFY_ROW_BUDGET, 0.75);
    assert_eq!(
        retained,
        vec![3, 2],
        "ratio 0.75 drops exactly one position"
    );
    let rows: Vec<usize> = retained.iter().map(|r| r + 1).collect();
    assert_ne!(rows, vec![4, 4], "the n=2 plan is NOT the uniform shape");
    assert_eq!(rows.iter().sum::<usize>(), 7);
    assert_eq!(chunk_ranges(&rows), vec![(0, 2)], "still a single chunk");
}

/// 2026-09-25: Below `verify_key::CANONICAL_KEY_MIN_WIDTH` the width gate
/// turns off the canonical re-pairing, not the pruning. The test composes
/// `select`, the gate and `verify_batch_order` as `plan` does.
///
/// At n=2 the ragged rows `{4, 3}` (R = 7 against the uniform 8) survive and
/// each sequence keeps its own depth. The canonical arm keeps the same rows
/// but moves the deepest row count to the lowest slot.
#[test]
fn below_the_gate_the_pairing_is_legacy_but_the_pruning_is_kept() {
    use metrale_model_layers::speculative::verify_key::{canonical_assignment, verify_batch_order};
    let c = [vec![-0.01f32, -0.02, -0.03], vec![-0.01f32, -0.02, -0.40]];
    let refs: Vec<&[f32]> = c.iter().map(|v| v.as_slice()).collect();
    let ks: Vec<usize> = select(&refs, 3, VERIFY_ROW_BUDGET, 0.75)
        .iter()
        .map(|r| r + 1)
        .collect();
    assert_eq!(ks, vec![4, 3]);
    // 2026-09-25: Seq 0 (the deeper one) sits on the higher pool slot, so the two arms
    // disagree — the case the gate is actually deciding.
    let slots = [5usize, 0];
    assert!(!canonical_assignment(slots.len()));

    let (order, depths) = verify_batch_order(&slots, &ks, canonical_assignment(slots.len()));
    let paired: Vec<(usize, usize)> = order.iter().map(|&i| (slots[i], ks[i])).collect();
    assert_eq!(
        depths.iter().sum::<usize>(),
        7,
        "the row saving must survive the gate — this is not a D-Cut kill switch"
    );
    assert_eq!(
        paired,
        vec![(5, 4), (0, 3)],
        "each sequence keeps the depth its confidence earned"
    );
    assert_eq!(depths, vec![4, 3]);
    assert_eq!(chunk_ranges(&depths), vec![(0, 2)]);

    // 2026-09-25: The canonical arm re-pairs: deepest onto the lowest slot.
    let (c_order, c_depths) = verify_batch_order(&slots, &ks, true);
    let c_paired: Vec<(usize, usize)> = c_order
        .iter()
        .zip(&c_depths)
        .map(|(&i, &k)| (slots[i], k))
        .collect();
    assert_eq!(c_paired, vec![(0, 4), (5, 3)]);
    assert_ne!(c_paired, paired);
    assert_eq!(c_depths.iter().sum::<usize>(), 7);
}

// 2026-09-25: Width bound. `can_batch_verify` refuses a batch wider than
// `VERIFY_WY_TABLE_SEQS`, and `mtp_step` sends every sequence of a refused
// chunk to the per-sequence verify loop. 2026-09-29: With 128 and a 256-row
// budget the row budget binds first at every rows >= 2; this pins that no
// chunk ever exceeds the width bound whichever binds.
#[test]
fn chunk_ranges_never_exceed_the_verify_width_bound() {
    // 2026-09-25: The width bound `can_batch_verify` enforces, read from the
    // model crate.
    const WIDTH_CAP: usize = metrale_model_layers::layer::VERIFY_WY_TABLE_SEQS;
    // 2026-09-29: Rows 2..=4, at widths past every row-derived cap (64/85/128).
    for rows in 2..=4usize {
        for n in 2..=300usize {
            let ks = vec![rows; n];
            for (lo, hi) in chunk_ranges(&ks) {
                assert!(
                    hi - lo <= WIDTH_CAP,
                    "rows={rows} n={n}: chunk ({lo},{hi}) is {} sequences wide, \
                     above the {WIDTH_CAP}-slot verify stash — can_batch_verify \
                     refuses it and the whole chunk serializes",
                    hi - lo
                );
            }
        }
    }
}

// 2026-09-25: Ragged (D-Cut) shapes must obey the width bound too: the cap
// is taken from `ks[lo]`, the chunk's deepest row count, so a chunk that
// starts deep and continues shallow gets the deep cap while admitting
// shallow rows.
#[test]
fn ragged_chunks_also_respect_the_width_bound() {
    // 2026-09-25: Deepest first, as `plan` returns it: 4 deep rows, then a long
    // shallow tail. 2026-09-29: seq_cap 256/4 = 64 admits a shallow tail; the
    // width and row bounds must both hold on every chunk.
    let mut ks = vec![4usize; 4];
    ks.extend(std::iter::repeat_n(2usize, 200));
    for (lo, hi) in chunk_ranges(&ks) {
        assert!(
            hi - lo <= metrale_model_layers::layer::VERIFY_WY_TABLE_SEQS,
            "ragged chunk ({lo},{hi}) is {} wide",
            hi - lo
        );
        // 2026-09-25: The row budget must still hold — the width clamp adds a bound, it
        // does not relax the existing one.
        let r: usize = ks[lo..hi].iter().sum();
        assert!(r <= VERIFY_ROW_BUDGET, "rows={r}");
        assert!(hi > lo, "empty range");
    }
}

// 2026-10-02: --draft-confidence-stop through `plan_with_stop`.

fn stop_ctx(dcut: bool) -> crate::scheduler::sched_ctx::SchedCtx {
    let mut ctx = crate::scheduler::sched_ctx::SchedCtx::for_test();
    let mut levers = crate::scheduler::levers::SchedLevers::defaults();
    levers.dcut_enabled = dcut;
    ctx.levers = std::sync::Arc::new(levers);
    ctx
}

/// 2026-10-02: Active sequences holding the given drafts' confidences (one
/// draft per confidence; draft ids are irrelevant here).
fn seqs_with(confs: &[&[f32]]) -> Vec<ActiveSeq> {
    confs
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let (mut a, _rx) = crate::scheduler::test_support::active_seq(i, 1);
            a.pending_drafts = (0..c.len() as u32).map(|d| 100 + d).collect();
            a.pending_draft_conf = c.to_vec();
            a
        })
        .collect()
}

const LN_HALF: f32 = -std::f32::consts::LN_2;

#[test]
fn the_stop_caps_each_chain_at_its_first_unconfident_draft() {
    let sched = stop_ctx(false);
    let mut active = seqs_with(&[
        &[-0.01, -0.02, -0.03],
        &[-0.01, -2.0, -0.1],
        &[-3.0, -0.1, -0.1],
    ]);
    let mut batchable = vec![0, 1, 2];
    let ks = plan_with_stop(&sched, &mut active, &mut batchable, 3, 4, Some(LN_HALF));
    // 2026-10-02: Deepest first (below the canonical width each keeps its own).
    assert_eq!(ks, vec![4, 3, 2]);
    assert_eq!(batchable, vec![0, 1, 2]);
    let held: Vec<usize> = active.iter().map(|a| a.pending_drafts.len()).collect();
    assert_eq!(held, vec![3, 2, 1], "drafts truncated to the planned depth");
}

#[test]
fn without_the_stop_or_d_cut_the_plan_is_uniform() {
    let sched = stop_ctx(false);
    let mut active = seqs_with(&[&[-0.01, -2.0, -0.1], &[-3.0, -0.1, -0.1]]);
    let mut batchable = vec![0, 1];
    let ks = plan_with_stop(&sched, &mut active, &mut batchable, 3, 4, None);
    assert_eq!(ks, vec![4, 4]);
    assert!(active.iter().all(|a| a.pending_drafts.len() == 3));
}

#[test]
fn the_stop_applies_above_the_d_cut_width_cap() {
    let sched = stop_ctx(true);
    let n = sched.levers.dcut_width_cap + 4;
    let confident: &[f32] = &[-0.01, -0.01, -0.01];
    let short: &[f32] = &[-0.01, -5.0, -0.01];
    let confs: Vec<&[f32]> = (0..n)
        .map(|i| if i % 2 == 0 { confident } else { short })
        .collect();
    let mut active = seqs_with(&confs);
    let mut batchable: Vec<usize> = (0..n).collect();
    let ks = plan_with_stop(&sched, &mut active, &mut batchable, 3, 4, Some(LN_HALF));
    let total: usize = ks.iter().sum();
    assert_eq!(
        total,
        (n / 2) * 4 + (n / 2) * 3,
        "half the chains stop after 2 drafts"
    );
}

#[test]
fn a_short_chain_joins_at_its_own_depth_when_re_pairing_cannot_hold_it() {
    // 2026-10-02: At the canonical width the depths would be re-paired
    // deepest-first in slot order; sequence 0 holds only one draft, so the
    // re-pairing is infeasible and every sequence keeps its own depth.
    let sched = stop_ctx(false);
    let n = metrale_model_layers::speculative::verify_key::CANONICAL_KEY_MIN_WIDTH;
    let full: &[f32] = &[-0.01, -0.01, -0.01];
    let one: &[f32] = &[-0.01];
    let confs: Vec<&[f32]> = (0..n).map(|i| if i == 0 { one } else { full }).collect();
    let mut active = seqs_with(&confs);
    let mut batchable: Vec<usize> = (0..n).collect();
    let ks = plan_with_stop(&sched, &mut active, &mut batchable, 3, 4, Some(LN_HALF));
    for (pos, &i) in batchable.iter().enumerate() {
        assert_eq!(ks[pos], active[i].pending_drafts.len() + 1);
    }
    assert_eq!(active[0].pending_drafts.len(), 1);
    assert_eq!(ks.iter().sum::<usize>(), 2 + (n - 1) * 4);
}
