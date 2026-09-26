// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: EP=2 MoE semantics, checked in one process with exact arithmetic.
//!
//! Owner: model-engine tests.
//! Invariants: none beyond the types.
//!
//! # The EP path this models
//!
//! The engine's EP MoE is masked-local plus all-reduce, not token dispatch:
//!
//! 1. Every rank runs the replicated router over all `num_experts`.
//! 2. A rank loads only the experts in its `ModelConfig::local_expert_range`;
//!    the others are `ExpertWeight::null()` (`weight_map/loaders_moe.rs`), and
//!    the grouped GEMM returns early for a null expert, leaving the zeroed
//!    output (`kernels/gb10/common/moe_w4a16_grouped_gemm.cu`, `B_expert == 0`).
//! 3. Each rank blends its partial weighted sum.
//! 4. `all_reduce` over the hidden vector sums the partials
//!    (`layers/moe/forward/ep_reduce.rs`).
//! 5. Under EP the shared expert is left out of the pre-reduce blend
//!    (`shared_for_blend` in `layers/moe/forward.rs`) and added once after the
//!    reduce (`ep_reduce_shared`); inside the blend it would be summed
//!    `world_size` times.
//!
//! Since expert outputs enter as a sum, this equals token dispatch in exact
//! arithmetic. `MoeLayer::forward_ep_dispatch` (`layers/moe/forward_ep.rs`)
//! has no caller.
//!
//! # What the tests check
//!
//! * Every (token, expert) pair runs on exactly one rank.
//! * Global expert ids and routing weights pass through the partition unchanged.
//! * The sum of the per-rank partials equals the single-process reference
//!   exactly.
//! * The shared expert is added once, not once per rank.
//! * A rank that owns none of a token's experts contributes an exact zero.
//!
//! The collective itself is not exercised. Expert outputs are small integers in
//! `f32`, so every sum is exact and `assert_eq!` on floats is sound.

use metrale_model_layers::layers::ep_dispatch::build_ep_routing_table;

const NUM_EXPERTS: usize = 8;
const EP_WORLD: usize = 2;
const HIDDEN: usize = 4;
const TOP_K: usize = 2;

/// 2026-09-25: Local expert range for `rank`, the same split as
/// `ModelConfig::local_expert_range`.
fn range_of(rank: usize) -> (usize, usize) {
    let per = NUM_EXPERTS / EP_WORLD;
    let start = rank * per;
    let end = if rank == EP_WORLD - 1 {
        NUM_EXPERTS
    } else {
        start + per
    };
    (start, end)
}

/// 2026-09-25: Stand-in for expert `e`'s output: an integer vector that differs
/// per expert, so a mis-routed token changes the result.
fn expert_out(e: u32) -> [f32; HIDDEN] {
    let base = (e as f32 + 1.0) * 10.0;
    [base, base + 1.0, base + 2.0, base + 3.0]
}

fn shared_out() -> [f32; HIDDEN] {
    [1.0, 2.0, 3.0, 4.0]
}

/// 2026-09-25: Single-process reference: weighted sum over every top-k expert,
/// plus the shared expert once.
fn reference(indices: &[u32], weights: &[f32], num_tokens: usize) -> Vec<[f32; HIDDEN]> {
    let mut out = vec![[0.0f32; HIDDEN]; num_tokens];
    for t in 0..num_tokens {
        for k in 0..TOP_K {
            let f = t * TOP_K + k;
            let e = expert_out(indices[f]);
            for h in 0..HIDDEN {
                out[t][h] += weights[f] * e[h];
            }
        }
        for h in 0..HIDDEN {
            out[t][h] += shared_out()[h];
        }
    }
    out
}

/// 2026-09-25: One rank's partial over the routed experts it owns, and the
/// number of pairs it ran. The shared expert is left out; it is added once
/// after the reduce.
fn rank_partial(
    indices: &[u32],
    weights: &[f32],
    num_tokens: usize,
    rank: usize,
) -> (Vec<[f32; HIDDEN]>, usize) {
    let (start, end) = range_of(rank);
    let table = build_ep_routing_table(indices, weights, num_tokens, TOP_K, start, end);

    let mut out = vec![[0.0f32; HIDDEN]; num_tokens];
    for i in 0..table.local_count() {
        let t = table.local_token_indices[i] as usize;
        let e = table.local_expert_ids[i];
        assert!(
            (e as usize) >= start && (e as usize) < end,
            "rank {rank} asked to execute non-owned expert {e}"
        );
        let w = table.local_weights[i];
        let v = expert_out(e);
        for h in 0..HIDDEN {
            out[t][h] += w * v[h];
        }
    }
    (out, table.local_count())
}

/// 2026-09-25: Sum across ranks, then add the shared expert once.
fn reduce_and_finish(partials: Vec<Vec<[f32; HIDDEN]>>, num_tokens: usize) -> Vec<[f32; HIDDEN]> {
    let mut out = vec![[0.0f32; HIDDEN]; num_tokens];
    for p in &partials {
        for t in 0..num_tokens {
            for h in 0..HIDDEN {
                out[t][h] += p[t][h];
            }
        }
    }
    for t in 0..num_tokens {
        for h in 0..HIDDEN {
            out[t][h] += shared_out()[h];
        }
    }
    out
}

/// 2026-09-25: Run one routing case and assert the EP result equals the
/// single-process one.
fn assert_ep_matches_reference(indices: &[u32], weights: &[f32], case: &str) {
    let num_tokens = indices.len() / TOP_K;

    let mut partials = Vec::new();
    let mut executed = 0usize;
    for rank in 0..EP_WORLD {
        let (p, n) = rank_partial(indices, weights, num_tokens, rank);
        executed += n;
        partials.push(p);
    }

    assert_eq!(
        executed,
        num_tokens * TOP_K,
        "{case}: {} pairs executed across ranks, expected {}",
        executed,
        num_tokens * TOP_K
    );

    let got = reduce_and_finish(partials, num_tokens);
    let want = reference(indices, weights, num_tokens);
    assert_eq!(
        got, want,
        "{case}: EP=2 result diverged from single process"
    );
}

#[test]
fn mixed_local_and_remote_matches_single_process() {
    // 2026-09-25: Seen from rank 0: t0 split, t1 both on rank 1, t2 both on
    // rank 0, t3 split.
    let indices = vec![1u32, 5, 6, 7, 0, 3, 2, 4];
    let weights = vec![0.6f32, 0.4, 0.5, 0.5, 0.25, 0.75, 0.125, 0.875];
    assert_ep_matches_reference(&indices, &weights, "mixed");
}

#[test]
fn all_local_to_rank0_matches_single_process() {
    let indices = vec![0u32, 1, 2, 3];
    let weights = vec![0.5f32, 0.5, 0.25, 0.75];
    assert_ep_matches_reference(&indices, &weights, "all-rank0");
}

#[test]
fn all_local_to_rank1_matches_single_process() {
    let indices = vec![4u32, 5, 6, 7];
    let weights = vec![0.5f32, 0.5, 0.25, 0.75];
    assert_ep_matches_reference(&indices, &weights, "all-rank1");
}

/// 2026-09-25: Rank 0 owns none of the token's experts and contributes an
/// exact zero partial.
#[test]
fn empty_partial_contributes_exact_zero() {
    let indices = vec![4u32, 7];
    let weights = vec![0.5f32, 0.5];
    let (p0, n0) = rank_partial(&indices, &weights, 1, 0);
    assert_eq!(n0, 0, "rank 0 owns no expert in this case");
    assert_eq!(p0, vec![[0.0f32; HIDDEN]], "empty partial must be zero");
    assert_ep_matches_reference(&indices, &weights, "empty-rank0");
}

/// 2026-09-25: Ownership is disjoint and complete, and ids and weights are
/// preserved.
#[test]
fn ownership_is_disjoint_complete_and_value_preserving() {
    let indices = vec![1u32, 5, 6, 7, 0, 3, 2, 4];
    let weights = vec![0.6f32, 0.4, 0.5, 0.5, 0.25, 0.75, 0.125, 0.875];
    let num_tokens = indices.len() / TOP_K;

    let mut claimed: Vec<(u32, u32, f32)> = Vec::new();
    for rank in 0..EP_WORLD {
        let (start, end) = range_of(rank);
        let t = build_ep_routing_table(&indices, &weights, num_tokens, TOP_K, start, end);

        assert_eq!(t.total_count(), num_tokens * TOP_K, "rank {rank} total");
        // 2026-09-25: No expert in this rank's range is classified remote.
        for e in &t.remote_expert_ids {
            assert!(
                !((*e as usize) >= start && (*e as usize) < end),
                "rank {rank} classified an owned expert as remote"
            );
        }
        for i in 0..t.local_count() {
            claimed.push((
                t.local_token_indices[i],
                t.local_expert_ids[i],
                t.local_weights[i],
            ));
        }
    }

    assert_eq!(claimed.len(), num_tokens * TOP_K);
    let mut sorted = claimed.clone();
    sorted.sort_by_key(|(t, e, _)| (*t, *e));
    sorted.dedup_by_key(|(t, e, _)| (*t, *e));
    assert_eq!(sorted.len(), claimed.len(), "a pair was claimed twice");

    for (t, e, w) in claimed {
        let mut found = false;
        for k in 0..TOP_K {
            let f = t as usize * TOP_K + k;
            if indices[f] == e {
                assert_eq!(w, weights[f], "weight altered for (t{t}, e{e})");
                found = true;
            }
        }
        assert!(
            found,
            "expert id {e} not in token {t}'s top-k — id was rewritten"
        );
    }
}

/// 2026-09-26: Negative control: adding the shared expert to each rank's
/// partial counts it `world_size` times, which is why `ep_reduce_shared`
/// (`layers/moe/forward/ep_reduce.rs`) adds it once after the reduce.
#[test]
fn shared_expert_added_before_reduce_would_double_count() {
    let indices = vec![1u32, 5];
    let weights = vec![0.5f32, 0.5];

    let mut wrong = vec![[0.0f32; HIDDEN]; 1];
    for rank in 0..EP_WORLD {
        let (p, _) = rank_partial(&indices, &weights, 1, rank);
        for h in 0..HIDDEN {
            wrong[0][h] += p[0][h] + shared_out()[h];
        }
    }
    let right = reference(&indices, &weights, 1);
    for h in 0..HIDDEN {
        assert_eq!(
            wrong[0][h] - right[0][h],
            shared_out()[h] * (EP_WORLD - 1) as f32,
            "double-count control did not behave as predicted"
        );
    }
    assert_ne!(wrong, right, "the negative control must diverge");
}
