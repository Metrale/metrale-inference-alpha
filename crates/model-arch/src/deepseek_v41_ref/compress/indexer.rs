// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

//! 2026-09-25: The indexer of DeepSeek-V4.1 compressed attention (the
//! reference's `Indexer.forward`: fp4 query heads against one shared fp4 key
//! per compressed position, rectified scores combined by `weights_proj`),
//! `select_candidate_blocks` (the block-level first stage of the top-k), the
//! shared slots, and `torch_cpu_topk_set`, the emulation of torch's CPU
//! `topk` tie order.
//!
//! Owner: model-arch, DeepSeek-V4.1 reference.
//! Invariants: none beyond the types.

use super::super::attn::apply_rotary;
use super::super::hc::rms_norm;
use super::super::moe::linear_bf16;
use super::super::to_bf16_rne;
use super::{FP4_BLOCK, fp4_quant_e8m0_inplace};

pub struct IndexerWeights<'a> {
    pub wq_b: &'a [f32],
    pub weights_proj: &'a [f32],
    // 2026-09-25: `wk` and `k_norm` are needed on kv sources only.
    pub wk: Option<&'a [f32]>,
    pub k_norm: Option<&'a [f32]>,
}

/// 2026-09-25: What source layers publish and later layers read: the latest
/// kv source's whole latent
/// cache `[max_groups][hd]` and index-key cache `[max_groups][index_hd]`, the
/// latest index selection `[queries][topk]` (offset past the window rows),
/// and the candidate mask `[queries][width]`.
#[derive(Default)]
pub struct SharedRuntime {
    pub compress_kv: Vec<f32>,
    pub index_k: Vec<f32>,
    pub topk_idxs: Vec<i32>,
    pub topk: usize,
    pub candidates: Vec<bool>,
    pub cand_width: usize,
}

pub struct IndexerCfg {
    pub n_heads: usize,
    pub index_hd: usize,
    pub rope_dim: usize,
    pub q_rank: usize,
    pub dim: usize,
    pub hd: usize,
    pub index_topk: usize,
    pub cand_topk_blocks: usize,
    pub cand_block: usize,
    pub eps: f32,
}

/// 2026-09-25: The index set torch's CPU `topk(k, largest=True)` returns for
/// `vals`, ties included, emulated as libstdc++'s introselect
/// (`__introselect`, `__move_median_to_first`, `__unguarded_partition`,
/// `__insertion_sort`) over (value, index) pairs with a descending
/// comparator, keeping the first `k` pairs. Among equal values (the
/// indexer's many -inf entries) the set is whatever that order leaves in
/// front. Only `k * 64 > n` is emulated (asserted). Pinned by
/// `topk_tie_order_matches_torch_cpu`.
pub fn torch_cpu_topk_set(vals: &[f32], k: usize) -> Vec<usize> {
    let n = vals.len();
    assert!(k <= n && k > 0);
    assert!(
        k * 64 > n,
        "torch takes the partial_sort branch for k*64 <= n; not emulated"
    );
    let mut q: Vec<(f32, usize)> = vals
        .iter()
        .copied()
        .enumerate()
        .map(|(i, v)| (v, i))
        .collect();
    // 2026-09-25: comp(x, y): x before y when x is NaN and y is not, or x.value > y.value.
    let comp = |x: &(f32, usize), y: &(f32, usize)| (x.0.is_nan() && !y.0.is_nan()) || x.0 > y.0;
    let nth = k - 1;
    let (mut first, mut last) = (0usize, n);
    let mut depth = 2 * (usize::BITS - n.leading_zeros()) as usize;
    while last - first > 3 {
        if depth == 0 {
            unreachable!("introselect depth limit reached");
        }
        depth -= 1;
        let mid = first + (last - first) / 2;
        // 2026-09-25: __move_median_to_first(first, first+1, mid, last-1)
        let (a, b, c) = (first + 1, mid, last - 1);
        let r = if comp(&q[a], &q[b]) {
            if comp(&q[b], &q[c]) {
                b
            } else if comp(&q[a], &q[c]) {
                c
            } else {
                a
            }
        } else if comp(&q[a], &q[c]) {
            a
        } else if comp(&q[b], &q[c]) {
            c
        } else {
            b
        };
        q.swap(first, r);
        // 2026-09-25: __unguarded_partition(first+1, last, pivot=first)
        let pivot = q[first];
        let (mut lo, mut hi) = (first + 1, last);
        let cut = loop {
            while comp(&q[lo], &pivot) {
                lo += 1;
            }
            hi -= 1;
            while comp(&pivot, &q[hi]) {
                hi -= 1;
            }
            if lo >= hi {
                break lo;
            }
            q.swap(lo, hi);
            lo += 1;
        };
        if cut <= nth {
            first = cut;
        } else {
            last = cut;
        }
    }
    // 2026-09-25: __insertion_sort(first, last)
    for i in (first + 1)..last {
        let val = q[i];
        if comp(&val, &q[first]) {
            for j in (first + 1..=i).rev() {
                q[j] = q[j - 1];
            }
            q[first] = val;
        } else {
            let mut j = i;
            while comp(&val, &q[j - 1]) {
                q[j] = q[j - 1];
                j -= 1;
            }
            q[j] = val;
        }
    }
    q[..k].iter().map(|&(_, i)| i).collect()
}

/// 2026-09-25: Candidate blocks: per query, score each `block`-wide block of
/// `logits[queries][width]` (unreachable positions at -inf) by its max, pin
/// the block holding position `compress_lens[q] - 1`, keep the top
/// `topk_blocks` blocks that are not all -inf. Returns the keep mask
/// `[queries][width]`.
pub fn select_candidate_blocks(
    logits: &[f32],
    queries: usize,
    width: usize,
    compress_lens: &[usize],
    topk_blocks: usize,
    block: usize,
) -> Vec<bool> {
    let nb = width.div_ceil(block);
    let mut keep = vec![false; queries * width];
    for q in 0..queries {
        let row = &logits[q * width..(q + 1) * width];
        let mut scores: Vec<f32> = (0..nb)
            .map(|b| {
                (b * block..((b + 1) * block).min(width))
                    .map(|i| row[i])
                    .fold(f32::NEG_INFINITY, f32::max)
            })
            .collect();
        let last = (compress_lens[q] as i64 - 1).div_euclid(block as i64);
        if last >= 0 && (last as usize) < nb {
            scores[last as usize] = f32::INFINITY;
        }
        for b in torch_cpu_topk_set(&scores, topk_blocks.min(nb)) {
            if scores[b] > f32::NEG_INFINITY {
                for i in b * block..((b + 1) * block).min(width) {
                    keep[q * width + i] = true;
                }
            }
        }
    }
    keep
}

/// 2026-09-25: The indexer forward. Returns `([queries][topk], topk)`
/// compressed-position indices, offset by `offset`, -1 where unreachable;
/// also publishes index keys when `latent` and `k_cache` are given (a kv source).
#[allow(clippy::too_many_arguments)]
pub fn indexer(
    x: &[f32],
    qr: &[f32],
    latent: Option<&[f32]>,
    seqlen: usize,
    start_pos: usize,
    offset: usize,
    ratio: usize,
    c: &IndexerCfg,
    w: &IndexerWeights,
    fc: &[(f32, f32)],
    k_cache: Option<&mut Vec<f32>>,
    shared: &mut SharedRuntime,
    is_candidate_source: bool,
    uses_candidates: bool,
) -> (Vec<i32>, usize) {
    let (nh, ihd, rd) = (c.n_heads, c.index_hd, c.rope_dim);
    let end_pos = start_pos + seqlen;

    if let (Some(lat), Some(cache)) = (latent, k_cache) {
        let groups = lat.len() / c.hd;
        let mut k = rms_norm(
            &linear_bf16(lat, w.wk.expect("wk"), groups, c.hd, ihd),
            w.k_norm.expect("k_norm"),
            groups,
            ihd,
            c.eps,
        );
        let pos: Vec<usize> = if start_pos == 0 {
            (0..groups).map(|g| g * ratio).collect()
        } else {
            vec![start_pos + 1 - ratio]
        };
        apply_rotary(&mut k, ihd, rd, &pos, fc, false);
        fp4_quant_e8m0_inplace(&mut k, FP4_BLOCK);
        let at = start_pos / ratio;
        cache[at * ihd..(at + groups) * ihd].copy_from_slice(&k);
        shared.index_k = cache.clone();
    }

    let mut q = linear_bf16(qr, w.wq_b, seqlen, c.q_rank, nh * ihd);
    let head_pos: Vec<usize> = (0..seqlen)
        .flat_map(|t| std::iter::repeat_n(start_pos + t, nh))
        .collect();
    apply_rotary(&mut q, ihd, rd, &head_pos, fc, false);
    fp4_quant_e8m0_inplace(&mut q, FP4_BLOCK);

    let width = end_pos / ratio;
    let index_k = &shared.index_k[..width * ihd];
    let wscale = (ihd as f32).powf(-0.5) * (nh as f32).powf(-0.5);
    let weights: Vec<f32> = linear_bf16(x, w.weights_proj, seqlen, c.dim, nh)
        .into_iter()
        .map(|v| to_bf16_rne(v * wscale))
        .collect();

    // 2026-09-25: index_score[q][t] = bf16(sum_h bf16(relu(bf16(q_h . k_t)) * weights[h]))
    let mut score = vec![0f32; seqlen * width];
    for t in 0..seqlen {
        for p in 0..width {
            let kr = &index_k[p * ihd..(p + 1) * ihd];
            let mut acc = 0f32;
            for h in 0..nh {
                let qv = &q[(t * nh + h) * ihd..(t * nh + h + 1) * ihd];
                let dot = to_bf16_rne(qv.iter().zip(kr).map(|(a, b)| a * b).sum::<f32>());
                acc += to_bf16_rne(dot.max(0.0) * weights[t * nh + h]);
            }
            score[t * width + p] = to_bf16_rne(acc);
        }
    }
    let compress_lens: Vec<usize> = if start_pos == 0 {
        (0..seqlen).map(|t| (t + 1) / ratio).collect()
    } else {
        vec![end_pos / ratio; seqlen]
    };
    if start_pos == 0 {
        for t in 0..seqlen {
            for p in compress_lens[t]..width {
                score[t * width + p] = f32::NEG_INFINITY;
            }
        }
    }
    if is_candidate_source {
        shared.candidates = select_candidate_blocks(
            &score,
            seqlen,
            width,
            &compress_lens,
            c.cand_topk_blocks,
            c.cand_block,
        );
        shared.cand_width = width;
    } else if uses_candidates {
        for i in 0..seqlen * width {
            if !shared.candidates[i] {
                score[i] = f32::NEG_INFINITY;
            }
        }
    }
    let topk = c.index_topk.min(end_pos / ratio);
    let mut out = Vec::with_capacity(seqlen * topk);
    for t in 0..seqlen {
        let row = &score[t * width..(t + 1) * width];
        let mut picked = torch_cpu_topk_set(row, topk);
        picked.sort_unstable();
        for i in picked {
            out.push(if i < compress_lens[t] {
                (i + offset) as i32
            } else {
                -1
            });
        }
    }
    (out, topk)
}
