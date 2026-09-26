// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: CPU reference for the GLM-5.3-Flash DSA kpool indexer and NoPE MLA.
//!
//! Owner: model-arch (GLM-5.3-Flash DSA reference).
//! Invariants:
//! - Nothing here runs on a GPU or reads a checkpoint; production code does not call it.
//! - Every index row `expand_selection` returns is written for its full `out_width`, with
//!   `INVALID` in every slot that holds no token.
//!
//! `examples/dsa_indexer_microtest.rs` checks it against goldens from `gen_dsa_indexer_golden.py`
//! and `gen_dsa_mla_golden.py`, which run the `transformers` `glm5_next` indexer and attention.
//!
//! * `k_norm` is a LayerNorm: it subtracts the mean and adds a bias
//!   (`indexer.k_norm.bias`). The DSA block's other norms (`q_a_layernorm`, `kv_a_layernorm`)
//!   have no bias.
//! * The pool softmax runs over the slot axis, separately for each `head_dim` channel.
//! * Pooling starts at the first valid token, not at slot 0: with left padding
//!   `[P, P, A, B, C, D]`, pool 0 is `[A, B, C, D]`.
//! * A pool is valid only if all `kpool` slots are valid. A trailing partial pool is not a pool;
//!   the tail append covers it, so a 7-token sequence has one pool.
//! * NoPE: `qk_rope_head_dim = 0`, so `kv_a_proj_with_mqa` emits `kv_lora_rank` values and there
//!   is no rope section.

/// 2026-09-25: Indexer and MLA geometry. No field has a default.
#[derive(Clone, Copy, Debug)]
pub struct DsaDims {
    pub hidden: usize,
    /// 2026-09-25: Indexer heads (`index_n_heads`), not the MLA head count.
    pub index_heads: usize,
    /// 2026-09-25: Indexer head dim (`index_head_dim`), not the MLA head dim.
    pub index_head_dim: usize,
    pub index_kpool: usize,
    pub index_topk: usize,
    pub always_select_tail: bool,
    pub q_lora_rank: usize,
    pub heads: usize,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    /// 2026-09-25: Zero on GLM-5.3-Flash (MODEL.toml); [`expand_kv`] asserts it is zero.
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
}

impl DsaDims {
    pub fn qk_head_dim(&self) -> usize {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }
    /// 2026-09-25: Pools selected per query: `index_topk / index_kpool`, capped by how many pools
    /// exist.
    pub fn select_k(&self, n_pools: usize) -> usize {
        (self.index_topk / self.index_kpool).min(n_pools)
    }
    /// 2026-09-25: Width of the emitted index row. The tail adds `kpool - 1` slots.
    pub fn out_width(&self) -> usize {
        self.index_topk
            + if self.always_select_tail {
                self.index_kpool - 1
            } else {
                0
            }
    }
    pub fn is_nope(&self) -> bool {
        self.qk_rope_head_dim == 0
    }
}

/// 2026-09-25: The invalid-index sentinel, the `-1` the golden generator writes.
pub const INVALID: i32 = -1;

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// 2026-09-25: LayerNorm over the trailing `d`: subtract the mean, divide by the standard
/// deviation, then `w * x + b`.
pub fn layer_norm(x: &[f32], w: &[f32], b: &[f32], d: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    for (row_in, row_out) in x.chunks_exact(d).zip(out.chunks_exact_mut(d)) {
        let mean = row_in.iter().sum::<f32>() / d as f32;
        let var = row_in.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for i in 0..d {
            row_out[i] = (row_in[i] - mean) * inv * w[i] + b[i];
        }
    }
    out
}

/// 2026-09-25: RMSNorm without bias, the contrast to [`layer_norm`] in the tests.
pub fn rms_norm(x: &[f32], w: &[f32], d: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    for (row_in, row_out) in x.chunks_exact(d).zip(out.chunks_exact_mut(d)) {
        let inv = 1.0 / (row_in.iter().map(|v| v * v).sum::<f32>() / d as f32 + eps).sqrt();
        for i in 0..d {
            row_out[i] = row_in[i] * inv * w[i];
        }
    }
    out
}

/// 2026-09-25: `y = x @ w^T` for `x: [m, k]`, `w: [n, k]` (torch `Linear` layout), no bias.
pub fn linear(x: &[f32], m: usize, k: usize, w: &[f32], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for i in 0..k {
                acc += x[row * k + i] * w[col * k + i];
            }
            out[row * n + col] = acc;
        }
    }
    out
}

/// 2026-09-25: The compressed k-pool candidates.
pub struct Pools {
    /// 2026-09-25: `[n_pools, index_head_dim]`: softmax-weighted average of the pool's keys.
    pub keys: Vec<f32>,
    /// 2026-09-25: `[n_pools, index_kpool]`: raw token index per slot, [`INVALID`] where the slot
    /// is not a valid token.
    pub indices: Vec<i32>,
    /// 2026-09-25: `[n_pools]`: 1 only when every slot is valid.
    pub valid: Vec<u8>,
    pub n_pools: usize,
}

/// 2026-09-25: Build the pools. `k`/`gate` are `[seq, index_head_dim]`, `valid` is `[seq]`. The
/// indexer golden records the output of `transformers`' `get_pooled_states` for comparison.
pub fn pool_states(
    k: &[f32],
    gate: &[f32],
    valid: &[u8],
    ape: &[f32],
    dims: DsaDims,
    seq: usize,
) -> Pools {
    let (d, kp) = (dims.index_head_dim, dims.index_kpool);
    let n_pools = seq.div_ceil(kp);
    // 2026-09-25: Pooling starts at the first valid token, so left padding is skipped.
    let first_key = valid.iter().position(|v| *v != 0).unwrap_or(seq) as i64;

    let mut keys = vec![0.0f32; n_pools * d];
    let mut indices = vec![INVALID; n_pools * kp];
    let mut pvalid = vec![0u8; n_pools];
    let mut logits = vec![0.0f32; kp];

    for p in 0..n_pools {
        let mut slot_valid = [false; 64];
        let mut slot_idx = [0usize; 64];
        let mut all = true;
        for s in 0..kp {
            let raw = first_key + (p * kp + s) as i64;
            let in_range = raw >= 0 && (raw as usize) < seq;
            let ok = in_range && valid[raw as usize] != 0;
            slot_valid[s] = ok;
            slot_idx[s] = if in_range { raw as usize } else { 0 };
            all &= ok;
            indices[p * kp + s] = if ok { raw as i32 } else { INVALID };
        }
        pvalid[p] = all as u8;

        // 2026-09-25: Softmax over the slot axis, independently per channel.
        for dd in 0..d {
            let mut mx = f32::NEG_INFINITY;
            for s in 0..kp {
                logits[s] = if slot_valid[s] {
                    gate[slot_idx[s] * d + dd] + ape[s * d + dd]
                } else {
                    f32::NEG_INFINITY
                };
                mx = mx.max(logits[s]);
            }
            let mut sum = 0.0f32;
            for s in 0..kp {
                let e = if logits[s] == f32::NEG_INFINITY {
                    0.0
                } else {
                    (logits[s] - mx).exp()
                };
                logits[s] = e;
                sum += e;
            }
            // 2026-09-25: A pool with no valid slot gets weight 0, so its key is 0, not NaN.
            let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
            let mut acc = 0.0f32;
            for s in 0..kp {
                if slot_valid[s] {
                    acc += logits[s] * inv * k[slot_idx[s] * d + dd];
                }
            }
            keys[p * d + dd] = acc;
        }
    }
    // 2026-09-25: Invalid pools are dropped from the axis. `n_pools`, and the `select_k` derived
    // from it, therefore depend on the padding, not only on the sequence length: a 7-token
    // sequence has one pool and a budget of one.
    let keep: Vec<usize> = (0..n_pools).filter(|p| pvalid[*p] != 0).collect();
    if keep.len() == n_pools {
        return Pools {
            keys,
            indices,
            valid: pvalid,
            n_pools,
        };
    }
    let mut ck = vec![0.0f32; keep.len() * d];
    let mut ci = vec![INVALID; keep.len() * kp];
    let mut cv = vec![0u8; keep.len()];
    for (j, p) in keep.iter().enumerate() {
        ck[j * d..(j + 1) * d].copy_from_slice(&keys[p * d..(p + 1) * d]);
        ci[j * kp..(j + 1) * kp].copy_from_slice(&indices[p * kp..(p + 1) * kp]);
        cv[j] = pvalid[*p];
    }
    Pools {
        keys: ck,
        indices: ci,
        valid: cv,
        n_pools: keep.len(),
    }
}

/// 2026-09-25: The original ids of the pools [`pool_states`] keeps. Computed from padding and
/// sequence length alone.
pub fn kept_pools(valid: &[u8], dims: DsaDims, seq: usize) -> Vec<i32> {
    let kp = dims.index_kpool;
    let n_pools = seq.div_ceil(kp);
    let first_key = valid.iter().position(|v| *v != 0).unwrap_or(seq) as i64;
    (0..n_pools)
        .filter(|p| {
            (0..kp).all(|s| {
                let raw = first_key + (p * kp + s) as i64;
                raw >= 0 && (raw as usize) < seq && valid[raw as usize] != 0
            })
        })
        .map(|p| p as i32)
        .collect()
}

/// 2026-09-25: Per-(query, pool) index score: `sum_h weights[h] * relu(scale * q_h . key)`.
///
/// `q` is `[q_rows, index_heads, index_head_dim]`; `weights` is `[q_rows, index_heads]` and must
/// already carry the `index_heads^-0.5` factor, as the golden generator applies it.
pub fn index_scores(
    q: &[f32],
    weights: &[f32],
    pools: &Pools,
    dims: DsaDims,
    q_rows: usize,
) -> Vec<f32> {
    let (h, d, p) = (dims.index_heads, dims.index_head_dim, pools.n_pools);
    let scale = (d as f32).powf(-0.5);
    let mut out = vec![0.0f32; q_rows * p];
    for r in 0..q_rows {
        for pp in 0..p {
            let mut acc = 0.0f32;
            for hh in 0..h {
                let mut dot = 0.0f32;
                for dd in 0..d {
                    dot += q[(r * h + hh) * d + dd] * pools.keys[pp * d + dd];
                }
                // 2026-09-25: ReLU after the scale. `relu(s * x) == s * relu(x)` only because the
                // scale is positive.
                acc += weights[r * h + hh] * (scale * dot).max(0.0);
            }
            out[r * p + pp] = acc;
        }
    }
    out
}

/// 2026-09-25: Which keys a query at `q_pos` may see: causal and not padding.
pub fn visible(valid_keys: &[u8], q_pos: usize, key_idx: usize) -> bool {
    key_idx <= q_pos && valid_keys[key_idx] != 0
}

/// 2026-09-25: Select up to `select_k` pools per query, ordered by higher score, then smaller
/// pool index, so ties resolve the same way on every run. Invalid candidates score `f32::MIN`.
pub fn topk_pools(
    scores: &[f32],
    valid_candidates: &[u8],
    n_pools: usize,
    q_rows: usize,
    select_k: usize,
) -> Vec<i32> {
    let mut out = vec![INVALID; q_rows * select_k];
    let mut buf: Vec<(f32, usize)> = Vec::with_capacity(n_pools);
    for r in 0..q_rows {
        buf.clear();
        for p in 0..n_pools {
            let s = if valid_candidates[r * n_pools + p] != 0 {
                scores[r * n_pools + p]
            } else {
                f32::MIN
            };
            buf.push((s, p));
        }
        buf.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
        for (j, (_, p)) in buf.iter().take(select_k).enumerate() {
            out[r * select_k + j] = *p as i32;
        }
    }
    out
}

/// 2026-09-25: Expand selected pools into raw token indices, append the visible tail, and pad with
/// [`INVALID`]. Every row is written for all `out_width` entries: the output starts filled with
/// [`INVALID`], and a padded query row stays that way.
#[allow(clippy::too_many_arguments)]
pub fn expand_selection(
    selected: &[i32],
    pools: &Pools,
    valid_candidates: &[u8],
    valid_keys: &[u8],
    q_positions: &[usize],
    q_mask: &[u8],
    dims: DsaDims,
    seq: usize,
    select_k: usize,
) -> Vec<i32> {
    let (kp, width) = (dims.index_kpool, dims.out_width());
    let q_rows = q_positions.len();
    let mut out = vec![INVALID; q_rows * width];

    let first_key = valid_keys.iter().position(|v| *v != 0).unwrap_or(seq) as i64;
    for r in 0..q_rows {
        let row = &mut out[r * width..(r + 1) * width];
        if q_mask[r] == 0 {
            continue;
        }
        let mut w = 0usize;
        for j in 0..select_k {
            let p = selected[r * select_k + j];
            let ok = p >= 0 && valid_candidates[r * pools.n_pools + p as usize] != 0;
            for s in 0..kp {
                row[w] = if ok {
                    pools.indices[p as usize * kp + s]
                } else {
                    INVALID
                };
                w += 1;
            }
        }
        if dims.always_select_tail {
            // 2026-09-25: The incomplete pool, as raw token indices.
            let vis_count = (0..seq)
                .filter(|k| visible(valid_keys, q_positions[r], *k))
                .count();
            let tail_count = vis_count % kp;
            let tail_start = first_key + vis_count as i64 - tail_count as i64;
            for t in 0..kp - 1 {
                let idx = tail_start + t as i64;
                let ok = t < tail_count
                    && idx >= 0
                    && (idx as usize) < seq
                    && visible(valid_keys, q_positions[r], idx as usize);
                row[w] = if ok { idx as i32 } else { INVALID };
                w += 1;
            }
        }
        debug_assert_eq!(
            w,
            width.min(select_k * kp + if dims.always_select_tail { kp - 1 } else { 0 })
        );
    }
    out
}

/// 2026-09-25: Turn an index row into the boolean visibility mask the attention reads. Duplicate
/// indices collapse, so a repeated token is attended once; out-of-range and [`INVALID`] entries
/// are dropped.
pub fn topk_to_mask(topk: &[i32], q_rows: usize, width: usize, kv_len: usize) -> Vec<u8> {
    let mut mask = vec![0u8; q_rows * kv_len];
    for r in 0..q_rows {
        for j in 0..width {
            let i = topk[r * width + j];
            if i >= 0 && (i as usize) < kv_len {
                mask[r * kv_len + i as usize] = 1;
            }
        }
    }
    mask
}

/// 2026-09-25: Public sigmoid; nothing in the DSA reference calls it.
pub fn sigmoid_f32(x: f32) -> f32 {
    sigmoid(x)
}

mod mla;
pub use mla::{expand_kv, mla_masked_attention};

#[cfg(test)]
mod tests;
