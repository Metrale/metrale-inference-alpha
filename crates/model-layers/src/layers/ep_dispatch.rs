// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Partition of top-k (token, expert) assignments into this rank's
//! experts and the other ranks' experts, for EP token dispatch.
//!
//! Its callers are `MoeLayer::forward_ep_dispatch` (`moe/forward_ep.rs`),
//! which has no caller and serves through `forward`, and the
//! `glm53_ep_semantics` test. The served EP MoE is masked-local plus
//! all-reduce (`moe/forward.rs`).
//!
//! Owner: model-layers (MoE).
//! Invariants: none beyond the types.

/// 2026-09-25: Top-k assignments split into a local bucket (expert in this
/// rank's range) and a remote bucket (any other expert). Entry `i` of a bucket
/// is the triple (`*_token_indices[i]`, `*_expert_ids[i]`, `*_weights[i]`), in
/// the order the assignments were given.
#[derive(Debug)]
pub struct EpRoutingTable {
    pub local_token_indices: Vec<u32>,
    /// 2026-09-25: Global expert ids, not rank-relative.
    pub local_expert_ids: Vec<u32>,
    pub local_weights: Vec<f32>,
    pub remote_token_indices: Vec<u32>,
    /// 2026-09-25: Global expert ids, not rank-relative.
    pub remote_expert_ids: Vec<u32>,
    pub remote_weights: Vec<f32>,
}

impl EpRoutingTable {
    pub fn local_count(&self) -> usize {
        self.local_token_indices.len()
    }

    pub fn remote_count(&self) -> usize {
        self.remote_token_indices.len()
    }

    /// 2026-09-25: Equals `num_tokens * top_k` of the build call.
    pub fn total_count(&self) -> usize {
        self.local_count() + self.remote_count()
    }
}

/// 2026-09-25: Build the table from row-major `[num_tokens, top_k]` expert ids
/// and weights. An expert is local when it lies in
/// `local_expert_start..local_expert_end`.
///
/// # Panics
/// If `gate_indices.len()` or `gate_weights.len()` is not
/// `num_tokens * top_k`.
pub fn build_ep_routing_table(
    gate_indices: &[u32],
    gate_weights: &[f32],
    num_tokens: usize,
    top_k: usize,
    local_expert_start: usize,
    local_expert_end: usize,
) -> EpRoutingTable {
    let total = num_tokens * top_k;
    assert_eq!(gate_indices.len(), total, "gate_indices length mismatch");
    assert_eq!(gate_weights.len(), total, "gate_weights length mismatch");

    let mut local_token_indices = Vec::with_capacity(total);
    let mut local_expert_ids = Vec::with_capacity(total);
    let mut local_weights = Vec::with_capacity(total);
    let mut remote_token_indices = Vec::with_capacity(total);
    let mut remote_expert_ids = Vec::with_capacity(total);
    let mut remote_weights = Vec::with_capacity(total);

    for token_idx in 0..num_tokens {
        for k in 0..top_k {
            let flat_idx = token_idx * top_k + k;
            let expert_id = gate_indices[flat_idx];
            let weight = gate_weights[flat_idx];
            let eid = expert_id as usize;

            if eid >= local_expert_start && eid < local_expert_end {
                local_token_indices.push(token_idx as u32);
                local_expert_ids.push(expert_id);
                local_weights.push(weight);
            } else {
                remote_token_indices.push(token_idx as u32);
                remote_expert_ids.push(expert_id);
                remote_weights.push(weight);
            }
        }
    }

    EpRoutingTable {
        local_token_indices,
        local_expert_ids,
        local_weights,
        remote_token_indices,
        remote_expert_ids,
        remote_weights,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_local() {
        let indices = vec![3u32, 7, 100, 200];
        let weights = vec![0.6f32, 0.4, 0.55, 0.45];
        let table = build_ep_routing_table(&indices, &weights, 2, 2, 0, 256);

        assert_eq!(table.local_count(), 4);
        assert_eq!(table.remote_count(), 0);
        assert_eq!(table.total_count(), 4);
        assert_eq!(table.local_token_indices, vec![0, 0, 1, 1]);
        assert_eq!(table.local_expert_ids, vec![3, 7, 100, 200]);
    }

    #[test]
    fn test_all_remote() {
        let indices = vec![300u32, 400, 256, 511];
        let weights = vec![0.6f32, 0.4, 0.55, 0.45];
        let table = build_ep_routing_table(&indices, &weights, 2, 2, 0, 256);

        assert_eq!(table.local_count(), 0);
        assert_eq!(table.remote_count(), 4);
        assert_eq!(table.remote_token_indices, vec![0, 0, 1, 1]);
        assert_eq!(table.remote_expert_ids, vec![300, 400, 256, 511]);
    }

    #[test]
    fn test_mixed_routing() {
        let indices = vec![
            10u32, 300, // 2026-09-25: token 0: expert 10 local, 300 remote.
            255, 256, // 2026-09-25: token 1: expert 255 local, 256 remote.
            400, 500, // 2026-09-25: token 2: both remote.
        ];
        let weights = vec![0.7f32, 0.3, 0.5, 0.5, 0.6, 0.4];
        let table = build_ep_routing_table(&indices, &weights, 3, 2, 0, 256);

        assert_eq!(table.local_count(), 2);
        assert_eq!(table.remote_count(), 4);
        assert_eq!(table.local_token_indices, vec![0, 1]);
        assert_eq!(table.local_expert_ids, vec![10, 255]);
        assert_eq!(table.local_weights, vec![0.7, 0.5]);
        assert_eq!(table.remote_token_indices, vec![0, 1, 2, 2]);
        assert_eq!(table.remote_expert_ids, vec![300, 256, 400, 500]);
        assert_eq!(table.remote_weights, vec![0.3, 0.5, 0.6, 0.4]);
    }

    #[test]
    fn test_rank1_perspective() {
        let indices = vec![
            10u32, 300, // 2026-09-25: token 0: expert 10 remote, 300 local.
            255, 256, // 2026-09-25: token 1: expert 255 remote, 256 local.
        ];
        let weights = vec![0.7f32, 0.3, 0.5, 0.5];
        let table = build_ep_routing_table(&indices, &weights, 2, 2, 256, 512);

        assert_eq!(table.local_count(), 2);
        assert_eq!(table.remote_count(), 2);
        assert_eq!(table.local_token_indices, vec![0, 1]);
        assert_eq!(table.local_expert_ids, vec![300, 256]);
        assert_eq!(table.remote_token_indices, vec![0, 1]);
        assert_eq!(table.remote_expert_ids, vec![10, 255]);
    }

    #[test]
    fn test_single_token() {
        let indices = vec![5u32, 260, 100];
        let weights = vec![0.5f32, 0.3, 0.2];
        let table = build_ep_routing_table(&indices, &weights, 1, 3, 0, 256);

        assert_eq!(table.local_count(), 2);
        assert_eq!(table.remote_count(), 1);
        assert_eq!(table.total_count(), 3);
    }

    #[test]
    #[should_panic(expected = "gate_indices length mismatch")]
    fn index_length_mismatch_panics() {
        let indices = vec![1u32, 2, 3];
        let weights = vec![0.5f32; 4];
        build_ep_routing_table(&indices, &weights, 2, 2, 0, 256);
    }

    #[test]
    #[should_panic(expected = "gate_weights length mismatch")]
    fn weight_length_mismatch_panics() {
        let indices = vec![1u32, 2, 3, 4];
        let weights = vec![0.5f32; 3];
        build_ep_routing_table(&indices, &weights, 2, 2, 0, 256);
    }
}
