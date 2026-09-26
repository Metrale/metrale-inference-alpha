// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The manifest a weight peer publishes, and the address and rail
//! arithmetic its clients use (model-weights `weight_tier_rdma.rs` and
//! `weight_lora_rdma.rs`). Compiled on every platform and independent of
//! `serve` and `shard`.
//!
//! Owner: storage (weight peer).
//! Invariants: none beyond the types.

use serde::{Deserialize, Serialize};

/// 2026-09-25: Where one tensor of a staged model lies in its shard file. The
/// client reads `[shard_base + offset_in_shard, + len)`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WeightTensorRecord {
    /// 2026-09-25: The tensor's name in the safetensors header.
    pub name: String,
    /// 2026-09-25: The safetensors dtype string (`"BF16"`, `"F8_E4M3"`, ...),
    /// which the base-weight client maps with `WeightDtype::from_safetensors_str`.
    pub dtype: String,
    pub shape: Vec<u64>,
    /// 2026-09-25: File offset of the tensor's first byte: 8 + header size +
    /// `data_offsets[0]` (`metrale_core::safetensors::tensor_span`).
    pub offset_in_shard: u64,
    /// 2026-09-25: Byte length, `data_offsets[1] - data_offsets[0]`.
    pub len: u64,
    /// 2026-09-25: Index into [`WeightManifest::shard_files`].
    pub shard_index: u32,
    /// 2026-09-25: True for tensors of `extra_weights.safetensors`, which the
    /// client never drops with its expert-skip filter (`should_skip_tensor` in
    /// `weight_tier_rdma.rs`).
    pub extra: bool,
}

/// 2026-09-25: A staged model's manifest, sent as length-prefixed JSON after
/// the client's model request (`write_weight_manifest`). The per-shard
/// `(base, rkey)` pairs are sent later, in the verbs handshake.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WeightManifest {
    pub version: u32,
    /// 2026-09-25: The directory the peer staged, as a string.
    pub model_id: String,
    /// 2026-09-25: Shard file names in shard order; the verbs `layers` vector
    /// lists the shard MRs in the same order.
    pub shard_files: Vec<String>,
    /// 2026-09-25: Byte length of each shard file, in shard order.
    pub shard_lens: Vec<u64>,
    pub tensors: Vec<WeightTensorRecord>,
}

impl WeightManifest {
    pub const VERSION: u32 = 1;

    /// 2026-09-25: The number of shard files, which the client checks against
    /// the number of MRs the peer publishes per rail.
    pub fn num_shards(&self) -> usize {
        self.shard_files.len()
    }

    /// 2026-09-25: The sum of `shard_lens`, which `serve` reserves on the
    /// `CommitLedger` once per staged model.
    pub fn total_shard_bytes(&self) -> u64 {
        self.shard_lens.iter().sum()
    }
}

/// 2026-09-25: The rail for tensor `tensor_index`: `tensor_index % n_rails`,
/// with `n_rails` taken as at least 1. The client's read loop calls it.
pub fn rail_for_tensor(tensor_index: usize, n_rails: usize) -> usize {
    tensor_index % n_rails.max(1)
}

/// 2026-09-25: The peer address of a tensor's first byte: the shard MR's base
/// plus `offset_in_shard`, which already counts the size prefix and header.
pub fn tensor_remote_addr(shard_base: u64, offset_in_shard: u64) -> u64 {
    shard_base + offset_in_shard
}

#[cfg(test)]
#[path = "manifest_tests.rs"]
mod tests;
