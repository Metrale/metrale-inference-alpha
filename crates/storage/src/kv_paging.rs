// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: KV blocks as a paging kind. `KvPagingBackend` keeps whole KV
//! blocks in a peer's paging arena (handshake kind `PagingKind::KV`); the
//! alternative is the one-sided `RdmaKvBackend`, whose client addresses a
//! fixed arena at `base + group_id × group_stride`.
//!
//! Owner: metrale-storage KV tier.
//! Invariants:
//! - `connect_kv_peer_backend` returns `RdmaKvBackend` when `METRALE_KV_PAGING`
//!   is unset or `0`, `KvPagingBackend` when it is `1`, and an error for any
//!   other value (`ns::kv_paging_selected`).
//! - A paging record is one KV block, `GroupLayout::block_bytes()`, keyed by
//!   `ns::wire_key` of the block's K-head-0 group id.

pub mod ns;

#[cfg(all(feature = "cuda", metrale_rdma_verbs))]
mod backend;
#[cfg(all(feature = "cuda", metrale_rdma_verbs))]
mod connect;

#[cfg(all(feature = "cuda", metrale_rdma_verbs))]
pub use backend::{KvPagingBackend, KvPagingConnect};
#[cfg(all(feature = "cuda", metrale_rdma_verbs))]
pub use connect::connect_kv_peer_backend;

/// 2026-09-25: The error a KV paging GET miss becomes. The `StorageBackend`
/// reads have no miss outcome, so a block the peer no longer holds fails the
/// read.
pub fn kv_miss_error(layer: u32, block: u32) -> anyhow::Error {
    anyhow::anyhow!(
        "kv-paging: block (layer {layer}, disk block {block}) is not on the peer — an \
         evicted KV block is unrecoverable (silent KV loss would corrupt long-context \
         output). Run the peer with --swap-cap-gb-kv 0 (unbounded KV disk) and size \
         --max-blade-gb / METRALE_KV_PAGING_ARENA_GB for the working set"
    )
}

#[cfg(test)]
mod isolation_tests;
