// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Paging core of the SSM snapshot tier: the peer's TCP control protocol
//! and paging loops, the client side of that protocol (`wire`), and `MmapSlotArena`
//! (`mmap_arena`). Re-exports the generic `tier` types it builds on.
//!
//! Owner: storage, SSM snapshot tier.
//! Invariants: none beyond the types.
//!
//! The peer runs a `Residency` over its RDMA-registered arena and an NVMe swap file,
//! and owns residency, so every client of one arena shares its cache. The paging
//! loops register no memory: blobs move between arena and swap under the rkeys each
//! connection registered in its handshake.

#![allow(dead_code)]

/// 2026-09-25: The generic paging types from `crate::tier`.
pub use crate::tier::{
    DirectSwapFile, MemSwapStore, Residency, SlotArena, SwapStats, SwapStore, VecSlotArena,
};

mod mmap_arena;
mod wire;

pub use mmap_arena::MmapSlotArena;
pub use wire::*;
