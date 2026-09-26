// SPDX-License-Identifier: MIT OR Apache-2.0
#![deny(warnings)]
#![deny(clippy::all)]

//! 2026-09-25: The generic tiered cache: a bounded hot arena over a cold record
//! store, written without CUDA or verbs calls.
//!
//! * [`SlotArena`]: the hot tier, `num_slots` fixed-size byte slots
//!   ([`VecSlotArena`] in host RAM, the cache peer's `MmapSlotArena`).
//! * [`SwapStore`]: the cold tier, fixed-size records ([`DirectSwapFile`] on
//!   disk, [`MemSwapStore`] in host RAM).
//! * [`Residency`]: the page table over both, from a `u64` key to one blob: an
//!   LRU of resident slots above an LRU of on-disk records, and read-pins. A
//!   put into a full arena spills the coldest unpinned resident to the store;
//!   with a disk cap, the coldest on-disk record is dropped and a later get of
//!   its key misses.
//!
//! Owner: storage (tier).
//! Invariants: none beyond the types.

mod aligned;
mod direct_swap;
mod mem;
pub mod pio;
mod residency;
mod traits;

pub mod entropy;
pub mod hash;

pub use direct_swap::DirectSwapFile;
pub use mem::{MemSwapStore, VecSlotArena};
pub use residency::Residency;
pub use traits::{SlotArena, SwapStats, SwapStore};
