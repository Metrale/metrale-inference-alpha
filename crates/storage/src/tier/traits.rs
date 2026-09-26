// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The two tier seams, [`SlotArena`] (hot) and [`SwapStore`] (cold),
//! and [`SwapStats`], the counters [`Residency`](crate::tier::Residency) keeps.
//!
//! Owner: storage (tier).
//! Invariants: none beyond the types.

use anyhow::Result;

/// 2026-09-25: The hot tier: `num_slots` slots of `slot_bytes` each. Implemented
/// by [`VecSlotArena`](crate::tier::VecSlotArena) (host RAM), the cache peer's
/// `MmapSlotArena` and model-engine's `TransportSlotArena`.
pub trait SlotArena: Send {
    fn slot_bytes(&self) -> usize;
    fn num_slots(&self) -> usize;
    /// 2026-09-25: Copy slot `slot` into `out`, whose length must be
    /// `slot_bytes()`.
    fn read_slot(&self, slot: usize, out: &mut [u8]) -> Result<()>;
    /// 2026-09-25: Copy `bytes`, whose length must be `slot_bytes()`, into slot
    /// `slot`.
    fn write_slot(&mut self, slot: usize, bytes: &[u8]) -> Result<()>;
}

/// 2026-09-25: The cold tier: records of `record_bytes` each, addressed by a
/// `disk_slot` index that [`Residency`](crate::tier::Residency) allocates and
/// reuses. Implemented by [`DirectSwapFile`](crate::tier::DirectSwapFile) (a
/// file) and [`MemSwapStore`](crate::tier::MemSwapStore) (host RAM).
pub trait SwapStore: Send {
    fn record_bytes(&self) -> usize;
    fn write_record(&mut self, disk_slot: usize, bytes: &[u8]) -> Result<()>;
    fn read_record(&self, disk_slot: usize, out: &mut [u8]) -> Result<()>;
    /// 2026-09-25: Called when `disk_slot` is freed. The default does nothing;
    /// `Residency` reuses the index from its free list either way.
    fn discard_record(&mut self, _disk_slot: usize) {}
}

// 2026-09-25: Boxes implement the traits, so a `Residency<Box<dyn SlotArena>,
// Box<dyn SwapStore>>` takes an arena and a store chosen at run time
// (model-engine `UnifiedSnapshotStore`).
impl<T: SlotArena + ?Sized> SlotArena for Box<T> {
    fn slot_bytes(&self) -> usize {
        (**self).slot_bytes()
    }
    fn num_slots(&self) -> usize {
        (**self).num_slots()
    }
    fn read_slot(&self, slot: usize, out: &mut [u8]) -> Result<()> {
        (**self).read_slot(slot, out)
    }
    fn write_slot(&mut self, slot: usize, bytes: &[u8]) -> Result<()> {
        (**self).write_slot(slot, bytes)
    }
}

impl<T: SwapStore + ?Sized> SwapStore for Box<T> {
    fn record_bytes(&self) -> usize {
        (**self).record_bytes()
    }
    fn write_record(&mut self, disk_slot: usize, bytes: &[u8]) -> Result<()> {
        (**self).write_record(disk_slot, bytes)
    }
    fn read_record(&self, disk_slot: usize, out: &mut [u8]) -> Result<()> {
        (**self).read_record(disk_slot, out)
    }
    fn discard_record(&mut self, disk_slot: usize) {
        (**self).discard_record(disk_slot)
    }
}

#[derive(Default, Debug, Clone)]
pub struct SwapStats {
    pub puts: u64,
    pub gets: u64,
    pub get_miss: u64,
    pub spills_to_disk: u64,
    pub faults_from_disk: u64,
    pub resident_hits: u64,
    /// 2026-09-25: On-disk records dropped by the disk cap; a later get of a
    /// dropped key is a miss.
    pub disk_evictions: u64,
}
