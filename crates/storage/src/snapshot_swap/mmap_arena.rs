// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `MmapSlotArena`, the `SlotArena` over the paging peer's mapped arena.
//!
//! Owner: storage, SSM snapshot tier.
//! Invariants:
//! - `read_slot` and `write_slot` touch memory only after checking the slot index and
//!   the buffer length; a mismatch is an error.

use anyhow::{Result, bail};

use super::SlotArena;

/// 2026-09-25: `SlotArena` over the peer's mapped arena, from a raw base pointer. The
/// peer copies between slots and its swap file; clients RDMA into and out of the same
/// slots. The arena registers nothing; the peer registers the mapping per connection.
pub struct MmapSlotArena {
    base: *mut u8,
    slot_bytes: usize,
    num_slots: usize,
}
unsafe impl Send for MmapSlotArena {}

impl MmapSlotArena {
    /// 2026-09-25: Wrap the mapping at `base`.
    ///
    /// # Safety
    /// `base` must be a valid, writable mapping of `>= num_slots*slot_bytes`
    /// bytes that outlives this arena.
    pub unsafe fn new(base: *mut u8, slot_bytes: usize, num_slots: usize) -> Self {
        Self {
            base,
            slot_bytes,
            num_slots,
        }
    }
    fn slot_ptr(&self, slot: usize) -> *mut u8 {
        // 2026-09-25: Both callers check `slot < num_slots` first.
        unsafe { self.base.add(slot * self.slot_bytes) }
    }
}

impl SlotArena for MmapSlotArena {
    fn slot_bytes(&self) -> usize {
        self.slot_bytes
    }
    fn num_slots(&self) -> usize {
        self.num_slots
    }
    fn read_slot(&self, slot: usize, out: &mut [u8]) -> Result<()> {
        if slot >= self.num_slots || out.len() != self.slot_bytes {
            bail!("read_slot({slot}) out of range / size mismatch");
        }
        unsafe {
            std::ptr::copy_nonoverlapping(self.slot_ptr(slot), out.as_mut_ptr(), self.slot_bytes)
        };
        Ok(())
    }
    fn write_slot(&mut self, slot: usize, bytes: &[u8]) -> Result<()> {
        if slot >= self.num_slots || bytes.len() != self.slot_bytes {
            bail!("write_slot({slot}) out of range / size mismatch");
        }
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.slot_ptr(slot), self.slot_bytes)
        };
        Ok(())
    }
}

#[cfg(test)]
#[path = "mmap_arena_tests.rs"]
mod tests;
