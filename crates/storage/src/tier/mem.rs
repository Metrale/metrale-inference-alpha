// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Host-RAM tiers: [`VecSlotArena`] (hot) and [`MemSwapStore`] (cold).
//!
//! Owner: storage (tier).
//! Invariants: none beyond the types.

use std::collections::HashMap;

use anyhow::{Result, bail};

use crate::tier::traits::{SlotArena, SwapStore};

/// 2026-09-25: [`SlotArena`] over one `Vec<u8>` of `slot_bytes * num_slots`
/// bytes, allocated and zeroed in `new`. The hot tier of model-engine's
/// host-RAM unified SSM stores.
pub struct VecSlotArena {
    buf: Vec<u8>,
    slot_bytes: usize,
    n: usize,
}

impl VecSlotArena {
    pub fn new(slot_bytes: usize, num_slots: usize) -> Self {
        Self {
            buf: vec![0u8; slot_bytes * num_slots],
            slot_bytes,
            n: num_slots,
        }
    }
}

impl SlotArena for VecSlotArena {
    fn slot_bytes(&self) -> usize {
        self.slot_bytes
    }
    fn num_slots(&self) -> usize {
        self.n
    }
    fn read_slot(&self, slot: usize, out: &mut [u8]) -> Result<()> {
        if slot >= self.n || out.len() != self.slot_bytes {
            bail!("VecSlotArena::read_slot({slot}) out of range / size mismatch");
        }
        let o = slot * self.slot_bytes;
        out.copy_from_slice(&self.buf[o..o + self.slot_bytes]);
        Ok(())
    }
    fn write_slot(&mut self, slot: usize, bytes: &[u8]) -> Result<()> {
        if slot >= self.n || bytes.len() != self.slot_bytes {
            bail!("VecSlotArena::write_slot({slot}) out of range / size mismatch");
        }
        let o = slot * self.slot_bytes;
        self.buf[o..o + self.slot_bytes].copy_from_slice(bytes);
        Ok(())
    }
}

/// 2026-09-25: [`SwapStore`] over a `HashMap` of heap records, with no bound of
/// its own. Model-engine's unified SSM store falls back to it when no swap
/// file can be used.
pub struct MemSwapStore {
    recs: HashMap<usize, Vec<u8>>,
    record_bytes: usize,
}

impl MemSwapStore {
    pub fn new(record_bytes: usize) -> Self {
        Self {
            recs: HashMap::new(),
            record_bytes,
        }
    }
}

impl SwapStore for MemSwapStore {
    fn record_bytes(&self) -> usize {
        self.record_bytes
    }
    fn write_record(&mut self, disk_slot: usize, bytes: &[u8]) -> Result<()> {
        if bytes.len() != self.record_bytes {
            bail!(
                "MemSwapStore::write_record: {} bytes, expected {}",
                bytes.len(),
                self.record_bytes
            );
        }
        self.recs.insert(disk_slot, bytes.to_vec());
        Ok(())
    }
    fn read_record(&self, disk_slot: usize, out: &mut [u8]) -> Result<()> {
        match self.recs.get(&disk_slot) {
            Some(v) => {
                out.copy_from_slice(v);
                Ok(())
            }
            None => bail!("MemSwapStore: no record {disk_slot}"),
        }
    }
    fn discard_record(&mut self, disk_slot: usize) {
        self.recs.remove(&disk_slot);
    }
}
