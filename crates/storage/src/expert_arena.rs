// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `ExpertArena`: one pinned host allocation (`cuMemAllocHost`)
//! divided into `num_slabs` slabs of `slots_per_slab` expert-record slots.
//! `new` requires the allocation's device address to equal its host address,
//! so a record written into a slot is read by device code at `slot_dev_va`
//! without a copy. The expert tiers decide what fills a slot.
//!
//! Owner: metrale-storage experts.
//! Invariants:
//! - `dev_base` equals the allocation's host address.
//! - `record_stride` is a non-zero multiple of 4096.

use anyhow::{Context, Result, bail};
use std::ffi::c_void;

use crate::cuda_min::PinnedBuffer;

/// 2026-09-25: Slabs of slots, each slot one expert record of `record_stride`
/// bytes.
pub struct ExpertArena {
    pinned: PinnedBuffer,
    /// 2026-09-25: Device address of the arena base; `new` checks that it
    /// equals the host address.
    dev_base: u64,
    num_slabs: u32,
    slots_per_slab: u32,
    record_stride: usize,
}

impl ExpertArena {
    /// 2026-09-25: Allocate `num_slabs * slots_per_slab * record_stride` pinned
    /// bytes. Fails on a zero dimension, a stride that is not a multiple of
    /// 4096, a size overflow, or a device address that differs from the host
    /// address.
    pub fn new(num_slabs: u32, slots_per_slab: u32, record_stride: usize) -> Result<Self> {
        if num_slabs == 0 || slots_per_slab == 0 || record_stride == 0 {
            bail!("ExpertArena: zero geometry ({num_slabs},{slots_per_slab},{record_stride})");
        }
        if !record_stride.is_multiple_of(4096) {
            bail!("ExpertArena: record_stride {record_stride} must be a 4 KiB multiple (O_DIRECT)");
        }
        let total = (num_slabs as usize)
            .checked_mul(slots_per_slab as usize)
            .and_then(|v| v.checked_mul(record_stride))
            .context("ExpertArena: size overflow")?;
        let pinned = PinnedBuffer::new(total)?;
        let dev_base = pinned.device_ptr()?;
        let host_base = pinned.ptr as u64;
        if dev_base != host_base {
            bail!(
                "ExpertArena: pinned host VA {host_base:#x} != device VA {dev_base:#x} \
                 — host is not unified-addressing (UMA zero-copy unavailable)"
            );
        }
        Ok(Self {
            pinned,
            dev_base,
            num_slabs,
            slots_per_slab,
            record_stride,
        })
    }

    pub fn num_slabs(&self) -> u32 {
        self.num_slabs
    }
    pub fn slots_per_slab(&self) -> u32 {
        self.slots_per_slab
    }
    pub fn record_stride(&self) -> usize {
        self.record_stride
    }

    fn linear_slot(&self, slab: u32, slot: u32) -> Result<usize> {
        if slab >= self.num_slabs || slot >= self.slots_per_slab {
            bail!(
                "ExpertArena: slot ({slab},{slot}) out of range ({},{})",
                self.num_slabs,
                self.slots_per_slab
            );
        }
        Ok((slab as usize) * (self.slots_per_slab as usize) + (slot as usize))
    }

    /// 2026-09-25: Host pointer of a slot, where its record is written.
    pub fn slot_host_ptr(&self, slab: u32, slot: u32) -> Result<*mut u8> {
        let i = self.linear_slot(slab, slot)?;
        // 2026-09-25: SAFETY: `linear_slot` checked i < num_slabs *
        // slots_per_slab, so the offset is inside the allocation.
        Ok(unsafe { (self.pinned.ptr as *mut u8).add(i * self.record_stride) })
    }

    /// 2026-09-25: Device address of a slot.
    pub fn slot_dev_va(&self, slab: u32, slot: u32) -> Result<u64> {
        let i = self.linear_slot(slab, slot)?;
        Ok(self.dev_base + (i as u64) * (self.record_stride as u64))
    }

    /// 2026-09-25: Base of the pinned allocation; with `total_bytes` it
    /// describes the whole arena.
    pub fn base_ptr(&self) -> *mut c_void {
        self.pinned.ptr
    }
    pub fn total_bytes(&self) -> usize {
        self.pinned.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // 2026-09-25: `ExpertArena::new` allocates pinned CUDA memory, so the test
    // needs a GPU.
    #[test]
    #[ignore = "requires GPU"]
    fn arena_slots_are_strided_and_same_va() {
        let _ctx = crate::cuda_min::CudaCtx::new(0).unwrap();
        let stride = 8192;
        let arena = ExpertArena::new(2, 3, stride).unwrap();
        assert_eq!(arena.total_bytes(), 2 * 3 * stride);
        let a = arena.slot_dev_va(0, 0).unwrap();
        let b = arena.slot_dev_va(0, 1).unwrap();
        let c = arena.slot_dev_va(1, 0).unwrap();
        assert_eq!(b - a, stride as u64);
        assert_eq!(c - a, 3 * stride as u64);
        assert_eq!(
            arena.slot_host_ptr(1, 2).unwrap() as u64,
            arena.slot_dev_va(1, 2).unwrap()
        );
        assert!(arena.slot_dev_va(2, 0).is_err());
    }
}
