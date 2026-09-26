// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The slot lifecycle of [`LoraWeights`]: per-slot ref counts,
//! last-used ticks, the cache-slot view the victim search reads, and the
//! per-slot table refresh after a swap.
//!
//! Owner: model-layers (lora).
//! Invariants:
//! - [`LoraWeights::acquire_slot`] returns either -1, having changed nothing,
//!   or the index whose count it incremented.
//! - [`LoraWeights::release_slot`] never takes a count below 0.

use std::sync::atomic::Ordering;

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::types::{LoraLayerWeights, LoraModule, LoraWeights, SlotView};

impl LoraWeights {
    /// 2026-09-25: Resolve `slot` (`>= 0` that slot, `-1` the active one),
    /// increment its ref count and stamp it most recently used. Returns the
    /// resolved index, which the caller passes to [`Self::release_slot`] even
    /// if `active` changes meanwhile, or -1 when the index is out of range.
    pub fn acquire_slot(&self, slot: i32) -> i32 {
        let resolved = if slot >= 0 {
            slot as usize
        } else {
            self.active
        };
        match self.ref_counts.get(resolved) {
            Some(rc) => {
                rc.fetch_add(1, Ordering::AcqRel);
                if let Some(lu) = self.last_used.get(resolved) {
                    let t = self.lru_tick.fetch_add(1, Ordering::Relaxed) + 1;
                    lu.store(t, Ordering::Relaxed);
                }
                resolved as i32
            }
            None => -1,
        }
    }

    /// 2026-09-25: Stamp `slot` most recently used without taking a ref. A
    /// promote calls it on the slot it just filled, so a second promote before
    /// the first request acquires does not pick that idle slot while an older
    /// idle slot exists.
    pub fn touch_slot(&self, slot: usize) {
        if let Some(lu) = self.last_used.get(slot) {
            let t = self.lru_tick.fetch_add(1, Ordering::Relaxed) + 1;
            lu.store(t, Ordering::Relaxed);
        }
    }

    /// 2026-09-25: Last-used tick of pool `slot`; larger is more recent, and an
    /// out-of-range slot reads 0.
    pub fn slot_last_used(&self, slot: usize) -> u64 {
        self.last_used
            .get(slot)
            .map(|lu| lu.load(Ordering::Relaxed))
            .unwrap_or(0)
    }

    /// 2026-09-25: Rewrite `slot`'s entry in every `[max_loras]` a/b pointer
    /// table from `layers`, 0 where the new adapter does not adapt the module,
    /// and its scale-table entry from `scale`. Without it a swapped-in adapter
    /// with different module coverage would keep the old adapter's entries.
    /// Both swaps call it: the disk swap (`pack_store_into_slot`) and the
    /// model's peer swap. Only the `[slot]` entries are written.
    pub fn refresh_slot_tables(
        &self,
        slot: usize,
        layers: &[Option<LoraLayerWeights>],
        scale: f32,
        gpu: &dyn GpuBackend,
    ) -> Result<()> {
        for ((layer, module), (a_dev, b_dev)) in &self.tables {
            let pair = layers
                .get(*layer)
                .and_then(|o| o.as_ref())
                .and_then(|lw| match module {
                    LoraModule::QProj => lw.q_proj.as_ref(),
                    LoraModule::KProj => lw.k_proj.as_ref(),
                    LoraModule::VProj => lw.v_proj.as_ref(),
                    LoraModule::OProj => lw.o_proj.as_ref(),
                    LoraModule::GateProj => lw.gate_proj.as_ref(),
                    LoraModule::UpProj => lw.up_proj.as_ref(),
                    LoraModule::DownProj => lw.down_proj.as_ref(),
                    LoraModule::OutProj => lw.out_proj.as_ref(),
                });
            let (a_ptr, b_ptr) = pair.map(|p| (p.a.weight.0, p.b.weight.0)).unwrap_or((0, 0));
            gpu.copy_h2d(&a_ptr.to_le_bytes(), DevicePtr(a_dev.0 + (slot * 8) as u64))?;
            gpu.copy_h2d(&b_ptr.to_le_bytes(), DevicePtr(b_dev.0 + (slot * 8) as u64))?;
        }
        if self.scale_table.0 != 0 {
            gpu.copy_h2d(
                &scale.to_le_bytes(),
                DevicePtr(self.scale_table.0 + (slot * 4) as u64),
            )?;
        }
        Ok(())
    }

    /// 2026-09-25: `(slot_index, SlotView)` for each cache slot
    /// `[pinned, max_loras)`, the input of `select_victim_slot`.
    pub fn cache_slot_views(&self) -> Vec<(usize, SlotView)> {
        (self.pinned..self.max_loras)
            .map(|k| {
                let filled = self.slots.get(k).is_some_and(|s| !s.name.is_empty());
                (
                    k,
                    SlotView {
                        filled,
                        ref_count: self.slot_ref_count(k),
                        last_used: self.slot_last_used(k),
                    },
                )
            })
            .collect()
    }

    /// 2026-09-25: Release a ref taken by [`Self::acquire_slot`], given the
    /// index it returned. -1 is a no-op, and the count saturates at 0.
    pub fn release_slot(&self, resolved: i32) {
        if resolved < 0 {
            return;
        }
        if let Some(rc) = self.ref_counts.get(resolved as usize) {
            let _ = rc.fetch_update(Ordering::Release, Ordering::Acquire, |v| {
                Some(v.saturating_sub(1))
            });
        }
    }

    /// 2026-09-25: In-flight ref count of pool `slot`, which the swaps and
    /// rotation check before replacing or leaving a slot; an out-of-range slot
    /// reads 0.
    pub fn slot_ref_count(&self, slot: usize) -> usize {
        self.ref_counts
            .get(slot)
            .map(|rc| rc.load(Ordering::Acquire))
            .unwrap_or(0)
    }
}
