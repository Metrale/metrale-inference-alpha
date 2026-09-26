// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The token overlay's `[max_loras]` device tables of pointers
//! and counts, one cell per adapter slot, allocated once when the set is
//! built. The kernels pick a row's cell by its slot (`seq_slot`, or `active`)
//! and skip a slot with a null `slot_map` or a zero count
//! (`token_overlay.cu`).
//!
//! Owner: model-layers (lora).
//! Invariants:
//! - `from_slots` returns a set only when every overlay has the same `vocab`.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::overlay_build::EmbedOverlay;

/// 2026-09-25: The overlay tables for the whole adapter pool, built from the
/// per-slot [`EmbedOverlay`]s. Model-engine keeps it only when some slot has an
/// overlay (`any_active`).
pub struct TokenOverlaySet {
    /// 2026-09-25: The per-slot overlays, kept so the buffers the tables point
    /// into stay allocated.
    pub overlays: Vec<Option<EmbedOverlay>>,
    /// 2026-09-25: `u64[max_loras]` of `i32[vocab]` `slot_map` pointers, 0 for
    /// a slot without an overlay.
    pub embed_slot_map_table: DevicePtr,
    /// 2026-09-25: `u64[max_loras]` of embed row pointers (`[n, h]` BF16).
    pub embed_rows_table: DevicePtr,
    /// 2026-09-25: `u32[max_loras]` embed row counts; the embed kernel skips a
    /// `slot_map` entry not below it. `n_override_table` is the lm_head count.
    pub embed_n_table: DevicePtr,
    /// 2026-09-25: The `slot_map` length of every overlay (0 when there is
    /// none); the kernels skip token ids `>= vocab`.
    pub vocab: u32,
    /// 2026-09-25: `u64[max_loras]` of lm_head row pointers.
    pub lmhead_rows_table: DevicePtr,
    /// 2026-09-25: `u64[max_loras]` of lm_head id pointers (`u32[n]`).
    pub lmhead_ids_table: DevicePtr,
    /// 2026-09-25: `u32[max_loras]` lm_head row counts; 0 skips the slot.
    pub n_override_table: DevicePtr,
    /// 2026-09-25: The largest lm_head row count, the lm_head kernel's
    /// `grid.y` (`token_overlay.cu`).
    pub max_n_override: u32,
}

fn mk_u64(gpu: &dyn GpuBackend, tab: &[u64]) -> Result<DevicePtr> {
    let bytes: Vec<u8> = tab.iter().flat_map(|p| p.to_le_bytes()).collect();
    let d = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(&bytes, d)?;
    Ok(d)
}

fn mk_u32(gpu: &dyn GpuBackend, tab: &[u32]) -> Result<DevicePtr> {
    let bytes: Vec<u8> = tab.iter().flat_map(|p| p.to_le_bytes()).collect();
    let d = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(&bytes, d)?;
    Ok(d)
}

impl TokenOverlaySet {
    /// 2026-09-25: Build the tables. A slot's lm_head cells are its own lm_head
    /// overlay when it has one; else, when `tied`, its embed rows and ids;
    /// else empty (count 0). An error when two overlays were built against
    /// different vocab sizes.
    pub fn from_slots(
        gpu: &dyn GpuBackend,
        overlays: Vec<Option<EmbedOverlay>>,
        max_loras: usize,
        tied: bool,
    ) -> Result<Self> {
        let mut slot_map_tab = vec![0u64; max_loras];
        let mut embed_rows_tab = vec![0u64; max_loras];
        let mut embed_n_tab = vec![0u32; max_loras];
        let mut lmhead_rows_tab = vec![0u64; max_loras];
        let mut lmhead_ids_tab = vec![0u64; max_loras];
        let mut n_override_tab = vec![0u32; max_loras];
        let mut max_n_override = 0u32;
        let mut vocab = 0u32;

        for (k, ov) in overlays.iter().enumerate() {
            let Some(ov) = ov else { continue };
            slot_map_tab[k] = ov.slot_map.0;
            embed_rows_tab[k] = ov.rows.0;
            embed_n_tab[k] = ov.n_override;
            // 2026-09-25: The kernels take one `vocab` bound for all slots.
            anyhow::ensure!(
                vocab == 0 || vocab == ov.vocab,
                "token-overlay: slot {k} was built against vocab {} but an \
                 earlier slot against {vocab}; refusing mixed-vocab overlay tables",
                ov.vocab
            );
            vocab = ov.vocab;
            let (rows, ids, n) = match (&ov.lmhead, tied) {
                (Some(lm), _) => (lm.rows.0, lm.ids_dev.0, lm.n_override),
                (None, true) => (ov.rows.0, ov.ids_dev.0, ov.n_override),
                (None, false) => (0, 0, 0),
            };
            lmhead_rows_tab[k] = rows;
            lmhead_ids_tab[k] = ids;
            n_override_tab[k] = n;
            max_n_override = max_n_override.max(n);
        }

        Ok(Self {
            embed_slot_map_table: mk_u64(gpu, &slot_map_tab)?,
            embed_rows_table: mk_u64(gpu, &embed_rows_tab)?,
            embed_n_table: mk_u32(gpu, &embed_n_tab)?,
            vocab,
            lmhead_rows_table: mk_u64(gpu, &lmhead_rows_tab)?,
            lmhead_ids_table: mk_u64(gpu, &lmhead_ids_tab)?,
            n_override_table: mk_u32(gpu, &n_override_tab)?,
            max_n_override,
            overlays,
        })
    }

    /// 2026-09-25: True when at least one slot has an overlay.
    pub fn any_active(&self) -> bool {
        self.overlays.iter().any(|o| o.is_some())
    }
}

#[cfg(test)]
#[path = "overlay_tables_tests.rs"]
mod tests;
