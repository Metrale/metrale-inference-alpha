// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Host staging and upload of the batched verify's R-row attention metadata
//! (positions, slots, causal lengths, block tables, seq slots) at the VMETA_* offsets.
//!
//! Owner: model-engine (speculative verify).
//! Invariants:
//! - Rows `r_total..r_up` (ghost rows) point only at the dummy KV block.

use anyhow::Result;
use metrale_model_layers::layer::AttnMetadataDev;

use super::super::super::types::TransformerModel;
use super::{VMETA_BT, VMETA_R, VMETA_SEQ_LENS, VMETA_SEQ_SLOT, VMETA_SLOTS};
use crate::traits::SequenceState;

impl TransformerModel {
    /// 2026-09-26: Fill and upload the metadata for `r_up` rows (active rows, then ghost
    /// rows) and return the `AttnMetadataDev` that points at it.
    pub(super) fn stage_verify_metadata(
        &self,
        seqs: &[&mut SequenceState],
        ks: &[usize],
        off: &[usize],
        bs: usize,
        r_total: usize,
        r_up: usize,
        stream: u64,
    ) -> Result<AttnMetadataDev> {
        // 2026-09-25: R-row attention metadata at `scratch + 32768`, at the
        // VMETA_* offsets. Consumers get absolute pointers through
        // `AttnMetadataDev`, and this step uploads its layout before dispatch,
        // so other paths using other layouts at the same base do not conflict.
        let meta_base = self.buffers.scratch().offset(32768);
        let max_blocks = self.max_blocks_per_seq;
        let mb = max_blocks as usize;

        let mut positions = [0u32; VMETA_R];
        let mut slots = [0i64; VMETA_R];
        let mut seq_lens = [0i32; VMETA_R];
        for (i, seq) in seqs.iter().enumerate() {
            for j in 0..ks[i] {
                let r = off[i] + j;
                let pos = seq.seq_len + j;
                positions[r] = pos as u32;
                let physical_block = seq.physical_block_for(pos / bs).unwrap_or(0);
                slots[r] = (physical_block as i64) * (bs as i64) + ((pos % bs) as i64);
                // 2026-09-25: Per-row causal clamp: row r attends through its own position.
                seq_lens[r] = (pos + 1) as i32;
            }
        }
        // 2026-09-25: Ghost rows (borrow only): position 0, a slot in the
        // dummy KV block, causal length 1, so no ghost row points at a live
        // sequence's KV.
        let dummy_kv = (self.dummy_kv_block as i64) * (bs as i64);
        for r in r_total..r_up {
            positions[r] = 0;
            slots[r] = dummy_kv;
            seq_lens[r] = 1;
        }
        // 2026-09-25: SAFETY: `positions` is `[0u32; VMETA_R]`, VMETA_R * 4
        // bytes, and `r_up <= VERIFY_ROW_CAP == VMETA_R` (see `r_up`), so
        // `r_up * 4` fits. The array is zero-initialised, and the fill loops
        // write rows `0..r_up` (`off` is the prefix sum of `ks`). `u32` is POD.
        let pos_bytes =
            unsafe { std::slice::from_raw_parts(positions.as_ptr() as *const u8, r_up * 4) };
        self.gpu.copy_h2d_async(pos_bytes, meta_base, stream)?;
        // 2026-09-25: SAFETY: `slots` is `[0i64; VMETA_R]` (VMETA_R * 8 bytes);
        // the same bound fits `r_up * 8`. Zero-initialised; `i64` is POD.
        let slot_bytes =
            unsafe { std::slice::from_raw_parts(slots.as_ptr() as *const u8, r_up * 8) };
        self.gpu
            .copy_h2d_async(slot_bytes, meta_base.offset(VMETA_SLOTS), stream)?;
        // 2026-09-25: SAFETY: `seq_lens` is `[0i32; VMETA_R]` (VMETA_R * 4
        // bytes); the same bound fits `r_up * 4`. Zero-initialised; `i32` is POD.
        let sl_bytes =
            unsafe { std::slice::from_raw_parts(seq_lens.as_ptr() as *const u8, r_up * 4) };
        self.gpu
            .copy_h2d_async(sl_bytes, meta_base.offset(VMETA_SEQ_LENS), stream)?;

        // 2026-09-25: Every row of sequence i gets sequence i's block table.
        // A ghost row has causal length 1, so only its entry 0 is read; it
        // points at the dummy KV block.
        let needed = r_up * mb;
        let mut bt_buf = vec![0i32; needed];
        for (i, seq) in seqs.iter().enumerate() {
            for j in 0..ks[i] {
                let row = off[i] + j;
                for (bi, &block) in seq.block_table.iter().enumerate().take(mb) {
                    bt_buf[row * mb + bi] = block as i32;
                }
            }
        }
        for row in r_total..r_up {
            bt_buf[row * mb] = self.dummy_kv_block as i32;
        }
        // 2026-09-25: SAFETY: `bt_buf` is `vec![0i32; needed]`, so its length
        // is `needed` and `needed * 4 == size_of_val(&bt_buf[..])`. The
        // zero-initialisation covers entries the fill loops skip when
        // `block_table.len() < mb`.
        let bt_bytes =
            unsafe { std::slice::from_raw_parts(bt_buf.as_ptr() as *const u8, needed * 4) };
        self.gpu
            .copy_h2d_async(bt_bytes, meta_base.offset(VMETA_BT), stream)?;

        // 2026-09-25: Every row takes `seqs[0].adapter_slot`.
        // `upload_seq_slot_uniform` returns `DevicePtr(0)` when no adapter is
        // loaded or the slot resolves to the active adapter.
        debug_assert!(
            r_up <= super::super::verify_e2::VERIFY_ROW_CAP,
            "verify seq_slot gap holds R ≤ VERIFY_ROW_CAP"
        );
        let seq_slot = self.upload_seq_slot_uniform(
            seqs[0].adapter_slot,
            r_up,
            meta_base.offset(VMETA_SEQ_SLOT),
            stream,
        )?;

        Ok(AttnMetadataDev {
            positions: meta_base,
            positions_h: meta_base,
            positions_w: meta_base,
            slot: meta_base.offset(VMETA_SLOTS),
            seq_len: meta_base.offset(VMETA_SEQ_LENS),
            block_table: meta_base.offset(VMETA_BT),
            max_blocks_per_seq: max_blocks,
            num_seqs: r_up as u32,
            seq_slot,
            moe_row_adapter: metrale_gpu_runtime::gpu::DevicePtr::NULL,
        })
    }
}
