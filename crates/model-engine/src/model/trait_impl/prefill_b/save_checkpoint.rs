// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: After a non-last prefill chunk, save an SSM snapshot at the chunk end
//! when that end is a prompt-tail or interval boundary, and insert it into the prefix
//! cache as an intermediate checkpoint that a later partial hit can restore from.
//!
//! Owner: model-engine prefill (SSM prefix cache).
//! Invariants: none beyond the types.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use anyhow::Result;
use metrale_cache::kv_cache::PagedKvCache;

use super::super::super::types::TransformerModel;
use crate::traits::SequenceState;

impl TransformerModel {
    pub(in crate::model) fn prefill_b_save_checkpoint(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        kv_cache: &mut PagedKvCache,
        chunk_start: usize,
        chunk_len: usize,
        stream: u64,
    ) -> Result<()> {
        if !self.ssm_snapshots.is_enabled() {
            return Ok(());
        }
        let bs = kv_cache.block_size();
        let end_token = chunk_start + chunk_len;
        let end_block = end_token / bs;
        // 2026-09-25: A chunk end at `tail` (the last block boundary below the prompt
        // end) or at `tail - bs` is a prompt-tail checkpoint: a warm next turn's
        // block-floored match lands at or one block below `tail` (`ssm_tail_boundary`).
        // `prefill_chunk_dispatch` ends a chunk at `tail - bs`.
        let tail = (tokens.len().saturating_sub(1) / bs) * bs;
        let is_prompt_tail = end_token == tail || (tail >= bs && end_token == tail - bs);
        // 2026-09-25: `--ssm-checkpoint-interval` filters chunk ends; it does not create
        // them. This runs only at a chunk end, so interval checkpoints are spaced by the
        // chunk size, at chunk ends whose block index is an interval multiple.
        let on_interval = self.ssm_checkpoint_interval > 0
            && end_block.is_multiple_of(self.ssm_checkpoint_interval);
        if end_block == 0 || !(is_prompt_tail || on_interval) {
            return Ok(());
        }
        // 2026-09-25: Skip the checkpoint when a block below `end_block` lies past
        // `seq.kv_valid_tokens`, the prefix whose K/V is known to be written (the same
        // cap as in `finalize_last`).
        if seq.kv_valid_tokens / bs < end_block {
            tracing::debug!(
                "Skip intermediate checkpoint at block {end_block}: \
                 kv_valid_tokens={} only covers {} complete blocks",
                seq.kv_valid_tokens,
                seq.kv_valid_tokens / bs,
            );
            return Ok(());
        }
        if std::env::var("METRALE_SSM_SAVE_DUMP").is_ok() {
            self.ssm_pool.debug_state_checksum(
                seq.slot_idx,
                self.gpu.as_ref(),
                stream,
                &format!("ckpt_save@{end_token}"),
            );
        }

        let snap_result = match self.ssm_snapshots.save(
            seq.slot_idx,
            seq.session_hash,
            self.seq_ssm_h_is_f16(seq),
            &self.ssm_pool,
            self.gpu.as_ref(),
            stream,
        ) {
            Ok(Some(id)) => Some(id),
            Ok(None) => {
                if self.ssm_snapshots.reclaim_from_cache(
                    self.prefix_cache.as_ref(),
                    kv_cache,
                    self.ssm_tier_store.as_deref(),
                    self.gpu.as_ref(),
                ) {
                    self.ssm_snapshots
                        .save(
                            seq.slot_idx,
                            seq.session_hash,
                            self.seq_ssm_h_is_f16(seq),
                            &self.ssm_pool,
                            self.gpu.as_ref(),
                            stream,
                        )
                        .ok()
                        .flatten()
                } else {
                    tracing::warn!(
                        "SSM snapshot pool exhausted and no evictable cached entries — \
                         dropping checkpoint for this chunk. Long-context prefix-cache \
                         hits will recompute SSM state. Consider raising \
                         --ssm-cache-slots."
                    );
                    None
                }
            }
            Err(e) => {
                tracing::warn!("SSM snapshot save error: {e}");
                None
            }
        };
        let Some(snap_id) = snap_result else {
            return Ok(());
        };
        // 2026-09-25: Per-sequence aux layer state (`collect_aux_states`) rides the
        // checkpoint, taken at the end of a completed pass. A model that carries aux
        // state refuses a snapshot without it on restore (`snap_agree::local_proposal`).
        let aux = self.collect_aux_states(seq, stream)?;
        if !aux.is_empty() {
            self.ssm_snapshots.set_aux(snap_id, aux);
        }

        let boundary_tokens = &tokens[..end_token];
        // 2026-09-25: No insert when the HSS window has slid (`hss_window_start() > 0`:
        // the front of the prefix has no physical blocks) or the block table is shorter
        // than `end_block`. The snapshot is reachable only through that insert, so it is
        // freed.
        let skip_boundary_insert = seq.hss_window_start() > 0 || end_block > seq.block_table.len();
        if skip_boundary_insert {
            self.ssm_snapshots.free(snap_id);
            return Ok(());
        }
        let boundary_blocks = &seq.block_table[..end_block];
        // 2026-09-25: No insert for a prefix with vision pads: the pad tokens are the same
        // for different images, so a later hit would restore another image's state.
        if self.tokens_have_vision_pad(boundary_tokens) {
            self.ssm_snapshots.free(snap_id);
            return Ok(());
        }
        let boundary_disk = if seq.disk_block_ids.len() >= end_block {
            &seq.disk_block_ids[..end_block]
        } else {
            &[][..]
        };
        // 2026-09-25: `matched_tokens = end_token` marks no block of this insert as
        // sequence-owned, so none takes the extra reference here; the last chunk's
        // insert takes those for the whole prompt (`PrefixCache::insert` doc).
        let acquired = self.prefix_cache.insert(
            boundary_tokens,
            boundary_blocks,
            boundary_disk,
            bs,
            end_token,
            seq.adapter_id,
        );
        super::super::super::block_mgmt::cache_acquires_refs(&acquired, kv_cache);
        if let Some(old) = self.prefix_cache.insert_intermediate_snapshot(
            boundary_tokens,
            boundary_blocks,
            boundary_disk,
            bs,
            snap_id,
            seq.session_hash,
            end_token,
            seq.adapter_id,
        ) {
            self.ssm_snapshots.free(old);
        }
        if is_prompt_tail {
            seq.tail_checkpoint_tokens = Some(end_token);
        }
        tracing::info!(
            "Intermediate SSM checkpoint saved at token {} (snapshot_id {}, block {})",
            end_token,
            snap_id,
            end_block,
        );
        Ok(())
    }
}
