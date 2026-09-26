// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

//! 2026-09-25: Prefill tails: the full-prompt cache-hit exit of `prefill_twophase`, and the
//! Marconi snapshot save plus prefix-cache insert that ends a prefill.
//!
//! Owner: model-engine.
//! Invariants:
//! - A prompt with vision pad tokens, or a sequence whose HSS window has slid, is never
//!   inserted into the prefix cache, and a snapshot saved for it is freed.

use anyhow::Result;
use metrale_cache::kv_cache::PagedKvCache;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::super::types::TransformerModel;
use crate::traits::SequenceState;
use metrale_model_layers::layers::ops;

impl TransformerModel {
    /// 2026-09-25: `prefill_twophase` exit when no prompt token needs processing: embed the
    /// last token, run the final norm and LM head on it, insert the prompt into the
    /// prefix cache, and return the decode logits pointer. No layer runs.
    pub(super) fn prefill_full_cache_hit(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        hidden: DevicePtr,
        h: u32,
        bs: usize,
        total_len: usize,
        stream: u64,
    ) -> Result<DevicePtr> {
        seq.tokens.extend_from_slice(tokens);
        seq.seq_len = total_len;
        let last_tok = tokens[total_len - 1];
        // 2026-09-25: SAFETY: 4 == `size_of::<u32>()` bytes over the single, fully
        // initialised `last_tok` local on the line above (an in-bounds copy out
        // of `tokens`); the slice never outlives that local.
        let last_tok_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(&last_tok as *const u32 as *const u8, 4) };
        let token_id_dev = self.buffers.scratch();
        self.gpu
            .copy_h2d_async(last_tok_bytes, token_id_dev, stream)?;
        if self.has_ngram_embedding() {
            // 2026-09-25: The n-gram hash for this token reads back over the tokens that
            // precede it in the same prompt.
            let lb = self.ngram_lookbehind();
            let cs = total_len.saturating_sub(lb + 1);
            self.embed_tokens_fused(&tokens[cs..total_len], 1, hidden, stream)?;
        } else {
            ops::batched_embed(
                self.gpu.as_ref(),
                self.batched_embed_kernel,
                token_id_dev,
                self.embed_tokens.weight,
                hidden,
                1,
                h,
                stream,
            )?;
        }
        self.scale_embeddings(hidden, 1usize, stream)?;
        let normed = self.buffers.norm_output();
        let eps = self.config.rms_norm_eps as f32;
        self.final_norm_apply(hidden, normed, 1, h, eps, stream)?;
        self.lm_head(normed, stream)?;
        // 2026-09-25: No layer ran, so the SSM state is unchanged and no snapshot is saved.
        if !self.tokens_have_vision_pad(tokens) && !self.hss_window_slid(seq) {
            let acquired = self.prefix_cache.insert(
                tokens,
                &seq.block_table,
                &seq.disk_block_ids,
                bs,
                seq.cached_prefix_tokens,
                seq.adapter_id,
            );
            super::super::block_mgmt::cache_acquires_refs(&acquired, &mut self.kv_cache.lock());
        }
        Ok(self.decode_logits_ptr())
    }

    /// 2026-09-25: `prefill_a`'s snapshot save and prefix-cache insert. It differs from
    /// [`Self::prefill_save_snapshot_and_insert`] in attaching no aux layer state to the
    /// snapshot.
    pub(super) fn prefill_save_snapshot_with_vision_gate(
        &self,
        tokens: &[u32],
        seq: &SequenceState,
        kv_cache: &mut PagedKvCache,
        bs: usize,
        stream: u64,
    ) {
        if self.ssm_snapshots.is_enabled() {
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
                    // 2026-09-25: Pool full: reclaim a slot from the prefix cache and retry once.
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
                        None
                    }
                }
                Err(_) => None,
            };
            if let Some(snap_id) = snap_result {
                // 2026-09-25: Order any later warm restore after this save's D2D (the
                // restore may run on a different stream).
                if let Err(e) = self.record_snapshot_save_dispatch(stream) {
                    tracing::warn!("prefill snapshot save: record snapshot event: {e}");
                }
                if self.tokens_have_vision_pad(tokens) || self.hss_window_slid(seq) {
                    // 2026-09-25: Vision prompt: the prefix cache sees token ids, not pixels,
                    // and image pad tokens are the same for different images, so neither the
                    // snapshot nor the blocks are admitted. HSS-slid: `block_table` no
                    // longer parallels the token stream (see `hss_window_slid`). The
                    // snapshot is freed because only the skipped insert would reference it.
                    self.ssm_snapshots.free(snap_id);
                } else {
                    let (displaced, acquired) = self.prefix_cache.insert_with_snapshot(
                        tokens,
                        &seq.block_table,
                        &seq.disk_block_ids,
                        bs,
                        snap_id,
                        seq.session_hash,
                        seq.cached_prefix_tokens,
                        seq.adapter_id,
                    );
                    super::super::block_mgmt::cache_acquires_refs(&acquired, kv_cache);
                    if let Some(old) = displaced {
                        self.ssm_snapshots.free(old);
                    }
                }
            } else if !self.tokens_have_vision_pad(tokens) && !self.hss_window_slid(seq) {
                let acquired = self.prefix_cache.insert(
                    tokens,
                    &seq.block_table,
                    &seq.disk_block_ids,
                    bs,
                    seq.cached_prefix_tokens,
                    seq.adapter_id,
                );
                super::super::block_mgmt::cache_acquires_refs(&acquired, kv_cache);
            }
        } else if !self.tokens_have_vision_pad(tokens) && !self.hss_window_slid(seq) {
            let acquired = self.prefix_cache.insert(
                tokens,
                &seq.block_table,
                &seq.disk_block_ids,
                bs,
                seq.cached_prefix_tokens,
                seq.adapter_id,
            );
            super::super::block_mgmt::cache_acquires_refs(&acquired, kv_cache);
        }
    }

    /// 2026-09-25: `prefill_twophase`'s snapshot save and prefix-cache insert. Save a Marconi
    /// SSM snapshot with the aux layer state (reclaiming a slot from the prefix cache once if
    /// the pool is full) and insert with it. With snapshots disabled or no snapshot saved,
    /// insert without one.
    pub(super) fn prefill_save_snapshot_and_insert(
        &self,
        tokens: &[u32],
        seq: &SequenceState,
        kv_cache: &mut PagedKvCache,
        bs: usize,
        stream: u64,
    ) {
        if self.ssm_snapshots.is_enabled() {
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
                    tracing::debug!("Snapshot pool full, reclaiming...");
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
                        tracing::debug!("Reclaim failed — no evictable snapshots");
                        None
                    }
                }
                Err(e) => {
                    tracing::warn!("SSM snapshot save error: {e}");
                    None
                }
            };
            if let Some(snap_id) = snap_result {
                // 2026-09-25: Order any later warm restore after this save's D2D (the
                // restore may run on a different stream).
                if let Err(e) = self.record_snapshot_save_dispatch(stream) {
                    tracing::warn!("prefill snapshot save [twophase]: record snapshot event: {e}");
                }
                if self.tokens_have_vision_pad(tokens) || self.hss_window_slid(seq) {
                    // 2026-09-25: Same gate as `prefill_save_snapshot_with_vision_gate`.
                    self.ssm_snapshots.free(snap_id);
                } else {
                    tracing::info!(
                        "Saved SSM snapshot {} for {} tokens ({} blocks) [twophase]",
                        snap_id,
                        tokens.len(),
                        seq.block_table.len(),
                    );
                    // 2026-09-25: Aux layer state rides the snapshot. A collection failure
                    // leaves the slot aux-less, and aux-carrying models decline such a slot
                    // on restore.
                    match self.collect_aux_states(seq, stream) {
                        Ok(aux) if !aux.is_empty() => {
                            self.ssm_snapshots.set_aux(snap_id, aux);
                        }
                        Ok(_) => {}
                        Err(e) => tracing::warn!("aux snapshot skipped: {e:#}"),
                    }
                    let (displaced, acquired) = self.prefix_cache.insert_with_snapshot(
                        tokens,
                        &seq.block_table,
                        &seq.disk_block_ids,
                        bs,
                        snap_id,
                        seq.session_hash,
                        seq.cached_prefix_tokens,
                        seq.adapter_id,
                    );
                    super::super::block_mgmt::cache_acquires_refs(&acquired, kv_cache);
                    if let Some(old) = displaced {
                        self.ssm_snapshots.free(old);
                    }
                }
            } else if !self.tokens_have_vision_pad(tokens) && !self.hss_window_slid(seq) {
                let acquired = self.prefix_cache.insert(
                    tokens,
                    &seq.block_table,
                    &seq.disk_block_ids,
                    bs,
                    seq.cached_prefix_tokens,
                    seq.adapter_id,
                );
                super::super::block_mgmt::cache_acquires_refs(&acquired, kv_cache);
            }
        } else if !self.tokens_have_vision_pad(tokens) && !self.hss_window_slid(seq) {
            let acquired = self.prefix_cache.insert(
                tokens,
                &seq.block_table,
                &seq.disk_block_ids,
                bs,
                seq.cached_prefix_tokens,
                seq.adapter_id,
            );
            super::super::block_mgmt::cache_acquires_refs(&acquired, kv_cache);
        }
    }
}
