// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `Model::prefill_twophase` for `TransformerModel`: embed the whole prompt,
//! then run each SSM layer as chunked projections, one GDN launch over the uncached range
//! and chunked post-processing, while attention layers prefill the range in one call.
//!
//! Owner: model-engine.
//! Invariants: none beyond the types.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, bail};
use metrale_cache::kv_cache::PagedKvCache;
use metrale_config::{LayerType, ModelConfig};
use metrale_gpu_runtime::buffers::BufferArena;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle, KernelHandle};

use super::super::block_mgmt::{
    apply_evicted_blocks, ensure_blocks_through_decode, ensure_blocks_through_prefill,
    extract_layer_refs, reuse_prefix_match_disk_ids,
};
use super::super::ssm_pool::SsmStatePool;
use super::super::ssm_snapshot::SsmSnapshotPool;
use super::super::types::{PinnedMetaStaging, TransformerModel};
use crate::traits::ModelForward;
use crate::traits::{ChunkedPrefillPageMetadata, Model, SequenceState};
use metrale_model_layers::layer::{
    AttnMetadataDev, ForwardContext, GdnPrefillBuffers, LayerState, SsmLayerState, TransformerLayer,
};
use metrale_model_layers::layers::ops;
use metrale_model_layers::speculative::DraftProposer;
use metrale_model_layers::weight_map::{DenseWeight, MtpWeights, QuantizedWeight};

mod marconi;
mod vision_rows;

impl TransformerModel {
    pub(super) fn prefill_twophase_dispatch(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        chunk_size: usize,
        stream: u64,
    ) -> Result<DevicePtr> {
        let total_len = tokens.len();
        if total_len == 0 {
            return Ok(DevicePtr::NULL);
        }

        // 2026-09-25: `hidden_states` holds `max_batch_tokens` rows, so a longer prompt
        // goes through `prefill_chunk` in `chunk_size` pieces instead.
        let arena_cap = self.buffers.max_batch_tokens();
        if total_len > arena_cap {
            tracing::info!(
                "Chunked SSM prefill: {total_len} tokens in {} chunks of {chunk_size} \
                 (arena_cap={arena_cap})",
                total_len.div_ceil(chunk_size),
            );
            let mut offset = 0;
            while offset < total_len {
                let remaining = total_len - offset;
                let chunk_len = remaining.min(chunk_size);
                let is_last = offset + chunk_len >= total_len;
                let logits = self.prefill_chunk(tokens, seq, offset, chunk_len, is_last, stream)?;
                offset += chunk_len;
                if is_last {
                    return Ok(logits);
                }
            }
            return Ok(DevicePtr::NULL);
        }

        // 2026-09-25: The GDN prefill buffers are null when the model has no GDN
        // linear-attention layers (`build_gdn_prefill_buffers`).
        if self.gdn_buf_qkv.is_null() {
            return self.prefill_chunk(tokens, seq, 0, total_len, true, stream);
        }

        if total_len > self.gdn_buf_max_len {
            tracing::info!(
                "prefill_twophase: total_len ({total_len}) > GDN buffer max ({}) \
                 falling back to chunked prefill",
                self.gdn_buf_max_len,
            );
            return self.prefill_chunk(tokens, seq, 0, total_len, true, stream);
        }

        // 2026-09-25: A multi-rank world (EP or TP, see `multi_rank_protocol_active`)
        // runs on the backend's default stream; otherwise on the caller's stream.
        let stream = if self.multi_rank_protocol_active() {
            self.gpu.default_stream()
        } else {
            stream
        };
        let h = self.config.hidden_size;
        let _bf16 = 2usize;
        let fp32 = 2usize;
        let hidden = self.buffers.hidden_states();
        let residual = self.buffers.residual();

        if self.comm.is_some() {
            self.buffers.zero_all(self.gpu.as_ref(), stream)?;
        } else {
            self.buffers.zero_all(self.gpu.as_ref(), stream)?;
        }

        let mut kv_cache = self.kv_cache.lock();

        {
            // 2026-09-25: SAFETY: `total_len` is `tokens.len()` (bound at fn entry and never
            // reassigned), so the byte length is exactly
            // `tokens.len() * size_of::<u32>()` over a live `&[u32]`.
            let token_ids_bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(tokens.as_ptr() as *const u8, total_len * 4) };
            let token_ids_dev = self.buffers.scratch();
            self.gpu
                .copy_h2d_async(token_ids_bytes, token_ids_dev, stream)?;
            if self.has_ngram_embedding() {
                // 2026-09-25: The prompt starts at position 0, so it has no earlier
                // n-gram context to pass.
                self.embed_tokens_fused(tokens, total_len, hidden, stream)?;
            } else {
                ops::batched_embed(
                    self.gpu.as_ref(),
                    self.batched_embed_kernel,
                    token_ids_dev,
                    self.embed_tokens.weight,
                    hidden,
                    total_len as u32,
                    h as u32,
                    stream,
                )?;
            }
            self.scale_embeddings(hidden, total_len, stream)?;
        }

        self.twophase_vision_rows(tokens, hidden, h, fp32, stream)?;

        // 2026-09-25: A prompt with vision pad tokens skips the prefix-cache lookup.
        let bs = kv_cache.block_size();
        let prefix_match = if self.tokens_have_vision_pad(tokens) {
            metrale_telemetry::prefix_cache::PrefixMatch::empty()
        } else {
            self.prefix_cache
                .lookup(tokens, bs, seq.session_hash, seq.adapter_id)
        };
        let matched = prefix_match.matched_tokens;
        seq.cached_prefix_tokens = matched;
        seq.cached_prefix_blocks = prefix_match.matched_blocks.len();
        seq.prompt_len = tokens.len();
        for &block_idx in &prefix_match.matched_blocks {
            kv_cache.inc_ref(block_idx);
            seq.block_table.push(block_idx);
        }
        reuse_prefix_match_disk_ids(
            &prefix_match.matched_disk_block_ids,
            &mut seq.disk_block_ids,
        );

        let (kv_write_start, marconi_skip) =
            self.twophase_marconi_restore(&prefix_match, seq, matched, total_len, stream)?;

        let blocks_needed = (total_len - 1) / bs + 1;
        ensure_blocks_through_prefill(
            seq,
            blocks_needed - 1,
            &mut kv_cache,
            self.prefix_cache.as_ref(),
            self.gpu.as_ref(),
            stream,
            self.levers.kv_poison,
        )?;

        let (proc_start, proc_count) = if marconi_skip && kv_write_start > 0 {
            (kv_write_start, total_len - kv_write_start)
        } else {
            (0, total_len)
        };

        if proc_count == 0 {
            return self
                .prefill_full_cache_hit(tokens, seq, hidden, h as u32, bs, total_len, stream);
        }

        // 2026-09-25: Re-embed only the uncached tokens, at hidden rows [0, proc_count).
        if proc_start > 0 {
            let uncached_tokens = &tokens[proc_start..];
            // 2026-09-25: SAFETY: this arm runs only when `proc_start > 0`, which the
            // `(proc_start, proc_count)` binding above reaches only via
            // `(kv_write_start, total_len - kv_write_start)`, so
            // `proc_count == total_len - proc_start == uncached_tokens.len()`
            // and the byte length is `uncached_tokens.len() * size_of::<u32>()`
            // over a live `&[u32]`. A `kv_write_start > total_len` panics
            // before this point.
            let token_ids_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(uncached_tokens.as_ptr() as *const u8, proc_count * 4)
            };
            let token_ids_dev = self.buffers.scratch();
            self.gpu
                .copy_h2d_async(token_ids_bytes, token_ids_dev, stream)?;
            if self.has_ngram_embedding() {
                let cs = proc_start.saturating_sub(self.ngram_lookbehind());
                self.embed_tokens_fused(
                    &tokens[cs..proc_start + proc_count],
                    proc_count,
                    hidden,
                    stream,
                )?;
            } else {
                ops::batched_embed(
                    self.gpu.as_ref(),
                    self.batched_embed_kernel,
                    token_ids_dev,
                    self.embed_tokens.weight,
                    hidden,
                    proc_count as u32,
                    h as u32,
                    stream,
                )?;
            }
            self.scale_embeddings(hidden, proc_count, stream)?;
        }

        let moe_scratch_bytes = proc_count * self.config.num_experts_per_tok * 4 * 2;
        let meta_offset = (moe_scratch_bytes + 7) & !7;
        let meta_base = self.buffers.scratch().offset(meta_offset);
        let slot_offset = (proc_count * 4 + 7) & !7;

        // 2026-09-25: With `proc_start > 0` the earlier K/V is already in the cache,
        // so attention reads it through the paged block table and slots come
        // from `fill_slots_from_block_table` instead of the host.
        let needs_paged = proc_start > 0;

        {
            // 2026-09-25: SAFETY: the model is used only from the scheduler thread (the
            // `Model` trait's `# Safety`), so no other reference to the staging exists.
            let stg = unsafe { &mut *self.pinned_staging.get() };
            stg.positions.clear();
            stg.positions
                .extend(proc_start as u32..(proc_start + proc_count) as u32);

            if !needs_paged {
                stg.slots.clear();
                stg.slots
                    .extend((proc_start..proc_start + proc_count).map(|i| {
                        let block_idx = seq
                            .physical_block_for(i / bs)
                            .unwrap_or(self.dummy_kv_block);
                        (block_idx as i64) * (bs as i64) + ((i % bs) as i64)
                    }));
            }

            // 2026-09-25: Rounding `slot_offset` up to 8 leaves up to 4 pad bytes after the
            // positions array that no copy writes; they are still initialised
            // (see the `pinned_pack` module docs).
            let mut pack = stg.packer_for(self.buffers.scratch_bytes().saturating_sub(meta_offset));
            pack.put_prefix_at("positions", 0, &stg.positions, proc_count)?;
            if !needs_paged {
                pack.put_prefix_at("slots", slot_offset, &stg.slots, proc_count)?;
            }
            self.gpu
                .copy_h2d_async_retained(pack.packed(), meta_base, stream)?;
        }

        if needs_paged {
            let current_blocks = seq.block_table.len();
            let upload_start = self
                .ensure_chunked_prefill_meta(seq, total_len, bs)?
                .uploaded_blocks;
            // 2026-09-25: Once the HSS window has slid, `block_table[i]` is absolute
            // block `hss_window_start() + i` (`physical_block_for`), so the delta
            // cannot be written at its absolute offset; the block-table upload is
            // skipped. The seq_len upload and slot fill below still run.
            if upload_start < current_blocks && seq.hss_window_start() == 0 {
                let new_blocks = &seq.block_table[upload_start..];
                // 2026-09-25: SAFETY: the length is `size_of_val(new_blocks)`, derived from
                // the slice itself, so it can never exceed it, over a live
                // `&[u32]` sub-slice of `seq.block_table`.
                let bt_bytes = unsafe {
                    std::slice::from_raw_parts(
                        new_blocks.as_ptr() as *const u8,
                        std::mem::size_of_val(new_blocks),
                    )
                };
                let block_table_base = seq.chunked_prefill_meta.as_ref().unwrap().block_table;
                self.gpu.copy_h2d_async(
                    bt_bytes,
                    block_table_base.offset(upload_start * std::mem::size_of::<u32>()),
                    stream,
                )?;
                seq.chunked_prefill_meta.as_mut().unwrap().uploaded_blocks = current_blocks;
            }

            let seq_len_val = (proc_start + proc_count) as u32;
            // 2026-09-25: SAFETY: exactly `size_of::<u32>()` bytes over the live, fully
            // initialised `seq_len_val` local on the line above.
            let seq_len_bytes = unsafe {
                std::slice::from_raw_parts(
                    &seq_len_val as *const u32 as *const u8,
                    std::mem::size_of::<u32>(),
                )
            };
            let seq_len_base = seq.chunked_prefill_meta.as_ref().unwrap().seq_len;
            self.gpu
                .copy_h2d_async(seq_len_bytes, seq_len_base, stream)?;

            let block_table_base = seq.chunked_prefill_meta.as_ref().unwrap().block_table;
            ops::fill_slots_from_block_table(
                self.gpu.as_ref(),
                self.fill_slots_kernel,
                meta_base.offset(slot_offset),
                block_table_base,
                proc_start as u32,
                proc_count as u32,
                bs as u32,
                stream,
            )?;
        }

        self.gpu.synchronize(stream)?;

        let (block_table_dev, seq_len_dev) = if needs_paged {
            let page_meta = seq.chunked_prefill_meta.as_ref().unwrap();
            (page_meta.block_table, page_meta.seq_len)
        } else {
            (DevicePtr::NULL, DevicePtr::NULL)
        };

        // 2026-09-25: Request-scoped LoRA routing: `DevicePtr(0)` when there is no LoRA or
        // the request resolves to the active adapter (installed-pair path);
        // `adapter_slot == -1` resolves to the active adapter.
        let seq_slot = self.upload_seq_slot_uniform(
            seq.adapter_slot,
            proc_count,
            self.buffers.lora_seq_slot(),
            stream,
        )?;

        let attn_metadata = AttnMetadataDev {
            positions: meta_base,
            positions_h: meta_base,
            positions_w: meta_base,
            slot: meta_base.offset(slot_offset),
            seq_len: seq_len_dev,
            block_table: block_table_dev,
            max_blocks_per_seq: seq.block_table.len() as u32,
            num_seqs: 1,
            seq_slot,
            moe_row_adapter: metrale_gpu_runtime::gpu::DevicePtr::NULL,
        };

        let ctx = ForwardContext {
            buffers: &self.buffers,
            hc_row_offset: 0,
            gpu: self.gpu.as_ref(),
            config: &self.config,
            dispatch: &self.dispatch,
            derived: &self.derived,
            levers: &self.levers,
            stats: &self.stats,
            attn_metadata: Some(attn_metadata),
            profile: self.profile,
            comm: self.comm_ref(),
            graph_capture: false,
            decode_step: false,
            // 2026-09-25: Marconi warm hit: GDN layers replay from a restored SSM state
            // and skip the FLA chunked kernel (see `ForwardContext::gdn_exact_replay`).
            gdn_exact_replay: marconi_skip,
            gdn_write_on_accept: false,
            token_ids: None,
            host_token_ids: None,
            // 2026-09-25: `None` unless the request routes to a non-active adapter slot.
            routed_lora_layers: self.routed_slot_layers(seq.adapter_slot),
            midchunk_capture: None,
            moe_lora_route: self.moe_lora_route(seq.adapter_slot),
        };

        // 2026-09-25: Marconi hit: the first `cached_prefix_tokens - proc_start` processed
        // tokens replay positions whose K/V is already in shared prefix-cache
        // blocks. Passing that count as the layer write floor keeps attention
        // layers from rewriting those blocks with recomputed K/V.
        let layer_kv_write_start = if marconi_skip {
            seq.cached_prefix_tokens
                .saturating_sub(proc_start)
                .min(proc_count)
        } else {
            kv_write_start
        };
        let gdn_bufs = GdnPrefillBuffers {
            qkv: self.gdn_buf_qkv,
            gate_beta: self.gdn_buf_gate_beta,
            output: self.gdn_buf_out,
            z: self.gdn_buf_z,
            total_len: proc_count,
        };

        for (i, layer) in self.layers.iter().enumerate() {
            if layer.is_ssm_layer() {
                // 2026-09-25: Projections per chunk, staged into the full-length GDN buffers.
                for chunk_start in (0..proc_count).step_by(chunk_size) {
                    let chunk_len = chunk_size.min(proc_count - chunk_start);
                    let hidden_chunk = hidden.offset(chunk_start * h * fp32);
                    let residual_chunk = residual.offset(chunk_start * h * fp32);
                    layer.prefill_phase1(
                        hidden_chunk,
                        residual_chunk,
                        chunk_len,
                        seq.layer_states[i].as_mut(),
                        &mut kv_cache,
                        proc_start + chunk_start,
                        &mut seq.block_table,
                        &mut seq.disk_block_ids,
                        &mut seq.disk_last_offloaded_per_layer,
                        layer_kv_write_start,
                        &gdn_bufs,
                        chunk_start,
                        &ctx,
                        stream,
                    )?;
                }

                // 2026-09-25: One GDN recurrence launch over the whole range.
                layer.prefill_gdn_full(seq.layer_states[i].as_mut(), &gdn_bufs, &ctx, stream)?;

                // 2026-09-25: Post-processing per chunk: gated RMS norm, output projection,
                // residual add and MoE.
                for chunk_start in (0..proc_count).step_by(chunk_size) {
                    let chunk_len = chunk_size.min(proc_count - chunk_start);
                    let hidden_chunk = hidden.offset(chunk_start * h * fp32);
                    let residual_chunk = residual.offset(chunk_start * h * fp32);
                    layer.prefill_phase3(
                        hidden_chunk,
                        residual_chunk,
                        chunk_len,
                        &gdn_bufs,
                        chunk_start,
                        &ctx,
                        stream,
                    )?;
                }
            } else {
                layer
                    .prefill(
                        hidden,
                        residual,
                        proc_count,
                        seq.layer_states[i].as_mut(),
                        &mut kv_cache,
                        proc_start,
                        &mut seq.block_table,
                        &mut seq.disk_block_ids,
                        &mut seq.disk_last_offloaded_per_layer,
                        layer_kv_write_start,
                        &ctx,
                        stream,
                    )
                    .map_err(|e| {
                        anyhow::anyhow!("Two-phase prefill attention layer {i} failed: {e}")
                    })?;
            }
        }

        seq.tokens.extend_from_slice(tokens);
        seq.seq_len = total_len;
        // 2026-09-25: `decode_ckpt_plan` skips `end_block == last_ckpt_block`, so the
        // first decode checkpoint is not taken in the block the prompt ends in.
        seq.last_decode_ckpt_block = seq.tokens.len() / bs;

        let last_hidden = hidden.offset((proc_count - 1) * h * fp32);
        let normed = self.buffers.norm_output();
        let eps = self.config.rms_norm_eps as f32;
        self.final_norm_apply(last_hidden, normed, 1, h as u32, eps, stream)?;

        self.lm_head(normed, stream)?;

        self.prefill_save_snapshot_and_insert(tokens, seq, &mut kv_cache, bs, stream);

        Ok(self.decode_logits_ptr())
    }
}
