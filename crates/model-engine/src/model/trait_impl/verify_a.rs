// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Eager multi-token verify (`Model::decode_verify`) and the per-token
//! attention loop that the graphed verify paths use for sliding-window layers.
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
use super::super::ssm_batched_copy::{StateCopy, run_ssm_state_copies};
use super::super::ssm_pool::SsmStatePool;
use super::super::ssm_snapshot::SsmSnapshotPool;
use super::super::types::{PinnedMetaStaging, TransformerModel};
use crate::traits::{ChunkedPrefillPageMetadata, Model, SequenceState};
use crate::traits::{ModelForward, ModelLogits};
use metrale_model_layers::layer::{
    AttnMetadataDev, ForwardContext, GdnPrefillBuffers, LayerState, SsmLayerState, TransformerLayer,
};
use metrale_model_layers::layers::ops;
use metrale_model_layers::speculative::DraftProposer;
use metrale_model_layers::weight_map::{DenseWeight, MtpWeights, QuantizedWeight};

impl TransformerModel {
    pub(super) fn decode_verify_dispatch(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        stream: u64,
    ) -> Result<Vec<u32>> {
        let k = tokens.len();
        if k == 0 {
            return Ok(Vec::new());
        }
        if k == 1 {
            let logits = self.decode(tokens[0], seq, stream)?;
            let tok = self.argmax_on_device(logits, stream)?;
            return Ok(vec![tok]);
        }

        // 2026-09-25: All K tokens go through each layer before the next: attention layers
        // one token at a time with that token's metadata, every other layer in one
        // `decode_batched` call.
        let stream = self.gpu.default_stream();
        let h = self.config.hidden_size;
        let bf16 = 2usize;
        let fp32 = 2usize;

        let hidden = self.buffers.hidden_states();
        let residual = self.buffers.residual();

        let mut kv_cache = self.kv_cache.lock();

        for (t, &token) in tokens.iter().enumerate() {
            let h_t = hidden.offset(t * h * fp32);
            self.embed(token, h_t, stream)?;
        }

        for (i, layer) in self.layers.iter().enumerate() {
            let layer_type = self.config.layer_type(i);

            // 2026-09-25: Full and sliding attention both take the per-token path: the
            // default `decode_batched` loop reuses one metadata upload for every token, so
            // all K would attend and write KV at the same position.
            if matches!(
                layer_type,
                metrale_config::LayerType::FullAttention
                    | metrale_config::LayerType::SlidingAttention
            ) {
                for t in 0..k {
                    let pos = seq.seq_len + t;
                    let bs = kv_cache.block_size();
                    let blocks_needed = (pos / bs) + 1;
                    ensure_blocks_through_decode(
                        seq,
                        blocks_needed - 1,
                        &mut kv_cache,
                        self.prefix_cache.as_ref(),
                        self.gpu.as_ref(),
                        stream,
                        self.levers.kv_poison,
                    )?;

                    let meta_base = self.buffers.scratch().offset(32768);
                    let max_blocks = seq.block_table.len() as u32;
                    let pos_val = pos as u32;
                    self.gpu
                        .copy_h2d_async(&pos_val.to_le_bytes(), meta_base, stream)?;
                    let block_idx = seq
                        .physical_block_for(pos / bs)
                        .unwrap_or(self.dummy_kv_block);
                    let global_slot = (block_idx as i64) * (bs as i64) + ((pos % bs) as i64);
                    self.gpu.copy_h2d_async(
                        &global_slot.to_le_bytes(),
                        meta_base.offset(8),
                        stream,
                    )?;
                    let actual_seq_len = (pos + 1) as i32;
                    self.gpu.copy_h2d_async(
                        &actual_seq_len.to_le_bytes(),
                        meta_base.offset(16),
                        stream,
                    )?;
                    let bt_i32: Vec<i32> = seq.block_table.iter().map(|&b| b as i32).collect();
                    // 2026-09-25: SAFETY: the length is derived from the source itself
                    // (`bt_i32.len() * 4 == size_of_val(&bt_i32[..])`), and `bt_i32` is a
                    // fresh `collect()` on the line above, so all `len` elements are
                    // initialised and nothing past `len` is read. `i32` is POD, and
                    // `copy_h2d_async` lets the source drop once it returns.
                    let bt_bytes: &[u8] = unsafe {
                        std::slice::from_raw_parts(bt_i32.as_ptr() as *const u8, bt_i32.len() * 4)
                    };
                    self.gpu
                        .copy_h2d_async(bt_bytes, meta_base.offset(256), stream)?;

                    // 2026-09-25: Request-scoped LoRA routing: one adapter for the sequence, a
                    // one-element slot buffer at +128 (slot +8, seq_len +16, block table
                    // +256). `DevicePtr(0)` selects the installed-pair path.
                    let seq_slot = self.upload_seq_slot_uniform(
                        seq.adapter_slot,
                        1,
                        meta_base.offset(128),
                        stream,
                    )?;

                    let attn_metadata = AttnMetadataDev {
                        positions: meta_base,
                        positions_h: meta_base,
                        positions_w: meta_base,
                        slot: meta_base.offset(8),
                        seq_len: meta_base.offset(16),
                        block_table: meta_base.offset(256),
                        max_blocks_per_seq: max_blocks,
                        num_seqs: 1,
                        seq_slot,
                        moe_row_adapter: metrale_gpu_runtime::gpu::DevicePtr::NULL,
                    };

                    let ctx = ForwardContext {
                        decode_step: false,
                        buffers: &self.buffers,
                        hc_row_offset: 0,
                        gpu: self.gpu.as_ref(),
                        config: &self.config,
                        dispatch: &self.dispatch,
                        derived: &self.derived,
                        levers: &self.levers,
                        stats: &self.stats,
                        attn_metadata: Some(attn_metadata),
                        profile: false,
                        comm: self.comm_ref(),
                        graph_capture: false,
                        gdn_exact_replay: false,
                        gdn_write_on_accept: false,
                        token_ids: None,
                        host_token_ids: None,
                        routed_lora_layers: None,
                        midchunk_capture: None,
                        moe_lora_route: self.decode_moe_route(),
                    };

                    let h_t = hidden.offset(t * h * fp32);
                    let r_t = residual.offset(t * h * fp32);
                    layer.decode(
                        h_t,
                        r_t,
                        seq.layer_states[i].as_mut(),
                        &mut kv_cache,
                        pos,
                        &mut seq.block_table,
                        &mut seq.disk_block_ids,
                        &mut seq.disk_last_offloaded_per_layer,
                        &ctx,
                        stream,
                    )?;
                }
            } else {
                let ctx = ForwardContext {
                    decode_step: false,
                    buffers: &self.buffers,
                    hc_row_offset: 0,
                    gpu: self.gpu.as_ref(),
                    config: &self.config,
                    dispatch: &self.dispatch,
                    derived: &self.derived,
                    levers: &self.levers,
                    stats: &self.stats,
                    attn_metadata: None,
                    profile: false,
                    comm: self.comm_ref(),
                    graph_capture: false,
                    gdn_exact_replay: false,
                    gdn_write_on_accept: false,
                    token_ids: None,
                    host_token_ids: None,
                    routed_lora_layers: None,
                    midchunk_capture: None,
                    moe_lora_route: self.decode_moe_route(),
                };

                layer.decode_batched(
                    hidden,
                    residual,
                    k,
                    seq.layer_states[i].as_mut(),
                    &mut kv_cache,
                    seq.seq_len,
                    &mut seq.block_table,
                    &mut seq.disk_block_ids,
                    &mut seq.disk_last_offloaded_per_layer,
                    &ctx,
                    stream,
                )?;
            }
        }

        let normed = self.buffers.norm_output();
        let eps = self.config.rms_norm_eps as f32;
        self.final_norm_apply(hidden, normed, k as u32, h as u32, eps, stream)?;

        self.lm_head_batched(normed, k as u32, self.buffers.logits(), stream)?;

        let vocab = self.config.vocab_size;
        let mut results = Vec::with_capacity(k);
        for t in 0..k {
            let logits_t = self.buffers.logits().offset(t * vocab * bf16);
            let out_ptr = self.buffers.scratch().offset(t * 4);
            ops::argmax_bf16(
                self.gpu.as_ref(),
                self.argmax_kernel,
                logits_t,
                out_ptr,
                vocab as u32,
                stream,
            )?;
        }
        let mut buf = vec![0u8; k * 4];
        self.gpu.copy_d2h(self.buffers.scratch(), &mut buf)?;
        for t in 0..k {
            let tok =
                u32::from_le_bytes([buf[t * 4], buf[t * 4 + 1], buf[t * 4 + 2], buf[t * 4 + 3]]);
            results.push(tok);
        }

        // 2026-09-25: All K tokens are appended and `seq_len` advances by K, as in the
        // graphed verify paths.
        for &token in tokens {
            seq.tokens.push(token);
        }
        seq.seq_len += k;

        Ok(results)
    }

    /// 2026-09-25: Per-token attention decode over K verify rows, uploading each token's
    /// metadata (position, KV slot, seq_len, block table) before its `decode()`, as the
    /// eager verify above does inline.
    ///
    /// For sliding-window attention layers in the graphed verify bodies, which would
    /// otherwise reach the default `decode_batched` loop and decode every token at the
    /// same position. It uploads from the host per token, so callers run it outside
    /// CUDA-graph capture; the verify bodies disable graphs for models with sliding layers.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn verify_attention_per_token(
        &self,
        layer: &dyn TransformerLayer,
        layer_idx: usize,
        hidden: DevicePtr,
        residual: DevicePtr,
        k: usize,
        seq: &mut SequenceState,
        kv_cache: &mut PagedKvCache,
        stream: u64,
    ) -> Result<()> {
        let h = self.config.hidden_size;
        let bf16 = 2usize;
        // 2026-09-25: Its own staging, not `scratch + 32768`, which holds the metadata the
        // full-attention layers still read (see the `verify_ptok_meta` field doc).
        const PTOK_META_BYTES: usize = 16 * 1024;
        let meta_base = match self.verify_ptok_meta.get() {
            Some(&p) => p,
            None => {
                let p = self.gpu.alloc(PTOK_META_BYTES)?;
                *self.verify_ptok_meta.get_or_init(|| p)
            }
        };
        // 2026-09-25: 4 blocks of slack for the table growth `ensure_blocks_through_decode`
        // may add in the loop.
        anyhow::ensure!(
            256 + (seq.block_table.len() + 4) * 4 <= PTOK_META_BYTES,
            "verify_attention_per_token: block table ({} blocks) exceeds the \
             16 KB per-token metadata staging",
            seq.block_table.len(),
        );
        for t in 0..k {
            let pos = seq.seq_len + t;
            let bs = kv_cache.block_size();
            let blocks_needed = (pos / bs) + 1;
            ensure_blocks_through_decode(
                seq,
                blocks_needed - 1,
                kv_cache,
                self.prefix_cache.as_ref(),
                self.gpu.as_ref(),
                stream,
                self.levers.kv_poison,
            )?;

            let max_blocks = seq.block_table.len() as u32;
            let pos_val = pos as u32;
            self.gpu
                .copy_h2d_async(&pos_val.to_le_bytes(), meta_base, stream)?;
            let block_idx = seq
                .physical_block_for(pos / bs)
                .unwrap_or(self.dummy_kv_block);
            let global_slot = (block_idx as i64) * (bs as i64) + ((pos % bs) as i64);
            self.gpu
                .copy_h2d_async(&global_slot.to_le_bytes(), meta_base.offset(8), stream)?;
            let actual_seq_len = (pos + 1) as i32;
            self.gpu
                .copy_h2d_async(&actual_seq_len.to_le_bytes(), meta_base.offset(16), stream)?;
            let bt_i32: Vec<i32> = seq.block_table.iter().map(|&b| b as i32).collect();
            // 2026-09-25: SAFETY: length derived from `bt_i32` itself (`len() * 4`), a
            // fresh `collect()` above; i32 is POD; `copy_h2d_async` lets the source
            // drop once it returns.
            let bt_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(bt_i32.as_ptr() as *const u8, bt_i32.len() * 4)
            };
            self.gpu
                .copy_h2d_async(bt_bytes, meta_base.offset(256), stream)?;

            let seq_slot =
                self.upload_seq_slot_uniform(seq.adapter_slot, 1, meta_base.offset(128), stream)?;

            let attn_metadata = AttnMetadataDev {
                positions: meta_base,
                positions_h: meta_base,
                positions_w: meta_base,
                slot: meta_base.offset(8),
                seq_len: meta_base.offset(16),
                block_table: meta_base.offset(256),
                max_blocks_per_seq: max_blocks,
                num_seqs: 1,
                seq_slot,
                moe_row_adapter: metrale_gpu_runtime::gpu::DevicePtr::NULL,
            };

            let ctx = ForwardContext {
                decode_step: false,
                buffers: &self.buffers,
                gpu: self.gpu.as_ref(),
                config: &self.config,
                dispatch: &self.dispatch,
                derived: &self.derived,
                levers: &self.levers,
                stats: &self.stats,
                attn_metadata: Some(attn_metadata),
                profile: false,
                comm: self.comm_ref(),
                graph_capture: false,
                gdn_exact_replay: false,
                gdn_write_on_accept: false,
                token_ids: None,
                hc_row_offset: 0,
                host_token_ids: None,
                routed_lora_layers: None,
                midchunk_capture: None,
                moe_lora_route: self.decode_moe_route(),
            };

            let h_t = hidden.offset(t * h * bf16);
            let r_t = residual.offset(t * h * bf16);
            layer.decode(
                h_t,
                r_t,
                seq.layer_states[layer_idx].as_mut(),
                kv_cache,
                pos,
                &mut seq.block_table,
                &mut seq.disk_block_ids,
                &mut seq.disk_last_offloaded_per_layer,
                &ctx,
                stream,
            )?;
        }
        Ok(())
    }
}
