// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The profiled decode step (`decode_profiled`) and the
//! self-speculative draft step (`decode_draft`).
//!
//! Owner: model-engine.
//! Invariants:
//! - Both steps push `token` and advance `seq_len` only after the forward
//!   succeeds; an error leaves them unchanged.

#![allow(unused_imports, dead_code)]

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, bail};
use metrale_cache::kv_cache::PagedKvCache;
use metrale_config::{LayerType, ModelConfig};
use metrale_gpu_runtime::buffers::BufferArena;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle, KernelHandle};

use super::block_mgmt::{
    apply_evicted_blocks, ensure_blocks_through_decode, ensure_blocks_through_prefill,
    extract_layer_refs, reuse_prefix_match_disk_ids,
};
use super::ssm_pool::SsmStatePool;
use super::ssm_snapshot::SsmSnapshotPool;
use super::types::{PinnedMetaStaging, TransformerModel};
use crate::traits::{ChunkedPrefillPageMetadata, Model, SequenceState};
use metrale_model_layers::layer::{
    AttnMetadataDev, ForwardContext, GdnPrefillBuffers, LayerState, SsmLayerState, TransformerLayer,
};
use metrale_model_layers::layers::ops;
use metrale_model_layers::speculative::DraftProposer;
use metrale_model_layers::weight_map::{DenseWeight, MtpWeights, QuantizedWeight};

impl TransformerModel {
    /// 2026-09-25: Profile mode: run each layer eagerly, synchronizing and
    /// timing it, and log the time in full-attention layers, in all other
    /// layers (as `ssm`) and in the head.
    pub(super) fn decode_profiled(
        &self,
        token: u32,
        hidden: DevicePtr,
        residual: DevicePtr,
        seq: &mut SequenceState,
        kv_cache: &mut PagedKvCache,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        use std::time::Instant;

        let num_attn = self.config.num_attention_layers();
        let mut attn_us = 0u64;
        let mut ssm_us = 0u64;
        // 2026-09-25: Per-op profiling for MLA models, and while
        // `seq.seq_len < seq.tokens.len() + 2`, which holds whenever `seq_len`
        // equals the token count.
        let is_mla = ctx.config.kv_lora_rank > 0;
        let detail = is_mla || seq.seq_len < seq.tokens.len() + 2;
        let inner_ctx = if detail {
            ctx
        } else {
            // 2026-09-25: The same context with per-op profiling off.
            &ForwardContext {
                buffers: ctx.buffers,
                hc_row_offset: ctx.hc_row_offset,
                gpu: ctx.gpu,
                config: ctx.config,
                dispatch: ctx.dispatch,
                derived: ctx.derived,
                levers: ctx.levers,
                stats: ctx.stats,
                attn_metadata: ctx.attn_metadata,
                profile: false,
                comm: ctx.comm,
                graph_capture: ctx.graph_capture,
                decode_step: false,
                gdn_exact_replay: false,
                gdn_write_on_accept: false,
                token_ids: None,
                host_token_ids: None,
                routed_lora_layers: ctx.routed_lora_layers,
                midchunk_capture: None,
                moe_lora_route: ctx.moe_lora_route,
            }
        };

        // 2026-09-25: DIAG logs (hidden state after the embedding and each
        // layer, then the top-5 logits) under the non-MLA half of the per-op
        // condition above.
        let diag = seq.seq_len < seq.tokens.len() + 2;
        if diag {
            self.gpu.synchronize(stream)?;
            let (vals, norm) = self.readback_f32(hidden, 8)?;
            tracing::info!(
                "DIAG tok={} after_embed (FP32): norm={:.4} vals={:.4?}",
                seq.seq_len,
                norm,
                &vals[..4]
            );
        }

        for (i, layer) in self.layers.iter().enumerate() {
            let t0 = Instant::now();
            layer.decode(
                hidden,
                residual,
                seq.layer_states[i].as_mut(),
                kv_cache,
                seq.seq_len,
                &mut seq.block_table,
                &mut seq.disk_block_ids,
                &mut seq.disk_last_offloaded_per_layer,
                inner_ctx,
                stream,
            )?;
            self.gpu.synchronize(stream)?;
            let elapsed = t0.elapsed().as_micros() as u64;
            if self.config.layer_type(i) == metrale_config::LayerType::FullAttention {
                attn_us += elapsed;
            } else {
                ssm_us += elapsed;
            }

            if diag {
                let (vals, norm) = self.readback_f32(hidden, 8)?;
                let lt = self.config.layer_type(i);
                tracing::info!(
                    "DIAG tok={} after_L{} ({:?}) [FP32]: norm={:.4} vals={:.4?}",
                    seq.seq_len,
                    i,
                    lt,
                    norm,
                    &vals[..4]
                );
            }
        }

        let t0 = Instant::now();
        let normed = self.buffers.norm_output();
        let h = self.config.hidden_size as u32;
        let eps = self.config.rms_norm_eps as f32;
        self.final_norm_apply(hidden, normed, 1, h, eps, stream)?;
        self.lm_head(normed, stream)?;
        self.gpu.synchronize(stream)?;
        let head_us = t0.elapsed().as_micros() as u64;

        if diag {
            let logits_ptr = self.buffers.logits();
            let v = self.config.vocab_size;
            let mut logit_buf = vec![0u8; v * 2];
            self.gpu.copy_d2h(logits_ptr, &mut logit_buf)?;
            let logits: Vec<f32> = logit_buf
                .chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    f32::from_bits((bits as u32) << 16)
                })
                .collect();
            let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            tracing::info!("DIAG tok={} top5_logits: {:?}", seq.seq_len, &indexed[..5]);
        }

        let total_us = attn_us + ssm_us + head_us;
        tracing::info!(
            "PROFILE tok={}: total={:.1}ms attn={:.1}ms({}) ssm={:.1}ms({}) head={:.1}ms",
            seq.seq_len,
            total_us as f64 / 1000.0,
            attn_us as f64 / 1000.0,
            num_attn,
            ssm_us as f64 / 1000.0,
            self.layers.len() - num_attn,
            head_us as f64 / 1000.0,
        );

        seq.tokens.push(token);
        seq.seq_len += 1;
        Ok(self.decode_logits_ptr())
    }

    /// 2026-09-25: Eager decode that skips the linear-attention (SSM) layers,
    /// for self-speculative drafting. It appends KV entries for the draft token
    /// and leaves the SSM state untouched.
    pub(super) fn decode_draft(
        &self,
        token: u32,
        seq: &mut SequenceState,
        _stream: u64,
    ) -> Result<DevicePtr> {
        let stream = self.gpu.default_stream();
        let hidden = self.buffers.hidden_states();
        let residual = self.buffers.residual();

        let mut kv_cache = self.kv_cache.lock();

        // 2026-09-25: `seq.tokens` is the history without `token` (pushed after
        // the forward), which is what `embed_ctx` requires.
        self.embed_ctx(&seq.tokens, token, hidden, stream)?;

        // 2026-09-25: Allocate KV blocks through the current position, then
        // upload the single-sequence attention metadata.
        let bs = kv_cache.block_size();
        let blocks_needed = (seq.seq_len / bs) + 1;
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

        let pos_val = seq.seq_len as u32;
        self.gpu
            .copy_h2d_async(&pos_val.to_le_bytes(), meta_base, stream)?;

        let block_idx = seq
            .physical_block_for(seq.seq_len / bs)
            .unwrap_or(self.dummy_kv_block);
        let global_slot = (block_idx as i64) * (bs as i64) + ((seq.seq_len % bs) as i64);
        self.gpu
            .copy_h2d_async(&global_slot.to_le_bytes(), meta_base.offset(8), stream)?;

        let actual_seq_len = (seq.seq_len + 1) as i32;
        self.gpu
            .copy_h2d_async(&actual_seq_len.to_le_bytes(), meta_base.offset(16), stream)?;

        let bt_i32: Vec<i32> = seq.block_table.iter().map(|&b| b as i32).collect();
        // 2026-09-25: SAFETY: the length is read back off `bt_i32` itself, so the span is
        // exactly `bt_i32.len() * size_of::<i32>()` bytes. `collect()` on the line
        // above initialises every one of those elements (a `collect` Vec has
        // len == the number of items yielded, never a with_capacity gap), and
        // `bt_i32` is only ever shared-borrowed here.
        let bt_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(bt_i32.as_ptr() as *const u8, bt_i32.len() * 4) };
        self.gpu
            .copy_h2d_async(bt_bytes, meta_base.offset(256), stream)?;

        // 2026-09-25: Route the draft through the request's adapter, at the
        // same `+128` offset as `decode_a`, so drafts come from the adapter the
        // verify uses.
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
            moe_row_adapter: DevicePtr::NULL,
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
            profile: false,
            comm: self.comm_ref(),
            graph_capture: false,
            decode_step: false,
            gdn_exact_replay: false,
            gdn_write_on_accept: false,
            token_ids: None,
            host_token_ids: None,
            routed_lora_layers: None,
            midchunk_capture: None,
            moe_lora_route: self.decode_moe_route(),
        };

        // 2026-09-25: Skip the linear-attention (SSM) layers; every other layer
        // runs.
        for (i, layer) in self.layers.iter().enumerate() {
            if self.config.layer_type(i) == LayerType::LinearAttention {
                continue;
            }
            layer.decode(
                hidden,
                residual,
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

        let normed = self.buffers.norm_output();
        let h = self.config.hidden_size as u32;
        let eps = self.config.rms_norm_eps as f32;
        self.final_norm_apply(hidden, normed, 1, h, eps, stream)?;
        self.lm_head(normed, stream)?;

        seq.tokens.push(token);
        seq.seq_len += 1;

        Ok(self.decode_logits_ptr())
    }
}
