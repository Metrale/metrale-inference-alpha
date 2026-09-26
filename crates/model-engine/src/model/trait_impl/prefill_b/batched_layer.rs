// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Per-layer dispatchers for the kernel-batched prefill: each call runs one layer over the packed tokens of all N streams.
//!
//! `prefill_attn_batched_layer` calls the layer's `prefill_inner_batched_q12`.
//! `prefill_ssm_batched_layer` runs the projections once over all tokens,
//! conv1d per stream, the GDN recurrence (one batched call for equal lengths,
//! otherwise a varlen call or a per-stream loop), and the post-GDN phase once.
//! `prefill_dense_batched_layer` has no caller in the tree.
//!
//! Owner: model-engine.
//! Invariants: none beyond the types.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use anyhow::Result;
use metrale_cache::kv_cache::PagedKvCache;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::super::super::types::TransformerModel;
use crate::traits::SequenceState;
use metrale_model_layers::layer::{
    AttnMetadataDev, BatchedAttnMetadata, ForwardContext, GdnPrefillBuffers, LayerState,
    TransformerLayer,
};

impl TransformerModel {
    /// 2026-09-25: Run one attention layer over the N streams' packed tokens.
    ///
    /// `hidden_stacked` and `residual_stacked` hold the streams back to back
    /// (stream b at Σ `proc_count` of the earlier streams). `meta` is the
    /// `BatchedAttnMetadata` from `stage_batched_attn_metadata`. `seqs`,
    /// `layer_idx` and `kv_write_starts` are not used.
    pub(in crate::model) fn prefill_attn_batched_layer(
        &self,
        layer: &dyn TransformerLayer,
        layer_idx: usize,
        hidden_stacked: DevicePtr,
        residual_stacked: DevicePtr,
        seqs: &mut [&mut SequenceState],
        kv_cache: &mut PagedKvCache,
        kv_write_starts: &[usize],
        seq_lens_start: usize,
        meta: &BatchedAttnMetadata,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        // 2026-09-25: `Qwen3AttentionLayer::prefill_inner_batched_q12` runs the
        // layer's `prefill_inner` on the packed input with `meta` as the batched
        // metadata. Every other layer type returns an error from it, and an
        // error fails the whole batch.
        debug_assert_eq!(seqs.len() as u32, meta.batch_size);
        let _ = (layer_idx, kv_write_starts);
        let num_tokens = meta.total_tokens as usize;
        let _ = seqs;
        layer.prefill_inner_batched_q12(
            hidden_stacked,
            residual_stacked,
            num_tokens,
            kv_cache,
            seq_lens_start,
            meta,
            ctx,
            stream,
        )
    }

    /// 2026-09-25: Run one SSM layer over the N streams' packed tokens. Stream
    /// b's recurrent state is `seqs[b].layer_states[layer_idx]`.
    pub(in crate::model) fn prefill_ssm_batched_layer(
        &self,
        layer: &dyn TransformerLayer,
        layer_idx: usize,
        hidden_stacked: DevicePtr,
        residual_stacked: DevicePtr,
        seqs: &mut [&mut SequenceState],
        kv_cache: &mut PagedKvCache,
        seqs_proc_start: &[usize],
        meta: &BatchedAttnMetadata,
        gdn_bufs: &GdnPrefillBuffers,
        h_state_ptrs_scratch_offset: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let n = seqs.len();
        debug_assert_eq!(n as u32, meta.batch_size);
        debug_assert_eq!(n, seqs_proc_start.len());

        // 2026-09-25: The projections run once over all packed tokens, conv1d
        // runs per stream (it advances that stream's conv state), then one L2
        // step runs over all tokens.
        let total_tokens = meta.total_tokens as usize;
        let _ = &kv_cache;
        layer.prefill_phase1_proj_batched(
            hidden_stacked,
            residual_stacked,
            total_tokens,
            gdn_bufs,
            ctx,
            stream,
        )?;
        for (b, seq) in seqs.iter_mut().enumerate() {
            let off = meta.cu_seqlens_host[b] as usize;
            let len = (meta.cu_seqlens_host[b + 1] - meta.cu_seqlens_host[b]) as usize;
            let layer_state = seq.layer_states[layer_idx].as_mut();
            layer.prefill_phase1_conv1d_one(layer_state, off, len, gdn_bufs, ctx, stream)?;
        }
        layer.prefill_phase1_l2_batched(total_tokens, gdn_bufs, ctx, stream)?;

        // 2026-09-25: GDN recurrence. With equal lengths, one batched call over
        // the per-stream `h_state` pointers staged at
        // `h_state_ptrs_scratch_offset`, which the caller places after the staged
        // metadata.
        let cu = &meta.cu_seqlens_host;
        let n_streams = seqs.len();
        let uniform = (1..=n_streams).all(|b| (cu[b] - cu[b - 1]) == (cu[1] - cu[0]));
        if uniform {
            let h_state_ptrs_dev =
                self.stage_h_state_ptrs(layer_idx, seqs, h_state_ptrs_scratch_offset, stream)?;
            layer.prefill_gdn_full_batched(
                h_state_ptrs_dev,
                gdn_bufs,
                meta.batch_size,
                meta.chunk_len,
                ctx,
                stream,
            )?;
            // 2026-09-25: With an f16-sized pool the pointers name FP32 staging
            // blobs (`stage_h_state_ptrs` widened into them); narrow them back
            // now that the kernel has run. A no-op on an FP32-sized pool.
            self.narrow_h_state_stages(layer_idx, seqs, stream)?;
        } else {
            // 2026-09-25: Differing lengths: try one varlen FLA call over
            // `cu_seqlens`. It declines when the `gdn_batched_fla` lever is off,
            // a head dimension is not 128, `cu_seqlens` or the FLA scratch is
            // NULL, or an FLA kernel is missing; each stream then runs the
            // single-stream GDN over its own slice.
            let mut total_nt = 0usize;
            let mut max_nc = 0u32;
            let mut max_sl = 0u32;
            for b in 0..n {
                let len = (cu[b + 1] - cu[b]) as u32;
                let ncc = len.div_ceil(64);
                total_nt += ncc as usize;
                max_nc = max_nc.max(ncc);
                max_sl = max_sl.max(len);
            }
            let h_state_ptrs_dev =
                self.stage_h_state_ptrs(layer_idx, seqs, h_state_ptrs_scratch_offset, stream)?;
            let did_varlen = layer.prefill_gdn_full_batched_fla_varlen(
                h_state_ptrs_dev,
                gdn_bufs,
                meta.batch_size,
                meta.cu_seqlens,
                max_nc,
                total_nt,
                max_sl,
                ctx,
                stream,
            )?;
            if did_varlen {
                // 2026-09-25: The same narrowing as the equal-length arm.
                self.narrow_h_state_stages(layer_idx, seqs, stream)?;
            }
            if !did_varlen {
                let nk = ctx.config.linear_num_key_heads;
                let kd = ctx.config.linear_key_head_dim;
                let nv = ctx.config.linear_num_value_heads;
                let vd = ctx.config.linear_value_head_dim;
                let key_dim = nk * kd;
                let value_dim = nv * vd;
                let conv_dim = key_dim * 2 + value_dim;
                let bf16 = 2usize;
                let fp32 = 4usize;
                for (b, seq) in seqs.iter_mut().enumerate() {
                    let off = cu[b] as usize;
                    let len = (cu[b + 1] - cu[b]) as usize;
                    let gb = GdnPrefillBuffers {
                        qkv: gdn_bufs.qkv.offset(off * conv_dim * bf16),
                        gate_beta: gdn_bufs.gate_beta.offset(off * (nv * 2) * fp32),
                        output: gdn_bufs.output.offset(off * value_dim * bf16),
                        z: gdn_bufs.z.offset(off * value_dim * bf16),
                        total_len: len,
                    };
                    let st = seq.layer_states[layer_idx].as_mut();
                    layer.prefill_gdn_full(st, &gb, ctx, stream)?;
                }
            }
        }

        // 2026-09-25: The post-GDN phase (gated RMS norm, output projection,
        // MoE, residuals) takes no per-stream state, so it runs once over all
        // packed tokens from offset 0.
        let total = meta.total_tokens as usize;
        layer.prefill_phase3(
            hidden_stacked,
            residual_stacked,
            total,
            gdn_bufs,
            0,
            ctx,
            stream,
        )?;

        let _ = meta;
        Ok(())
    }

    /// 2026-09-25: Run one layer with no per-stream state over all packed
    /// tokens in a single `layer.prefill` call, passing the first stream's
    /// layer state and block table. No caller in the tree uses it.
    pub(in crate::model) fn prefill_dense_batched_layer(
        &self,
        layer: &dyn TransformerLayer,
        layer_idx: usize,
        hidden_stacked: DevicePtr,
        residual_stacked: DevicePtr,
        total_tokens: usize,
        seqs: &mut [&mut SequenceState],
        kv_cache: &mut PagedKvCache,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if seqs.is_empty() {
            return Ok(());
        }
        let first_seq = &mut **seqs.first_mut().unwrap();
        layer.prefill(
            hidden_stacked,
            residual_stacked,
            total_tokens,
            first_seq.layer_states[layer_idx].as_mut(),
            kv_cache,
            0,
            &mut first_seq.block_table,
            &mut first_seq.disk_block_ids,
            &mut first_seq.disk_last_offloaded_per_layer,
            0,
            ctx,
            stream,
        )
    }
}
