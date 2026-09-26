// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: DFlash decode and verify of one sequence in a single forward over `M = 1 + drafts` rows.
//!
//! Row 0 is the accepted token and rows `1..M` are the drafts, so every
//! weight is read once for the decode and the verify. The DFlash hidden
//! capture is taken from row 0 only, so the drafter conditions on an accepted
//! token's per-layer hidden, never on a draft's.
//!
//! The `unsafe { from_raw_parts(..) }` blocks view `Vec`s or arrays of `u32`,
//! `i32` or `i64` (POD, no padding) as bytes for `copy_h2d_async`, whose
//! contract lets the source drop once it returns.
//!
//! Owner: model-engine (speculative verify).
//! Invariants:
//! - `seq.tokens` and `seq.seq_len` change only on `Ok`. An `Err` can leave
//!   KV blocks allocated and layer state written.

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
use crate::traits::{ChunkedPrefillPageMetadata, Model, SequenceState};
use metrale_model_layers::layer::{
    AttnMetadataDev, ForwardContext, GdnPrefillBuffers, LayerState, SsmLayerState, TransformerLayer,
};
use metrale_model_layers::layers::ops;
use metrale_model_layers::speculative::DraftProposer;
use metrale_model_layers::weight_map::{DenseWeight, MtpWeights, QuantizedWeight};

impl TransformerModel {
    /// 2026-09-25: `tokens` is `[accepted_token, draft_0, ..., draft_{k-1}]`.
    /// Returns the argmax of each of the `tokens.len()` rows: row `j` is the
    /// token predicted after `tokens[j]`. On `Ok`, `seq.tokens` gains `tokens`
    /// and `seq.seq_len` grows by `tokens.len()`.
    pub(super) fn decode_and_verify_fused_dispatch(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        _stream: u64,
    ) -> Result<Vec<u32>> {
        let stream = self.gpu.default_stream();
        let h = self.config.hidden_size;
        let bf16 = 2usize;
        let m = tokens.len();
        let vocab = self.config.vocab_size;

        let hidden = self.buffers.hidden_states();
        let residual = self.buffers.residual();

        let mut kv_cache = self.kv_cache.lock();

        // 2026-09-25: Everything up to the graph section changes per step
        // and runs outside any capture.
        for (t, &tok) in tokens.iter().enumerate() {
            self.embed(tok, hidden.offset(t * h * bf16), stream)?;
        }

        let bs = kv_cache.block_size();
        for t in 0..m {
            let pos = seq.seq_len + t;
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
        }

        // 2026-09-25: M-row attention metadata at `scratch + 32768`: positions
        // at +0, seq_slot at +128, slots at +256, seq_lens at +512, block
        // table at +768.
        let meta_base = self.buffers.scratch().offset(32768);
        let max_blocks = self.max_blocks_per_seq;

        let positions: Vec<u32> = (0..m).map(|t| (seq.seq_len + t) as u32).collect();
        // 2026-09-25: SAFETY: `positions` is collected from `0..m`, so its
        // length is `m` and `m * 4 == size_of_val(&positions[..])`. `u32` is
        // POD.
        let pos_bytes =
            unsafe { std::slice::from_raw_parts(positions.as_ptr() as *const u8, m * 4) };
        self.gpu.copy_h2d_async(pos_bytes, meta_base, stream)?;

        let mut slots = vec![0i64; m];
        for t in 0..m {
            let pos = seq.seq_len + t;
            let block_idx = pos / bs;
            let block_offset = pos % bs;
            let physical_block = seq.physical_block_for(block_idx).unwrap_or(0);
            slots[t] = (physical_block as i64) * (bs as i64) + (block_offset as i64);
        }
        // 2026-09-25: SAFETY: `slots` is `vec![0i64; m]`, so its length is `m`
        // and `m * 8 == size_of_val(&slots[..])`. `i64` is POD.
        let slot_bytes = unsafe { std::slice::from_raw_parts(slots.as_ptr() as *const u8, m * 8) };
        self.gpu
            .copy_h2d_async(slot_bytes, meta_base.offset(256), stream)?;

        // 2026-09-25: Causal lengths: row t attends over `seq_len + t + 1`
        // keys, itself included.
        let seq_lens_meta: Vec<i32> = (0..m).map(|t| (seq.seq_len + t + 1) as i32).collect();
        // 2026-09-25: SAFETY: `seq_lens_meta` is collected from `0..m`, so its
        // length is `m` and `m * 4 == size_of_val(&seq_lens_meta[..])`. `i32`
        // is POD.
        let sl_bytes =
            unsafe { std::slice::from_raw_parts(seq_lens_meta.as_ptr() as *const u8, m * 4) };
        self.gpu
            .copy_h2d_async(sl_bytes, meta_base.offset(512), stream)?;

        // 2026-09-25: Block table: M copies of this sequence's table.
        let mb = max_blocks as usize;
        let needed = m * mb;
        let mut bt_buf_vec;
        let mut bt_buf_stack = [0i32; 1024];
        let bt_buf: &mut [i32] = if needed <= 1024 {
            &mut bt_buf_stack[..needed]
        } else {
            bt_buf_vec = vec![0i32; needed];
            &mut bt_buf_vec
        };
        for row in 0..m {
            for (j, &block) in seq.block_table.iter().enumerate().take(mb) {
                bt_buf[row * mb + j] = block as i32;
            }
        }
        // 2026-09-25: SAFETY: `bt_buf.len() == needed` in both arms of the
        // `if needed <= 1024` above (`bt_buf_stack[..needed]` or
        // `vec![0i32; needed]`), so `needed * 4 == size_of_val(bt_buf)`. Both
        // arms are zero-initialised, which covers entries the fill loop skips
        // when `block_table.len() < mb`.
        let bt_bytes =
            unsafe { std::slice::from_raw_parts(bt_buf.as_ptr() as *const u8, needed * 4) };
        self.gpu
            .copy_h2d_async(bt_bytes, meta_base.offset(768), stream)?;

        // 2026-09-25: One sequence, one adapter: `upload_seq_slot_uniform`
        // writes M equal entries at +128 before any capture, or returns
        // `DevicePtr(0)` when no adapter is loaded or the sequence uses the
        // active one. The +128..+256 gap holds 32 u32 entries.
        debug_assert!(m <= 32, "fused verify seq_slot +128 gap holds M ≤ 32");
        let seq_slot =
            self.upload_seq_slot_uniform(seq.adapter_slot, m, meta_base.offset(128), stream)?;

        let metadata = AttnMetadataDev {
            positions: meta_base,
            positions_h: meta_base,
            positions_w: meta_base,
            slot: meta_base.offset(256),
            seq_len: meta_base.offset(512),
            block_table: meta_base.offset(768),
            max_blocks_per_seq: max_blocks,
            num_seqs: m as u32,
            seq_slot,
            moe_row_adapter: metrale_gpu_runtime::gpu::DevicePtr::NULL,
        };

        // 2026-09-25: Once FP8 KV calibration has frozen, turn graphs back on.
        if self.config.fp8_kv_calibration_tokens > 0
            && self
                .suppress_graphs
                .load(std::sync::atomic::Ordering::Relaxed)
            && self.fp8_calibration_frozen()
        {
            self.suppress_graphs
                .store(false, std::sync::atomic::Ordering::Relaxed);
            tracing::info!("FP8 calibration frozen — re-enabling CUDA graphs (DFlash fused)");
        }

        let hss_engaged = kv_cache.config().cache_blocks_per_seq.is_some();
        // 2026-09-25: `lora_eager` (`METRALE_LORA_EAGER`) keeps a model with an
        // adapter eager.
        let lora_eager = self.lora.is_some() && self.levers.lora_eager;
        let use_graphs = self.comm.is_none()
            && !self
                .suppress_graphs
                .load(std::sync::atomic::Ordering::Relaxed)
            && !hss_engaged
            && !lora_eager
            // 2026-09-25: Sliding-window layers verify through the per-token
            // loop (`verify_attention_per_token`), whose per-token metadata
            // uploads cannot be captured.
            && !(0..self.layers.len())
                .any(|i| self.config.layer_type(i) == LayerType::SlidingAttention);

        let ctx = ForwardContext {
            buffers: &self.buffers,
            hc_row_offset: 0,
            gpu: self.gpu.as_ref(),
            config: &self.config,
            dispatch: &self.dispatch,
            derived: &self.derived,
            levers: &self.levers,
            stats: &self.stats,
            attn_metadata: Some(metadata),
            profile: false,
            comm: self.comm_ref(),
            graph_capture: use_graphs,
            decode_step: false,
            gdn_exact_replay: false,
            gdn_write_on_accept: false,
            token_ids: None,
            host_token_ids: None,
            routed_lora_layers: None,
            midchunk_capture: None,
            moe_lora_route: self.decode_moe_route(),
        };

        // 2026-09-25: Replay the graph cached for `(slot, M)`, or run the
        // forward, under capture when graphs are on.
        let cache_key = (seq.slot_idx, m);
        let mut graph_cache = if use_graphs {
            Some(self.fused_graph.lock())
        } else {
            None
        };

        let cached_for_slot = graph_cache
            .as_ref()
            .and_then(|c| c.get(&cache_key).copied());
        if let Some(graph) = cached_for_slot
            && graph.0 != 0
        {
            self.gpu.launch_graph(graph, stream)?;
        }
        let need_run = cached_for_slot.is_none();
        if need_run {
            let seq_lens_vec: Vec<usize> = (0..m).map(|t| seq.seq_len + t).collect();
            let block_tables_vec: Vec<Vec<u32>> = vec![seq.block_table.clone(); m];

            if use_graphs {
                self.gpu.begin_capture(stream)?;
            }

            for (layer_idx, layer) in self.layers.iter().enumerate() {
                let layer_type = self.config.layer_type(layer_idx);

                if layer_type == LayerType::FullAttention {
                    if hss_engaged {
                        // 2026-09-25: HSS: `decode_multi_seq` takes no disk
                        // block ids, so attention runs through
                        // `decode_batched`, which does.
                        layer.decode_batched(
                            hidden,
                            residual,
                            m,
                            seq.layer_states[layer_idx].as_mut(),
                            &mut kv_cache,
                            seq.seq_len,
                            &mut seq.block_table,
                            &mut seq.disk_block_ids,
                            &mut seq.disk_last_offloaded_per_layer,
                            &ctx,
                            stream,
                        )?;
                    } else {
                        let mut dummy_states: Vec<Box<dyn LayerState>> = (0..m)
                            .map(|_| layer.alloc_state(self.gpu.as_ref()))
                            .collect::<Result<_>>()?;
                        let mut refs: Vec<&mut (dyn LayerState + 'static)> =
                            dummy_states.iter_mut().map(|s| s.as_mut()).collect();
                        layer.decode_multi_seq(
                            hidden,
                            residual,
                            m,
                            &mut refs,
                            &mut kv_cache,
                            &seq_lens_vec,
                            &block_tables_vec,
                            &ctx,
                            stream,
                        )?;
                    }
                } else if layer_type == LayerType::SlidingAttention {
                    // 2026-09-25: Sliding-window attention: one metadata
                    // upload per token. Graphs are off for these models
                    // (`use_graphs`).
                    self.verify_attention_per_token(
                        layer.as_ref(),
                        layer_idx,
                        hidden,
                        residual,
                        m,
                        seq,
                        &mut kv_cache,
                        stream,
                    )?;
                } else {
                    layer.decode_batched(
                        hidden,
                        residual,
                        m,
                        seq.layer_states[layer_idx].as_mut(),
                        &mut kv_cache,
                        seq.seq_len,
                        &mut seq.block_table,
                        &mut seq.disk_block_ids,
                        &mut seq.disk_last_offloaded_per_layer,
                        &ctx,
                        stream,
                    )?;
                }
                // 2026-09-25: DFlash capture from row 0, the accepted token.
                self.try_dflash_capture(layer_idx, 0, stream)?;
            }

            let normed = self.buffers.norm_output();
            self.final_norm_apply(
                hidden,
                normed,
                m as u32,
                h as u32,
                self.config.rms_norm_eps as f32,
                stream,
            )?;

            self.lm_head_batched(normed, m as u32, self.buffers.logits(), stream)?;

            // 2026-09-25: Argmax of each row into scratch, a fixed address, so a
            // replay writes the same place.
            let argmax_out = self.buffers.scratch();
            for t in 0..m {
                let logits_t = self.buffers.logits().offset(t * vocab * bf16);
                let out_t = argmax_out.offset(t * 4);
                ops::argmax_bf16(
                    self.gpu.as_ref(),
                    self.argmax_kernel,
                    logits_t,
                    out_t,
                    vocab as u32,
                    stream,
                )?;
            }

            if use_graphs {
                let graph = self.gpu.end_capture(stream)?;
                if graph.0 != 0 {
                    tracing::info!(
                        "DFlash fused CUDA graph captured (slot={}, M={})",
                        seq.slot_idx,
                        m
                    );
                    if let Some(ref mut cache) = graph_cache {
                        cache.insert(cache_key, graph);
                    }
                    self.gpu.launch_graph(graph, stream)?;
                }
            }
        }

        let out_ptr = self.buffers.scratch();
        let mut buf = vec![0u8; m * 4];
        self.gpu.copy_d2h(out_ptr, &mut buf)?;
        let result: Vec<u32> = (0..m)
            .map(|t| {
                let b = t * 4;
                u32::from_le_bytes([buf[b], buf[b + 1], buf[b + 2], buf[b + 3]])
            })
            .collect();

        for &t in tokens {
            seq.tokens.push(t);
        }
        seq.seq_len += m;

        Ok(result)
    }
}
