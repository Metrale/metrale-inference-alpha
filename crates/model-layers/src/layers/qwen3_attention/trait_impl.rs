// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `TransformerLayer` for `Qwen3AttentionLayer`: forwards decode,
//! prefill and multi-sequence decode to their inner implementations, carries the
//! QSA indexer's per-sequence state, and runs the MoE transposes on both FFN slots.
//!
//! Owner: model-layers (qwen3 attention).
//! Invariants:
//! - `decode_graph_unsupported` and `has_aux_state` are true exactly when the
//!   layer has a QSA indexer.
//! - `release_state` leaves `AttnLayerState::qsa` empty on every return, `Err`
//!   included: the carry is taken before the fallible free.

use anyhow::Result;
use metrale_cache::kv_cache::PagedKvCache;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::Qwen3AttentionLayer;
use crate::layer::{
    BatchedAttnMetadata, EmptyLayerState, ForwardContext, LayerState, TransformerLayer,
};
use crate::layer::{
    LayerAuxState, LayerCapabilities, LayerGraphHooks, LayerSplitPrefill, LayerWeightSetup,
    LayerWriteOnAccept,
};
use crate::layers::FfnComponent;

mod decode_inner;
mod multi_seq;
mod prefill_inner;

/// 2026-09-25: Debug: synchronise `stream`, read a BF16 tensor back and log its
/// L2 norm, max |x| and first four values. A failed copy logs nothing.
pub(super) fn diag_norm(
    gpu: &dyn GpuBackend,
    ptr: DevicePtr,
    n_elements: usize,
    stream: u64,
    label: &str,
) {
    let _ = gpu.synchronize(stream);
    let mut buf = vec![0u16; n_elements];
    // 2026-09-25: SAFETY: `buf` is `vec![0u16; n_elements]` on the line above, so
    // `buf.len() == n_elements` and `n_elements * 2 == buf.len() *
    // size_of::<u16>()` — the span is exactly the Vec's buffer, all of it
    // zero-initialised. `bytes` is the sole reference derived from `buf` while it
    // is live: it is dead after the `copy_d2h` below, before `buf.iter()` runs.
    let bytes =
        unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, n_elements * 2) };
    if gpu.copy_d2h(ptr, bytes).is_err() {
        return;
    }
    let vals: Vec<f32> = buf
        .iter()
        .map(|&b| f32::from_bits((b as u32) << 16))
        .collect();
    let norm: f32 = vals.iter().map(|v| v * v).sum::<f32>().sqrt();
    let max_abs: f32 = vals.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let f4 = if vals.len() >= 4 {
        format!(
            "[{:.4},{:.4},{:.4},{:.4}]",
            vals[0], vals[1], vals[2], vals[3]
        )
    } else {
        format!("{:?}", &vals[..vals.len().min(4)])
    };
    tracing::info!("DIAG {label}: norm={norm:.4} max={max_abs:.4} first4={f4} n={n_elements}");
}

/// 2026-09-25: Debug: the FP32 twin of `diag_norm`. Its callers are the
/// DeepSeek-V4 prefill and multi-sequence decode diagnostics, which log the
/// hyper-connection `post` and `comb` tensors.
pub fn diag_norm_f32(
    gpu: &dyn GpuBackend,
    ptr: DevicePtr,
    n_elements: usize,
    stream: u64,
    label: &str,
) {
    let _ = gpu.synchronize(stream);
    let mut buf = vec![0f32; n_elements];
    // 2026-09-25: SAFETY: `buf` is `vec![0f32; n_elements]` on the line above, so
    // `buf.len() == n_elements` and `n_elements * 4 == buf.len() *
    // size_of::<f32>()` — the span is exactly the Vec's buffer, all of it
    // zero-initialised. `bytes` is the sole reference derived from `buf` while it
    // is live: it is dead after the `copy_d2h` below, before `buf.iter()` runs.
    let bytes =
        unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, n_elements * 4) };
    if gpu.copy_d2h(ptr, bytes).is_err() {
        return;
    }
    let norm: f32 = buf.iter().map(|v| v * v).sum::<f32>().sqrt();
    let max_abs: f32 = buf.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let f4 = if buf.len() >= 4 {
        format!("[{:.4},{:.4},{:.4},{:.4}]", buf[0], buf[1], buf[2], buf[3])
    } else {
        format!("{:?}", &buf[..buf.len().min(4)])
    };
    tracing::info!(
        "DIAG {label}: norm={norm:.4} max={max_abs:.4} first4={f4} n={n_elements} (FP32)"
    );
}

impl TransformerLayer for Qwen3AttentionLayer {
    fn decode(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.decode_inner(
            hidden,
            residual,
            state,
            kv_cache,
            seq_len,
            block_table,
            disk_block_ids,
            disk_last_offloaded_per_layer,
            ctx,
            stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn prefill(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len_start: usize,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        kv_write_start: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.prefill_inner(
            hidden,
            residual,
            num_tokens,
            state,
            kv_cache,
            seq_len_start,
            block_table,
            disk_block_ids,
            disk_last_offloaded_per_layer,
            kv_write_start,
            None,
            ctx,
            stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_multi_seq<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_seqs: usize,
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        kv_cache: &mut PagedKvCache,
        seq_lens: &[usize],
        block_tables: &[Vec<u32>],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.decode_multi_seq_inner(
            hidden,
            residual,
            num_seqs,
            states,
            kv_cache,
            seq_lens,
            block_tables,
            ctx,
            stream,
        )
    }

    fn alloc_state(&self, _gpu: &dyn GpuBackend) -> Result<Box<dyn LayerState>> {
        Ok(Box::new(crate::layer::AttnLayerState::default()))
    }

    /// 2026-09-25: Free the QSA indexer carry this sequence attached on first use.
    ///
    /// `alloc_state` returns an empty `AttnLayerState`; the carry is created
    /// later by `helpers::qsa_seq_state` (or by `restore_aux`). `take()` makes
    /// this idempotent and leaves the state as `alloc_state` produced it.
    ///
    /// A layer with no QSA indexer never populates the field, so `take()`
    /// yields `None` and this returns `Ok(())`.
    fn release_state(&self, state: &mut dyn LayerState, gpu: &dyn GpuBackend) -> Result<()> {
        let Some(attn) = state
            .as_any_mut()
            .downcast_mut::<crate::layer::AttnLayerState>()
        else {
            return Ok(());
        };
        let Some(mut st) = attn.qsa.take() else {
            return Ok(());
        };
        let Some(qsa) = self.qsa.as_ref() else {
            // 2026-09-25: A carry without an indexer cannot be sized or freed,
            // so this is an error rather than a guess.
            anyhow::bail!("release_state: QSA seq state present but layer has no QSA indexer");
        };
        qsa.release_seq_state(&mut st, gpu)
    }
}

impl LayerCapabilities for Qwen3AttentionLayer {
    fn uses_local_mla_prefill(&self) -> bool {
        self.mla.is_some()
    }

    fn fp8_calibration_frozen(&self) -> Option<bool> {
        self.fp8_calibration
            .as_ref()
            .map(|cal| !cal.is_calibrating())
    }

    /// 2026-09-25: QSA selection copies the scores to the host and sorts them
    /// there (`qsa_select.rs`), which a captured graph cannot contain. Inside the
    /// inert bound the layer runs dense attention, so a graph captured then
    /// would replay dense attention after selection starts.
    fn decode_graph_unsupported(&self) -> bool {
        self.qsa.is_some()
    }
}

impl LayerWeightSetup for Qwen3AttentionLayer {
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }

    fn transpose_moe_for_prefill(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &metrale_config::ModelConfig,
    ) -> Result<()> {
        if let FfnComponent::Moe(moe) = &mut self.ffn {
            moe.transpose_for_prefill(gpu, config)?;
        }
        if let Some(FfnComponent::Moe(moe)) = self.moe_ffn.as_mut() {
            moe.transpose_for_prefill(gpu, config)?;
        }
        Ok(())
    }

    fn transpose_moe_gate_up_for_prefill(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &metrale_config::ModelConfig,
    ) -> Result<()> {
        if let FfnComponent::Moe(moe) = &mut self.ffn {
            moe.transpose_gate_up_for_prefill(gpu, config)?;
        }
        if let Some(FfnComponent::Moe(moe)) = self.moe_ffn.as_mut() {
            moe.transpose_gate_up_for_prefill(gpu, config)?;
        }
        Ok(())
    }

    fn set_moe_down_transpose_scratch(
        &mut self,
        scratch_packed: DevicePtr,
        scratch_scale: DevicePtr,
        packed_ptrs_t: DevicePtr,
        scale_ptrs_t: DevicePtr,
    ) {
        if let FfnComponent::Moe(moe) = &mut self.ffn {
            moe.set_down_transpose_scratch(
                scratch_packed,
                scratch_scale,
                packed_ptrs_t,
                scale_ptrs_t,
            );
        }
        if let Some(FfnComponent::Moe(moe)) = self.moe_ffn.as_mut() {
            moe.set_down_transpose_scratch(
                scratch_packed,
                scratch_scale,
                packed_ptrs_t,
                scale_ptrs_t,
            );
        }
    }

    fn transpose_moe_for_prefill_unified(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &metrale_config::ModelConfig,
    ) -> Result<()> {
        if let FfnComponent::Moe(moe) = &mut self.ffn {
            moe.transpose_for_prefill_unified(gpu, config)?;
        }
        if let Some(FfnComponent::Moe(moe)) = self.moe_ffn.as_mut() {
            moe.transpose_for_prefill_unified(gpu, config)?;
        }
        Ok(())
    }

    fn transpose_moe_for_prefill_hybrid(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &metrale_config::ModelConfig,
    ) -> Result<()> {
        if let FfnComponent::Moe(moe) = &mut self.ffn {
            moe.transpose_for_prefill_hybrid(gpu, config)?;
        }
        if let Some(FfnComponent::Moe(moe)) = self.moe_ffn.as_mut() {
            moe.transpose_for_prefill_hybrid(gpu, config)?;
        }
        Ok(())
    }
}

impl LayerWriteOnAccept for Qwen3AttentionLayer {}
impl LayerGraphHooks for Qwen3AttentionLayer {}

impl LayerAuxState for Qwen3AttentionLayer {
    fn has_aux_state(&self) -> bool {
        self.qsa.is_some()
    }

    fn snapshot_aux(
        &self,
        state: &dyn LayerState,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<Option<Vec<u8>>> {
        let Some(qsa) = self.qsa.as_ref() else {
            return Ok(None);
        };
        let attn = state
            .as_any()
            .downcast_ref::<crate::layer::AttnLayerState>()
            .ok_or_else(|| anyhow::anyhow!("QSA host layer state is not AttnLayerState"))?;
        match attn.qsa.as_ref() {
            Some(st) => Ok(Some(qsa.snapshot_aux(st, gpu, stream)?)),
            // 2026-09-25: The carry is created on the sequence's first QSA use
            // (`helpers::qsa_seq_state`); without it there is nothing to carry.
            None => Ok(None),
        }
    }

    fn restore_aux(
        &self,
        state: &mut dyn LayerState,
        blob: &[u8],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let qsa = self
            .qsa
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("restore_aux: no QSA on this layer"))?;
        let attn = state
            .as_any_mut()
            .downcast_mut::<crate::layer::AttnLayerState>()
            .ok_or_else(|| anyhow::anyhow!("QSA host layer state is not AttnLayerState"))?;
        if attn.qsa.is_none() {
            attn.qsa = Some(qsa.new_seq_state(gpu)?);
        }
        qsa.restore_aux(attn.qsa.as_mut().expect("just created"), blob, gpu, stream)
    }
}

impl LayerSplitPrefill for Qwen3AttentionLayer {
    /// 2026-09-25: Batched attention prefill over stacked streams: `prefill_inner`
    /// with `batched_meta = Some`, an empty layer state and empty per-stream
    /// block and disk lists. The streams' block tables travel in
    /// `BatchedAttnMetadata::block_table_ptrs`. The caller is the model's
    /// `prefill_attn_batched_layer`.
    fn prefill_inner_batched_q12(
        &self,
        hidden_stacked: DevicePtr,
        residual_stacked: DevicePtr,
        num_tokens: usize,
        kv_cache: &mut PagedKvCache,
        seq_len_start: usize,
        batched_meta: &BatchedAttnMetadata,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let mut empty_state = EmptyLayerState;
        let mut empty_block_table: Vec<u32> = Vec::new();
        let mut empty_disk_block_ids: Vec<u32> = Vec::new();
        let mut empty_disk_last: Vec<u32> = Vec::new();
        self.prefill_inner(
            hidden_stacked,
            residual_stacked,
            num_tokens,
            &mut empty_state,
            kv_cache,
            seq_len_start,
            &mut empty_block_table,
            &mut empty_disk_block_ids,
            &mut empty_disk_last,
            0,
            Some(batched_meta),
            ctx,
            stream,
        )
    }
}
