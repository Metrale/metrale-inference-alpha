// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

//! 2026-09-25: `impl TransformerLayer for DeepSeekV41Layer`: decode and
//! prefill both go through `step`; model-level graph capture, multi-sequence
//! decode and decode rollback are declined.
//!
//! Owner: model-arch, DeepSeek-V4.1.
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::{DeepSeekV41Layer, V41LayerState};
use crate::attn_v41::AttnV41LayerState;
use metrale_model_layers::layer::{ForwardContext, LayerState};

impl metrale_model_layers::layer::TransformerLayer for DeepSeekV41Layer {
    #[allow(clippy::too_many_arguments)]
    fn decode(
        &self,
        hidden: DevicePtr,
        _residual: DevicePtr,
        state: &mut dyn LayerState,
        _kv_cache: &mut metrale_cache::kv_cache::PagedKvCache,
        seq_len: usize,
        _block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.step(hidden, 1, seq_len, state, ctx, stream)
    }

    #[allow(clippy::too_many_arguments)]
    fn prefill(
        &self,
        hidden: DevicePtr,
        _residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        _kv_cache: &mut metrale_cache::kv_cache::PagedKvCache,
        seq_len_start: usize,
        _block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        _kv_write_start: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.step(hidden, num_tokens, seq_len_start, state, ctx, stream)
    }

    fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn LayerState>> {
        Ok(Box::new(V41LayerState {
            attn: AttnV41LayerState::new(gpu, &self.rt.attn_cfg, self.role)?,
            graphs: None,
        }))
    }

    /// 2026-09-25: Destroy this sequence's captured segments first (they bake
    /// its buffers), then free the attention buffers `alloc_state` allocated
    /// (`AttnV41LayerState::free`). A state of another type is left alone.
    fn release_state(&self, state: &mut dyn LayerState, gpu: &dyn GpuBackend) -> Result<()> {
        if let Some(st) = state.as_any_mut().downcast_mut::<V41LayerState>() {
            if let Some(g) = st.graphs.take() {
                g.destroy(gpu)?;
            }
            st.attn.free(gpu)?;
        }
        Ok(())
    }
}

impl metrale_model_layers::layer::LayerCapabilities for DeepSeekV41Layer {
    fn decode_graph_unsupported(&self) -> bool {
        true
    }

    fn decode_multi_seq_unsupported(&self) -> bool {
        true
    }

    /// 2026-09-25: Declined: lowering `seq_len` rewinds none of this layer's
    /// state. The window ring and the ratio > 1 compressor state keep rows of
    /// the dropped tokens, and `SharedV41::compress_len` only grows.
    fn decode_rollback_unsupported(&self) -> bool {
        true
    }
}

impl metrale_model_layers::layer::LayerWeightSetup for DeepSeekV41Layer {}
impl metrale_model_layers::layer::LayerWriteOnAccept for DeepSeekV41Layer {}
impl metrale_model_layers::layer::LayerGraphHooks for DeepSeekV41Layer {}
impl metrale_model_layers::layer::LayerAuxState for DeepSeekV41Layer {}
impl metrale_model_layers::layer::LayerSplitPrefill for DeepSeekV41Layer {}
