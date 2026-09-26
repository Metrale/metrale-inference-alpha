// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `LayerSplitPrefill`, the phased and batched prefill entry points: the SSM phase 1
//! pieces, the GDN recurrence per stream, batched and varlen, phase 3, and the batched
//! attention prefill.
//!
//! Owner: model-layers.
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_cache::kv_cache::PagedKvCache;
use metrale_gpu_runtime::gpu::DevicePtr;

use crate::layer::{BatchedAttnMetadata, ForwardContext, GdnPrefillBuffers, LayerState};

/// 2026-09-26: A supertrait of `TransformerLayer`; see the module header.
pub trait LayerSplitPrefill {
    /// 2026-09-25: Phase 1 projections (norm, QKVZ, gates) over all stacked tokens of a
    /// batched prefill at once. The caller then runs `prefill_phase1_conv1d_one` per
    /// request and `prefill_phase1_l2_batched`. The default returns an error.
    fn prefill_phase1_proj_batched(
        &self,
        hidden_stacked: DevicePtr,
        residual_stacked: DevicePtr,
        total_tokens: usize,
        gdn_bufs: &GdnPrefillBuffers,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let _ = (
            hidden_stacked,
            residual_stacked,
            total_tokens,
            gdn_bufs,
            ctx,
            stream,
        );
        anyhow::bail!("prefill_phase1_proj_batched: only implemented for SSM layers")
    }

    /// 2026-09-25: Conv1d over one request's tokens of a batched prefill, advancing its
    /// conv state and writing its slice of `gdn_bufs.qkv`. The default returns an error.
    fn prefill_phase1_conv1d_one(
        &self,
        state: &mut dyn LayerState,
        token_offset: usize,
        len: usize,
        gdn_bufs: &GdnPrefillBuffers,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let _ = (state, token_offset, len, gdn_bufs, ctx, stream);
        anyhow::bail!("prefill_phase1_conv1d_one: only implemented for SSM layers")
    }

    /// 2026-09-25: L2 norm over the whole stacked `gdn_bufs.qkv`, after every request's
    /// conv1d. The default returns an error.
    fn prefill_phase1_l2_batched(
        &self,
        total_tokens: usize,
        gdn_bufs: &GdnPrefillBuffers,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let _ = (total_tokens, gdn_bufs, ctx, stream);
        anyhow::bail!("prefill_phase1_l2_batched: only implemented for SSM layers")
    }

    /// 2026-09-25: Split SSM prefill, phase 2: the GDN recurrence over all
    /// `gdn_bufs.total_len` tokens, reading packed QKV and gate/beta and writing `output`.
    /// Default: nothing.
    fn prefill_gdn_full(
        &self,
        _state: &mut dyn LayerState,
        _gdn_bufs: &GdnPrefillBuffers,
        _ctx: &ForwardContext,
        _stream: u64,
    ) -> Result<()> {
        Ok(())
    }

    /// 2026-09-25: Prefill of one attention layer over the stacked tokens of several
    /// streams, with per-stream metadata from `batched_meta`. The default returns an
    /// error; the Qwen3 attention layer overrides it.
    fn prefill_inner_batched_q12(
        &self,
        _hidden_stacked: DevicePtr,
        _residual_stacked: DevicePtr,
        _num_tokens: usize,
        _kv_cache: &mut PagedKvCache,
        _seq_len_start: usize,
        _batched_meta: &BatchedAttnMetadata,
        _ctx: &ForwardContext,
        _stream: u64,
    ) -> Result<()> {
        anyhow::bail!("prefill_inner_batched_q12: not implemented for this layer type")
    }

    /// 2026-09-25: GDN recurrence for `batch_size` streams of equal length `chunk_len` in
    /// one call. `h_state_ptrs` is a device array of one h-state pointer per stream
    /// (`TransformerModel::stage_h_state_ptrs`), and the streams' tokens lie back to back
    /// in `gdn_bufs`. `prefill_ssm_batched_layer` calls it only when every stream has
    /// the same length. The default returns an error.
    fn prefill_gdn_full_batched(
        &self,
        _h_state_ptrs: DevicePtr,
        _gdn_bufs: &GdnPrefillBuffers,
        _batch_size: u32,
        _chunk_len: u32,
        _ctx: &ForwardContext,
        _stream: u64,
    ) -> Result<()> {
        anyhow::bail!(
            "prefill_gdn_full_batched: layer does not implement batched GDN \
             — caller should fall back to per-stream prefill_gdn_full"
        )
    }

    /// 2026-09-25: GDN recurrence for streams of different lengths in one FLA call,
    /// driven by `cu_seqlens`. `Ok(true)` if it ran; `Ok(false)` if not eligible, and the
    /// caller then runs `prefill_gdn_full` per stream. Default `Ok(false)`.
    #[allow(clippy::too_many_arguments)]
    fn prefill_gdn_full_batched_fla_varlen(
        &self,
        _h_state_ptrs: DevicePtr,
        _gdn_bufs: &GdnPrefillBuffers,
        _batch_size: u32,
        _cu_seqlens: DevicePtr,
        _max_num_chunks: u32,
        _total_nt: usize,
        _max_seqlen: u32,
        _ctx: &ForwardContext,
        _stream: u64,
    ) -> Result<bool> {
        Ok(false)
    }

    /// 2026-09-25: Split SSM prefill, phase 3: for `num_tokens` tokens at `token_offset`,
    /// gated RMSNorm of the GDN output with Z, the output projection, the residual add and
    /// the FFN. Default: nothing.
    #[allow(clippy::too_many_arguments)]
    fn prefill_phase3(
        &self,
        _hidden: DevicePtr,
        _residual: DevicePtr,
        _num_tokens: usize,
        _gdn_bufs: &GdnPrefillBuffers,
        _token_offset: usize,
        _ctx: &ForwardContext,
        _stream: u64,
    ) -> Result<()> {
        Ok(())
    }
}
