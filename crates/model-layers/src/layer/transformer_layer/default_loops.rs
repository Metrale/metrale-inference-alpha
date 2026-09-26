// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The per-token fallbacks behind the default `TransformerLayer::prefill`,
//! `decode_batched` and `decode_multi_seq`: each calls the layer's `decode` once per
//! token or sequence.
//!
//! Owner: model-layers.
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_cache::kv_cache::PagedKvCache;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::TransformerLayer;
use crate::layer::{ForwardContext, LayerState};

#[allow(clippy::too_many_arguments)]
pub(super) fn prefill_default(
    layer: &(impl TransformerLayer + ?Sized),
    hidden: DevicePtr,
    residual: DevicePtr,
    num_tokens: usize,
    state: &mut dyn LayerState,
    kv_cache: &mut PagedKvCache,
    seq_len_start: usize,
    block_table: &mut Vec<u32>,
    disk_block_ids: &mut Vec<u32>,
    disk_last_offloaded_per_layer: &mut Vec<u32>,
    ctx: &ForwardContext,
    stream: u64,
) -> Result<()> {
    let h = ctx.config.hidden_size;
    for t in 0..num_tokens {
        let offset = t * h * 2;
        let h_t = hidden.offset(offset);
        let r_t = residual.offset(offset);
        layer.decode(
            h_t,
            r_t,
            state,
            kv_cache,
            seq_len_start + t,
            block_table,
            disk_block_ids,
            disk_last_offloaded_per_layer,
            ctx,
            stream,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn decode_batched_default(
    layer: &(impl TransformerLayer + ?Sized),
    hidden: DevicePtr,
    residual: DevicePtr,
    num_tokens: usize,
    state: &mut dyn LayerState,
    kv_cache: &mut PagedKvCache,
    seq_len: usize,
    block_table: &mut Vec<u32>,
    disk_block_ids: &mut Vec<u32>,
    disk_last_offloaded_per_layer: &mut Vec<u32>,
    ctx: &ForwardContext,
    stream: u64,
) -> Result<()> {
    let h = ctx.config.hidden_size;
    for t in 0..num_tokens {
        let offset = (t * h * 2) as u64;
        let h_t = hidden.offset(offset as usize);
        let r_t = residual.offset(offset as usize);
        layer.decode(
            h_t,
            r_t,
            state,
            kv_cache,
            seq_len + t,
            block_table,
            disk_block_ids,
            disk_last_offloaded_per_layer,
            ctx,
            stream,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn decode_multi_seq_default<'a, 'b: 'a>(
    layer: &(impl TransformerLayer + ?Sized),
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
    let h = ctx.config.hidden_size;
    for i in 0..num_seqs {
        let offset = i * h * 2;
        let h_i = hidden.offset(offset);
        let r_i = residual.offset(offset);
        let mut bt = block_tables[i].clone();
        // 2026-09-25: This fallback passes empty disk-block vectors to `decode`. The
        // Qwen3 attention layer, which reads them, overrides `decode_multi_seq`.
        let mut stub_disk = Vec::<u32>::new();
        let mut stub_last_offloaded = Vec::<u32>::new();
        layer.decode(
            h_i,
            r_i,
            states[i],
            kv_cache,
            seq_lens[i],
            &mut bt,
            &mut stub_disk,
            &mut stub_last_offloaded,
            ctx,
            stream,
        )?;
    }
    Ok(())
}
