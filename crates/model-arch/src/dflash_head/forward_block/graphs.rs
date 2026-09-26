// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The paged layer calls of `forward_block`: each layer's `PagedLayerArgs` and the
//! replay of the cached CUDA subgraphs for one block width.
//!
//! Owner: model-arch (DFlash drafter).
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GraphHandle};
use metrale_model_layers::layer::ForwardContext;

use super::dims::BlockDims;
use crate::dflash_head::BlockDiffusionDraftHead;
use crate::dflash_head::forward_block_layer_paged::PagedLayerArgs;

/// 2026-09-26: The paged layer body's inputs for layer `layer_idx`, or `None` off the paged
/// path or before its cache slots are built.
pub(super) fn paged_layer_args(
    d: &BlockDims<'_>,
    layer_idx: usize,
    slot_mapping_gamma_opt: Option<DevicePtr>,
    stream: u64,
    block_dump_armed: bool,
) -> Option<PagedLayerArgs> {
    let BlockDims {
        n_seq,
        h: h_local,
        q_dim: q_dim_local,
        kv_dim: kv_dim_local,
        inter: inter_local,
        inv_sqrt_d: inv_sqrt_d_local,
        option_b_block_table,
        option_b_ctx_count,
        option_b_on,
        batch,
        ..
    } = *d;
    if !option_b_on {
        return None;
    }
    let bt = option_b_block_table?;
    let slot_mapping = slot_mapping_gamma_opt?;
    Some(PagedLayerArgs {
        layer_idx,
        ctx_count: option_b_ctx_count,
        h: h_local,
        q_dim: q_dim_local,
        kv_dim: kv_dim_local,
        inter: inter_local,
        inv_sqrt_d: inv_sqrt_d_local,
        slot_mapping_gamma: slot_mapping,
        block_table_dev: bt,
        stream,
        block_dump: block_dump_armed,
        n_seq: n_seq as u32,
        seq_block_tables: batch.map(|x| x.block_tables.clone()).unwrap_or_default(),
    })
}

impl BlockDiffusionDraftHead {
    /// 2026-09-26: Replays `graphs` (`[pre_0, post_0, ..., tail]`) with attention run eagerly
    /// between each layer's pre and post; a zero handle runs that part eagerly.
    pub(super) fn replay_block_graphs(
        &self,
        ctx: &ForwardContext,
        stream: u64,
        graphs: &[GraphHandle],
        tail_slot: usize,
        make_paged_args: &impl Fn(usize) -> Option<PagedLayerArgs>,
        run_tail: &impl Fn() -> Result<()>,
    ) -> Result<()> {
        let gpu = ctx.gpu;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let args = make_paged_args(layer_idx).expect("option_b args available");

            let pre_handle = graphs[layer_idx * 2];
            if pre_handle.0 != 0 {
                gpu.launch_graph(pre_handle, stream)?;
            } else {
                // 2026-09-25: A zero handle marks a capture that came back
                // empty; that slot always runs eagerly.
                self.forward_block_layer_pre_attn(layer, &args, ctx)?;
            }

            // 2026-09-25: A replayed pre-attention subgraph returns no pool
            // pointers, so they are read from the cache here.
            let (k_pool, v_pool) = {
                let cache = self.kv_cache.lock();
                (cache.k_pool_ptr(layer_idx), cache.v_pool_ptr(layer_idx))
            };
            self.forward_block_layer_attention(&args, ctx, k_pool, v_pool)?;

            let post_handle = graphs[layer_idx * 2 + 1];
            if post_handle.0 != 0 {
                gpu.launch_graph(post_handle, stream)?;
            } else {
                self.forward_block_layer_post_attn(layer, &args, ctx)?;
            }
        }

        let tail_handle = graphs[tail_slot];
        if tail_handle.0 != 0 {
            gpu.launch_graph(tail_handle, stream)?;
        } else {
            run_tail()?;
        }
        Ok(())
    }
}
