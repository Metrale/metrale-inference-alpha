// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The attention dispatch of `prefill_attention_paged` for a
//! batched chunk: one launch over every stream, each stream's KV pages through
//! `block_table_ptrs[b]` (staged by `stage_batched_attn_metadata` in
//! model-engine).
//!
//! Only BF16, FP8 and NVFP4 caches at head_dim <= 256 are handled, with the
//! BR=64 kernels when `chunk_len >= 256`. Any other layer returns an error,
//! and the model-engine caller returns that error rather than retrying per
//! stream. The kernels take one `q_offset` (`seq_len_start`) for the whole
//! batch.
//!
//! Owner: model-layers (attention).
//! Invariants: none beyond the types.

#![allow(unused_imports, clippy::too_many_arguments)]

use anyhow::Result;
use metrale_cache::kv_cache::{KvCacheDtype, PagedKvCache};
use metrale_gpu_runtime::gpu::DevicePtr;

use super::super::Qwen3AttentionLayer;
use crate::layer::{BatchedAttnMetadata, ForwardContext};
use crate::layers::ops;

#[allow(dead_code)]
pub(in crate::layers::qwen3_attention) struct PagedAttnBatchedArgs<'a> {
    pub q_contiguous: DevicePtr,
    pub attn_out: DevicePtr,
    pub seq_len_start: usize,
    pub nq: u32,
    pub nkv: u32,
    pub hd: u32,
    pub bs: usize,
    pub inv_sqrt_d: f32,
    pub kv_len: u32,
    pub batched_meta: &'a BatchedAttnMetadata,
    pub stream: u64,
}

impl Qwen3AttentionLayer {
    /// 2026-09-25: Runs batched paged attention over every stream of the batch
    /// into `attn_out`.
    #[allow(dead_code)]
    pub(in crate::layers::qwen3_attention) fn prefill_attention_paged_attn_batched(
        &self,
        kv_cache: &mut PagedKvCache,
        ctx: &ForwardContext,
        args: &PagedAttnBatchedArgs<'_>,
    ) -> Result<()> {
        let PagedAttnBatchedArgs {
            q_contiguous,
            attn_out,
            seq_len_start,
            nq,
            nkv,
            hd,
            bs,
            inv_sqrt_d,
            kv_len,
            batched_meta,
            stream,
        } = *args;
        let bs_u = bs as u32;
        let chunk_len = batched_meta.chunk_len;
        let batch_size = batched_meta.batch_size;
        let block_table_ptrs = batched_meta.block_table_ptrs;
        // 2026-09-25: Per-stream Q offsets (`cu_seqlens`) and KV extents
        // (`kv_lens`). Both are NULL for a same-length batch, where the kernels
        // use `b * chunk_len` and the scalar `kv_len`. `chunk_len` is the
        // longest stream and bounds the grid.
        let cu_seqlens = batched_meta.cu_seqlens;
        let kv_lens = batched_meta.kv_lens;

        // 2026-09-25: head_dim > 256 is an error here;
        // `check_kernel_batched_eligible` keeps such models out of batched
        // prefill, and `prefill_attention_paged` refuses MLA layers.
        if hd > 256 {
            anyhow::bail!(
                "prefill_attention_paged_attn_batched: HDIM={} (>256) not supported \
                 in batched mode (layer {}). Caller should fall back to per-stream.",
                hd,
                self.attn_layer_idx
            );
        }
        // 2026-09-25: The same rule as the `allow_chunk_zero` argument
        // `kernel_batched_eligible` passes (first-chunk lever or varlen). A
        // batch admitted there and refused here fails after its streams were
        // changed.
        let allow_first_chunk = crate::layers::ops::prefill_batched_first_chunk_enabled()
            || crate::layers::ops::prefill_varlen_enabled();
        if seq_len_start == 0 && !allow_first_chunk {
            anyhow::bail!(
                "prefill_attention_paged_attn_batched: seq_len_start=0 not supported \
                 (batched kernels are paged-only; non-paged BR=32 batched kernel is \
                 not yet shipped). Caller should fall back to per-stream."
            );
        }

        let use_br64 = chunk_len >= 256;
        let (fp8_k_scale, fp8_v_scale) = self.effective_fp8_scales();
        let q_offset_u32 = seq_len_start as u32;

        match (self.kv_dtype, use_br64) {
            (KvCacheDtype::Nvfp4, false) => {
                if self.prefill_attn_paged_nvfp4_batched_k.0 == 0 {
                    anyhow::bail!(
                        "prefill_attn_paged_nvfp4_batched kernel not loaded — \
                         rebuild metrale-kernels (commit 4ec2cf2)."
                    );
                }
                ops::prefill_attention_paged_nvfp4_batched(
                    ctx.gpu,
                    self.prefill_attn_paged_nvfp4_batched_k,
                    q_contiguous,
                    kv_cache.k_pool_ptr(self.attn_layer_idx),
                    kv_cache.v_pool_ptr(self.attn_layer_idx),
                    attn_out,
                    block_table_ptrs,
                    batch_size,
                    cu_seqlens,
                    kv_lens,
                    chunk_len,
                    kv_len,
                    q_offset_u32,
                    nq,
                    nkv,
                    hd,
                    bs_u,
                    self.sliding_window.unwrap_or(0),
                    inv_sqrt_d,
                    kv_cache.block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
                    kv_cache.nvfp4_data_bytes() as u64,
                    stream,
                )?;
            }
            (KvCacheDtype::Nvfp4, true) => {
                if self.prefill_attn_paged_nvfp4_batched_64_k.0 == 0 {
                    anyhow::bail!(
                        "prefill_attn_paged_nvfp4_batched_64 kernel not loaded — \
                         rebuild metrale-kernels (commit 4ec2cf2)."
                    );
                }
                ops::prefill_attention_paged_nvfp4_batched_64(
                    ctx.gpu,
                    self.prefill_attn_paged_nvfp4_batched_64_k,
                    q_contiguous,
                    kv_cache.k_pool_ptr(self.attn_layer_idx),
                    kv_cache.v_pool_ptr(self.attn_layer_idx),
                    attn_out,
                    block_table_ptrs,
                    batch_size,
                    cu_seqlens,
                    kv_lens,
                    chunk_len,
                    kv_len,
                    q_offset_u32,
                    nq,
                    nkv,
                    hd,
                    bs_u,
                    self.sliding_window.unwrap_or(0),
                    inv_sqrt_d,
                    kv_cache.block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
                    kv_cache.nvfp4_data_bytes() as u64,
                    stream,
                )?;
            }
            (KvCacheDtype::Fp8, false) => {
                if self.prefill_attn_paged_fp8_batched_k.0 == 0 {
                    anyhow::bail!(
                        "prefill_attn_paged_fp8_batched kernel not loaded — \
                         rebuild metrale-kernels (commit 4ec2cf2)."
                    );
                }
                ops::prefill_attention_paged_fp8_batched(
                    ctx.gpu,
                    self.prefill_attn_paged_fp8_batched_k,
                    q_contiguous,
                    kv_cache.k_pool_ptr(self.attn_layer_idx),
                    kv_cache.v_pool_ptr(self.attn_layer_idx),
                    attn_out,
                    block_table_ptrs,
                    batch_size,
                    cu_seqlens,
                    kv_lens,
                    chunk_len,
                    kv_len,
                    q_offset_u32,
                    nq,
                    nkv,
                    hd,
                    bs_u,
                    self.sliding_window.unwrap_or(0),
                    inv_sqrt_d,
                    fp8_k_scale,
                    fp8_v_scale,
                    kv_cache.block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
                    stream,
                )?;
            }
            (KvCacheDtype::Fp8, true) => {
                if self.prefill_attn_paged_fp8_batched_64_k.0 == 0 {
                    anyhow::bail!(
                        "prefill_attn_paged_fp8_batched_64 kernel not loaded — \
                         rebuild metrale-kernels (commit 4ec2cf2)."
                    );
                }
                ops::prefill_attention_paged_fp8_batched_64(
                    ctx.gpu,
                    self.prefill_attn_paged_fp8_batched_64_k,
                    q_contiguous,
                    kv_cache.k_pool_ptr(self.attn_layer_idx),
                    kv_cache.v_pool_ptr(self.attn_layer_idx),
                    attn_out,
                    block_table_ptrs,
                    batch_size,
                    cu_seqlens,
                    kv_lens,
                    chunk_len,
                    kv_len,
                    q_offset_u32,
                    nq,
                    nkv,
                    hd,
                    bs_u,
                    self.sliding_window.unwrap_or(0),
                    inv_sqrt_d,
                    fp8_k_scale,
                    fp8_v_scale,
                    kv_cache.block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
                    stream,
                )?;
            }
            (KvCacheDtype::Bf16, false) => {
                if self.prefill_attn_paged_batched_k.0 == 0 {
                    anyhow::bail!(
                        "prefill_attn_paged_batched kernel not loaded — \
                         rebuild metrale-kernels (commit 4ec2cf2)."
                    );
                }
                ops::prefill_attention_paged_batched(
                    ctx.gpu,
                    self.prefill_attn_paged_batched_k,
                    q_contiguous,
                    kv_cache.k_pool_ptr(self.attn_layer_idx),
                    kv_cache.v_pool_ptr(self.attn_layer_idx),
                    attn_out,
                    block_table_ptrs,
                    batch_size,
                    cu_seqlens,
                    kv_lens,
                    chunk_len,
                    kv_len,
                    q_offset_u32,
                    nq,
                    nkv,
                    hd,
                    bs_u,
                    self.sliding_window.unwrap_or(0),
                    inv_sqrt_d,
                    stream,
                )?;
            }
            (KvCacheDtype::Bf16, true) => {
                if self.prefill_attn_paged_batched_64_k.0 == 0 {
                    anyhow::bail!(
                        "prefill_attn_paged_batched_64 kernel not loaded — \
                         rebuild metrale-kernels (commit 4ec2cf2)."
                    );
                }
                ops::prefill_attention_paged_batched_64(
                    ctx.gpu,
                    self.prefill_attn_paged_batched_64_k,
                    q_contiguous,
                    kv_cache.k_pool_ptr(self.attn_layer_idx),
                    kv_cache.v_pool_ptr(self.attn_layer_idx),
                    attn_out,
                    block_table_ptrs,
                    batch_size,
                    cu_seqlens,
                    kv_lens,
                    chunk_len,
                    kv_len,
                    q_offset_u32,
                    nq,
                    nkv,
                    hd,
                    bs_u,
                    self.sliding_window.unwrap_or(0),
                    inv_sqrt_d,
                    stream,
                )?;
            }
            (dtype, _) => {
                anyhow::bail!(
                    "prefill_attention_paged_attn_batched: kv_dtype {:?} not yet supported \
                     in batched mode (layer {}). Falls back to per-stream.",
                    dtype,
                    self.attn_layer_idx
                );
            }
        }

        // 2026-09-25: `q_contiguous` and `attn_out` point at every stream's
        // stacked rows; the kernels find a stream's rows from `cu_seqlens`, or at
        // `b * chunk_len` when it is NULL.
        let _ = nq;
        Ok(())
    }
}
