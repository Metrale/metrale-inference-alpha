// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Paged decode attention for one layer, dispatched on the KV dtype: the V4-Flash MLA
//! kernels, split-K or non-split NVFP4, the turbo and asymmetric kernels, and split-K, GQA-packed
//! or non-split BF16 and FP8.
//!
//! Owner: model-layers attention decode.
//! Invariants:
//! - A GQA-packed kernel is launched only on a non-split path.

#![allow(unused_imports)]

use anyhow::Result;
use metrale_cache::kv_cache::{KvCacheDtype, PagedKvCache};
use metrale_cache::kv_dequant::{
    NVFP4_E2M1_LUT, TURBO4_LUT, dequant_4bit_block_to_bf16, dequant_fp8_to_bf16,
    dequant_turbo3_block_to_bf16, dequant_turbo8_block_to_bf16,
};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

mod bf16_fp8;
mod nvfp4_turbo;

impl Qwen3AttentionLayer {
    pub(in super::super) fn run_paged_decode(
        &self,
        gpu: &dyn GpuBackend,
        q: DevicePtr,
        kv_cache: &PagedKvCache,
        output: DevicePtr,
        block_table: DevicePtr,
        seq_lens: DevicePtr,
        max_blocks_per_seq: u32,
        num_seqs: u32,
        num_q_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        block_size: u32,
        inv_sqrt_d: f32,
        q_stride: u32,
        workspace: DevicePtr,
        // 2026-09-25: `ModelLevers::max_decode_seqs`, which the `legacy` split count reads through
        // `split_ref_seqs`.
        max_decode_seqs: u32,
        stream: u64,
    ) -> Result<()> {
        // 2026-09-25: An MLA layer with a rope part (`mla.rope > 0`) is treated as V4-Flash: its
        // `kv_lora_rank + rope`-wide cache rows go to the MLA decode kernels for the NVFP4 and FP8
        // KV dtypes.
        let is_v4_flash = self.mla.as_ref().map(|m| m.rope > 0).unwrap_or(false);

        match self.kv_dtype {
            // 2026-09-25: V4-Flash MLA decode, ahead of the generic arms.
            KvCacheDtype::Nvfp4 if is_v4_flash => {
                let mla = self.mla.as_ref().unwrap();
                let kv_cache_dim = (mla.kv_lora_rank + mla.rope) as u32;
                tracing::info!(
                    "V4-Flash MLA decode (NVFP4): q_head_dim={}, kv_cache_dim={}",
                    head_dim,
                    kv_cache_dim
                );
                ops::mla_paged_decode_nvfp4(
                    gpu,
                    self.mla_paged_decode_k,
                    q,
                    kv_cache.k_pool_ptr(self.attn_layer_idx),
                    kv_cache.v_pool_ptr(self.attn_layer_idx),
                    output,
                    block_table,
                    seq_lens,
                    max_blocks_per_seq,
                    num_q_heads,
                    num_kv_heads,
                    head_dim,
                    kv_cache_dim,
                    block_size,
                    inv_sqrt_d,
                    kv_cache.block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
                    kv_cache.nvfp4_data_bytes() as u64,
                    num_seqs,
                    stream,
                )
            }
            KvCacheDtype::Fp8 if is_v4_flash => {
                let mla = self.mla.as_ref().unwrap();
                let kv_cache_dim = (mla.kv_lora_rank + mla.rope) as u32;
                tracing::info!(
                    "V4-Flash MLA decode (FP8): q_head_dim={}, kv_cache_dim={}",
                    head_dim,
                    kv_cache_dim
                );
                let (k_scale, v_scale) = self.effective_fp8_scales();
                // 2026-09-25: `inv_sqrt_d` is the caller's `effective_attn_scale(head_dim)`, with
                // the query head dim `nope + rope` (448 + 64 = 512 in
                // `kernels/gb10/deepseek-v4-flash/MODEL.toml`), not the 576-wide cache row. The
                // compressed arm is a flat FP8 pool and a block count: a layer without a compressor
                // passes a null pool and 0 blocks; a compressor layer passes blocks `[0,
                // v4_comp_pool_filled)`, written by prefill and by the decode-time append in
                // `attention_forward_v4`.
                let (comp_pool, comp_blocks) = match mla.compressor {
                    Some(c) => (
                        c.pool,
                        self.v4_comp_pool_filled
                            .load(std::sync::atomic::Ordering::Relaxed),
                    ),
                    None => (metrale_gpu_runtime::gpu::DevicePtr::NULL, 0u32),
                };
                ops::mla_paged_decode_fp8(
                    gpu,
                    self.mla_paged_decode_fp8_k,
                    q,
                    kv_cache.k_pool_ptr(self.attn_layer_idx),
                    kv_cache.v_pool_ptr(self.attn_layer_idx),
                    output,
                    block_table,
                    seq_lens,
                    max_blocks_per_seq,
                    num_q_heads,
                    num_kv_heads,
                    head_dim,
                    kv_cache_dim,
                    block_size,
                    inv_sqrt_d,
                    k_scale,
                    v_scale,
                    kv_cache.block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
                    num_seqs,
                    // 2026-09-25: The V4 sliding window; prefill's `V4_WINDOW` (`cache_skip_v4.rs`)
                    // is the same 128.
                    128,
                    self.mla.as_ref().unwrap().attn_sink,
                    comp_pool,
                    comp_blocks,
                    stream,
                )
            }
            KvCacheDtype::Nvfp4 => self.run_paged_decode_nvfp4(
                gpu,
                q,
                kv_cache,
                output,
                block_table,
                seq_lens,
                max_blocks_per_seq,
                num_seqs,
                num_q_heads,
                num_kv_heads,
                head_dim,
                block_size,
                inv_sqrt_d,
                q_stride,
                workspace,
                max_decode_seqs,
                stream,
            ),
            // 2026-09-25: Turbo4, Turbo3 and Turbo2 use the NVFP4 paged-decode interface (block
            // stride and data-section size).
            KvCacheDtype::Turbo4 | KvCacheDtype::Turbo3 | KvCacheDtype::Turbo2 => {
                let kernel = if head_dim > 256 && self.paged_decode_512_k.0 != 0 {
                    self.paged_decode_512_k
                } else {
                    self.paged_decode_k
                };
                let data_bytes = match self.kv_dtype {
                    KvCacheDtype::Turbo3 => kv_cache.turbo3_data_bytes() as u64,
                    KvCacheDtype::Turbo2 => kv_cache.turbo2_data_bytes() as u64,
                    _ => kv_cache.turbo4_data_bytes() as u64,
                };
                ops::paged_decode_attn_nvfp4(
                    gpu,
                    kernel,
                    q,
                    kv_cache.k_pool_ptr(self.attn_layer_idx),
                    kv_cache.v_pool_ptr(self.attn_layer_idx),
                    output,
                    block_table,
                    seq_lens,
                    max_blocks_per_seq,
                    num_seqs,
                    num_q_heads,
                    num_kv_heads,
                    head_dim,
                    block_size,
                    inv_sqrt_d,
                    q_stride,
                    kv_cache.block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
                    data_bytes,
                    stream,
                )
            }
            // 2026-09-25: Turbo8 (FP8 data with per-group BF16 scales), through the same interface.
            KvCacheDtype::Turbo8 => {
                let kernel = if head_dim > 256 && self.paged_decode_512_k.0 != 0 {
                    self.paged_decode_512_k
                } else {
                    self.paged_decode_k
                };
                ops::paged_decode_attn_nvfp4(
                    gpu,
                    kernel,
                    q,
                    kv_cache.k_pool_ptr(self.attn_layer_idx),
                    kv_cache.v_pool_ptr(self.attn_layer_idx),
                    output,
                    block_table,
                    seq_lens,
                    max_blocks_per_seq,
                    num_seqs,
                    num_q_heads,
                    num_kv_heads,
                    head_dim,
                    block_size,
                    inv_sqrt_d,
                    q_stride,
                    kv_cache.block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
                    kv_cache.turbo8_data_bytes() as u64,
                    stream,
                )
            }
            KvCacheDtype::Bf16KTurbo3V => {
                // 2026-09-25: One kernel reads K as BF16 and V as turbo3.
                let sliding = self.sliding_window.unwrap_or(0);
                ops::paged_decode_attn_bf16k_turbo3v(
                    gpu,
                    self.paged_decode_k,
                    q,
                    kv_cache.k_pool_ptr(self.attn_layer_idx),
                    kv_cache.v_pool_ptr(self.attn_layer_idx),
                    output,
                    block_table,
                    seq_lens,
                    max_blocks_per_seq,
                    num_seqs,
                    num_q_heads,
                    num_kv_heads,
                    head_dim,
                    block_size,
                    inv_sqrt_d,
                    q_stride,
                    kv_cache.v_block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
                    kv_cache.turbo3_data_bytes() as u64,
                    sliding,
                    stream,
                )
            }
            KvCacheDtype::Bf16KTurbo4V => {
                // 2026-09-25: One kernel reads K as BF16 and V as turbo4.
                let sliding = self.sliding_window.unwrap_or(0);
                ops::paged_decode_attn_bf16k_turbo4v(
                    gpu,
                    self.paged_decode_k,
                    q,
                    kv_cache.k_pool_ptr(self.attn_layer_idx),
                    kv_cache.v_pool_ptr(self.attn_layer_idx),
                    output,
                    block_table,
                    seq_lens,
                    max_blocks_per_seq,
                    num_seqs,
                    num_q_heads,
                    num_kv_heads,
                    head_dim,
                    block_size,
                    inv_sqrt_d,
                    q_stride,
                    kv_cache.v_block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
                    kv_cache.nvfp4_data_bytes() as u64,
                    sliding,
                    stream,
                )
            }
            KvCacheDtype::Bf16KTurbo2V => {
                // 2026-09-25: One kernel reads K as BF16 and V as turbo2.
                let sliding = self.sliding_window.unwrap_or(0);
                ops::paged_decode_attn_bf16k_turbo2v(
                    gpu,
                    self.paged_decode_k,
                    q,
                    kv_cache.k_pool_ptr(self.attn_layer_idx),
                    kv_cache.v_pool_ptr(self.attn_layer_idx),
                    output,
                    block_table,
                    seq_lens,
                    max_blocks_per_seq,
                    num_seqs,
                    num_q_heads,
                    num_kv_heads,
                    head_dim,
                    block_size,
                    inv_sqrt_d,
                    q_stride,
                    kv_cache.v_block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
                    kv_cache.turbo2_data_bytes() as u64,
                    sliding,
                    stream,
                )
            }
            KvCacheDtype::Turbo4KTurbo3V
            | KvCacheDtype::Turbo4KTurbo8V
            | KvCacheDtype::Turbo3KTurbo8V => self.run_paged_decode_turbo_kv(
                gpu,
                q,
                kv_cache,
                output,
                block_table,
                seq_lens,
                max_blocks_per_seq,
                num_seqs,
                num_q_heads,
                num_kv_heads,
                head_dim,
                block_size,
                inv_sqrt_d,
                q_stride,
                stream,
            ),
            KvCacheDtype::Fp8KTurbo3V | KvCacheDtype::Fp8KTurbo4V | KvCacheDtype::Fp8KTurbo2V => {
                self.run_paged_decode_fp8k_turbo_v(
                    gpu,
                    q,
                    kv_cache,
                    output,
                    block_table,
                    seq_lens,
                    max_blocks_per_seq,
                    num_seqs,
                    num_q_heads,
                    num_kv_heads,
                    head_dim,
                    block_size,
                    inv_sqrt_d,
                    q_stride,
                    stream,
                )
            }
            KvCacheDtype::Bf16 => self.run_paged_decode_bf16(
                gpu,
                q,
                kv_cache,
                output,
                block_table,
                seq_lens,
                max_blocks_per_seq,
                num_seqs,
                num_q_heads,
                num_kv_heads,
                head_dim,
                block_size,
                inv_sqrt_d,
                q_stride,
                workspace,
                max_decode_seqs,
                stream,
            ),
            _ => self.run_paged_decode_fp8(
                gpu,
                q,
                kv_cache,
                output,
                block_table,
                seq_lens,
                max_blocks_per_seq,
                num_seqs,
                num_q_heads,
                num_kv_heads,
                head_dim,
                block_size,
                inv_sqrt_d,
                q_stride,
                workspace,
                max_decode_seqs,
                stream,
            ),
        }
    }
}
