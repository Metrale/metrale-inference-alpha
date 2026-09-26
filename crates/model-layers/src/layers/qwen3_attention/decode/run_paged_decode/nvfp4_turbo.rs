// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The NVFP4 and mixed-turbo arms of `run_paged_decode`: split-K or non-split NVFP4,
//! turbo K with a different turbo V, and FP8 K with turbo V.
//!
//! Owner: model-layers attention decode.
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_cache::kv_cache::{KvCacheDtype, PagedKvCache};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::super::super::Qwen3AttentionLayer;
use super::super::splitk_dispatch;
use crate::layers::ops;

impl Qwen3AttentionLayer {
    /// 2026-09-26: The `KvCacheDtype::Nvfp4` arm (not V4-Flash).
    pub(super) fn run_paged_decode_nvfp4(
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
        max_decode_seqs: u32,
        stream: u64,
    ) -> Result<()> {
        // 2026-09-25: The split count comes from `splitk_dispatch::num_splits`, which does
        // not move with the co-batched count under `auto` or a pinned count, and under
        // `legacy` only when `num_seqs` exceeds `max_decode_seqs`.
        let num_splits =
            splitk_dispatch::num_splits(num_q_heads, head_dim, num_seqs, max_decode_seqs);

        if splitk_dispatch::splits_are_worth_it(num_splits) {
            splitk_dispatch::log_decode_route(
                splitk_dispatch::RouteArm::Nvfp4,
                splitk_dispatch::ROUTE_SPLITK_NVFP4,
                num_splits,
            );
            let splitk_k = self
                .paged_decode_splitk_k
                .expect("split-K kernel required for NVFP4");
            let reduce_k = self
                .paged_decode_reduce_k
                .expect("reduce kernel required for NVFP4");
            ops::paged_decode_attn_splitk_nvfp4(
                gpu,
                splitk_k,
                q,
                kv_cache.k_pool_ptr(self.attn_layer_idx),
                kv_cache.v_pool_ptr(self.attn_layer_idx),
                workspace,
                block_table,
                seq_lens,
                max_blocks_per_seq,
                num_q_heads,
                num_kv_heads,
                head_dim,
                block_size,
                inv_sqrt_d,
                num_splits,
                q_stride,
                kv_cache.block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
                kv_cache.nvfp4_data_bytes() as u64,
                num_seqs,
                stream,
            )?;
            ops::paged_decode_attn_reduce_nvfp4(
                gpu,
                reduce_k,
                workspace,
                output,
                seq_lens,
                num_q_heads,
                head_dim,
                num_splits,
                num_seqs,
                stream,
            )
        } else {
            splitk_dispatch::log_decode_route(
                splitk_dispatch::RouteArm::Nvfp4,
                splitk_dispatch::ROUTE_NONSPLIT_NVFP4,
                num_splits,
            );
            ops::paged_decode_attn_nvfp4(
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
                kv_cache.block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
                kv_cache.nvfp4_data_bytes() as u64,
                stream,
            )
        }
    }

    /// 2026-09-26: The `Turbo4KTurbo3V`, `Turbo4KTurbo8V` and `Turbo3KTurbo8V` arm.
    pub(super) fn run_paged_decode_turbo_kv(
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
        stream: u64,
    ) -> Result<()> {
        // 2026-09-25: K and V are different turbo dtypes, so each side passes its own block
        // stride and data-section size.
        let sliding = self.sliding_window.unwrap_or(0);
        let k_block_stride = kv_cache.k_block_stride_bytes_for_layer(self.attn_layer_idx) as u64;
        let v_block_stride = kv_cache.v_block_stride_bytes_for_layer(self.attn_layer_idx) as u64;
        let k_pool = kv_cache.k_pool_ptr(self.attn_layer_idx);
        let v_pool = kv_cache.v_pool_ptr(self.attn_layer_idx);
        match self.kv_dtype {
            KvCacheDtype::Turbo4KTurbo3V => ops::paged_decode_attn_turbo4k_turbo3v(
                gpu,
                self.paged_decode_k,
                q,
                k_pool,
                v_pool,
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
                k_block_stride,
                kv_cache.nvfp4_data_bytes() as u64,
                v_block_stride,
                kv_cache.turbo3_data_bytes() as u64,
                sliding,
                stream,
            ),
            KvCacheDtype::Turbo4KTurbo8V => ops::paged_decode_attn_turbo4k_turbo8v(
                gpu,
                self.paged_decode_k,
                q,
                k_pool,
                v_pool,
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
                k_block_stride,
                kv_cache.nvfp4_data_bytes() as u64,
                v_block_stride,
                kv_cache.turbo8_data_bytes() as u64,
                sliding,
                stream,
            ),
            KvCacheDtype::Turbo3KTurbo8V => ops::paged_decode_attn_turbo3k_turbo8v(
                gpu,
                self.paged_decode_k,
                q,
                k_pool,
                v_pool,
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
                k_block_stride,
                kv_cache.turbo3_data_bytes() as u64,
                v_block_stride,
                kv_cache.turbo8_data_bytes() as u64,
                sliding,
                stream,
            ),
            _ => unreachable!(),
        }
    }

    /// 2026-09-26: The `Fp8KTurbo3V`, `Fp8KTurbo4V` and `Fp8KTurbo2V` arm.
    pub(super) fn run_paged_decode_fp8k_turbo_v(
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
        stream: u64,
    ) -> Result<()> {
        // 2026-09-25: K as FP8 with the per-tensor `k_scale`, V as turbo2, 3 or 4.
        let sliding = self.sliding_window.unwrap_or(0);
        let (k_scale, _) = self.effective_fp8_scales();
        let v_block_stride = kv_cache.v_block_stride_bytes_for_layer(self.attn_layer_idx) as u64;
        let k_pool = kv_cache.k_pool_ptr(self.attn_layer_idx);
        let v_pool = kv_cache.v_pool_ptr(self.attn_layer_idx);
        match self.kv_dtype {
            KvCacheDtype::Fp8KTurbo3V => ops::paged_decode_attn_fp8k_turbo3v(
                gpu,
                self.paged_decode_k,
                q,
                k_pool,
                v_pool,
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
                k_scale,
                q_stride,
                v_block_stride,
                kv_cache.turbo3_data_bytes() as u64,
                sliding,
                stream,
            ),
            KvCacheDtype::Fp8KTurbo4V => ops::paged_decode_attn_fp8k_turbo4v(
                gpu,
                self.paged_decode_k,
                q,
                k_pool,
                v_pool,
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
                k_scale,
                q_stride,
                v_block_stride,
                kv_cache.nvfp4_data_bytes() as u64,
                sliding,
                stream,
            ),
            KvCacheDtype::Fp8KTurbo2V => ops::paged_decode_attn_fp8k_turbo2v(
                gpu,
                self.paged_decode_k,
                q,
                k_pool,
                v_pool,
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
                k_scale,
                q_stride,
                v_block_stride,
                kv_cache.turbo2_data_bytes() as u64,
                sliding,
                stream,
            ),
            _ => unreachable!(),
        }
    }
}
