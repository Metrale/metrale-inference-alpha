// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The BF16 and FP8 arms of `run_paged_decode`: split-K, GQA-packed or non-split.
//!
//! Owner: model-layers attention decode.
//! Invariants:
//! - A GQA-packed kernel is launched only on a non-split path.

use anyhow::Result;
use metrale_cache::kv_cache::PagedKvCache;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::super::super::Qwen3AttentionLayer;
use super::super::splitk_dispatch::{self, SplitkPlan};
use crate::layers::ops;

impl Qwen3AttentionLayer {
    /// 2026-09-26: The `KvCacheDtype::Bf16` arm.
    pub(super) fn run_paged_decode_bf16(
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
        // 2026-09-25: A layer the loader gave a sliding window (`set_sliding_window`)
        // attends only to the last `sliding_window` positions; every other layer passes 0.
        let sliding = self.sliding_window.unwrap_or(0);
        // 2026-09-25: BF16 split-K runs only where `bf16_splitk_pair` finds the Hopper twin
        // (`kernels/hopper/common/paged_decode_bf16_splitk_hopper.cu`); otherwise this
        // falls through to the non-split kernels below.
        let bf16_splitk = self.bf16_splitk_pair(head_dim);
        let num_splits =
            splitk_dispatch::num_splits(num_q_heads, head_dim, num_seqs, max_decode_seqs);
        if let (true, Some(pair)) = (
            splitk_dispatch::splits_are_worth_it(num_splits),
            bf16_splitk,
        ) {
            splitk_dispatch::log_decode_route(
                splitk_dispatch::RouteArm::Bf16,
                pair.name,
                num_splits,
            );
            return self.launch_splitk_bf16(
                gpu,
                &pair,
                SplitkPlan {
                    num_splits,
                    num_q_heads,
                    num_kv_heads,
                    head_dim,
                    block_size,
                    max_blocks_per_seq,
                    num_seqs,
                    inv_sqrt_d,
                    q_stride,
                    sliding_window: sliding,
                },
                q,
                kv_cache.k_pool_ptr(self.attn_layer_idx),
                kv_cache.v_pool_ptr(self.attn_layer_idx),
                workspace,
                output,
                block_table,
                seq_lens,
                stream,
            );
        }
        // 2026-09-25: GQA-packed kernel: one CTA per (kv_head, seq) instead of per (q_head,
        // seq), loading each K and V row once for the whole query group. `gqa_pack_kernel`
        // checks the lever, the shape and the handle. The kernel keeps the unpacked
        // kernel's per-head arithmetic and order, with the loads hoisted
        // (`paged_decode_attn_bf16_gqa.cu`).
        if let Some(gqa_k) = splitk_dispatch::gqa_pack_kernel(
            self.paged_decode_bf16_gqa_k,
            num_q_heads,
            num_kv_heads,
            head_dim,
        ) {
            splitk_dispatch::log_decode_route(
                splitk_dispatch::RouteArm::Bf16,
                splitk_dispatch::ROUTE_GQA_BF16,
                num_splits,
            );
            return ops::paged_decode_attn_bf16_gqa(
                gpu,
                gqa_k,
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
                sliding,
                stream,
            );
        }
        splitk_dispatch::log_decode_route(
            splitk_dispatch::RouteArm::Bf16,
            splitk_dispatch::ROUTE_NONSPLIT_BF16,
            num_splits,
        );
        // 2026-09-25: The HDIM=512 kernel for heads wider than 256, when loaded.
        let kernel = if head_dim > 256 && self.paged_decode_512_k.0 != 0 {
            self.paged_decode_512_k
        } else {
            self.paged_decode_k
        };
        ops::paged_decode_attn_bf16(
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
            sliding,
            stream,
        )
    }

    /// 2026-09-26: The fallback arm: FP8 and every dtype no other arm names.
    pub(super) fn run_paged_decode_fp8(
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
        // 2026-09-25: FP8 paged decode, with the split count from
        // `splitk_dispatch::num_splits`.
        let num_splits =
            splitk_dispatch::num_splits(num_q_heads, head_dim, num_seqs, max_decode_seqs);
        splitk_dispatch::trace_splits(self.attn_layer_idx, num_seqs, num_q_heads, num_splits);

        let (k_scale, v_scale) = self.effective_fp8_scales();
        let sliding = self.sliding_window.unwrap_or(0);
        let plan = SplitkPlan {
            num_splits,
            num_q_heads,
            num_kv_heads,
            head_dim,
            block_size,
            max_blocks_per_seq,
            num_seqs,
            inv_sqrt_d,
            q_stride,
            sliding_window: sliding,
        };

        if let (true, Some(pair)) = (
            splitk_dispatch::splits_are_worth_it(num_splits),
            self.fp8_splitk_pair(head_dim),
        ) {
            splitk_dispatch::log_decode_route(
                splitk_dispatch::RouteArm::Fp8,
                pair.name,
                num_splits,
            );
            self.launch_splitk_fp8(
                gpu,
                &pair,
                plan,
                q,
                kv_cache.k_pool_ptr(self.attn_layer_idx),
                kv_cache.v_pool_ptr(self.attn_layer_idx),
                workspace,
                output,
                block_table,
                seq_lens,
                k_scale,
                v_scale,
                kv_cache.cache_stride() as u64,
                stream,
            )
        } else if let Some(gqa_k) = splitk_dispatch::gqa_pack_kernel(
            self.paged_decode_fp8_gqa_k,
            num_q_heads,
            num_kv_heads,
            head_dim,
        ) {
            // 2026-09-25: GQA-packed kernel, as in the BF16 arm; only on the non-split
            // branch, because the packed grid `(num_kv_heads, num_seqs)` would change the
            // split partition and so the merge tree.
            splitk_dispatch::log_decode_route(
                splitk_dispatch::RouteArm::Fp8,
                splitk_dispatch::ROUTE_GQA_FP8,
                num_splits,
            );
            ops::paged_decode_attn_fp8_gqa(
                gpu,
                gqa_k,
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
                k_scale,
                v_scale,
                q_stride,
                kv_cache.cache_stride() as u64,
                sliding,
                stream,
            )
        } else {
            splitk_dispatch::log_decode_route(
                splitk_dispatch::RouteArm::Fp8,
                splitk_dispatch::ROUTE_NONSPLIT_FP8,
                num_splits,
            );
            // 2026-09-25: The HDIM=512 kernel for heads wider than 256, when loaded.
            let fp8_kernel = if head_dim > 256 && self.paged_decode_512_k.0 != 0 {
                self.paged_decode_512_k
            } else {
                self.paged_decode_k
            };
            ops::paged_decode_attn_fp8(
                gpu,
                fp8_kernel,
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
                k_scale,
                v_scale,
                q_stride,
                kv_cache.cache_stride() as u64,
                sliding,
                stream,
            )
        }
    }
}
