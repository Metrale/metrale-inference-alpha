// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The attention core of `MtpHead::forward_one`: cache the row's K/V and run the
//! paged decode attention over the drafter KV cache, BF16 or FP8 by `kv_bf16`.
//!
//! Owner: model-layers (MTP head).
//! Invariants: the caller holds the drafter KV-cache lock across the call.

use anyhow::Result;
use metrale_cache::kv_cache::PagedKvCache;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::MtpHead;
use crate::layer::ForwardContext;
use crate::layers::ops;

impl MtpHead {
    pub(super) fn mtp_attend(
        &self,
        ctx: &ForwardContext,
        kv_cache: &PagedKvCache,
        q_out: DevicePtr,
        k_out: DevicePtr,
        v_out: DevicePtr,
        attn_out: DevicePtr,
        meta_base: DevicePtr,
        max_blocks: u32,
        bs: usize,
        nq: u32,
        nkv: u32,
        hd: u32,
        kv_stride: u32,
        inv_sqrt_d: f32,
        stream: u64,
    ) -> Result<()> {
        if self.kv_bf16 {
            ops::reshape_and_cache(
                ctx.gpu,
                self.reshape_cache_k,
                k_out,
                v_out,
                kv_cache.k_pool_ptr(self.attn_layer_idx),
                kv_cache.v_pool_ptr(self.attn_layer_idx),
                meta_base.offset(8),
                1,
                nkv,
                hd,
                bs as u32,
                kv_stride,
                kv_stride,
                kv_cache.cache_stride() as u64,
                stream,
            )?;
            ops::paged_decode_attn_bf16(
                ctx.gpu,
                self.paged_decode_k,
                q_out,
                kv_cache.k_pool_ptr(self.attn_layer_idx),
                kv_cache.v_pool_ptr(self.attn_layer_idx),
                attn_out,
                meta_base.offset(256),
                meta_base.offset(16),
                max_blocks,
                1,
                nq,
                nkv,
                hd,
                bs as u32,
                inv_sqrt_d,
                nq * hd,
                0,
                stream,
            )?;
        } else {
            ops::reshape_and_cache_fp8(
                ctx.gpu,
                self.reshape_cache_k,
                k_out,
                v_out,
                kv_cache.k_pool_ptr(self.attn_layer_idx),
                kv_cache.v_pool_ptr(self.attn_layer_idx),
                meta_base.offset(8),
                1,
                nkv,
                hd,
                bs as u32,
                1.0,
                1.0,
                kv_stride,
                kv_stride,
                kv_cache.cache_stride() as u64,
                stream,
            )?;
            ops::paged_decode_attn_fp8(
                ctx.gpu,
                self.paged_decode_k,
                q_out,
                kv_cache.k_pool_ptr(self.attn_layer_idx),
                kv_cache.v_pool_ptr(self.attn_layer_idx),
                attn_out,
                meta_base.offset(256),
                meta_base.offset(16),
                max_blocks,
                1,
                nq,
                nkv,
                hd,
                bs as u32,
                inv_sqrt_d,
                1.0,
                1.0,
                nq * hd,
                kv_cache.cache_stride() as u64,
                0,
                stream,
            )?;
        }
        Ok(())
    }
}
