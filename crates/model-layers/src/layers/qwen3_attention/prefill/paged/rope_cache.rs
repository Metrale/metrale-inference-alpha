// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: RoPE and the paged KV-cache write of `prefill_attention_paged`, including the
//! fused-K rewrite of a BF16 cache's K side.
//!
//! Owner: model-layers (attention).
//! Invariants:
//! - Rows below `kv_write_floor` are not written to the cache.

use anyhow::Result;
use metrale_cache::kv_cache::PagedKvCache;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::super::super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

impl Qwen3AttentionLayer {
    /// 2026-09-26: Called by `prefill_attention_paged` once the positions and slots are
    /// resolved, before attention.
    pub(super) fn prefill_paged_rope_cache_write(
        &self,
        kv_cache: &mut PagedKvCache,
        q_contiguous: DevicePtr,
        k_contiguous: DevicePtr,
        v_contiguous: DevicePtr,
        raw_k_scratch: Option<DevicePtr>,
        bmeta_positions: DevicePtr,
        bmeta_positions_h: DevicePtr,
        bmeta_positions_w: DevicePtr,
        bmeta_slot: DevicePtr,
        n: u32,
        nq: u32,
        nkv: u32,
        hd: u32,
        bs: usize,
        num_tokens: usize,
        kv_write_floor: usize,
        kv_dim: usize,
        bf16: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if self.mla.is_some() {
            // 2026-09-25: Not reached: MLA layers returned above.
        } else if !self.yarn_inv_freq.is_null() {
            ops::rope_yarn_scaled(
                ctx.gpu,
                self.rope_yarn_scaled_k,
                q_contiguous,
                k_contiguous,
                bmeta_positions,
                n,
                nq,
                nkv,
                hd,
                self.rotary_dim_override
                    .unwrap_or(ctx.config.rotary_dim() as u32),
                self.yarn_inv_freq,
                self.yarn_attention_factor,
                stream,
            )?;
        } else if let Some(ref mla) = self.mla {
            // 2026-09-25: Not reached: MLA layers returned above.
            if !mla.yarn_inv_freq.is_null() {
                ops::rope_yarn(
                    ctx.gpu,
                    self.rope_yarn_k,
                    q_contiguous,
                    k_contiguous,
                    bmeta_positions,
                    n,
                    nq,
                    nkv,
                    hd,
                    ctx.config.rotary_dim() as u32,
                    mla.yarn_inv_freq,
                    ctx.config.rope_theta as f32,
                    stream,
                )?;
            } else {
                ops::rope(
                    ctx.gpu,
                    self.rope_k,
                    q_contiguous,
                    k_contiguous,
                    bmeta_positions,
                    n,
                    nq,
                    nkv,
                    hd,
                    self.rotary_dim_override
                        .unwrap_or(ctx.config.rotary_dim() as u32),
                    self.rope_theta_override
                        .unwrap_or(ctx.config.rope_theta as f32),
                    stream,
                )?;
            }
        } else if self.rope_proportional && self.rope_proportional_k.0 != 0 {
            let rope_angles = self
                .rotary_dim_override
                .unwrap_or(ctx.config.rotary_dim() as u32);
            ops::rope_proportional(
                ctx.gpu,
                self.rope_proportional_k,
                q_contiguous,
                k_contiguous,
                bmeta_positions,
                n,
                nq,
                nkv,
                hd,
                rope_angles,
                self.rope_theta_override
                    .unwrap_or(ctx.config.rope_theta as f32),
                stream,
            )?;
        } else if self.mrope_interleaved && self.rope_mrope_interleaved_k.0 != 0 {
            ops::rope_mrope_interleaved(
                ctx.gpu,
                self.rope_mrope_interleaved_k,
                q_contiguous,
                k_contiguous,
                bmeta_positions,
                bmeta_positions_h,
                bmeta_positions_w,
                n,
                nq,
                nkv,
                hd,
                self.rotary_dim_override
                    .unwrap_or(ctx.config.rotary_dim() as u32),
                self.rope_theta_override
                    .unwrap_or(ctx.config.rope_theta as f32),
                stream,
            )?;
        } else {
            ops::rope(
                ctx.gpu,
                self.rope_k,
                q_contiguous,
                k_contiguous,
                bmeta_positions,
                n,
                nq,
                nkv,
                hd,
                self.rotary_dim_override
                    .unwrap_or(ctx.config.rotary_dim() as u32),
                self.rope_theta_override
                    .unwrap_or(ctx.config.rope_theta as f32),
                stream,
            )?;
        }

        // 2026-09-25: Write K/V to the paged cache for the rows from
        // `kv_write_floor` on (`nkv` heads of `hd`). The rows below the floor
        // are positions already in the cache, and their cached K/V stay as
        // they are.
        let wf = kv_write_floor.min(num_tokens);
        if self.mla.is_none() && wf < num_tokens {
            self.write_kv_cache(
                ctx.gpu,
                k_contiguous.offset(wf * kv_dim * bf16),
                v_contiguous.offset(wf * kv_dim * bf16),
                kv_cache,
                // 2026-09-25: slot_mapping entries are int64 (8 bytes each)
                bmeta_slot.offset(wf * 8),
                n - wf as u32,
                nkv,
                hd,
                bs as u32,
                nkv * hd,
                nkv * hd,
                stream,
                ctx.graph_capture,
            )?;
            // 2026-09-25: Fused K path: rewrite the K side of a BF16 cache from
            // the raw K copy, with k_norm and RoPE in FP32 and one BF16 rounding.
            // The V side stays as written above.
            if let Some(raw_k) = raw_k_scratch
                && !self.attn.k_norm.weight.is_null()
            {
                use metrale_cache::kv_cache::KvCacheDtype;
                if kv_cache.dtype() == KvCacheDtype::Bf16 {
                    ops::fused_k_norm_rope_cache_write_bf16_mrope(
                        ctx.gpu,
                        self.fused_k_norm_rope_mrope_cache_write_bf16_k,
                        raw_k.offset(wf * kv_dim * bf16),
                        self.attn.k_norm.weight,
                        // 2026-09-25: positions are u32 (4 bytes), slots int64 (8 bytes)
                        bmeta_positions.offset(wf * 4),
                        bmeta_positions_h.offset(wf * 4),
                        bmeta_positions_w.offset(wf * 4),
                        kv_cache.k_pool_ptr(self.attn_layer_idx),
                        bmeta_slot.offset(wf * 8),
                        n - wf as u32,
                        nkv,
                        hd,
                        self.rotary_dim_override
                            .unwrap_or(ctx.config.rotary_dim() as u32),
                        bs as u32,
                        ctx.config.rms_norm_eps as f32,
                        self.rope_theta_override
                            .unwrap_or(ctx.config.rope_theta as f32),
                        stream,
                    )?;
                }
            }
        }
        Ok(())
    }
}
