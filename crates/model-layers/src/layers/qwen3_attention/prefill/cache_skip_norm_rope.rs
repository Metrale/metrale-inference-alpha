// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The Q/K/V norm and RoPE steps of the cache-skip prefill
//! (`cache_skip.rs`): Q split from the gate, the Q, K and V RMS norms, RoPE
//! on Q and K, and their op dumps.
//!
//! Owner: model-layers (attention).
//! Invariants: `prefill_attention_with_cache_skip` calls these helpers where
//! their statements run, so every launch keeps its order and arguments.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

impl Qwen3AttentionLayer {
    /// 2026-09-26: The Q split (from the gate, fused with the Q norm and MRoPE
    /// when `q_rope_fused`), the Q and K RMS norms, the `k_post_norm` op dump
    /// and the V norm of `prefill_attention_with_cache_skip`, in that order.
    pub(super) fn cache_skip_qk_norms(
        &self,
        ctx: &ForwardContext,
        q_rope_fused: bool,
        qg_out: DevicePtr,
        q_contiguous: DevicePtr,
        k_contiguous: DevicePtr,
        v_contiguous: DevicePtr,
        positions: DevicePtr,
        positions_h: DevicePtr,
        positions_w: DevicePtr,
        n: u32,
        nq: u32,
        nkv: u32,
        hd: u32,
        q_proj_dim: usize,
        q_dim: usize,
        kv_dim: usize,
        num_tokens: usize,
        bf16: usize,
        eps: f32,
        stream: u64,
    ) -> Result<()> {
        if q_rope_fused {
            ops::deinterleave_qg_split_qnorm_mrope(
                ctx.gpu,
                self.deinterleave_qg_split_qnorm_mrope_k,
                qg_out,
                q_contiguous,
                self.attn.q_norm.weight,
                positions,
                positions_h,
                positions_w,
                n,
                nq,
                hd,
                q_proj_dim as u32,
                self.rotary_dim_override
                    .unwrap_or(ctx.config.rotary_dim() as u32),
                eps,
                self.rope_theta_override
                    .unwrap_or(ctx.config.rope_theta as f32),
                stream,
            )?;
        } else if self.gated && !self.attn.q_norm.weight.is_null() {
            ops::deinterleave_qg_split_qnorm(
                ctx.gpu,
                self.deinterleave_qg_split_qnorm_k,
                qg_out,
                q_contiguous,
                self.attn.q_norm.weight,
                n,
                nq,
                hd,
                q_proj_dim as u32,
                eps,
                stream,
            )?;
        } else if self.gated {
            ops::deinterleave_qg_split(
                ctx.gpu,
                self.deinterleave_qg_split_k,
                qg_out,
                q_contiguous,
                n,
                nq,
                hd,
                q_proj_dim as u32,
                stream,
            )?;
        } else if self.mla.is_some() {
            if self.attn_layer_idx == 0 && ctx.config.model_type == "mistral" {
                ctx.gpu.synchronize(stream)?;
                let v_chk = k_contiguous.offset(num_tokens * kv_dim * bf16);
                crate::layers::qwen3_attention::trait_impl::diag_norm(
                    ctx.gpu,
                    v_chk,
                    (nkv * hd) as usize,
                    stream,
                    "L0 V BEFORE Q_copy",
                );
            }
            ctx.gpu
                .copy_d2d_async(qg_out, q_contiguous, num_tokens * q_dim * bf16, stream)
                .map_err(|e| anyhow::anyhow!("MLA Q copy failed: {e}"))?;
        } else {
            // 2026-09-25: Ungated: `q_contiguous` aliases `qg_out` (see its
            // binding), so only the Q norm runs here.
            debug_assert_eq!(q_contiguous.0, qg_out.0);
            if let Some(ref q_norm_full) = self.attn.q_norm_full {
                ops::rms_norm(
                    ctx.gpu,
                    self.rms_norm_w_k,
                    q_contiguous,
                    q_norm_full,
                    q_contiguous,
                    n,
                    nq * hd,
                    eps,
                    stream,
                )?;
            } else if !self.attn.q_norm.weight.is_null() {
                if self.rms_norm_w_warp_row_k.0 != 0 && ops::rms_norm_short_row_eligible(nq * n, hd)
                {
                    ops::rms_norm_warp_row(
                        ctx.gpu,
                        self.rms_norm_w_warp_row_k,
                        q_contiguous,
                        &self.attn.q_norm,
                        q_contiguous,
                        nq * n,
                        hd,
                        eps,
                        stream,
                    )?;
                } else {
                    ops::rms_norm(
                        ctx.gpu,
                        self.rms_norm_w_k,
                        q_contiguous,
                        &self.attn.q_norm,
                        q_contiguous,
                        nq * n,
                        hd,
                        eps,
                        stream,
                    )?;
                }
            }
        }
        if let Some(ref k_norm_full) = self.attn.k_norm_full {
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_w_k,
                k_contiguous,
                k_norm_full,
                k_contiguous,
                n,
                nkv * hd,
                eps,
                stream,
            )?;
        } else if !self.attn.k_norm.weight.is_null() {
            if self.rms_norm_w_warp_row_k.0 != 0 && ops::rms_norm_short_row_eligible(nkv * n, hd) {
                ops::rms_norm_warp_row(
                    ctx.gpu,
                    self.rms_norm_w_warp_row_k,
                    k_contiguous,
                    &self.attn.k_norm,
                    k_contiguous,
                    nkv * n,
                    hd,
                    eps,
                    stream,
                )?;
            } else {
                ops::rms_norm(
                    ctx.gpu,
                    self.rms_norm_w_k,
                    k_contiguous,
                    &self.attn.k_norm,
                    k_contiguous,
                    nkv * n,
                    hd,
                    eps,
                    stream,
                )
                .map_err(|e| {
                    anyhow::anyhow!("k_norm rms_norm failed: nkv={nkv} n={n} hd={hd}: {e}")
                })?;
            }
        }

        // 2026-09-25: METRALE_OP_DUMP: K after k_norm, before RoPE.
        if num_tokens > 0 {
            let kv_dim_e = (nkv * hd) as usize;
            super::super::op_dump::dump_bf16(
                ctx.gpu,
                k_contiguous,
                (num_tokens - 1) * kv_dim_e * bf16,
                kv_dim_e,
                self.attn_layer_idx,
                "k_post_norm",
                stream,
            )?;
        }

        // 2026-09-25: The V norm, when `set_k_eq_v` or `set_v_norm` installed
        // one: an RMS norm of each V head with `v_norm_weight`. V gets no RoPE.
        if let Some(v_norm_w) = self.v_norm_weight.as_ref() {
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_w_k,
                v_contiguous,
                v_norm_w,
                v_contiguous,
                nkv * n,
                hd,
                eps,
                stream,
            )?;
        }
        Ok(())
    }

    /// 2026-09-26: RoPE on Q and K, then the `k_post_rope` and `q_post_rope` op
    /// dumps, of `prefill_attention_with_cache_skip`.
    pub(super) fn cache_skip_rope(
        &self,
        ctx: &ForwardContext,
        q_rope_fused: bool,
        q_contiguous: DevicePtr,
        k_contiguous: DevicePtr,
        positions: DevicePtr,
        positions_h: DevicePtr,
        positions_w: DevicePtr,
        n: u32,
        nq: u32,
        nkv: u32,
        hd: u32,
        num_tokens: usize,
        bf16: usize,
        stream: u64,
    ) -> Result<()> {
        // 2026-09-25: RoPE on Q and K.
        if self.mla.is_some() {
            // 2026-09-25: Not reached: MLA layers returned above.
        } else if q_rope_fused {
            ops::rope_mrope_interleaved_k_only(
                ctx.gpu,
                self.rope_mrope_interleaved_k_only_k,
                k_contiguous,
                positions,
                positions_h,
                positions_w,
                n,
                nkv,
                hd,
                self.rotary_dim_override
                    .unwrap_or(ctx.config.rotary_dim() as u32),
                self.rope_theta_override
                    .unwrap_or(ctx.config.rope_theta as f32),
                stream,
            )
            .map_err(|e| anyhow::anyhow!("rope_mrope_interleaved_k_only failed: {e}"))?;
        } else if !self.yarn_inv_freq.is_null() {
            ops::rope_yarn_scaled(
                ctx.gpu,
                self.rope_yarn_scaled_k,
                q_contiguous,
                k_contiguous,
                positions,
                n,
                nq,
                nkv,
                hd,
                self.rotary_dim_override
                    .unwrap_or(ctx.config.rotary_dim() as u32),
                self.yarn_inv_freq,
                self.yarn_attention_factor,
                stream,
            )
            .map_err(|e| anyhow::anyhow!("rope_yarn_scaled failed: {e}"))?;
        } else if self.rope_proportional && self.rope_proportional_k.0 != 0 {
            let rope_angles = self
                .rotary_dim_override
                .unwrap_or(ctx.config.rotary_dim() as u32);
            ops::rope_proportional(
                ctx.gpu,
                self.rope_proportional_k,
                q_contiguous,
                k_contiguous,
                positions,
                n,
                nq,
                nkv,
                hd,
                rope_angles,
                self.rope_theta_override
                    .unwrap_or(ctx.config.rope_theta as f32),
                stream,
            )
            .map_err(|e| anyhow::anyhow!("rope_proportional failed: {e}"))?;
        } else {
            ops::rope(
                ctx.gpu,
                self.rope_k,
                q_contiguous,
                k_contiguous,
                positions,
                n,
                nq,
                nkv,
                hd,
                self.rotary_dim_override
                    .unwrap_or(ctx.config.rotary_dim() as u32),
                self.rope_theta_override
                    .unwrap_or(ctx.config.rope_theta as f32),
                stream,
            )
            .map_err(|e| anyhow::anyhow!("rope failed: {e}"))?;
        }

        // 2026-09-25: METRALE_OP_DUMP: K and Q after RoPE.
        if num_tokens > 0 {
            let kv_dim_e = (nkv * hd) as usize;
            let q_dim_e = (nq * hd) as usize;
            super::super::op_dump::dump_bf16(
                ctx.gpu,
                k_contiguous,
                (num_tokens - 1) * kv_dim_e * bf16,
                kv_dim_e,
                self.attn_layer_idx,
                "k_post_rope",
                stream,
            )?;
            super::super::op_dump::dump_bf16(
                ctx.gpu,
                q_contiguous,
                (num_tokens - 1) * q_dim_e * bf16,
                q_dim_e,
                self.attn_layer_idx,
                "q_post_rope",
                stream,
            )?;
        }
        Ok(())
    }
}
