// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Nemotron MoE prefill, sorted path: batched routing, the routed
//! rows sorted by expert, grouped up / down GEMMs, the weighted unpermute, the
//! shared expert's relu² and down projection, and the residual add. The setup
//! is in `NemotronMoeLayer::prefill`.
//!
//! Owner: model-arch (Nemotron-H).
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::NemotronMoeLayer;
use metrale_model_layers::layer::ForwardContext;
use metrale_model_layers::layers::ops;

/// 2026-09-25: Values computed in `prefill` and passed to either expert path.
pub(super) struct SortedPrefillCtx {
    pub n: u32,
    pub num_tokens: usize,
    pub h: usize,
    pub inter: u32,
    pub shared_inter: u32,
    pub num_experts: u32,
    pub top_k: u32,
    pub scale: f32,
    pub latent: u32,
    pub gate_logits: DevicePtr,
    pub indices_dev: DevicePtr,
    pub weights_dev: DevicePtr,
    pub normed: DevicePtr,
    pub hidden: DevicePtr,
    pub latent_base: Option<DevicePtr>,
    pub shared_up_out_base: DevicePtr,
}

impl NemotronMoeLayer {
    pub(super) fn prefill_sorted_path(
        &self,
        p: &SortedPrefillCtx,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let total_expanded = p.n * p.top_k;
        let ne = p.num_experts as usize;
        let te = total_expanded as usize;

        KernelLaunch::new(ctx.gpu, self.topk_sigmoid_batched_k)
            .grid([1, p.n, 1])
            .block([256, 1, 1])
            .arg_ptr(p.gate_logits)
            .arg_ptr(self.weights.e_score_correction_bias.weight)
            .arg_ptr(p.indices_dev)
            .arg_ptr(p.weights_dev)
            .arg_u32(p.num_experts)
            .arg_u32(p.top_k)
            .arg_u32(if ctx.config.norm_topk_prob { 1 } else { 0 })
            .arg_f32(p.scale)
            .arg_u32(p.n)
            .launch(stream)?;

        // 2026-09-25: The sort's outputs reuse the `gate_logits` buffer, which
        // the routing launch above has consumed.
        let sorted_token_ids = p.gate_logits;
        let sorted_expert_ids = p.gate_logits.offset(te * 4);
        let expert_offsets = p.gate_logits.offset(te * 4 * 2);
        let token_to_perm = p.gate_logits.offset(te * 4 * 2 + (ne + 1) * 4);
        ops::moe_sort_by_expert(
            ctx.gpu,
            self.moe_sort_k,
            p.indices_dev,
            sorted_token_ids,
            sorted_expert_ids,
            expert_offsets,
            token_to_perm,
            total_expanded,
            p.num_experts,
            p.top_k,
            stream,
        )?;

        let is_latent = self.moe_latent_size > 0;
        let expert_input = if is_latent {
            p.latent_base.unwrap()
        } else {
            p.normed
        };
        let expert_k = if is_latent { p.latent } else { p.h as u32 };
        let expert_out_dim = if is_latent { p.latent } else { p.h as u32 };

        let expert_up_out = ctx.buffers.expert_up_out();
        // 2026-09-25: The arena buffers are reused across requests, so a row a
        // GEMM does not write would keep an earlier request's values; they are
        // zeroed first unless `METRALE_MOE_NO_ZERO_INTERMEDIATES` is set (any
        // value).
        if ctx.levers.moe_zero_intermediates {
            ctx.gpu.memset_async(
                expert_up_out,
                0,
                (total_expanded as usize) * (p.inter as usize) * 2,
                stream,
            )?;
        }
        let avg_per_expert = (p.num_tokens * p.top_k as usize).div_ceil(ne);
        // 2026-09-25: `max_m_tiles` is the grouped GEMM's grid.y, counted here in
        // 64-row tiles, so it must cover the most rows any one expert receives.
        // That count is known only on the device after the sort, so the bound is
        // the worst case, every routed row on one expert. With
        // `METRALE_MOE_MAX_M_TILES_ESTIMATE` set (any value) it is twice the
        // average instead, and the rows of an expert over that bound are not
        // computed.
        let max_m_tiles = if ctx.levers.moe_max_m_tiles_estimate {
            (avg_per_expert * 2).div_ceil(64).max(1) as u32
        } else {
            (total_expanded as usize).div_ceil(64).max(1) as u32
        };
        // 2026-09-25: W4A4 up GEMM on LatentMoE layers: the latent activations
        // quantized to NVFP4 once, relu² fused (`moe_w4a4_grouped_gemm_relu2`);
        // from 512 tokens, and only when `METRALE_MOE_W4A4` is set (any value).
        let w4a4_up = is_latent
            && p.n >= 512
            && self.moe_w4a4_grouped_k.0 != 0
            && self.quantize_nvfp4_k.0 != 0
            && ctx.buffers.fp8_act_bytes() >= (p.n as usize) * (p.latent as usize)
            && ctx.levers.moe_w4a4;
        if w4a4_up {
            let a4 = ctx.buffers.fp8_act();
            let a4_sf = a4.offset((p.n as usize) * (p.latent as usize) / 2);
            ops::quantize_bf16_to_nvfp4(
                ctx.gpu,
                self.quantize_nvfp4_k,
                expert_input,
                a4,
                a4_sf,
                p.n,
                p.latent,
                stream,
            )?;
            ops::moe_w4a4_grouped_gemm_relu2(
                ctx.gpu,
                self.moe_w4a4_grouped_k,
                a4,
                a4_sf,
                self.up_ptrs.packed_ptrs,
                self.up_ptrs.scale_ptrs,
                self.up_ptrs.scale2_vals,
                expert_up_out,
                expert_offsets,
                sorted_token_ids,
                p.num_experts,
                p.inter,
                expert_k,
                max_m_tiles,
                stream,
            )?;
        } else if let Some(ref upt) = self.up_ptrs_t {
            ops::moe_w4a16_grouped_gemm_ptrtable_n128(
                ctx.gpu,
                self.moe_grouped_gemm_n128_k,
                expert_input,
                upt.packed_ptrs,
                upt.scale_ptrs,
                upt.scale2_vals,
                expert_up_out,
                expert_offsets,
                sorted_token_ids,
                p.num_experts,
                p.inter,
                expert_k,
                max_m_tiles,
                stream,
            )?;
            // 2026-09-25: This branch has no fused epilogue: apply relu^2 elementwise.
            let relu2_n = total_expanded * p.inter;
            KernelLaunch::new(ctx.gpu, self.moe_relu2_elementwise_k)
                .grid([div_ceil(relu2_n, 256), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(expert_up_out)
                .arg_u32(relu2_n)
                .launch(stream)?;
        } else {
            // 2026-09-25: relu² is fused into the up GEMM when
            // `moe_w4a16_grouped_gemm_ptrtable_relu2` resolved; otherwise a
            // separate elementwise pass.
            let fused = self.moe_grouped_gemm_relu2_k.0 != 0;
            ops::moe_w4a16_grouped_gemm_ptrtable_n128(
                ctx.gpu,
                if fused {
                    self.moe_grouped_gemm_relu2_k
                } else {
                    self.moe_grouped_gemm_k
                },
                expert_input,
                self.up_ptrs.packed_ptrs,
                self.up_ptrs.scale_ptrs,
                self.up_ptrs.scale2_vals,
                expert_up_out,
                expert_offsets,
                sorted_token_ids,
                p.num_experts,
                p.inter,
                expert_k,
                max_m_tiles,
                stream,
            )?;
            if !fused {
                let relu2_n = total_expanded * p.inter;
                KernelLaunch::new(ctx.gpu, self.moe_relu2_elementwise_k)
                    .grid([div_ceil(relu2_n, 256), 1, 1])
                    .block([256, 1, 1])
                    .arg_ptr(expert_up_out)
                    .arg_u32(relu2_n)
                    .launch(stream)?;
            }
        }

        let expert_down_out = ctx.buffers.expert_down_out();
        if ctx.levers.moe_zero_intermediates {
            ctx.gpu.memset_async(
                expert_down_out,
                0,
                (total_expanded as usize) * (expert_out_dim as usize) * 2,
                stream,
            )?;
        }
        if let Some(ref dpt) = self.down_ptrs_t {
            ops::moe_w4a16_grouped_gemm_ptrtable_n128(
                ctx.gpu,
                self.moe_grouped_gemm_n128_k,
                expert_up_out,
                dpt.packed_ptrs,
                dpt.scale_ptrs,
                dpt.scale2_vals,
                expert_down_out,
                expert_offsets,
                DevicePtr::NULL,
                p.num_experts,
                expert_out_dim,
                p.inter,
                max_m_tiles,
                stream,
            )?;
        } else {
            ops::moe_w4a16_grouped_gemm_ptrtable_n128(
                ctx.gpu,
                self.moe_grouped_gemm_k,
                expert_up_out,
                self.down_ptrs.packed_ptrs,
                self.down_ptrs.scale_ptrs,
                self.down_ptrs.scale2_vals,
                expert_down_out,
                expert_offsets,
                DevicePtr::NULL,
                p.num_experts,
                expert_out_dim,
                p.inter,
                max_m_tiles,
                stream,
            )?;
        }

        let routed_out = ctx.buffers.moe_output();
        ops::moe_unpermute_reduce_indexed(
            ctx.gpu,
            self.moe_unpermute_reduce_k,
            expert_down_out,
            routed_out,
            token_to_perm,
            p.weights_dev,
            expert_out_dim,
            p.n,
            p.top_k,
            stream,
        )?;

        // 2026-09-25: The shared expert's up output came from `prefill_shared_up`:
        // relu² in place, then the down projection.
        let shared_down_out = ctx.buffers.ssm_deinterleaved();
        let shared_relu2_n = p.n * p.shared_inter;
        KernelLaunch::new(ctx.gpu, self.moe_relu2_elementwise_k)
            .grid([div_ceil(shared_relu2_n, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(p.shared_up_out_base)
            .arg_u32(shared_relu2_n)
            .launch(stream)?;
        // 2026-09-25: Down-projection arms, first match: native FP8; W4A4, from
        // 512 tokens and only when `METRALE_SHARED_W4A4_DOWN` is set (any value);
        // the pre-dequant FP8 copy, derived from the NVFP4 weights; transposed
        // NVFP4; plain `w4a16_gemm`.
        let native_down = self
            .weights
            .shared_down_fp8
            .as_ref()
            .filter(|_| self.w8a16_gemm_pipelined_k.0 != 0 || self.w8a16_gemm_k.0 != 0);
        let w4a4_down = native_down.is_none()
            && p.n >= 512
            && self.w4a4_gemm_k.0 != 0
            && self.quantize_nvfp4_k.0 != 0
            && ctx.buffers.fp8_act_bytes() >= (p.shared_inter as usize) * (p.n as usize)
            && ctx.levers.shared_w4a4_down;
        if let Some(fp8w) = native_down {
            let (kern, pipelined) = if self.w8a16_gemm_pipelined_k.0 != 0 {
                (self.w8a16_gemm_pipelined_k, true)
            } else {
                (self.w8a16_gemm_k, false)
            };
            let f = if pipelined {
                ops::w8a16_gemm_pipelined
            } else {
                ops::w8a16_gemm
            };
            f(
                ctx.gpu,
                kern,
                p.shared_up_out_base,
                fp8w.weight,
                fp8w.row_scale,
                shared_down_out,
                p.n,
                p.h as u32,
                p.shared_inter,
                stream,
            )?;
        } else if w4a4_down {
            let a4 = ctx.buffers.fp8_act();
            let a4_sf = a4.offset((p.n as usize) * (p.shared_inter as usize) / 2);
            ops::quantize_bf16_to_nvfp4(
                ctx.gpu,
                self.quantize_nvfp4_k,
                p.shared_up_out_base,
                a4,
                a4_sf,
                p.n,
                p.shared_inter,
                stream,
            )?;
            ops::w4a4_gemm_mfast(
                ctx.gpu,
                self.w4a4_gemm_k,
                a4,
                a4_sf,
                &self.weights.shared_down,
                shared_down_out,
                p.n,
                p.h as u32,
                p.shared_inter,
                stream,
            )?;
        } else if let Some(w_fp8) = self.shared_down_pd_fp8 {
            ops::fp8_gemm_m128_mfast(
                ctx.gpu,
                self.fp8_gemm_m128_k,
                p.shared_up_out_base,
                w_fp8,
                shared_down_out,
                p.n,
                p.h as u32,
                p.shared_inter,
                stream,
            )?;
        } else if let Some(ref sdt) = self.shared_down_t {
            if p.n > 128 && self.w4a16_gemm_t_m128_k.0 != 0 {
                ops::w4a16_gemm_n128_m128(
                    ctx.gpu,
                    self.w4a16_gemm_t_m128_k,
                    p.shared_up_out_base,
                    sdt,
                    shared_down_out,
                    p.n,
                    p.h as u32,
                    p.shared_inter,
                    stream,
                )?;
            } else {
                ops::w4a16_gemm_n128(
                    ctx.gpu,
                    self.w4a16_gemm_t_k,
                    p.shared_up_out_base,
                    sdt,
                    shared_down_out,
                    p.n,
                    p.h as u32,
                    p.shared_inter,
                    stream,
                )?;
            }
        } else {
            ops::w4a16_gemm(
                ctx.gpu,
                self.w4a16_gemm_k,
                p.shared_up_out_base,
                &self.weights.shared_down,
                shared_down_out,
                p.n,
                p.h as u32,
                p.shared_inter,
                stream,
            )?;
        }

        if is_latent {
            let fc2_out = ctx.buffers.attn_output();
            if let Some(w_fp8) = self.fc2_pd_fp8 {
                ops::fp8_gemm_m128_mfast(
                    ctx.gpu,
                    self.fp8_gemm_m128_k,
                    routed_out,
                    w_fp8,
                    fc2_out,
                    p.n,
                    p.h as u32,
                    p.latent,
                    stream,
                )?;
            } else {
                let fc2 = self.weights.fc2_latent_proj.as_ref().unwrap();
                self.dense_gemm_prefill(
                    ctx.gpu, routed_out, fc2, fc2_out, p.n, p.h as u32, p.latent, stream,
                )?;
            }
            ops::residual_add(
                ctx.gpu,
                self.residual_add_k,
                fc2_out,
                shared_down_out,
                (p.num_tokens * p.h) as u32,
                stream,
            )?;
            ops::residual_add(
                ctx.gpu,
                self.residual_add_k,
                p.hidden,
                fc2_out,
                (p.num_tokens * p.h) as u32,
                stream,
            )?;
        } else {
            ops::residual_add(
                ctx.gpu,
                self.residual_add_k,
                routed_out,
                shared_down_out,
                (p.num_tokens * p.h) as u32,
                stream,
            )?;
            ops::residual_add(
                ctx.gpu,
                self.residual_add_k,
                p.hidden,
                routed_out,
                (p.num_tokens * p.h) as u32,
                stream,
            )?;
        }
        Ok(())
    }
}
