// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Single-token decode for [`super::NemotronMoeLayer`]:
//!   - `decode_direct_moe` (`moe_latent_size == 0`): experts work on the
//!     hidden vector `[H]`;
//!   - `decode_latent_moe` (`moe_latent_size > 0`): experts work in the latent
//!     space `[L]`, with the fc1 / fc2 projections between hidden and latent.
//!
//! Owner: model-arch (Nemotron-H).
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::NemotronMoeLayer;
use metrale_model_layers::layer::ForwardContext;
use metrale_model_layers::layers::ops;

impl NemotronMoeLayer {
    /// 2026-09-25: Single-token decode: norm, gate and routing, then the direct
    /// or latent MoE.
    pub(super) fn decode_inner(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size as u32;
        let inter = self.moe_inter as u32;
        let shared_inter = ctx.config.shared_expert_intermediate_size as u32;
        let num_experts = ctx.config.num_experts as u32;
        let top_k = self.top_k as u32;
        let eps = ctx.config.rms_norm_eps as f32;
        let scale = ctx.config.routed_scaling_factor as f32;

        let normed = ctx.buffers.norm_output();
        ops::rms_norm_residual(
            ctx.gpu,
            self.rms_norm_residual_k,
            hidden,
            &self.input_norm,
            normed,
            residual,
            1,
            h,
            eps,
            stream,
        )?;

        let gate_logits = ctx.buffers.gate_logits();
        ops::dense_gemv(
            ctx.gpu,
            self.dense_gemv_k,
            normed,
            &self.weights.gate,
            gate_logits,
            num_experts,
            h,
            stream,
        )?;

        let scratch = ctx.buffers.scratch();
        let indices_dev = scratch;
        let weights_dev = scratch.offset(top_k as usize * 4);
        ops::moe_topk_sigmoid(
            ctx.gpu,
            self.topk_sigmoid_k,
            gate_logits,
            self.weights.e_score_correction_bias.weight,
            indices_dev,
            weights_dev,
            num_experts,
            top_k,
            ctx.config.norm_topk_prob,
            scale,
            stream,
        )?;

        if self.moe_latent_size > 0 {
            self.decode_latent_moe(
                hidden,
                normed,
                indices_dev,
                weights_dev,
                ctx,
                stream,
                h,
                inter,
                shared_inter,
                top_k,
            )
        } else {
            self.decode_direct_moe(
                hidden,
                normed,
                indices_dev,
                weights_dev,
                ctx,
                stream,
                h,
                inter,
                shared_inter,
                top_k,
            )
        }
    }

    /// 2026-09-25: Direct MoE: the routed experts work on the hidden vector.
    pub(super) fn decode_direct_moe(
        &self,
        hidden: DevicePtr,
        normed: DevicePtr,
        indices_dev: DevicePtr,
        weights_dev: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
        h: u32,
        inter: u32,
        shared_inter: u32,
        top_k: u32,
    ) -> Result<()> {
        let expert_up_out = ctx.buffers.expert_up_out();
        ops::moe_expert_gemv(
            ctx.gpu,
            self.moe_expert_gemv_k,
            normed,
            self.up_ptrs.packed_ptrs,
            self.up_ptrs.scale_ptrs,
            self.up_ptrs.scale2_vals,
            expert_up_out,
            indices_dev,
            inter,
            h,
            top_k,
            0,
            stream,
        )?;

        // 2026-09-25: Shared expert up: the checkpoint's own FP8 bytes
        // (`w8a16_gemv`) when the loader kept them and the kernel resolved, else
        // the NVFP4 GEMV.
        let shared_up_out = ctx.buffers.ssm_qkvz();
        match self
            .weights
            .shared_up_fp8
            .as_ref()
            .filter(|_| self.w8a16_gemv_k.0 != 0)
        {
            Some(fp8w) => ops::w8a16_gemv(
                ctx.gpu,
                self.w8a16_gemv_k,
                normed,
                fp8w.weight,
                fp8w.row_scale,
                shared_up_out,
                shared_inter,
                h,
                stream,
            )?,
            None => ops::w4a16_decode_gemv(
                ctx.gpu,
                self.w4a16_gemv_k,
                self.w4a16_gemv_sw_k,
                ctx.levers.gemv_sw,
                normed,
                &self.weights.shared_up,
                shared_up_out,
                shared_inter,
                h,
                stream,
            )?,
        }

        let expert_down_out = ctx.buffers.expert_down_out();
        let shared_down_out = ctx.buffers.ssm_deinterleaved();
        let smem = (shared_inter.max(inter) as usize) * 4;

        // 2026-09-25: The fused relu²+down kernel reads NVFP4 weights only. With a
        // native-FP8 shared down_proj the shared expert runs here instead (relu²
        // in place, then `w8a16_gemv`), and the fused launch gets grid.y =
        // `top_k`: its shared slot is `expert_slot == top_k`, so it never runs.
        let native_shared_down = self
            .weights
            .shared_down_fp8
            .as_ref()
            .filter(|_| self.w8a16_gemv_k.0 != 0 && self.moe_relu2_elementwise_k.0 != 0);
        if let Some(fp8w) = native_shared_down {
            KernelLaunch::new(ctx.gpu, self.moe_relu2_elementwise_k)
                .grid([div_ceil(shared_inter, 256), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(shared_up_out)
                .arg_u32(shared_inter)
                .launch(stream)?;
            ops::w8a16_gemv(
                ctx.gpu,
                self.w8a16_gemv_k,
                shared_up_out,
                fp8w.weight,
                fp8w.row_scale,
                shared_down_out,
                h,
                shared_inter,
                stream,
            )?;
        }
        let fused_slots = if native_shared_down.is_some() {
            top_k
        } else {
            top_k + 1
        };

        KernelLaunch::new(ctx.gpu, self.relu2_down_shared_k)
            .grid([div_ceil(h, 8), fused_slots, 1])
            .block([128, 1, 1])
            .shared_mem(smem as u32)
            .arg_ptr(expert_up_out)
            .arg_ptr(self.down_ptrs.packed_ptrs)
            .arg_ptr(self.down_ptrs.scale_ptrs)
            .arg_ptr(self.down_ptrs.scale2_vals)
            .arg_ptr(expert_down_out)
            .arg_ptr(indices_dev)
            .arg_ptr(shared_up_out)
            .arg_ptr(self.weights.shared_down.weight)
            .arg_ptr(self.weights.shared_down.weight_scale)
            .arg_f32(self.weights.shared_down.weight_scale_2)
            .arg_ptr(shared_down_out)
            .arg_u32(h)
            .arg_u32(inter)
            .arg_u32(shared_inter)
            .arg_u32(h)
            .arg_u32(top_k)
            .launch(stream)?;

        let output = ctx.buffers.moe_output();
        KernelLaunch::new(ctx.gpu, self.weighted_sum_scale_k)
            .grid([div_ceil(h, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(output)
            .arg_ptr(expert_down_out)
            .arg_ptr(weights_dev)
            .arg_ptr(shared_down_out)
            .arg_u32(h)
            .arg_u32(top_k)
            .arg_f32(1.0f32)
            .launch(stream)?;

        ops::residual_add(ctx.gpu, self.residual_add_k, hidden, output, h, stream)
    }

    /// 2026-09-25: LatentMoE: the routed experts work in the latent space
    /// `[moe_latent_size]`.
    ///
    /// fc1_latent(normed) → latent `[L]`
    /// routed up(latent) → `[inter]`, relu²+down → `[L]`
    /// weighted_sum → combined `[L]`
    /// fc2_latent(combined) → routed_out `[H]`
    /// shared up(normed) → `[shared_inter]`, relu²+down → shared_out `[H]`
    /// output = routed_out + shared_out, added to `hidden`
    pub(super) fn decode_latent_moe(
        &self,
        hidden: DevicePtr,
        normed: DevicePtr,
        indices_dev: DevicePtr,
        weights_dev: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
        h: u32,
        inter: u32,
        shared_inter: u32,
        top_k: u32,
    ) -> Result<()> {
        let latent = self.moe_latent_size as u32;
        let fc1 = self.weights.fc1_latent_proj.as_ref().unwrap();
        let fc2 = self.weights.fc2_latent_proj.as_ref().unwrap();

        let latent_buf = ctx.buffers.ssm_ba();
        ops::dense_gemv(
            ctx.gpu,
            self.dense_gemv_k,
            normed,
            fc1,
            latent_buf,
            latent,
            h,
            stream,
        )?;

        let expert_up_out = ctx.buffers.expert_up_out();
        ops::moe_expert_gemv(
            ctx.gpu,
            self.moe_expert_gemv_k,
            latent_buf,
            self.up_ptrs.packed_ptrs,
            self.up_ptrs.scale_ptrs,
            self.up_ptrs.scale2_vals,
            expert_up_out,
            indices_dev,
            inter,
            latent,
            top_k,
            0,
            stream,
        )?;

        // 2026-09-25: Shared expert up: the checkpoint's native FP8 bytes when
        // present; then the NVFP4 copy is `QuantizedWeight::null()` and must not
        // reach `w4a16_gemv`.
        let shared_up_out = ctx.buffers.ssm_qkvz();
        match self
            .weights
            .shared_up_fp8
            .as_ref()
            .filter(|_| self.w8a16_gemv_k.0 != 0)
        {
            Some(fp8w) => ops::w8a16_gemv(
                ctx.gpu,
                self.w8a16_gemv_k,
                normed,
                fp8w.weight,
                fp8w.row_scale,
                shared_up_out,
                shared_inter,
                h,
                stream,
            )?,
            None => ops::w4a16_decode_gemv(
                ctx.gpu,
                self.w4a16_gemv_k,
                self.w4a16_gemv_sw_k,
                ctx.levers.gemv_sw,
                normed,
                &self.weights.shared_up,
                shared_up_out,
                shared_inter,
                h,
                stream,
            )?,
        }

        // 2026-09-25: Fused relu²+down: the routed rows are `latent` wide and the
        // shared row `h` wide (the kernel's `N` and `N_shared` arguments).
        let expert_down_out = ctx.buffers.expert_down_out();
        let shared_down_out = ctx.buffers.ssm_deinterleaved();
        let max_n = h.max(latent);
        let smem = (shared_inter.max(inter) as usize) * 4;

        // 2026-09-25: The same native-FP8 shared down_proj substitution as
        // `decode_direct_moe`.
        let native_shared_down = self
            .weights
            .shared_down_fp8
            .as_ref()
            .filter(|_| self.w8a16_gemv_k.0 != 0 && self.moe_relu2_elementwise_k.0 != 0);
        if let Some(fp8w) = native_shared_down {
            KernelLaunch::new(ctx.gpu, self.moe_relu2_elementwise_k)
                .grid([div_ceil(shared_inter, 256), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(shared_up_out)
                .arg_u32(shared_inter)
                .launch(stream)?;
            ops::w8a16_gemv(
                ctx.gpu,
                self.w8a16_gemv_k,
                shared_up_out,
                fp8w.weight,
                fp8w.row_scale,
                shared_down_out,
                h,
                shared_inter,
                stream,
            )?;
        }
        let fused_slots = if native_shared_down.is_some() {
            top_k
        } else {
            top_k + 1
        };

        KernelLaunch::new(ctx.gpu, self.relu2_down_shared_k)
            .grid([div_ceil(max_n, 8), fused_slots, 1])
            .block([128, 1, 1])
            .shared_mem(smem as u32)
            .arg_ptr(expert_up_out)
            .arg_ptr(self.down_ptrs.packed_ptrs)
            .arg_ptr(self.down_ptrs.scale_ptrs)
            .arg_ptr(self.down_ptrs.scale2_vals)
            .arg_ptr(expert_down_out)
            .arg_ptr(indices_dev)
            .arg_ptr(shared_up_out)
            .arg_ptr(self.weights.shared_down.weight)
            .arg_ptr(self.weights.shared_down.weight_scale)
            .arg_f32(self.weights.shared_down.weight_scale_2)
            .arg_ptr(shared_down_out)
            .arg_u32(latent)
            .arg_u32(inter)
            .arg_u32(shared_inter)
            .arg_u32(h)
            .arg_u32(top_k)
            .launch(stream)?;

        // 2026-09-25: Weighted sum in latent space, `[top_k, L]` → `[L]`, into
        // `latent_buf`, whose fc1 output the up GEMV has already read.
        // `moe_weighted_sum_scale` always adds `shared_down[idx]`, so a zeroed
        // `expert_gate_out[..L]` stands in for it. The routing weights already
        // carry `routed_scaling_factor` (the top-k kernel applied it), so the
        // sum's own factor is 1.0.
        let combined_latent = latent_buf;
        let dummy_shared = ctx.buffers.expert_gate_out();
        ctx.gpu
            .memset_async(dummy_shared, 0, latent as usize * 2, stream)?;
        KernelLaunch::new(ctx.gpu, self.weighted_sum_scale_k)
            .grid([div_ceil(latent, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(combined_latent)
            .arg_ptr(expert_down_out)
            .arg_ptr(weights_dev)
            .arg_ptr(dummy_shared)
            .arg_u32(latent)
            .arg_u32(top_k)
            .arg_f32(1.0f32)
            .launch(stream)?;

        let routed_out = ctx.buffers.moe_output();
        ops::dense_gemv(
            ctx.gpu,
            self.dense_gemv_k,
            combined_latent,
            fc2,
            routed_out,
            h,
            latent,
            stream,
        )?;

        ops::residual_add(
            ctx.gpu,
            self.residual_add_k,
            routed_out,
            shared_down_out,
            h,
            stream,
        )?;
        ops::residual_add(ctx.gpu, self.residual_add_k, hidden, routed_out, h, stream)?;

        Ok(())
    }
}
