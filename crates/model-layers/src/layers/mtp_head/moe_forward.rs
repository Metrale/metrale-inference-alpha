// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Single-row FFN forwards of the MTP head that run through
//! `MtpHead::gemv` (the dense FFN and the per-expert MoE), and the deferred
//! draft-token readback.
//!
//! Owner: model-layers (MTP head).
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::MtpHead;
use crate::layer::ForwardContext;
use crate::layers::ops;

impl MtpHead {
    /// 2026-09-25: Read the draft token id that the last `forward_one` call
    /// with `draft_embed_target = Some(..)` stored on the device, with one
    /// 4-byte `copy_d2h`.
    pub(super) fn read_deferred_draft_token(&self, gpu: &dyn GpuBackend) -> Result<u32> {
        let mut buf = [0u8; 4];
        gpu.copy_d2h(self.draft_token_id_dev, &mut buf)?;
        Ok(u32::from_le_bytes(buf))
    }

    /// 2026-09-25: Dense FFN of one row,
    /// `out = down_proj(silu(gate_proj(x)) * up_proj(x))`, into `moe_output`.
    /// Panics when the head has no dense FFN.
    pub(super) fn dense_ffn_forward_generic(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        let h = ctx.config.hidden_size as u32;
        let inter = if ctx.config.intermediate_size > 0 {
            ctx.config.intermediate_size as u32
        } else {
            ctx.config.moe_intermediate_size as u32
        };
        let (gate_w, up_w, down_w) = self
            .dense_ffn_generic
            .as_ref()
            .expect("dense_ffn_forward_generic called without dense_ffn_generic populated");

        let gate_out = ctx.buffers.expert_gate_out();
        let up_out = ctx.buffers.expert_up_out();

        self.gemv(ctx.gpu, input, gate_w, gate_out, inter, h, stream)?;
        self.gemv(ctx.gpu, input, up_w, up_out, inter, h, stream)?;

        ops::moe_silu_mul(
            ctx.gpu,
            self.moe_silu_mul_k.unwrap(),
            gate_out,
            up_out,
            gate_out,
            inter,
            stream,
        )?;

        let output = ctx.buffers.moe_output();
        self.gemv(ctx.gpu, gate_out, down_w, output, h, inter, stream)?;
        Ok(output)
    }

    /// 2026-09-25: MoE of one row over per-expert FP8 or BF16 weights, into
    /// `moe_output`. Synchronizes the stream to read the top-k expert ids on
    /// the host, then runs the GEMVs of each selected expert and of the
    /// shared expert. Panics unless the head holds per-expert weights
    /// (`moe_experts_generic`).
    pub(super) fn moe_forward_generic(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.moe_intermediate_size as u32;
        let num_experts = ctx.config.num_experts as u32;
        let top_k = ctx.config.num_experts_per_tok as u32;

        let gate_logits = ctx.buffers.gate_logits();
        ops::dense_gemv(
            ctx.gpu,
            self.dense_gemv_k.unwrap(),
            input,
            &self.moe_gate,
            gate_logits,
            num_experts,
            h,
            stream,
        )?;

        let scratch = ctx.buffers.scratch();
        let indices_dev = scratch;
        let weights_dev = scratch.offset(top_k as usize * 4);
        ops::moe_topk_softmax(
            ctx.gpu,
            self.moe_topk_k.unwrap(),
            gate_logits,
            indices_dev,
            weights_dev,
            num_experts,
            top_k,
            ctx.config.norm_topk_prob,
            stream,
        )?;

        ctx.gpu.synchronize(stream)?;
        let mut idx_buf = vec![0u8; top_k as usize * 4];
        ctx.gpu.copy_d2h(indices_dev, &mut idx_buf)?;
        let expert_ids: Vec<u32> = (0..top_k as usize)
            .map(|i| {
                u32::from_le_bytes([
                    idx_buf[i * 4],
                    idx_buf[i * 4 + 1],
                    idx_buf[i * 4 + 2],
                    idx_buf[i * 4 + 3],
                ])
            })
            .collect();

        let experts = self.moe_experts_generic.as_ref().unwrap();
        let expert_gate_out = ctx.buffers.expert_gate_out();
        let expert_up_out = ctx.buffers.expert_up_out();
        let expert_down_out = ctx.buffers.expert_down_out();

        for (slot, &eid) in expert_ids.iter().enumerate() {
            let (ref gate_w, ref up_w, _) = experts[eid as usize];
            let g_out = expert_gate_out.offset(slot * inter as usize * 2);
            let u_out = expert_up_out.offset(slot * inter as usize * 2);
            self.gemv(ctx.gpu, input, gate_w, g_out, inter, h, stream)?;
            self.gemv(ctx.gpu, input, up_w, u_out, inter, h, stream)?;
        }

        for slot in 0..top_k as usize {
            let g = expert_gate_out.offset(slot * inter as usize * 2);
            let u = expert_up_out.offset(slot * inter as usize * 2);
            ops::moe_silu_mul(
                ctx.gpu,
                self.moe_silu_mul_k.unwrap(),
                g,
                u,
                g,
                inter,
                stream,
            )?;
        }

        for (slot, &eid) in expert_ids.iter().enumerate() {
            let (_, _, ref down_w) = experts[eid as usize];
            let silu_out = expert_gate_out.offset(slot * inter as usize * 2);
            let d_out = expert_down_out.offset(slot * h as usize * 2);
            self.gemv(ctx.gpu, silu_out, down_w, d_out, h, inter, stream)?;
        }

        let (sh_gate, sh_up, sh_down) = self.moe_shared_generic.as_ref().unwrap();
        let shared_gate_scratch = ctx.buffers.logits();
        let shared_up_scratch = ctx.buffers.ssm_qkvz();
        self.gemv(
            ctx.gpu,
            input,
            sh_gate,
            shared_gate_scratch,
            inter,
            h,
            stream,
        )?;
        self.gemv(ctx.gpu, input, sh_up, shared_up_scratch, inter, h, stream)?;
        ops::moe_silu_mul(
            ctx.gpu,
            self.moe_silu_mul_k.unwrap(),
            shared_gate_scratch,
            shared_up_scratch,
            shared_gate_scratch,
            inter,
            stream,
        )?;
        let shared_out = ctx.buffers.attn_output();
        self.gemv(
            ctx.gpu,
            shared_gate_scratch,
            sh_down,
            shared_out,
            h,
            inter,
            stream,
        )?;

        let output = ctx.buffers.moe_output();
        ops::moe_weighted_sum_blend(
            ctx.gpu,
            self.moe_weighted_sum_blend_k.unwrap(),
            output,
            expert_down_out,
            weights_dev,
            shared_out,
            input,
            self.shared_expert_gate.weight,
            h,
            top_k,
            h,
            stream,
        )?;

        Ok(output)
    }
}
