// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `ModelResource` for `BufferArena`: free every buffer the arena owns.
//!
//! Owner: gpu-runtime.
//! Invariants:
//! - `release` calls `free` on every pointer field, whatever earlier frees
//!   return, then sets every pointer field to `DevicePtr::NULL` and returns the
//!   first error.
//! - A second `release` frees only NULL pointers.

use super::*;

/// 2026-09-25: The destructure in `release` names every field, with no `..`, so
/// a field added to `BufferArena` does not compile until it is handled here.
/// Free the new buffer rather than adding a wildcard: a missed free is a leak
/// that shows only as the next model failing to fit.
impl metrale_core::scope::ModelResource<dyn GpuBackend> for BufferArena {
    fn label(&self) -> &'static str {
        "buffer arena"
    }

    fn release(&mut self, gpu: &dyn GpuBackend) -> anyhow::Result<()> {
        let Self {
            // 2026-09-25: Not allocations; named so the destructure stays exhaustive.
            sizes: _,
            max_batch_tokens: _,
            decode_meta: _,
            hidden_states,
            residual,
            norm_output,
            qkv_output,
            attn_output,
            gate_logits,
            gate_logits_f32,
            moe_router_in_f32,
            moe_output,
            logits,
            ssm_qkvz,
            ssm_ba,
            ssm_deinterleaved,
            ssm_gates,
            ssm_conv_out_f32,
            scratch,
            expert_gate_out,
            expert_up_out,
            expert_down_out,
            splitk_workspace,
            o_latent,
            norm_unit_w,
            hc_streams,
            hc_post,
            hc_comb,
            hc_lowrank_scratch,
            qsa_select_scratch,
            gdn_fla_scratch,
            ssd_scratch,
            token_ids,
            ffn_act_q8,
            ffn_act_a,
            ffn_act_scale,
            ffn_act_scale_kmajor,
            ffn_gate_up_fused,
            fp8_act,
            moe_fp8_scratch,
            fp8_act_scale,
            fp8_act_scale_kmajor,
            lora_xa,
            lora_delta,
            lora_hact,
            lora_seq_slot,
            q2_dequant_scratch,
            q2_act_q8,
            ssm_rowwise_w_bf16,
            ssm_rowwise_w_bf16_used: _,
        } = self;
        // 2026-09-25: Free every pointer, then set it to NULL, so a second call
        // frees only NULL pointers: `ModelResource::release` must be idempotent.
        let owned = [
            *hidden_states,
            *residual,
            *norm_output,
            *qkv_output,
            *attn_output,
            *gate_logits,
            *gate_logits_f32,
            *moe_router_in_f32,
            *moe_output,
            *logits,
            *ssm_qkvz,
            *ssm_ba,
            *ssm_deinterleaved,
            *ssm_gates,
            *ssm_conv_out_f32,
            *scratch,
            *expert_gate_out,
            *expert_up_out,
            *expert_down_out,
            *splitk_workspace,
            *o_latent,
            *norm_unit_w,
            *hc_streams,
            *hc_lowrank_scratch,
            *qsa_select_scratch,
            *hc_post,
            *hc_comb,
            *gdn_fla_scratch,
            *ssd_scratch,
            *token_ids,
            *ffn_act_q8,
            *ffn_act_a,
            *ffn_act_scale,
            *ffn_act_scale_kmajor,
            *ffn_gate_up_fused,
            *fp8_act,
            *moe_fp8_scratch,
            *fp8_act_scale,
            *fp8_act_scale_kmajor,
            *lora_xa,
            *lora_delta,
            *lora_hact,
            *lora_seq_slot,
            *q2_dequant_scratch,
            *q2_act_q8,
            *ssm_rowwise_w_bf16,
        ];
        let mut first_error = None;
        for ptr in owned {
            if let Err(e) = gpu.free(ptr)
                && first_error.is_none()
            {
                first_error = Some(e);
            }
        }
        *hidden_states = DevicePtr::NULL;
        *residual = DevicePtr::NULL;
        *norm_output = DevicePtr::NULL;
        *qkv_output = DevicePtr::NULL;
        *attn_output = DevicePtr::NULL;
        *gate_logits = DevicePtr::NULL;
        *gate_logits_f32 = DevicePtr::NULL;
        *moe_router_in_f32 = DevicePtr::NULL;
        *moe_output = DevicePtr::NULL;
        *logits = DevicePtr::NULL;
        *ssm_qkvz = DevicePtr::NULL;
        *ssm_ba = DevicePtr::NULL;
        *ssm_deinterleaved = DevicePtr::NULL;
        *ssm_gates = DevicePtr::NULL;
        *ssm_conv_out_f32 = DevicePtr::NULL;
        *scratch = DevicePtr::NULL;
        *expert_gate_out = DevicePtr::NULL;
        *expert_up_out = DevicePtr::NULL;
        *expert_down_out = DevicePtr::NULL;
        *splitk_workspace = DevicePtr::NULL;
        *o_latent = DevicePtr::NULL;
        *norm_unit_w = DevicePtr::NULL;
        *hc_streams = DevicePtr::NULL;
        *hc_lowrank_scratch = DevicePtr::NULL;
        *qsa_select_scratch = DevicePtr::NULL;
        *hc_post = DevicePtr::NULL;
        *hc_comb = DevicePtr::NULL;
        *gdn_fla_scratch = DevicePtr::NULL;
        *ssd_scratch = DevicePtr::NULL;
        *token_ids = DevicePtr::NULL;
        *ffn_act_q8 = DevicePtr::NULL;
        *ffn_act_a = DevicePtr::NULL;
        *ffn_act_scale = DevicePtr::NULL;
        *ffn_act_scale_kmajor = DevicePtr::NULL;
        *ffn_gate_up_fused = DevicePtr::NULL;
        *fp8_act = DevicePtr::NULL;
        *moe_fp8_scratch = DevicePtr::NULL;
        *fp8_act_scale = DevicePtr::NULL;
        *fp8_act_scale_kmajor = DevicePtr::NULL;
        *lora_xa = DevicePtr::NULL;
        *lora_delta = DevicePtr::NULL;
        *lora_hact = DevicePtr::NULL;
        *lora_seq_slot = DevicePtr::NULL;
        *q2_dequant_scratch = DevicePtr::NULL;
        *q2_act_q8 = DevicePtr::NULL;
        *ssm_rowwise_w_bf16 = DevicePtr::NULL;
        match first_error {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}
