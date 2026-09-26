// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The q, k and v LoRA deltas of single-token `attention_forward`.
//!
//! Owner: model-layers attention decode.
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::super::super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

impl Qwen3AttentionLayer {
    /// 2026-09-26: Fold the k_proj and v_proj LoRA deltas into `k_out` and `v_out`; a no-op
    /// without a resident adapter.
    pub(super) fn apply_kv_lora(
        &self,
        ctx: &ForwardContext,
        normed: DevicePtr,
        k_out: DevicePtr,
        v_out: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        if let Some(ref lw) = self.lora {
            // 2026-09-25: With a per-sequence slot buffer (`seq_slot != 0`) and a routing table,
            // the delta for this request's adapter goes through the bgmv; otherwise the installed
            // active pair is applied.
            let seq_slot = ctx
                .attn_metadata
                .map(|m| m.seq_slot)
                .unwrap_or(DevicePtr(0));
            if let Some(ref pair) = lw.k {
                if seq_slot.0 != 0
                    && let Some(ref route) = lw.k_route
                {
                    ops::lora_delta::apply_lora_bgmv(
                        ctx.gpu,
                        &lw.kernels,
                        route,
                        normed,
                        k_out,
                        seq_slot,
                        1,
                        pair.k_in,
                        pair.n_out,
                        ctx.buffers.lora_xa(),
                        stream,
                    )?;
                } else {
                    ops::lora_delta::apply_lora_delta(
                        ctx.gpu,
                        &lw.kernels,
                        pair,
                        normed,
                        k_out,
                        1,
                        ctx.buffers.lora_xa(),
                        ctx.buffers.lora_delta(),
                        stream,
                    )?;
                }
            }
            if let Some(ref pair) = lw.v {
                if seq_slot.0 != 0
                    && let Some(ref route) = lw.v_route
                {
                    ops::lora_delta::apply_lora_bgmv(
                        ctx.gpu,
                        &lw.kernels,
                        route,
                        normed,
                        v_out,
                        seq_slot,
                        1,
                        pair.k_in,
                        pair.n_out,
                        ctx.buffers.lora_xa(),
                        stream,
                    )?;
                } else {
                    ops::lora_delta::apply_lora_delta(
                        ctx.gpu,
                        &lw.kernels,
                        pair,
                        normed,
                        v_out,
                        1,
                        ctx.buffers.lora_xa(),
                        ctx.buffers.lora_delta(),
                        stream,
                    )?;
                }
            }
        }
        Ok(())
    }

    /// 2026-09-25: Fold the q_proj LoRA delta into the raw q_proj output at `q_out` (on a gated
    /// model the interleaved `[Q|gate]`), before `deinterleave_qg`. Same routing as the K/V deltas:
    /// the bgmv when this step carries a `seq_slot` and the module has a route, else the installed
    /// active pair. A no-op when no q adapter is resident.
    pub(super) fn apply_q_lora(
        &self,
        ctx: &ForwardContext,
        normed: DevicePtr,
        q_out: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let Some(ref lw) = self.lora else {
            return Ok(());
        };
        let Some(ref pair) = lw.q else {
            return Ok(());
        };
        let seq_slot = ctx
            .attn_metadata
            .map(|m| m.seq_slot)
            .unwrap_or(DevicePtr(0));
        if seq_slot.0 != 0
            && let Some(ref route) = lw.q_route
        {
            ops::lora_delta::apply_lora_bgmv(
                ctx.gpu,
                &lw.kernels,
                route,
                normed,
                q_out,
                seq_slot,
                1,
                pair.k_in,
                pair.n_out,
                ctx.buffers.lora_xa(),
                stream,
            )?;
        } else {
            ops::lora_delta::apply_lora_delta(
                ctx.gpu,
                &lw.kernels,
                pair,
                normed,
                q_out,
                1,
                ctx.buffers.lora_xa(),
                ctx.buffers.lora_delta(),
                stream,
            )?;
        }
        Ok(())
    }
}
