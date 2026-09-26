// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The per-head gate of `prefill_attention_paged`: one scalar per head from the normed
//! input, applied to that head's attention output with the layer's activation.
//!
//! Owner: model-layers (attention).
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::super::super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::weight_map::DenseWeight;

impl Qwen3AttentionLayer {
    /// 2026-09-26: Called by `prefill_attention_paged` when the layer has `head_gate_weight`
    /// (`g_proj`).
    pub(super) fn prefill_paged_head_gate(
        &self,
        g_proj: &DenseWeight,
        normed: DevicePtr,
        q_contiguous: DevicePtr,
        attn_out: DevicePtr,
        n: u32,
        nq: u32,
        hd: u32,
        h: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let gate_buf = q_contiguous; // 2026-09-25: nothing reads Q after attention and QSA
        // 2026-09-25: `normed [n, H] x g_proj^T [H, nq] -> gate_buf [n, nq]`,
        // through cuBLASLt when the `attn` scope is armed, otherwise
        // `dense_gemm_tc`.
        if ctx.dispatch.cublas.attn {
            ops::cublas_bf16_proj_dense(normed, g_proj.weight, gate_buf, n, nq, h, stream)?;
        } else {
            ops::dense_gemm_tc(
                ctx.gpu,
                self.dense_gemm_tc_k,
                normed,
                g_proj,
                gate_buf,
                n,
                nq,
                h,
                stream,
            )?;
        }
        match self.head_gate_activation {
            super::super::super::types::HeadGateActivation::Sigmoid => {
                ops::sigmoid_gate_mul_head_broadcast(
                    ctx.gpu,
                    self.sigmoid_gate_head_broadcast_k,
                    attn_out,
                    gate_buf,
                    attn_out,
                    nq,
                    hd,
                    n,
                    stream,
                )?;
            }
            super::super::super::types::HeadGateActivation::Softplus => {
                ops::softplus_gate_mul_head_broadcast(
                    ctx.gpu,
                    self.softplus_gate_head_broadcast_k,
                    attn_out,
                    gate_buf,
                    attn_out,
                    nq,
                    hd,
                    n,
                    stream,
                )?;
            }
        }
        Ok(())
    }
}
