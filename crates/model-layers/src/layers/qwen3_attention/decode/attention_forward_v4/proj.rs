// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The weight-format dispatch of the `attention_forward_v4` projections wq_a, wq_b,
//! wkv_a and wo_b: an NVFP4 GEMV, else an FP8 (W8A16) GEMV, else a BF16 dense GEMV.
//!
//! Owner: model-layers attention decode.
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::super::super::{MlaWeights, Qwen3AttentionLayer};
use crate::layer::ForwardContext;
use crate::layers::ops;

impl Qwen3AttentionLayer {
    /// 2026-09-26: `normed [h]` -> `q_latent [q_lora]`.
    pub(super) fn v4_wq_a(
        &self,
        ctx: &ForwardContext,
        mla: &MlaWeights,
        normed: DevicePtr,
        q_latent: DevicePtr,
        q_lora: u32,
        h: u32,
        stream: u64,
    ) -> Result<()> {
        if let Some(ref wqa_nvfp4) = mla.wq_a_nvfp4 {
            self.nvfp4_decode_gemv(
                ctx.gpu,
                ctx.levers.gemv_sw,
                normed,
                wqa_nvfp4,
                q_latent,
                q_lora,
                h,
                stream,
            )
        } else if let Some(ref wqa_fp8) = mla.wq_a_fp8 {
            ops::w8a16_gemv(
                ctx.gpu,
                self.w8a16_gemv_k,
                normed,
                wqa_fp8.weight,
                wqa_fp8.row_scale,
                q_latent,
                q_lora,
                h,
                stream,
            )
        } else {
            ops::dense_gemv(
                ctx.gpu,
                self.dense_gemv_k,
                normed,
                &mla.wq_a,
                q_latent,
                q_lora,
                h,
                stream,
            )
        }
    }

    /// 2026-09-26: `q_latent [q_lora]` -> `q_out [q_dim]`.
    pub(super) fn v4_wq_b(
        &self,
        ctx: &ForwardContext,
        mla: &MlaWeights,
        q_latent: DevicePtr,
        q_out: DevicePtr,
        q_dim: u32,
        q_lora: u32,
        stream: u64,
    ) -> Result<()> {
        if let Some(ref wqb_nvfp4) = mla.wq_b_nvfp4 {
            self.nvfp4_decode_gemv(
                ctx.gpu,
                ctx.levers.gemv_sw,
                q_latent,
                wqb_nvfp4,
                q_out,
                q_dim,
                q_lora,
                stream,
            )
        } else if let Some(ref wqb_fp8) = mla.wq_b_fp8 {
            ops::w8a16_gemv(
                ctx.gpu,
                self.w8a16_gemv_k,
                q_latent,
                wqb_fp8.weight,
                wqb_fp8.row_scale,
                q_out,
                q_dim,
                q_lora,
                stream,
            )
        } else {
            ops::dense_gemv(
                ctx.gpu,
                self.dense_gemv_k,
                q_latent,
                &mla.wq_b,
                q_out,
                q_dim,
                q_lora,
                stream,
            )
        }
    }

    /// 2026-09-26: `normed [h]` -> `k_out [kv_dim]`.
    pub(super) fn v4_wkv_a(
        &self,
        ctx: &ForwardContext,
        mla: &MlaWeights,
        normed: DevicePtr,
        k_out: DevicePtr,
        kv_dim: u32,
        h: u32,
        stream: u64,
    ) -> Result<()> {
        if let Some(ref wkva_nvfp4) = mla.wkv_a_nvfp4 {
            self.nvfp4_decode_gemv(
                ctx.gpu,
                ctx.levers.gemv_sw,
                normed,
                wkva_nvfp4,
                k_out,
                kv_dim,
                h,
                stream,
            )
        } else if let Some(ref wkva_fp8) = mla.wkv_a_fp8 {
            ops::w8a16_gemv(
                ctx.gpu,
                self.w8a16_gemv_k,
                normed,
                wkva_fp8.weight,
                wkva_fp8.row_scale,
                k_out,
                kv_dim,
                h,
                stream,
            )
        } else {
            ops::dense_gemv(
                ctx.gpu,
                self.dense_gemv_k,
                normed,
                &mla.wkv_a,
                k_out,
                kv_dim,
                h,
                stream,
            )
        }
    }

    /// 2026-09-26: `o_latent [latent_dim]` -> `o_out [h]` (no NVFP4 arm).
    pub(super) fn v4_wo_b(
        &self,
        ctx: &ForwardContext,
        mla: &MlaWeights,
        o_latent: DevicePtr,
        o_out: DevicePtr,
        h: u32,
        latent_dim: u32,
        stream: u64,
    ) -> Result<()> {
        if let Some(ref wob_fp8) = mla.wo_b_fp8 {
            ops::w8a16_gemv(
                ctx.gpu,
                self.w8a16_gemv_k,
                o_latent,
                wob_fp8.weight,
                wob_fp8.row_scale,
                o_out,
                h,
                latent_dim,
                stream,
            )
        } else {
            ops::dense_gemv(
                ctx.gpu,
                self.dense_gemv_k,
                o_latent,
                &mla.wo_b,
                o_out,
                h,
                latent_dim,
                stream,
            )
        }
    }
}
