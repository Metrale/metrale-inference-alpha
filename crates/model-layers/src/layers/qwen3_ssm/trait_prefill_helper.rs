// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The GDN out_proj dispatch for prefill
//! (`prefill_out_proj_dispatch`), and the TP reduce plus LoRA step that every
//! GDN path runs after its out_proj (`ssm_tp_all_reduce`).
//!
//! Owner: model-layers, GDN/SSM layer (`qwen3_ssm`).
//! Invariants:
//! - `ssm_tp_all_reduce` enqueues the TP all-reduce (only when
//!   `tp_world_size > 1` and a communicator is present) before it applies the
//!   out_proj LoRA delta; an all-reduce error returns before the delta.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::Qwen3SsmLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

impl Qwen3SsmLayer {
    /// 2026-09-25: Sum the row-parallel `out_proj` partials across TP ranks,
    /// then apply the `out_proj` LoRA delta.
    ///
    /// Each rank runs the GDN over its own value heads and projects with its
    /// slice of `out_proj`, so its `[num_tokens, h]` BF16 buffer holds a
    /// partial sum. The all-reduce runs only when `tp_world_size > 1` and a
    /// communicator is present. The LoRA delta goes after it: added before,
    /// it would be summed once per rank.
    ///
    /// Every GDN path ends its out_proj here: `ssm_forward`,
    /// `decode_batched_inner`, `try_decode_multi_seq_ssm_batched`,
    /// `prefill_block` and `prefill_phase3_inner`.
    pub(super) fn ssm_tp_all_reduce(
        &self,
        out_proj_buf: DevicePtr,
        normed_out: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if ctx.config.tp_world_size > 1
            && let Some(comm) = ctx.comm
        {
            let bytes = num_tokens * ctx.config.hidden_size * 2;
            comm.all_reduce_async(out_proj_buf.0, bytes, stream)?;
        }
        self.apply_lora_out_proj(ctx, normed_out, out_proj_buf, num_tokens as u32, stream)
    }

    pub(super) fn prefill_out_proj_dispatch(
        &self,
        ctx: &ForwardContext,
        normed_out_buf: DevicePtr,
        out_proj_buf: DevicePtr,
        k: u32,
        h: usize,
        value_dim: usize,
        stream: u64,
    ) -> Result<()> {
        let force_w8a8 = matches!(std::env::var("METRALE_FP8_W8A8").ok().as_deref(), Some("1"));
        // 2026-09-25: Per-row FP8 weights (`out_proj_fp8w_rowwise`), dequantised
        // once per layer to BF16 in the arena slab (`rowwise_bf16.rs`) and run
        // on cuBLASLt. First, because it is the only arm that does not
        // re-quantise the weight.
        if let Some(ref fp8w) = self.out_proj_fp8w_rowwise {
            let w_bf16 = self.rowwise_out_proj_bf16(ctx, fp8w, stream)?;
            return ops::cublas_bf16_proj_dense(
                normed_out_buf,
                w_bf16,
                out_proj_buf,
                k,
                h as u32,
                value_dim as u32,
                stream,
            );
        }
        if ctx.dispatch.cutlass_nvfp4_ssm_out
            && let Some(ref nvfp4_t) = self.out_proj_nvfp4_t
        {
            ops::log_cutlass_nvfp4_route(ctx.gpu, "ssm_out_nvfp4", k, h as u32, value_dim as u32);
            ops::cutlass_nvfp4_proj(
                ctx,
                normed_out_buf,
                nvfp4_t,
                out_proj_buf,
                k,
                h as u32,
                value_dim as u32,
                stream,
            )
        } else if ctx.dispatch.cutlass_nvfp4_ssm_out
            && let Some(ref fp8w) = self.out_proj_fp8w
        {
            ops::log_cutlass_nvfp4_route(ctx.gpu, "ssm_out_fp8pack", k, h as u32, value_dim as u32);
            ops::cutlass_nvfp4_proj_from_fp8(
                ctx,
                normed_out_buf,
                fp8w,
                out_proj_buf,
                k,
                h as u32,
                value_dim as u32,
                stream,
            )
        } else if let Some(ref dense_out) = self.out_proj_dense {
            ops::dense_gemm_bf16_pipelined(
                ctx.gpu,
                self.dense_gemm_pipelined_k,
                normed_out_buf,
                dense_out,
                out_proj_buf,
                k,
                h as u32,
                value_dim as u32,
                stream,
            )
        // 2026-09-25: cuBLASLt W8A8 (`prefill_out_w8a8.rs`) when
        // `prefill_out_proj_w8a8_selected` accepts, which needs `ssm` in the
        // cuBLAS scope. It precedes the METRALE_FP8_W8A8 arm below, which runs
        // the in-tree W8A8 kernel.
        } else if let Some(ref fp8w) = self.out_proj_fp8w
            && self.prefill_out_proj_w8a8_selected(ctx, k, h as u32, value_dim as u32, fp8w)
        {
            self.log_out_proj_prefill_route(ctx, true);
            self.prefill_out_proj_w8a8_cublas(
                ctx,
                normed_out_buf,
                fp8w,
                out_proj_buf,
                k,
                h as u32,
                value_dim as u32,
                stream,
            )
        } else if force_w8a8
            && let Some(ref fp8w) = self.out_proj_fp8w
            && self.per_token_group_quant_fp8_k.available()
            && self.fp8_gemm_t_blockscaled_k.0 != 0
        {
            tracing::debug!(
                "ssm prefill: out_proj via W8A8+FP32-epilogue (M={k} K={h} N={value_dim})"
            );
            let m = k as usize;
            let k_dim = h;
            // 2026-09-25: The activation quant uses the arena's `fp8_act`
            // scratch; nothing is allocated here.
            let a_fp8_buf = ctx.buffers.fp8_act();
            let a_scale_buf = ctx.buffers.fp8_act_scale();
            debug_assert!(m * k_dim <= ctx.buffers.fp8_act_bytes());
            ops::per_token_group_quant_fp8(
                ctx.gpu,
                self.per_token_group_quant_fp8_k,
                normed_out_buf,
                a_fp8_buf,
                a_scale_buf,
                k,
                k_dim as u32,
                stream,
            )?;
            ops::fp8_gemm_t_blockscaled(
                ctx.gpu,
                self.fp8_gemm_t_blockscaled_k,
                a_fp8_buf,
                a_scale_buf,
                fp8w.weight,
                fp8w.row_scale,
                out_proj_buf,
                k,
                value_dim as u32,
                h as u32,
                stream,
            )?;
            Ok(())
        } else if let Some(ref fp8w) = self.out_proj_fp8w
            && self.w8a16_gemm_pipelined_k.0 != 0
        {
            // 2026-09-25: Log once that W8A16 served out_proj, so a missing
            // W8A8 line is not read as a lost log.
            self.log_out_proj_prefill_route(ctx, false);
            ops::w8a16_gemm_pipelined(
                ctx.gpu,
                self.w8a16_gemm_pipelined_k,
                normed_out_buf,
                fp8w.weight,
                fp8w.row_scale,
                out_proj_buf,
                k,
                h as u32,
                value_dim as u32,
                stream,
            )
        } else if let Some(ref fp8w) = self.out_proj_fp8w
            && self.w8a16_gemm_k.0 != 0
        {
            ops::w8a16_gemm(
                ctx.gpu,
                self.w8a16_gemm_k,
                normed_out_buf,
                fp8w.weight,
                fp8w.row_scale,
                out_proj_buf,
                k,
                h as u32,
                value_dim as u32,
                stream,
            )
        } else if let Some(fp8) = self.out_proj_fp8 {
            if k > 128 {
                ops::fp8_gemm_n128_m128(
                    ctx.gpu,
                    self.fp8_gemm_t_m128_k,
                    normed_out_buf,
                    fp8,
                    out_proj_buf,
                    k,
                    h as u32,
                    value_dim as u32,
                    stream,
                )
            } else {
                ops::fp8_gemm_n128(
                    ctx.gpu,
                    self.fp8_gemm_k,
                    normed_out_buf,
                    fp8,
                    out_proj_buf,
                    k,
                    h as u32,
                    value_dim as u32,
                    stream,
                )
            }
        } else if let Some(ref nvfp4_t) = self.out_proj_nvfp4_t {
            ops::w4a16_gemm_n128(
                ctx.gpu,
                self.w4a16_gemm_t_k,
                normed_out_buf,
                nvfp4_t,
                out_proj_buf,
                k,
                h as u32,
                value_dim as u32,
                stream,
            )
        } else {
            ops::w4a16_gemm(
                ctx.gpu,
                self.w4a16_gemm_k,
                normed_out_buf,
                &self.ssm.out_proj,
                out_proj_buf,
                k,
                h as u32,
                value_dim as u32,
                stream,
            )
        }
        .map_err(|e| anyhow::anyhow!("ssm prefill: out_proj GEMM failed: {e}"))
    }
}
