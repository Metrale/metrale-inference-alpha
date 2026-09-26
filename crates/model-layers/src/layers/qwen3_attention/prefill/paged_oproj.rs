// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The O projection of `prefill_attention_paged`, `[n, nq*hd]` to
//! `[n, h]` in the `norm_output` buffer, with the weight-format dispatch, the
//! LoRA delta and the `METRALE_OP_DUMP` hook.
//!
//! Owner: model-layers (qwen3 attention).
//! Invariants: none beyond the types.
//!
//! `prefill/alloc_tests.rs` checks that an FP8 O weight with
//! `METRALE_CUBLAS_GEMM=attn` allocates nothing on the first call.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

impl Qwen3AttentionLayer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_attention_paged_oproj(
        &self,
        attn_out: DevicePtr,
        n: u32,
        h: u32,
        nq: u32,
        hd: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        let o_out = ctx.buffers.norm_output();
        // 2026-09-25: A packed Q2_0 weight goes through `try_q2_prefill`.
        if let Some(r) =
            self.try_q2_prefill(ctx, self.o_weight.as_ref(), attn_out, o_out, n, stream)
        {
            r?;
            return Ok(o_out);
        }
        let force_w8a8 = ctx.dispatch.fp8_blockscaled_prefill;
        // 2026-09-25: Opt-in W4A4, under the same `METRALE_ATTN_W4A4` presence
        // check as the Q/K/V projections (`paged_qkv.rs`): quantize `attn_out`
        // to NVFP4 and multiply by the NVFP4 `attn.o_proj`.
        let w4a4 = n >= 256
            && self.w4a4_gemm_k.0 != 0
            && self.quantize_nvfp4_k.0 != 0
            && ctx.buffers.fp8_act_bytes() >= (n as usize) * (nq as usize) * (hd as usize)
            && std::env::var("METRALE_ATTN_W4A4").is_ok();
        if w4a4 {
            let kd = nq * hd;
            let a4 = ctx.buffers.fp8_act();
            let a4_sf = a4.offset((n as usize) * (kd as usize) / 2);
            ops::quantize_bf16_to_nvfp4(
                ctx.gpu,
                self.quantize_nvfp4_k,
                attn_out,
                a4,
                a4_sf,
                n,
                kd,
                stream,
            )?;
            ops::w4a4_gemm_mfast(
                ctx.gpu,
                self.w4a4_gemm_k,
                a4,
                a4_sf,
                &self.attn.o_proj,
                o_out,
                n,
                h,
                kd,
                stream,
            )?;
        } else if ctx.dispatch.cutlass_nvfp4_attn_o
            && let Some(ref nvfp4_t) = self.o_nvfp4_t
        {
            ops::log_cutlass_nvfp4_route(ctx.gpu, "attn_o", n, h, nq * hd);
            ops::cutlass_nvfp4_proj(ctx, attn_out, nvfp4_t, o_out, n, h, nq * hd, stream)?;
        } else if ctx.dispatch.cutlass_nvfp4_attn_o
            && let Some(fp8w) = self.o_weight.as_ref().and_then(|w| w.as_fp8())
        {
            ops::log_cutlass_nvfp4_route(ctx.gpu, "attn_o", n, h, nq * hd);
            ops::cutlass_nvfp4_proj_from_fp8(ctx, attn_out, fp8w, o_out, n, h, nq * hd, stream)?;
        // 2026-09-25: With `METRALE_CUBLAS_GEMM=attn` an FP8 O weight reaches
        // cuBLASLt through the W8A8 route below (`prefill_w8a8.rs`), which
        // neither dequantizes nor allocates.
        } else if force_w8a8
            && let Some(fp8w) = self.o_weight.as_ref().and_then(|w| w.as_fp8())
            && self.per_token_group_quant_fp8_k.available()
            && self.fp8_gemm_t_blockscaled_k.0 != 0
        {
            // 2026-09-25: C[M, N] = A[M, K] @ B[N, K]ᵀ with A = `attn_out`
            // `[n, nq*hd]`, B = the FP8 O weight `[h, nq*hd]`, C = `o_out` `[n, h]`.
            let m = n as usize;
            let k_dim = (nq * hd) as usize;
            let n_out = h as usize;
            // 2026-09-25: Arena scratch; the quantization and the GEMM run on one
            // stream, so no host sync is needed.
            let a_fp8_buf = ctx.buffers.fp8_act();
            let a_scale_buf = ctx.buffers.fp8_act_scale();
            debug_assert!(m * k_dim <= ctx.buffers.fp8_act_bytes());
            ops::per_token_group_quant_fp8(
                ctx.gpu,
                self.per_token_group_quant_fp8_k,
                attn_out,
                a_fp8_buf,
                a_scale_buf,
                n,
                nq * hd,
                stream,
            )?;
            self.attn_prefill_w8a8_gemm(
                ctx,
                a_fp8_buf,
                a_scale_buf,
                fp8w,
                o_out,
                ctx.buffers.norm_output_bytes(),
                n,
                n_out as u32,
                k_dim as u32,
                stream,
            )?;
        } else if let Some(ref fp8t) = self.o_fp8w_t
            && self.w8a16_gemm_t_pipelined_k.0 != 0
        {
            // 2026-09-25: The transposed FP8 weight through the pipelined kernel
            // when it is loaded; otherwise the non-pipelined `w8a16_gemm_t` below.
            ops::w8a16_gemm_t_pipelined(
                ctx.gpu,
                self.w8a16_gemm_t_pipelined_k,
                attn_out,
                fp8t.weight_t,
                fp8t.scale_t,
                o_out,
                n,
                h,
                nq * hd,
                stream,
            )?;
        } else if let Some(ref fp8t) = self.o_fp8w_t
            && self.w8a16_gemm_t_k.0 != 0
        {
            ops::w8a16_gemm_t(
                ctx.gpu,
                self.w8a16_gemm_t_k,
                attn_out,
                fp8t.weight_t,
                fp8t.scale_t,
                o_out,
                n,
                h,
                nq * hd,
                stream,
            )?;
        } else if self.o_weight.as_ref().and_then(|w| w.as_fp8()).is_some()
            && self.w8a16_gemm_k.0 != 0
        {
            let fp8w = self.o_weight.as_ref().and_then(|w| w.as_fp8()).unwrap();
            ops::w8a16_gemm(
                ctx.gpu,
                self.w8a16_gemm_k,
                attn_out,
                fp8w.weight,
                fp8w.row_scale,
                o_out,
                n,
                h,
                nq * hd,
                stream,
            )?;
        } else if let Some(fp8) = self.o_fp8 {
            if n > 128 {
                ops::fp8_gemm_n128_m128(
                    ctx.gpu,
                    self.fp8_gemm_t_m128_k,
                    attn_out,
                    fp8,
                    o_out,
                    n,
                    h,
                    nq * hd,
                    stream,
                )?;
            } else {
                ops::fp8_gemm_n128(
                    ctx.gpu,
                    self.fp8_gemm_k,
                    attn_out,
                    fp8,
                    o_out,
                    n,
                    h,
                    nq * hd,
                    stream,
                )?;
            }
        } else if let Some(ref nvfp4_t) = self.o_nvfp4_t {
            if n > 128 {
                self.w4a16_gemm_m128_dispatch(
                    ctx.gpu,
                    ctx.dispatch,
                    attn_out,
                    nvfp4_t,
                    o_out,
                    n,
                    h,
                    nq * hd,
                    stream,
                )?;
            } else {
                ops::w4a16_gemm_n128(
                    ctx.gpu,
                    self.w4a16_gemm_t_k,
                    attn_out,
                    nvfp4_t,
                    o_out,
                    n,
                    h,
                    nq * hd,
                    stream,
                )?;
            }
        } else if let Some(o_bf16) = self.o_dense_bf16.as_ref() {
            // 2026-09-25: A BF16 O weight installed by the loader
            // (`set_o_dense_bf16`): cuBLASLt when `METRALE_CUBLAS_GEMM` names
            // `attn` and n > 1, else the pipelined GEMM when loaded, else
            // `dense_gemm`.
            if ctx.dispatch.cublas.attn && n > 1 {
                ops::cublas_bf16_proj_dense(attn_out, o_bf16.weight, o_out, n, h, nq * hd, stream)?;
            } else if self.dense_gemm_pipelined_k.0 != 0 {
                ops::dense_gemm_bf16_pipelined(
                    ctx.gpu,
                    self.dense_gemm_pipelined_k,
                    attn_out,
                    o_bf16,
                    o_out,
                    n,
                    h,
                    nq * hd,
                    stream,
                )?;
            } else {
                ops::dense_gemm(
                    ctx.gpu,
                    self.dense_gemm_k,
                    attn_out,
                    o_bf16,
                    o_out,
                    n,
                    h,
                    nq * hd,
                    stream,
                )?;
            }
        } else {
            ops::w4a16_gemm(
                ctx.gpu,
                self.w4a16_gemm_k,
                attn_out,
                &self.attn.o_proj,
                o_out,
                n,
                h,
                nq * hd,
                stream,
            )?;
        }
        // 2026-09-25: LoRA delta, o_out[n, h] += scale * (attn_out[n, nq*hd] @ Aᵀ) @ Bᵀ,
        // before the op dump, so dumps show the adapted output.
        if let Some(ref lw) = self.lora
            && let Some(ref pair) = lw.o
        {
            debug_assert_eq!(pair.k_in, nq * hd);
            debug_assert_eq!(pair.n_out, h);
            // 2026-09-25: As in `paged_qkv.rs`: a prefill routed to a non-active
            // slot folds that slot's O pair (indexed by the global `lw.layer_idx`)
            // through `apply_lora_delta`. It is checked before the bgmv branch.
            let routed_pair = ctx.routed_lora_layers.and_then(|ls| {
                crate::lora::select_routed_pair(ls, lw.layer_idx, crate::lora::LoraModule::OProj)
            });
            // 2026-09-25: With a per-request slot buffer and an O route, the bgmv
            // runs; unlike the Q/K/V projections, it needs no
            // `METRALE_LORA_PREFILL_BGMV` here. Otherwise the installed pair folds.
            let seq_slot = ctx
                .attn_metadata
                .map(|m| m.seq_slot)
                .unwrap_or(DevicePtr(0));
            if let Some(routed_pair) = routed_pair {
                debug_assert_eq!(routed_pair.k_in, nq * hd);
                debug_assert_eq!(routed_pair.n_out, h);
                ops::lora_delta::apply_lora_delta(
                    ctx.gpu,
                    &lw.kernels,
                    routed_pair,
                    attn_out,
                    o_out,
                    n,
                    ctx.buffers.lora_xa(),
                    ctx.buffers.lora_delta(),
                    stream,
                )?;
            } else if seq_slot.0 != 0
                && let Some(ref route) = lw.o_route
            {
                ops::lora_delta::apply_lora_bgmv(
                    ctx.gpu,
                    &lw.kernels,
                    route,
                    attn_out,
                    o_out,
                    seq_slot,
                    n,
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
                    attn_out,
                    o_out,
                    n,
                    ctx.buffers.lora_xa(),
                    ctx.buffers.lora_delta(),
                    stream,
                )?;
            }
        }
        // 2026-09-25: `METRALE_OP_DUMP`: the last token's O projection row.
        let bf16 = 2usize;
        let num_tokens = n as usize;
        if num_tokens > 0 {
            super::super::op_dump::dump_bf16(
                ctx.gpu,
                o_out,
                (num_tokens - 1) * h as usize * bf16,
                h as usize,
                self.attn_layer_idx,
                "o_proj",
                stream,
            )?;
        }
        Ok(o_out)
    }
}
