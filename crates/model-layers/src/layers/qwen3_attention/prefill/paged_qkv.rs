// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The Q, K and V projections of `prefill_attention_paged` for a
//! non-MLA layer, with the weight-format dispatch and the LoRA delta.
//!
//! Owner: model-layers (qwen3 attention).
//! Invariants:
//! - Outputs: Q (`q_proj_dim` wide) in `qkv_output`; K at `ssm_qkvz`; V at
//!   `ssm_qkvz + num_tokens * kv_dim`, all BF16.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

/// 2026-09-25: Which projection, and so which of the layer's weights.
pub(super) enum Proj {
    Q,
    K,
    V,
}

impl Qwen3AttentionLayer {
    /// 2026-09-25: Run the Q, K and V projections, in that order, into the
    /// buffers the module header names.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_attention_paged_qkv(
        &self,
        normed: DevicePtr,
        n: u32,
        h: u32,
        nkv: u32,
        hd: u32,
        q_proj_dim: usize,
        kv_dim: usize,
        num_tokens: usize,
        bf16: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        // 2026-09-25: Opt-in W4A4: with `METRALE_ATTN_W4A4` set (any value), at
        // least 256 tokens, the kernels loaded, an NVFP4 q weight and room in
        // `fp8_act`, `normed` is quantized to NVFP4 once and shared by every
        // projection whose weight is NVFP4.
        let w4a4 = n >= 256
            && self.w4a4_gemm_k.0 != 0
            && self.quantize_nvfp4_k.0 != 0
            && self.q_weight.as_ref().and_then(|w| w.as_nvfp4()).is_some()
            && ctx.buffers.fp8_act_bytes() >= (n as usize) * (h as usize)
            && std::env::var("METRALE_ATTN_W4A4").is_ok();
        let a4 = if w4a4 {
            let a4 = ctx.buffers.fp8_act();
            let a4_sf = a4.offset((n as usize) * (h as usize) / 2);
            ops::quantize_bf16_to_nvfp4(
                ctx.gpu,
                self.quantize_nvfp4_k,
                normed,
                a4,
                a4_sf,
                n,
                h,
                stream,
            )?;
            Some((a4, a4_sf))
        } else {
            None
        };
        let qg_out = ctx.buffers.qkv_output();
        self.prefill_one_proj(
            Proj::Q,
            normed,
            qg_out,
            n,
            q_proj_dim as u32,
            h,
            a4,
            ctx,
            stream,
        )?;
        // 2026-09-25: `METRALE_OP_DUMP`: the last token's q_proj row, all
        // `q_proj_dim` values (Q and gate interleaved on a gated layer).
        super::super::op_dump::dump_bf16(
            ctx.gpu,
            qg_out,
            (num_tokens - 1) * q_proj_dim * bf16,
            q_proj_dim,
            self.attn_layer_idx,
            "q_proj_full",
            stream,
        )?;

        let k_contiguous = ctx.buffers.ssm_qkvz();
        self.prefill_one_proj(
            Proj::K,
            normed,
            k_contiguous,
            n,
            nkv * hd,
            h,
            a4,
            ctx,
            stream,
        )?;
        super::super::op_dump::dump_bf16(
            ctx.gpu,
            k_contiguous,
            (num_tokens - 1) * kv_dim * bf16,
            kv_dim,
            self.attn_layer_idx,
            "k_proj",
            stream,
        )?;

        let v_contiguous = k_contiguous.offset(num_tokens * kv_dim * bf16);
        self.prefill_one_proj(
            Proj::V,
            normed,
            v_contiguous,
            n,
            nkv * hd,
            h,
            a4,
            ctx,
            stream,
        )?;
        super::super::op_dump::dump_bf16(
            ctx.gpu,
            v_contiguous,
            (num_tokens - 1) * kv_dim * bf16,
            kv_dim,
            self.attn_layer_idx,
            "v_proj",
            stream,
        )?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn prefill_one_proj(
        &self,
        proj: Proj,
        normed: DevicePtr,
        out: DevicePtr,
        n: u32,
        out_dim: u32,
        h: u32,
        a4: Option<(DevicePtr, DevicePtr)>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let (fp8w_t, weight_opt, fp8, nvfp4_t, dense, label) = match proj {
            Proj::Q => (
                self.q_fp8w_t.as_ref(),
                self.q_weight.as_ref(),
                self.q_fp8,
                self.q_nvfp4_t.as_ref(),
                &self.attn.q_proj,
                "attn_q",
            ),
            Proj::K => (
                self.k_fp8w_t.as_ref(),
                self.k_weight.as_ref(),
                self.k_fp8,
                self.k_nvfp4_t.as_ref(),
                &self.attn.k_proj,
                "attn_k",
            ),
            Proj::V => (
                self.v_fp8w_t.as_ref(),
                self.v_weight.as_ref(),
                self.v_fp8,
                self.v_nvfp4_t.as_ref(),
                &self.attn.v_proj,
                "attn_v",
            ),
        };

        // 2026-09-25: A packed Q2_0 weight goes through `try_q2_prefill`
        // (`prefill_weights.rs`).
        if let Some(r) = self.try_q2_prefill(ctx, weight_opt, normed, out, n, stream) {
            return r;
        }

        // 2026-09-25: W4A4: the shared NVFP4 activations times the NVFP4 weight.
        if let (Some((a4p, a4sf)), Some(nvfp4)) = (a4, weight_opt.and_then(|w| w.as_nvfp4())) {
            let _ = label;
            return ops::w4a4_gemm_mfast(
                ctx.gpu,
                self.w4a4_gemm_k,
                a4p,
                a4sf,
                nvfp4,
                out,
                n,
                out_dim,
                h,
                stream,
            );
        }

        let force_w8a8 = ctx.dispatch.fp8_blockscaled_prefill;
        // 2026-09-25: Then, first match: CUTLASS NVFP4 (`cutlass_nvfp4_attn_qkv`)
        // from the transposed NVFP4 or the FP8 weight; W8A8 (FP8 activations,
        // FP32 epilogue) on the non-transposed FP8 weight when
        // `fp8_blockscaled_prefill` holds, quantizing `normed` for this
        // projection; the transposed FP8 GEMMs; `w8a16_gemm`; the `fp8` weight;
        // NVFP4, transposed then plain; BF16.
        if ctx.dispatch.cutlass_nvfp4_attn_qkv(label)
            && let Some(nvfp4_t) = nvfp4_t
        {
            ops::log_cutlass_nvfp4_route(ctx.gpu, label, n, out_dim, h);
            ops::cutlass_nvfp4_proj(ctx, normed, nvfp4_t, out, n, out_dim, h, stream)?;
        } else if ctx.dispatch.cutlass_nvfp4_attn_qkv(label)
            && let Some(fp8w) = weight_opt.and_then(|w| w.as_fp8())
        {
            ops::log_cutlass_nvfp4_route(ctx.gpu, label, n, out_dim, h);
            ops::cutlass_nvfp4_proj_from_fp8(ctx, normed, fp8w, out, n, out_dim, h, stream)?;
        } else if force_w8a8
            && let Some(fp8w) = weight_opt.and_then(|w| w.as_fp8())
            && self.per_token_group_quant_fp8_k.available()
            && self.fp8_gemm_t_blockscaled_k.0 != 0
        {
            let m = n as usize;
            let k_dim = h as usize;
            // 2026-09-25: Arena scratch; nothing is allocated.
            let a_fp8_buf = ctx.buffers.fp8_act();
            let a_scale_buf = ctx.buffers.fp8_act_scale();
            debug_assert!(m * k_dim <= ctx.buffers.fp8_act_bytes());
            ops::per_token_group_quant_fp8(
                ctx.gpu,
                self.per_token_group_quant_fp8_k,
                normed,
                a_fp8_buf,
                a_scale_buf,
                n,
                h,
                stream,
            )?;
            ops::fp8_gemm_t_blockscaled(
                ctx.gpu,
                self.fp8_gemm_t_blockscaled_k,
                a_fp8_buf,
                a_scale_buf,
                fp8w.weight,
                fp8w.row_scale,
                out,
                n,
                out_dim,
                h,
                stream,
            )?;
        } else if let Some(fp8t) = fp8w_t
            && self.w8a16_gemm_t_m128_k.0 != 0
        {
            // 2026-09-25: The transposed FP8 weight (`weight_t`, `scale_t`)
            // through the M128 tile kernel.
            ops::w8a16_gemm_n128_m128(
                ctx.gpu,
                self.w8a16_gemm_t_m128_k,
                normed,
                fp8t.weight_t,
                fp8t.scale_t,
                out,
                n,
                out_dim,
                h,
                stream,
            )?;
        } else if let Some(fp8t) = fp8w_t {
            ops::w8a16_gemm_t(
                ctx.gpu,
                self.w8a16_gemm_t_k,
                normed,
                fp8t.weight_t,
                fp8t.scale_t,
                out,
                n,
                out_dim,
                h,
                stream,
            )?;
        } else if weight_opt.and_then(|w| w.as_fp8()).is_some() && self.w8a16_gemm_k.0 != 0 {
            let fp8w = weight_opt.and_then(|w| w.as_fp8()).unwrap();
            ops::w8a16_gemm(
                ctx.gpu,
                self.w8a16_gemm_k,
                normed,
                fp8w.weight,
                fp8w.row_scale,
                out,
                n,
                out_dim,
                h,
                stream,
            )?;
        } else if let Some(fp8p) = fp8 {
            if n > 128 {
                ops::fp8_gemm_n128_m128(
                    ctx.gpu,
                    self.fp8_gemm_t_m128_k,
                    normed,
                    fp8p,
                    out,
                    n,
                    out_dim,
                    h,
                    stream,
                )?;
            } else {
                ops::fp8_gemm_n128(
                    ctx.gpu,
                    self.fp8_gemm_k,
                    normed,
                    fp8p,
                    out,
                    n,
                    out_dim,
                    h,
                    stream,
                )?;
            }
        } else if let Some(nvfp4_t) = nvfp4_t {
            if n > 128 {
                self.w4a16_gemm_m128_dispatch(
                    ctx.gpu,
                    ctx.dispatch,
                    normed,
                    nvfp4_t,
                    out,
                    n,
                    out_dim,
                    h,
                    stream,
                )?;
            } else {
                ops::w4a16_gemm_n128(
                    ctx.gpu,
                    self.w4a16_gemm_t_k,
                    normed,
                    nvfp4_t,
                    out,
                    n,
                    out_dim,
                    h,
                    stream,
                )?;
            }
        } else if let Some(nvfp4) = weight_opt.and_then(|w| w.as_nvfp4()) {
            ops::w4a16_gemm(
                ctx.gpu,
                self.w4a16_gemm_k,
                normed,
                nvfp4,
                out,
                n,
                out_dim,
                h,
                stream,
            )?;
        } else {
            // 2026-09-25: BF16 `attn.{q,k,v}_proj`: cuBLASLt when
            // `METRALE_CUBLAS_GEMM` names `attn` and n > 1, else the pipelined
            // tensor-core GEMM when loaded, else `dense_gemm`.
            if ctx.dispatch.cublas.attn && n > 1 {
                ops::cublas_bf16_proj_dense(normed, dense.weight, out, n, out_dim, h, stream)?;
            } else if self.dense_gemm_pipelined_k.0 != 0 {
                ops::dense_gemm_bf16_pipelined(
                    ctx.gpu,
                    self.dense_gemm_pipelined_k,
                    normed,
                    dense,
                    out,
                    n,
                    out_dim,
                    h,
                    stream,
                )?;
            } else {
                ops::dense_gemm(
                    ctx.gpu,
                    self.dense_gemm_k,
                    normed,
                    dense,
                    out,
                    n,
                    out_dim,
                    h,
                    stream,
                )?;
            }
        }
        // 2026-09-25: LoRA delta, out += scale * (normed @ Aᵀ) @ Bᵀ. For a gated
        // Q it folds into the interleaved `[Q | gate]` output, before the caller
        // splits it (`paged.rs`). It runs before the caller's `METRALE_OP_DUMP`,
        // so dumps show the adapted output.
        if let Some(ref lw) = self.lora {
            let (pair, route, module) = match proj {
                Proj::Q => (
                    lw.q.as_ref(),
                    lw.q_route.as_ref(),
                    Some(crate::lora::LoraModule::QProj),
                ),
                Proj::K => (
                    lw.k.as_ref(),
                    lw.k_route.as_ref(),
                    Some(crate::lora::LoraModule::KProj),
                ),
                Proj::V => (
                    lw.v.as_ref(),
                    lw.v_route.as_ref(),
                    Some(crate::lora::LoraModule::VProj),
                ),
            };
            if let Some(pair) = pair {
                debug_assert_eq!(pair.k_in, h);
                debug_assert_eq!(pair.n_out, out_dim);
                // 2026-09-25: A prefill routed to a non-active slot
                // (`ctx.routed_lora_layers`) folds that slot's pair for this
                // layer and module through the same `apply_lora_delta` as the
                // active adapter. The pool is indexed by global layer, so the
                // index is `lw.layer_idx`, not `attn_layer_idx`.
                // `select_routed_pair` returns `None` when the slot does not adapt
                // this module; the branches below then run.
                let routed_pair = ctx.routed_lora_layers.and_then(|ls| {
                    module.and_then(|m| crate::lora::select_routed_pair(ls, lw.layer_idx, m))
                });
                // 2026-09-25: The per-request slot buffer (`seq_slot`), when the
                // prefill uploaded one.
                let seq_slot = ctx
                    .attn_metadata
                    .map(|m| m.seq_slot)
                    .unwrap_or(DevicePtr(0));
                if let Some(routed_pair) = routed_pair {
                    // 2026-09-25: Checked before the bgmv branch, because a routed
                    // prefill can also meet that branch's conditions.
                    debug_assert_eq!(routed_pair.k_in, h);
                    debug_assert_eq!(routed_pair.n_out, out_dim);
                    ops::lora_delta::apply_lora_delta(
                        ctx.gpu,
                        &lw.kernels,
                        routed_pair,
                        normed,
                        out,
                        n,
                        ctx.buffers.lora_xa(),
                        ctx.buffers.lora_delta(),
                        stream,
                    )?;
                } else if seq_slot.0 != 0
                    && let Some(route) = route
                    && crate::lora::prefill_bgmv_forced()
                {
                    // 2026-09-25: Opt-in (`METRALE_LORA_PREFILL_BGMV=1`), with a slot
                    // buffer and a route: the per-row bgmv.
                    ops::lora_delta::apply_lora_bgmv(
                        ctx.gpu,
                        &lw.kernels,
                        route,
                        normed,
                        out,
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
                        normed,
                        out,
                        n,
                        ctx.buffers.lora_xa(),
                        ctx.buffers.lora_delta(),
                        stream,
                    )?;
                }
            }
        }
        Ok(())
    }
}
