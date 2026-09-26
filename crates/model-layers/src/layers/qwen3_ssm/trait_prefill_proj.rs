// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: QKVZ projection dispatch for the GDN prefill
//! (`prefill_qkvz_proj`), shared by `prefill_block`, `prefill_phase1_inner`
//! and `prefill_phase1_proj_batched_inner`.
//!
//! Owner: model-layers, GDN/SSM layer (`qwen3_ssm`).
//! Invariants: none beyond the types.

use super::*;

impl Qwen3SsmLayer {
    /// 2026-09-25: QKVZ projection, plus the deinterleave when QKVZ is
    /// interleaved; the sequential `[Q|K|V|Z]` rows end up in
    /// `deinterleaved`. The first matching arm runs.
    ///
    /// `force_bf16` (METRALE_GDN_BF16_WEIGHTS=1) skips the row-wise arm and
    /// runs the BF16 weight on cuBLASLt ahead of the W8A8, W8A16, FP8 and
    /// NVFP4 arms. The packed Q2_0 arm and the CUTLASS / cuBLAS-FP8 dispatch
    /// arms still come before it.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_qkvz_proj(
        &self,
        normed: DevicePtr,
        deinterleaved: DevicePtr,
        k: u32,
        qkvz_size: usize,
        h: usize,
        nk: usize,
        kd: usize,
        vpg: usize,
        vd: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let proj_dst = if self.sequential_qkvz {
            deinterleaved
        } else {
            ctx.buffers.ssm_qkvz()
        };
        // 2026-09-25: Packed Q2_0 QKVZ (`qkvz_q2`). The loader installs it
        // only on layers built with `new_sequential` (qwen35_dense.rs), so
        // `proj_dst` is `deinterleaved` and returning before the deinterleave
        // is correct.
        if self.qkvz_q2.is_some() {
            let scratch = ctx.buffers.q2_dequant_scratch();
            let act_q8 = ctx.buffers.q2_act_q8();
            self.qkvz_q2_prefill_gemm(ctx.gpu, normed, proj_dst, scratch, act_q8, k, stream)?;
            return Ok(());
        }
        let force_bf16 = matches!(
            std::env::var("METRALE_GDN_BF16_WEIGHTS").ok().as_deref(),
            Some("1")
        );
        // 2026-09-25: Per-row FP8 weights (`qkvz_fp8w_rowwise`), dequantised
        // once per layer to BF16 in the arena slab (`rowwise_bf16.rs`) and run
        // on cuBLASLt. FP8 E4M3 is exact in BF16, so this arm keeps the
        // checkpoint's precision, and it runs first for that reason.
        // `cublaslt::fp8_gemm_act_weight_t_rowwise` returned NOT_SUPPORTED on
        // sm_121 (measured 2026-08-15), hence BF16. `force_bf16` still takes
        // precedence.
        if !force_bf16 && let Some(ref fp8w) = self.qkvz_fp8w_rowwise {
            // 2026-09-25: This arm returns before the CUTLASS and cuBLAS arms
            // below; warn once when one of them is enabled, since it would
            // otherwise go unused without a trace.
            if ctx.dispatch.cutlass_nvfp4_qkvz || ctx.dispatch.cutlass_gemm {
                static SHADOW_WARNED: std::sync::atomic::AtomicBool =
                    std::sync::atomic::AtomicBool::new(false);
                if !SHADOW_WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    tracing::warn!(
                        "METRALE_FP8_ROWWISE is shadowing an enabled CUTLASS/cuBLAS QKVZ \
                         prefill arm: the row-wise arm keeps the checkpoint's precision, \
                         the shadowed arms would consume the re-quantised NVFP4 copy. \
                         Unset METRALE_FP8_ROWWISE to get the CUTLASS path back."
                    );
                }
            }
            let w_bf16 = self.rowwise_qkvz_bf16(ctx, fp8w, stream)?;
            ops::cublas_bf16_proj_dense(
                normed,
                w_bf16,
                proj_dst,
                k,
                qkvz_size as u32,
                h as u32,
                stream,
            )?;
            return Ok(());
        }
        let force_w8a8 = ctx.dispatch.fp8_blockscaled_prefill;
        // 2026-09-25: The cuBLASLt path for FP8 QKVZ lives inside the W8A8 arm
        // below (`qkvz_w8a8_gemm`, `prefill_w8a8.rs`): it reads the FP8 weight
        // directly and allocates nothing. `force_w8a8` is the dispatch's
        // `fp8_blockscaled_prefill`, on unless METRALE_FP8_SINGLE_SCALE=1.
        if ctx.dispatch.cutlass_nvfp4_qkvz
            && let Some(ref nvfp4_t) = self.qkvz_nvfp4_t
        {
            ops::log_cutlass_nvfp4_route(ctx.gpu, "ssm_qkvz_nvfp4", k, qkvz_size as u32, h as u32);
            ops::cutlass_nvfp4_proj(
                ctx,
                normed,
                nvfp4_t,
                proj_dst,
                k,
                qkvz_size as u32,
                h as u32,
                stream,
            )?;
        } else if ctx.dispatch.cutlass_nvfp4_qkvz
            && let Some(ref fp8w) = self.qkvz_fp8w
        {
            ops::log_cutlass_nvfp4_route(
                ctx.gpu,
                "ssm_qkvz_fp8pack",
                k,
                qkvz_size as u32,
                h as u32,
            );
            ops::cutlass_nvfp4_proj_from_fp8(
                ctx,
                normed,
                fp8w,
                proj_dst,
                k,
                qkvz_size as u32,
                h as u32,
                stream,
            )?;
        } else if ctx.dispatch.cutlass_gemm
            && let Some(ref fp8w) = self.qkvz_fp8w
        {
            ops::cutlass_bf16_proj(
                ctx.gpu,
                ctx.derived,
                normed,
                fp8w,
                proj_dst,
                k,
                qkvz_size as u32,
                h as u32,
                stream,
            )?;
        } else if ctx.dispatch.cublas_fp8
            && let Some(ref fp8w) = self.qkvz_fp8w
        {
            ops::cublas_fp8_rowwise_proj(
                ctx.gpu,
                ctx.derived,
                normed,
                ctx.buffers.fp8_act(),
                ctx.buffers.fp8_act_scale(),
                fp8w,
                proj_dst,
                k,
                qkvz_size as u32,
                h as u32,
                stream,
            )?;
        } else if force_bf16 {
            // 2026-09-25: cuBLASLt on the BF16 weight; there is no dequant
            // step. Measured 2026-08-15 on unsloth/Qwen3.8-27B-NVFP4: through
            // the scalar `dense_gemm` this lever cut prefill from 507 to 137
            // tok/s.
            ops::cublas_bf16_proj_dense(
                normed,
                self.ssm.in_proj_qkvz.weight,
                proj_dst,
                k,
                qkvz_size as u32,
                h as u32,
                stream,
            )
            .map_err(|e| {
                anyhow::anyhow!(
                    "ssm prefill: QKVZ BF16 cuBLASLt GEMM failed (M={k}, N={qkvz_size}): {e}"
                )
            })?;
        } else if force_w8a8
            && let Some(ref fp8w) = self.qkvz_fp8w
            && self.per_token_group_quant_fp8_k.available()
            && self.fp8_gemm_t_blockscaled_k.0 != 0
        {
            tracing::debug!(
                "ssm prefill: QKVZ via block-scaled FP8 (W8A8+FP32-epilogue, M={k} K={h} N={qkvz_size})"
            );
            let k_dim = h;
            // 2026-09-25: Arena scratch, `cublas_fp8_m_pad(k)` (16-row padded)
            // rows, because the cuBLASLt arm reads that many. The quant and the
            // GEMM are ordered on one stream.
            let m_pad = ops::cublas_fp8_m_pad(k) as usize;
            let a_fp8_buf = ctx.buffers.fp8_act();
            let a_scale_buf = ctx.buffers.fp8_act_scale();
            debug_assert!(m_pad * k_dim <= ctx.buffers.fp8_act_bytes());
            debug_assert!(m_pad * k_dim.div_ceil(128) * 4 <= ctx.buffers.fp8_act_scale_bytes());
            // 2026-09-25: Per-token 1x128 FP8 quant of the activation, then a
            // block-scaled FP8 GEMM that folds both scale sets in an FP32
            // epilogue: cuBLASLt or the in-tree kernel, chosen by
            // `qkvz_w8a8_gemm`.
            ops::per_token_group_quant_fp8(
                ctx.gpu,
                self.per_token_group_quant_fp8_k,
                normed,
                a_fp8_buf,
                a_scale_buf,
                k,
                k_dim as u32,
                stream,
            )?;
            // 2026-09-25: `proj_dst` is one of two arena buffers, picked by
            // `sequential_qkvz`; the cuBLASLt arm writes padded rows into it,
            // so the bound is the capacity of the one it got.
            let dst_capacity = if self.sequential_qkvz {
                ctx.buffers.ssm_deinterleaved_bytes()
            } else {
                ctx.buffers.ssm_qkvz_bytes()
            };
            self.qkvz_w8a8_gemm(
                ctx,
                a_fp8_buf,
                a_scale_buf,
                fp8w,
                proj_dst,
                dst_capacity,
                k,
                qkvz_size as u32,
                h as u32,
                stream,
            )?;
        } else if let Some(ref fp8w) = self.qkvz_fp8w
            && self.w8a16_gemm_pipelined_k.0 != 0
        {
            // 2026-09-25: Block-scaled W8A16 (per-128-block FP32 weight scales)
            // through the pipelined kernel; without it, the base `w8a16_gemm`
            // arm below runs.
            ops::w8a16_gemm_pipelined(
                ctx.gpu,
                self.w8a16_gemm_pipelined_k,
                normed,
                fp8w.weight,
                fp8w.row_scale,
                proj_dst,
                k,
                qkvz_size as u32,
                h as u32,
                stream,
            )
            .map_err(|e| {
                anyhow::anyhow!(
                    "ssm prefill: QKVZ w8a16_gemm_pipelined failed (M={k}, N={qkvz_size}): {e}"
                )
            })?;
        } else if let Some(ref fp8w) = self.qkvz_fp8w
            && self.w8a16_gemm_k.0 != 0
        {
            // 2026-09-25: The base block-scaled W8A16 GEMM, for targets without
            // `w8a16_gemm_pipelined`.
            ops::w8a16_gemm(
                ctx.gpu,
                self.w8a16_gemm_k,
                normed,
                fp8w.weight,
                fp8w.row_scale,
                proj_dst,
                k,
                qkvz_size as u32,
                h as u32,
                stream,
            )
            .map_err(|e| {
                anyhow::anyhow!(
                    "ssm prefill: QKVZ w8a16_gemm (block-scaled) failed (M={k}, N={qkvz_size}): {e}"
                )
            })?;
        } else if let Some(fp8) = self.qkvz_fp8 {
            ops::fp8_gemm_n128(
                ctx.gpu,
                self.fp8_gemm_k,
                normed,
                fp8,
                proj_dst,
                k,
                qkvz_size as u32,
                h as u32,
                stream,
            )
            .map_err(|e| {
                anyhow::anyhow!("ssm prefill: QKVZ FP8 GEMM failed (M={k}, N={qkvz_size}): {e}")
            })?;
        } else if let Some(ref nvfp4_t) = self.qkvz_nvfp4_t {
            if k > 128 {
                ops::w4a16_gemm_n128_m128(
                    ctx.gpu,
                    self.w4a16_gemm_t_m128_k,
                    normed,
                    nvfp4_t,
                    proj_dst,
                    k,
                    qkvz_size as u32,
                    h as u32,
                    stream,
                )
                .map_err(|e| {
                    anyhow::anyhow!(
                        "ssm prefill: QKVZ m128 GEMM failed (M={k}, N={qkvz_size}): {e}"
                    )
                })?;
            } else {
                ops::w4a16_gemm_n128(
                    ctx.gpu,
                    self.w4a16_gemm_t_k,
                    normed,
                    nvfp4_t,
                    proj_dst,
                    k,
                    qkvz_size as u32,
                    h as u32,
                    stream,
                )
                .map_err(|e| {
                    anyhow::anyhow!("ssm prefill: QKVZ GEMM failed (M={k}, N={qkvz_size}): {e}")
                })?;
            }
        } else if let Some(ref nvfp4) = self.qkvz_nvfp4 {
            ops::w4a16_gemm(
                ctx.gpu,
                self.w4a16_gemm_k,
                normed,
                nvfp4,
                proj_dst,
                k,
                qkvz_size as u32,
                h as u32,
                stream,
            )
            .map_err(|e| {
                anyhow::anyhow!("ssm prefill: QKVZ GEMM failed (M={k}, N={qkvz_size}): {e}")
            })?;
        } else {
            // 2026-09-25: BF16 weights (the qwen4_exp loader keeps its GDN
            // projections BF16 unless METRALE_QWEN4EXP_BF16_GDN=0): cuBLASLt,
            // and the scalar `dense_gemm` if that call fails. Measured
            // 2026-08-26 on Qwen3.8-Flash-Next: through `dense_gemm` this arm
            // took 147 ms per call.
            if ops::cublas_bf16_proj_dense(
                normed,
                self.ssm.in_proj_qkvz.weight,
                proj_dst,
                k,
                qkvz_size as u32,
                h as u32,
                stream,
            )
            .is_err()
            {
                ops::dense_gemm(
                    ctx.gpu,
                    self.dense_gemm_k,
                    normed,
                    &self.ssm.in_proj_qkvz,
                    proj_dst,
                    k,
                    qkvz_size as u32,
                    h as u32,
                    stream,
                )?;
            }
        }
        if !self.sequential_qkvz {
            ops::deinterleave_qkvz(
                ctx.gpu,
                self.deinterleave_k,
                proj_dst,
                deinterleaved,
                k,
                nk as u32,
                kd as u32,
                vpg as u32,
                vd as u32,
                stream,
            )?;
        }
        Ok(())
    }
}
