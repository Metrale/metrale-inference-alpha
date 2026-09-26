// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The `in_proj_qkvz` and `out_proj` projections of the
//! batched-decode GDN mixer, and the GEMV/GEMM tier they share.
//!
//! Owner: model-layers, GDN/SSM layer (`qwen3_ssm`).
//! Invariants:
//! - The NVFP4 batched-GEMV arms return an error instead of launching for
//!   more than 16 rows.
//!
//! `ssm_batched.rs` keeps the phase order; this file picks the GEMM for each
//! projection at `n` rows: cuBLASLt W8A8 (`decode_w8a8_proj.rs`), the
//! block-scaled W8A16 GEMV tiers, the pipelined or base W8A16 GEMM, cuBLASLt
//! BF16, or the NVFP4 tile GEMM and batched GEMV.

use super::super::decode_w8a8_proj::SsmDecodeProj;
use super::super::*;
use super::ssm_batched::ssm_tc_proj_min_n;

/// 2026-09-25: The GEMV/GEMM handles this step's row count selects, resolved
/// once (`batched_proj_tier`) and passed to both projections, which run at the
/// same row count.
pub(super) struct BatchedProjTier {
    /// 2026-09-25: `w8a16_gemm_pipelined` is loaded.
    pub(super) w8a16_pipe: bool,
    /// 2026-09-25: The contiguous block-scaled W8A16 GEMV wrapper for this row
    /// count; `gemv_batch_k` is its handle.
    pub(super) gemv_batch: ops::ContiguousBatchGemv,
    pub(super) gemv_batch_k: KernelHandle,
    /// 2026-09-25: That GEMV tier is loaded, `n <= 16`, and the
    /// `ssm_gemv_batch4` lever is on (METRALE_SSM_GEMV_BATCH4=0 turns it off).
    pub(super) use_batch4: bool,
    /// 2026-09-25: The NVFP4 batched-GEMV handle for this row count.
    pub(super) fp4_gemv_batch_k: KernelHandle,
}

impl Qwen3SsmLayer {
    pub(super) fn batched_proj_tier(&self, n: usize) -> BatchedProjTier {
        let w8a16_pipe = self.w8a16_gemm_pipelined_k.0 != 0;
        // 2026-09-25: `w8a16_gemv_batch4` serves up to 4 rows,
        // `w8a16_gemv_batch16` up to 16; the wrapper and handle are chosen
        // together.
        let (gemv_batch, gemv_batch_k): (ops::ContiguousBatchGemv, KernelHandle) = if n <= 4 {
            (ops::w8a16_gemv_batch4, self.w8a16_gemv_batch4_k)
        } else {
            (ops::w8a16_gemv_batch16, self.w8a16_gemv_batch16_k)
        };
        let use_batch4 = gemv_batch_k.0 != 0
            && n <= 16
            && crate::layers::ops::ModelLevers::get().ssm_gemv_batch4;
        // 2026-09-25: NVFP4: the narrowest loaded `w4a16_batchm` tier covering
        // `n`, else `w4a16_gemv_batch16`.
        let narrow = self.w4a16_batchm.kernel(n as u32);
        let fp4_gemv_batch_k = if narrow.0 != 0 {
            narrow
        } else {
            self.w4a16_gemv_batch16_k
        };
        BatchedProjTier {
            w8a16_pipe,
            gemv_batch,
            gemv_batch_k,
            use_batch4,
            fp4_gemv_batch_k,
        }
    }

    /// 2026-09-25: Batched QKVZ projection: one `[n, h] -> [n, qkvz]`
    /// projection over all `n` rows, written into `deinterleaved`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn ms_batched_qkvz(
        &self,
        ctx: &ForwardContext,
        tier: &BatchedProjTier,
        n: usize,
        normed_base: DevicePtr,
        deinterleaved: DevicePtr,
        qkvz_size: usize,
        h: usize,
        stream: u64,
    ) -> Result<()> {
        let BatchedProjTier {
            w8a16_pipe,
            gemv_batch,
            gemv_batch_k,
            use_batch4,
            fp4_gemv_batch_k,
        } = *tier;
        if let Some(ref fp8) = self.qkvz_fp8w {
            // 2026-09-25: cuBLASLt W8A8 when `try_ssm_decode_w8a8` accepts
            // (5..=16 rows, `DECODE_W8A8_ROWS`); otherwise the arms below.
            if self.try_ssm_decode_w8a8(
                ctx,
                SsmDecodeProj::Qkvz,
                normed_base,
                fp8,
                deinterleaved,
                ctx.buffers.ssm_deinterleaved_bytes(),
                n,
                qkvz_size as u32,
                h as u32,
                stream,
            )? {
                // 2026-09-25: `try_ssm_decode_w8a8` already ran the projection.
            } else if use_batch4 {
                gemv_batch(
                    ctx.gpu,
                    gemv_batch_k,
                    normed_base,
                    fp8.weight,
                    fp8.row_scale,
                    deinterleaved,
                    n as u32,
                    qkvz_size as u32,
                    h as u32,
                    stream,
                )?;
            } else if w8a16_pipe {
                ops::w8a16_gemm_pipelined(
                    ctx.gpu,
                    self.w8a16_gemm_pipelined_k,
                    normed_base,
                    fp8.weight,
                    fp8.row_scale,
                    deinterleaved,
                    n as u32,
                    qkvz_size as u32,
                    h as u32,
                    stream,
                )?;
            } else {
                ops::w8a16_gemm(
                    ctx.gpu,
                    self.w8a16_gemm_k,
                    normed_base,
                    fp8.weight,
                    fp8.row_scale,
                    deinterleaved,
                    n as u32,
                    qkvz_size as u32,
                    h as u32,
                    stream,
                )?;
            }
        } else if let Some(ref nvfp4) = self.qkvz_nvfp4 {
            match (ssm_tc_proj_min_n(), self.qkvz_nvfp4_t.as_ref()) {
                (Some(min_n), Some(nvfp4_t)) if n >= min_n => {
                    // 2026-09-25: Tile GEMM on the transposed twin. `ms_proj_gemm`
                    // takes the 128-row M-tile when `ssm_m128_min_m` admits `n`
                    // and ceil(N/128) CTAs cover the SMs.
                    self.ms_proj_gemm(
                        ctx.gpu,
                        normed_base,
                        nvfp4_t,
                        deinterleaved,
                        n as u32,
                        qkvz_size as u32,
                        h as u32,
                        stream,
                    )?;
                }
                // 2026-09-25: `nvfp4_proj_small_m` over all `n` rows (the
                // W4A16 batched GEMV, or W4A4 under `--w4a4-downcast`), written
                // straight into `deinterleaved` (QKVZ is sequential here).
                _ => {
                    // 2026-09-25: The eligibility check in `ssm_batched.rs`
                    // admits this arm only at n <= 16; the ensure fails fast if
                    // that changes.
                    anyhow::ensure!(
                        n <= 16,
                        "SSM batchm QKVZ GEMV caps at M=16 (n={n}); tile-GEMM twins required"
                    );
                    ops::w4a4_proj::nvfp4_proj_small_m(
                        ctx.gpu,
                        fp4_gemv_batch_k,
                        normed_base,
                        nvfp4,
                        deinterleaved,
                        n as u32,
                        qkvz_size as u32,
                        h as u32,
                        stream,
                    )?
                }
            }
        } else {
            ops::cublas_bf16_proj_dense(
                normed_base,
                self.ssm.in_proj_qkvz.weight,
                deinterleaved,
                n as u32,
                qkvz_size as u32,
                h as u32,
                stream,
            )?;
        }
        Ok(())
    }

    /// 2026-09-25: Batched out_proj: one `[n, value_dim] -> [n, h]`
    /// projection over all `n` rows. Writes nothing when no out_proj weight is present;
    /// the eligibility check in `ssm_batched.rs` requires one.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn ms_batched_out_proj(
        &self,
        ctx: &ForwardContext,
        tier: &BatchedProjTier,
        n: usize,
        normed_out_base: DevicePtr,
        ssm_out_base: DevicePtr,
        h: usize,
        value_dim: usize,
        stream: u64,
    ) -> Result<()> {
        let BatchedProjTier {
            w8a16_pipe,
            gemv_batch,
            gemv_batch_k,
            use_batch4,
            fp4_gemv_batch_k,
        } = *tier;
        if let Some(ref fp8) = self.out_proj_fp8w {
            // 2026-09-25: cuBLASLt W8A8 when `try_ssm_decode_w8a8` accepts;
            // otherwise the arms below.
            if self.try_ssm_decode_w8a8(
                ctx,
                SsmDecodeProj::OutProj,
                normed_out_base,
                fp8,
                ssm_out_base,
                ctx.buffers.moe_output_bytes(),
                n,
                h as u32,
                value_dim as u32,
                stream,
            )? {
                // 2026-09-25: `try_ssm_decode_w8a8` already ran the projection.
            } else if use_batch4 {
                gemv_batch(
                    ctx.gpu,
                    gemv_batch_k,
                    normed_out_base,
                    fp8.weight,
                    fp8.row_scale,
                    ssm_out_base,
                    n as u32,
                    h as u32,
                    value_dim as u32,
                    stream,
                )?;
            } else if w8a16_pipe {
                ops::w8a16_gemm_pipelined(
                    ctx.gpu,
                    self.w8a16_gemm_pipelined_k,
                    normed_out_base,
                    fp8.weight,
                    fp8.row_scale,
                    ssm_out_base,
                    n as u32,
                    h as u32,
                    value_dim as u32,
                    stream,
                )?;
            } else {
                ops::w8a16_gemm(
                    ctx.gpu,
                    self.w8a16_gemm_k,
                    normed_out_base,
                    fp8.weight,
                    fp8.row_scale,
                    ssm_out_base,
                    n as u32,
                    h as u32,
                    value_dim as u32,
                    stream,
                )?;
            }
        } else if let Some(ref out_proj_dense) = self.out_proj_dense {
            ops::cublas_bf16_proj_dense(
                normed_out_base,
                out_proj_dense.weight,
                ssm_out_base,
                n as u32,
                h as u32,
                value_dim as u32,
                stream,
            )?;
        } else if self.qkvz_nvfp4.is_some() {
            match (ssm_tc_proj_min_n(), self.out_proj_nvfp4_t.as_ref()) {
                (Some(min_n), Some(nvfp4_t)) if n >= min_n => {
                    // 2026-09-25: Tile GEMM on the transposed twin, as in the
                    // QKVZ arm.
                    self.ms_proj_gemm(
                        ctx.gpu,
                        normed_out_base,
                        nvfp4_t,
                        ssm_out_base,
                        n as u32,
                        h as u32,
                        value_dim as u32,
                        stream,
                    )?;
                }
                // 2026-09-25: `nvfp4_proj_small_m` over all `n` rows on
                // `ssm.out_proj`, the weight the per-sequence `ssm_forward`
                // uses in an NVFP4 build.
                _ => {
                    // 2026-09-25: Admitted only at n <= 16, as in the QKVZ arm.
                    anyhow::ensure!(
                        n <= 16,
                        "SSM batchm out_proj GEMV caps at M=16 (n={n}); tile-GEMM twins required"
                    );
                    ops::w4a4_proj::nvfp4_proj_small_m(
                        ctx.gpu,
                        fp4_gemv_batch_k,
                        normed_out_base,
                        &self.ssm.out_proj,
                        ssm_out_base,
                        n as u32,
                        h as u32,
                        value_dim as u32,
                        stream,
                    )?
                }
            }
        }
        Ok(())
    }
}
