// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The out_proj of `decode_batched_inner`: the dispatch on row count and on the
//! out_proj weight copies the layer holds, writing `[num_tokens, h]` BF16 into `moe_output`.
//!
//! Owner: model-layers (Qwen3 SSM layer).
//! Invariants:
//! - Exactly one arm runs per call. With no usable out_proj weight the last arm returns an
//!   error instead of launching `w4a16_gemm` on a null `ssm.out_proj`.

use super::*;

impl Qwen3SsmLayer {
    /// 2026-09-26: Output projection of the `num_tokens` normed rows at `normed_out_buf`
    /// into `out_proj_buf`, before the tensor-parallel all-reduce.
    pub(super) fn batched_out_proj(
        &self,
        ctx: &ForwardContext,
        d: &BatchedDims,
        normed_out_buf: DevicePtr,
        out_proj_buf: DevicePtr,
    ) -> Result<()> {
        let BatchedDims {
            num_tokens,
            k,
            h,
            value_dim,
            bf16,
            stream,
            ..
        } = *d;
        if let Some(ref dense_out) = self.out_proj_dense {
            ops::dense_gemm(
                ctx.gpu,
                self.dense_gemm_k,
                normed_out_buf,
                dense_out,
                out_proj_buf,
                k,
                h as u32,
                value_dim as u32,
                stream,
            )?;
        } else if (2..=4).contains(&num_tokens)
            && let Some(ref fp8) = self.out_proj_fp8w
        {
            // 2026-09-25: 2..=4 rows with `out_proj_fp8w`, as for QKVZ: `w8a16_gemv_batch4`,
            // or `w8a16_gemv` per row when it is not linked.
            if self.w8a16_gemv_batch4_k.0 != 0 {
                ops::w8a16_gemv_batch4(
                    ctx.gpu,
                    self.w8a16_gemv_batch4_k,
                    normed_out_buf,
                    fp8.weight,
                    fp8.row_scale,
                    out_proj_buf,
                    num_tokens as u32,
                    h as u32,
                    value_dim as u32,
                    stream,
                )?;
            } else {
                for t in 0..num_tokens {
                    ops::w8a16_gemv(
                        ctx.gpu,
                        self.w8a16_gemv_k,
                        normed_out_buf.offset(t * value_dim * bf16),
                        fp8.weight,
                        fp8.row_scale,
                        out_proj_buf.offset(t * h * bf16),
                        h as u32,
                        value_dim as u32,
                        stream,
                    )?;
                }
            }
        } else if (4..=ops::w4a4_proj::proj_max_rows() as usize).contains(&num_tokens)
            && !self.ssm.out_proj.weight.is_null()
            && self.w4a16_batchm_kernel(num_tokens).0 != 0
        {
            // 2026-09-25: 4..=`proj_max_rows()` rows with an NVFP4 out_proj: the batched
            // NVFP4 GEMV tier for this row count, one weight pass for all rows.
            ops::w4a4_proj::nvfp4_proj_small_m(
                ctx.gpu,
                self.w4a16_batchm_kernel(num_tokens),
                normed_out_buf,
                &self.ssm.out_proj,
                out_proj_buf,
                num_tokens as u32,
                h as u32,
                value_dim as u32,
                stream,
            )?;
        } else if (5..=16).contains(&num_tokens)
            && self.w8a16_gemv_batch16_k.0 != 0
            && let Some(ref fp8) = self.out_proj_fp8w
        {
            // 2026-09-25: 5..=16 rows with `out_proj_fp8w`: `w8a16_gemv_batch16`, as for
            // QKVZ, after the NVFP4 arm.
            ops::w8a16_gemv_batch16(
                ctx.gpu,
                self.w8a16_gemv_batch16_k,
                normed_out_buf,
                fp8.weight,
                fp8.row_scale,
                out_proj_buf,
                k,
                h as u32,
                value_dim as u32,
                stream,
            )?;
        } else if num_tokens > 4
            && (self.w8a16_gemm_pipelined_k.0 != 0 || self.w8a16_gemm_k.0 != 0)
            && let Some(ref fp8) = self.out_proj_fp8w
        {
            // 2026-09-25: More than 4 rows with `out_proj_fp8w` that the arms above did not
            // take: the block-scaled W8A16 tile GEMM, as for QKVZ. Without this arm a
            // native-FP8 GDN layer would reach `w4a16_gemm` on its null `ssm.out_proj`.
            if self.w8a16_gemm_pipelined_k.0 != 0 {
                ops::w8a16_gemm_pipelined_by_m(
                    ctx.gpu,
                    self.w8a16_gemm_pipelined_k,
                    self.w8a16_gemm_pipelined_m32_k,
                    normed_out_buf,
                    fp8.weight,
                    fp8.row_scale,
                    out_proj_buf,
                    k,
                    h as u32,
                    value_dim as u32,
                    stream,
                )?;
            } else {
                ops::w8a16_gemm(
                    ctx.gpu,
                    self.w8a16_gemm_k,
                    normed_out_buf,
                    fp8.weight,
                    fp8.row_scale,
                    out_proj_buf,
                    k,
                    h as u32,
                    value_dim as u32,
                    stream,
                )?;
            }
        } else if num_tokens > VERIFY_TGEMM_MIN_TOKENS
            && let Some(ref nvfp4_t) = self.out_proj_nvfp4_t
            && verify_outproj_tgemm_enabled()
        {
            // 2026-09-25: More than `VERIFY_TGEMM_MIN_TOKENS` rows with an NVFP4 transposed
            // twin: `ms_proj_gemm`, the call the multi-seq decode out_proj makes on this
            // weight, ahead of the FP8 prefill-copy arm below. Not bitwise the FP8 arm's
            // output: a different lossy copy, with BF16 activations.
            self.ms_proj_gemm(
                ctx.gpu,
                normed_out_buf,
                nvfp4_t,
                out_proj_buf,
                num_tokens as u32,
                h as u32,
                value_dim as u32,
                stream,
            )?;
        } else if num_tokens == 3 {
            ops::w4a16_gemv_batch3(
                ctx.gpu,
                self.w4a16_gemv_batch3_k,
                normed_out_buf,
                &self.ssm.out_proj,
                out_proj_buf,
                h as u32,
                value_dim as u32,
                stream,
            )?;
        } else if num_tokens == 2 {
            ops::w4a16_gemv_batch2(
                ctx.gpu,
                self.w4a16_gemv_batch2_k,
                normed_out_buf,
                &self.ssm.out_proj,
                out_proj_buf,
                h as u32,
                value_dim as u32,
                stream,
            )?;
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
                )?;
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
                )?;
            }
        } else if let Some(ref nvfp4_t) = self.out_proj_nvfp4_t {
            if self.w4a16_gemm_t_m128_v2_k.0 != 0 {
                ops::w4a16_gemm_n128_m128_v2(
                    ctx.gpu,
                    self.w4a16_gemm_t_m128_v2_k,
                    normed_out_buf,
                    nvfp4_t,
                    out_proj_buf,
                    k,
                    h as u32,
                    value_dim as u32,
                    stream,
                )?;
            } else {
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
                )?;
            }
        } else {
            // 2026-09-25: A null `ssm.out_proj` returns an error rather than launching, as
            // for the QKVZ dense slot.
            anyhow::ensure!(
                !self.ssm.out_proj.weight.is_null(),
                "batched GDN out_proj dispatch: no usable weight for num_tokens={num_tokens} \
                 (quant slot NULL; fp8w={}, dense={}, gemm kernels pipelined/base: {:#x}/{:#x})",
                self.out_proj_fp8w.is_some(),
                self.out_proj_dense.is_some(),
                self.w8a16_gemm_pipelined_k.0,
                self.w8a16_gemm_k.0,
            );
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
            )?;
        }
        Ok(())
    }
}
