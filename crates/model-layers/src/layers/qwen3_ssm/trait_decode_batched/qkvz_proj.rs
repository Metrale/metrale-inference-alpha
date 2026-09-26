// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The QKVZ projection of `decode_batched_inner`: the dispatch on row count and
//! on the QKVZ weight copies the layer holds, writing `num_tokens` rows of `qkvz_size` BF16
//! into `proj_dst`.
//!
//! Owner: model-layers (Qwen3 SSM layer).
//! Invariants:
//! - Exactly one arm runs per call. With no usable QKVZ weight the last arm returns an error
//!   instead of launching `dense_gemm` on a null dense slot.

use super::*;

impl Qwen3SsmLayer {
    /// 2026-09-26: QKVZ projection of the `num_tokens` normed rows at `normed` into
    /// `proj_dst`, called once per [`Qwen3SsmLayer::decode_batched_inner`].
    pub(super) fn batched_qkvz_proj(
        &self,
        ctx: &ForwardContext,
        d: &BatchedDims,
        normed: DevicePtr,
        proj_dst: DevicePtr,
    ) -> Result<()> {
        let BatchedDims {
            num_tokens,
            k,
            h,
            bf16,
            qkvz_size,
            stream,
            ..
        } = *d;
        if let Some(ref q2) = self.qkvz_q2 {
            // 2026-09-25: Packed Q2_0 QKVZ: one `q2_0_gemv_vec` per row, the kernel the
            // single-token decode runs (`ssm_forward`).
            for t in 0..num_tokens {
                ops::q2_0_gemv_vec(
                    ctx.gpu,
                    self.q2_0_gemv_k,
                    normed.offset(t * h * bf16),
                    q2,
                    proj_dst.offset(t * qkvz_size * bf16),
                    stream,
                )?;
            }
        // 2026-09-25: 2..=4 rows with a block-scaled FP8 QKVZ (`qkvz_fp8w`, the only
        // QKVZ weight a native-FP8 GDN checkpoint holds): one `w8a16_gemv_batch4` weight
        // pass (M <= 4), or `w8a16_gemv` per row when that kernel is not linked.
        } else if (2..=4).contains(&num_tokens)
            && let Some(ref fp8) = self.qkvz_fp8w
        {
            if self.w8a16_gemv_batch4_k.0 != 0 {
                ops::w8a16_gemv_batch4(
                    ctx.gpu,
                    self.w8a16_gemv_batch4_k,
                    normed,
                    fp8.weight,
                    fp8.row_scale,
                    proj_dst,
                    num_tokens as u32,
                    qkvz_size as u32,
                    h as u32,
                    stream,
                )?;
            } else {
                for t in 0..num_tokens {
                    ops::w8a16_gemv(
                        ctx.gpu,
                        self.w8a16_gemv_k,
                        normed.offset(t * h * bf16),
                        fp8.weight,
                        fp8.row_scale,
                        proj_dst.offset(t * qkvz_size * bf16),
                        qkvz_size as u32,
                        h as u32,
                        stream,
                    )?;
                }
            }
        } else if (5..=ops::w4a4_proj::proj_max_rows() as usize).contains(&num_tokens)
            && self.w4a16_batchm.kernel(num_tokens as u32).0 != 0
            && let Some(ref nvfp4) = self.qkvz_nvfp4
        {
            // 2026-09-25: 5..=`proj_max_rows()` rows with an NVFP4 QKVZ:
            // `nvfp4_proj_small_m` with the `w4a16_batchm` tier for this row count.
            ops::w4a4_proj::nvfp4_proj_small_m(
                ctx.gpu,
                self.w4a16_batchm.kernel(num_tokens as u32),
                normed,
                nvfp4,
                proj_dst,
                num_tokens as u32,
                qkvz_size as u32,
                h as u32,
                stream,
            )?;
        } else if (5..=16).contains(&num_tokens)
            && self.w8a16_gemv_batch16_k.0 != 0
            && let Some(ref fp8) = self.qkvz_fp8w
        {
            // 2026-09-25: 5..=16 rows with `qkvz_fp8w`: `w8a16_gemv_batch16`, the MAX_M=16
            // instantiation of the `w8a16_gemv_batch4` template. One weight pass serves
            // every row, and each row is bitwise the scalar `w8a16_gemv` the single-token
            // decode runs (checked by the `w8a16_batch_bitparity_microtest` example). It
            // sits after the NVFP4 GEMV arm, so at the rows both arms cover a layer holding
            // both formats takes NVFP4.
            ops::w8a16_gemv_batch16(
                ctx.gpu,
                self.w8a16_gemv_batch16_k,
                normed,
                fp8.weight,
                fp8.row_scale,
                proj_dst,
                k,
                qkvz_size as u32,
                h as u32,
                stream,
            )?;
        } else if num_tokens > 4
            && (self.w8a16_gemm_pipelined_k.0 != 0 || self.w8a16_gemm_k.0 != 0)
            && let Some(ref fp8) = self.qkvz_fp8w
        {
            // 2026-09-25: More than 4 rows with `qkvz_fp8w` that the arms above did not
            // take: the block-scaled W8A16 tile GEMM the SSM prefill uses
            // (`trait_prefill_proj.rs`), the pipelined kernel when linked. Without this arm
            // a native-FP8 GDN layer would fall through to `dense_gemm` on its null dense
            // slot.
            if self.w8a16_gemm_pipelined_k.0 != 0 {
                // 2026-09-25: `w8a16_gemm_pipelined_by_m` runs the 32-row M-tile twin at
                // 1..=32 rows when it is linked and K is whole 128-wide scale blocks, the
                // 128-row tile otherwise.
                ops::w8a16_gemm_pipelined_by_m(
                    ctx.gpu,
                    self.w8a16_gemm_pipelined_k,
                    self.w8a16_gemm_pipelined_m32_k,
                    normed,
                    fp8.weight,
                    fp8.row_scale,
                    proj_dst,
                    k,
                    qkvz_size as u32,
                    h as u32,
                    stream,
                )?;
            } else {
                ops::w8a16_gemm(
                    ctx.gpu,
                    self.w8a16_gemm_k,
                    normed,
                    fp8.weight,
                    fp8.row_scale,
                    proj_dst,
                    k,
                    qkvz_size as u32,
                    h as u32,
                    stream,
                )?;
            }
        } else if num_tokens == 4 {
            if let Some(ref nvfp4) = self.qkvz_nvfp4 {
                ops::w4a4_proj::nvfp4_proj_small_m(
                    ctx.gpu,
                    self.w4a16_batchm.kernel(num_tokens as u32),
                    normed,
                    nvfp4,
                    proj_dst,
                    num_tokens as u32,
                    qkvz_size as u32,
                    h as u32,
                    stream,
                )?;
            } else if let Some(ref fp8w) = self.qkvz_fp8w {
                ops::w8a16_gemv_batch4(
                    ctx.gpu,
                    self.w8a16_gemv_batch4_k,
                    normed,
                    fp8w.weight,
                    fp8w.row_scale,
                    proj_dst,
                    num_tokens as u32,
                    qkvz_size as u32,
                    h as u32,
                    stream,
                )?;
            } else {
                for t in 0..4u32 {
                    ops::dense_gemv(
                        ctx.gpu,
                        self.dense_gemv_k,
                        normed.offset(t as usize * h * bf16),
                        &self.ssm.in_proj_qkvz,
                        proj_dst.offset(t as usize * qkvz_size * bf16),
                        qkvz_size as u32,
                        h as u32,
                        stream,
                    )?;
                }
            }
        } else if num_tokens == 3 {
            if let Some(ref nvfp4) = self.qkvz_nvfp4 {
                ops::w4a16_gemv_batch3(
                    ctx.gpu,
                    self.w4a16_gemv_batch3_k,
                    normed,
                    nvfp4,
                    proj_dst,
                    qkvz_size as u32,
                    h as u32,
                    stream,
                )?;
            } else {
                for t in 0..3u32 {
                    ops::dense_gemv(
                        ctx.gpu,
                        self.dense_gemv_k,
                        normed.offset(t as usize * h * bf16),
                        &self.ssm.in_proj_qkvz,
                        proj_dst.offset(t as usize * qkvz_size * bf16),
                        qkvz_size as u32,
                        h as u32,
                        stream,
                    )?;
                }
            }
        } else if num_tokens == 2 {
            if let Some(ref nvfp4) = self.qkvz_nvfp4 {
                ops::w4a16_gemv_batch2(
                    ctx.gpu,
                    self.w4a16_gemv_batch2_k,
                    normed,
                    nvfp4,
                    proj_dst,
                    qkvz_size as u32,
                    h as u32,
                    stream,
                )?;
            } else {
                ops::dense_gemv_batch2(
                    ctx.gpu,
                    self.dense_gemv_batch2_k,
                    normed,
                    &self.ssm.in_proj_qkvz,
                    proj_dst,
                    qkvz_size as u32,
                    h as u32,
                    qkvz_size as u32,
                    stream,
                )?;
            }
        } else if qkvz_verify_nvfp4_wins(
            num_tokens,
            self.qkvz_fp8.is_some(),
            self.qkvz_nvfp4_t.is_some(),
            self.deep_k_gemm(h as u32).0 != 0,
            qkvz_nvfp4_decode_off(),
        ) && let Some(ref nvfp4_t) = self.qkvz_nvfp4_t
        {
            // 2026-09-25: More than `VERIFY_TGEMM_MIN_TOKENS` rows with both copies: the
            // NVFP4 transposed twin (0.5625 bytes per weight) through `ms_proj_gemm`, the
            // call the multi-seq decode QKVZ makes on this weight
            // (`trait_decode_multi_seq/ssm_batched_proj.rs`), instead of the 1-byte FP8
            // prefill copy. The output is not bitwise the FP8 arm's: it reads a different
            // lossy copy with BF16 activations.
            self.ms_proj_gemm(
                ctx.gpu,
                normed,
                nvfp4_t,
                proj_dst,
                num_tokens as u32,
                qkvz_size as u32,
                h as u32,
                stream,
            )?;
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
            )?;
        } else if let Some(ref nvfp4_t) = self.qkvz_nvfp4_t {
            // 2026-09-25: `w4a16_gemm_n128_m128_v2` when linked, else the m128 tile above
            // 128 rows, else n128.
            if self.w4a16_gemm_t_m128_v2_k.0 != 0 {
                ops::w4a16_gemm_n128_m128_v2(
                    ctx.gpu,
                    self.w4a16_gemm_t_m128_v2_k,
                    normed,
                    nvfp4_t,
                    proj_dst,
                    k,
                    qkvz_size as u32,
                    h as u32,
                    stream,
                )?;
            } else if k > 128 {
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
                )?;
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
                )?;
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
            )?;
        } else {
            // 2026-09-25: A `dense_gemm` on a null weight is an illegal device access that
            // leaves the CUDA context unusable, so a null dense slot returns an error.
            anyhow::ensure!(
                !self.ssm.in_proj_qkvz.weight.is_null(),
                "batched GDN QKVZ dispatch: no usable weight for num_tokens={num_tokens} \
                 (dense slot NULL; fp8w={}, nvfp4={}, gemm kernels pipelined/base: {:#x}/{:#x})",
                self.qkvz_fp8w.is_some(),
                self.qkvz_nvfp4.is_some(),
                self.w8a16_gemm_pipelined_k.0,
                self.w8a16_gemm_k.0,
            );
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
        Ok(())
    }
}
