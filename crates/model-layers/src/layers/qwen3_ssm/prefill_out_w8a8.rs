// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The W8A8 block-scaled cuBLASLt arm of the GDN `out_proj`
//! prefill projection, beside `prefill_w8a8.rs`'s `in_proj_qkvz` arm.
//!
//! `prefill_out_proj_dispatch` (`trait_prefill_helper.rs`) tries it after the
//! row-wise, CUTLASS and BF16-dense arms and before the `METRALE_FP8_W8A8=1`
//! in-tree W8A8 arm and the W8A16 kernels. W8A8 quantizes the activation to
//! E4M3 per 128-wide K group; the model-arch example
//! `native_fp8_prefill_proj_w8a8_microtest` gates this projection at cosine
//! >= 0.999 and rel_rms <= 3e-2 against the W8A16 reference.
//!
//! Owner: model-layers (qwen3_ssm).
//! Invariants:
//! - The activation, its scales and the k-major scale copy live in arena
//!   buffers (`fp8_act`, `fp8_act_scale`, `fp8_act_scale_kmajor`); nothing
//!   here allocates a buffer for them.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, KernelHandle};

use super::Qwen3SsmLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::weight_map::{Fp8Weight, WeightQuantFormat};

/// 2026-09-25: `METRALE_SSM_OUT_W8A16_ONLY`, read by presence (any value,
/// empty and `0` included) once per process. When set, the SSM `out_proj`
/// prefill never takes the cuBLASLt W8A8 arm.
pub(super) fn ssm_out_w8a16_only() -> bool {
    static ONLY: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ONLY.get_or_init(|| std::env::var_os("METRALE_SSM_OUT_W8A16_ONLY").is_some())
}

/// 2026-09-25: Whether the SSM `out_proj` prefill takes the cuBLASLt arm. A
/// pure function, so `prefill_out_w8a8_tests.rs` checks each clause without a
/// GPU.
///
/// All of these must hold:
/// * `!w8a16_only` (`METRALE_SSM_OUT_W8A16_ONLY` unset).
/// * `cublas_ssm`: `METRALE_CUBLAS_GEMM` names the `ssm` family.
/// * `fp8_blockscaled_prefill` (`METRALE_FP8_SINGLE_SCALE` unset).
/// * `m > 4`, the floor `dense_ffn_w8a8_prefill`'s selector also uses.
/// * `scale_format == Fp8BlockScaled`: cuBLASLt reads the weight scales as a
///   `[N/128, K/128]` BLK128x128 grid.
/// * `n` and `k` are multiples of 128, and `blk128x128_stride_ok(k)`
///   (`K/128` a multiple of 4, which cuBLAS requires of the scale stride).
/// * Room for the padded rows (`m_pad = ceil16(m)`), which
///   `cublas_fp8_proj_prequant` writes: `m_pad * n` BF16 of output,
///   `m_pad * k` FP8 of activation and `m_pad * K/128` FP32 of scales.
/// * The quantizer resolved, and under the default k-major scale layout the
///   `fp8_act_scale_to_kmajor` kernel and a scratch of `m_pad * K/128` FP32.
#[allow(clippy::too_many_arguments)]
pub(super) fn out_proj_cublas_selected(
    cublas_ssm: bool,
    fp8_blockscaled_prefill: bool,
    w8a16_only: bool,
    scale_format: WeightQuantFormat,
    m: u32,
    n: u32,
    k: u32,
    out_capacity_bytes: usize,
    act_capacity_bytes: usize,
    act_scale_capacity_bytes: usize,
    quant_k: ops::Fp8ActQuant,
    scale_kmajor_k: KernelHandle,
    scale_kmajor_buf: DevicePtr,
    scale_kmajor_capacity_bytes: usize,
) -> bool {
    let m_pad = ops::cublas_fp8_m_pad(m) as usize;
    let kg = k as usize / 128;
    let kmajor_ready = scale_kmajor_k.0 != 0
        && scale_kmajor_buf.0 != 0
        && scale_kmajor_capacity_bytes >= m_pad * kg * 4;
    !w8a16_only
        && cublas_ssm
        && fp8_blockscaled_prefill
        && m > 4
        && scale_format == WeightQuantFormat::Fp8BlockScaled
        && n.is_multiple_of(128)
        && k.is_multiple_of(128)
        && metrale_gpu_runtime::cublaslt::scale_layout::blk128x128_stride_ok(k as usize)
        && m_pad * (n as usize) * 2 <= out_capacity_bytes
        && m_pad * (k as usize) <= act_capacity_bytes
        && m_pad * kg * 4 <= act_scale_capacity_bytes
        && quant_k.available()
        && (kmajor_ready || !ops::cublas_scale_layout_kmajor())
}

impl Qwen3SsmLayer {
    /// 2026-09-25: [`out_proj_cublas_selected`] with this layer's handles and
    /// the arena capacities; the output capacity checked is `moe_output`, the
    /// buffer the prefill passes as `out_proj`. `n` is the hidden size and `k`
    /// the GDN value dim.
    pub(super) fn prefill_out_proj_w8a8_selected(
        &self,
        ctx: &ForwardContext,
        m: u32,
        n: u32,
        k: u32,
        fp8w: &Fp8Weight,
    ) -> bool {
        out_proj_cublas_selected(
            ctx.dispatch.cublas.ssm,
            ctx.dispatch.fp8_blockscaled_prefill,
            ssm_out_w8a16_only(),
            fp8w.scale_format,
            m,
            n,
            k,
            ctx.buffers.moe_output_bytes(),
            ctx.buffers.fp8_act_bytes(),
            ctx.buffers.fp8_act_scale_bytes(),
            self.per_token_group_quant_fp8_k,
            self.fp8_act_scale_kmajor_k,
            ctx.buffers.fp8_act_scale_kmajor(),
            ctx.buffers.fp8_act_scale_kmajor_bytes(),
        )
    }

    /// 2026-09-25: `out[m, n] = quant(normed_out)[m, k] @ weight[n, k]ᵀ` through
    /// cuBLASLt, with the per-token 1x128 activation scales and the 128x128
    /// weight scales applied in an FP32 epilogue.
    ///
    /// Quantizes `normed_out` into the arena's `fp8_act` / `fp8_act_scale` and
    /// hands both to `ops::cublas_fp8_proj_prequant`, which zeroes the pad
    /// rows, writes the k-major scale copy and runs the matmul, on `stream`.
    ///
    /// The caller must have checked
    /// [`Qwen3SsmLayer::prefill_out_proj_w8a8_selected`]; this method does not
    /// re-check and has no fallback.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_out_proj_w8a8_cublas(
        &self,
        ctx: &ForwardContext,
        normed_out: DevicePtr,
        fp8w: &Fp8Weight,
        out: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        let a_fp8 = ctx.buffers.fp8_act();
        let a_scale = ctx.buffers.fp8_act_scale();
        // 2026-09-25: `cublas_fp8_proj_prequant` zeroes and then reads rows
        // `m..ceil16(m)` of the activation; the selector checked these sizes.
        let m_pad = ops::cublas_fp8_m_pad(m) as usize;
        debug_assert!(m_pad * k as usize <= ctx.buffers.fp8_act_bytes());
        debug_assert!(m_pad * (k as usize / 128) * 4 <= ctx.buffers.fp8_act_scale_bytes());
        ops::per_token_group_quant_fp8(
            ctx.gpu,
            self.per_token_group_quant_fp8_k,
            normed_out,
            a_fp8,
            a_scale,
            m,
            k,
            stream,
        )?;
        ops::cublas_fp8_proj_prequant(
            ctx.gpu,
            self.fp8_act_scale_kmajor_k,
            a_fp8,
            a_scale,
            ctx.buffers.fp8_act_scale_kmajor(),
            fp8w,
            out,
            m,
            n,
            k,
            stream,
        )
    }

    /// 2026-09-25: Log, once per model, which kernel the SSM `out_proj` prefill
    /// uses. Called by the cuBLASLt arm (`cublas = true`) and by the
    /// `w8a16_gemm_pipelined` arm (`false`) of `prefill_out_proj_dispatch`.
    pub(super) fn log_out_proj_prefill_route(&self, ctx: &ForwardContext, cublas: bool) {
        if ctx.stats.once("log:ssm_out_proj_prefill") {
            if cublas {
                tracing::info!(
                    "[metrale] SSM out_proj prefill: W8A8 block-scaled via cuBLASLt \
                     (per-token 1x128 act scales x 128x128 weight scales, FP32 epilogue). \
                     METRALE_CUBLAS_GEMM=ssm selected it; METRALE_SSM_OUT_W8A16_ONLY restores W8A16. \
                     This arm allocates nothing."
                );
            } else {
                tracing::info!(
                    "[metrale] SSM out_proj prefill: W8A16 (BF16 act x FP8 weight). \
                     W8A8 cuBLASLt not selected — add `ssm` to METRALE_CUBLAS_GEMM; see #917/#928."
                );
            }
        }
    }
}

#[cfg(test)]
#[path = "prefill_out_w8a8_tests.rs"]
mod tests;
