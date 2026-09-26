// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The W8A8 block-scaled GEMM of the GDN fused `in_proj_qkvz`
//! prefill projection: cuBLASLt (`ops::cublas_fp8_proj_prequant`) when
//! `qkvz_cublas_selected` holds, else the in-tree `fp8_gemm_t_blockscaled`.
//!
//! The row-wise arm, the CUTLASS/cuBLAS lever arms and
//! `METRALE_GDN_BF16_WEIGHTS` are tried before this one in
//! `trait_prefill_proj.rs`, and `METRALE_FP8_SINGLE_SCALE=1` disables it.
//!
//! Owner: model-layers (qwen3_ssm).
//! Invariants:
//! - Both arms read the FP8 activation and its per-token 1x128 scales from
//!   the arena's `fp8_act` / `fp8_act_scale`, and the cuBLASLt arm writes the
//!   k-major scale copy to the arena's `fp8_act_scale_kmajor`; nothing here
//!   allocates a buffer for them.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, KernelHandle};

use super::Qwen3SsmLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::weight_map::{Fp8Weight, WeightQuantFormat};

/// 2026-09-25: Whether the QKVZ W8A8 prefill takes the cuBLASLt arm. A pure
/// function, so `prefill_w8a8_tests.rs` checks each clause without a GPU.
///
/// All of these must hold:
/// * `cublas_ssm`: `METRALE_CUBLAS_GEMM` names the `ssm` family.
/// * `scale_format == Fp8BlockScaled`: cuBLASLt reads the weight scales as a
///   `[N/128, K/128]` BLK128x128 grid.
/// * `n` and `k` are multiples of 128, and `blk128x128_stride_ok(k)`
///   (`K/128` a multiple of 4, which cuBLAS requires of the scale stride).
/// * `out` holds `m_pad * n` BF16: the helper writes the padded rows.
/// * Under the default k-major scale layout (`cublas_scale_layout_kmajor`),
///   the `fp8_act_scale_to_kmajor` kernel and its scratch exist and the
///   scratch holds `m_pad * K/128` FP32. Without the transpose cuBLASLt reads
///   the scales in the wrong order: measured 2026-09-11 on H100, rel_rms
///   7.7e-2 against the in-tree kernel on identical FP8 bytes.
#[allow(clippy::too_many_arguments)]
pub(super) fn qkvz_cublas_selected(
    cublas_ssm: bool,
    scale_format: WeightQuantFormat,
    m_pad: u32,
    n: u32,
    k: u32,
    out_capacity_bytes: usize,
    scale_kmajor_k: KernelHandle,
    scale_kmajor_buf: DevicePtr,
    scale_kmajor_capacity_bytes: usize,
) -> bool {
    let kmajor_ready = scale_kmajor_k.0 != 0
        && scale_kmajor_buf.0 != 0
        && scale_kmajor_capacity_bytes >= (m_pad as usize) * (k as usize / 128) * 4;
    cublas_ssm
        && scale_format == WeightQuantFormat::Fp8BlockScaled
        && n.is_multiple_of(128)
        && k.is_multiple_of(128)
        && metrale_gpu_runtime::cublaslt::scale_layout::blk128x128_stride_ok(k as usize)
        && (m_pad as usize) * (n as usize) * 2 <= out_capacity_bytes
        && (kmajor_ready || !ops::cublas_scale_layout_kmajor())
}

impl Qwen3SsmLayer {
    /// 2026-09-25: `out[m, n] = a_fp8[m, k] @ weight[n, k]ᵀ` with the activation
    /// and weight block scales applied in an FP32 epilogue: cuBLASLt when
    /// [`qkvz_cublas_selected`] holds, else `fp8_gemm_t_blockscaled`.
    ///
    /// `out_capacity_bytes` is the size of the arena buffer behind `out`
    /// (`ssm_deinterleaved` on a sequential model, `ssm_qkvz` otherwise).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn qkvz_w8a8_gemm(
        &self,
        ctx: &ForwardContext,
        a_fp8: DevicePtr,
        a_scale: DevicePtr,
        fp8w: &Fp8Weight,
        out: DevicePtr,
        out_capacity_bytes: usize,
        m: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        let m_pad = ops::cublas_fp8_m_pad(m);
        let cublas = qkvz_cublas_selected(
            ctx.dispatch.cublas.ssm,
            fp8w.scale_format,
            m_pad,
            n,
            k,
            out_capacity_bytes,
            self.fp8_act_scale_kmajor_k,
            ctx.buffers.fp8_act_scale_kmajor(),
            ctx.buffers.fp8_act_scale_kmajor_bytes(),
        );
        if ctx.stats.once("log:ssm_qkvz_w8a8_prefill") {
            let how = if cublas { "cuBLASLt" } else { "kernel" };
            tracing::info!(
                "[metrale] SSM QKVZ prefill: W8A8 block-scaled via {how} \
                 (per-token 1x128 act scales x 128x128 weight scales, FP32 epilogue). \
                 METRALE_CUBLAS_GEMM=ssm selects cuBLASLt; neither arm allocates."
            );
        }
        if cublas {
            return ops::cublas_fp8_proj_prequant(
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
            );
        }
        ops::fp8_gemm_t_blockscaled(
            ctx.gpu,
            self.fp8_gemm_t_blockscaled_k,
            a_fp8,
            a_scale,
            fp8w.weight,
            fp8w.row_scale,
            out,
            m,
            n,
            k,
            stream,
        )
    }
}

#[cfg(test)]
#[path = "prefill_w8a8_tests.rs"]
mod tests;
