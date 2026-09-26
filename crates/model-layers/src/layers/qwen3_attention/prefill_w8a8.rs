// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The attention prefill O projection's W8A8 block-scaled GEMM:
//! cuBLASLt when `METRALE_CUBLAS_GEMM` names `attn` and every clause holds,
//! otherwise the in-tree `fp8_gemm_t_blockscaled`. Neither route allocates or
//! dequantizes a BF16 copy of the weight; the scratch is arena buffers.
//!
//! Owner: model-layers (qwen3 attention).
//! Invariants:
//! - The cuBLASLt route is taken only when its padded output,
//!   `ceil16(m) * n` BF16, fits in `out_capacity_bytes`.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::weight_map::{Fp8Weight, WeightQuantFormat};

impl Qwen3AttentionLayer {
    /// 2026-09-25: `out[m, n] = a_fp8[m, k] @ weight[n, k]ᵀ`, both block-scale
    /// sets folded in an FP32 epilogue.
    ///
    /// `out_capacity_bytes` is the size of `out`'s arena buffer. cuBLASLt is
    /// given `ceil16(m)` rows and writes all of them
    /// (`ops::cublas_fp8_proj_prequant` zeroes the pad activations), and a
    /// prefill `m` need not be a multiple of 16.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn attn_prefill_w8a8_gemm(
        &self,
        ctx: &ForwardContext,
        a_fp8: DevicePtr,
        a_scale: DevicePtr,
        w: &Fp8Weight,
        out: DevicePtr,
        out_capacity_bytes: usize,
        m: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        let m_pad = ops::cublas_fp8_m_pad(m);
        let kmajor_ready = self.fp8_act_scale_kmajor_k.0 != 0
            && ctx.buffers.fp8_act_scale_kmajor().0 != 0
            && ctx.buffers.fp8_act_scale_kmajor_bytes()
                >= (m_pad as usize) * (k as usize / 128) * 4;
        let cublas = ctx.dispatch.cublas.attn
            && w.scale_format == WeightQuantFormat::Fp8BlockScaled
            && n.is_multiple_of(128)
            && k.is_multiple_of(128)
            && metrale_gpu_runtime::cublaslt::scale_layout::blk128x128_stride_ok(k as usize)
            && (m_pad as usize) * (n as usize) * 2 <= out_capacity_bytes
            && (kmajor_ready || !ops::cublas_scale_layout_kmajor());
        if ctx.stats.once("log:attn_w8a8_prefill") {
            let how = if cublas { "cuBLASLt" } else { "kernel" };
            tracing::info!(
                "[metrale] attention prefill: W8A8 block-scaled via {how} \
                 (per-token 1x128 act scales x 128x128 weight scales, FP32 epilogue). \
                 METRALE_CUBLAS_GEMM=attn selects cuBLASLt; neither arm allocates."
            );
        }
        if cublas {
            return ops::cublas_fp8_proj_prequant(
                ctx.gpu,
                self.fp8_act_scale_kmajor_k,
                a_fp8,
                a_scale,
                ctx.buffers.fp8_act_scale_kmajor(),
                w,
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
            w.weight,
            w.row_scale,
            out,
            m,
            n,
            k,
            stream,
        )
    }
}
