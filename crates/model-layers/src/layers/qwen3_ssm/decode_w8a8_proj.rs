// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: W8A8 block-scaled cuBLASLt arm for the SSM/GDN decode projections (`in_proj_qkvz`, `out_proj`) at 5..=16 rows.
//!
//! Owner: model-layers (qwen3 SSM).
//! Invariants:
//! - A projection takes this arm only when every clause of
//!   `ops::decode_w8a8_selected` holds, including that the padded write
//!   extent fits `out_capacity_bytes`; otherwise `Ok(false)` leaves it to the
//!   caller's GEMV arms.
//! - The rows' arithmetic moves from W8A16 to W8A8 (E4M3 activations with
//!   per-token 1x128 scales, the checkpoint's FP8 weights and 128x128 block
//!   scales). `METRALE_NO_W8A8_DECODE_PROJ` keeps the GEMV arms.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::Qwen3SsmLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::weight_map::Fp8Weight;

/// 2026-09-25: Which of the two SSM decode projections a route line is about.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SsmDecodeProj {
    Qkvz,
    OutProj,
}

impl SsmDecodeProj {
    fn label(self) -> &'static str {
        match self {
            Self::Qkvz => "in_proj_qkvz",
            Self::OutProj => "out_proj",
        }
    }

    fn log_key(self) -> &'static str {
        match self {
            Self::Qkvz => "log:ssm_qkvz_w8a8_decode",
            Self::OutProj => "log:ssm_out_proj_w8a8_decode",
        }
    }
}

/// 2026-09-25: The [`ops::CublasScope`] bit that arms this family (`ssm`), in one
/// function so a test and the dispatch site read the same bit.
/// `METRALE_CUBLAS_GEMM=ffn` does not arm it.
pub(super) fn ssm_decode_family_armed(scope: ops::CublasScope) -> bool {
    scope.ssm
}

impl Qwen3SsmLayer {
    /// 2026-09-25: The activation-quant scratch for the decode W8A8 arm: the same
    /// arena buffers the SSM prefill projection uses (`fp8_act`,
    /// `fp8_act_scale`, `fp8_act_scale_kmajor`).
    fn decode_w8a8_scratch(&self, ctx: &ForwardContext) -> ops::DecodeW8a8Scratch {
        ops::DecodeW8a8Scratch {
            act_fp8: ctx.buffers.fp8_act(),
            act_fp8_bytes: ctx.buffers.fp8_act_bytes(),
            act_scale: ctx.buffers.fp8_act_scale(),
            act_scale_bytes: ctx.buffers.fp8_act_scale_bytes(),
            act_scale_kmajor: ctx.buffers.fp8_act_scale_kmajor(),
            act_scale_kmajor_bytes: ctx.buffers.fp8_act_scale_kmajor_bytes(),
            quant_k: self.per_token_group_quant_fp8_k,
            scale_kmajor_k: self.fp8_act_scale_kmajor_k,
        }
    }

    /// 2026-09-25: Whether one SSM decode projection takes the W8A8 cuBLASLt arm.
    fn ssm_decode_w8a8_selected(
        &self,
        ctx: &ForwardContext,
        plan: &ops::DecodeW8a8Plan,
        fp8w: &Fp8Weight,
    ) -> bool {
        ops::decode_w8a8_selected(
            ssm_decode_family_armed(ctx.dispatch.cublas),
            ops::w8a8_decode_proj_disabled(),
            plan,
            fp8w.scale_format,
            &self.decode_w8a8_scratch(ctx),
        )
    }

    /// 2026-09-25: Route one SSM decode projection through cuBLASLt W8A8 if every
    /// clause holds; `Ok(false)` leaves it to the caller's GEMV arms.
    ///
    /// `rows` is the caller's row count, `n_out` the projection's output width
    /// and `k` its contract width. `out_capacity_bytes` is the allocated size
    /// of `out`, which bounds the padded write: the padded rows are written.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn try_ssm_decode_w8a8(
        &self,
        ctx: &ForwardContext,
        which: SsmDecodeProj,
        act_bf16: DevicePtr,
        fp8w: &Fp8Weight,
        out: DevicePtr,
        out_capacity_bytes: usize,
        rows: usize,
        n_out: u32,
        k: u32,
        stream: u64,
    ) -> Result<bool> {
        let plan = ops::DecodeW8a8Plan::contiguous(rows, n_out, k, out_capacity_bytes);
        if !self.ssm_decode_w8a8_selected(ctx, &plan, fp8w) {
            return Ok(false);
        }
        self.ssm_decode_w8a8_proj(ctx, which, act_bf16, fp8w, out, &plan, stream)?;
        Ok(true)
    }

    /// 2026-09-25: Run one SSM decode projection through cuBLASLt W8A8: quantize
    /// the activation into the shared scratch (`per_token_group_quant_fp8`,
    /// then `fp8_act_scale_to_kmajor` when the K-major layout is selected),
    /// then one matmul.
    fn ssm_decode_w8a8_proj(
        &self,
        ctx: &ForwardContext,
        which: SsmDecodeProj,
        act_bf16: DevicePtr,
        fp8w: &Fp8Weight,
        out: DevicePtr,
        plan: &ops::DecodeW8a8Plan,
        stream: u64,
    ) -> Result<()> {
        let scratch = self.decode_w8a8_scratch(ctx);
        self.log_decode_w8a8_route(ctx, which, plan);
        ops::decode_w8a8_quant_act(
            ctx.gpu,
            &scratch,
            act_bf16,
            plan.rows as u32,
            plan.k,
            stream,
        )?;
        ops::decode_w8a8_gemm(&scratch, fp8w, out, plan, stream)
    }

    /// 2026-09-25: Log once per projection that its rows took W8A8, so a quality
    /// report can state which arithmetic produced it; the line names the switch
    /// that restores the GEMV arms.
    fn log_decode_w8a8_route(
        &self,
        ctx: &ForwardContext,
        which: SsmDecodeProj,
        plan: &ops::DecodeW8a8Plan,
    ) {
        if ctx.stats.once(which.log_key()) {
            tracing::info!(
                "[metrale] SSM {} decode (n={} rows, N={} K={}): W8A8 block-scaled via cuBLASLt \
                 (per-token 1x128 act scales x 128x128 weight scales, FP32 epilogue; \
                 vLLM-equivalent FP8 numerics), replacing w8a16_gemv_batch16. \
                 METRALE_NO_W8A8_DECODE_PROJ restores the GEMV tier; M=1 decode is untouched.",
                which.label(),
                plan.rows,
                plan.n,
                plan.k,
            );
        }
    }
}

#[cfg(test)]
#[path = "decode_w8a8_proj_tests.rs"]
mod tests;
