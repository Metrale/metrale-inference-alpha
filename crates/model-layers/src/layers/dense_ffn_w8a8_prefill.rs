// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The W8A8 block-scaled arm of the native-FP8 dense FFN: which projections take it, and the GEMM it runs.
//!
//! Owner: model-layers (dense FFN).
//! Invariants:
//! - `w8a8_prefill_selected` holds only for `4 < m <= max_m`, `Fp8BlockScaled`
//!   weights, `k` and `n` multiples of 128, the quantizer and GEMM handles
//!   loaded, `METRALE_FFN_W8A16_ONLY` unset and
//!   `dispatch.fp8_blockscaled_prefill` on.
//!
//! The arm is reachable only on a layer that holds native-FP8 dense-FFN
//! weights. The one loader that installs them, `qwen35_dense.rs`, does so only
//! when `ffn_fp8_arm_selected` holds: `METRALE_DENSE_FP8` is exactly `1`,
//! `config.tp_world_size.max(1) == 1`, the variant is
//! `Nvfp4Variant::Fp8Dequanted`, and the layer's `mlp.gate_proj` is block-scaled
//! FP8 on disk (`proj_is_native_fp8`). Without it neither route log in this file
//! is emitted.
//!
//! The activation is quantized to E4M3 with one FP32 scale per token per
//! 128-wide K group (`per_token_group_quant_fp8`). `w8a8_gemm` then runs
//! cuBLASLt `fp8_gemm_act_weight_t_blkscaled` when `dispatch.cublas.ffn` holds
//! (`METRALE_CUBLAS_GEMM` lists `ffn`, or is `1`, `true` or `all`) and its other
//! clauses pass, and the in-tree `ops::fp8_gemm_t_blockscaled` otherwise.
//! `examples/native_fp8_ffn_w8a8_microtest.rs` grades W8A8 against the W8A16
//! reference at cosine >= 0.999 and relative RMS <= 2%.
//!
//! The upper M bound is per target: `kernels/<hw>/HARDWARE.toml` `[defaults]
//! w8a8_prefill_max_m_widening` / `_narrowing`. gb10 declares 64 and 384 beside
//! its measurement; every other target has no cap (`u32::MAX`).

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, KernelHandle};

use super::DenseFfnLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::weight_map::{Fp8Weight, WeightQuantFormat};

/// 2026-09-25: `METRALE_FFN_W8A16_ONLY`, by presence (any value, including
/// empty or `0`): no dense-FFN projection takes the W8A8 arm. Read once per
/// process (`OnceLock`).
pub fn ffn_w8a16_only() -> bool {
    static ONLY: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ONLY.get_or_init(|| std::env::var_os("METRALE_FFN_W8A16_ONLY").is_some())
}

/// 2026-09-25: The W8A8 upper M bound for a projection of shape `[n, k]`:
/// `w8a8_prefill_max_m_widening` when `n > k` (gate/up) and
/// `w8a8_prefill_max_m_narrowing` otherwise (down), from
/// [`ops::target_defaults::resolved`].
pub(crate) fn max_m_for(n: u32, k: u32) -> u32 {
    let levers = ops::target_defaults::resolved();
    if n > k {
        levers.w8a8_prefill_max_m_widening.value
    } else {
        levers.w8a8_prefill_max_m_narrowing.value
    }
}

/// 2026-09-25: The W8A8 selection rule, as a pure function, so the CPU tests
/// can pin every clause without a `ForwardContext`. `w8a16_only` and `max_m`
/// are arguments because their production readers are process-global.
///
/// Clauses:
///
/// * `m > 4`: `m <= 4` belongs to the `w8a16_gemv_batch4` rung.
/// * `fp8_blockscaled_prefill`: cleared by `METRALE_FP8_SINGLE_SCALE=1`.
/// * `Fp8BlockScaled`: the GEMM indexes `row_scale` as the `[N/128, K/128]`
///   grid, which per-row and single-scale weights do not have.
/// * `k % 128 == 0`: the activation quantizer emits one scale per 128-wide K
///   group.
/// * `n % 128 == 0`: the weight scale grid is `[N/128, K/128]`.
/// * `m <= max_m`: the per-target upper bound. `u32::MAX` is no cap and 0
///   turns the arm off.
/// * the quantizer and the GEMM handles are loaded.
#[allow(clippy::too_many_arguments)]
pub(crate) fn w8a8_prefill_selected(
    m: u32,
    n: u32,
    k: u32,
    scale_format: WeightQuantFormat,
    fp8_blockscaled_prefill: bool,
    quant_k: ops::Fp8ActQuant,
    gemm_k: KernelHandle,
    w8a16_only: bool,
    max_m: u32,
) -> bool {
    !w8a16_only
        && m > 4
        && m <= max_m
        && fp8_blockscaled_prefill
        && scale_format == WeightQuantFormat::Fp8BlockScaled
        && k.is_multiple_of(128)
        && n.is_multiple_of(128)
        && quant_k.available()
        && gemm_k.0 != 0
}

impl DenseFfnLayer {
    /// 2026-09-25: Whether one dense-FFN projection takes the W8A8 arm.
    ///
    /// Beyond [`w8a8_prefill_selected`] this requires the arena's dense-FFN
    /// activation scratch (`ffn_act_a`, `ffn_act_scale`), which is 0 for MoE
    /// configs. It is a gate and not an assert: a null scratch would be a launch
    /// writing to address 0.
    pub(crate) fn prefill_w8a8_selected(
        &self,
        ctx: &ForwardContext,
        m: u32,
        n: u32,
        k: u32,
        w: &Fp8Weight,
    ) -> bool {
        w8a8_prefill_selected(
            m,
            n,
            k,
            w.scale_format,
            ctx.dispatch.fp8_blockscaled_prefill,
            self.per_token_group_quant_fp8_k,
            self.fp8_gemm_t_blockscaled_k,
            ffn_w8a16_only(),
            max_m_for(n, k),
        ) && ctx.buffers.ffn_act_a().0 != 0
            && ctx.buffers.ffn_act_scale().0 != 0
    }

    /// 2026-09-26: Quantize `act[m, k]` BF16 into the arena's dense-FFN scratch as
    /// FP8 E4M3 plus one FP32 scale per token per 128-wide K group; returns
    /// `(a_fp8, a_scale)`.
    ///
    /// The arena has one scratch pair, so the caller consumes the result before
    /// the next call. `prefill_fp8` (reached from `forward_prefill_inner`) issues the gate/up GEMMs, the
    /// SiLU-mul and the down quantization in order on one stream. `k` is `hidden`
    /// or `intermediate`; the scratch is sized for the larger.
    pub(crate) fn w8a8_quant_act(
        &self,
        ctx: &ForwardContext,
        act: DevicePtr,
        m: u32,
        k: u32,
        stream: u64,
    ) -> Result<(DevicePtr, DevicePtr)> {
        let a_fp8 = ctx.buffers.ffn_act_a();
        let a_scale = ctx.buffers.ffn_act_scale();
        // 2026-09-25: Padded extents: the cuBLASLt arm reads `ceil16(m)` rows.
        let rows = ops::cublas_fp8_m_pad(m) as usize;
        debug_assert!(rows * k as usize <= ctx.buffers.ffn_act_a_bytes());
        debug_assert!(rows * (k as usize / 128) * 4 <= ctx.buffers.ffn_act_scale_bytes());
        ops::per_token_group_quant_fp8(
            ctx.gpu,
            self.per_token_group_quant_fp8_k,
            act,
            a_fp8,
            a_scale,
            m,
            k,
            stream,
        )?;
        Ok((a_fp8, a_scale))
    }

    /// 2026-09-25: `out[m, n] = a_fp8[m, k] @ weight[n, k]^T` with both scale sets
    /// applied in FP32: cuBLASLt when `dispatch.cublas.ffn` holds and the clauses
    /// below pass, else the in-tree kernel.
    ///
    /// `out_capacity_bytes` is the allocated size of `out`. The cuBLASLt helper
    /// rounds M up to 16 and writes those rows too, so an `out` smaller than the
    /// padded extent sends the GEMM to the in-tree kernel instead.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn w8a8_gemm(
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
        let m_pad = ops::cublas_fp8_m_pad(m) as usize;
        let padded_out_bytes = m_pad * n as usize * 2;
        // 2026-09-25: The cuBLASLt arm also needs:
        //
        // * output room for the padded rows;
        // * when the k-major scale layout is in use (the default;
        //   `METRALE_CUBLAS_SCALE_LAYOUT=rowmajor` turns it off), the
        //   `fp8_act_scale_to_kmajor` kernel and its scratch. cuBLASLt reads the
        //   activation scales token-contiguous. Measured on H100 2026-09-11 without
        //   the transpose: rel_rms 7.7e-2 against the in-tree kernel;
        // * `k / 128` a multiple of 4 (`blk128x128_stride_ok`): the weight scales
        //   are handed over as the checkpoint's `[N/128, K/128]` grid.
        let scale_layout_ready = ctx.buffers.ffn_act_scale_kmajor().0 != 0
            && self.fp8_act_scale_kmajor_k.0 != 0
            && ctx.buffers.ffn_act_scale_kmajor_bytes() >= m_pad * (k as usize / 128) * 4;
        let cublas = ctx.dispatch.cublas.ffn
            && padded_out_bytes <= out_capacity_bytes
            && metrale_gpu_runtime::cublaslt::scale_layout::blk128x128_stride_ok(k as usize)
            && (scale_layout_ready || !ops::cublas_scale_layout_kmajor());
        self.log_w8a8_prefill_route(ctx, cublas);
        if cublas {
            return ops::cublas_fp8_proj_prequant(
                ctx.gpu,
                self.fp8_act_scale_kmajor_k,
                a_fp8,
                a_scale,
                ctx.buffers.ffn_act_scale_kmajor(),
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

    /// 2026-09-25: Logs once per model (`log:ffn_w8a8_prefill`) which GEMM the
    /// W8A8 arm used.
    fn log_w8a8_prefill_route(&self, ctx: &ForwardContext, cublas: bool) {
        if ctx.stats.once("log:ffn_w8a8_prefill") {
            let how = if cublas { "cuBLASLt" } else { "kernel" };
            tracing::info!(
                "[metrale] dense FFN prefill: W8A8 block-scaled via {how} \
                 (per-token 1x128 act scales x 128x128 weight scales, FP32 epilogue; \
                 vLLM-equivalent FP8 numerics). METRALE_FFN_W8A16_ONLY=1 restores W8A16."
            );
        }
    }

    /// 2026-09-25: Logs once per model (`log:ffn_w8a16_prefill`) when a prefill
    /// takes the W8A8 arm for neither gate/up nor down while the W8A8 arm is
    /// reachable (neither the batch16 nor a tensor-core tier claims `m`).
    pub(crate) fn log_w8a16_prefill_route(&self, ctx: &ForwardContext) {
        if ctx.stats.once("log:ffn_w8a16_prefill") {
            tracing::info!(
                "[metrale] dense FFN prefill: W8A16 (BF16 act x FP8 weight). \
                 W8A8 not selected — see #917/#928."
            );
        }
    }
}

#[cfg(test)]
#[path = "dense_ffn_w8a8_prefill_tests.rs"]
mod tests;
