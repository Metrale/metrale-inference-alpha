// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: W8A8 block-scaled cuBLASLt route for the cache-skip Q/K/V
//! prefill (chunk 0, `prefill/cache_skip_qkv.rs`), under
//! `METRALE_CUBLAS_GEMM=attn`. Later chunks go through `prefill/paged_qkv.rs`,
//! which quantizes the activation once per projection; here the three
//! projections share one input, so it is quantized once for all three.
//!
//! Owner: model-layers (qwen3 attention).
//! Invariants:
//! - The route is all three projections or none (`cache_skip_qkv_cublas_selected`
//!   is called once per chain), and it writes Q, then K, then V on one stream.
//! - cuBLASLt writes `ceil16(M)` rows. Q is alone in `qkv_output`, which needs
//!   `ceil16(M) * q_proj_dim` BF16. K sits at `ssm_qkvz` and V right after it,
//!   at row M: K's pad rows (`M..ceil16(M)`, at most 15) land in V's region and
//!   V's own write covers them, because `m >= 16` gives `ceil16(M) - M < M`.
//!   Only V's pad rows survive, so `ssm_qkvz` must hold `(M + ceil16(M)) * kv_dim`
//!   BF16 (`cache_skip_qkv_extents`). Both capacities are checked before the
//!   route is taken; otherwise the chain runs the W8A16 kernels.
//!
//! W8A8 quantizes the activation to E4M3 per 128-wide K group, so it is lossier
//! than W8A16. `examples/native_fp8_prefill_proj_w8a8_microtest` gates it at
//! cosine >= 0.999 and relative RMS <= 3e-2 against W8A16 at the attention
//! shapes. `METRALE_ATTN_QKV_W8A16_ONLY` (presence) keeps W8A16.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, KernelHandle};

use super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::weight_map::{Fp8Weight, WeightQuantFormat};

/// 2026-09-25: `METRALE_ATTN_QKV_W8A16_ONLY`, a presence check (any value,
/// empty and `0` included) read once per process, keeps the cache-skip Q/K/V
/// prefill on W8A16, like `METRALE_FFN_W8A16_ONLY`.
pub(super) fn attn_qkv_w8a16_only() -> bool {
    static ONLY: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ONLY.get_or_init(|| std::env::var_os("METRALE_ATTN_QKV_W8A16_ONLY").is_some())
}

/// 2026-09-25: The extents this chain writes, in BF16 elements, for `m` real
/// rows and the cuBLASLt pad. The selector's capacity clauses use it.
///
/// Returns `(q_elems_in_qkv_output, kv_elems_in_ssm_qkvz)`.
pub(super) fn cache_skip_qkv_extents(m: u32, q_proj_dim: u32, kv_dim: u32) -> (usize, usize) {
    let m_pad = ops::cublas_fp8_m_pad(m) as usize;
    // 2026-09-25: `v` starts at row `m` of `ssm_qkvz` and writes `m_pad` rows.
    (
        m_pad * q_proj_dim as usize,
        (m as usize + m_pad) * kv_dim as usize,
    )
}

/// 2026-09-25: Whether the whole cache-skip Q/K/V chain takes the cuBLASLt W8A8
/// route. A pure function, so CPU tests pin each clause.
///
/// One decision for all three projections: the pad-row argument in the module
/// header needs K and V on the same route, in that order.
///
/// Clauses:
///
/// * `!w8a16_only`: the switch above.
/// * `cublas_attn`: `METRALE_CUBLAS_GEMM` names `attn`.
/// * `fp8_blockscaled_prefill`: off when `METRALE_FP8_SINGLE_SCALE` is set.
/// * `m >= 16`: the covering argument for K's pad rows (module header). The
///   dense FFN's W8A8 prefill uses `m > 4`.
/// * all three weights `Fp8BlockScaled`: cuBLASLt reads the weight scales as a
///   `[N/128, K/128]` grid.
/// * `q_n`, `kv_n` and `k` multiples of 128: that grid, and one activation scale
///   per 128 of K.
/// * `blk128x128_stride_ok(k)` (`k % 512 == 0`): the weight scale column stride
///   `K/128` must be a multiple of 4.
/// * room in both destination buffers for the padded extents above.
/// * activation scratch for `ceil16(M) * K` FP8 and its scales, pad rows
///   included.
/// * the quantizer kernel, and, when the k-major scale layout is in use, its
///   adapter (kernel, scratch, capacity); `cublas_fp8_proj_prequant` refuses
///   to run without it. Measured 2026-09-11 on H100: without the transpose the
///   result was wrong (relative RMS 7.7e-2 on the same FP8 bytes).
#[allow(clippy::too_many_arguments)]
pub(super) fn cache_skip_qkv_cublas_selected(
    cublas_attn: bool,
    fp8_blockscaled_prefill: bool,
    w8a16_only: bool,
    formats: [Option<WeightQuantFormat>; 3],
    m: u32,
    q_n: u32,
    kv_n: u32,
    k: u32,
    qkv_output_capacity_bytes: usize,
    ssm_qkvz_capacity_bytes: usize,
    act_capacity_bytes: usize,
    act_scale_capacity_bytes: usize,
    quant_k: ops::Fp8ActQuant,
    scale_kmajor_k: KernelHandle,
    scale_kmajor_buf: DevicePtr,
    scale_kmajor_capacity_bytes: usize,
) -> bool {
    let m_pad = ops::cublas_fp8_m_pad(m) as usize;
    let kg = k as usize / 128;
    let (q_elems, kv_elems) = cache_skip_qkv_extents(m, q_n, kv_n);
    let kmajor_ready = scale_kmajor_k.0 != 0
        && scale_kmajor_buf.0 != 0
        && scale_kmajor_capacity_bytes >= m_pad * kg * 4;
    !w8a16_only
        && cublas_attn
        && fp8_blockscaled_prefill
        && m >= 16
        && formats
            .iter()
            .all(|f| *f == Some(WeightQuantFormat::Fp8BlockScaled))
        && q_n.is_multiple_of(128)
        && kv_n.is_multiple_of(128)
        && k.is_multiple_of(128)
        && metrale_gpu_runtime::cublaslt::scale_layout::blk128x128_stride_ok(k as usize)
        && q_elems * 2 <= qkv_output_capacity_bytes
        && kv_elems * 2 <= ssm_qkvz_capacity_bytes
        && m_pad * (k as usize) <= act_capacity_bytes
        && m_pad * kg * 4 <= act_scale_capacity_bytes
        && quant_k.available()
        && (kmajor_ready || !ops::cublas_scale_layout_kmajor())
}

impl Qwen3AttentionLayer {
    /// 2026-09-25: Whether this layer's cache-skip Q/K/V chain runs on
    /// cuBLASLt W8A8. `m` is the chunk's token count, `k` the hidden size. The
    /// caller (`cache_skip_qkv.rs`) asks once per chain.
    pub(super) fn cache_skip_qkv_w8a8_selected(
        &self,
        ctx: &ForwardContext,
        m: u32,
        q_proj_dim: u32,
        kv_dim: u32,
        k: u32,
    ) -> bool {
        let fmt = |w: Option<&crate::weight_map::QuantWeight>| {
            w.and_then(|w| w.as_fp8()).map(|f| f.scale_format)
        };
        cache_skip_qkv_cublas_selected(
            ctx.dispatch.cublas.attn,
            ctx.dispatch.fp8_blockscaled_prefill,
            attn_qkv_w8a16_only(),
            [
                fmt(self.q_weight.as_ref()),
                fmt(self.k_weight.as_ref()),
                fmt(self.v_weight.as_ref()),
            ],
            m,
            q_proj_dim,
            kv_dim,
            k,
            ctx.buffers.qkv_output_bytes(),
            ctx.buffers.ssm_qkvz_bytes(),
            ctx.buffers.fp8_act_bytes(),
            ctx.buffers.fp8_act_scale_bytes(),
            self.per_token_group_quant_fp8_k,
            self.fp8_act_scale_kmajor_k,
            ctx.buffers.fp8_act_scale_kmajor(),
            ctx.buffers.fp8_act_scale_kmajor_bytes(),
        )
    }

    /// 2026-09-25: Quantize `normed[m, k]` once into the arena's `fp8_act` and
    /// `fp8_act_scale` for all three projections, on the chain's stream.
    pub(super) fn cache_skip_qkv_w8a8_quant(
        &self,
        ctx: &ForwardContext,
        normed: DevicePtr,
        m: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        let m_pad = ops::cublas_fp8_m_pad(m) as usize;
        debug_assert!(m_pad * k as usize <= ctx.buffers.fp8_act_bytes());
        debug_assert!(m_pad * (k as usize / 128) * 4 <= ctx.buffers.fp8_act_scale_bytes());
        ops::per_token_group_quant_fp8(
            ctx.gpu,
            self.per_token_group_quant_fp8_k,
            normed,
            ctx.buffers.fp8_act(),
            ctx.buffers.fp8_act_scale(),
            m,
            k,
            stream,
        )
    }

    /// 2026-09-25: One projection of the chain through cuBLASLt,
    /// `out[m, n] = a_fp8[m, k] @ weight[n, k]ᵀ` with both block-scale sets
    /// folded in an FP32 epilogue, on the activation `cache_skip_qkv_w8a8_quant`
    /// wrote. The caller must have checked `cache_skip_qkv_w8a8_selected`; this
    /// neither re-checks nor falls back.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn cache_skip_qkv_w8a8_gemm(
        &self,
        ctx: &ForwardContext,
        fp8w: &Fp8Weight,
        out: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        ops::cublas_fp8_proj_prequant(
            ctx.gpu,
            self.fp8_act_scale_kmajor_k,
            ctx.buffers.fp8_act(),
            ctx.buffers.fp8_act_scale(),
            ctx.buffers.fp8_act_scale_kmajor(),
            fp8w,
            out,
            m,
            n,
            k,
            stream,
        )
    }

    /// 2026-09-25: Log the route of the first cache-skip Q/K/V prefill
    /// (`ctx.stats.once`); either route writes the line.
    pub(super) fn log_cache_skip_qkv_route(&self, ctx: &ForwardContext, cublas: bool) {
        if ctx.stats.once("log:attn_cache_skip_qkv_prefill") {
            if cublas {
                tracing::info!(
                    "[metrale] attention Q/K/V prefill (chunk 0, cache-skip): W8A8 block-scaled \
                     via cuBLASLt, activation quantized once for all three \
                     (per-token 1x128 act scales x 128x128 weight scales, FP32 epilogue). \
                     METRALE_CUBLAS_GEMM=attn selected it; METRALE_ATTN_QKV_W8A16_ONLY restores \
                     W8A16. This arm allocates nothing."
                );
            } else {
                tracing::info!(
                    "[metrale] attention Q/K/V prefill (chunk 0, cache-skip): W8A16 \
                     (BF16 act x FP8 weight). W8A8 cuBLASLt not selected — add `attn` to \
                     METRALE_CUBLAS_GEMM; see #917/#928."
                );
            }
        }
    }
}

#[cfg(test)]
#[path = "prefill_qkv_w8a8_tests.rs"]
mod tests;
