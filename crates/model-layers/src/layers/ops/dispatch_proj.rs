// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: cuBLASLt projection routes (block-scaled FP8, BF16) and the
//! FP8-to-BF16 weight dequant.
//!
//! Owner: model-layers ops.
//! Invariants:
//! - The block-scaled FP8 route hands cuBLASLt `cublas_fp8_m_pad(m)` rows, and
//!   zeroes the pad rows' FP8 bytes and activation scales before the GEMM.

#![allow(unused_imports)]

use super::*;

// 2026-09-25: `ops.rs` loads this file with `#[path]`, so a plain
// `mod cutlass;` would resolve in `ops/`, not `ops/dispatch_proj/`; hence the
// explicit `#[path]`.
#[path = "dispatch_proj/cutlass.rs"]
mod cutlass;
pub use cutlass::*;

/// 2026-09-25: Which VEC128 activation-scale layout the cuBLASLt block-scaled
/// arm passes, from `METRALE_CUBLAS_SCALE_LAYOUT`, read once per process.
///
/// * `true` (unset, or any value but `rowmajor`): `[K/128, ceil16(M)]`, tokens
///   contiguous, the layout `metrale_gpu_runtime::cublaslt::scale_layout`
///   quotes from the cuBLAS manual.
/// * `false` (`rowmajor`): the quantizer's `[M, K/128]` passed as is. It is a
///   measurement control: measured 2026-09-11 on H100, it gave rel_rms 7.7e-2
///   against the in-tree kernel.
pub fn cublas_scale_layout_kmajor() -> bool {
    static KMAJOR: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *KMAJOR.get_or_init(|| {
        !matches!(
            std::env::var("METRALE_CUBLAS_SCALE_LAYOUT").as_deref(),
            Ok("rowmajor")
        )
    })
}

/// 2026-09-25: Launch `fp8_act_scale_to_kmajor`: copy the quantizer's
/// row-major `[M, K/128]` FP32 activation scales into `[K/128, M_pad]`, with
/// zeros in the `M..M_pad` pad rows. `a_scale` is only read.
///
/// `metrale_gpu_runtime::cublaslt::scale_layout` holds the same index math and
/// its CPU tests.
pub fn fp8_act_scale_to_kmajor(
    gpu: &dyn metrale_gpu_runtime::gpu::GpuBackend,
    kernel: metrale_gpu_runtime::gpu::KernelHandle,
    a_scale: metrale_gpu_runtime::gpu::DevicePtr,
    a_scale_kmajor: metrale_gpu_runtime::gpu::DevicePtr,
    m: u32,
    m_pad: u32,
    k: u32,
    stream: u64,
) -> anyhow::Result<()> {
    use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};
    let l = k / 128;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(m_pad, 256), l, 1])
        .block([256, 1, 1])
        .arg_ptr(a_scale)
        .arg_ptr(a_scale_kmajor)
        .arg_u32(m)
        .arg_u32(m_pad)
        .arg_u32(l)
        .launch(stream)
}

/// 2026-09-25: Quantize the activation with `per_token_group_quant_fp8` (FP8
/// plus one scale per token and 128 of K), then run
/// [`cublas_fp8_proj_prequant`]. The FP8 weight and its 128x128 block scales
/// go to cuBLASLt as they are, with no dequant.
///
/// The scratch buffers need the extents listed on
/// [`cublas_fp8_proj_prequant`].
#[allow(clippy::too_many_arguments)]
pub fn cublas_fp8_proj(
    gpu: &dyn metrale_gpu_runtime::gpu::GpuBackend,
    ptg_quant_k: Fp8ActQuant,
    scale_kmajor_k: metrale_gpu_runtime::gpu::KernelHandle,
    act_bf16: metrale_gpu_runtime::gpu::DevicePtr,
    act_fp8_scratch: metrale_gpu_runtime::gpu::DevicePtr,
    act_scale_scratch: metrale_gpu_runtime::gpu::DevicePtr,
    act_scale_kmajor_scratch: metrale_gpu_runtime::gpu::DevicePtr,
    fp8w: &crate::weight_map::Fp8Weight,
    out: metrale_gpu_runtime::gpu::DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> anyhow::Result<()> {
    per_token_group_quant_fp8(
        gpu,
        ptg_quant_k,
        act_bf16,
        act_fp8_scratch,
        act_scale_scratch,
        m,
        k,
        stream,
    )?;
    cublas_fp8_proj_prequant(
        gpu,
        scale_kmajor_k,
        act_fp8_scratch,
        act_scale_scratch,
        act_scale_kmajor_scratch,
        fp8w,
        out,
        m,
        n,
        k,
        stream,
    )
}

/// 2026-09-25: [`cublas_fp8_proj`] for an activation the caller already ran
/// through `per_token_group_quant_fp8`, so one quantization can feed several
/// projections (the dense FFN's gate and up share one).
///
/// With [`cublas_scale_layout_kmajor`] on, the scales are first transposed
/// into `act_scale_kmajor`; a zero kernel handle or scratch pointer is an
/// error.
///
/// cuBLASLt is handed `ceil16(M)` rows ([`cublas_fp8_m_pad`]), so:
///
/// * `out` must hold `ceil16(M) * N` BF16 elements; the pad rows are written.
/// * `act_fp8` must hold `ceil16(M) * K` bytes; the pad rows are zeroed here.
/// * `act_scale_kmajor` must hold `ceil16(M) * (K/128)` f32, and `act_scale`
///   `M * (K/128)` f32, or `ceil16(M) * (K/128)` under `rowmajor`, where the
///   pad rows are zeroed in place.
#[allow(clippy::too_many_arguments)]
pub fn cublas_fp8_proj_prequant(
    gpu: &dyn metrale_gpu_runtime::gpu::GpuBackend,
    scale_kmajor_k: metrale_gpu_runtime::gpu::KernelHandle,
    act_fp8: metrale_gpu_runtime::gpu::DevicePtr,
    act_scale: metrale_gpu_runtime::gpu::DevicePtr,
    act_scale_kmajor: metrale_gpu_runtime::gpu::DevicePtr,
    fp8w: &crate::weight_map::Fp8Weight,
    out: metrale_gpu_runtime::gpu::DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> anyhow::Result<()> {
    let m_pad = cublas_fp8_m_pad(m);
    let kg = (k / 128) as usize;
    if m_pad > m {
        // 2026-09-25: A zero scale alone is not enough: the FP8 dot product
        // still reads the pad bytes, and `NaN * 0.0` is `NaN`.
        gpu.memset_async(
            act_fp8.offset(m as usize * k as usize),
            0,
            (m_pad - m) as usize * k as usize,
            stream,
        )?;
    }
    let b_scale = if cublas_scale_layout_kmajor() {
        if scale_kmajor_k.0 == 0 || act_scale_kmajor.0 == 0 {
            anyhow::bail!(
                "cuBLASLt block-scaled FP8 needs the fp8_act_scale_to_kmajor adapter \
                 (kernel={:#x}, scratch={:#x}) — see cublas_scale_layout_kmajor()",
                scale_kmajor_k.0,
                act_scale_kmajor.0
            );
        }
        // 2026-09-25: The adapter writes every `[K/128, m_pad]` slot, pad rows
        // included, so the scale pad needs no memset.
        fp8_act_scale_to_kmajor(
            gpu,
            scale_kmajor_k,
            act_scale,
            act_scale_kmajor,
            m,
            m_pad,
            k,
            stream,
        )?;
        act_scale_kmajor
    } else {
        // 2026-09-25: In the `rowmajor` layout the pad rows are a contiguous
        // tail, so they are zeroed in place.
        if m_pad > m {
            gpu.memset_async(
                act_scale.offset(m as usize * kg * 4),
                0,
                (m_pad - m) as usize * kg * 4,
                stream,
            )?;
        }
        act_scale
    };
    metrale_gpu_runtime::cublaslt::fp8_gemm_act_weight_t_blkscaled(
        act_fp8.0,
        b_scale.0,
        fp8w.weight.0,
        fp8w.row_scale.0,
        out.0,
        m_pad,
        n,
        k,
        stream,
    )
}

/// 2026-09-25: The M extent [`cublas_fp8_proj_prequant`] hands cuBLASLt; callers
/// bound their output buffers against it.
pub fn cublas_fp8_m_pad(m: u32) -> u32 {
    m.div_ceil(16) * 16
}

/// 2026-09-25: Dequantize a block-scaled or per-row FP8 weight `[N,K]` to BF16
/// into `dst`, a caller-owned buffer of [`dequant_fp8_bf16_bytes`] bytes.
///
/// One kernel serves both scale layouts; the block geometry passed to it
/// selects between them:
///
///   block-scaled   block_n = block_k = 128, sk = K/128
///   per-row        block_n = 1, block_k = K, sk = 1
pub fn dequant_fp8_bf16_into(
    gpu: &dyn metrale_gpu_runtime::gpu::GpuBackend,
    fp8w: &crate::weight_map::Fp8Weight,
    dst: metrale_gpu_runtime::gpu::DevicePtr,
    stream: u64,
) -> anyhow::Result<()> {
    use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};
    let (n, kk) = (fp8w.n, fp8w.k);
    let per_row = fp8w.scale_format == crate::weight_map::WeightQuantFormat::Fp8PerRow;
    let (block_n, block_k, sk) = if per_row {
        (1u32, kk, 1u32)
    } else {
        (128u32, 128u32, kk / 128)
    };
    let kernel = gpu.kernel(
        "dequant_fp8_blockscaled_bf16",
        "dequant_fp8_blockscaled_bf16",
    )?;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(kk, 64), div_ceil(n, 4), 1])
        .block([64, 4, 1])
        .arg_ptr(fp8w.weight)
        .arg_ptr(fp8w.row_scale)
        .arg_ptr(dst)
        .arg_u32(n)
        .arg_u32(kk)
        .arg_u32(block_n)
        .arg_u32(block_k)
        .arg_u32(sk)
        .arg_u32(1)
        .launch(stream)
}

/// 2026-09-25: BF16 bytes [`dequant_fp8_bf16_into`] writes for `fp8w`.
pub fn dequant_fp8_bf16_bytes(fp8w: &crate::weight_map::Fp8Weight) -> usize {
    fp8w.n as usize * fp8w.k as usize * 2
}

/// 2026-09-25: Route a projection `out[M,N] = act[M,K] @ weightᵀ` through
/// cuBLASLt BF16 for a weight that is already BF16 `[N,K]`.
pub fn cublas_bf16_proj_dense(
    act: metrale_gpu_runtime::gpu::DevicePtr,
    weight_bf16: metrale_gpu_runtime::gpu::DevicePtr,
    out: metrale_gpu_runtime::gpu::DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> anyhow::Result<()> {
    metrale_gpu_runtime::cublaslt::bf16_gemm_act_weight_t(
        act.0,
        weight_bf16.0,
        out.0,
        m,
        n,
        k,
        stream,
    )
}

/// 2026-09-25: [`cublas_bf16_proj_dense`] writing an FP32 output buffer, for
/// consumers whose next kernel reads FP32.
pub fn cublas_bf16_proj_dense_f32_out(
    act: metrale_gpu_runtime::gpu::DevicePtr,
    weight_bf16: metrale_gpu_runtime::gpu::DevicePtr,
    out: metrale_gpu_runtime::gpu::DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> anyhow::Result<()> {
    metrale_gpu_runtime::cublaslt::bf16_gemm_act_weight_t_f32_out(
        act.0,
        weight_bf16.0,
        out.0,
        m,
        n,
        k,
        stream,
    )
}
