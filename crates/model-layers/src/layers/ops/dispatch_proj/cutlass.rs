// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: CUTLASS projection routes: BF16 from an FP8 weight, and NVFP4
//! from an NVFP4 or an FP8 weight.
//!
//! Owner: model-layers ops.
//! Invariants:
//! - Each derived weight is memoized in the model's `DerivedWeights` under the
//!   source weight's device pointer, and only after it was built without
//!   error.

use super::*;

/// 2026-09-25: [`dequant_fp8_bf16_into`] into a new allocation, memoized by
/// the FP8 weight's pointer.
///
/// The allocation is a plain `gpu.alloc` with no `BufferSizes` entry. A new
/// caller should pass its own budgeted destination to
/// [`dequant_fp8_bf16_into`] instead.
fn dequant_fp8_bf16_cached(
    gpu: &dyn metrale_gpu_runtime::gpu::GpuBackend,
    derived: &super::DerivedWeights,
    fp8w: &crate::weight_map::Fp8Weight,
    stream: u64,
) -> anyhow::Result<u64> {
    let cache_key = fp8w.weight.0;
    if let Some(hit) = derived.get_ptr(super::Derivation::Bf16, cache_key) {
        return Ok(hit);
    }
    let out = gpu.alloc(dequant_fp8_bf16_bytes(fp8w))?;
    dequant_fp8_bf16_into(gpu, fp8w, out, stream)?;
    derived.insert_ptr(super::Derivation::Bf16, cache_key, out.0);
    Ok(out.0)
}

/// 2026-09-25: [`dequant_fp8_bf16_into`] into a new allocation that the caller
/// must free.
fn dequant_fp8_bf16_uncached(
    gpu: &dyn metrale_gpu_runtime::gpu::GpuBackend,
    fp8w: &crate::weight_map::Fp8Weight,
    stream: u64,
) -> anyhow::Result<metrale_gpu_runtime::gpu::DevicePtr> {
    let out = gpu.alloc(dequant_fp8_bf16_bytes(fp8w))?;
    dequant_fp8_bf16_into(gpu, fp8w, out, stream)?;
    Ok(out)
}

/// 2026-09-25: Route a projection `out[M,N] = act[M,K] @ weightᵀ` through
/// CUTLASS BF16, against the FP8 weight's memoized BF16 copy.
#[allow(clippy::too_many_arguments)]
pub fn cutlass_bf16_proj(
    gpu: &dyn metrale_gpu_runtime::gpu::GpuBackend,
    derived: &super::DerivedWeights,
    act: metrale_gpu_runtime::gpu::DevicePtr,
    fp8w: &crate::weight_map::Fp8Weight,
    out: metrale_gpu_runtime::gpu::DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> anyhow::Result<()> {
    let w_bf16 = dequant_fp8_bf16_cached(gpu, derived, fp8w, stream)?;
    metrale_gpu_runtime::cutlass::bf16_gemm_act_weight_t(act.0, w_bf16, out.0, m, n, k, stream)
}

#[allow(clippy::too_many_arguments)]
/// 2026-09-25: Transpose an NVFP4 weight's packed bytes from `[K/2,N]` into the
/// `[N,K/2]` layout the CUTLASS GEMM reads, memoized by the source weight's
/// pointer.
fn cutlass_nvfp4_weight_transposed_cached(
    gpu: &dyn metrale_gpu_runtime::gpu::GpuBackend,
    derived: &super::DerivedWeights,
    weight_t: &crate::weight_map::QuantizedWeight,
    n: u32,
    k: u32,
    stream: u64,
) -> anyhow::Result<u64> {
    let cache_key = weight_t.weight.0;
    if let Some(hit) = derived.get_ptr(super::Derivation::CutlassNvfp4Transposed, cache_key) {
        return Ok(hit);
    }
    let dst = gpu.alloc((n as usize) * (k as usize) / 2)?;
    metrale_gpu_runtime::cutlass::transpose_nvfp4_packed_kton(
        weight_t.weight.0,
        dst.0,
        n,
        k,
        stream,
    )?;
    gpu.synchronize(stream)?;
    derived.insert_ptr(super::Derivation::CutlassNvfp4Transposed, cache_key, dst.0);
    Ok(dst.0)
}

#[allow(clippy::too_many_arguments)]
/// 2026-09-25: Route a projection `out[M,N] = act[M,K] @ weightᵀ` through
/// CUTLASS NVFP4. `weight_t` is an NVFP4 weight with `[K/2,N]` packed bytes;
/// the bytes are transposed once to `[N,K/2]` and memoized.
pub fn cutlass_nvfp4_proj(
    ctx: &crate::layer::ForwardContext<'_>,
    act: metrale_gpu_runtime::gpu::DevicePtr,
    weight_t: &crate::weight_map::QuantizedWeight,
    out: metrale_gpu_runtime::gpu::DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> anyhow::Result<()> {
    let (gpu, derived) = (ctx.gpu, ctx.derived);
    let packed = cutlass_nvfp4_weight_transposed_cached(gpu, derived, weight_t, n, k, stream)?;
    metrale_gpu_runtime::cutlass::nvfp4_gemm_bf16_act_weight_t(
        act.0,
        packed,
        weight_t.weight_scale.0,
        weight_t.weight_scale_2,
        out.0,
        m,
        n,
        k,
        stream,
    )
}

fn cutlass_nvfp4_weight_from_fp8_cached(
    gpu: &dyn metrale_gpu_runtime::gpu::GpuBackend,
    derived: &super::DerivedWeights,
    fp8w: &crate::weight_map::Fp8Weight,
    stream: u64,
) -> anyhow::Result<(u64, u64)> {
    let cache_key = fp8w.weight.0;
    if let Some(hit) = derived.get_pair(super::Derivation::CutlassNvfp4FromFp8, cache_key) {
        return Ok(hit);
    }

    let n = fp8w.n as usize;
    let k = fp8w.k as usize;
    let w_bf16 = dequant_fp8_bf16_uncached(gpu, fp8w, stream)?;
    let packed_t = gpu.alloc(n * k / 2)?;
    let scale_t = gpu.alloc(n * k / 16)?;
    metrale_gpu_runtime::cutlass::pack_bf16_weight_to_nvfp4_t(
        w_bf16.0, packed_t.0, scale_t.0, fp8w.n, fp8w.k, stream,
    )?;
    gpu.synchronize(stream)?;
    gpu.free(w_bf16)?;
    derived.insert_pair(
        super::Derivation::CutlassNvfp4FromFp8,
        cache_key,
        (packed_t.0, scale_t.0),
    );
    Ok((packed_t.0, scale_t.0))
}

/// 2026-09-25: CUTLASS NVFP4 projection for an FP8 weight. On first use the
/// weight is dequantized into a transient BF16 buffer, packed into NVFP4 data
/// and scales, and the pair is memoized for later calls.
#[allow(clippy::too_many_arguments)]
pub fn cutlass_nvfp4_proj_from_fp8(
    ctx: &crate::layer::ForwardContext<'_>,
    act: metrale_gpu_runtime::gpu::DevicePtr,
    fp8w: &crate::weight_map::Fp8Weight,
    out: metrale_gpu_runtime::gpu::DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> anyhow::Result<()> {
    let (gpu, derived) = (ctx.gpu, ctx.derived);
    let (packed_t, scale_t) = cutlass_nvfp4_weight_from_fp8_cached(gpu, derived, fp8w, stream)?;
    metrale_gpu_runtime::cutlass::nvfp4_gemm_bf16_act_weight_t(
        act.0, packed_t, scale_t, 1.0, out.0, m, n, k, stream,
    )
}
