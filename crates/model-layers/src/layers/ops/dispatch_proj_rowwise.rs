// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Row-wise FP8 projection through cuBLASLt, with the weight's
//! per-row FP8 pair passed through or re-quantized from block-scaled FP8.
//!
//! Owner: model-layers ops.
//! Invariants:
//! - A weight tagged `Fp8PerRow` is used through its own pointers; nothing is
//!   allocated or memoized for it.
//! - The requant panics on a format other than `Fp8PerRow` or
//!   `Fp8BlockScaled`, before it reads the scales.

/// 2026-09-25: `(weight, scale)` as they are when `fp8w` is tagged
/// `Fp8PerRow`, else `None`. Pure, so it is tested without a GPU.
pub(super) fn rowwise_pair_passthrough(fp8w: &crate::weight_map::Fp8Weight) -> Option<(u64, u64)> {
    use crate::weight_map::WeightQuantFormat;
    (fp8w.scale_format == WeightQuantFormat::Fp8PerRow).then_some((fp8w.weight.0, fp8w.row_scale.0))
}

/// 2026-09-25: Re-quantize a block-scaled FP8 weight `[N,K]` to row-wise FP8
/// (E4M3 plus one FP32 scale per row) on the GPU, through a transient BF16
/// copy, and memoize it by the FP8 weight's pointer. Returns
/// `(fp8_weight_ptr, per_row_scale_ptr)`. A weight that is already row-wise
/// returns its own pointers.
fn requant_weight_rowwise_fp8_cached(
    gpu: &dyn metrale_gpu_runtime::gpu::GpuBackend,
    derived: &super::DerivedWeights,
    fp8w: &crate::weight_map::Fp8Weight,
    stream: u64,
) -> anyhow::Result<(u64, u64)> {
    use crate::weight_map::WeightQuantFormat;
    use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

    // 2026-09-25: A mixed-precision compressed-tensors checkpoint (for example
    // unsloth/Qwen3.8-27B-NVFP4) ships its attention and GDN projections as FP8
    // with a per-channel scale, which is already the pair this function
    // produces. Re-quantizing it would round it through BF16 for nothing.
    if let Some(pair) = rowwise_pair_passthrough(fp8w) {
        return Ok(pair);
    }
    // 2026-09-25: The conversion reads `row_scale` as a `[N/128, K/128]` FP32
    // grid, so any other scale format is refused here.
    fp8w.scale_format
        .expect(WeightQuantFormat::Fp8BlockScaled, "rowwise-fp8 requant");
    let cache_key = fp8w.weight.0;
    if let Some(hit) = derived.get_pair(super::Derivation::RowwiseFp8, cache_key) {
        return Ok(hit);
    }
    let (n, k) = (fp8w.n, fp8w.k);
    // 2026-09-25: Block-scaled FP8 to a transient BF16 `[N,K]`.
    let bf16 = gpu.alloc(n as usize * k as usize * 2)?;
    let block = 128u32;
    let sk = k / block;
    let dq = gpu.kernel(
        "dequant_fp8_blockscaled_bf16",
        "dequant_fp8_blockscaled_bf16",
    )?;
    KernelLaunch::new(gpu, dq)
        .grid([div_ceil(k, 64), div_ceil(n, 4), 1])
        .block([64, 4, 1])
        .arg_ptr(fp8w.weight)
        .arg_ptr(fp8w.row_scale)
        .arg_ptr(bf16)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(block)
        .arg_u32(block)
        .arg_u32(sk)
        .arg_u32(1)
        .launch(stream)?;
    // 2026-09-25: BF16 to row-wise FP8 `[N,K]` plus a per-row scale `[N]`.
    let w_fp8 = gpu.alloc(n as usize * k as usize)?;
    let w_scale = gpu.alloc(n as usize * 4)?;
    let qk = gpu.kernel("quant_rowwise_fp8", "quant_rowwise_fp8")?;
    KernelLaunch::new(gpu, qk)
        .grid([n, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(bf16)
        .arg_ptr(w_fp8)
        .arg_ptr(w_scale)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)?;
    gpu.synchronize(stream)?; // 2026-09-25: finish the re-quant before freeing the BF16 copy
    gpu.free(bf16)?;
    derived.insert_pair(
        super::Derivation::RowwiseFp8,
        cache_key,
        (w_fp8.0, w_scale.0),
    );
    Ok((w_fp8.0, w_scale.0))
}

/// 2026-09-25: Route a projection through row-wise FP8 cuBLASLt. The weight
/// pair comes from `requant_weight_rowwise_fp8_cached`; the activation is
/// quantized per token on every call.
///
/// cuBLASLt is handed `ceil16(m)` rows, so `act_fp8_scratch` must hold
/// `ceil16(m) * k` bytes, `act_scale_scratch` `ceil16(m)` f32 and `out`
/// `ceil16(m) * n` BF16 elements. The pad rows of both scratch buffers are
/// zeroed here, and the pad output rows are written.
#[allow(clippy::too_many_arguments)]
pub fn cublas_fp8_rowwise_proj(
    gpu: &dyn metrale_gpu_runtime::gpu::GpuBackend,
    derived: &super::DerivedWeights,
    act_bf16: metrale_gpu_runtime::gpu::DevicePtr,
    act_fp8_scratch: metrale_gpu_runtime::gpu::DevicePtr,
    act_scale_scratch: metrale_gpu_runtime::gpu::DevicePtr,
    fp8w: &crate::weight_map::Fp8Weight,
    out: metrale_gpu_runtime::gpu::DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> anyhow::Result<()> {
    use metrale_gpu_runtime::kernel_args::KernelLaunch;
    let (w_fp8, w_scale) = requant_weight_rowwise_fp8_cached(gpu, derived, fp8w, stream)?;
    // 2026-09-25: Per-token quant of the activation to FP8 `[M,K]` plus scale `[M]`.
    let qk = gpu.kernel("quant_rowwise_fp8", "quant_rowwise_fp8")?;
    KernelLaunch::new(gpu, qk)
        .grid([m, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(act_bf16)
        .arg_ptr(act_fp8_scratch)
        .arg_ptr(act_scale_scratch)
        .arg_u32(m)
        .arg_u32(k)
        .launch(stream)?;
    // 2026-09-25: Pad M to 16, as `cublas_fp8_m_pad` does, and zero both the
    // pad scales and the pad activation rows: a zero scale drops their
    // contribution, and zeroed bytes cannot carry a NaN into the accumulator.
    let m_pad = m.div_ceil(16) * 16;
    if m_pad > m {
        let pad_rows = (m_pad - m) as usize;
        gpu.memset_async(
            act_scale_scratch.offset(m as usize * 4),
            0,
            pad_rows * 4,
            stream,
        )?;
        gpu.memset_async(
            act_fp8_scratch.offset(m as usize * k as usize),
            0,
            pad_rows * k as usize,
            stream,
        )?;
    }
    metrale_gpu_runtime::cublaslt::fp8_gemm_act_weight_t_rowwise(
        act_fp8_scratch.0,
        act_scale_scratch.0,
        w_fp8,
        w_scale,
        out.0,
        m_pad,
        n,
        k,
        stream,
    )
}

#[cfg(test)]
mod rowwise_passthrough_tests {
    use super::{requant_weight_rowwise_fp8_cached, rowwise_pair_passthrough};
    use crate::layers::ops::DerivedWeights;
    use crate::weight_map::{Fp8Weight, WeightQuantFormat};
    use metrale_gpu_runtime::gpu::DevicePtr;
    use metrale_gpu_runtime::gpu::mock::MockGpuBackend;

    fn weight(scale_format: WeightQuantFormat) -> Fp8Weight {
        Fp8Weight {
            weight: DevicePtr(0xBEEF),
            row_scale: DevicePtr(0x5CA1E),
            n: 4096,
            k: 5120,
            scale_format,
        }
    }

    /// 2026-09-25: A weight that already has per-row scales is passed through
    /// with its own pointers.
    #[test]
    fn an_already_rowwise_weight_passes_through_verbatim() {
        let w = weight(WeightQuantFormat::Fp8PerRow);
        assert_eq!(
            rowwise_pair_passthrough(&w),
            Some((w.weight.0, w.row_scale.0)),
            "the checkpoint's own pointers, not a converted copy"
        );
    }

    /// 2026-09-25: Every format other than `Fp8PerRow` returns `None`.
    #[test]
    fn other_formats_still_requantize() {
        for f in [
            WeightQuantFormat::Fp8BlockScaled,
            WeightQuantFormat::Fp8SingleScale,
            WeightQuantFormat::Bf16,
            WeightQuantFormat::Nvfp4,
        ] {
            assert_eq!(
                rowwise_pair_passthrough(&weight(f)),
                None,
                "{f:?} is not a row-wise pair and must not be passed through"
            );
        }
    }

    #[test]
    fn cached_requant_returns_rowwise_checkpoint_pointers_without_gpu_work() {
        let gpu = MockGpuBackend::new();
        let w = weight(WeightQuantFormat::Fp8PerRow);

        assert_eq!(
            requant_weight_rowwise_fp8_cached(&gpu, &DerivedWeights::new(), &w, 0).unwrap(),
            (w.weight.0, w.row_scale.0)
        );
        assert_eq!(gpu.alloc_count(), 0, "passthrough must not allocate a copy");
    }
}
