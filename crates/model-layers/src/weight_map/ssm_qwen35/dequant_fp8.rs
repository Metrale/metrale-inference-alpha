// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Block-scaled FP8 dequantization of a slice addressed by device pointer, for fused expert tensors.
//!
//! Owner: model-layers (weight loading).
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

/// 2026-09-25: Dequantize a block-scaled FP8 weight `[n, k]` with scales
/// `[sn, sk]`, both given as device pointers, into a new BF16 buffer, using the
/// same kernel as `dequant_fp8_blockscaled_to_bf16`, and synchronize. For a
/// slice of a fused expert tensor, which has no key of its own. The caller
/// frees the result.
#[allow(clippy::too_many_arguments)]
pub(super) fn dequant_fp8_block_slice_bf16(
    gpu: &dyn GpuBackend,
    weight_ptr: DevicePtr,
    scale_ptr: DevicePtr,
    n: usize,
    k: usize,
    sn: usize,
    sk: usize,
    scale_is_f32: bool,
) -> Result<DevicePtr> {
    use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};
    let out = gpu.alloc(n * k * 2)?;
    let block_n = (n / sn) as u32;
    let block_k = (k / sk) as u32;
    let stream = gpu.default_stream();
    let kernel = gpu.kernel(
        "dequant_fp8_blockscaled_bf16",
        "dequant_fp8_blockscaled_bf16",
    )?;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(k as u32, 64), div_ceil(n as u32, 4), 1])
        .block([64, 4, 1])
        .arg_ptr(weight_ptr)
        .arg_ptr(scale_ptr)
        .arg_ptr(out)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .arg_u32(block_n)
        .arg_u32(block_k)
        .arg_u32(sk as u32)
        .arg_u32(scale_is_f32 as u32)
        .launch(stream)?;
    gpu.synchronize(stream)?;
    Ok(out)
}
