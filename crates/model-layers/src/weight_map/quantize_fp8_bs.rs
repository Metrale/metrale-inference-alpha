// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Load-time BF16 to FP8 E4M3 quantization with one FP32 scale per 128x128 block.
//!
//! Owner: model-layers (weight loading).
//! Invariants:
//! - `FP8_BLOCK` equals `FP8_BLOCK` in `kernels/gb10/common/moe_fp8_grouped_gemm.cu`.
//!
//! The FP8 grouped MoE GEMM reads block scales `[N/128, K/128]`, not per-row
//! ones, and cannot tell the two apart by shape; `Fp8Weight::scale_format`
//! records which one a weight carries.

use anyhow::Result;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use super::{DenseWeight, Fp8Weight, WeightQuantFormat};

/// 2026-09-25: Elements per block scale, on both axes.
const FP8_BLOCK: usize = 128;

/// 2026-09-25: Quantize an `[n, k]` BF16 dense weight to block-scaled FP8 E4M3
/// on the GPU, as an `Fp8Weight` tagged `Fp8BlockScaled` with FP32 scales
/// `[ceil(n/128), ceil(k/128)]`.
///
/// It does not synchronize `stream` and does not free the BF16 source; the
/// caller must synchronize before freeing it.
pub fn quantize_to_fp8_blockscaled(
    bf16_weight: &DenseWeight,
    n: usize,
    k: usize,
    gpu: &dyn GpuBackend,
    quantize_kernel: metrale_gpu_runtime::gpu::KernelHandle,
    stream: u64,
) -> Result<Fp8Weight> {
    anyhow::ensure!(
        n > 0 && k > 0,
        "quantize_to_fp8_blockscaled: empty [{n},{k}]"
    );

    let n_blocks = n.div_ceil(FP8_BLOCK);
    let k_blocks = k.div_ceil(FP8_BLOCK);

    let weight_buf = gpu.alloc(n * k)?;
    let scale_buf = gpu.alloc(n_blocks * k_blocks * 4)?;

    KernelLaunch::new(gpu, quantize_kernel)
        .grid([k_blocks as u32, n_blocks as u32, 1])
        .block([256, 1, 1])
        .arg_ptr(bf16_weight.weight)
        .arg_ptr(weight_buf)
        .arg_ptr(scale_buf)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .launch(stream)?;

    Ok(Fp8Weight {
        weight: weight_buf,
        row_scale: scale_buf,
        n: n as u32,
        k: k as u32,
        scale_format: WeightQuantFormat::Fp8BlockScaled,
    })
}
