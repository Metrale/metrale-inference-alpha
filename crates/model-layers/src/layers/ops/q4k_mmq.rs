// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Launchers for the Q4_K MMQ GEMM that the dense FFN uses in
//! prefill under `METRALE_FFN_MMQ`, built on the vendored llama.cpp MMQ headers
//! (`kernels/gb10/qwen3.6-27b/nvfp4/q4k_mmq.cu`, `q4k_quantize.cu`), plus the
//! weight conversion and the q8_1 activation quantizer.
//!
//! `DenseFfn::ensure_q4k_weight` converts each NVFP4 weight once (dequantize
//! to BF16, then quantize to `block_q4_K`) and caches the result. Each prefill
//! quantizes its activations to q8_1 and runs the GEMM with a BF16 store.
//!
//! Owner: model-layers ops.
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

/// 2026-09-25: Weights per `block_q4_K`.
pub const QK_K: u32 = 256;
/// 2026-09-25: `sizeof(block_q4_K)`.
pub const Q4K_BLOCK_BYTES: usize = 144;
/// 2026-09-25: Dynamic shared memory of the Q4_K MMQ kernel at `mmq_x = mmq_y =
/// 128`: `mmq_get_nbytes_shared` in `q4k_vendor/mmq.cuh` with an x tile of 76
/// ints per row. Above 48 KiB, so the registry raises the kernel's dynamic
/// shared memory limit (`registry.rs`).
pub const Q4K_MMQ_SMEM: u32 = 57856;
const CUDA_QUANTIZE_BLOCK_SIZE_MMQ: u32 = 128;

/// 2026-09-25: Bytes of the Q4_K form of an `[nrows, n_per_row]` weight;
/// `n_per_row` must be a multiple of 256.
pub fn q4k_weight_bytes(nrows: u32, n_per_row: u32) -> usize {
    (nrows as usize) * (n_per_row as usize / QK_K as usize) * Q4K_BLOCK_BYTES
}

/// 2026-09-25: q8_1 activation scratch bytes for `[m, k]`: 4 bytes per value with
/// `k` rounded up to 256, plus 1 MiB. `ffn_act_q8` in
/// `gpu-runtime/src/buffers/sizes.rs` uses the same expression with `k` the
/// larger of the hidden and intermediate sizes.
pub fn q8_1_scratch_bytes(m: u32, k: u32) -> usize {
    let kpad = div_ceil(k, QK_K) * QK_K;
    (m as usize) * (kpad as usize) * 4 + (1 << 20)
}

/// 2026-09-25: Dequantize an NVFP4 weight `[n, k]` (packed E2M1, E4M3 group
/// scales and the per-tensor `scale2`) to BF16 `[n, k]`, one block per row.
pub fn dequant_nvfp4_to_bf16(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    packed: DevicePtr,
    scales: DevicePtr,
    out_bf16: DevicePtr,
    scale2: f32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([n, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(packed)
        .arg_ptr(scales)
        .arg_ptr(out_bf16)
        .arg_f32(scale2)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// 2026-09-25: Quantize BF16 weights `[nrows, n_per_row]` to `block_q4_K`, one
/// thread per 256-value block.
pub fn quantize_weight_q4k(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input_bf16: DevicePtr,
    out_q4k: DevicePtr,
    nrows: u32,
    n_per_row: u32,
    stream: u64,
) -> Result<()> {
    let total_sb = (nrows as u64) * (n_per_row as u64 / QK_K as u64);
    let grid_x = div_ceil(total_sb as u32, 128);
    KernelLaunch::new(gpu, kernel)
        .grid([grid_x, 1, 1])
        .block([128, 1, 1])
        .arg_ptr(input_bf16)
        .arg_ptr(out_q4k)
        .arg_u32(nrows)
        .arg_u32(n_per_row)
        .launch(stream)
}

/// 2026-09-25: Quantize BF16 activations `[m, k]` to q8_1 in `out_q8`, with `k`
/// padded to 256. The scale layout (DS4, D4 or D2S6) is the kernel's.
pub fn quantize_act_q8_1(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input_bf16: DevicePtr,
    out_q8: DevicePtr,
    m: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    let kpad = div_ceil(k, QK_K) * QK_K;
    let grid_y = div_ceil(kpad, 4 * CUDA_QUANTIZE_BLOCK_SIZE_MMQ);
    KernelLaunch::new(gpu, kernel)
        .grid([m, grid_y, 1])
        .block([CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 1, 1])
        .arg_ptr(input_bf16)
        .arg_ptr(out_q8)
        .arg_u64(k as u64)
        .arg_u64(k as u64)
        .arg_u64(kpad as u64)
        .arg_u32(m)
        .launch(stream)
}

/// 2026-09-25: `C[m, n] = A[m, k] x W[n, k]` in BF16, with `a_q8` in q8_1 and
/// `w_q4k` in `block_q4_K`, on a 128 x 128 tile. `kernel_wc` is used when `n`
/// is not a multiple of 128.
pub fn q4k_mmq_gemm(
    gpu: &dyn GpuBackend,
    kernel_nc: KernelHandle,
    kernel_wc: KernelHandle,
    a_q8: DevicePtr,
    w_q4k: DevicePtr,
    out_bf16: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    let kernel = if !n.is_multiple_of(128) {
        kernel_wc
    } else {
        kernel_nc
    };
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 128), div_ceil(m, 128), 1])
        .block([32, 8, 1])
        .shared_mem(Q4K_MMQ_SMEM)
        .arg_ptr(w_q4k)
        .arg_ptr(a_q8)
        .arg_ptr(out_bf16)
        .arg_u32(n)
        .arg_u32(m)
        .arg_u32(k)
        .arg_u32(k / QK_K)
        .arg_u32(m)
        .arg_u32(n)
        .launch(stream)
}
