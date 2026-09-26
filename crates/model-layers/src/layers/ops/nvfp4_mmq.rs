// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Launchers for the NVFP4 W4A4 MMQ GEMM that the dense FFN uses in
//! prefill, built on the vendored llama.cpp MMQ headers
//! (`kernels/gb10/qwen3.6-27b/nvfp4/nvfp4_mmq.cu`, `q4k_vendor/`), plus its
//! weight repack, activation quantizers and scale2 folds.
//!
//! The MMA applies only the per-16 E4M3 scales, so each GEMM output lacks the
//! per-tensor FP32 `weight_scale_2`. `DenseFfn` folds it in: into the SiLU-mul
//! for gate and up ([`nvfp4_silu_mul_scaled`] or [`nvfp4_silu_mul_quant`]),
//! and with [`nvfp4_scale_bf16`] for down. The arm runs when
//! `ModelLevers::ffn_nvfp4_mmq` is on (unless `METRALE_NO_FFN_NVFP4_MMQ` is set)
//! and its kernels resolved.
//!
//! Owner: model-layers ops.
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

/// 2026-09-25: Weights per repacked NVFP4 block: 36 bytes, 32 of E2M1 nibbles and
/// 4 UE4M3 scales.
pub const QK_NVFP4: u32 = 64;
pub const NVFP4_BLOCK_BYTES: usize = 36;
/// 2026-09-25: `block_fp4_mmq`, the activation format: 256 values in 144 bytes,
/// the size of `block_q8_1_mmq` (`q4k_vendor/mmq.cuh`).
const FP4_MMQ_Y_BLOCK_VALS: u32 = 256;
const FP4_MMQ_Y_BLOCK_BYTES: usize = 144;
/// 2026-09-25: Dynamic shared memory of the 128-wide M tile.
pub const NVFP4_MMQ_SMEM: u32 = nvfp4_mmq_smem(128);

/// 2026-09-25: Dynamic shared memory for M tile `mmq_x`, in bytes: `mmq_x` ids,
/// a y tile of `mmq_x * 36` ints padded to 256 ints, and an x tile of `128 * 76`
/// ints. This is `mmq_get_nbytes_shared` in `q4k_vendor/mmq.cuh` with
/// `mmq_y = 128`, `MMQ_MMA_TILE_X_K_FP4 = 76` and 8 warps of 32.
pub const fn nvfp4_mmq_smem(mmq_x: u32) -> u32 {
    let y = mmq_x * 36;
    let y_padded = y.div_ceil(256) * 256;
    4 * (mmq_x + y_padded + 128 * 76)
}
const QUANT_BLOCK_THREADS: u32 = 128;

/// 2026-09-25: Bytes of the repacked form of an `[n, k]` weight; `k` must be a
/// multiple of 64.
pub fn nvfp4_mmq_weight_bytes(n: u32, k: u32) -> usize {
    (n as usize) * (k as usize / QK_NVFP4 as usize) * NVFP4_BLOCK_BYTES
}

/// 2026-09-25: `block_fp4_mmq` activation bytes for `[m, k]`, with `k` rounded up
/// to 256, plus 1 MiB of slack. It never exceeds
/// [`q8_1_scratch_bytes`](super::q8_1_scratch_bytes)`(m, k)`, the size of the
/// `ffn_act_q8` scratch it is written into.
pub fn fp4_act_scratch_bytes(m: u32, k: u32) -> usize {
    let bpc = div_ceil(k, FP4_MMQ_Y_BLOCK_VALS) as usize;
    (m as usize) * bpc * FP4_MMQ_Y_BLOCK_BYTES + (1 << 20)
}

/// 2026-09-25: Repack a checkpoint NVFP4 weight (E2M1 `[n, k/2]`, low nibble =
/// even k, and E4M3 `[n, k/16]` scales) for the MMQ kernel. Each row holds
/// `k/64` blocks as all nibble bytes (32 per block) followed by all scale bytes
/// (4 per block). Codes and scale bytes are copied, not requantized.
pub fn nvfp4_mmq_repack(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    packed: DevicePtr,
    scales: DevicePtr,
    out_blocks: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    let nblocks = (n as u64) * (k as u64 / QK_NVFP4 as u64);
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(nblocks as u32, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(packed)
        .arg_ptr(scales)
        .arg_ptr(out_blocks)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// 2026-09-25: Quantize BF16 activations `[m, k]` to `block_fp4_mmq` in `out_y`:
/// E2M1 values with one UE4M3 scale per 16, `k` padded to 256, one thread per
/// 16-value group.
pub fn nvfp4_mmq_quantize_act(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input_bf16: DevicePtr,
    out_y: DevicePtr,
    m: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    let kpad = div_ceil(k, FP4_MMQ_Y_BLOCK_VALS) * FP4_MMQ_Y_BLOCK_VALS;
    let grid_y = div_ceil(kpad, 16 * QUANT_BLOCK_THREADS);
    KernelLaunch::new(gpu, kernel)
        .grid([m, grid_y, 1])
        .block([QUANT_BLOCK_THREADS, 1, 1])
        .arg_ptr(input_bf16)
        .arg_ptr(out_y)
        .arg_u64(k as u64)
        .arg_u64(k as u64)
        .arg_u64(kpad as u64)
        .arg_u32(m)
        .launch(stream)
}

/// 2026-09-25: `C[m, n] = A[m, k] x W[n, k]` in BF16, without `weight_scale_2`,
/// on the 128-wide M tile. `a_fp4` is `block_fp4_mmq`, `w_nvfp4` the
/// [`nvfp4_mmq_repack`] output. `kernel_wc` is used when `n` is not a multiple
/// of 128.
pub fn nvfp4_mmq_gemm(
    gpu: &dyn GpuBackend,
    kernel_nc: KernelHandle,
    kernel_wc: KernelHandle,
    a_fp4: DevicePtr,
    w_nvfp4: DevicePtr,
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
        .shared_mem(NVFP4_MMQ_SMEM)
        .arg_ptr(w_nvfp4)
        .arg_ptr(a_fp4)
        .arg_ptr(out_bf16)
        .arg_u32(n)
        .arg_u32(m)
        .arg_u32(k)
        .arg_u32(k / QK_NVFP4)
        .arg_u32(m)
        .arg_u32(n)
        .launch(stream)
}

/// 2026-09-25: [`nvfp4_mmq_gemm`] with the M tile `mmq_x` chosen by the caller;
/// the kernels must be that tile's entries. `nvfp4_mmq.cu` instantiates 16, 32,
/// 64 and 128. A 128-wide tile issues MMAs for all 128 columns whatever `m` is,
/// so a small `m` wants a small tile. `grid.y = ceil(m / mmq_x)`, and each M tile
/// streams the whole weight again, so `m` must not exceed `mmq_x`
/// (debug-asserted).
#[allow(clippy::too_many_arguments)]
pub fn nvfp4_mmq_gemm_tiled(
    gpu: &dyn GpuBackend,
    kernel_nc: KernelHandle,
    kernel_wc: KernelHandle,
    mmq_x: u32,
    a_fp4: DevicePtr,
    w_nvfp4: DevicePtr,
    out_bf16: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    debug_assert!(
        matches!(mmq_x, 16 | 32 | 64 | 128),
        "mmq_x must be an instantiated tile"
    );
    debug_assert!(
        m <= mmq_x,
        "grid.y>1 would re-stream the weights per M-tile"
    );
    let kernel = if !n.is_multiple_of(128) {
        kernel_wc
    } else {
        kernel_nc
    };
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 128), div_ceil(m, mmq_x), 1])
        .block([32, 8, 1])
        .shared_mem(nvfp4_mmq_smem(mmq_x))
        .arg_ptr(w_nvfp4)
        .arg_ptr(a_fp4)
        .arg_ptr(out_bf16)
        .arg_u32(n)
        .arg_u32(m)
        .arg_u32(k)
        .arg_u32(k / QK_NVFP4)
        .arg_u32(m)
        .arg_u32(n)
        .launch(stream)
}
/// 2026-09-25: `silu(gate * gate_scale) * (up * up_scale)` quantized straight to
/// `block_fp4_mmq` in `out_y` for the down GEMM, without writing a BF16
/// intermediate. `k` is the intermediate width, padded to 256. The kernel
/// applies no clamp.
#[allow(clippy::too_many_arguments)]
pub fn nvfp4_silu_mul_quant(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate: DevicePtr,
    up: DevicePtr,
    out_y: DevicePtr,
    gate_scale: f32,
    up_scale: f32,
    m: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    let kpad = div_ceil(k, FP4_MMQ_Y_BLOCK_VALS) * FP4_MMQ_Y_BLOCK_VALS;
    let grid_y = div_ceil(kpad, 16 * QUANT_BLOCK_THREADS);
    KernelLaunch::new(gpu, kernel)
        .grid([m, grid_y, 1])
        .block([QUANT_BLOCK_THREADS, 1, 1])
        .arg_ptr(gate)
        .arg_ptr(up)
        .arg_ptr(out_y)
        .arg_f32(gate_scale)
        .arg_f32(up_scale)
        .arg_u64(k as u64)
        .arg_u64(kpad as u64)
        .arg_u32(m)
        .launch(stream)
}

/// 2026-09-25: `data[i] *= scale` in place over `total` BF16 values; the caller
/// passes the down projection's `weight_scale_2`.
pub fn nvfp4_scale_bf16(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    data: DevicePtr,
    scale: f32,
    total: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(total, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(data)
        .arg_f32(scale)
        .arg_u32(total)
        .launch(stream)
}

/// 2026-09-25: `out[i] = silu(gate[i] * gate_scale) * (up[i] * up_scale)` over
/// `total` BF16 values. The kernel applies no clamp.
#[allow(clippy::too_many_arguments)]
pub fn nvfp4_silu_mul_scaled(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate: DevicePtr,
    up: DevicePtr,
    out: DevicePtr,
    gate_scale: f32,
    up_scale: f32,
    total: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(total, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(gate)
        .arg_ptr(up)
        .arg_ptr(out)
        .arg_f32(gate_scale)
        .arg_f32(up_scale)
        .arg_u32(total)
        .launch(stream)
}
