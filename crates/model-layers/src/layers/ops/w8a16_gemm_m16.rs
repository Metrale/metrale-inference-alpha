// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Launchers for the tensor-core W8A16 decode GEMM with a 16-row M
//! tile (`kernels/hopper/common/w8a16_gemm_m16.cu`, module `w8a16_gemm_m16`;
//! entries `w8a16_gemm_m16`, `w8a16_gemm_m16_n64`, `w8a16_gemm_m16_strided`).
//!
//! Where `w8a16_gemv_batch16` runs 16 scalar FFMA per weight byte, this kernel
//! runs `mma.sync.m16n8k16`, which reduces 16 K-products in the tensor core's
//! own order. It is therefore not bit-identical to the scalar GEMVs. Its
//! contract is `within_m16_tc_budget` (`dense_ffn_m16_tc_oracle.rs`): within
//! `M16_TC_MAX_ULP` BF16 ULP of the reference, or an absolute error under
//! `m16_tc_acc_floor` for the reduction depth and the row's RMS; the
//! `native_fp8_ffn_m16_tc_microtest` example applies it. The callers gate it
//! on the target-declared `ffn_m16_tc` (dense FFN) and `attn_m16_tc`
//! (attention) levers.
//!
//! The 128-K block scale is folded once per block onto an FP32 outer
//! accumulator, never per element and never into BF16.
//!
//! Grid: (ceil(N/N_TILE), 1, 1)  Block: (128, 1, 1).
//!
//! Owner: model-layers ops.
//! Invariants: none beyond the types.

use anyhow::{Result, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

/// 2026-09-25: N columns one CTA owns on the default instantiation. The launch
/// grid and `m16_tc_kernel` (`dense_ffn_m16_tc.rs`) both use it; the kernel's
/// `M16_N_TILE` must equal it.
pub const W8A16_GEMM_M16_N_TILE: u32 = 32;

/// 2026-09-25: The wide instantiation's N tile (`w8a16_gemm_m16_n64`, kernel
/// `M16_N_TILE_WIDE`), selected for the dense FFN by
/// `METRALE_FFN_M16_TC_NTILE=64`. It halves the CTA count for a given N.
pub const W8A16_GEMM_M16_N_TILE_WIDE: u32 = 64;

/// 2026-09-25: The shared signature of [`w8a16_gemm_m16`] and
/// [`w8a16_gemm_m16_n64`], so `m16_tc_kernel` (`dense_ffn_m16_tc.rs`) can
/// return either as one function pointer.
pub type ContiguousM16Gemm = fn(
    &dyn GpuBackend,
    KernelHandle,
    DevicePtr,
    DevicePtr,
    DevicePtr,
    DevicePtr,
    u32,
    u32,
    u32,
    u64,
) -> Result<()>;

/// 2026-09-25: Contiguous `input` `[m, k]` BF16 and `output` `[m, n]` BF16;
/// `weight` / `block_scale` are the raw `w8a16_gemv` pointers (`[N, K]` FP8
/// E4M3 and `[N/128, K/128]` FP32).
///
/// Returns an error for `m` outside `1..=16` or `k` not a multiple of 128: the
/// kernel's M tile is the MMA's 16 rows, and rows past it would be left
/// unwritten rather than fail. Callers with 17..=32 rows run it twice on
/// contiguous row halves, as `dense_ffn_m16_tc.rs` does.
#[allow(clippy::too_many_arguments)]
pub fn w8a16_gemm_m16(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: DevicePtr,
    block_scale: DevicePtr,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    ensure!(
        (1..=16).contains(&m),
        "w8a16_gemm_m16: m={m} outside 1..=16 (kernel M tile)"
    );
    ensure!(
        k.is_multiple_of(128),
        "w8a16_gemm_m16: K={k} not a multiple of 128 (block-scale granularity)"
    );
    launch_contiguous(
        gpu,
        kernel,
        W8A16_GEMM_M16_N_TILE,
        input,
        weight,
        block_scale,
        output,
        m,
        n,
        k,
        stream,
    )
}

/// 2026-09-25: The `N_TILE=64` instantiation of [`w8a16_gemm_m16`]: the same
/// arguments and guards, `ceil(N/64)` CTAs instead of `ceil(N/32)`.
#[allow(clippy::too_many_arguments)]
pub fn w8a16_gemm_m16_n64(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: DevicePtr,
    block_scale: DevicePtr,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    ensure!(
        (1..=16).contains(&m),
        "w8a16_gemm_m16_n64: m={m} outside 1..=16 (kernel M tile)"
    );
    ensure!(
        k.is_multiple_of(128),
        "w8a16_gemm_m16_n64: K={k} not a multiple of 128 (block-scale granularity)"
    );
    launch_contiguous(
        gpu,
        kernel,
        W8A16_GEMM_M16_N_TILE_WIDE,
        input,
        weight,
        block_scale,
        output,
        m,
        n,
        k,
        stream,
    )
}

/// 2026-09-25: The launch both contiguous instantiations share; only the CTA
/// width differs.
#[allow(clippy::too_many_arguments)]
fn launch_contiguous(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    n_tile: u32,
    input: DevicePtr,
    weight: DevicePtr,
    block_scale: DevicePtr,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, n_tile), 1, 1])
        .block([128, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight)
        .arg_ptr(block_scale)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// 2026-09-25: Strided [`w8a16_gemm_m16`]: `a_row_stride` / `c_row_stride` are
/// the A and C row pitches in elements, for callers whose rows are not
/// contiguous. The multi-seq decode QKV buffer (`qkv_fp8_batch.rs`) is
/// `[n, per_seq_qkv]` with Q at offset 0, K after Q and V after K in every row,
/// so one launch per projection writes all `m` rows into their slots. Same
/// argument order as `w8a16_gemv_batch16_strided`.
///
/// `a_row_stride` must keep each activation row 16-byte aligned (a multiple of
/// 8 BF16): the kernel stages A in 16-byte `cp.async` chunks.
#[allow(clippy::too_many_arguments)]
pub fn w8a16_gemm_m16_strided(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: DevicePtr,
    block_scale: DevicePtr,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    a_row_stride: u32,
    c_row_stride: u32,
    stream: u64,
) -> Result<()> {
    ensure!(
        (1..=16).contains(&m),
        "w8a16_gemm_m16_strided: m={m} outside 1..=16 (kernel M tile)"
    );
    ensure!(
        k.is_multiple_of(128),
        "w8a16_gemm_m16_strided: K={k} not a multiple of 128 (block-scale granularity)"
    );
    ensure!(
        a_row_stride >= k && c_row_stride >= n,
        "w8a16_gemm_m16_strided: row pitches (a={a_row_stride}, c={c_row_stride}) \
         must cover the used extents (k={k}, n={n})"
    );
    ensure!(
        a_row_stride.is_multiple_of(8),
        "w8a16_gemm_m16_strided: a_row_stride={a_row_stride} must keep rows \
         16B-aligned (cp.async stages A in 16-byte chunks)"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, W8A16_GEMM_M16_N_TILE), 1, 1])
        .block([128, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight)
        .arg_ptr(block_scale)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(a_row_stride)
        .arg_u32(c_row_stride)
        .launch(stream)
}
