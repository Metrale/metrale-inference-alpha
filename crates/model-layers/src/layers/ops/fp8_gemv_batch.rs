// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Launchers for the batched FP8-weight GEMVs: the row-scaled
//! `dense_gemv_fp8w_batch2` and `fp8_gemv_rowscale_batch{8,16}_rt2`, and the
//! block-scaled `w8a16_gemv_batch*` family.
//!
//! Owner: model-layers ops.
//! Invariants:
//! - The `fp8_gemv_rowscale_batch*_rt2` and `w8a16_gemv_batch{4,16}*`
//!   launchers return an error, without launching, when `m` is outside
//!   `1..=MAX_M` of their kernel.

use anyhow::{Result, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::Fp8DenseWeight;

/// 2026-09-25: Register-tiled row-scaled FP8 GEMV for up to 8 rows, two
/// outputs per 64-lane group (kernel `fp8_gemv_rowscale_batch8_rt2`, module
/// `fp8_gemv_rt`). `input` `[M, K]` BF16, `output` `[M, N]` BF16; the kernel
/// applies `row_scale[n]` at write-out. The DFlash drafter's FP8 path calls it
/// (`dflash_head/forward_block.rs`). Refuses `m` outside `1..=8` and a `k`
/// that is not a multiple of 16.
#[allow(clippy::too_many_arguments)]
pub fn fp8_gemv_rowscale_batch8_rt2(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &Fp8DenseWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    ensure!(
        (1..=8).contains(&m),
        "fp8_gemv_rowscale_batch8_rt2: m={m} outside 1..=8 (kernel MAX_M)"
    );
    ensure!(
        k.is_multiple_of(16),
        "fp8_gemv_rowscale_batch8_rt2: K={k} not a multiple of 16"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 8), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.row_scale)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// 2026-09-25: The 16-row instantiation of the same kernel template
/// (`fp8_gemv_rowscale_batch16_rt2`, module `fp8_gemv_rt`), with the launch
/// geometry of [`fp8_gemv_rowscale_batch8_rt2`]. Refuses `m` outside
/// `1..=16` and a `k` that is not a multiple of 16.
#[allow(clippy::too_many_arguments)]
pub fn fp8_gemv_rowscale_batch16_rt2(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &Fp8DenseWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    ensure!(
        (1..=16).contains(&m),
        "fp8_gemv_rowscale_batch16_rt2: m={m} outside 1..=16 (kernel MAX_M)"
    );
    ensure!(
        k.is_multiple_of(16),
        "fp8_gemv_rowscale_batch16_rt2: K={k} not a multiple of 16"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 8), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.row_scale)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// 2026-09-25: FP8-weight GEMV for two rows in one pass over the weight:
/// `input` `[2, K]` BF16, `output` `[2, N]` BF16, scaled by
/// `weight.row_scale`. Not bit-identical to two `dense_gemv_fp8w` launches:
/// the kernel adds four products to the accumulator at once, where
/// `dense_gemv_fp8w` adds each product separately.
pub fn dense_gemv_fp8w_batch2(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &Fp8DenseWeight,
    output: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.row_scale)
        .arg_ptr(output)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// 2026-09-25: The signature of the contiguous batched GEMV launchers, such as
/// [`w8a16_gemv_batch4`] and [`w8a16_gemv_batch16`], so a caller that picks a
/// tier by row count can hold the launcher and its handle as one pair.
pub type ContiguousBatchGemv = fn(
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

/// 2026-09-25: Block-scaled FP8 GEMV for up to 4 rows
/// (`w8a16_gemv_batchm_impl<4>`); one pass over the weight serves every row.
/// `input` `[M, K]` BF16, `output` `[M, N]` BF16; `weight` `[N, K]` FP8 E4M3
/// and `block_scale` `[N/128, K/128]` FP32, as for `w8a16_gemv`.
///
/// Per row, the K order and the separate FP32 additions match `w8a16_gemv`
/// (gb10 kernels); `w8a16_batch_bitparity_microtest` compares the two byte
/// for byte.
///
/// Refuses `m > 4`: the kernel computes only rows below its MAX_M and never
/// writes the rest. [`w8a16_gemv_batch16`] takes the same arguments for
/// 5..=16 rows.
#[allow(clippy::too_many_arguments)]
pub fn w8a16_gemv_batch4(
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
        (1..=4).contains(&m),
        "w8a16_gemv_batch4: m={m} outside 1..=4 (kernel MAX_M; use w8a16_gemv_batch16)"
    );
    contiguous_batch_launch(
        gpu,
        kernel,
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

/// 2026-09-25: The 16-row instantiation of the same template (kernel
/// `w8a16_gemv_batch16`, module `w8a16_gemv_batch4`), with the launch
/// geometry and per-row numerics of [`w8a16_gemv_batch4`].
#[allow(clippy::too_many_arguments)]
pub fn w8a16_gemv_batch16(
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
        "w8a16_gemv_batch16: m={m} outside 1..=16 (kernel MAX_M)"
    );
    contiguous_batch_launch(
        gpu,
        kernel,
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

/// 2026-09-25: Launch body of the two contiguous entry points; they differ
/// only in the MAX_M bound they check.
#[allow(clippy::too_many_arguments)]
fn contiguous_batch_launch(
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
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight)
        .arg_ptr(block_scale)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// 2026-09-25: Launches a batch-2 block-scaled FP8 GEMV. No kernel under
/// `kernels/` defines a `w8a16_gemv_batch2` entry point.
#[allow(clippy::too_many_arguments)]
pub fn w8a16_gemv_batch2(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: DevicePtr,
    block_scale: DevicePtr,
    output: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight)
        .arg_ptr(block_scale)
        .arg_ptr(output)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// 2026-09-25: Strided variant of [`w8a16_gemv_batch4`] (up to 4 rows). It
/// reads and writes rows at caller-given pitches, so one projection of the
/// multi-sequence decode QKV buffer `[n, per_seq_qkv]` is written for every
/// row in one launch.
///
/// Layout: `input` `[M, a_row_stride]` BF16, of which the first `k` elements
/// of each row are read; `weight`/`block_scale` as for `w8a16_gemv`
/// (`[N, K]` FP8 E4M3 and `[N/128, K/128]` FP32); `output` `[M, c_row_stride]`
/// BF16, of which the first `n` elements of each row are written. Strides are
/// in elements. The launcher refuses pitches that do not cover `k` and `n`,
/// and an `a_row_stride` that is not a multiple of 8 (the kernel loads
/// activations as 16-byte `uint4`).
///
/// Per-row numerics as [`w8a16_gemv_batch4`]; `native_fp8_qkv_batch_microtest`
/// compares it with `w8a16_gemv`.
///
/// Kernel: `w8a16_gemv_batch4_strided` (module `w8a16_gemv_batch4`).
#[allow(clippy::too_many_arguments)]
pub fn w8a16_gemv_batch4_strided(
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
        (1..=4).contains(&m),
        "w8a16_gemv_batch4_strided: m={m} outside 1..=4 (kernel MAX_M)"
    );
    strided_batch_launch(
        gpu,
        kernel,
        input,
        weight,
        block_scale,
        output,
        m,
        n,
        k,
        a_row_stride,
        c_row_stride,
        stream,
    )
}

/// 2026-09-25: The 16-row variant of [`w8a16_gemv_batch4_strided`] (kernel
/// `w8a16_gemv_batch16_strided`, module `w8a16_gemv_batch4`), with the same
/// launch geometry.
#[allow(clippy::too_many_arguments)]
pub fn w8a16_gemv_batch16_strided(
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
        "w8a16_gemv_batch16_strided: m={m} outside 1..=16 (kernel MAX_M)"
    );
    strided_batch_launch(
        gpu,
        kernel,
        input,
        weight,
        block_scale,
        output,
        m,
        n,
        k,
        a_row_stride,
        c_row_stride,
        stream,
    )
}

/// 2026-09-25: Launch body of the two strided entry points; they differ only
/// in the MAX_M bound they check.
#[allow(clippy::too_many_arguments)]
fn strided_batch_launch(
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
        a_row_stride >= k && c_row_stride >= n,
        "w8a16_gemv batch strided: row pitches (a={a_row_stride}, c={c_row_stride}) \
         must cover the used extents (k={k}, n={n})"
    );
    ensure!(
        a_row_stride.is_multiple_of(8),
        "w8a16_gemv batch strided: a_row_stride={a_row_stride} must keep rows \
         16B-aligned (uint4 activation loads)"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
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
