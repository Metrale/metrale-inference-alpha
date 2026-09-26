// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Launchers for the N-column-blocked W8A16 batched GEMV
//! (`kernels/hopper/common/w8a16_gemv_ncol.cu`, module `w8a16_gemv_ncol`).
//!
//! It computes what `w8a16_gemv_batch16` computes, with the same per-row
//! reduction order, but one thread owns `N_COLS` adjacent output columns, so
//! the activation loads and BF16->FP32 converts are shared by `N_COLS` weight
//! bytes. The `native_fp8_attn_decode_batch_microtest` example compares the
//! outputs bit for bit.
//!
//! The four entry points keep the `ContiguousBatchGemv` / `StridedBatchGemv`
//! signatures of the `w8a16_gemv_batch{4,16}` launchers, so a decode tier picks
//! a rung by swapping a function pointer and a handle.
//!
//! Owner: model-layers ops.
//! Invariants: none beyond the types.

use anyhow::{Result, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

/// 2026-09-25: Output columns one thread owns. The kernel is instantiated for
/// both; `METRALE_ATTN_NCOL_WIDTH` picks one (`attn_ncol_gemv::NcolWidth`).
const COLS_PER_THREAD_2: u32 = 2;
const COLS_PER_THREAD_4: u32 = 4;

/// 2026-09-25: Output groups per block: `w8a16_gemv_ncol.cu`'s
/// `N_GROUPS_PER_BLOCK`, 64 threads each, as in `w8a16_gemv_batch4.cu`. A block
/// covers `GROUPS_PER_BLOCK * n_cols` columns, the grid divisor below.
const GROUPS_PER_BLOCK: u32 = 4;

/// 2026-09-25: `N_COLS=2`, contiguous A `[M, K]` and C `[M, N]`.
///
/// Kernel: `w8a16_gemv_batch16_ncol2` (module `w8a16_gemv_ncol`).
/// Grid: (ceil(N/8), 1, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn w8a16_gemv_batch16_ncol2(
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
    ncol_launch(
        gpu,
        kernel,
        input,
        weight,
        block_scale,
        output,
        m,
        n,
        k,
        k,
        n,
        COLS_PER_THREAD_2,
        false,
        stream,
    )
}

/// 2026-09-25: `N_COLS=4`, contiguous A and C. Grid: (ceil(N/16), 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn w8a16_gemv_batch16_ncol4(
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
    ncol_launch(
        gpu,
        kernel,
        input,
        weight,
        block_scale,
        output,
        m,
        n,
        k,
        k,
        n,
        COLS_PER_THREAD_4,
        false,
        stream,
    )
}

/// 2026-09-25: `N_COLS=2`, explicit A and C row pitches in elements, for the
/// multi-seq `[n, per_seq_qkv]` decode buffer, as `w8a16_gemv_batch16_strided`.
///
/// Kernel: `w8a16_gemv_batch16_ncol2_strided` (module `w8a16_gemv_ncol`).
#[allow(clippy::too_many_arguments)]
pub fn w8a16_gemv_batch16_ncol2_strided(
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
    ncol_launch(
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
        COLS_PER_THREAD_2,
        true,
        stream,
    )
}

/// 2026-09-25: `N_COLS=4`, explicit row pitches.
#[allow(clippy::too_many_arguments)]
pub fn w8a16_gemv_batch16_ncol4_strided(
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
    ncol_launch(
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
        COLS_PER_THREAD_4,
        true,
        stream,
    )
}

/// 2026-09-25: Shared launch body. `strided` selects the argument tail: the
/// contiguous entry points pass `a_row_stride = K` / `c_row_stride = N` inside
/// the kernel and take neither.
///
/// Returns an error for `m` outside `1..=16`, `k` not a multiple of 16, pitches
/// shorter than `k` / `n`, or an `a_row_stride` that is not a multiple of 8.
#[allow(clippy::too_many_arguments)]
fn ncol_launch(
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
    n_cols: u32,
    strided: bool,
    stream: u64,
) -> Result<()> {
    // 2026-09-25: The kernel is `w8a16_gemv_ncol_impl<16, N_COLS>`: above 16
    // rows it would leave rows 16.. unwritten rather than fail.
    ensure!(
        (1..=16).contains(&m),
        "w8a16_gemv_batch16_ncol{n_cols}: m={m} outside 1..=16 (kernel MAX_M)"
    );
    ensure!(
        k.is_multiple_of(16),
        "w8a16_gemv_batch16_ncol{n_cols}: K={k} not a multiple of 16 (uint4 loads)"
    );
    ensure!(
        a_row_stride >= k && c_row_stride >= n,
        "w8a16_gemv_batch16_ncol{n_cols}: row pitches (a={a_row_stride}, c={c_row_stride}) \
         must cover the used extents (k={k}, n={n})"
    );
    ensure!(
        a_row_stride.is_multiple_of(8),
        "w8a16_gemv_batch16_ncol{n_cols}: a_row_stride={a_row_stride} must keep rows \
         16B-aligned (uint4 activation loads)"
    );
    let mut launch = KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, GROUPS_PER_BLOCK * n_cols), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight)
        .arg_ptr(block_scale)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k);
    if strided {
        launch = launch.arg_u32(a_row_stride).arg_u32(c_row_stride);
    }
    launch.launch(stream)
}
