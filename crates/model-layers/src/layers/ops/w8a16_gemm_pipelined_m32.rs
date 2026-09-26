// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `w8a16_gemm_pipelined_m32`, the 32-row M-tile twin of
//! [`super::w8a16_gemm_pipelined`], and the rule for when the twin runs.
//!
//! At 1..=32 rows the 128-row tile of `w8a16_gemm_pipelined` is mostly
//! padding. The twin is a 32x32 tile with 128-K steps and caller-supplied A/C
//! row pitches (kernel header: `kernels/gb10/common/w8a16_gemm_pipelined_m32.cu`),
//! so one kernel serves the contiguous GDN in_proj_qkvz/out_proj arms
//! ([`w8a16_gemm_pipelined_by_m`]), the attention o_proj and the strided
//! multi-seq Q/K/V tier.
//!
//! Numerics: it uses the same sub-MMA windows and fold points as
//! `w8a16_gemm_pipelined`, and the `native_fp8_gdn_proj_m32_microtest` example
//! checks its output byte for byte against it. Against the scalar GEMV family
//! it reassociates the K reduction, as the `*_m16` tiers do.
//!
//! The kernel has its own module (`w8a16_gemm_pipelined_m32`). Callers resolve
//! it with `try_target_kernel`, so a target without the module gets a zero
//! handle and its callers keep to their other kernels.
//!
//! Owner: model-layers ops.
//! Invariants: none beyond the types.

use anyhow::{Result, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

/// 2026-09-25: Rows one M tile covers (`PM32_M_TILE`). Above it the kernel
/// still runs, with `grid.y = ceil(M / 32)` and one weight pass per tile;
/// [`w8a16_pipelined_prefers_m32`] sends contiguous callers to the 128-tile
/// kernel past this bound.
pub const W8A16_M32_TILE_ROWS: u32 = 32;

/// 2026-09-25: N tile (`PM32_N_TILE`), the grid.x granularity.
const W8A16_M32_TILE_COLS: u32 = 32;

/// 2026-09-25: K granularity: one FP8 scale block per K-step (`PM32_K_STEP ==
/// PM32_FP8_BLOCK`), which is why `K % 128 == 0` is required rather than
/// padded.
pub const W8A16_M32_K_BLOCK: u32 = 128;

/// 2026-09-25: The selection rule between the 128-tile `w8a16_gemm_pipelined`
/// and its 32-tile twin at a contiguous call site: the twin takes `1..=32`
/// rows when its handle is nonzero and `K` is a positive multiple of 128.
/// Above 32 rows the twin would need `grid.y > 1`, streaming the weight once
/// per 32-row tile. [`w8a16_gemm_pipelined_by_m`] applies it.
pub fn w8a16_pipelined_prefers_m32(m: u32, k: u32, m32_kernel: KernelHandle) -> bool {
    (1..=W8A16_M32_TILE_ROWS).contains(&m)
        && k >= W8A16_M32_K_BLOCK
        && k.is_multiple_of(W8A16_M32_K_BLOCK)
        && m32_kernel.0 != 0
}

/// 2026-09-25: Strided launch of `w8a16_gemm_pipelined_m32`.
///
/// `input` `[M, a_row_stride]` BF16 (first `k` of each row read), `weight`
/// `[N, K]` FP8 E4M3 with `block_scale` `[N/128, K/128]` FP32, `output`
/// `[M, c_row_stride]` BF16 (first `n` of each row written). Strides in
/// elements. The A rows feed 16-byte `cp.async` chunks, so `a_row_stride`
/// must keep them 16 B-aligned (multiple of 8).
///
/// Returns an error for `m == 0`, for `k` not a positive multiple of 128, for
/// pitches shorter than `k` / `n`, and for an `a_row_stride` that is not a
/// multiple of 8. Same argument order as `w8a16_gemv_batch{4,16}_strided`, so
/// the multi-seq Q/K/V tier holds it as one more `StridedBatchGemv` arm.
///
/// Grid: (ceil(N/32), ceil(M/32), 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn w8a16_gemm_pipelined_m32_strided(
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
        m >= 1,
        "w8a16_gemm_pipelined_m32: m=0 has no rows to compute"
    );
    ensure!(
        k >= W8A16_M32_K_BLOCK && k.is_multiple_of(W8A16_M32_K_BLOCK),
        "w8a16_gemm_pipelined_m32: K={k} must be a positive multiple of \
         {W8A16_M32_K_BLOCK} (one FP8 scale block per K-step)"
    );
    ensure!(
        a_row_stride >= k && c_row_stride >= n,
        "w8a16_gemm_pipelined_m32: row pitches (a={a_row_stride}, c={c_row_stride}) \
         must cover the used extents (k={k}, n={n})"
    );
    ensure!(
        a_row_stride.is_multiple_of(8),
        "w8a16_gemm_pipelined_m32: a_row_stride={a_row_stride} must keep rows \
         16B-aligned (cp.async A tile)"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([
            div_ceil(n, W8A16_M32_TILE_COLS),
            div_ceil(m, W8A16_M32_TILE_ROWS),
            1,
        ])
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

/// 2026-09-25: Contiguous `[M, K]` -> `[M, N]` launch of the same kernel: the
/// pitches are `k` and `n`. Same signature as [`super::w8a16_gemm_pipelined`].
#[allow(clippy::too_many_arguments)]
pub fn w8a16_gemm_pipelined_m32(
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
    w8a16_gemm_pipelined_m32_strided(
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
        stream,
    )
}

/// 2026-09-25: Contiguous block-scaled W8A16 GEMM, tile chosen by
/// [`w8a16_pipelined_prefers_m32`]: the 32-tile twin on `m32_kernel`, else the
/// 128-tile `w8a16_gemm_pipelined` on `full_kernel`. The GDN `in_proj_qkvz` /
/// `out_proj` batched-verify arms (`trait_decode_batched.rs`) call it.
#[allow(clippy::too_many_arguments)]
pub fn w8a16_gemm_pipelined_by_m(
    gpu: &dyn GpuBackend,
    full_kernel: KernelHandle,
    m32_kernel: KernelHandle,
    input: DevicePtr,
    weight: DevicePtr,
    block_scale: DevicePtr,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    if w8a16_pipelined_prefers_m32(m, k, m32_kernel) {
        w8a16_gemm_pipelined_m32(
            gpu,
            m32_kernel,
            input,
            weight,
            block_scale,
            output,
            m,
            n,
            k,
            stream,
        )
    } else {
        super::w8a16_gemm_pipelined(
            gpu,
            full_kernel,
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
}

#[cfg(test)]
#[path = "w8a16_gemm_pipelined_m32_tests.rs"]
mod tests;
