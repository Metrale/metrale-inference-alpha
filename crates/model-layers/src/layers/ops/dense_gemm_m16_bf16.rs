// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Launch wrappers for the tensor-core BF16 decode GEMM with a 16-row M tile
//! (`dense_gemm_m16_bf16`, `dense_gemm_m16_bf16_n64`), the BF16 LM head's arm for 5 to 16
//! rows.
//!
//! Owner: model-layers ops (kernels in `kernels/hopper/common/dense_gemm_m16_bf16.cu`).
//! Invariants: a launch has `1 <= m <= 16`, `k` a multiple of 64, row pitches that cover
//! `k` and `n`, and `a_row_stride` a multiple of 8; `launch` returns an error otherwise.
//!
//! Why it exists: measured 2026-09-11 on 1xH100 (Qwen3.8-27B-FP8 with a BF16 LM head,
//! decode batch 16), the batched scalar GEMV `dense_gemv_bf16_batchm` took 3,571 µs for
//! the head, about 710 GB/s, against 798 µs at batch 1, so the batched GEMV was FP32-FMA
//! bound. This kernel feeds each 16-row tile to `mma.sync.m16n8k16` instead.
//!
//! Numerics: an MMA sums K products in the tensor core's own order, so outputs can
//! differ from the scalar GEMV. The accepted difference is
//! [`crate::layers::dense_ffn::m16_tc::within_m16_tc_budget`]: at most 2 ordinal BF16
//! ULP, or an absolute error under the accumulation floor. The GPU check is the
//! model-arch example `native_bf16_lm_head_m16_microtest`. Whether the head uses this
//! arm is the target's `lm_head_m16_tc` default (on in `kernels/hopper/HARDWARE.toml`,
//! off for gb10, b200 and b300), which `METRALE_LM_HEAD_M16_TC` overrides
//! (`target_defaults.rs`); the head's route is in `model-engine lm_head_batched.rs`.
//!
//! There is no block scale and no dequant: B is BF16 already. Grid `ceil(N / N_TILE)`,
//! block 128.

use crate::weight_map::DenseWeight;
use anyhow::{Result, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

/// 2026-09-25: N columns one CTA owns in the default instantiation. It must equal the
/// kernel's `DGM16_N_TILE`.
pub const DENSE_GEMM_M16_BF16_N_TILE: u32 = 32;

/// 2026-09-25: N tile of the wide instantiation `dense_gemm_m16_bf16_n64` (kernel
/// `DGM16_N_TILE_WIDE`), selected by `METRALE_LM_HEAD_M16_TC_NTILE=64`. It launches half
/// the CTAs for a given N, so the A tile is re-read from L2 half as many times.
pub const DENSE_GEMM_M16_BF16_N_TILE_WIDE: u32 = 64;

/// 2026-09-25: The kernel's M tile. Rows past it are not computed, so the wrapper
/// returns an error rather than leave part of the output stale.
pub const DENSE_GEMM_M16_BF16_MAX_M: u32 = 16;

/// 2026-09-25: K granularity: the kernel's 4-stage cp.async pipeline advances 64
/// elements per step (`DGM16_K_STEP`), which also keeps every weight row 16-byte
/// aligned.
pub const DENSE_GEMM_M16_BF16_K_STEP: u32 = 64;

/// 2026-09-25: The signature both instantiations share, so a caller can pick one and
/// hold a single function pointer.
pub type DenseM16Bf16Gemm = fn(
    &dyn GpuBackend,
    KernelHandle,
    DevicePtr,
    &DenseWeight,
    DevicePtr,
    u32,
    u32,
    u32,
    u32,
    u32,
    u64,
) -> Result<()>;

/// 2026-09-25: The default 32-wide CTA. `input` is `[m, a_row_stride]` BF16 with `k`
/// used, `weight.weight` is the `[n, k]` BF16 weight as loaded, and `output` is
/// `[m, c_row_stride]` BF16 with `n` used.
#[allow(clippy::too_many_arguments)]
pub fn dense_gemm_m16_bf16(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    a_row_stride: u32,
    c_row_stride: u32,
    stream: u64,
) -> Result<()> {
    launch(
        gpu,
        kernel,
        DENSE_GEMM_M16_BF16_N_TILE,
        "dense_gemm_m16_bf16",
        input,
        weight,
        output,
        m,
        n,
        k,
        a_row_stride,
        c_row_stride,
        stream,
    )
}

/// 2026-09-25: The `N_TILE = 64` instantiation of the same kernel template: the same
/// arguments as [`dense_gemm_m16_bf16`], `ceil(n / 64)` CTAs instead of `ceil(n / 32)`.
#[allow(clippy::too_many_arguments)]
pub fn dense_gemm_m16_bf16_n64(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    a_row_stride: u32,
    c_row_stride: u32,
    stream: u64,
) -> Result<()> {
    launch(
        gpu,
        kernel,
        DENSE_GEMM_M16_BF16_N_TILE_WIDE,
        "dense_gemm_m16_bf16_n64",
        input,
        weight,
        output,
        m,
        n,
        k,
        a_row_stride,
        c_row_stride,
        stream,
    )
}

/// 2026-09-25: The checks and launch both instantiations share; only `n_tile` differs.
#[allow(clippy::too_many_arguments)]
fn launch(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    n_tile: u32,
    who: &str,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    a_row_stride: u32,
    c_row_stride: u32,
    stream: u64,
) -> Result<()> {
    ensure!(
        (1..=DENSE_GEMM_M16_BF16_MAX_M).contains(&m),
        "{who}: m={m} outside 1..={DENSE_GEMM_M16_BF16_MAX_M} (kernel M tile; \
         rows past it are never computed, not a launch failure)"
    );
    ensure!(
        k.is_multiple_of(DENSE_GEMM_M16_BF16_K_STEP),
        "{who}: K={k} not a multiple of {DENSE_GEMM_M16_BF16_K_STEP} \
         (cp.async pipeline step, and what keeps each [n, k] weight row 16B-aligned)"
    );
    ensure!(
        a_row_stride >= k && c_row_stride >= n,
        "{who}: row pitches (a={a_row_stride}, c={c_row_stride}) must cover the \
         used extents (k={k}, n={n})"
    );
    ensure!(
        a_row_stride.is_multiple_of(8),
        "{who}: a_row_stride={a_row_stride} must keep rows 16B-aligned \
         (cp.async stages A in 16-byte chunks)"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, n_tile), 1, 1])
        .block([128, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(a_row_stride)
        .arg_u32(c_row_stride)
        .launch(stream)
}

/// 2026-09-25: Host simulation of the kernel's staging, fragment and store index maps,
/// checked against a reference GEMM.
#[cfg(test)]
#[path = "dense_gemm_m16_bf16_tests.rs"]
mod tests;

/// 2026-09-25: Host simulation of the accumulation floor that `within_m16_tc_budget`
/// applies, and its margins.
#[cfg(test)]
#[path = "dense_gemm_m16_bf16_floor_tests.rs"]
mod floor_tests;
