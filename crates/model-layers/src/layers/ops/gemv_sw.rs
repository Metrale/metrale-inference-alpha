// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Decode NVFP4 W4A16 GEMV launchers whose grid is tied to the kernel's `N_PER_BLOCK` / `N_PER_BLOCK_SW` defines.
//!
//! Owner: model-layers ops.
//! Invariants:
//! - `w4a16_gemv` computes 4 outputs per 256-thread block (64 threads each) and
//!   `w4a16_gemv_sw` computes 8 (one warp each), so a launch that swaps the kernel must also
//!   swap the grid. [`w4a16_decode_gemv`] picks both together.
//! - `gemv_sw_tests::cuda_n_per_block_matches_rust_ssot` checks that every compiled copy of the
//!   kernel sources defines the two constants below.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::QuantizedWeight;

/// 2026-09-25: Outputs per 256-thread block of `w4a16_gemv`; equals `#define N_PER_BLOCK` in
/// kernels/gb10/common/w4a16_gemv.cu.
pub const W4A16_GEMV_OUTS_PER_BLOCK: u32 = 4;

/// 2026-09-25: Outputs per 256-thread block of the single-warp kernels (`w4a16_gemv_sw` and the
/// `_sw` kernels in w4a16_gemv_fused.cu); equals `#define N_PER_BLOCK_SW`.
pub const W4A16_GEMV_SW_OUTS_PER_BLOCK: u32 = 8;

pub fn w4a16_gemv_grid_x(n: u32) -> u32 {
    div_ceil(n, W4A16_GEMV_OUTS_PER_BLOCK)
}

pub fn w4a16_gemv_sw_grid_x(n: u32) -> u32 {
    div_ceil(n, W4A16_GEMV_SW_OUTS_PER_BLOCK)
}

/// 2026-09-25: Polarity of the `METRALE_NO_GEMV_SW` kill switch, given the variable's value: the
/// single-warp GEMV is on unless the value is exactly `"1"`, so `=0` and an empty value leave
/// it on.
pub fn gemv_sw_from(no_gemv_sw: Option<&str>) -> bool {
    no_gemv_sw != Some("1")
}

/// 2026-09-25: Use the single-warp kernel only when the lever is on and its handle resolved
/// (a missing kernel is `KernelHandle(0)`).
pub fn use_gemv_sw(lever: bool, sw_handle: KernelHandle) -> bool {
    lever && sw_handle.0 != 0
}

/// 2026-09-25: Single-warp-per-output W4A16 GEMV for M=1, grid `ceil(N/8)`.
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_sw(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([w4a16_gemv_sw_grid_x(n), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// 2026-09-25: The launch of [`w4a16_gemv_sw`] for callers that hold the NVFP4 packed weight,
/// scale and global scale as separate values rather than a [`QuantizedWeight`] (the GLM-5.3
/// MLP in `model-arch/src/glm5next_mlp/forward.rs`). It keeps the grid on
/// [`w4a16_gemv_sw_grid_x`].
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_sw_raw(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    packed: DevicePtr,
    scale: DevicePtr,
    scale_2: f32,
    output: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([w4a16_gemv_sw_grid_x(n), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(packed)
        .arg_ptr(scale)
        .arg_f32(scale_2)
        .arg_ptr(output)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// 2026-09-25: Decode GEMV: the single-warp kernel when [`use_gemv_sw`] allows it, otherwise
/// the base `w4a16_gemv`, each with its own grid.
#[allow(clippy::too_many_arguments)]
pub fn w4a16_decode_gemv(
    gpu: &dyn GpuBackend,
    gemv: KernelHandle,
    gemv_sw: KernelHandle,
    use_sw: bool,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    if use_gemv_sw(use_sw, gemv_sw) {
        w4a16_gemv_sw(gpu, gemv_sw, input, weight, output, n, k, stream)
    } else {
        super::quant_dispatch::w4a16_gemv(gpu, gemv, input, weight, output, n, k, stream)
    }
}

#[cfg(test)]
#[path = "gemv_sw_tests.rs"]
mod tests;
