// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Launchers for the routed+shared expert GEMVs of one token (NVFP4, FP8, BF16) and of two tokens (BF16).
//!
//! Owner: model-layers ops.
//! Invariants: none beyond the types.
//!
//! Single-token kernels: `blockIdx.y < top_k` is a routed expert, looked up through the device
//! pointer tables; `blockIdx.y == top_k` is the shared expert, with direct weight pointers.

use super::*;

/// 2026-09-25: NVFP4 gate and up GEMVs of the routed experts and the shared expert in one launch.
#[allow(clippy::too_many_arguments)]
pub fn moe_expert_gate_up_shared(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    gate_packed_ptrs: DevicePtr,
    gate_scale_ptrs: DevicePtr,
    gate_scale2_vals: DevicePtr,
    gate_out: DevicePtr,
    up_packed_ptrs: DevicePtr,
    up_scale_ptrs: DevicePtr,
    up_scale2_vals: DevicePtr,
    up_out: DevicePtr,
    expert_indices: DevicePtr,
    sh_gate: &QuantizedWeight,
    sh_gate_out: DevicePtr,
    sh_up: &QuantizedWeight,
    sh_up_out: DevicePtr,
    n: u32,
    k: u32,
    top_k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 8), top_k + 1, 2])
        .block([128, 1, 1])
        .arg_ptr(input)
        .arg_ptr(gate_packed_ptrs)
        .arg_ptr(gate_scale_ptrs)
        .arg_ptr(gate_scale2_vals)
        .arg_ptr(gate_out)
        .arg_ptr(up_packed_ptrs)
        .arg_ptr(up_scale_ptrs)
        .arg_ptr(up_scale2_vals)
        .arg_ptr(up_out)
        .arg_ptr(expert_indices)
        .arg_ptr(sh_gate.weight)
        .arg_ptr(sh_gate.weight_scale)
        .arg_f32(sh_gate.weight_scale_2)
        .arg_ptr(sh_gate_out)
        .arg_ptr(sh_up.weight)
        .arg_ptr(sh_up.weight_scale)
        .arg_f32(sh_up.weight_scale_2)
        .arg_ptr(sh_up_out)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(top_k)
        .launch(stream)
}

/// 2026-09-25: NVFP4 SiLU+down GEMVs in one launch. Routed experts read `gate_out`/`up_out`;
/// the shared expert reads `sh_gate_in`/`sh_up_in`. The kernel's `s_act` is dynamic shared
/// memory of K floats.
#[allow(clippy::too_many_arguments)]
pub fn moe_expert_silu_down_shared(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate_out: DevicePtr,
    up_out: DevicePtr,
    packed_ptrs: DevicePtr,
    scale_ptrs: DevicePtr,
    scale2_vals: DevicePtr,
    output: DevicePtr,
    expert_indices: DevicePtr,
    sh_gate_in: DevicePtr,
    sh_up_in: DevicePtr,
    sh_down: &QuantizedWeight,
    sh_down_out: DevicePtr,
    n: u32,
    k: u32,
    top_k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 8), top_k + 1, 1])
        .block([128, 1, 1])
        .shared_mem(k * 4)
        .arg_ptr(gate_out)
        .arg_ptr(up_out)
        .arg_ptr(packed_ptrs)
        .arg_ptr(scale_ptrs)
        .arg_ptr(scale2_vals)
        .arg_ptr(output)
        .arg_ptr(expert_indices)
        .arg_ptr(sh_gate_in)
        .arg_ptr(sh_up_in)
        .arg_ptr(sh_down.weight)
        .arg_ptr(sh_down.weight_scale)
        .arg_f32(sh_down.weight_scale_2)
        .arg_ptr(sh_down_out)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(top_k)
        .launch(stream)
}

/// 2026-09-25: FP8 variant of [`moe_expert_gate_up_shared`]: two pointer tables per projection
/// (weights, scales) and the shared expert as direct `Fp8Weight` pointers.
#[allow(clippy::too_many_arguments)]
pub fn moe_expert_gate_up_shared_fp8(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    gate_weight_ptrs: DevicePtr,
    gate_scale_ptrs: DevicePtr,
    gate_out: DevicePtr,
    up_weight_ptrs: DevicePtr,
    up_scale_ptrs: DevicePtr,
    up_out: DevicePtr,
    expert_indices: DevicePtr,
    sh_gate: &Fp8Weight,
    sh_gate_out: DevicePtr,
    sh_up: &Fp8Weight,
    sh_up_out: DevicePtr,
    n: u32,
    k: u32,
    top_k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 8), top_k + 1, 2])
        .block([128, 1, 1])
        .arg_ptr(input)
        .arg_ptr(gate_weight_ptrs)
        .arg_ptr(gate_scale_ptrs)
        .arg_ptr(gate_out)
        .arg_ptr(up_weight_ptrs)
        .arg_ptr(up_scale_ptrs)
        .arg_ptr(up_out)
        .arg_ptr(expert_indices)
        .arg_ptr(sh_gate.weight)
        .arg_ptr(sh_gate.row_scale)
        .arg_ptr(sh_gate_out)
        .arg_ptr(sh_up.weight)
        .arg_ptr(sh_up.row_scale)
        .arg_ptr(sh_up_out)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(top_k)
        .launch(stream)
}

/// 2026-09-25: BF16 variant of [`moe_expert_gate_up_shared_fp8`], with no scale tables, for
/// experts installed by `MoeLayer::set_bf16_experts`.
#[allow(clippy::too_many_arguments)]
pub fn moe_expert_gate_up_shared_bf16(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    gate_weight_ptrs: DevicePtr,
    gate_out: DevicePtr,
    up_weight_ptrs: DevicePtr,
    up_out: DevicePtr,
    expert_indices: DevicePtr,
    sh_gate_weight: DevicePtr,
    sh_gate_out: DevicePtr,
    sh_up_weight: DevicePtr,
    sh_up_out: DevicePtr,
    n: u32,
    k: u32,
    top_k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 8), top_k + 1, 2])
        .block([128, 1, 1])
        .arg_ptr(input)
        .arg_ptr(gate_weight_ptrs)
        .arg_ptr(gate_out)
        .arg_ptr(up_weight_ptrs)
        .arg_ptr(up_out)
        .arg_ptr(expert_indices)
        .arg_ptr(sh_gate_weight)
        .arg_ptr(sh_gate_out)
        .arg_ptr(sh_up_weight)
        .arg_ptr(sh_up_out)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(top_k)
        .launch(stream)
}

/// 2026-09-25: BF16 variant of [`moe_expert_silu_down_shared_fp8`], with no scale tables.
#[allow(clippy::too_many_arguments)]
pub fn moe_expert_silu_down_shared_bf16(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate_out: DevicePtr,
    up_out: DevicePtr,
    down_weight_ptrs: DevicePtr,
    output: DevicePtr,
    expert_indices: DevicePtr,
    sh_gate_in: DevicePtr,
    sh_up_in: DevicePtr,
    sh_down_weight: DevicePtr,
    sh_down_out: DevicePtr,
    n: u32,
    k: u32,
    top_k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 8), top_k + 1, 1])
        .block([128, 1, 1])
        .arg_ptr(gate_out)
        .arg_ptr(up_out)
        .arg_ptr(down_weight_ptrs)
        .arg_ptr(output)
        .arg_ptr(expert_indices)
        .arg_ptr(sh_gate_in)
        .arg_ptr(sh_up_in)
        .arg_ptr(sh_down_weight)
        .arg_ptr(sh_down_out)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(top_k)
        .launch(stream)
}

/// 2026-09-25: [`moe_expert_gate_up_shared_bf16`] for two tokens in one launch (called from
/// `MoeLayer::forward_k2`). `blockIdx.y` in `[0, 2*top_k)` is a routed expert (token
/// `y / top_k`, slot `y % top_k`, output row `token * top_k + slot`); `blockIdx.y == 2*top_k` is
/// the shared expert, computed for both tokens into rows 0 and 1.
#[allow(clippy::too_many_arguments)]
pub fn moe_expert_gate_up_shared_bf16_batch2(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    gate_weight_ptrs: DevicePtr,
    gate_out: DevicePtr,
    up_weight_ptrs: DevicePtr,
    up_out: DevicePtr,
    expert_indices: DevicePtr,
    sh_gate_weight: DevicePtr,
    sh_gate_out: DevicePtr,
    sh_up_weight: DevicePtr,
    sh_up_out: DevicePtr,
    n: u32,
    k: u32,
    top_k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 8), 2 * top_k + 1, 2])
        .block([128, 1, 1])
        .arg_ptr(input)
        .arg_ptr(gate_weight_ptrs)
        .arg_ptr(gate_out)
        .arg_ptr(up_weight_ptrs)
        .arg_ptr(up_out)
        .arg_ptr(expert_indices)
        .arg_ptr(sh_gate_weight)
        .arg_ptr(sh_gate_out)
        .arg_ptr(sh_up_weight)
        .arg_ptr(sh_up_out)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(top_k)
        .launch(stream)
}

/// 2026-09-25: [`moe_expert_silu_down_shared_bf16`] for two tokens, with the
/// [`moe_expert_gate_up_shared_bf16_batch2`] row layout.
#[allow(clippy::too_many_arguments)]
pub fn moe_expert_silu_down_shared_bf16_batch2(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate_out: DevicePtr,
    up_out: DevicePtr,
    down_weight_ptrs: DevicePtr,
    output: DevicePtr,
    expert_indices: DevicePtr,
    sh_gate_in: DevicePtr,
    sh_up_in: DevicePtr,
    sh_down_weight: DevicePtr,
    sh_down_out: DevicePtr,
    n: u32,
    k: u32,
    top_k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 8), 2 * top_k + 1, 1])
        .block([128, 1, 1])
        .arg_ptr(gate_out)
        .arg_ptr(up_out)
        .arg_ptr(down_weight_ptrs)
        .arg_ptr(output)
        .arg_ptr(expert_indices)
        .arg_ptr(sh_gate_in)
        .arg_ptr(sh_up_in)
        .arg_ptr(sh_down_weight)
        .arg_ptr(sh_down_out)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(top_k)
        .launch(stream)
}

/// 2026-09-25: FP8 variant of [`moe_expert_silu_down_shared`]: two pointer tables (weights,
/// scales) and the shared down weight as a direct `Fp8Weight` pointer.
#[allow(clippy::too_many_arguments)]
pub fn moe_expert_silu_down_shared_fp8(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate_out: DevicePtr,
    up_out: DevicePtr,
    down_weight_ptrs: DevicePtr,
    down_scale_ptrs: DevicePtr,
    output: DevicePtr,
    expert_indices: DevicePtr,
    sh_gate_in: DevicePtr,
    sh_up_in: DevicePtr,
    sh_down: &Fp8Weight,
    sh_down_out: DevicePtr,
    n: u32,
    k: u32,
    top_k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 8), top_k + 1, 1])
        .block([128, 1, 1])
        .arg_ptr(gate_out)
        .arg_ptr(up_out)
        .arg_ptr(down_weight_ptrs)
        .arg_ptr(down_scale_ptrs)
        .arg_ptr(output)
        .arg_ptr(expert_indices)
        .arg_ptr(sh_gate_in)
        .arg_ptr(sh_up_in)
        .arg_ptr(sh_down.weight)
        .arg_ptr(sh_down.row_scale)
        .arg_ptr(sh_down_out)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(top_k)
        .launch(stream)
}
