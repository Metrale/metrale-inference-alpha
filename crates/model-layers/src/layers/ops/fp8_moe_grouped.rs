// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Launchers for the cross-row grouped FP8 MoE decode kernels
//! (`moe_shared_expert_fused_fp8_grouped.cu`, `moe_fp8_grouped_blend.cu`).
//!
//! Owner: model-layers ops.
//! Invariants: none beyond the types.
//!
//! Rows are grouped by expert (`moe_sort_by_expert`). A gate+up or SiLU+down
//! CTA owns one active expert and applies each weight row it streams to every
//! row routed to that expert; the batch-2/3 kernels put one (token, slot) pair
//! on each `blockIdx.y` instead. Intermediates are laid out by sorted
//! position, and the blend maps each slot back through `token_to_perm`.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::Fp8Weight;

/// 2026-09-25: Rows per register pass inside the grouped kernels. Must equal
/// `GROUP_ROWS` in the `.cu`; the SiLU+down launch sizes its dynamic shared
/// memory from it.
pub const FP8_GROUPED_ROWS_PER_PASS: u32 = 8;

/// 2026-09-25: Output columns per SiLU+down CTA. Must equal
/// `DOWN_COLS_PER_CTA` in the `.cu`.
pub const FP8_GROUPED_DOWN_COLS_PER_CTA: u32 = 32;

/// 2026-09-25: Dynamic shared memory, in bytes, the SiLU+down kernel needs for
/// intermediate width `k`.
pub fn fp8_grouped_silu_down_smem_bytes(k: u32) -> usize {
    FP8_GROUPED_ROWS_PER_PASS as usize * k as usize * 4
}

/// 2026-09-25: Cap on active experts, which sizes the grouped grids' Y extent:
/// `num_tokens * top_k` rows can reach at most that many distinct experts. It
/// does not depend on the routing, so a captured graph stays valid for every
/// routing.
pub fn fp8_grouped_active_cap(num_tokens: u32, top_k: u32, num_experts: u32) -> u32 {
    (num_tokens * top_k).min(num_experts)
}

/// 2026-09-25: Builds the compacted active-expert list from `expert_offsets`:
/// `active_experts[0..count]` in ascending order, and `active_count[0] =
/// count`. Launch it on the stream of the grouped kernels that read it.
pub fn moe_fp8_grouped_compact(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    expert_offsets: DevicePtr,
    active_experts: DevicePtr,
    active_count: DevicePtr,
    num_experts: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([1, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(expert_offsets)
        .arg_ptr(active_experts)
        .arg_ptr(active_count)
        .arg_u32(num_experts)
        .launch(stream)
}

/// 2026-09-25: Grouped FP8 gate+up. `cap` is [`fp8_grouped_active_cap`]; the
/// extra `blockIdx.y` is the shared expert. Rows of `gate_out`/`up_out` are
/// sorted positions.
#[allow(clippy::too_many_arguments)]
pub fn moe_expert_gate_up_shared_fp8_grouped(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    gp_w: DevicePtr,
    gp_s: DevicePtr,
    gate_out: DevicePtr,
    up_w: DevicePtr,
    up_s: DevicePtr,
    up_out: DevicePtr,
    expert_offsets: DevicePtr,
    sorted_token_ids: DevicePtr,
    active_experts: DevicePtr,
    active_count: DevicePtr,
    sh_gate: &Fp8Weight,
    sh_gate_out: DevicePtr,
    sh_up: &Fp8Weight,
    sh_up_out: DevicePtr,
    n: u32,
    k: u32,
    cap: u32,
    num_tokens: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 8), cap + 1, 2])
        .block([128, 1, 1])
        .arg_ptr(input)
        .arg_ptr(gp_w)
        .arg_ptr(gp_s)
        .arg_ptr(gate_out)
        .arg_ptr(up_w)
        .arg_ptr(up_s)
        .arg_ptr(up_out)
        .arg_ptr(expert_offsets)
        .arg_ptr(sorted_token_ids)
        .arg_ptr(active_experts)
        .arg_ptr(active_count)
        .arg_ptr(sh_gate.weight)
        .arg_ptr(sh_gate.row_scale)
        .arg_ptr(sh_gate_out)
        .arg_ptr(sh_up.weight)
        .arg_ptr(sh_up.row_scale)
        .arg_ptr(sh_up_out)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(cap)
        .arg_u32(num_tokens)
        .launch(stream)
}

/// 2026-09-25: Grouped FP8 SiLU+down: 256 threads, 8 warps of 4 output
/// columns each. `k` is the intermediate width; rows of
/// `gate_out`/`up_out`/`output` are sorted positions.
#[allow(clippy::too_many_arguments)]
pub fn moe_expert_silu_down_shared_fp8_grouped(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate_out: DevicePtr,
    up_out: DevicePtr,
    down_w: DevicePtr,
    down_s: DevicePtr,
    output: DevicePtr,
    expert_offsets: DevicePtr,
    active_experts: DevicePtr,
    active_count: DevicePtr,
    sh_gate_in: DevicePtr,
    sh_up_in: DevicePtr,
    sh_down: &Fp8Weight,
    sh_down_out: DevicePtr,
    n: u32,
    k: u32,
    cap: u32,
    num_tokens: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, FP8_GROUPED_DOWN_COLS_PER_CTA), cap + 1, 1])
        .block([256, 1, 1])
        .shared_mem(fp8_grouped_silu_down_smem_bytes(k) as u32)
        .arg_ptr(gate_out)
        .arg_ptr(up_out)
        .arg_ptr(down_w)
        .arg_ptr(down_s)
        .arg_ptr(output)
        .arg_ptr(expert_offsets)
        .arg_ptr(active_experts)
        .arg_ptr(active_count)
        .arg_ptr(sh_gate_in)
        .arg_ptr(sh_up_in)
        .arg_ptr(sh_down.weight)
        .arg_ptr(sh_down.row_scale)
        .arg_ptr(sh_down_out)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(cap)
        .arg_u32(num_tokens)
        .launch(stream)
}

/// 2026-09-25: Grouped blend, one `blockIdx.y` per token. `expert_out` rows
/// are sorted positions; `token_to_perm[token * top_k + k]` maps a slot to its
/// row. `gate_weight` may be NULL (ungated shared expert).
#[allow(clippy::too_many_arguments)]
pub fn moe_weighted_sum_blend_fp8_grouped(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    output: DevicePtr,
    expert_out: DevicePtr,
    expert_weights: DevicePtr,
    token_to_perm: DevicePtr,
    shared_out: DevicePtr,
    input: DevicePtr,
    gate_weight: DevicePtr,
    hidden: u32,
    top_k: u32,
    k: u32,
    num_tokens: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(hidden, 256), num_tokens, 1])
        .block([256, 1, 1])
        .arg_ptr(output)
        .arg_ptr(expert_out)
        .arg_ptr(expert_weights)
        .arg_ptr(token_to_perm)
        .arg_ptr(shared_out)
        .arg_ptr(input)
        .arg_ptr(gate_weight)
        .arg_u32(hidden)
        .arg_u32(top_k)
        .arg_u32(k)
        .launch(stream)
}
