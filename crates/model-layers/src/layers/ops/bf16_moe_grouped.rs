// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: Launchers for the grouped BF16 MoE decode kernels
//! (`kernels/gb10/common/moe_bf16_grouped_tc.cu`, the BF16 point of the tensor-core grouped
//! family). The layout is that of the grouped FP8 and NVFP4 decodes: rows grouped by expert by
//! `moe_fp8_grouped_sort`, intermediates by sorted position, the blend
//! `moe_weighted_sum_blend_fp8_grouped`. Weights are row-major BF16 per expert (pointer tables).
//!
//! Owner: model-layers ops.
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::Fp8GroupedGeometry;

/// 2026-10-02: `moe_expert_gate_up_act_bf16_grouped_tc` (`BTC_GU_COLS` columns per CTA; writes the
/// SiLU product as BF16 hi + lo).
pub const BF16_GROUPED_GATE_UP_TC: Fp8GroupedGeometry = Fp8GroupedGeometry {
    cols_per_cta: 64,
    rows_per_pass: 8,
    threads: 128,
};

/// 2026-10-02: `moe_expert_down_act_bf16_grouped_tc` (`BTC_DOWN_COLS` columns per CTA).
pub const BF16_GROUPED_DOWN_TC: Fp8GroupedGeometry = Fp8GroupedGeometry {
    cols_per_cta: 128,
    rows_per_pass: 8,
    threads: 128,
};

/// 2026-10-02: Whether the BF16 tensor-core kernels take an `n`-column, `k`-deep projection: `k`
/// in whole load groups (two 32-K chunks; 128 for margin), `n` in whole CTAs of `geometry`.
pub fn bf16_grouped_tc_shape_ok(n: u32, k: u32, geometry: Fp8GroupedGeometry) -> bool {
    n > 0 && k > 0 && k.is_multiple_of(128) && n.is_multiple_of(geometry.cols_per_cta)
}

/// 2026-10-02: Grouped BF16 gate+up and SiLU. `gate_ptrs` / `up_ptrs` are device tables of each
/// routed expert's `[n, k]` BF16 weight; `sh_gate` / `sh_up` the shared expert's. Writes the SiLU
/// product as BF16 hi + lo rows (`[positions, 2n]` into `act`, `[num_tokens, 2n]` into `sh_act`).
#[allow(clippy::too_many_arguments)]
pub fn moe_expert_gate_up_act_bf16_grouped(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    gate_ptrs: DevicePtr,
    up_ptrs: DevicePtr,
    act: DevicePtr,
    expert_offsets: DevicePtr,
    sorted_token_ids: DevicePtr,
    active_experts: DevicePtr,
    active_count: DevicePtr,
    sh_gate: DevicePtr,
    sh_up: DevicePtr,
    sh_act: DevicePtr,
    n: u32,
    k: u32,
    cap: u32,
    num_tokens: u32,
    stream: u64,
) -> Result<()> {
    let g = BF16_GROUPED_GATE_UP_TC;
    KernelLaunch::new(gpu, kernel)
        .grid([
            div_ceil(n, g.cols_per_cta),
            cap + div_ceil(num_tokens, g.rows_per_pass),
            1,
        ])
        .block([g.threads, 1, 1])
        .arg_ptr(input)
        .arg_ptr(gate_ptrs)
        .arg_ptr(up_ptrs)
        .arg_ptr(act)
        .arg_ptr(expert_offsets)
        .arg_ptr(sorted_token_ids)
        .arg_ptr(active_experts)
        .arg_ptr(active_count)
        .arg_ptr(sh_gate)
        .arg_ptr(sh_up)
        .arg_ptr(sh_act)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(cap)
        .arg_u32(num_tokens)
        .launch(stream)
}

/// 2026-10-02: Grouped BF16 down over the hi + lo SiLU rows. `k` is the intermediate width; rows
/// of `act` and `output` are sorted positions, rows of `sh_act` and `sh_down_out` tokens.
#[allow(clippy::too_many_arguments)]
pub fn moe_expert_down_act_bf16_grouped(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    act: DevicePtr,
    down_ptrs: DevicePtr,
    output: DevicePtr,
    expert_offsets: DevicePtr,
    active_experts: DevicePtr,
    active_count: DevicePtr,
    sh_act: DevicePtr,
    sh_down: DevicePtr,
    sh_down_out: DevicePtr,
    n: u32,
    k: u32,
    cap: u32,
    num_tokens: u32,
    stream: u64,
) -> Result<()> {
    let g = BF16_GROUPED_DOWN_TC;
    KernelLaunch::new(gpu, kernel)
        .grid([
            div_ceil(n, g.cols_per_cta),
            cap + div_ceil(num_tokens, g.rows_per_pass),
            1,
        ])
        .block([g.threads, 1, 1])
        .arg_ptr(act)
        .arg_ptr(down_ptrs)
        .arg_ptr(output)
        .arg_ptr(expert_offsets)
        .arg_ptr(active_experts)
        .arg_ptr(active_count)
        .arg_ptr(sh_act)
        .arg_ptr(sh_down)
        .arg_ptr(sh_down_out)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(cap)
        .arg_u32(num_tokens)
        .launch(stream)
}
