// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-27: Launchers for the grouped NVFP4 MoE decode kernels
//! (`kernels/gb10/common/moe_nvfp4_grouped.cu`).
//!
//! Owner: model-layers ops.
//! Invariants: none beyond the types.
//!
//! The layout is that of the grouped FP8 decode (`fp8_moe_grouped.rs`): rows grouped by
//! expert and the active experts listed by `moe_fp8_grouped_sort`, intermediates laid out
//! by sorted position, and the blend
//! `moe_weighted_sum_blend_fp8_grouped`. Only the weight format differs: row-major
//! NVFP4 (packed E2M1, E4M3 block scales of 16, per-tensor scale 2).

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::Fp8GroupedGeometry;
use crate::weight_map::QuantizedWeight;

/// 2026-09-27: Output columns per gate+up CTA. Must equal `NG_GU_COLS_PER_CTA` in the `.cu`.
pub const NVFP4_GROUPED_GATE_UP_COLS_PER_CTA: u32 = 16;

/// 2026-09-27: Rows per gate+up pass. Must equal `NG_GU_ROWS` in the `.cu`.
pub const NVFP4_GROUPED_GATE_UP_ROWS_PER_PASS: u32 = 4;

/// 2026-09-27: Output columns per down CTA. Must equal `NG_DOWN_COLS_PER_CTA` in the `.cu`.
pub const NVFP4_GROUPED_DOWN_COLS_PER_CTA: u32 = 128;

/// 2026-09-27: Rows per down pass. Must equal `NG_DOWN_ROWS` in the `.cu`.
pub const NVFP4_GROUPED_DOWN_ROWS_PER_PASS: u32 = 4;

/// 2026-10-02: `moe_expert_gate_up_act_nvfp4_grouped` (CUDA cores, FP32 SiLU products).
pub const NVFP4_GROUPED_GATE_UP_SCALAR: Fp8GroupedGeometry = Fp8GroupedGeometry {
    cols_per_cta: NVFP4_GROUPED_GATE_UP_COLS_PER_CTA,
    rows_per_pass: NVFP4_GROUPED_GATE_UP_ROWS_PER_PASS,
    threads: 128,
};

/// 2026-10-02: `moe_expert_down_act_nvfp4_grouped` (reads FP32 SiLU products).
pub const NVFP4_GROUPED_DOWN_SCALAR: Fp8GroupedGeometry = Fp8GroupedGeometry {
    cols_per_cta: NVFP4_GROUPED_DOWN_COLS_PER_CTA,
    rows_per_pass: NVFP4_GROUPED_DOWN_ROWS_PER_PASS,
    threads: 256,
};

/// 2026-10-02: Rows per pass of the tensor-core kernels (`moe_nvfp4_grouped_tc.cu`,
/// `TC_ROWS`), gate+up and down alike.
pub const NVFP4_GROUPED_TC_ROWS_PER_PASS: u32 = 8;

/// 2026-10-02: `moe_expert_gate_up_act_nvfp4_grouped_tc` (`NTC_GU_COLS` columns per CTA; writes
/// the SiLU product as BF16 hi + lo).
pub const NVFP4_GROUPED_GATE_UP_TC: Fp8GroupedGeometry = Fp8GroupedGeometry {
    cols_per_cta: 64,
    rows_per_pass: NVFP4_GROUPED_TC_ROWS_PER_PASS,
    threads: 128,
};

/// 2026-10-02: `moe_expert_down_act_nvfp4_grouped_tc` (`NTC_DOWN_COLS` columns per CTA; reads
/// the hi + lo SiLU rows).
pub const NVFP4_GROUPED_DOWN_TC: Fp8GroupedGeometry = Fp8GroupedGeometry {
    cols_per_cta: 128,
    rows_per_pass: NVFP4_GROUPED_TC_ROWS_PER_PASS,
    threads: 128,
};

/// 2026-10-02: Whether the tensor-core kernels take an `n`-column, `k`-deep projection: `k` in
/// whole load groups of two 128-wide chunks, `n` in whole CTAs of `geometry`.
pub fn nvfp4_grouped_tc_shape_ok(n: u32, k: u32, geometry: Fp8GroupedGeometry) -> bool {
    n > 0 && k > 0 && k.is_multiple_of(256) && n.is_multiple_of(geometry.cols_per_cta)
}

/// 2026-09-27: One NVFP4 projection across the routed experts: device tables of each
/// expert's packed and block-scale pointers, and its per-tensor scales as f32.
#[derive(Clone, Copy, Debug)]
pub struct Nvfp4ExpertTables {
    pub packed_ptrs: DevicePtr,
    pub scale_ptrs: DevicePtr,
    pub scale2_vals: DevicePtr,
}

/// 2026-09-27: Grouped NVFP4 gate+up and SiLU over `k`-wide input rows. `cap` is
/// `fp8_grouped_active_cap`; the first `ceil(num_tokens / geometry.rows_per_pass)` block rows
/// are the shared expert. Writes the FP32 product `silu(bf16(gate)) * bf16(up)`,
/// `[positions, n]` for the routed experts into `act` and `[num_tokens, n]` for the shared
/// expert into `sh_act` (2026-10-02: the tensor-core kernel writes it as BF16 hi + lo in the
/// same bytes, which only its down kernel reads).
#[allow(clippy::too_many_arguments)]
pub fn moe_expert_gate_up_act_nvfp4_grouped(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    geometry: Fp8GroupedGeometry,
    input: DevicePtr,
    gate: Nvfp4ExpertTables,
    up: Nvfp4ExpertTables,
    act: DevicePtr,
    expert_offsets: DevicePtr,
    sorted_token_ids: DevicePtr,
    active_experts: DevicePtr,
    active_count: DevicePtr,
    sh_gate: &QuantizedWeight,
    sh_up: &QuantizedWeight,
    sh_act: DevicePtr,
    n: u32,
    k: u32,
    cap: u32,
    num_tokens: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([
            div_ceil(n, geometry.cols_per_cta),
            cap + div_ceil(num_tokens, geometry.rows_per_pass),
            1,
        ])
        .block([geometry.threads, 1, 1])
        .arg_ptr(input)
        .arg_ptr(gate.packed_ptrs)
        .arg_ptr(gate.scale_ptrs)
        .arg_ptr(gate.scale2_vals)
        .arg_ptr(up.packed_ptrs)
        .arg_ptr(up.scale_ptrs)
        .arg_ptr(up.scale2_vals)
        .arg_ptr(act)
        .arg_ptr(expert_offsets)
        .arg_ptr(sorted_token_ids)
        .arg_ptr(active_experts)
        .arg_ptr(active_count)
        .arg_ptr(sh_gate.weight)
        .arg_ptr(sh_gate.weight_scale)
        .arg_f32(sh_gate.weight_scale_2)
        .arg_ptr(sh_up.weight)
        .arg_ptr(sh_up.weight_scale)
        .arg_f32(sh_up.weight_scale_2)
        .arg_ptr(sh_act)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(cap)
        .arg_u32(num_tokens)
        .launch(stream)
}

/// 2026-09-27: Grouped NVFP4 down over the SiLU product. `k` is the intermediate width; rows
/// of `act` and `output` are sorted positions, rows of `sh_act` and `sh_down_out` tokens.
#[allow(clippy::too_many_arguments)]
pub fn moe_expert_down_act_nvfp4_grouped(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    geometry: Fp8GroupedGeometry,
    act: DevicePtr,
    down: Nvfp4ExpertTables,
    output: DevicePtr,
    expert_offsets: DevicePtr,
    active_experts: DevicePtr,
    active_count: DevicePtr,
    sh_act: DevicePtr,
    sh_down: &QuantizedWeight,
    sh_down_out: DevicePtr,
    n: u32,
    k: u32,
    cap: u32,
    num_tokens: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([
            div_ceil(n, geometry.cols_per_cta),
            cap + div_ceil(num_tokens, geometry.rows_per_pass),
            1,
        ])
        .block([geometry.threads, 1, 1])
        .arg_ptr(act)
        .arg_ptr(down.packed_ptrs)
        .arg_ptr(down.scale_ptrs)
        .arg_ptr(down.scale2_vals)
        .arg_ptr(output)
        .arg_ptr(expert_offsets)
        .arg_ptr(active_experts)
        .arg_ptr(active_count)
        .arg_ptr(sh_act)
        .arg_ptr(sh_down.weight)
        .arg_ptr(sh_down.weight_scale)
        .arg_f32(sh_down.weight_scale_2)
        .arg_ptr(sh_down_out)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(cap)
        .arg_u32(num_tokens)
        .launch(stream)
}
