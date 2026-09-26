// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Launchers for the FP8 MoE batch-2/3 decode kernels, the NVFP4
//! transposed-weight single-token MoE kernels, the uint8 transposes, and
//! `causal_conv1d_fwd` and `bf16_concat`.
//!
//! Owner: model-layers ops.
//! Invariants: none beyond the types.

#![allow(unused_imports)]

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::layers::moe;
use crate::weight_map::{DenseWeight, Fp8DenseWeight, Fp8Weight, QuantizedWeight};

use super::*;

/// 2026-09-25: Causal depthwise conv1d followed by SiLU over a full sequence
/// (kernel `causal_conv1d_fwd`), launched without a bias. Input
/// `[batch, dim, seq_len]` BF16 (channel-first), weight `[dim, d_conv]` BF16,
/// output `[batch, dim, seq_len]` BF16.
#[allow(clippy::too_many_arguments)]
pub fn conv1d_fwd(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    batch: u32,
    dim: u32,
    seq_len: u32,
    d_conv: u32,
    stream: u64,
) -> Result<()> {
    let block_x = std::cmp::min(seq_len, 1024);
    KernelLaunch::new(gpu, kernel)
        .grid([dim, batch, 1])
        .block([block_x, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(DevicePtr::NULL)
        .arg_ptr(output)
        .arg_u32(batch)
        .arg_u32(dim)
        .arg_u32(seq_len)
        .arg_u32(d_conv)
        .launch(stream)
}

/// 2026-09-25: BF16 concatenation (kernel `bf16_concat`):
/// `out[0..N] = a[0..N]`, `out[N..2N] = b[0..N]`.
pub fn bf16_concat(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a: DevicePtr,
    b: DevicePtr,
    output: DevicePtr,
    n: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(b)
        .arg_ptr(output)
        .arg_u32(n)
        .launch(stream)
}

/// 2026-09-25: FP8 fused gate+up GEMV over the routed experts and the shared
/// expert, for two tokens (`moe/forward_k2.rs`).
#[allow(clippy::too_many_arguments)]
pub fn moe_expert_gate_up_shared_fp8_batch2(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    gp_w: DevicePtr,
    gp_s: DevicePtr,
    gate_out: DevicePtr,
    up_w: DevicePtr,
    up_s: DevicePtr,
    up_out: DevicePtr,
    indices: DevicePtr,
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
        .grid([div_ceil(n, 8), 2 * (top_k + 1), 2])
        .block([128, 1, 1])
        .arg_ptr(input)
        .arg_ptr(gp_w)
        .arg_ptr(gp_s)
        .arg_ptr(gate_out)
        .arg_ptr(up_w)
        .arg_ptr(up_s)
        .arg_ptr(up_out)
        .arg_ptr(indices)
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

/// 2026-09-25: FP8 fused SiLU+down GEMV over the routed experts and the shared
/// expert, for two tokens.
#[allow(clippy::too_many_arguments)]
pub fn moe_expert_silu_down_shared_fp8_batch2(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate_out: DevicePtr,
    up_out: DevicePtr,
    dp_w: DevicePtr,
    dp_s: DevicePtr,
    output: DevicePtr,
    indices: DevicePtr,
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
        .grid([div_ceil(n, 8), 2 * (top_k + 1), 1])
        .block([128, 1, 1])
        .arg_ptr(gate_out)
        .arg_ptr(up_out)
        .arg_ptr(dp_w)
        .arg_ptr(dp_s)
        .arg_ptr(output)
        .arg_ptr(indices)
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

/// 2026-09-25: FP8 fused gate+up GEMV over the routed experts and the shared
/// expert, for three tokens (`moe/forward_k3.rs`).
#[allow(clippy::too_many_arguments)]
pub fn moe_expert_gate_up_shared_fp8_batch3(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    gp_w: DevicePtr,
    gp_s: DevicePtr,
    gate_out: DevicePtr,
    up_w: DevicePtr,
    up_s: DevicePtr,
    up_out: DevicePtr,
    indices: DevicePtr,
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
        .grid([div_ceil(n, 8), 3 * (top_k + 1), 2])
        .block([128, 1, 1])
        .arg_ptr(input)
        .arg_ptr(gp_w)
        .arg_ptr(gp_s)
        .arg_ptr(gate_out)
        .arg_ptr(up_w)
        .arg_ptr(up_s)
        .arg_ptr(up_out)
        .arg_ptr(indices)
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

/// 2026-09-25: FP8 fused SiLU+down GEMV over the routed experts and the shared
/// expert, for three tokens.
#[allow(clippy::too_many_arguments)]
pub fn moe_expert_silu_down_shared_fp8_batch3(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate_out: DevicePtr,
    up_out: DevicePtr,
    dp_w: DevicePtr,
    dp_s: DevicePtr,
    output: DevicePtr,
    indices: DevicePtr,
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
        .grid([div_ceil(n, 8), 3 * (top_k + 1), 1])
        .block([128, 1, 1])
        .arg_ptr(gate_out)
        .arg_ptr(up_out)
        .arg_ptr(dp_w)
        .arg_ptr(dp_s)
        .arg_ptr(output)
        .arg_ptr(indices)
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

#[allow(clippy::too_many_arguments)]
/// 2026-09-25: Single-matrix uint8 transpose `[rows, cols] -> [cols, rows]` on
/// the GPU (`transpose_u8.cu`, 32x32 shared-memory tiles). The quantized
/// weight transposes in `weight_map/quantized/transpose.rs` call it.
pub fn transpose_u8(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    src: DevicePtr,
    dst: DevicePtr,
    rows: u32,
    cols: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(cols, 32), div_ceil(rows, 32), 1])
        .block([32, 8, 1])
        .arg_ptr(src)
        .arg_ptr(dst)
        .arg_u32(rows)
        .arg_u32(cols)
        .launch(stream)
}

/// 2026-09-25: Per-expert uint8 transpose for the MoE down_proj relayout:
/// reads expert `e`'s `[rows, cols]` matrix through `src_ptrs[e]` and writes
/// it as `[cols, rows]` through `dst_ptrs[e]`. A NULL entry in either table
/// skips that expert.
pub fn moe_transpose_u8_batched(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    src_ptrs: DevicePtr,
    dst_ptrs: DevicePtr,
    rows: u32,
    cols: u32,
    num_experts: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(cols, 32), div_ceil(rows, 32), num_experts])
        .block([32, 8, 1])
        .arg_ptr(src_ptrs)
        .arg_ptr(dst_ptrs)
        .arg_u32(rows)
        .arg_u32(cols)
        .launch(stream)
}

// 2026-09-25: Launchers for the transposed-weight decode MoE kernels
// (`kernels/gb10/common/moe_shared_expert_fused*_t.cu`). Their expert pointer
// tables and shared-expert weights must hold the transposed layouts, `[K/2, N]`
// for NVFP4 and `[K, N]` for FP8; the kernels take no layout flag.

// 2026-09-25: Block size of the transposed-layout decode kernels: one warp,
// one output per thread. Must equal `BLOCK_SIZE` in those `.cu` files, which
// index outputs as `blockIdx.x * BLOCK_SIZE + threadIdx.x`.
pub(super) const T_BLOCK: u32 = 32;

/// 2026-09-25: NVFP4 fused gate+up GEMV over transposed weights, for one
/// token.
#[allow(clippy::too_many_arguments)]
pub fn moe_expert_gate_up_shared_t(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    gate_packed_t_ptrs: DevicePtr,
    gate_scale_t_ptrs: DevicePtr,
    gate_scale2_vals: DevicePtr,
    gate_out: DevicePtr,
    up_packed_t_ptrs: DevicePtr,
    up_scale_t_ptrs: DevicePtr,
    up_scale2_vals: DevicePtr,
    up_out: DevicePtr,
    expert_indices: DevicePtr,
    sh_gate_t: &QuantizedWeight,
    sh_gate_out: DevicePtr,
    sh_up_t: &QuantizedWeight,
    sh_up_out: DevicePtr,
    n: u32,
    k: u32,
    top_k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, T_BLOCK), top_k + 1, 2])
        .block([T_BLOCK, 1, 1])
        .arg_ptr(input)
        .arg_ptr(gate_packed_t_ptrs)
        .arg_ptr(gate_scale_t_ptrs)
        .arg_ptr(gate_scale2_vals)
        .arg_ptr(gate_out)
        .arg_ptr(up_packed_t_ptrs)
        .arg_ptr(up_scale_t_ptrs)
        .arg_ptr(up_scale2_vals)
        .arg_ptr(up_out)
        .arg_ptr(expert_indices)
        .arg_ptr(sh_gate_t.weight)
        .arg_ptr(sh_gate_t.weight_scale)
        .arg_f32(sh_gate_t.weight_scale_2)
        .arg_ptr(sh_gate_out)
        .arg_ptr(sh_up_t.weight)
        .arg_ptr(sh_up_t.weight_scale)
        .arg_f32(sh_up_t.weight_scale_2)
        .arg_ptr(sh_up_out)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(top_k)
        .launch(stream)
}

/// 2026-09-25: NVFP4 fused SiLU+down GEMV over transposed weights, for one
/// token.
#[allow(clippy::too_many_arguments)]
pub fn moe_expert_silu_down_shared_t(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate_out: DevicePtr,
    up_out: DevicePtr,
    packed_t_ptrs: DevicePtr,
    scale_t_ptrs: DevicePtr,
    scale2_vals: DevicePtr,
    output: DevicePtr,
    expert_indices: DevicePtr,
    sh_gate_in: DevicePtr,
    sh_up_in: DevicePtr,
    sh_down_t: &QuantizedWeight,
    sh_down_out: DevicePtr,
    n: u32,
    k: u32,
    top_k: u32,
    stream: u64,
) -> Result<()> {
    let smem_bytes = (k as usize * std::mem::size_of::<f32>()) as u32;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, T_BLOCK), top_k + 1, 1])
        .block([T_BLOCK, 1, 1])
        .shared_mem(smem_bytes)
        .arg_ptr(gate_out)
        .arg_ptr(up_out)
        .arg_ptr(packed_t_ptrs)
        .arg_ptr(scale_t_ptrs)
        .arg_ptr(scale2_vals)
        .arg_ptr(output)
        .arg_ptr(expert_indices)
        .arg_ptr(sh_gate_in)
        .arg_ptr(sh_up_in)
        .arg_ptr(sh_down_t.weight)
        .arg_ptr(sh_down_t.weight_scale)
        .arg_f32(sh_down_t.weight_scale_2)
        .arg_ptr(sh_down_out)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(top_k)
        .launch(stream)
}
