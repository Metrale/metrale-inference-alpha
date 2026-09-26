// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Launchers for the batched GDN prefill kernels, which run
//! several streams' prefill chunks in one launch.
//!
//! Owner: model-layers ops (GDN).
//! Invariants: none beyond the types.

#![allow(unused_imports)]

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

// 2026-09-25: The caller's contract for every launcher in this file:
//   1. `h_state_ptrs` is a device array of `batch_size` per-stream h-state
//      pointers;
//   2. Q/K/V, gate/beta and the output are stacked per stream: stream `b`
//      starts at `b * seq_len * stride`;
//   3. every stream has the same `seq_len`, since the kernels take one.

/// 2026-09-25: Batched persistent prefill whose shared memory the caller
/// computes (`gated_delta_rule_prefill_wy64_batched` or
/// `gated_delta_rule_prefill_persistent_wy4_batched`).
#[allow(clippy::too_many_arguments)]
pub fn gdn_prefill_persistent_smem_batched(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    h_state_ptrs: DevicePtr,
    query: DevicePtr,
    key: DevicePtr,
    value: DevicePtr,
    gate: DevicePtr,
    beta: DevicePtr,
    output: DevicePtr,
    batch_size: u32,
    seq_len: u32,
    num_k_heads: u32,
    num_v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    qk_stride: u32,
    v_stride: u32,
    gb_stride: u32,
    smem: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_v_heads, batch_size, 1])
        .block([128, 1, 1])
        .shared_mem(smem)
        .arg_ptr(h_state_ptrs)
        .arg_ptr(query)
        .arg_ptr(key)
        .arg_ptr(value)
        .arg_ptr(gate)
        .arg_ptr(beta)
        .arg_ptr(output)
        .arg_u32(batch_size)
        .arg_u32(seq_len)
        .arg_u32(num_k_heads)
        .arg_u32(num_v_heads)
        .arg_u32(k_dim)
        .arg_u32(v_dim)
        .arg_u32(qk_stride)
        .arg_u32(v_stride)
        .arg_u32(gb_stride)
        .launch(stream)
}

/// 2026-09-25: Batched persistent prefill with H plus double-buffered k and q
/// in shared memory (`gated_delta_rule_prefill_persistent_batched`).
#[allow(clippy::too_many_arguments)]
pub fn gdn_prefill_persistent_batched(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    h_state_ptrs: DevicePtr,
    query: DevicePtr,
    key: DevicePtr,
    value: DevicePtr,
    gate: DevicePtr,
    beta: DevicePtr,
    output: DevicePtr,
    batch_size: u32,
    seq_len: u32,
    num_k_heads: u32,
    num_v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    qk_stride: u32,
    v_stride: u32,
    gb_stride: u32,
    stream: u64,
) -> Result<()> {
    let smem = k_dim * v_dim * 4 + 4 * k_dim * 4;
    KernelLaunch::new(gpu, kernel)
        .grid([num_v_heads, batch_size, 1])
        .block([128, 1, 1])
        .shared_mem(smem)
        .arg_ptr(h_state_ptrs)
        .arg_ptr(query)
        .arg_ptr(key)
        .arg_ptr(value)
        .arg_ptr(gate)
        .arg_ptr(beta)
        .arg_ptr(output)
        .arg_u32(batch_size)
        .arg_u32(seq_len)
        .arg_u32(num_k_heads)
        .arg_u32(num_v_heads)
        .arg_u32(k_dim)
        .arg_u32(v_dim)
        .arg_u32(qk_stride)
        .arg_u32(v_stride)
        .arg_u32(gb_stride)
        .launch(stream)
}

/// 2026-09-25: Batched split4 prefill
/// (`gated_delta_rule_prefill_split4_batched`).
#[allow(clippy::too_many_arguments)]
pub fn gdn_prefill_split4_batched(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    h_state_ptrs: DevicePtr,
    query: DevicePtr,
    key: DevicePtr,
    value: DevicePtr,
    gate: DevicePtr,
    beta: DevicePtr,
    output: DevicePtr,
    batch_size: u32,
    seq_len: u32,
    num_k_heads: u32,
    num_v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    qk_stride: u32,
    v_stride: u32,
    gb_stride: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_v_heads * 4, batch_size, 1])
        .block([32, 1, 1])
        .shared_mem(4 * k_dim * 4)
        .arg_ptr(h_state_ptrs)
        .arg_ptr(query)
        .arg_ptr(key)
        .arg_ptr(value)
        .arg_ptr(gate)
        .arg_ptr(beta)
        .arg_ptr(output)
        .arg_u32(batch_size)
        .arg_u32(seq_len)
        .arg_u32(num_k_heads)
        .arg_u32(num_v_heads)
        .arg_u32(k_dim)
        .arg_u32(v_dim)
        .arg_u32(qk_stride)
        .arg_u32(v_stride)
        .arg_u32(gb_stride)
        .launch(stream)
}
