// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Launchers for the 3-token GDN verify decode and the
//! WY-chunkwise 2- and 3-token verify decodes.
//!
//! Owner: model-layers ops (GDN).
//! Invariants: none beyond the types.

#![allow(unused_imports)]

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::layers::moe;
use crate::weight_map::{DenseWeight, Fp8DenseWeight, Fp8Weight, QuantizedWeight};

use super::*;

/// 2026-09-25: Three tokens through the GDN recurrence in one launch
/// (`gated_delta_rule_chunk3`). The states after the first and second tokens
/// go to `h_state_inter0` and `h_state_inter1`, for rollback when drafts are
/// rejected.
#[allow(clippy::too_many_arguments)]
pub fn gdn_decode_chunk3(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    h_state: DevicePtr,
    query: DevicePtr,
    key: DevicePtr,
    value: DevicePtr,
    gate: DevicePtr,
    beta: DevicePtr,
    output: DevicePtr,
    h_state_inter0: DevicePtr,
    h_state_inter1: DevicePtr,
    batch_size: u32,
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
        .grid([num_v_heads, batch_size, 1])
        .block([128, 1, 1])
        .arg_ptr(h_state)
        .arg_ptr(query)
        .arg_ptr(key)
        .arg_ptr(value)
        .arg_ptr(gate)
        .arg_ptr(beta)
        .arg_ptr(output)
        .arg_ptr(h_state_inter0)
        .arg_ptr(h_state_inter1)
        .arg_u32(batch_size)
        .arg_u32(num_k_heads)
        .arg_u32(num_v_heads)
        .arg_u32(k_dim)
        .arg_u32(v_dim)
        .arg_u32(qk_stride)
        .arg_u32(v_stride)
        .arg_u32(gb_stride)
        .launch(stream)
}

/// 2026-09-25: WY-chunkwise 2-token verify decode (`gated_delta_rule_wy2`).
/// One pass over H computes both tokens' `H^T k` products, the WY correction
/// is applied, and a second pass updates H and writes the intermediate state.
#[allow(clippy::too_many_arguments)]
pub fn gdn_decode_wy2(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    h_state: DevicePtr,
    query: DevicePtr,
    key: DevicePtr,
    value: DevicePtr,
    gate: DevicePtr,
    beta: DevicePtr,
    output: DevicePtr,
    h_state_intermediate: DevicePtr,
    batch_size: u32,
    num_k_heads: u32,
    num_v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    qk_stride: u32,
    v_stride: u32,
    gb_stride: u32,
    // 2026-09-25: false: the state arguments are contiguous bases indexed by
    // `(b * num_v_heads + vh)`, which assumes every intermediate shares
    // h_state's per-sequence stride. The SSM pool does not lay them out that
    // way, so the contiguous form is correct only at batch_size == 1.
    // true: device pointer tables, one entry per sequence.
    state_is_table: bool,
    stream: u64,
) -> Result<()> {
    // 2026-09-25: `ensure!`, because a `debug_assert!` compiles out of release
    // builds, where the misaddressing would be silent.
    anyhow::ensure!(
        state_is_table || batch_size == 1,
        "gdn_decode_wy2: contiguous state addressing is only valid at \
         batch_size==1 (got {batch_size}) — the intermediate's pool stride is \
         num_intermediates x h_state's, so sequence 1's Hi0 would land on \
         sequence 0's Hi1. Stage pointer tables and pass state_is_table=true."
    );
    KernelLaunch::new(gpu, kernel)
        .grid([num_v_heads, batch_size, 1])
        .block([128, 1, 1])
        .arg_ptr(h_state)
        .arg_ptr(query)
        .arg_ptr(key)
        .arg_ptr(value)
        .arg_ptr(gate)
        .arg_ptr(beta)
        .arg_ptr(output)
        .arg_ptr(h_state_intermediate)
        .arg_u32(batch_size)
        .arg_u32(num_k_heads)
        .arg_u32(num_v_heads)
        .arg_u32(k_dim)
        .arg_u32(v_dim)
        .arg_u32(qk_stride)
        .arg_u32(v_stride)
        .arg_u32(gb_stride)
        .arg_u32(u32::from(state_is_table))
        .launch(stream)
}

/// 2026-09-25: WY-chunkwise 3-token verify decode (`gated_delta_rule_wy3`):
/// the [`gdn_decode_wy2`] scheme for three tokens, with two intermediate
/// states.
#[allow(clippy::too_many_arguments)]
pub fn gdn_decode_wy3(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    h_state: DevicePtr,
    query: DevicePtr,
    key: DevicePtr,
    value: DevicePtr,
    gate: DevicePtr,
    beta: DevicePtr,
    output: DevicePtr,
    h_state_inter0: DevicePtr,
    h_state_inter1: DevicePtr,
    batch_size: u32,
    num_k_heads: u32,
    num_v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    qk_stride: u32,
    v_stride: u32,
    gb_stride: u32,
    // 2026-09-25: false: the state arguments are contiguous bases indexed by
    // `(b * num_v_heads + vh)`, which assumes every intermediate shares
    // h_state's per-sequence stride. The SSM pool does not lay them out that
    // way, so the contiguous form is correct only at batch_size == 1.
    // true: device pointer tables, one entry per sequence.
    state_is_table: bool,
    stream: u64,
) -> Result<()> {
    // 2026-09-25: `ensure!`, because a `debug_assert!` compiles out of release
    // builds, where the misaddressing would be silent.
    anyhow::ensure!(
        state_is_table || batch_size == 1,
        "gdn_decode_wy3: contiguous state addressing is only valid at \
         batch_size==1 (got {batch_size}) — the intermediates' pool stride is \
         num_intermediates x h_state's, so sequence 1's Hi0 would land on \
         sequence 0's Hi1. Stage pointer tables and pass state_is_table=true."
    );
    KernelLaunch::new(gpu, kernel)
        .grid([num_v_heads, batch_size, 1])
        .block([128, 1, 1])
        .arg_ptr(h_state)
        .arg_ptr(query)
        .arg_ptr(key)
        .arg_ptr(value)
        .arg_ptr(gate)
        .arg_ptr(beta)
        .arg_ptr(output)
        .arg_ptr(h_state_inter0)
        .arg_ptr(h_state_inter1)
        .arg_u32(batch_size)
        .arg_u32(num_k_heads)
        .arg_u32(num_v_heads)
        .arg_u32(k_dim)
        .arg_u32(v_dim)
        .arg_u32(qk_stride)
        .arg_u32(v_stride)
        .arg_u32(gb_stride)
        .arg_u32(u32::from(state_is_table))
        .launch(stream)
}
