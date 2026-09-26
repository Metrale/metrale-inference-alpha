// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Launchers for the WY-chunkwise verify decodes at K = 4 and
//! K >= 5 (contiguous and pointer-table forms), and the 2-token conv1d update.
//!
//! Owner: model-layers ops (GDN).
//! Invariants: none beyond the types.

// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(unused_imports)]

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::layers::moe;
use crate::weight_map::{DenseWeight, Fp8DenseWeight, Fp8Weight, QuantizedWeight};

use super::*;

/// 2026-09-25: WY-chunkwise 4-token verify decode (`gated_delta_rule_wy4`):
/// the [`gdn_decode_wy2`] scheme for four tokens, with three intermediate
/// states.
#[allow(clippy::too_many_arguments)]
pub fn gdn_decode_wy4(
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
    h_state_inter2: DevicePtr,
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
    // true: device pointer tables, one entry per sequence; any batched verify
    // passes true.
    state_is_table: bool,
    stream: u64,
) -> Result<()> {
    // 2026-09-25: `ensure!`, because a `debug_assert!` compiles out of release
    // builds, where the misaddressing would be silent.
    anyhow::ensure!(
        state_is_table || batch_size == 1,
        "gdn_decode_wy4: contiguous state addressing is only valid at \
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
        .arg_ptr(h_state_inter2)
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

/// 2026-09-25: WY-chunkwise verify decode for K tokens, contiguous form: the
/// launch for `gated_delta_rule_wy5` .. `wy16` (and their `_f16` twins) from
/// `gated_delta_rule_wyn.cu`, and for `gated_delta_rule_wy17`. K is
/// compile-time in the kernel, so the `kernel` handle selects it. The kernel
/// writes Hi_0..Hi_{K-2} and the final H.
///
/// Hi_t is at `h_state_inter_base + t * inter_stride_floats`, plus the
/// per-(b, vh) offset; the stride is in the kernel's h element size.
#[allow(clippy::too_many_arguments)]
pub fn gdn_decode_wyn(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    h_state: DevicePtr,
    query: DevicePtr,
    key: DevicePtr,
    value: DevicePtr,
    gate: DevicePtr,
    beta: DevicePtr,
    output: DevicePtr,
    h_state_inter_base: DevicePtr,
    inter_stride_floats: u32,
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
    // 2026-09-25: The contiguous form offsets both h_state and the
    // intermediates by `(b * num_v_heads + vh) * hv`, which is right only at
    // batch_size == 1. Batched verify goes through [`gdn_decode_wyn_table`].
    anyhow::ensure!(
        batch_size == 1,
        "gdn_decode_wyn: contiguous state addressing is only valid at batch_size==1 \
         (got {batch_size}); port the wy4 `state_is_table` pointer-table form first"
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
        .arg_ptr(h_state_inter_base)
        .arg_u32(inter_stride_floats)
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

/// 2026-09-25: Cross-sequence batched-verify launch of the wyN pointer-table
/// twins (`gated_delta_rule_wy{5..16}_table` and their `_f16_table` forms).
/// `state_is_table` is compiled into the symbol, so the argument list is the
/// same as [`gdn_decode_wyn`], but the state arguments are tables:
/// - `h_tables`: the layer's staged table slice; slab 0 holds one
///   per-sequence H base pointer per entry (`VERIFY_WY_TABLE_SEQS` entries).
/// - `hi_tables`: slab 1 (Hi0); the following Hi slabs are
///   `VERIFY_WY_TABLE_SEQS` entries apart.
/// - `slab_entry_stride` must be `VERIFY_WY_TABLE_SEQS as u32`: the kernel
///   reads the contiguous form's stride argument as the number of pointer
///   entries between Hi slabs.
///
/// The tables are staged by `upload_verify_wy_tables` (model-engine
/// `verify_e2.rs`).
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub fn gdn_decode_wyn_table(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    h_tables: DevicePtr,
    query: DevicePtr,
    key: DevicePtr,
    value: DevicePtr,
    gate: DevicePtr,
    beta: DevicePtr,
    output: DevicePtr,
    hi_tables: DevicePtr,
    slab_entry_stride: u32,
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
    anyhow::ensure!(
        !h_tables.is_null() && !hi_tables.is_null(),
        "gdn_decode_wyn_table: staged pointer tables are required (NULL table \
         means upload_verify_wy_tables declined — run the per-sequence loop)"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([num_v_heads, batch_size, 1])
        .block([128, 1, 1])
        .arg_ptr(h_tables)
        .arg_ptr(query)
        .arg_ptr(key)
        .arg_ptr(value)
        .arg_ptr(gate)
        .arg_ptr(beta)
        .arg_ptr(output)
        .arg_ptr(hi_tables)
        .arg_u32(slab_entry_stride)
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

/// 2026-09-25: Conv1d update + SiLU for two tokens per sequence in one launch
/// (`causal_conv1d_update_chunk2`), one thread per channel. The conv state
/// after the first token goes to `conv_state_intermediate`, for rollback.
#[allow(clippy::too_many_arguments)]
pub fn conv1d_update_chunk2(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    conv_state: DevicePtr,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    conv_state_intermediate: DevicePtr,
    d_inner: u32,
    d_conv: u32,
    batch_size: u32,
    stream: u64,
) -> Result<()> {
    let bias_ptr = DevicePtr::NULL;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(d_inner, 256), batch_size, 1])
        .block([256, 1, 1])
        .arg_ptr(conv_state)
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(bias_ptr)
        .arg_ptr(output)
        .arg_ptr(conv_state_intermediate)
        .arg_u32(batch_size)
        .arg_u32(d_inner)
        .arg_u32(d_conv)
        .launch(stream)
}
