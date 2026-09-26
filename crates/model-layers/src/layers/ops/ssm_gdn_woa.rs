// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Launchers for the GDN write-on-accept K=4 verify
//! (`kernels/gb10/common/gated_delta_rule_wy4_woa.cu`): the verify kernel, its
//! post-verdict fold, and the engaged-word clear.
//!
//! Owner: model-layers ops (GDN).
//! Invariants: none beyond the types.
//!
//! provenance-id: 526f6e616c6420522e205374657369616b

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

/// 2026-09-25: Write-on-accept K=4 verify (`gated_delta_rule_wy4_woa`). It
/// writes the output but no state, and stashes the per-row update terms for
/// [`gdn_wy4_fold`]. The state arguments are pointer tables.
#[allow(clippy::too_many_arguments)]
pub fn gdn_decode_wy4_woa(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    h_table: DevicePtr,
    query: DevicePtr,
    key: DevicePtr,
    value: DevicePtr,
    gate: DevicePtr,
    beta: DevicePtr,
    output: DevicePtr,
    stash: DevicePtr,
    batch_size: u32,
    num_k_heads: u32,
    num_v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    qk_stride: u32,
    v_stride: u32,
    gb_stride: u32,
    stash_seq_floats: u32,
    engaged_flag: DevicePtr,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(
        !h_table.is_null() && !stash.is_null() && !engaged_flag.is_null(),
        "gdn_decode_wy4_woa: null table/stash/flag"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([num_v_heads, batch_size, 1])
        .block([128, 1, 1])
        .shared_mem(k_dim * v_dim * 4)
        .arg_ptr(h_table)
        .arg_ptr(query)
        .arg_ptr(key)
        .arg_ptr(value)
        .arg_ptr(gate)
        .arg_ptr(beta)
        .arg_ptr(output)
        .arg_ptr(stash)
        .arg_u32(batch_size)
        .arg_u32(num_k_heads)
        .arg_u32(num_v_heads)
        .arg_u32(k_dim)
        .arg_u32(v_dim)
        .arg_u32(qk_stride)
        .arg_u32(v_stride)
        .arg_u32(gb_stride)
        .arg_u32(stash_seq_floats)
        .arg_ptr(engaged_flag)
        .launch(stream)
}

/// 2026-09-25: Post-verdict fold (`gated_delta_rule_wy4_fold`). When the
/// engaged word is set, it applies stashed rows `0..na_tab[b]` to H, reading
/// and writing H once. Otherwise the parent kernel ran, and a partial accept
/// (`na < k_rows`) restores `Hi(na - 1)` from `hi_tables`.
#[allow(clippy::too_many_arguments)]
pub fn gdn_wy4_fold(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    h_table: DevicePtr,
    stash: DevicePtr,
    na_tab: DevicePtr,
    hi_tables: DevicePtr,
    slab_entries: u32,
    engaged_flag: DevicePtr,
    k_rows: u32,
    batch_size: u32,
    num_k_heads: u32,
    num_v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    stash_seq_floats: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_v_heads, batch_size, 1])
        .block([128, 1, 1])
        .arg_ptr(h_table)
        .arg_ptr(stash)
        .arg_ptr(na_tab)
        .arg_ptr(hi_tables)
        .arg_u32(slab_entries)
        .arg_ptr(engaged_flag)
        .arg_u32(k_rows)
        .arg_u32(batch_size)
        .arg_u32(num_k_heads)
        .arg_u32(num_v_heads)
        .arg_u32(k_dim)
        .arg_u32(v_dim)
        .arg_u32(stash_seq_floats)
        .launch(stream)
}

/// 2026-09-25: Clear a layer's write-on-accept engaged word. The caller issues
/// it at the start of a batched verify that requested write-on-accept, on the
/// stream of the launches that follow.
pub fn gdn_wy4_flag_clear(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    flag: DevicePtr,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([1, 1, 1])
        .block([32, 1, 1])
        .arg_ptr(flag)
        .launch(stream)
}
