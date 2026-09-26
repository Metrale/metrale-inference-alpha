// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Launchers for the device-side MoE expert LoRA folds:
//! `moe_lora_grouped_down` over the expert-sorted prefill rows and
//! `moe_lora_gather_bgmv` over the slot-major decode rows, plus the host packing
//! of their per-expert tables.
//!
//! Both read the expert routing (`expert_offsets` or `indices`) on the device.
//! The host loop `crate::lora::expert_apply::apply_expert_lora_sorted` needs a
//! host copy of `expert_offsets` and one launch per adapted expert, which a CUDA
//! graph cannot capture.
//!
//! Owner: model-layers ops.
//! Invariants:
//! - An expert whose `a_table` or `b_table` cell is `0`, or whose id is at or
//!   above the table length, folds nothing (`moe_lora_grouped_down.cu`,
//!   `moe_lora_gather_bgmv.cu`).

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::lora_delta::LoraKernels;

/// 2026-09-25: Device tables for one projection's expert LoRA pairs, indexed by
/// expert id: the expert-keyed counterpart of the slot-keyed
/// [`super::lora_delta::LoraRoute`]. `MoeLayer::build_expert_route` builds one
/// per projection (down, gate, up) at adapter install, and the router fold
/// builds a one-entry route whose expert 0 is the router pair.
///
/// `n_experts` is the table length, max adapted expert id + 1, not the layer's
/// expert count. The grouped launcher uses it as `grid.z`.
#[derive(Debug, Clone, Copy)]
pub struct MoeExpertRoute {
    /// 2026-09-25: `[n_experts]` u64 device array of `A_e` addresses (`0` = expert unadapted).
    pub a_table: DevicePtr,
    /// 2026-09-25: `[n_experts]` u64 device array of `B_e` addresses (`0` = expert unadapted).
    pub b_table: DevicePtr,
    /// 2026-09-25: `[n_experts]` f32 device array of per-expert `scale_e` (`0.0` where unadapted).
    pub scale_table: DevicePtr,
    pub n_experts: u32,
    /// 2026-09-25: Contraction dim of the shrink stage: the projection's input width.
    pub k_in: u32,
    /// 2026-09-25: Output dim of the expand stage: the projection's output width.
    pub n_out: u32,
    /// 2026-09-25: Padded rank: the contraction dim of the expand stage and the
    /// row stride of `B_e`.
    pub max_rank: u32,
}

/// 2026-09-25: Pack `(expert_id, a_addr, b_addr, scale)` entries into dense
/// tables indexed by expert id, with `0` / `0.0` at every unadapted id and
/// `n_experts = max expert_id + 1`. Returns `None` when `entries` is empty.
/// A duplicate expert id keeps its last entry.
pub fn pack_expert_tables(entries: &[(u16, u64, u64, f32)]) -> Option<ExpertTables> {
    let max_e = entries.iter().map(|(e, ..)| *e).max()?;
    let n = max_e as usize + 1;
    let mut a = vec![0u64; n];
    let mut b = vec![0u64; n];
    let mut scale = vec![0.0f32; n];
    for &(e, a_addr, b_addr, sc) in entries {
        let i = e as usize;
        a[i] = a_addr;
        b[i] = b_addr;
        scale[i] = sc;
    }
    Some(ExpertTables {
        a,
        b,
        scale,
        n_experts: n as u32,
    })
}

/// 2026-09-25: Host-side tables from [`pack_expert_tables`], before upload.
#[derive(Debug, Clone, PartialEq)]
pub struct ExpertTables {
    pub a: Vec<u64>,
    pub b: Vec<u64>,
    pub scale: Vec<f32>,
    pub n_experts: u32,
}

/// 2026-09-25: Fold `base_out[r] += scale_e * (x[row(r)] @ A_e^T) @ B_e^T` for
/// the sorted rows `r` in the window `[row_offset, row_end)`, where `e` is the
/// expert whose `expert_offsets` span holds `r`.
///
/// - `x`: BF16. With `x_gather == 0` (down) it is `[te, k_in]` in sorted order
///   and row `r` is read. With `x_gather == 1` (gate/up) it is the token-major
///   `[num_tokens, k_in]` input and row `sorted_token_ids[r]` is read.
/// - `base_out`: `[te, n_out]` BF16 in sorted order, folded in place.
/// - `expert_offsets`: device `[num_experts + 1]` i32 prefix sum.
/// - `sorted_token_ids`: device `[te]` i32, sorted row to token.
/// - `moe_row_adapter`: device `[num_tokens]` i32, where a row whose token maps
///   to a value `< 0` is skipped; `DevicePtr::NULL` folds every row.
/// - `xa`: fixed-address BF16 shrink scratch indexed by the local row
///   `r - row_offset`, so it needs `row_end - row_offset` rows of `max_rank`.
///
/// The argument order must match the kernel parameters in
/// `kernels/gb10/common/moe_lora_grouped_down.cu`: `cuLaunchKernel` does not
/// check types.
#[allow(clippy::too_many_arguments)]
pub fn moe_lora_grouped_down(
    gpu: &dyn GpuBackend,
    kernels: &LoraKernels,
    route: &MoeExpertRoute,
    x: DevicePtr,
    base_out: DevicePtr,
    expert_offsets: DevicePtr,
    sorted_token_ids: DevicePtr,
    moe_row_adapter: DevicePtr,
    xa: DevicePtr,
    row_offset: u32,
    row_end: u32,
    x_gather: u32,
    stream: u64,
) -> Result<()> {
    use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

    anyhow::ensure!(
        kernels.moe_down_shrink_k.0 != 0 && kernels.moe_down_expand_fold_k.0 != 0,
        "moe_lora_grouped_down kernels unresolved (module `moe_lora_grouped_down` missing \
         from the compiled kernel set — CUDA build required)"
    );
    // 2026-09-25: grid.y covers the window, not all of `te`: an expert's span
    // clipped to the window has at most `window` rows, and the kernels start each
    // expert's tiles at `max(m_start, row_offset)`.
    let window = row_end.saturating_sub(row_offset);
    let wc = div_ceil(window, MLG_M_TILE).max(1);

    // 2026-09-25: Shrink: `xa[r - row_offset] = x[row(r)] @ A_e^T`, stored as BF16.
    KernelLaunch::new(gpu, kernels.moe_down_shrink_k)
        .grid([div_ceil(route.max_rank, 4), wc, route.n_experts])
        .block([256, 1, 1])
        .arg_ptr(x)
        .arg_ptr(expert_offsets)
        .arg_ptr(sorted_token_ids)
        .arg_ptr(moe_row_adapter)
        .arg_ptr(route.a_table)
        .arg_ptr(xa)
        .arg_u32(route.n_experts)
        .arg_u32(route.max_rank)
        .arg_u32(route.k_in)
        .arg_u32(x_gather)
        .arg_u32(row_offset)
        .arg_u32(row_end)
        .launch(stream)?;

    // 2026-09-25: Expand and fold: `base_out[r] += scale_e * (xa[r - row_offset] @ B_e^T)`.
    KernelLaunch::new(gpu, kernels.moe_down_expand_fold_k)
        .grid([div_ceil(route.n_out, 4), wc, route.n_experts])
        .block([256, 1, 1])
        .arg_ptr(xa)
        .arg_ptr(expert_offsets)
        .arg_ptr(sorted_token_ids)
        .arg_ptr(moe_row_adapter)
        .arg_ptr(route.b_table)
        .arg_ptr(route.scale_table)
        .arg_ptr(base_out)
        .arg_u32(route.n_experts)
        .arg_u32(route.n_out)
        .arg_u32(route.max_rank)
        .arg_u32(row_offset)
        .arg_u32(row_end)
        .launch(stream)
}

/// 2026-09-25: `grid.y` of both grouped-fold kernels for the window
/// `[row_offset, row_end)`: `ceil((row_end - row_offset) / MLG_M_TILE)`, at
/// least 1. It is the same expression as the launcher's `wc`, kept separate so
/// tests can check the window math without a GPU.
pub fn grouped_down_wc(row_offset: u32, row_end: u32) -> u32 {
    use metrale_gpu_runtime::kernel_args::div_ceil;
    div_ceil(row_end.saturating_sub(row_offset), MLG_M_TILE).max(1)
}

/// 2026-09-25: Contiguous `[start, end)` windows of at most `cap` rows covering
/// `0..total_rows`. Panics when `cap` is 0.
pub fn grouped_down_windows(total_rows: u32, cap: u32) -> impl Iterator<Item = (u32, u32)> {
    assert!(cap > 0, "grouped LoRA scratch capacity must be nonzero");
    (0..total_rows)
        .step_by(cap as usize)
        .map(move |start| (start, start.saturating_add(cap).min(total_rows)))
}

/// 2026-09-25: Rows per grouped-fold tile. It must equal the `MLG_M_TILE`
/// `#define` in `moe_lora_grouped_down.cu`, or the host grid and the kernel's
/// tiles cover different rows.
pub const MLG_M_TILE: u32 = 64;

/// 2026-09-25: `(shrink, expand)` grids of the gather fold: `[ceil(out/4),
/// n_slots, 1]`, where `out` is `max_rank` for the shrink and `n_out` for the
/// expand. Each 256-thread block computes 4 outputs, 64 threads per output
/// (`GBGMV_N_PER_BLOCK` in `moe_lora_gather_bgmv.cu`).
pub fn gather_bgmv_grids(max_rank: u32, n_out: u32, n_slots: u32) -> ([u32; 3], [u32; 3]) {
    use metrale_gpu_runtime::kernel_args::div_ceil;
    (
        [div_ceil(max_rank, 4), n_slots, 1],
        [div_ceil(n_out, 4), n_slots, 1],
    )
}

/// 2026-09-25: The token that owns flat `(token, slot)` row `row`: the gather
/// kernels' `row / top_k`.
pub fn gather_row_token(row: u32, top_k: u32) -> u32 {
    row / top_k
}

/// 2026-09-25: Fold `base_out[row] += scale_e * (x[x_row] @ A_e^T) @ B_e^T` for
/// every flat `(token, slot)` row in `[0, n_slots)`, where `e = indices[row]`.
/// This is the unsorted counterpart of [`moe_lora_grouped_down`]: the expert
/// comes from `indices` instead of an `expert_offsets` span.
///
/// - `x`: BF16. With `x_gather == 0` (down) it is `[n_slots, k_in]` and row
///   `row` is read. With `x_gather == 1` (gate/up) it is `[num_tokens, k_in]`
///   and row `row / top_k` is read.
/// - `base_out`: `[n_slots, n_out]` BF16, folded in place.
/// - `indices`: device `[n_slots]` u32 expert id per row.
/// - `row_adapter`: device `[num_tokens]` i32, where a token `< 0` is skipped;
///   `DevicePtr::NULL` folds every row.
/// - `xa`: fixed-address `[n_slots, max_rank]` BF16 shrink scratch.
///
/// The argument order must match the kernel parameters in
/// `kernels/gb10/common/moe_lora_gather_bgmv.cu`: `cuLaunchKernel` does not
/// check types.
#[allow(clippy::too_many_arguments)]
pub fn moe_lora_gather_bgmv(
    gpu: &dyn GpuBackend,
    kernels: &LoraKernels,
    route: &MoeExpertRoute,
    x: DevicePtr,
    base_out: DevicePtr,
    indices: DevicePtr,
    row_adapter: DevicePtr,
    xa: DevicePtr,
    n_slots: u32,
    top_k: u32,
    x_gather: u32,
    stream: u64,
) -> Result<()> {
    use metrale_gpu_runtime::kernel_args::KernelLaunch;

    anyhow::ensure!(
        kernels.moe_gather_shrink_k.0 != 0 && kernels.moe_gather_expand_fold_k.0 != 0,
        "moe_lora_gather_bgmv kernels unresolved (module `moe_lora_gather_bgmv` missing \
         from the compiled kernel set — CUDA build required)"
    );
    let (shrink_grid, expand_grid) = gather_bgmv_grids(route.max_rank, route.n_out, n_slots);

    // 2026-09-25: Shrink: `xa[row] = x[x_row] @ A_e^T`, stored as BF16.
    KernelLaunch::new(gpu, kernels.moe_gather_shrink_k)
        .grid(shrink_grid)
        .block([256, 1, 1])
        .arg_ptr(x)
        .arg_ptr(indices)
        .arg_ptr(row_adapter)
        .arg_ptr(route.a_table)
        .arg_ptr(xa)
        .arg_u32(n_slots)
        .arg_u32(top_k)
        .arg_u32(route.n_experts)
        .arg_u32(route.max_rank)
        .arg_u32(route.k_in)
        .arg_u32(x_gather)
        .launch(stream)?;

    // 2026-09-25: Expand and fold: `base_out[row] += scale_e * (xa[row] @ B_e^T)`.
    KernelLaunch::new(gpu, kernels.moe_gather_expand_fold_k)
        .grid(expand_grid)
        .block([256, 1, 1])
        .arg_ptr(xa)
        .arg_ptr(indices)
        .arg_ptr(row_adapter)
        .arg_ptr(route.b_table)
        .arg_ptr(route.scale_table)
        .arg_ptr(base_out)
        .arg_u32(n_slots)
        .arg_u32(top_k)
        .arg_u32(route.n_experts)
        .arg_u32(route.n_out)
        .arg_u32(route.max_rank)
        .launch(stream)
}

#[cfg(test)]
#[path = "moe_lora_grouped_tests.rs"]
mod tests;
