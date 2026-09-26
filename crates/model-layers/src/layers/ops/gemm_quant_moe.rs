// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Launchers for MoE routing and grouped GEMMs: the fused NVFP4
//! gate top-k, the tile work-list builder, and the FP8, W8A8 and BF16 grouped
//! GEMMs.
//!
//! Owner: model-layers ops.
//! Invariants: none beyond the types.

use super::*;

/// 2026-09-25: Fused gate GEMV and top-k for one token (kernel
/// `moe_gate_topk_fused`): `gate[num_experts] = A[K] @ B_gate[num_experts,
/// K]^T` with an NVFP4 gate weight, then the top-k expert indices and weights
/// `exp(logit - max)` over those k, divided by their sum when `normalize` is
/// nonzero. One 256-thread CTA; `K` BF16 values of dynamic shared memory hold
/// the input.
#[allow(clippy::too_many_arguments)]
pub fn moe_gate_topk_fused(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    gate_weight: &QuantizedWeight,
    expert_indices: DevicePtr,
    expert_weights: DevicePtr,
    num_experts: u32,
    k: u32,
    top_k: u32,
    normalize: u32,
    stream: u64,
) -> Result<()> {
    let smem_bytes = k as usize * 2;
    KernelLaunch::new(gpu, kernel)
        .grid([1, 1, 1])
        .block([256, 1, 1])
        .shared_mem(smem_bytes as u32)
        .arg_ptr(input)
        .arg_ptr(gate_weight.weight)
        .arg_ptr(gate_weight.weight_scale)
        .arg_f32(gate_weight.weight_scale_2)
        .arg_ptr(expert_indices)
        .arg_ptr(expert_weights)
        .arg_u32(num_experts)
        .arg_u32(k)
        .arg_u32(top_k)
        .arg_u32(normalize)
        .launch(stream)
}

/// 2026-09-25: Build the compacted work-list for the grouped-GEMM grid: one
/// item per (expert, m-tile, n-tile) of every expert that has rows and a
/// non-NULL weight pointer, experts in ascending order. One CTA; thread 0
/// writes `worklist[2w] = expert`, `worklist[2w + 1] = (m-tile index << 6) |
/// n-tile index`, and `total_tiles[0] = w`.
///
/// `n_tiles` is `ceil(N / 64)` and `m_tile` is 128, as `PM4_N_TILE` and
/// `PM4_M_TILE` in `moe_fp8_grouped_gemm.cu`. The n-tile index has 6 bits, so
/// `n_tiles` must be at most 64; the kernel asserts it. `expert_offsets` is
/// `[num_experts + 1]` and `weight_ptrs` `[num_experts]`; an expert whose
/// pointer is NULL gets no items.
///
/// No event orders this before the grouped GEMM that reads the list, so
/// launch that GEMM on the same `stream`.
#[allow(clippy::too_many_arguments)]
pub fn moe_build_tile_worklist(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    expert_offsets: DevicePtr,
    weight_ptrs: DevicePtr,
    worklist: DevicePtr,
    total_tiles: DevicePtr,
    num_experts: u32,
    n_tiles: u32,
    m_tile: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([1, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(expert_offsets)
        .arg_ptr(weight_ptrs)
        .arg_ptr(worklist)
        .arg_ptr(total_tiles)
        .arg_u32(num_experts)
        .arg_u32(n_tiles)
        .arg_u32(m_tile)
        .launch(stream)
}

/// 2026-09-25: FP8 grouped GEMM for sorted MoE prefill over the work-list from
/// [`moe_build_tile_worklist`], which must run first on the same `stream`.
///
/// The kernel strides over the work-list by `gridDim.x`, so the grid is
/// `max_tiles` (the caller's upper bound on the tile count) clamped to
/// `1..=16384`. Extra CTAs exit at once; a grid smaller than the list is
/// slower but still covers every tile.
///
/// `input` `[total_tokens, K]` BF16; `weight_ptrs[e]` points at `[N, K]` FP8
/// and `scale_ptrs[e]` at `[N/128, K/128]` FP32; `output`
/// `[total_expanded, N]` BF16; `expert_offsets` `[num_experts + 1]`;
/// `sorted_token_ids` `[total_expanded]`, or NULL to read input row `i` for
/// output row `i`.
#[allow(clippy::too_many_arguments)]
pub fn moe_fp8_grouped_gemm(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight_ptrs: DevicePtr,
    scale_ptrs: DevicePtr,
    output: DevicePtr,
    expert_offsets: DevicePtr,
    sorted_token_ids: DevicePtr,
    num_experts: u32,
    n: u32,
    k: u32,
    worklist: DevicePtr,
    total_tiles: DevicePtr,
    max_tiles: u32,
    stream: u64,
) -> Result<()> {
    const MAX_GRID_CTAS: u32 = 16384;
    let grid_ctas = max_tiles.clamp(1, MAX_GRID_CTAS);
    // 2026-09-25: The block must equal `PM4_THREADS` of the kernel this target
    // compiles: 512 (16 warps) in `kernels/strix-hip/common/moe_fp8_grouped_gemm.cu`
    // for the `metrale_hip` build, and 256 in
    // `kernels/gb10/common/moe_fp8_grouped_gemm.cu` everywhere else.
    #[cfg(metrale_hip)]
    let block = [512u32, 1, 1];
    #[cfg(not(metrale_hip))]
    let block = [256u32, 1, 1];
    KernelLaunch::new(gpu, kernel)
        .grid([grid_ctas, 1, 1])
        .block(block)
        .arg_ptr(input)
        .arg_ptr(weight_ptrs)
        .arg_ptr(scale_ptrs)
        .arg_ptr(output)
        .arg_ptr(expert_offsets)
        .arg_ptr(sorted_token_ids)
        .arg_u32(num_experts)
        .arg_u32(n)
        .arg_u32(k)
        .arg_ptr(worklist)
        .arg_ptr(total_tiles)
        .launch(stream)
}

/// 2026-09-25: W8A8 grouped MoE GEMM on a dense grid
/// `(ceil(N/64), max_m_tiles, num_experts)`. `a_fp8` comes from
/// [`per_token_group_quant_fp8`]; `a_scale` (`[total_tokens, K/128]` FP32) and
/// the per-block `b_scale` (`scale_ptrs[e]` → `[N/128, K/128]` FP32) are
/// applied in an FP32 epilogue per 128-element K-group. `weight_ptrs[e]`
/// points at `[N, K]` FP8; `output` is `[total_expanded, N]` BF16;
/// `sorted_token_ids` is `[total_expanded]` or NULL.
#[allow(clippy::too_many_arguments)]
pub fn moe_w8a8_grouped_gemm(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a_fp8: DevicePtr,
    a_scale: DevicePtr,
    weight_ptrs: DevicePtr,
    scale_ptrs: DevicePtr,
    output: DevicePtr,
    expert_offsets: DevicePtr,
    sorted_token_ids: DevicePtr,
    num_experts: u32,
    n: u32,
    k: u32,
    max_m_tiles: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 64), max_m_tiles, num_experts])
        .block([128, 1, 1])
        .arg_ptr(a_fp8)
        .arg_ptr(a_scale)
        .arg_ptr(weight_ptrs)
        .arg_ptr(scale_ptrs)
        .arg_ptr(output)
        .arg_ptr(expert_offsets)
        .arg_ptr(sorted_token_ids)
        .arg_u32(num_experts)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// 2026-09-25: [`moe_w8a8_grouped_gemm`] over the work-list from
/// [`moe_build_tile_worklist`] (kernel `moe_w8a8_grouped_gemm_pm4`, same
/// module), which must run first on the same `stream`. The grid follows the
/// rule of [`moe_fp8_grouped_gemm`]: `max_tiles` clamped to `1..=16384`.
#[allow(clippy::too_many_arguments)]
pub fn moe_w8a8_grouped_gemm_pm4(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a_fp8: DevicePtr,
    a_scale: DevicePtr,
    weight_ptrs: DevicePtr,
    scale_ptrs: DevicePtr,
    output: DevicePtr,
    expert_offsets: DevicePtr,
    sorted_token_ids: DevicePtr,
    num_experts: u32,
    n: u32,
    k: u32,
    worklist: DevicePtr,
    total_tiles: DevicePtr,
    max_tiles: u32,
    stream: u64,
) -> Result<()> {
    const MAX_GRID_CTAS: u32 = 16384;
    let grid_ctas = max_tiles.clamp(1, MAX_GRID_CTAS);
    // 2026-09-25: 256 threads, `W8PM4_THREADS` in
    // `kernels/gb10/common/moe_w8a8_grouped_gemm.cu`.
    KernelLaunch::new(gpu, kernel)
        .grid([grid_ctas, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(a_fp8)
        .arg_ptr(a_scale)
        .arg_ptr(weight_ptrs)
        .arg_ptr(scale_ptrs)
        .arg_ptr(output)
        .arg_ptr(expert_offsets)
        .arg_ptr(sorted_token_ids)
        .arg_u32(num_experts)
        .arg_u32(n)
        .arg_u32(k)
        .arg_ptr(worklist)
        .arg_ptr(total_tiles)
        .launch(stream)
}

/// 2026-09-25: BF16 grouped GEMM for sorted MoE prefill, without scales:
/// `input` `[total_tokens, K]` BF16, `weight_ptrs[e]` → `[N, K]` BF16,
/// `output` `[total_expanded, N]` BF16, `sorted_token_ids` `[total_expanded]`
/// or NULL. The MoE layer's long-prefill path uses it when its experts are
/// BF16, which the loader sets up when `METRALE_FP8_DEQUANT_MOE_TO_BF16=1`
/// dequantizes FP8 experts at load (`qwen35/load_layers.rs`).
#[allow(clippy::too_many_arguments)]
pub fn moe_bf16_grouped_gemm(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight_ptrs: DevicePtr,
    output: DevicePtr,
    expert_offsets: DevicePtr,
    sorted_token_ids: DevicePtr,
    num_experts: u32,
    n: u32,
    k: u32,
    max_m_tiles: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 64), max_m_tiles, num_experts])
        .block([128, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight_ptrs)
        .arg_ptr(output)
        .arg_ptr(expert_offsets)
        .arg_ptr(sorted_token_ids)
        .arg_u32(num_experts)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}
