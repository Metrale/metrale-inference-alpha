// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Pure dispatch decisions for `forward_k2`; no launches.
//!
//! Owner: model-layers (MoE).
//! Invariants: none beyond the types.

/// 2026-09-25: Block width for the originals-layout NVFP4 batch2 MoE kernels:
/// 128 (one warp per output pair) or 256 (two warps per output pair, joined
/// through shared memory). The kernels derive the split from `blockDim.x`, so
/// both widths are valid and the switch at hidden_size 3072 is a tuning choice.
pub(crate) fn batch2_block_width(hidden_size: usize) -> u32 {
    if hidden_size >= 3072 { 256 } else { 128 }
}

/// 2026-09-25: True when `forward_k2` must send the rows to `forward_batched`:
/// E8M0 routed experts cannot use the GROUP_SIZE-16 batch2 kernel.
pub(crate) fn k2_e8m0_needs_per_token(scale_kind: crate::weight_map::WeightQuantFormat) -> bool {
    matches!(scale_kind, crate::weight_map::WeightQuantFormat::Mxfp4E8m0)
}
