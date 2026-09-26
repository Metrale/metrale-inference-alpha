// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The `impl VisionEncoder` body. The child modules add methods to the
//! inherent impl, beside a few free functions such as `derive_max_patches`.
//!
//! Owner: model-layers (vision).
//! Invariants: none beyond the types.
//!
//! - `init`: `new`, the scratch allocation, and the free fn `derive_max_patches`
//! - `pos_embed`: `resample_pos_embed(_into)`, `build_rope_cossin(_into)`
//! - `patch_embed`: `patch_embed`, `patch_embed_batched`
//! - `vit_block`: `vit_gemm_bias`, `vit_block`, `vit_block_batched`; its child
//!   `vit_block/attn_gemm.rs` holds `vit_attention_gemm`
//! - `merger`: `apply_merger`
//! - `forward`: `forward`, `forward_batched`, the oversized-batch fallback
//! - `utils`: `gpu_copy_bf16`, `maybe_dump_buf`

mod forward;
pub(crate) mod init;
mod merger;
mod patch_embed;
mod pos_embed;
mod utils;
mod vit_block;

/// 2026-09-25: Convert an f32 to BF16 bits, rounding to nearest even. A NaN becomes
/// the canonical quiet NaN `0x7fc0`.
#[inline]
pub(super) fn f32_to_bf16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    if (bits & 0x7fff_ffff) > 0x7f80_0000 {
        return 0x7fc0;
    }
    let rounding = 0x7fff + ((bits >> 16) & 1);
    ((bits.wrapping_add(rounding)) >> 16) as u16
}
