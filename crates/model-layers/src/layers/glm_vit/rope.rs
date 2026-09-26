// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Block-major patch ids, and the 2D axial RoPE tables built from them on the host.
//!
//! The 2×2 downsample treats each run of four consecutive tokens as one
//! spatial block (`merge.rs`), which is right only while the tokens are
//! block-major; a raster order still produces an embedding, a wrong one. The
//! pure functions here fix that order and are unit-tested.
//!
//! Owner: model-layers (vision).
//! Invariants: none beyond the types.

use super::GlmVit;

/// 2026-09-25: Per-token `(h_id, w_id)` in block-major order: the outer loops walk
/// the `merge × merge` blocks in raster order, the inner loops the positions
/// inside a block. This is the order
/// `vision_preprocess_glm::block_major_patch_index` gives the pixel patches.
pub fn block_major_position_ids(grid_h: usize, grid_w: usize, merge: usize) -> Vec<(u32, u32)> {
    let m = merge.max(1);
    let (bh, bw) = (grid_h / m, grid_w / m);
    let mut ids = Vec::with_capacity(bh * bw * m * m);
    for a in 0..bh {
        for c in 0..bw {
            for b in 0..m {
                for d in 0..m {
                    ids.push(((a * m + b) as u32, (c * m + d) as u32));
                }
            }
        }
    }
    ids
}

/// 2026-09-25: `(cos, sin)` tables of `[num_tokens, head_dim]` f32, in the layout
/// `glm_vit_qknorm_rope_deint` reads.
///
/// With `inv_freq` of `head_dim/4` entries, the per-token angle row is
/// `cat([h*inv_freq, w*inv_freq])` (length `head_dim/2`), written twice to
/// fill `head_dim`. The kernel rotates every dimension of the head.
pub fn build_axial_rope_tables(
    ids: &[(u32, u32)],
    inv_freq: &[f32],
    head_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let n_freq = inv_freq.len();
    let mut cos = vec![0.0f32; ids.len() * head_dim];
    let mut sin = vec![0.0f32; ids.len() * head_dim];
    for (t, &(h_id, w_id)) in ids.iter().enumerate() {
        for d in 0..half {
            // 2026-09-25: The first `n_freq` angles use the h id, the rest the w id.
            let angle = if d < n_freq {
                h_id as f32 * inv_freq[d]
            } else {
                w_id as f32 * inv_freq[d - n_freq]
            };
            let (s, c) = angle.sin_cos();
            cos[t * head_dim + d] = c;
            cos[t * head_dim + half + d] = c;
            sin[t * head_dim + d] = s;
            sin[t * head_dim + half + d] = s;
        }
    }
    (cos, sin)
}

impl GlmVit {
    /// 2026-09-25: Build this image's rope tables on the host and upload them as BF16.
    pub(super) fn upload_rope_tables(
        &self,
        grid_h: usize,
        grid_w: usize,
        cos_dst: metrale_gpu_runtime::gpu::DevicePtr,
        sin_dst: metrale_gpu_runtime::gpu::DevicePtr,
        gpu: &dyn metrale_gpu_runtime::gpu::GpuBackend,
        stream: u64,
    ) -> anyhow::Result<()> {
        let ids = block_major_position_ids(grid_h, grid_w, self.spatial_merge_size);
        let (cos, sin) = build_axial_rope_tables(&ids, &self.rope_inv_freq, self.head_dim);
        gpu.copy_h2d_async(&to_bf16_bytes(&cos), cos_dst, stream)?;
        gpu.copy_h2d_async(&to_bf16_bytes(&sin), sin_dst, stream)?;
        Ok(())
    }
}

/// 2026-09-25: Round-to-nearest-even f32 → BF16, as little-endian bytes.
pub(super) fn to_bf16_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 2);
    for &x in v {
        let bits = x.to_bits();
        let lsb = (bits >> 16) & 1;
        let rounded = bits.wrapping_add(0x7fff + lsb);
        out.extend_from_slice(&((rounded >> 16) as u16).to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-09-25: Four consecutive tokens are one spatial block, and the blocks
    /// walk in raster order.
    #[test]
    fn position_ids_are_block_major_not_raster() {
        let ids = block_major_position_ids(4, 4, 2);
        assert_eq!(ids.len(), 16);
        assert_eq!(&ids[0..4], &[(0, 0), (0, 1), (1, 0), (1, 1)]);
        assert_eq!(&ids[4..8], &[(0, 2), (0, 3), (1, 2), (1, 3)]);
        assert_eq!(&ids[8..12], &[(2, 0), (2, 1), (3, 0), (3, 1)]);
        assert_ne!(ids[2], (0, 2));
    }

    /// 2026-09-25: A non-square grid catches an h/w transposition that a square
    /// one hides.
    #[test]
    fn position_ids_handle_a_non_square_grid() {
        let ids = block_major_position_ids(26, 46, 2);
        assert_eq!(ids.len(), 26 * 46);
        assert_eq!(ids[0], (0, 0));
        assert_eq!(ids[23 * 4 - 1], (1, 45));
        assert_eq!(*ids.last().unwrap(), (25, 45));
        let mut seen: Vec<(u32, u32)> = ids.clone();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), ids.len(), "block-major must be a permutation");
    }

    /// 2026-09-25: The second half of each row repeats the first, and in the
    /// first half the h angles come before the w angles.
    #[test]
    fn axial_tables_duplicate_the_half_row() {
        let head_dim = 8;
        let inv_freq = vec![1.0f32, 0.5];
        let ids = [(3u32, 5u32)];
        let (cos, sin) = build_axial_rope_tables(&ids, &inv_freq, head_dim);
        let expect: [f32; 4] = [3.0 * 1.0, 3.0 * 0.5, 5.0 * 1.0, 5.0 * 0.5];
        for (d, &angle) in expect.iter().enumerate() {
            assert!((cos[d] - angle.cos()).abs() < 1e-6, "cos[{d}]");
            assert!((sin[d] - angle.sin()).abs() < 1e-6, "sin[{d}]");
            assert_eq!(cos[d], cos[head_dim / 2 + d]);
            assert_eq!(sin[d], sin[head_dim / 2 + d]);
        }
    }

    /// 2026-09-25: Token 0 sits at (0, 0), so every angle is zero: cos 1, sin 0.
    #[test]
    fn the_origin_token_is_an_identity_rotation() {
        let inv_freq: Vec<f32> = (0..16)
            .map(|k| 10_000f32.powf(-2.0 * k as f32 / 32.0))
            .collect();
        let ids = block_major_position_ids(32, 32, 2);
        let (cos, sin) = build_axial_rope_tables(&ids, &inv_freq, 64);
        for d in 0..64 {
            assert_eq!(cos[d], 1.0);
            assert_eq!(sin[d], 0.0);
        }
    }

    #[test]
    fn bf16_conversion_rounds_to_nearest_even() {
        assert_eq!(to_bf16_bytes(&[1.0]), 0x3f80u16.to_le_bytes().to_vec());
        // 2026-09-25: 0x3f80_8000 (1 + 2^-8) is a tie, which rounds to the even mantissa.
        let halfway = f32::from_bits(0x3f80_8000);
        assert_eq!(to_bf16_bytes(&[halfway]), 0x3f80u16.to_le_bytes().to_vec());
        let above = f32::from_bits(0x3f80_8001);
        assert_eq!(to_bf16_bytes(&[above]), 0x3f81u16.to_le_bytes().to_vec());
    }
}
