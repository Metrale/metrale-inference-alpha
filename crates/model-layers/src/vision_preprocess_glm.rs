// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: GLM image geometry: `smart_resize` to a canvas aligned to
//! `patch_size × spatial_merge_size`, an aspect-preserving fit that never
//! upscales, a black pad to the canvas, and the 2×2-block-major patch order.
//! `vision_preprocess` uses it when `VisionConfig::block_major_patches` is set.
//!
//! Owner: model-layers (vision input).
//! Invariants: none beyond the types.

use image::RgbImage;
use metrale_config::VisionConfig;

/// 2026-09-25: Snap each side to the nearest multiple of `factor`, then rescale
/// by the area bounds if the snapped canvas violates them.
///
/// `factor = patch_size * merge_size`, and the bounds are areas in pixels.
/// Returns `(h_bar, w_bar)`, both multiples of `factor` and at least one
/// factor each. The initial snap rounds (so a 360-pixel side at factor 28
/// becomes 364, and the canvas may be larger than the input), the over-budget
/// branch floors, and the under-budget branch ceils.
pub fn smart_resize(h: u32, w: u32, factor: u32, min_pixels: u64, max_pixels: u64) -> (u32, u32) {
    let f = factor.max(1) as f64;
    let (hf, wf) = (h.max(1) as f64, w.max(1) as f64);
    let snap = |v: f64| ((v / f).round() as u32).max(1) * factor;
    let (mut h_bar, mut w_bar) = (snap(hf), snap(wf));

    let area = |a: u32, b: u32| u64::from(a) * u64::from(b);
    if area(h_bar, w_bar) > max_pixels && max_pixels > 0 {
        let beta = (hf * wf / max_pixels as f64).sqrt();
        h_bar = (((hf / beta) / f).floor() as u32).max(1) * factor;
        w_bar = (((wf / beta) / f).floor() as u32).max(1) * factor;
    } else if area(h_bar, w_bar) < min_pixels {
        let beta = (min_pixels as f64 / (hf * wf)).sqrt();
        h_bar = (((hf * beta) / f).ceil() as u32).max(1) * factor;
        w_bar = (((wf * beta) / f).ceil() as u32).max(1) * factor;
    }
    (h_bar, w_bar)
}

/// 2026-09-25: `(min_pixels, max_pixels)` for `smart_resize`, as pixel areas.
///
/// GLM states its budget in merged tokens (`min_image_tokens`,
/// `max_image_tokens`), converted as `tokens × factor²`: one merged token
/// covers one `factor × factor` pixel block. `max_pixels` is the least of:
/// - the encoder's capacity, from `derive_max_patches`, the function `GlmVit`
///   sizes its buffers with;
/// - the declared token budget, when nonzero;
/// - `operator_max`, when given;
///
/// and never below one merged token. With no bound the encoder holds 6400
/// patches (`FALLBACK_MAX_PATCHES`), while GLM's default budget of 8000 merged
/// tokens is 32000 patches, so the encoder term binds.
pub fn pixel_bounds(vcfg: &VisionConfig, operator_max: Option<usize>) -> (u64, u64) {
    let factor = (vcfg.patch_size * vcfg.spatial_merge_size).max(1) as u64;
    let per_token = factor * factor;
    let min_pixels = vcfg.min_image_tokens as u64 * per_token;
    let model_max = vcfg.max_image_tokens as u64 * per_token;
    let (encoder_patches, _) = crate::layers::vision_encoder::enc_impl::init::derive_max_patches(
        vcfg.max_pixels.or(operator_max),
        vcfg.patch_size,
    );
    let encoder_max =
        encoder_patches as u64 * (vcfg.patch_size.max(1) * vcfg.patch_size.max(1)) as u64;
    let mut max_pixels = encoder_max;
    if model_max > 0 {
        max_pixels = max_pixels.min(model_max);
    }
    if let Some(p) = operator_max.filter(|&p| p > 0) {
        max_pixels = max_pixels.min(p as u64);
    }
    (min_pixels, max_pixels.max(per_token))
}

/// 2026-09-25: Fit `img` inside the `smart_resize` canvas preserving aspect,
/// then zero-pad right/bottom to the canvas exactly. Returns the padded canvas.
///
/// - The pad is black pixels applied before normalisation, so padded cells
///   normalise to `(0 - mean)/std`, not `0.0`.
/// - The fit never upscales: image c of the reference fixtures (640x360) sits
///   unresampled in a 644x364 canvas.
///
/// `tests/glm_vision_preprocess_pin.rs` compares the output with the
/// reference's `pixel_values` for the fixture images.
pub fn resize_and_pad(
    img: &RgbImage,
    vcfg: &VisionConfig,
    operator_max: Option<usize>,
) -> RgbImage {
    let factor = (vcfg.patch_size * vcfg.spatial_merge_size) as u32;
    let (min_pixels, max_pixels) = pixel_bounds(vcfg, operator_max);
    let (target_h, target_w) =
        smart_resize(img.height(), img.width(), factor, min_pixels, max_pixels);

    // 2026-09-25: The largest scale that keeps both sides inside the canvas,
    // capped at 1.0: `smart_resize` rounds to the grid, so its canvas is often a
    // few pixels larger than the input.
    let scale = (target_h as f64 / img.height().max(1) as f64)
        .min(target_w as f64 / img.width().max(1) as f64)
        .min(1.0);
    let content_h = ((img.height() as f64 * scale).round() as u32).clamp(1, target_h);
    let content_w = ((img.width() as f64 * scale).round() as u32).clamp(1, target_w);

    // 2026-09-25: When the content size equals the source size, the source
    // pixels are used as they are rather than passed through the filter.
    let fitted = if (content_h, content_w) == (img.height(), img.width()) {
        img.clone()
    } else {
        image::imageops::resize(
            img,
            content_w,
            content_h,
            image::imageops::FilterType::CatmullRom,
        )
    };
    if content_h == target_h && content_w == target_w {
        return fitted;
    }
    let mut canvas = RgbImage::new(target_w, target_h);
    image::imageops::replace(&mut canvas, &fitted, 0, 0);
    canvas
}

/// 2026-09-25: Index of the patch at `(patch_row, patch_col)` in the
/// 2×2-block-major stream the GLM tower expects.
///
/// `((h_block * w_blocks + w_block) * merge + row_in_block) * merge +
/// col_in_block`. The tower's 2×2 downsample folds four consecutive tokens
/// into one block, which is correct only in this order.
pub fn block_major_patch_index(
    patch_row: usize,
    patch_col: usize,
    grid_w: usize,
    merge: usize,
) -> usize {
    let m = merge.max(1);
    let w_blocks = grid_w / m;
    (((patch_row / m) * w_blocks + patch_col / m) * m + patch_row % m) * m + patch_col % m
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-09-25: Fixture image c: 640×360 at factor 28 becomes a 644×364
    /// canvas, a 46×26 patch grid, and 23×13 = 299 merged tokens.
    #[test]
    fn smart_resize_reproduces_the_golden_non_square_canvas() {
        let (h, w) = smart_resize(360, 640, 28, 16 * 784, 8000 * 784);
        assert_eq!((h, w), (364, 644));
        assert_eq!((h / 14) * (w / 14), 26 * 46);
        assert_eq!((h / 28) * (w / 28), 299);
    }

    /// 2026-09-25: A square 448 input, like fixture images a and b, is already
    /// aligned: the canvas is itself, 32×32 patches.
    #[test]
    fn an_aligned_square_is_left_alone() {
        assert_eq!(smart_resize(448, 448, 28, 16 * 784, 8000 * 784), (448, 448));
    }

    /// 2026-09-25: Over budget floors onto the grid, so the result is inside the
    /// bound.
    #[test]
    fn an_over_budget_image_is_floored_inside_the_bound() {
        let max = 100u64 * 784;
        let (h, w) = smart_resize(4000, 4000, 28, 16 * 784, max);
        assert_eq!(h % 28, 0);
        assert_eq!(w % 28, 0);
        assert!(
            u64::from(h) * u64::from(w) <= max,
            "{h}x{w} exceeds the {max}px budget"
        );
    }

    /// 2026-09-25: Under budget ceils up to the minimum, which is in tokens, so a
    /// 1×1 image becomes a multi-patch grid.
    #[test]
    fn a_tiny_image_is_raised_to_the_minimum_token_count() {
        let min = 16u64 * 784;
        let (h, w) = smart_resize(1, 1, 28, min, 8000 * 784);
        assert!(u64::from(h) * u64::from(w) >= min, "{h}x{w} under {min}");
        assert_eq!((h % 28, w % 28), (0, 0));
    }

    fn glm_cfg() -> VisionConfig {
        VisionConfig {
            patch_size: 14,
            spatial_merge_size: 2,
            min_image_tokens: 16,
            max_image_tokens: 8000,
            ..VisionConfig::default()
        }
    }

    /// 2026-09-25: A tighter operator bound lowers the budget; a looser one raises
    /// it only to the encoder's ceiling (`CEILING_MAX_PATCHES`), never to the
    /// declared 8000 merged tokens. `pixel_bounds` sizes the encoder term from
    /// the same operator bound here (`vcfg.max_pixels` is `None`).
    #[test]
    fn the_operator_bound_moves_the_budget_only_as_far_as_the_encoder_follows() {
        let vcfg = glm_cfg();
        let (_, tight) = pixel_bounds(&vcfg, Some(512 * 512));
        assert!(
            tight <= 512 * 512,
            "a tighter flag must lower the bound: {tight}"
        );
        assert_eq!(tight / (14 * 14), 1337, "512x512 at patch 14 is 1337 rows");

        let (_, loose) = pixel_bounds(&vcfg, Some(64 * 1024 * 1024));
        let (_, none) = pixel_bounds(&vcfg, None);
        assert!(
            loose > none,
            "a looser flag may raise the bound: {loose} vs {none}"
        );
        // 2026-09-25: Only to the encoder's ceiling, not to the declared budget.
        assert_eq!(
            loose / (14 * 14),
            16_384,
            "the encoder's CEILING_MAX_PATCHES"
        );
        assert!(
            loose < 8000 * 784,
            "never as far as the declared token budget"
        );
    }

    /// 2026-09-25: The declared budget exceeds the encoder's default capacity, so
    /// the host bound comes down to the encoder's.
    #[test]
    fn the_encoder_capacity_caps_the_checkpoints_own_token_budget() {
        let vcfg = glm_cfg();
        let (_, resolved) = pixel_bounds(&vcfg, None);
        let declared = 8000u64 * 784;
        assert!(
            resolved < declared,
            "GLM declares {declared}px (8000 merged tokens); the encoder's default \
             allocation cannot hold that, so the host bound must be lower, got {resolved}"
        );
        // 2026-09-25: `FALLBACK_MAX_PATCHES`, what `derive_max_patches` returns
        // when nothing bounds the image.
        assert_eq!(resolved / (14 * 14), 6400);
        // 2026-09-25: A square image at that bound yields no more patches than
        // the encoder holds.
        let side = (resolved as f64).sqrt() as u64;
        assert!((side / 14) * (side / 14) <= 6400);
    }

    /// 2026-09-25: A checkpoint that declares no token budget still gets the
    /// encoder's.
    #[test]
    fn a_checkpoint_without_a_token_budget_still_gets_a_bound() {
        let vcfg = VisionConfig {
            max_image_tokens: 0,
            ..glm_cfg()
        };
        let (_, resolved) = pixel_bounds(&vcfg, None);
        assert_eq!(resolved / (14 * 14), 6400);
        let (_, with_flag) = pixel_bounds(&vcfg, Some(512 * 512));
        assert!(with_flag <= 512 * 512);
        assert_eq!(with_flag / (14 * 14), 1337);
    }

    /// 2026-09-25: The canvas may be larger than the input, and the content is
    /// padded, not stretched, to fill it.
    #[test]
    fn a_canvas_larger_than_the_input_pads_rather_than_upscales() {
        let vcfg = glm_cfg();
        let img = RgbImage::new(640, 360);
        let out = resize_and_pad(&img, &vcfg, None);
        assert_eq!(
            (out.height(), out.width()),
            (364, 644),
            "the smart_resize canvas"
        );
        // 2026-09-25: A blank image has no pixel values to check;
        // `tests/glm_vision_preprocess_pin.rs` checks them against the reference.
        assert_eq!((out.height() / 14) * (out.width() / 14), 26 * 46);
    }

    /// 2026-09-25: Block-major is a permutation of raster over the same grid, and
    /// the first four indices are one 2×2 block rather than one row of four.
    #[test]
    fn block_major_index_is_a_permutation_grouping_2x2_blocks() {
        let (gh, gw, m) = (26usize, 46usize, 2usize);
        let mut seen = vec![usize::MAX; gh * gw];
        for ph in 0..gh {
            for pw in 0..gw {
                let i = block_major_patch_index(ph, pw, gw, m);
                assert!(i < gh * gw, "index {i} out of range");
                assert_eq!(seen[i], usize::MAX, "index {i} used twice");
                seen[i] = ph * gw + pw;
            }
        }
        // 2026-09-25: Tokens 0..4 are the 2×2 block at the origin, in (row, col)
        // order.
        assert_eq!(&seen[0..4], &[0, 1, gw, gw + 1]);
        // 2026-09-25: Token 4 starts the next block along w, not the next column.
        assert_eq!(seen[4], 2);
    }

    /// 2026-09-25: Raster and block-major agree only on a grid one block wide;
    /// on a wider grid they differ from the third patch on.
    #[test]
    fn block_major_differs_from_raster_on_any_real_grid() {
        // 2026-09-25: One block wide (grid_w = 2): the two orders coincide.
        for (row, col) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
            assert_eq!(
                block_major_patch_index(row, col, 2, 2),
                row * 2 + col,
                "1 block wide must equal raster at ({row}, {col})"
            );
        }
        // 2026-09-25: Two blocks wide: patch (0, 2) is raster index 2 but
        // block-major 4, because tokens 2 and 3 are the first block's second row.
        assert_eq!(block_major_patch_index(0, 2, 4, 2), 4);
        assert_eq!(block_major_patch_index(1, 0, 4, 2), 2);
    }
}
