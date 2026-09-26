// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The vision-token counts the geometry leg predicts, from an
//! image's size, the patch and merge sizes, and the serve's declared area
//! bound.
//!
//! Owner: bench, vision.
//! Invariants: none beyond the types.

/// 2026-09-26: Snap a side to the grid as the engine's `target_size_for` does
/// at scale 1: `max(round(side / grid), 1) * grid`. `f32::round` rounds half
/// away from zero, so 336 snaps to 352 at grid 32
/// (`the_rounding_mode_is_pinned`).
pub fn snap(side: u32, grid: u32) -> u32 {
    (((side as f32) / (grid as f32)).round() as u32).max(1) * grid
}

/// 2026-09-26: Merged vision tokens for a `w × h` image at native size.
///
/// Both sides snap to `patch × merge`, then each `patch × patch` square is one
/// patch and each `merge × merge` block of patches one token. At patch 16,
/// merge 2 a 448×448 image is 784 patches and 196 tokens; 196 was measured
/// against a live server on 2026-08-14.
pub fn expected_vision_tokens(w: u32, h: u32, patch: u32, merge: u32) -> u32 {
    let grid = patch * merge;
    let sw = snap(w, grid);
    let sh = snap(h, grid);
    (sw / patch) * (sh / patch) / (merge * merge)
}

/// 2026-09-26: The engine's long-side clamp when no area bound is given; a copy
/// of `FALLBACK_MAX_DIM` in `model-layers/src/vision_preprocess.rs`.
pub const FALLBACK_MAX_DIM: u32 = 1280;

/// 2026-09-26: The long-side ceiling applied with or without an area bound; a
/// copy of `ABS_MAX_DIM` in `model-layers/src/vision_preprocess.rs`. It binds
/// when an area bound admits a long side over 4096, such as a 64×8192 strip.
pub const ABS_MAX_DIM: u32 = 4096;

/// 2026-09-26: The `(w, h)` the server encodes for a `w × h` source.
///
/// Copies the scale and grid rounding of
/// `metrale_model_layers::vision_preprocess::target_size_for`, which is
/// crate-private and in a crate metrale-bench does not link. It omits the
/// engine's last step, which shrinks a side one grid unit at a time while the
/// rounded area is over `max_pixels`. `max_pixels` is an area, as with
/// `--vision-max-pixels`; `None` applies the 1280 px fallback clamp.
pub fn served_size(w: u32, h: u32, grid: u32, max_pixels: Option<u64>) -> (u32, u32) {
    let long_side = w.max(h) as f32;
    let area = (w as f32) * (h as f32);
    let bound_scale = match max_pixels.filter(|&p| p > 0) {
        // 2026-09-26: A declared area bound replaces the fallback clamp.
        Some(p) => ((p as f32) / area).sqrt(),
        None => (FALLBACK_MAX_DIM as f32) / long_side,
    };
    let abs_scale = (ABS_MAX_DIM as f32) / long_side;
    let scale = bound_scale.min(abs_scale).min(1.0);
    let tw = ((w as f32 * scale / grid as f32).round() as u32).max(1) * grid;
    let th = ((h as f32 * scale / grid as f32).round() as u32).max(1) * grid;
    (tw, th)
}

/// 2026-09-26: Merged vision tokens for `w × h` under a declared area bound.
///
/// A serve with a bound downscales a larger image and answers
/// (`preprocess_image_with_max_pixels`), so the prediction takes the bound as
/// input. `max_pixels == 0` means no bound was declared and returns
/// [`expected_vision_tokens`], on the premise that the checkpoint's own bound
/// (16777216 px in the Qwen3.6 preprocessor configs) is above every fixture.
/// It is not `served_size(.., None)`, which applies the 1280 px fallback clamp.
pub fn expected_vision_tokens_bounded(
    w: u32,
    h: u32,
    patch: u32,
    merge: u32,
    max_pixels: u64,
) -> u32 {
    if max_pixels == 0 {
        return expected_vision_tokens(w, h, patch, merge);
    }
    let (tw, th) = served_size(w, h, patch * merge, Some(max_pixels));
    (tw / patch) * (th / patch) / (merge * merge)
}

#[cfg(test)]
#[path = "geometry_tests.rs"]
mod geometry_tests;
