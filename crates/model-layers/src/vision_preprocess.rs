// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: CPU-side image preprocessing for the vision encoder. Decodes a
//! base64 image, resizes it onto a grid of `patch_size × spatial_merge_size`
//! cells, normalises it with the checkpoint's mean and std, and lays the
//! patches out as a flat `f32` tensor.
//!
//! Owner: model-layers (vision input).
//! Invariants: none beyond the types.

use anyhow::{Context, Result, bail};
use image::{DynamicImage, ImageDecoder, ImageFormat, ImageReader, Limits};
use metrale_config::VisionConfig;

/// 2026-09-25: Normalisation stats of the video path (`video_preprocess.rs`).
/// The image path reads `VisionConfig::image_mean` / `image_std` instead.
pub(crate) const MEAN: [f32; 3] = [0.5, 0.5, 0.5];
pub(crate) const STD: [f32; 3] = [0.5, 0.5, 0.5];

/// 2026-09-25: Long-side cap applied only when the caller passes no `max_pixels`
/// bound.
const FALLBACK_MAX_DIM: u32 = 1280;

/// 2026-09-25: Long-side ceiling applied with or without a `max_pixels` bound:
/// `max_pixels` is an area, so on its own it admits an unbounded long side (a
/// 1×N strip). The ceiling is applied before grid snapping.
const ABS_MAX_DIM: u32 = 4096;

/// 2026-09-25: Decoder limit on either side, checked against the image header
/// before pixels are decoded.
const DECODE_MAX_SIDE: u32 = 16_384;

/// 2026-09-25: Decoder allocation limit for one image, below the `image` crate's
/// default of 512 MiB (which the crate documents as non-strict). 192 MiB admits
/// an 8000×8000 RGB8 buffer (183 MiB).
const DECODE_MAX_ALLOC: u64 = 192 * 1024 * 1024;

/// 2026-09-25: Split a base64 `data:` URI (or a bare base64 string) into its
/// declared MIME type and decoded bytes. The MIME is empty when the input has
/// no `data:` header. The video path shares it and names the MIME in its
/// errors.
pub(crate) fn decode_data_uri_bytes(data_uri: &str) -> Result<(String, Vec<u8>)> {
    let (mime, b64) = if let Some(pos) = data_uri.find(",base64,") {
        (
            data_uri[..pos].trim_start_matches("data:").to_string(),
            &data_uri[pos + 8..],
        )
    } else if let Some(rest) = data_uri.strip_prefix("data:") {
        match rest.find(',') {
            Some(p) => (
                rest[..p].trim_end_matches(";base64").to_string(),
                &rest[p + 1..],
            ),
            None => (String::new(), data_uri),
        }
    } else {
        (String::new(), data_uri)
    };

    let bytes = base64::Engine::decode(&base64::engine::general_purpose::STANDARD, b64.trim())
        .context("base64 decode failed")?;
    Ok((mime, bytes))
}

/// 2026-09-25: Decode a base64 data URI or raw base64 string into a `DynamicImage`.
fn decode_image(data_uri: &str) -> Result<DynamicImage> {
    let (_mime, bytes) = decode_data_uri_bytes(data_uri)?;

    let fmt = image::guess_format(&bytes).unwrap_or(ImageFormat::Jpeg);
    // 2026-09-25: Decode through `ImageReader` so the limits are ours. The free
    // function `load_from_memory_with_format` applies `Limits::default()`: a
    // non-strict 512 MiB allocation limit and no dimension cap. The dimension
    // limit rejects a header that declares, say, 65535×65535 before any buffer
    // is reserved.
    let mut reader = ImageReader::new(std::io::Cursor::new(&bytes));
    reader.set_format(fmt);
    let mut limits = Limits::default();
    limits.max_image_width = Some(DECODE_MAX_SIDE);
    limits.max_image_height = Some(DECODE_MAX_SIDE);
    limits.max_alloc = Some(DECODE_MAX_ALLOC);
    reader.limits(limits);

    // 2026-09-25: Apply EXIF orientation. A camera often stores the sensor's raw
    // pixels and records the upright rotation in a tag, so decoding without it
    // hands the model a rotated image. The tag is read from the decoder, hence
    // `into_decoder`; a format without orientation reports `NoTransforms` and
    // the image is left as decoded.
    let mut decoder = reader.into_decoder().context("image decode failed")?;
    let orientation = decoder
        .orientation()
        .unwrap_or(image::metadata::Orientation::NoTransforms);
    let mut img = DynamicImage::from_decoder(decoder).context("image decode failed")?;
    if orientation != image::metadata::Orientation::NoTransforms {
        tracing::debug!("applying EXIF orientation {orientation:?}");
        img.apply_orientation(orientation);
    }
    Ok(img)
}

/// 2026-09-25: Reject a vision config whose geometry cannot drive the
/// preprocessor.
///
/// `parse_vision_config` reports a missing key as `0`. A zero `patch_size` or
/// `spatial_merge_size` would reach a division by zero or a 0×0 target, and a
/// zero `temporal_patch_size` an empty patch. This fails with a named error
/// and assumes no default.
fn validate_geometry(vcfg: &VisionConfig) -> Result<()> {
    if vcfg.patch_size == 0 {
        bail!("vision_config.patch_size is 0 (missing or invalid in the checkpoint's config.json)");
    }
    if vcfg.spatial_merge_size == 0 {
        bail!("vision_config.spatial_merge_size is 0 (missing or invalid in config.json)");
    }
    if vcfg.temporal_patch_size == 0 {
        bail!("vision_config.temporal_patch_size is 0 (missing or invalid in config.json)");
    }
    Ok(())
}

/// 2026-09-25: Compute the target (H, W):
/// - Both sides are positive multiples of `grid_unit = patch_size × spatial_merge_size`.
/// - With a `max_pixels` bound, the target area is at most
///   `max(max_pixels, grid_unit²)`; the bound replaces [`FALLBACK_MAX_DIM`]
///   rather than combining with it. Without one, the long side is scaled to
///   at most [`FALLBACK_MAX_DIM`] before snapping.
/// - The long side is scaled to at most [`ABS_MAX_DIM`] before snapping.
/// - Aspect ratio is kept up to rounding to the grid.
/// - The continuous scale never upscales; snapping may round a side up by
///   less than half a grid unit.
///
/// `max_pixels` is an area in pixels. The serve resolves it from
/// `--vision-max-pixels`, else `METRALE_VISION_MAX_PIXELS`, else the
/// checkpoint's processor config. The video path calls this too.
pub(crate) fn target_size_for(
    orig_h: u32,
    orig_w: u32,
    grid_unit: u32,
    max_pixels: Option<usize>,
) -> (u32, u32) {
    target_size_with_max_pixels(orig_h, orig_w, grid_unit, max_pixels)
}

fn target_size_with_max_pixels(
    orig_h: u32,
    orig_w: u32,
    grid_unit: u32,
    max_pixels: Option<usize>,
) -> (u32, u32) {
    let long_side = orig_h.max(orig_w) as f32;
    let area = (orig_h as f32) * (orig_w as f32);
    let bound_scale = match max_pixels.filter(|&p| p > 0) {
        Some(p) => ((p as f32) / area).sqrt(),
        None => (FALLBACK_MAX_DIM as f32) / long_side,
    };
    let abs_scale = (ABS_MAX_DIM as f32) / long_side;
    let scale = bound_scale.min(abs_scale).min(1.0);
    let mut target_h =
        ((orig_h as f32 * scale / grid_unit as f32).round() as u32).max(1) * grid_unit;
    let mut target_w =
        ((orig_w as f32 * scale / grid_unit as f32).round() as u32).max(1) * grid_unit;

    // 2026-09-25: Nearest-grid rounding can push the area past the bound. Shrink
    // one grid unit at a time, on the axis that keeps the closer source aspect
    // ratio, stopping at one grid cell, the smallest target.
    if let Some(max_pixels) = max_pixels.filter(|&p| p > 0) {
        let grid_area = u64::from(grid_unit) * u64::from(grid_unit);
        let max_area = (max_pixels as u64).max(grid_area);
        let area = |h: u32, w: u32| u64::from(h) * u64::from(w);
        let aspect_error = |h: u32, w: u32| {
            if orig_h == 0 {
                0.0
            } else {
                ((w as f64 / h as f64) - (orig_w as f64 / orig_h as f64)).abs()
            }
        };

        while area(target_h, target_w) > max_area {
            let shorter_h = target_h.checked_sub(grid_unit).filter(|&h| h >= grid_unit);
            let shorter_w = target_w.checked_sub(grid_unit).filter(|&w| w >= grid_unit);
            match (shorter_h, shorter_w) {
                (Some(h), Some(w)) => {
                    let h_error = aspect_error(h, target_w);
                    let w_error = aspect_error(target_h, w);
                    if h_error < w_error
                        || (h_error == w_error && area(h, target_w) >= area(target_h, w))
                    {
                        target_h = h;
                    } else {
                        target_w = w;
                    }
                }
                (Some(h), None) => target_h = h,
                (None, Some(w)) => target_w = w,
                (None, None) => break,
            }
        }
    }
    (target_h, target_w)
}

/// 2026-09-25: Preprocess one base64-encoded image with no `max_pixels` bound.
///
/// Returns:
/// - `pixels`: flat `f32` tensor shaped `[P, C × T × H_p × W_p]` where:
///   - `P = (H/patch_size) × (W/patch_size)` — number of patches
///   - `C = 3` channels, `T = temporal_patch_size` (image duplicated), `H_p = W_p = patch_size`
/// - `grid_h`: number of patches along height
/// - `grid_w`: number of patches along width
pub fn preprocess_image(data_uri: &str, vcfg: &VisionConfig) -> Result<(Vec<f32>, usize, usize)> {
    preprocess_image_with_max_pixels(data_uri, vcfg, None)
}

/// 2026-09-25: [`preprocess_image`] with an optional area bound `max_pixels`
/// (see `target_size_for`).
pub fn preprocess_image_with_max_pixels(
    data_uri: &str,
    vcfg: &VisionConfig,
    max_pixels: Option<usize>,
) -> Result<(Vec<f32>, usize, usize)> {
    // 2026-09-25: Before anything divides by them.
    validate_geometry(vcfg)?;
    let img = decode_image(data_uri)?;
    let img = img.to_rgb8();
    let (orig_w, orig_h) = (img.width(), img.height());

    let grid_unit = (vcfg.patch_size * vcfg.spatial_merge_size) as u32;

    // 2026-09-25: The resize policy follows `VisionConfig::block_major_patches`.
    // GLM fits the content inside a `smart_resize` canvas and zero-pads the
    // remainder; otherwise the image is resized onto the snapped canvas.
    let img = if vcfg.block_major_patches {
        crate::vision_preprocess_glm::resize_and_pad(&img, vcfg, max_pixels)
    } else {
        let (th, tw) = target_size_with_max_pixels(orig_h, orig_w, grid_unit, max_pixels);
        // 2026-09-25: CatmullRom is the `image` crate's bicubic filter.
        image::imageops::resize(&img, tw, th, image::imageops::FilterType::CatmullRom)
    };
    let (th, tw) = (img.height(), img.width());

    let ps = vcfg.patch_size;
    let tp = vcfg.temporal_patch_size;
    let grid_h = (th as usize) / ps;
    let grid_w = (tw as usize) / ps;
    let num_patches = grid_h * grid_w;
    let patch_dim = 3 * tp * ps * ps;
    let mut pixels = vec![0.0f32; num_patches * patch_dim];

    // 2026-09-25: Mean and std come from the checkpoint's `VisionConfig`: 0.5 on
    // every channel for the Qwen family (`VisionConfig::default`), CLIP values
    // for GLM (`parse_glm5_next`).
    let (mean, std) = (vcfg.image_mean, vcfg.image_std);

    // 2026-09-25: The temporal axis is filled by repeating the image `tp` times.
    // Layout: `[P, C, T, Hp, Wp]`, stored as `[P, C*T*Hp*Wp]` row-major.
    //
    // Patch order is 2×2-block-major when `block_major_patches` (GLM), whose
    // downsample folds four consecutive tokens into one block; raster order
    // otherwise.
    for ph in 0..grid_h {
        for pw in 0..grid_w {
            let patch_idx = if vcfg.block_major_patches {
                crate::vision_preprocess_glm::block_major_patch_index(
                    ph,
                    pw,
                    grid_w,
                    vcfg.spatial_merge_size,
                )
            } else {
                ph * grid_w + pw
            };
            for c in 0..3usize {
                for t in 0..tp {
                    for py in 0..ps {
                        for px in 0..ps {
                            let pixel_y = ph * ps + py;
                            let pixel_x = pw * ps + px;
                            let raw =
                                img.get_pixel(pixel_x as u32, pixel_y as u32)[c] as f32 / 255.0;
                            let norm = (raw - mean[c]) / std[c];
                            let off = c * (tp * ps * ps) + t * (ps * ps) + py * ps + px;
                            pixels[patch_idx * patch_dim + off] = norm;
                        }
                    }
                }
            }
        }
    }

    Ok((pixels, grid_h, grid_w))
}

#[cfg(test)]
#[path = "vision_preprocess_tests.rs"]
mod tests;
