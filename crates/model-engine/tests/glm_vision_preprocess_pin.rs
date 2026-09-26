// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The GLM image preprocessor against the reference processor's
//! `pixel_values`, for the three `glm_vit_golden` fixture images, on the CPU.
//!
//! Owner: model-engine tests.
//! Invariants: none beyond the types.
//!
//! `glm_vit_golden` feeds the encoder the reference's `pixel_values`, so it does
//! not test how they are made: the `smart_resize` canvas, scaling or padding,
//! the pad value, the normalisation constants and the 2x2-block-major patch
//! order. This test does.
//!
//! `vision_preprocess_glm::resize_and_pad` implements the two rules the
//! reference showed:
//!
//! * the pad is black pixels applied before normalisation, so padded cells
//!   hold `(0 - mean)/std`, not `0.0`;
//! * the fit never upscales: image c's 640x360 content sits unresampled in a
//!   644x364 canvas.
//!
//! Run (no GPU, no kernels needed):
//!
//! ```text
//! METRALE_GLM_VISION=1 \
//! GLM_VIT_GOLDEN_DIR=/path/to/ws2-vision \
//! cargo test -p metrale-model-engine --test glm_vision_preprocess_pin -- --ignored --nocapture
//! ```

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};

#[path = "glm_vit_golden/fixtures.rs"]
mod fixtures;
use fixtures::read_npy_f32;

/// 2026-09-25: Ceiling on the max-abs error against the reference's `pixel_values`.
///
/// It is far below one 1/255 pixel step, which is at least 0.014 in normalised
/// units (the largest GLM `image_std` is 0.2758), so any disagreement about a
/// pixel's value fails.
const MAX_ABS: f32 = 1e-5;

struct Fixture {
    name: &'static str,
    png: &'static str,
    grid_h: usize,
    grid_w: usize,
}

const FIXTURES: [Fixture; 3] = [
    Fixture {
        name: "a_noise_448x448",
        png: "a_noise_448x448.png",
        grid_h: 32,
        grid_w: 32,
    },
    Fixture {
        name: "b_geometric_448x448",
        png: "b_geometric_448x448.png",
        grid_h: 32,
        grid_w: 32,
    },
    // 2026-09-25: 640x360 is not a multiple of 28 (patch 14 x merge 2), so this is
    // the only fixture whose canvas differs from the image and gets a pad. It is
    // also the only non-square grid, where an h/w swap in the block-major
    // index shows.
    Fixture {
        name: "c_gradient_640x360",
        png: "c_gradient_640x360.png",
        grid_h: 26,
        grid_w: 46,
    },
];

fn golden_dir() -> PathBuf {
    PathBuf::from(
        std::env::var("GLM_VIT_GOLDEN_DIR").expect("set GLM_VIT_GOLDEN_DIR (see module doc)"),
    )
}

/// 2026-09-25: The PNG at `path` as a `data:` URI, the input
/// `preprocess_image_with_max_pixels` decodes.
fn png_data_uri(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let b64 = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &bytes);
    Ok(format!("data:image/png;base64,{b64}"))
}

#[test]
#[ignore = "requires the reference fixtures in GLM_VIT_GOLDEN_DIR and METRALE_GLM_VISION=1"]
fn glm_preprocess_matches_the_reference_pixel_values() -> Result<()> {
    ensure!(
        metrale_config::glm_vision_enabled(),
        "set METRALE_GLM_VISION=1: with the gate off `parse_glm5_next` leaves \
         config.vision at None and there is no GLM vision config to test"
    );
    let root = golden_dir();
    let gold = root.join("golden");
    let cfg = metrale_config::parse_config(&std::fs::read_to_string(root.join("config.json"))?)
        .context("parse the checkpoint's own config.json")?;
    let v = cfg
        .vision
        .context("config.vision is None even with the gate on")?;
    ensure!(
        v.block_major_patches,
        "the GLM parser must declare block-major patch order"
    );
    println!(
        "mean={:?} std={:?} patch={} merge={} min/max_image_tokens={}/{}",
        v.image_mean,
        v.image_std,
        v.patch_size,
        v.spatial_merge_size,
        v.min_image_tokens,
        v.max_image_tokens
    );

    let mut failures = Vec::new();
    println!("\n| image | grid (h x w) | patches | max-abs | mean-abs | pad cells |");
    println!("|---|---|---|---|---|---|");
    for f in &FIXTURES {
        let uri = png_data_uri(&gold.join(f.png))?;
        let (pixels, grid_h, grid_w) =
            metrale_model_layers::vision_preprocess::preprocess_image_with_max_pixels(
                &uri, &v, None,
            )
            .with_context(|| format!("preprocess {}", f.name))?;
        // 2026-09-25: The grid is checked before the pixels, so a smart_resize
        // or no-upscale error reports as a geometry error.
        ensure!(
            (grid_h, grid_w) == (f.grid_h, f.grid_w),
            "{}: Metrale Engine produced a {grid_h}x{grid_w} patch grid, the reference a {}x{}",
            f.name,
            f.grid_h,
            f.grid_w
        );

        let (shape, expect) = read_npy_f32(&gold.join(format!("{}_pixel_values.npy", f.name)))?;
        let patch_dim = 3 * v.temporal_patch_size * v.patch_size * v.patch_size;
        ensure!(
            shape == vec![grid_h * grid_w, patch_dim] && pixels.len() == expect.len(),
            "{}: golden pixel_values {shape:?} vs Metrale Engine {} floats",
            f.name,
            pixels.len()
        );

        // 2026-09-25: A padded cell normalises to -mean/std. `pad_cells` counts
        // the reference's cells at that value and is only printed; a wrong pad
        // fails the max-abs comparison.
        let pad_ch: Vec<f32> = (0..3).map(|c| -v.image_mean[c] / v.image_std[c]).collect();
        let mut pad_cells = 0usize;
        let (mut max_abs, mut sum_abs) = (0.0f32, 0.0f64);
        for (i, (&a, &b)) in pixels.iter().zip(&expect).enumerate() {
            ensure!(a.is_finite(), "{}: non-finite value at {i}", f.name);
            let d = (a - b).abs();
            max_abs = max_abs.max(d);
            sum_abs += d as f64;
            let channel = (i % patch_dim) / (v.temporal_patch_size * v.patch_size * v.patch_size);
            if (b - pad_ch[channel]).abs() < 1e-5 {
                pad_cells += 1;
            }
        }
        println!(
            "| {} | {grid_h} x {grid_w} | {} | {max_abs:.3e} | {:.3e} | {pad_cells} |",
            f.name,
            grid_h * grid_w,
            sum_abs / pixels.len() as f64
        );
        if max_abs > MAX_ABS {
            failures.push(format!("{}: max-abs {max_abs:.3e} > {MAX_ABS:.0e}", f.name));
        }
    }

    if !failures.is_empty() {
        bail!(
            "Metrale Engine's GLM preprocessing does not match the reference processor:\n  {}",
            failures.join("\n  ")
        );
    }
    println!("\nPASS: 3 fixtures match the reference pixel_values within {MAX_ABS:.0e}");
    Ok(())
}
