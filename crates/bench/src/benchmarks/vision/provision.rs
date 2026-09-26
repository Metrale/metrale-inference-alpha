// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The vision fixtures, written to `~/.metrale/artifacts/vision` on
//! `load()`.
//!
//! The images are compiled into the binary from `crates/bench/assets/vision/`
//! and written out only when a file's bytes differ (`write_asset_bytes`),
//! behind a content-derived [`Stamp`], so a run that already provisioned
//! writes nothing. The written directory lets an operator send the exact bytes
//! a run used with any HTTP client.
//!
//! The same files are in `tests/fixtures/images/`, and
//! `assets_match_the_repo_fixtures` fails when the two copies differ. They are
//! copied into the crate because `cargo package` does not include a path
//! outside it. `scripts/gen_test_images.py` generates `01` to `09`; it does
//! not generate `10` to `16`.
//!
//! Owner: bench, vision.
//! Invariants:
//! - `provision` commits the stamp only after every fixture write succeeded.

use anyhow::Result;

use crate::artifacts::{ArtifactStore, Stamp, write_asset_bytes};

/// 2026-09-26: The geometry ladder as `(name, bytes, width, height)`: `01` to
/// `08` square, landscape and portrait sizes up to 1280x720, `09` over the
/// 1280 px fallback clamp, `10` and `11` geometry extremes, and `12` to `14`
/// decode variants at 224x224.
pub const FIXTURES: &[(&str, &[u8], u32, u32)] = &[
    (
        "01_square_224.png",
        include_bytes!("../../../assets/vision/01_square_224.png"),
        224,
        224,
    ),
    (
        "02_square_336.png",
        include_bytes!("../../../assets/vision/02_square_336.png"),
        336,
        336,
    ),
    (
        "03_landscape_512x384.png",
        include_bytes!("../../../assets/vision/03_landscape_512x384.png"),
        512,
        384,
    ),
    (
        "04_wide_640x360.png",
        include_bytes!("../../../assets/vision/04_wide_640x360.png"),
        640,
        360,
    ),
    (
        "05_square_768.png",
        include_bytes!("../../../assets/vision/05_square_768.png"),
        768,
        768,
    ),
    (
        "06_wide_1024x576.png",
        include_bytes!("../../../assets/vision/06_wide_1024x576.png"),
        1024,
        576,
    ),
    (
        "07_hd_1280x720.png",
        include_bytes!("../../../assets/vision/07_hd_1280x720.png"),
        1280,
        720,
    ),
    (
        "08_portrait_480x854.png",
        include_bytes!("../../../assets/vision/08_portrait_480x854.png"),
        480,
        854,
    ),
    // 2026-09-26: Every fixture above has a long side of at most 1280 px, so
    // its count is the same whether the engine honours the checkpoint's area
    // bound or falls back to the 1280 px clamp. 1600x900 is 1.44M px, under the
    // 16777216 px of the Qwen3.6 preprocessor configs, so at native size it is
    // 1400 tokens; under the clamp it is served at 1280x736, 920 tokens.
    (
        "09_over_clamp_1600x900.png",
        include_bytes!("../../../assets/vision/09_over_clamp_1600x900.png"),
        1600,
        900,
    ),
    // 2026-09-26: 8x8 snaps up to one 32 px grid unit, one merged token: the
    // smallest target the preprocessor produces (`.max(1)` in
    // `target_size_for`).
    (
        "10_tiny_8x8.png",
        include_bytes!("../../../assets/vision/10_tiny_8x8.png"),
        8,
        8,
    ),
    // 2026-09-26: 64x2048 has a small area and a long side over the 1280 px
    // fallback clamp: 128 tokens at native size, 40 under the clamp (served at
    // 32x1280).
    (
        "11_strip_64x2048.png",
        include_bytes!("../../../assets/vision/11_strip_64x2048.png"),
        64,
        2048,
    ),
    // 2026-09-26: Decode variants at 224x224, so each asserts the same 49
    // tokens as `01`; only the pixel format differs. `12` is RGBA with a
    // varying alpha channel, which the engine's `to_rgb8()` drops.
    (
        "12_rgba_224.png",
        include_bytes!("../../../assets/vision/12_rgba_224.png"),
        224,
        224,
    ),
    // 2026-09-26: Single-channel (grayscale) JPEG.
    (
        "13_gray_224.jpg",
        include_bytes!("../../../assets/vision/13_gray_224.jpg"),
        224,
        224,
    ),
    // 2026-09-26: 16-bit grayscale PNG.
    (
        "14_png16_224.png",
        include_bytes!("../../../assets/vision/14_png16_224.png"),
        224,
        224,
    ),
];

/// 2026-09-26: The EXIF-orientation pair, kept out of the geometry ladder
/// because the integrity leg checks what the model reports, not a token count.
///
/// Both store the same pixels, red on the top half and blue on the bottom, at
/// 224x224. `15` carries `Orientation = 6` ("rotate 90 CW to display"); `16`
/// carries no EXIF. The engine applies the tag
/// (`model-layers/src/vision_preprocess.rs`), which carries the stored top
/// edge to the right, so the tagged image must read "right" and the untagged
/// one "top".
pub const EXIF_PAIR: &[(&str, &[u8])] = &[
    (
        "15_exif_rot90_224.jpg",
        include_bytes!("../../../assets/vision/15_exif_rot90_224.jpg"),
    ),
    (
        "16_exif_none_224.jpg",
        include_bytes!("../../../assets/vision/16_exif_none_224.jpg"),
    ),
];

/// 2026-09-26: Content-derived over every fixture's name and bytes, so a
/// changed image re-provisions without a version bump.
fn stamp_value() -> String {
    crate::benchmarks::content_stamp(
        "vision-fixtures-v1",
        FIXTURES
            .iter()
            .map(|(name, bytes, _, _)| (*name, *bytes))
            .chain(EXIF_PAIR.iter().copied()),
    )
}

/// 2026-09-26: The artifact directory name: `~/.metrale/artifacts/vision`.
pub const PLUGIN_ID: &str = "vision";

/// 2026-09-26: Write the fixtures. With a current stamp it writes nothing.
pub fn provision(store: &ArtifactStore) -> Result<std::path::PathBuf> {
    let dir = store.plugin_dir(PLUGIN_ID)?;
    let stamp = Stamp::new(&dir, ".provisioned", stamp_value());
    if stamp.is_current() {
        return Ok(dir);
    }
    for (name, bytes, _, _) in FIXTURES {
        write_asset_bytes(&dir, name, bytes)?;
    }
    for (name, bytes) in EXIF_PAIR {
        write_asset_bytes(&dir, name, bytes)?;
    }
    // 2026-09-26: Last: a stamp written before the writes complete would mark
    // a partial directory as current.
    stamp.commit()?;
    Ok(dir)
}

#[cfg(test)]
#[path = "provision_tests.rs"]
mod provision_tests;
