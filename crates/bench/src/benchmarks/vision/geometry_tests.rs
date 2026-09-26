// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for the vision-token predictions: native counts, grid
//! rounding, the ladder's discriminating rungs, and the declared area bound.
//!
//! Owner: bench, vision.
//! Invariants: none beyond the types.

use super::*;

#[test]
fn token_count_is_quadratic_in_the_side() {
    // 2026-09-26: Doubling the side quadruples the tokens, which catches an
    // off-by-one in the merge divisor that one fixed expectation would not.
    let a = expected_vision_tokens(224, 224, 16, 2);
    let b = expected_vision_tokens(448, 448, 16, 2);
    let c = expected_vision_tokens(896, 896, 16, 2);
    assert_eq!(a, 49);
    assert_eq!(b, a * 4, "{a} -> {b} is not quadratic");
    assert_eq!(c, b * 4, "{b} -> {c} is not quadratic");
}

#[test]
fn every_ladder_size_has_a_defined_expectation() {
    let got: Vec<_> = crate::benchmarks::vision::provision::FIXTURES
        .iter()
        .map(|&(name, _, w, h)| (name, w, h, expected_vision_tokens(w, h, 16, 2)))
        .collect();
    assert_eq!(
        got,
        vec![
            ("01_square_224.png", 224, 224, 49),
            ("02_square_336.png", 336, 336, 121),
            ("03_landscape_512x384.png", 512, 384, 192),
            ("04_wide_640x360.png", 640, 360, 220),
            ("05_square_768.png", 768, 768, 576),
            ("06_wide_1024x576.png", 1024, 576, 576),
            ("07_hd_1280x720.png", 1280, 720, 920),
            ("08_portrait_480x854.png", 480, 854, 405),
            ("09_over_clamp_1600x900.png", 1600, 900, 1400),
            ("10_tiny_8x8.png", 8, 8, 1),
            ("11_strip_64x2048.png", 64, 2048, 128),
            ("12_rgba_224.png", 224, 224, 49),
            ("13_gray_224.jpg", 224, 224, 49),
            ("14_png16_224.png", 224, 224, 49),
        ]
    );
}

#[test]
fn portrait_and_landscape_of_the_same_shape_agree() {
    // 2026-09-26: Transposing does not change the count, which rejects an
    // expectation that uses one side twice; square fixtures cannot show that.
    assert_eq!(
        expected_vision_tokens(512, 384, 16, 2),
        expected_vision_tokens(384, 512, 16, 2)
    );
}

#[test]
fn snap_never_returns_zero() {
    // 2026-09-26: A sub-grid image still produces one grid unit, not 0x0.
    assert_eq!(snap(1, 32), 32);
    assert_eq!(snap(15, 32), 32);
    assert_eq!(expected_vision_tokens(1, 1, 16, 2), 1);
}

/// 2026-09-26: At least one fixture has a different count under the 1280 px
/// long-side clamp than at native size, or the geometry leg could not catch
/// an engine falling back to the clamp. `09_over_clamp_1600x900` and
/// `11_strip_64x2048` both do.
#[test]
fn the_ladder_can_actually_detect_a_regression_to_the_old_clamp() {
    /// 2026-09-26: The 1280 px clamp: scale so the long side is at most 1280,
    /// never upscaling.
    fn under_old_clamp(w: u32, h: u32) -> u32 {
        let long = w.max(h) as f32;
        let s = (1280.0 / long).min(1.0);
        expected_vision_tokens(
            ((w as f32) * s).round() as u32,
            ((h as f32) * s).round() as u32,
            16,
            2,
        )
    }

    let ladder: Vec<(u32, u32)> = crate::benchmarks::vision::provision::FIXTURES
        .iter()
        .map(|&(_, _, w, h)| (w, h))
        .collect();
    let discriminating: Vec<(u32, u32)> = ladder
        .iter()
        .copied()
        .filter(|&(w, h)| under_old_clamp(w, h) != expected_vision_tokens(w, h, 16, 2))
        .collect();

    assert!(
        !discriminating.is_empty(),
        "every fixture in the ladder sits at or under the 1280px clamp, so a \
         regression to it would change no expectation and the geometry leg \
         would pass on a broken engine. Add a fixture above 1280 on the long \
         side."
    );

    // 2026-09-26: Pin the 1600x900 rung's two counts.
    assert_eq!(under_old_clamp(1600, 900), 920);
    assert_eq!(expected_vision_tokens(1600, 900, 16, 2), 1400);
}

#[test]
fn the_rounding_mode_is_pinned() {
    // 2026-09-26: `f32::round` rounds half away from zero, as the engine's
    // `target_size_for` does; half-even would move 336 and 720 by one grid
    // unit.
    assert_eq!(snap(224, 32), 224, "already exact");
    assert_eq!(snap(336, 32), 352, "10.5 rounds away from zero");
    assert_eq!(snap(360, 32), 352, "11.25 rounds down");
    assert_eq!(snap(720, 32), 736, "22.5 rounds away from zero");
    assert_eq!(snap(854, 32), 864, "26.69 rounds up");
}

/// 2026-09-26: Every fixture in the geometry ladder, as `(w, h)`, in
/// `provision::FIXTURES` order.
const LADDER: [(u32, u32); 14] = [
    (224, 224),
    (336, 336),
    (512, 384),
    (640, 360),
    (768, 768),
    (1024, 576),
    (1280, 720),
    (480, 854),
    (1600, 900),
    (8, 8),
    (64, 2048),
    (224, 224),
    (224, 224),
    (224, 224),
];

#[test]
fn the_mirror_matches_the_engines_anchors() {
    // 2026-09-26: The 1600x900 rung: 1400 tokens at native size, 920 under the
    // 1280 px fallback clamp, which serves it at 1280x736.
    assert_eq!(
        expected_vision_tokens(1600, 900, 16, 2),
        1400,
        "unbounded: the checkpoint's own bound is far above 1.44M px"
    );
    let (tw, th) = served_size(1600, 900, 32, None);
    assert_eq!((tw, th), (1280, 736), "the 1280px fallback clamp");
    assert_eq!(
        (tw / 16) * (th / 16) / 4,
        920,
        "the figure the fallback clamp produces, per provision::FIXTURES"
    );
}

#[test]
fn zero_means_nothing_was_declared() {
    // 2026-09-26: 0 is the driver's `vision_max_pixels` default and predicts
    // native geometry.
    for (w, h) in LADDER {
        assert_eq!(
            expected_vision_tokens_bounded(w, h, 16, 2, 0),
            expected_vision_tokens(w, h, 16, 2),
            "{w}x{h} moved when no bound was declared"
        );
    }
}

#[test]
fn a_declared_bound_moves_exactly_the_fixtures_above_it() {
    // 2026-09-26: A 262144 px bound moves exactly the five fixtures whose area
    // exceeds it.
    const CAP: u64 = 262_144;
    let moved: Vec<(u32, u32)> = LADDER
        .iter()
        .copied()
        .filter(|&(w, h)| {
            expected_vision_tokens_bounded(w, h, 16, 2, CAP) != expected_vision_tokens(w, h, 16, 2)
        })
        .collect();
    assert_eq!(
        moved,
        vec![
            (768, 768),
            (1024, 576),
            (1280, 720),
            (480, 854),
            (1600, 900)
        ],
        "exactly the five fixtures whose area exceeds {CAP}"
    );
    for &(w, h) in &moved {
        assert!(
            (w as u64) * (h as u64) > CAP,
            "{w}x{h} moved but is inside the bound"
        );
    }
}

#[test]
fn a_declared_bound_never_upscales() {
    // 2026-09-26: A bound only shrinks. The 8x8 and 64x2048 rungs are inside
    // 262144 px, where `sqrt(bound / area)` is over 1 and only `.min(1.0)`
    // keeps them native.
    assert_eq!(expected_vision_tokens_bounded(8, 8, 16, 2, 262_144), 1);
    assert_eq!(
        expected_vision_tokens_bounded(64, 2048, 16, 2, 262_144),
        128
    );
}

#[test]
fn the_discriminating_rung_stays_discriminating_under_a_bound() {
    // 2026-09-26: Under a 262144 px bound the 1600x900 rung predicts 252, which
    // still differs from the 920 of the fallback clamp.
    let honoured = expected_vision_tokens_bounded(1600, 900, 16, 2, 262_144);
    assert_eq!(honoured, 252);
    let (tw, th) = served_size(1600, 900, 32, None);
    assert_ne!(
        honoured,
        (tw / 16) * (th / 16) / 4,
        "a declared bound must not make the fallback-clamp defect indistinguishable"
    );
}

#[test]
fn the_absolute_long_side_ceiling_still_applies_under_a_bound() {
    // 2026-09-26: A large area bound does not lift the long-side ceiling:
    // 64x8192 is 512K px, and its 8192 side is scaled to 4096 or less.
    let (_, th) = served_size(64, 8192, 32, Some(16_777_216));
    assert!(th <= ABS_MAX_DIM, "{th} exceeds the absolute ceiling");
}
