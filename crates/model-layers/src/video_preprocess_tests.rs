// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests for video decode, frame sampling and temporal grouping.
//! The sampling arithmetic sets a video's token count, so it is tested
//! directly at the boundaries as well as through decoded GIFs.
//!
//! Owner: model-layers (vision input).
//! Invariants: none beyond the types.

use super::*;
use crate::video_decode_ffmpeg::FfmpegPolicy;

/// 2026-09-25: The default policy: no subprocess. The GIF cases pass without
/// ffmpeg, so the in-process path does not depend on it.
fn no_ffmpeg() -> FfmpegPolicy {
    FfmpegPolicy::default()
}

fn cfg() -> VisionConfig {
    VisionConfig {
        depth: 2,
        hidden_size: 32,
        num_heads: 2,
        patch_size: 16,
        temporal_patch_size: 2,
        spatial_merge_size: 2,
        intermediate_size: 64,
        out_hidden_size: 32,
        deepstack_visual_indexes: vec![],
        image_pad_token_id: 248_056,
        video_pad_token_id: 248_057,
        max_pixels: None,
        ..VisionConfig::default()
    }
}

/// 2026-09-25: The count is a whole number of temporal groups whenever one full
/// group is achievable. A clip with fewer frames than one group is returned
/// as it is, and `preprocess_video` refuses it (see
/// `one_frame_cannot_make_a_group_at_tp2`).
#[test]
fn the_frame_count_is_always_a_whole_number_of_groups() {
    const TP: usize = 2;
    for n in 1..64usize {
        for fps in [1.0f32, 2.0, 5.0, 7.5, 30.0] {
            let got = sample_indices(n, 30.0, fps, 4, 768, TP);
            if n < TP {
                assert_eq!(got.len(), n, "n={n}: too short for a group, report as-is");
                continue;
            }
            assert_eq!(
                got.len() % TP,
                0,
                "n={n} fps={fps} produced {} frames, not a whole number of groups",
                got.len()
            );
            assert!(got.len() >= TP, "n={n} fps={fps} dropped below one group");
            assert!(got.len() <= n, "n={n} fps={fps} sampled more than it had");
        }
    }
}

#[test]
fn indices_are_in_range_and_non_decreasing() {
    let got = sample_indices(100, 30.0, 2.0, 4, 768, 2);
    assert!(!got.is_empty());
    for w in got.windows(2) {
        assert!(w[0] <= w[1], "indices went backwards: {got:?}");
    }
    assert!(*got.last().unwrap() < 100);
}

/// 2026-09-25: A 30 fps, 100-frame clip is 3.33 s; at 2 fps that rounds to 7
/// frames, then down to 6 for three whole groups.
#[test]
fn a_clip_is_sampled_to_the_requested_rate() {
    let got = sample_indices(100, 30.0, 2.0, 4, 768, 2);
    assert_eq!(got.len(), 6, "got {got:?}");
    // 2026-09-25: Spread across the whole clip, not clustered at the start.
    assert_eq!(got[0], 0);
    assert_eq!(*got.last().unwrap(), 99);
}

/// 2026-09-25: Never upsample: asking 30 fps of a 4-frame clip returns those 4
/// frames, not repeats.
#[test]
fn a_short_clip_is_never_padded_up_to_the_minimum() {
    let got = sample_indices(4, 2.0, 30.0, 16, 768, 2);
    assert_eq!(got.len(), 4);
    assert_eq!(got, vec![0, 1, 2, 3]);
}

#[test]
fn the_max_frames_ceiling_is_honoured() {
    let got = sample_indices(10_000, 30.0, 30.0, 4, 768, 2);
    assert_eq!(got.len(), 768);
}

#[test]
fn an_inverted_frame_band_uses_the_ceiling() {
    let got = sample_indices(100, 30.0, 30.0, 16, 4, 2);
    assert_eq!(got.len(), 4);
    assert_eq!(got[0], 0);
    assert_eq!(*got.last().unwrap(), 99);
}

/// 2026-09-25: A zero or non-finite rate falls back to `DEFAULT_FPS` instead of
/// dividing by zero or returning nothing.
#[test]
fn zero_and_nonfinite_rates_fall_back_rather_than_exploding() {
    for (native, target, valid_native, valid_target) in [
        (0.0f32, 1.0f32, DEFAULT_FPS, 1.0),
        (30.0, 0.0, 30.0, DEFAULT_FPS),
        (f32::NAN, 1.0, DEFAULT_FPS, 1.0),
        (30.0, f32::INFINITY, 30.0, DEFAULT_FPS),
    ] {
        let got = sample_indices(100, native, target, 4, 768, 2);
        let expected = sample_indices(100, valid_native, valid_target, 4, 768, 2);
        assert_eq!(got, expected, "native={native} target={target}");
    }
}

#[test]
fn an_empty_clip_samples_to_nothing() {
    assert!(sample_indices(0, 30.0, 2.0, 4, 768, 2).is_empty());
}

/// 2026-09-25: A single frame cannot fill a tp=2 group. Sampling returns it,
/// and `preprocess_video` refuses it
/// (`a_single_frame_gif_is_refused_rather_than_treated_as_a_still`).
#[test]
fn one_frame_cannot_make_a_group_at_tp2() {
    let got = sample_indices(1, 30.0, 2.0, 4, 768, 2);
    assert_eq!(got.len(), 1, "sampling reports the single frame it has");
    assert!(got.len() < 2, "and it is short of one tp=2 group");
}

/// 2026-09-25: Build an animated GIF of `n` solid-color frames at `size`x`size`,
/// 100 ms each, as a base64 `data:` URI.
fn make_gif(n: u16, size: u16) -> String {
    use image::codecs::gif::GifEncoder;
    use image::{Delay, Frame, RgbaImage};
    let mut buf: Vec<u8> = Vec::new();
    {
        let mut enc = GifEncoder::new(&mut buf);
        for i in 0..n {
            // 2026-09-25: A distinct color per frame, so frame order shows in the
            // pixel data.
            let v = (20 + (i as u32 * 37) % 200) as u8;
            let img = RgbaImage::from_pixel(
                size as u32,
                size as u32,
                // 2026-09-25: `255 - v` cannot underflow: v is at most 219.
                image::Rgba([v, 40, 255 - v, 255]),
            );
            let mut f = Frame::from_parts(img, 0, 0, Delay::from_numer_denom_ms(100, 1));
            f = Frame::from_parts(f.into_buffer(), 0, 0, Delay::from_numer_denom_ms(100, 1));
            enc.encode_frame(f).expect("encode frame");
        }
    }
    use base64::Engine as _;
    format!(
        "data:image/gif;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(&buf)
    )
}

#[test]
fn an_animated_gif_decodes_to_its_frames() {
    let uri = make_gif(6, 64);
    let (frames, fps) = decode_frames(&uri, 10.0, &no_ffmpeg()).expect("decode");
    assert_eq!(frames.len(), 6);
    // 2026-09-25: 100 ms per frame is 10 fps.
    assert!((fps - 10.0).abs() < 0.5, "fps was {fps}");
}

#[test]
fn gif_magic_wins_over_a_mislabelled_mime() {
    let uri = make_gif(4, 64).replacen("data:image/gif", "data:video/mp4", 1);
    let (frames, fps) = decode_frames(&uri, 10.0, &no_ffmpeg()).expect("decode by magic");
    assert_eq!(frames.len(), 4);
    assert!((fps - 10.0).abs() < 0.5, "fps was {fps}");
}

/// 2026-09-25: The shape the encoder takes: one buffer per temporal group, each
/// `grid_h * grid_w * (C * tp * patch²)` long.
#[test]
fn grouping_produces_correctly_shaped_buffers() {
    let uri = make_gif(8, 64);
    let v = preprocess_video(&uri, &cfg(), None, 10.0, &no_ffmpeg()).expect("preprocess");
    assert_eq!(v.grid_h, 4, "64px / 16 = 4 patches");
    assert_eq!(v.grid_w, 4);
    assert_eq!(v.grid_t, 4, "8 frames at tp=2 = 4 groups");
    assert_eq!(v.groups.len(), 4);
    let expect = v.grid_h * v.grid_w * 3 * 2 * 16 * 16;
    for (i, g) in v.groups.iter().enumerate() {
        assert_eq!(g.len(), expect, "group {i} is the wrong length");
    }
}

/// 2026-09-25: Token accounting: temporal groups times the merged plane.
#[test]
fn pad_count_is_groups_times_the_merged_plane() {
    let uri = make_gif(8, 64);
    let v = preprocess_video(&uri, &cfg(), None, 10.0, &no_ffmpeg()).expect("preprocess");
    assert_eq!(v.pad_count(2), 16);
    assert_eq!(v.pad_count(1), 4 * 16);
}

/// 2026-09-25: Distinct frames land in distinct temporal slots. A still repeats
/// one frame across the tp axis; a video that did so would pass every shape
/// check above.
#[test]
fn the_two_frames_of_a_group_are_actually_different() {
    let uri = make_gif(4, 64);
    let v = preprocess_video(&uri, &cfg(), None, 10.0, &no_ffmpeg()).expect("preprocess");
    let ps = 16usize;
    let g = &v.groups[0];
    // 2026-09-25: Offsets of t=0 and t=1 within channel 0 of patch 0.
    let a = g[0];
    let b = g[ps * ps];
    assert_ne!(
        a, b,
        "t=0 and t=1 hold identical pixels — the frames were duplicated, not paired"
    );
}

/// 2026-09-25: Consecutive groups hold different frames: group 1 is not a copy
/// of group 0.
#[test]
fn consecutive_groups_hold_different_frames() {
    let uri = make_gif(4, 64);
    let v = preprocess_video(&uri, &cfg(), None, 10.0, &no_ffmpeg()).expect("preprocess");
    assert_eq!(v.groups.len(), 2);
    assert_ne!(v.groups[0][0], v.groups[1][0], "group 1 repeated group 0");
}

/// 2026-09-25: The `max_pixels` area bound shrinks a video's grid as it does a
/// still's, and leaves its temporal extent alone.
#[test]
fn the_area_bound_shrinks_the_grid() {
    let uri = make_gif(4, 256);
    let big = preprocess_video(&uri, &cfg(), None, 10.0, &no_ffmpeg()).expect("unbounded");
    let small = preprocess_video(&uri, &cfg(), Some(64 * 64), 10.0, &no_ffmpeg()).expect("bounded");
    assert!(
        small.grid_h < big.grid_h,
        "bound {}x{} did not shrink {}x{}",
        small.grid_h,
        small.grid_w,
        big.grid_h,
        big.grid_w
    );
    assert_eq!(
        small.grid_t, big.grid_t,
        "the bound is spatial, not temporal"
    );
}

/// 2026-09-25: With ffmpeg disabled, an mp4 is refused with its MIME named and
/// the ffmpeg hint, not misparsed as a GIF.
#[test]
fn an_mp4_is_refused_by_name_with_a_conversion_hint() {
    use base64::Engine as _;
    let uri = format!(
        "data:video/mp4;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(b"\x00\x00\x00\x20ftypmp42")
    );
    let err = format!("{:#}", decode_frames(&uri, 10.0, &no_ffmpeg()).unwrap_err());
    assert!(err.contains("mp4"), "{err}");
    assert!(err.contains("ffmpeg"), "no conversion hint: {err}");
}

#[test]
fn a_single_frame_gif_is_refused_rather_than_treated_as_a_still() {
    let uri = make_gif(1, 64);
    let err = format!(
        "{:#}",
        preprocess_video(&uri, &cfg(), None, 10.0, &no_ffmpeg()).unwrap_err()
    );
    assert!(err.contains("temporal group"), "{err}");
}

#[test]
fn garbage_is_an_error_not_a_panic() {
    assert!(decode_frames("data:video/gif;base64,bm90LWEtZ2lm", 10.0, &no_ffmpeg()).is_err());
    assert!(decode_frames("!!! not base64 !!!", 10.0, &no_ffmpeg()).is_err());
}

#[test]
fn invalid_geometry_is_refused_before_dividing_by_it() {
    let uri = make_gif(4, 64);
    let mut c = cfg();
    c.temporal_patch_size = 0;
    assert!(preprocess_video(&uri, &c, None, 10.0, &no_ffmpeg()).is_err());
    let mut c = cfg();
    c.patch_size = 0;
    assert!(preprocess_video(&uri, &c, None, 10.0, &no_ffmpeg()).is_err());
}
