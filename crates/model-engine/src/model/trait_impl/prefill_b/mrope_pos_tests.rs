// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests for the MRoPE (T, H, W) position rule in `mrope_pos::build`.
//!
//! Owner: model-engine prefill.
//! Invariants: none beyond the types.

use super::*;

const IMG: u32 = 248_056;
const VID: u32 = 248_057;
const TXT: u32 = 7;

fn run(
    tokens: &[u32],
    grids: &[(usize, usize, usize)],
    start: u32,
) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let (mut t, mut h, mut w) = (Vec::new(), Vec::new(), Vec::new());
    build(
        tokens,
        grids,
        0,
        grids.len(),
        start,
        IMG,
        VID,
        &mut t,
        &mut h,
        &mut w,
    );
    (t, h, w)
}

#[test]
fn text_only_is_the_identity_ramp_on_all_three_streams() {
    let (t, h, w) = run(&[TXT; 5], &[], 0);
    assert_eq!(t, vec![0, 1, 2, 3, 4]);
    assert_eq!(h, t);
    assert_eq!(w, t);
}

#[test]
fn the_streams_start_where_the_caller_says() {
    let (t, h, w) = run(&[TXT; 3], &[], 100);
    assert_eq!(t, vec![100, 101, 102]);
    assert_eq!(h, t);
    assert_eq!(w, t);
}

/// 2026-09-25: A 2×3 image: T constant across the run, H by row, W by column, then
/// the running position advances by max(gh, gw) = 3.
#[test]
fn an_image_holds_t_constant_and_advances_by_its_long_side() {
    let tokens = [TXT, IMG, IMG, IMG, IMG, IMG, IMG, TXT];
    let (t, h, w) = run(&tokens, &[(1, 2, 3)], 0);

    // 2026-09-25: text at 0, image run based at 1, then text.
    assert_eq!(t, vec![0, 1, 1, 1, 1, 1, 1, 4]);
    assert_eq!(h, vec![0, 1, 1, 1, 2, 2, 2, 4]);
    assert_eq!(w, vec![0, 1, 2, 3, 1, 2, 3, 4]);
    // 2026-09-25: The trailing text token is at 1 + max(2, 3) = 4.
    assert_eq!(*t.last().unwrap(), 4);
}

/// 2026-09-25: An image is the `t_len = 1` case, so `build` must agree with the
/// image-only rule `old_rule`, written out independently below.
#[test]
fn the_image_rule_is_unchanged_by_the_temporal_axis() {
    fn old_rule(tokens: &[u32], grids: &[(usize, usize)]) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
        let (mut t, mut h, mut w) = (Vec::new(), Vec::new(), Vec::new());
        let (mut pos, mut idx, mut i) = (0u32, 0usize, 0usize);
        while i < tokens.len() {
            if tokens[i] == IMG && idx < grids.len() {
                let (gh, gw) = grids[idx];
                let run = gh * gw;
                let base = pos;
                for k in 0..run {
                    t.push(base);
                    h.push(base + (k / gw.max(1)) as u32);
                    w.push(base + (k % gw.max(1)) as u32);
                }
                pos += gh.max(gw) as u32;
                i += run;
                idx += 1;
            } else {
                t.push(pos);
                h.push(pos);
                w.push(pos);
                pos += 1;
                i += 1;
            }
        }
        (t, h, w)
    }

    for (gh, gw) in [(1usize, 1usize), (2, 3), (3, 2), (4, 4), (1, 7), (7, 1)] {
        let mut tokens = vec![TXT, TXT];
        tokens.extend(std::iter::repeat_n(IMG, gh * gw));
        tokens.extend([TXT, TXT]);
        assert_eq!(
            run(&tokens, &[(1, gh, gw)], 0),
            old_rule(&tokens, &[(gh, gw)]),
            "the {gh}x{gw} image moved when the temporal axis was added"
        );
    }
}

#[test]
fn two_images_each_advance_the_running_position() {
    let tokens = [IMG, IMG, IMG, IMG, IMG, TXT];
    // 2026-09-25: 1x2 at base 0 (advances 2), then 1x3 at base 2 (advances 3), text at 5.
    let (t, h, w) = run(&tokens, &[(1, 1, 2), (1, 1, 3)], 0);
    assert_eq!(t, vec![0, 0, 2, 2, 2, 5]);
    assert_eq!(h, vec![0, 0, 2, 2, 2, 5]);
    assert_eq!(w, vec![0, 1, 2, 3, 4, 5]);
}

/// 2026-09-25: Three temporal groups over a 2×2 grid: T advances once per group while
/// H and W restart each group, and the item advances the running position by
/// max(3, 2, 2) = 3.
#[test]
fn a_video_advances_t_once_per_temporal_group() {
    let tokens = [
        TXT, VID, VID, VID, VID, VID, VID, VID, VID, VID, VID, VID, VID, TXT,
    ];
    let (t, h, w) = run(&tokens, &[(3, 2, 2)], 0);

    // 2026-09-25: base = 1. Group g gets T = 1 + g, four tokens each.
    assert_eq!(
        t,
        vec![0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4],
        "T must step per group"
    );
    // 2026-09-25: H and W repeat their 2x2 pattern in every group.
    assert_eq!(h, vec![0, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 4]);
    assert_eq!(w, vec![0, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 4]);
    // 2026-09-25: Trailing text at 1 + max(3, 2, 2) = 4.
    assert_eq!(*t.last().unwrap(), 4);
}

/// 2026-09-25: A clip longer than it is wide advances by its temporal extent, so the
/// text after it does not share positions with the end of the clip.
#[test]
fn a_long_clip_advances_by_its_temporal_extent() {
    let t_len = 10usize;
    let mut tokens = vec![TXT];
    tokens.extend(std::iter::repeat_n(VID, t_len));
    tokens.push(TXT);
    let (t, _, _) = run(&tokens, &[(t_len, 1, 1)], 0);
    assert_eq!(t[0], 0);
    assert_eq!(&t[1..=t_len], &(1..=t_len as u32).collect::<Vec<_>>()[..]);
    assert_eq!(
        *t.last().unwrap(),
        1 + t_len as u32,
        "the clip's temporal extent did not advance the running position"
    );
}

/// 2026-09-25: One video and one image in the same request: a mis-scan of the video
/// would shift everything after it.
#[test]
fn a_video_and_an_image_interleave_correctly() {
    let mut tokens = vec![TXT];
    tokens.extend(std::iter::repeat_n(VID, 2));
    tokens.push(TXT);
    tokens.extend(std::iter::repeat_n(IMG, 2 * 2));
    tokens.push(TXT);
    let (t, _, _) = run(&tokens, &[(2, 1, 1), (1, 2, 2)], 0);
    // 2026-09-25: text 0; video base 1, groups at 1 and 2, advance max(2,1,1) = 2;
    // text 3; image base 4, T = 4 four times, advance max(2,2) = 2; text 6.
    assert_eq!(t, vec![0, 1, 2, 3, 4, 4, 4, 4, 6]);
}

/// 2026-09-25: A single-group video gets the same positions as an image of the same grid.
#[test]
fn a_one_group_video_matches_the_equivalent_image() {
    let img_tokens = [TXT, IMG, IMG, IMG, IMG, TXT];
    let vid_tokens = [TXT, VID, VID, VID, VID, TXT];
    assert_eq!(
        run(&img_tokens, &[(1, 2, 2)], 0),
        run(&vid_tokens, &[(1, 2, 2)], 0)
    );
}

/// 2026-09-25: A request owns `grids[grid_base..grid_hi]` of a shared vector and consumes
/// only those items.
#[test]
fn the_grid_window_bounds_which_items_are_consumed() {
    let tokens = [IMG, IMG, IMG, IMG];
    let grids = [(1, 1, 4), (1, 2, 2), (1, 4, 1)];
    let (mut t, mut h, mut w) = (Vec::new(), Vec::new(), Vec::new());
    // 2026-09-25: Owns only `grids[1..2]`, the 2x2 item.
    build(&tokens, &grids, 1, 2, 0, IMG, VID, &mut t, &mut h, &mut w);
    assert_eq!(t, vec![0, 0, 0, 0], "consumed its one owned item");
    assert_eq!(h, vec![0, 0, 1, 1]);
    assert_eq!(w, vec![0, 1, 0, 1]);
}

/// 2026-09-25: Pad tokens with no owned grid left follow the text rule.
#[test]
fn pads_beyond_the_owned_grids_do_not_panic() {
    let tokens = [IMG, IMG, IMG, IMG, IMG];
    let (t, _, _) = run(&tokens, &[(1, 2, 2)], 0);
    assert_eq!(t.len(), 5, "one token in, one position out");
    assert_eq!(t, vec![0, 0, 0, 0, 2]);
}

/// 2026-09-25: With complete pad runs, each stream has one position per token. The pack
/// step (`put_prefix_at` in `upload_meta`) refuses a stream shorter than `proc_count`.
#[test]
fn the_three_streams_always_match_the_token_count() {
    for grids in [
        vec![],
        vec![(1usize, 2usize, 2usize)],
        vec![(3, 2, 2)],
        vec![(2, 1, 1), (1, 3, 3)],
    ] {
        let total: usize = grids.iter().map(|(t, h, w)| t * h * w).sum();
        let mut tokens = vec![TXT, TXT];
        tokens.extend(std::iter::repeat_n(IMG, total));
        tokens.push(TXT);
        let (t, h, w) = run(&tokens, &grids, 0);
        assert_eq!(t.len(), tokens.len(), "T stream length, grids={grids:?}");
        assert_eq!(h.len(), tokens.len(), "H stream length, grids={grids:?}");
        assert_eq!(w.len(), tokens.len(), "W stream length, grids={grids:?}");
    }
}

/// 2026-09-25: A zero in a grid neither divides by zero nor stalls the scan.
#[test]
fn a_degenerate_grid_is_survivable() {
    let (t, h, w) = run(&[IMG, TXT], &[(0, 0, 0)], 7);
    assert_eq!(t, vec![7, 8]);
    assert_eq!(h, t);
    assert_eq!(w, t);
}
