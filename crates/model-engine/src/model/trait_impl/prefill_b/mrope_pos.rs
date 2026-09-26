// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: MRoPE (T, H, W) position streams for a prefill chunk: where every
//! vision token sits in the three rotary streams. Pure arithmetic, called by
//! `upload_meta`, so the rule is testable without a GPU.
//!
//! Owner: model-engine prefill.
//! Invariants: none beyond the types.

/// 2026-09-25: Append the (T, H, W) streams for `chunk_tokens` to the three output
/// vectors, starting the running position `pos` at `start_pos`.
///
/// - A text token takes `T = H = W = pos` and advances `pos` by one.
/// - A pad token, while an owned item `grids[grid_base..grid_hi]` remains, starts
///   that item: `t_len` temporal groups over a `gh × gw` grid, `t_len * gh * gw`
///   positions from `base = pos`. Entry `k` of group `g` takes `T = base + g`,
///   `H = base + row`, `W = base + col`; afterwards `pos` advances by
///   `max(t_len, gh, gw)`. A zero `t_len` counts as one, and an empty grid
///   (`gh * gw == 0`) as one position per group.
/// - A pad token with no owned item left follows the text rule.
///
/// An image is the `t_len = 1` case: T stays at `base` and the advance is
/// `max(gh, gw)`. Image and video pad tokens are consumed the same way.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build(
    chunk_tokens: &[u32],
    grids: &[(usize, usize, usize)],
    grid_base: usize,
    grid_hi: usize,
    start_pos: u32,
    image_pad: u32,
    video_pad: u32,
    t_out: &mut Vec<u32>,
    h_out: &mut Vec<u32>,
    w_out: &mut Vec<u32>,
) {
    let is_pad = |tok: u32| tok == image_pad || tok == video_pad;
    let mut pos = start_pos;
    let mut item = grid_base;
    let mut i = 0usize;
    while i < chunk_tokens.len() {
        if is_pad(chunk_tokens[i]) && item < grid_hi {
            let (t_len, gh, gw) = grids[item];
            let t_len = t_len.max(1);
            let plane = (gh * gw).max(1);
            let run_len = t_len * plane;
            let base = pos;
            for k in 0..run_len {
                // 2026-09-25: [group, row, col] order, the order in which `embed_chunk.rs`
                // splices encoder rows (one per pad token, in sequence).
                let g = (k / plane) as u32;
                let within = k % plane;
                let row = (within / gw.max(1)) as u32;
                let col = (within % gw.max(1)) as u32;
                t_out.push(base + g);
                h_out.push(base + row);
                w_out.push(base + col);
            }
            // 2026-09-25: The item's extent on every axis, so the next token starts clear
            // of all three streams; a long clip can be longer than it is wide.
            pos += t_len.max(gh).max(gw) as u32;
            i += run_len;
            item += 1;
        } else {
            t_out.push(pos);
            h_out.push(pos);
            w_out.push(pos);
            pos += 1;
            i += 1;
        }
    }
}

#[cfg(test)]
#[path = "mrope_pos_tests.rs"]
mod tests;
