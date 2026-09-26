// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Video token geometry: tokens per temporal group, and the check
//! that temporal groups scale with clip duration.
//!
//! A clip's absolute token count depends on the server's `--video-fps`, which
//! the benchmark neither sends nor reads, so the geometry leg asserts a
//! relation between three durations instead of absolute counts.
//!
//! Owner: bench, video.
//! Invariants: none beyond the types.

/// 2026-09-26: Merged tokens one temporal group of a `w x h` clip occupies:
/// the still-image count, `vision::geometry::expected_vision_tokens`.
pub fn tokens_per_group(w: u32, h: u32, patch: u32, merge: u32) -> u32 {
    crate::benchmarks::vision::geometry::expected_vision_tokens(w, h, patch, merge)
}

/// 2026-09-26: What the geometry leg concluded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ratio {
    /// 2026-09-26: Temporal groups inferred for the shortest clip.
    pub unit_groups: usize,
    /// 2026-09-26: Template overhead implied, in tokens.
    pub overhead: usize,
}

/// 2026-09-26: Check that temporal groups scale with duration, from the totals
/// of three clips at 1x, 2x and 4x, and infer the group count and template
/// overhead.
///
/// Two totals cannot test this: any difference `t2 - t1` fits a 2:1 ratio
/// once the implied overhead absorbs the rest. With three the system is
/// over-determined:
///
/// ```text
///   t4 - t2  ==  2 * (t2 - t1)
/// ```
///
/// which involves neither the overhead nor the tokens-per-group figure. It
/// also returns `None` when `t2 - t1` is zero or not a whole number of
/// `plane`s, when `plane` is zero, when the implied overhead is negative, and
/// on overflow.
pub fn check_proportional(t1: usize, t2: usize, t4: usize, plane: usize) -> Option<Ratio> {
    let d21 = t2.checked_sub(t1)?;
    let d42 = t4.checked_sub(t2)?;
    if d21 == 0 || d42 != d21.checked_mul(2)? {
        return None;
    }
    if plane == 0 || d21 % plane != 0 {
        return None;
    }
    let unit_groups = d21 / plane;
    // 2026-09-26: A 1x total smaller than one unit of groups would imply a
    // negative overhead.
    let step = unit_groups.checked_mul(plane)?;
    let overhead = t1.checked_sub(step)?;
    Some(Ratio {
        unit_groups,
        overhead,
    })
}

#[cfg(test)]
#[path = "geometry_tests.rs"]
mod geometry_tests;
