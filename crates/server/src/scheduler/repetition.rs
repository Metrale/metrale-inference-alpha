// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Fuzzy repetition detector for the tail of a response, and
//! its tests.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

/// 2026-09-25: Detect a fuzzy repetition loop in the tail of `tokens`.
///
/// Returns `Some((pattern_len, mis_a, mis_b))` if the last `3 * pattern_len`
/// tokens are three consecutive near-copies of one window: windows A↔B and
/// B↔C each differ in at most `max(pattern_len / tolerance_div, 1)`
/// positions. Returns `None` otherwise, and always below 90 tokens.
///
/// Pattern lengths 40 down to 15 are tried, so the longest match is
/// returned. Three copies are required so that a block written twice, such
/// as two similar signatures, does not match. The decode path passes
/// MODEL.toml `[behavior].fuzzy_repeat_tolerance_div`, 12 when unset.
pub fn detect_fuzzy_repetition(
    tokens: &[u32],
    tolerance_div: usize,
) -> Option<(usize, usize, usize)> {
    let len = tokens.len();
    if len < 90 {
        return None;
    }
    for pattern_len in (15..=40).rev() {
        if len < pattern_len * 3 {
            continue;
        }
        let base = len - pattern_len * 3;
        let max_mismatches = (pattern_len / tolerance_div).max(1);
        let mut mis_a = 0usize;
        let mut mis_b = 0usize;
        for i in 0..pattern_len {
            if tokens[base + i] != tokens[base + pattern_len + i] {
                mis_a += 1;
            }
            if tokens[base + pattern_len + i] != tokens[base + 2 * pattern_len + i] {
                mis_b += 1;
            }
            if mis_a > max_mismatches && mis_b > max_mismatches {
                break;
            }
        }
        if mis_a <= max_mismatches && mis_b <= max_mismatches {
            return Some((pattern_len, mis_a, mis_b));
        }
    }
    None
}

#[cfg(test)]
mod fuzzy_repetition_tests {
    use super::detect_fuzzy_repetition;

    const DEFAULT_TOLERANCE_DIV: usize =
        crate::scheduler::helpers::WatchdogParams::DEFAULT_FUZZY_TOLERANCE_DIV;

    #[test]
    fn returns_none_below_minimum_output() {
        let tokens: Vec<u32> = (0..50).collect();
        assert_eq!(
            detect_fuzzy_repetition(&tokens, DEFAULT_TOLERANCE_DIV),
            None
        );
    }

    #[test]
    fn returns_none_on_non_repeating_output() {
        let tokens: Vec<u32> = (0..120).collect();
        assert_eq!(
            detect_fuzzy_repetition(&tokens, DEFAULT_TOLERANCE_DIV),
            None
        );
    }

    #[test]
    fn detects_exact_triple_repetition() {
        // 2026-09-25: 30 + 3 * 20 = 90 tokens, exactly the minimum length.
        let pattern: Vec<u32> = (1000..1020).collect();
        let mut tokens: Vec<u32> = (0..30).collect();
        for _ in 0..3 {
            tokens.extend_from_slice(&pattern);
        }
        let hit = detect_fuzzy_repetition(&tokens, DEFAULT_TOLERANCE_DIV);
        assert!(
            matches!(hit, Some((20, 0, 0))),
            "expected exact 20-tok x3 detection, got {:?}",
            hit
        );
    }

    #[test]
    fn does_not_fire_on_double_boilerplate() {
        let pattern: Vec<u32> = (500..520).collect();
        let mut tokens: Vec<u32> = (0..40).collect();
        tokens.extend_from_slice(&pattern);
        tokens.extend_from_slice(&pattern);
        for t in 600..660 {
            tokens.push(t);
        }
        assert_eq!(
            detect_fuzzy_repetition(&tokens, DEFAULT_TOLERANCE_DIV),
            None
        );
    }

    #[test]
    fn detects_fuzzy_triple_within_tolerance() {
        // 2026-09-25: One mismatch per adjacent pair; the tolerance is
        // 24 / 12 = 2.
        let base: Vec<u32> = (2000..2024).collect();
        let mut copy_b = base.clone();
        copy_b[5] = 9999;
        let mut copy_c = copy_b.clone();
        copy_c[10] = 8888;
        let mut tokens: Vec<u32> = (0..20).collect();
        tokens.extend_from_slice(&base);
        tokens.extend_from_slice(&copy_b);
        tokens.extend_from_slice(&copy_c);
        let hit = detect_fuzzy_repetition(&tokens, DEFAULT_TOLERANCE_DIV);
        assert!(
            hit.is_some(),
            "expected detection of near-identical triple, got None"
        );
    }

    #[test]
    fn does_not_fire_on_exact_boundary_double_match() {
        let pattern: Vec<u32> = (3000..3020).collect();
        let mut mutated = pattern.clone();
        mutated[4] = 9001;
        mutated[12] = 9002;
        let mut tokens: Vec<u32> = (0..50).collect();
        tokens.extend_from_slice(&pattern);
        tokens.extend_from_slice(&mutated);
        for t in 4000..4060 {
            tokens.push(t);
        }
        assert_eq!(
            detect_fuzzy_repetition(&tokens, DEFAULT_TOLERANCE_DIV),
            None
        );
    }
}
