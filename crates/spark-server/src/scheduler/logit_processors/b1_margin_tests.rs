// SPDX-License-Identifier: AGPL-3.0-only

use super::low_margin_in_body;

#[test]
fn inactive_body_never_reads_logits() {
    for (inside, chars) in [(false, 0), (false, 1), (true, 0)] {
        let logits = std::iter::from_fn(|| -> Option<f32> {
            panic!("inactive B1 observer must not scan logits")
        });
        assert_eq!(low_margin_in_body(logits, inside, chars), None);
    }
}

#[test]
fn active_body_preserves_margin_threshold_and_indices() {
    assert_eq!(
        low_margin_in_body([1.0, 3.0, 2.0].into_iter(), true, 1),
        Some((1.0, 1, 2))
    );
    assert_eq!(
        low_margin_in_body([3.0, 3.0, 1.0].into_iter(), true, 1),
        Some((0.0, 0, 1))
    );
    assert_eq!(low_margin_in_body([3.0, 1.5].into_iter(), true, 1), None);
    assert_eq!(
        low_margin_in_body([3.0, 1.500001].into_iter(), true, 1),
        Some((3.0 - 1.500001, 0, 1))
    );
}

#[test]
fn active_body_matches_historical_scan_including_nonfinite() {
    // Original scan is the oracle; preserve its strict comparisons and NaN
    // behavior, rather than substituting a sort or sampler tie convention.
    fn historical(logits: &[f32]) -> Option<(f32, u32, u32)> {
        let (mut top1, mut top2) = ((0, f32::NEG_INFINITY), (0, f32::NEG_INFINITY));
        for (i, &value) in logits.iter().enumerate() {
            if value > top1.1 {
                top2 = top1;
                top1 = (i as u32, value);
            } else if value > top2.1 {
                top2 = (i as u32, value);
            }
        }
        let gap = top1.1 - top2.1;
        (gap < 1.5).then_some((gap, top1.0, top2.0))
    }
    let values = [
        f32::NEG_INFINITY,
        -1.0,
        -0.0,
        0.0,
        1.0,
        f32::INFINITY,
        f32::NAN,
    ];
    assert_eq!(low_margin_in_body([].into_iter(), true, 1), None);
    for x in values {
        for y in values {
            for z in values {
                let logits = [x, y, z];
                assert_eq!(
                    low_margin_in_body(logits.into_iter(), true, 1),
                    historical(&logits),
                    "logits={logits:?}"
                );
            }
        }
    }
}
