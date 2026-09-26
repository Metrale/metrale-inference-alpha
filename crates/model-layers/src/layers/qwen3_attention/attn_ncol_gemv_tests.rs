// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Band, lever and handle-presence edges of `ncol_plan`, the N-column-blocked decode
//! tier's selection rule. Numeric parity is not tested here.
//!
//! Owner: model-layers attention decode.
//! Invariants: none beyond the types.

use super::{NcolWidth, ncol_plan};

/// 2026-09-25: Both handles present, lever on.
fn plan(m: usize, width: NcolWidth) -> Option<NcolWidth> {
    ncol_plan(m, width, true, true, true)
}

#[test]
fn selects_across_the_batch16_band() {
    for m in [5, 6, 8, 12, 15, 16] {
        assert_eq!(
            plan(m, NcolWidth::Two),
            Some(NcolWidth::Two),
            "m={m} is inside the 5..=16 band"
        );
    }
}

#[test]
fn width_four_is_selected_when_asked_for() {
    assert_eq!(plan(16, NcolWidth::Four), Some(NcolWidth::Four));
    assert_eq!(NcolWidth::Four.cols(), 4);
    assert_eq!(NcolWidth::Two.cols(), 2);
}

/// 2026-09-25: Rows 0..=4 are below the band.
#[test]
fn declines_below_the_band() {
    for m in [0, 1, 2, 3, 4] {
        assert_eq!(plan(m, NcolWidth::Two), None, "m={m} is below the band");
    }
}

/// 2026-09-25: The upper edge is the kernel's `MAX_M` of 16. The kernel would compute rows 0..15
/// and leave the rest unwritten; the op wrapper refuses such a launch.
#[test]
fn declines_above_the_kernel_max_m() {
    for m in [17, 20, 32] {
        assert_eq!(plan(m, NcolWidth::Two), None, "m={m} is above MAX_M=16");
    }
}

#[test]
fn lever_off_declines_every_width() {
    for m in [5, 8, 16] {
        assert_eq!(ncol_plan(m, NcolWidth::Two, true, true, false), None);
        assert_eq!(ncol_plan(m, NcolWidth::Four, true, true, false), None);
    }
}

/// 2026-09-25: With only one instantiation loaded, the other width declines rather than
/// substituting.
#[test]
fn a_missing_entry_point_declines_that_width_only() {
    assert_eq!(ncol_plan(16, NcolWidth::Four, true, false, true), None);
    assert_eq!(
        ncol_plan(16, NcolWidth::Two, true, false, true),
        Some(NcolWidth::Two)
    );
    assert_eq!(ncol_plan(16, NcolWidth::Two, false, true, true), None);
    assert_eq!(
        ncol_plan(16, NcolWidth::Four, false, true, true),
        Some(NcolWidth::Four)
    );
}
