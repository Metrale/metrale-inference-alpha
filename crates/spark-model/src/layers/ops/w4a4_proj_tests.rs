// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

#[test]
fn routes_every_27b_projection_shape_up_to_32_rows() {
    for k in [5120u32, 6144, 17408] {
        for m in 1..=32 {
            assert!(w4a4_route(m, 5120, k, true), "m={m} k={k}");
        }
        assert!(!w4a4_route(33, 5120, k, true), "33 rows exceed mx32");
        assert!(!w4a4_route(4, 5120, k, false), "opt-in");
    }
}

#[test]
fn declines_what_the_kernel_or_scratch_cannot_hold() {
    assert!(!w4a4_route(4, 5120, 5120 + 32, true), "K % 64");
    assert!(!w4a4_route(4, 5120, W4A4_MAX_K + 64, true), "scratch K");
    assert!(!w4a4_route(0, 5120, 5120, true));
}

/// Distinct fake handles so a pick can be identified by value.
fn state() -> W4a4State {
    let k = |v: u64| KernelHandle(v);
    W4a4State {
        quant: k(1),
        mx8: k(8),
        mx16: k(16),
        mx32: k(32),
        mx16_nt2: k(162),
        mx32_nt4: k(324),
        aq: DevicePtr::NULL,
        a_scale: DevicePtr::NULL,
        a_gs: DevicePtr::NULL,
        audit_ref: DevicePtr::NULL,
    }
}

fn pick(s: &W4a4State, m: u32, nt: u32) -> (u64, u32) {
    let (k, rows) = mx_pick(s, m, nt);
    (k.0, rows)
}

#[test]
fn nt1_picks_the_historical_kernels_at_16_rows_per_cta() {
    let s = state();
    for (m, want) in [(1, 8), (8, 8), (9, 16), (16, 16), (17, 32), (32, 32)] {
        assert_eq!(pick(&s, m, 1), (want, 16), "m={m}");
    }
}

#[test]
fn tile_factor_picks_the_twin_and_its_grid_rows() {
    let s = state();
    // The 1..=8-row arm has no twin: mx8 at every tile factor.
    assert_eq!(pick(&s, 8, 4), (8, 16));
    // 9..=16 caps at NT=2.
    assert_eq!(pick(&s, 9, 4), (162, 32));
    assert_eq!(pick(&s, 16, 4), (162, 32));
    assert_eq!(pick(&s, 17, 4), (324, 64));
    assert_eq!(pick(&s, 32, 4), (324, 64));
}
