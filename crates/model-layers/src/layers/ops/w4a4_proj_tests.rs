// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests for the W4A4 route predicate and the pure launch planning
//! (`mx_pick`, `mx_plan`) on fake kernel handles and a 48-SM state.
//!
//! Owner: model-layers ops.
//! Invariants: none beyond the types.

use super::*;

#[test]
fn routes_every_27b_projection_shape_up_to_32_rows() {
    for k in [5120u32, 6144, 17408] {
        for m in 1..=32 {
            assert!(w4a4_route(m, 5120, k, true, false), "m={m} k={k}");
        }
        assert!(!w4a4_route(33, 5120, k, true, false), "33 rows exceed mx32");
        assert!(!w4a4_route(4, 5120, k, false, false), "opt-in");
    }
}

#[test]
fn wide_extends_the_route_to_64_rows_and_no_further() {
    for m in 1..=64 {
        assert!(w4a4_route(m, 5120, 5120, true, true), "m={m}");
    }
    assert!(
        !w4a4_route(65, 5120, 5120, true, true),
        "65 rows exceed mx64"
    );
    assert!(
        !w4a4_route(40, 5120, 5120, false, true),
        "wide without downcast"
    );
}

#[test]
fn declines_what_the_kernel_or_scratch_cannot_hold() {
    assert!(!w4a4_route(4, 5120, 5120 + 32, true, false), "K % 64");
    assert!(
        !w4a4_route(4, 5120, W4A4_MAX_K + 64, true, false),
        "scratch K"
    );
    assert!(!w4a4_route(0, 5120, 5120, true, false));
}

/// 2026-09-25: Distinct fake handles, so a pick can be identified by value.
fn state() -> W4a4State {
    let k = |v: u64| KernelHandle(v);
    W4a4State {
        quant: k(1),
        mx8: k(8),
        mx16: k(16),
        mx32: k(32),
        mx16_nt2: k(162),
        mx32_nt4: k(324),
        mx16_ps: k(1600),
        mx32_ps: k(3200),
        sms: 48,
        mx64: k(64),
        mx64_nt2: k(642),
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
    for (m, want) in [
        (1, 8),
        (8, 8),
        (9, 16),
        (16, 16),
        (17, 32),
        (32, 32),
        (33, 64),
        (64, 64),
    ] {
        assert_eq!(pick(&s, m, 1), (want, 16), "m={m}");
    }
}

#[test]
fn tile_factor_picks_the_twin_and_its_grid_rows() {
    let s = state();
    // 2026-09-25: The 1..=8-row arm has no twin: mx8 at every tile factor.
    assert_eq!(pick(&s, 8, 4), (8, 16));
    assert_eq!(pick(&s, 9, 4), (162, 32));
    assert_eq!(pick(&s, 16, 4), (162, 32));
    assert_eq!(pick(&s, 17, 4), (324, 64));
    assert_eq!(pick(&s, 32, 4), (324, 64));
    assert_eq!(pick(&s, 33, 4), (642, 32));
    assert_eq!(pick(&s, 64, 4), (642, 32));
}

/// 2026-09-25: A launch as comparable values: (kernel, rows per CTA or 0, sst,
/// smem).
type Plan = (u64, u32, u32, u32);

fn plan(m: u32, n: u32, k: u32, nt: u32, ps: bool) -> Plan {
    match mx_plan(&state(), m, n, k, nt, ps) {
        MxLaunch::Tiles {
            kernel,
            rows_per_cta,
        } => (kernel.0, rows_per_cta, 0, 0),
        MxLaunch::Persistent { kernel, sst, smem } => (kernel.0, 0, sst, smem),
    }
}

fn tiles(kernel: u64, rows_per_cta: u32) -> Plan {
    (kernel, rows_per_cta, 0, 0)
}

fn persistent(kernel: u64, sst: u32, smem: u32) -> Plan {
    (kernel, 0, sst, smem)
}

#[test]
fn persistent_serves_the_wide_k5120_projections() {
    // 2026-09-25: At K=5120 the whole stripe (5 chunks) fits for 9..=32 rows;
    // at 17..=32 rows it takes 100,352 of the 101,376 B.
    for n in [17408, 16384, 12288, 10240, 6144] {
        for m in [17, 32] {
            let want = persistent(3200, 5, 100_352);
            assert_eq!(plan(m, n, 5120, 4, true), want, "m={m} n={n}");
        }
        for m in [9, 16] {
            let want = persistent(1600, 5, 54_272);
            assert_eq!(plan(m, n, 5120, 4, true), want, "m={m} n={n}");
        }
    }
}

#[test]
fn persistent_declines_what_it_cannot_stage_or_amortise() {
    // 2026-09-25: At 32 rows the K=6144 and K=17408 stripes do not fit in
    // shared memory, so the twin serves them.
    assert_eq!(plan(32, 5120, 6144, 4, true), tiles(324, 64));
    assert_eq!(plan(32, 5120, 17408, 4, true), tiles(324, 64));
    // 2026-09-25: At 16 rows the K=6144 stripe fits, but N=5120 is 320 tiles,
    // under 8 per SM; the K=17408 stripe does not fit.
    assert_eq!(plan(16, 5120, 6144, 4, true), tiles(162, 32));
    assert_eq!(plan(16, 5120, 17408, 4, true), tiles(162, 32));
    // 2026-09-25: N=1024 is 64 tiles, under 8 per SM.
    assert_eq!(plan(32, 1024, 5120, 4, true), tiles(324, 64));
    // 2026-09-25: The persistent entries stop at 32 rows, so 33..=64 rows take
    // the wide kernels even with PS on.
    assert_eq!(plan(33, 17408, 5120, 4, true), tiles(642, 32));
    assert_eq!(plan(64, 17408, 5120, 4, true), tiles(642, 32));
    assert_eq!(plan(64, 17408, 5120, 1, true), tiles(64, 16));
    // 2026-09-25: The smallest N that takes the persistent entry is exactly 8
    // tiles per SM.
    let edge = 8 * 48 * 16;
    assert_eq!(plan(32, edge, 5120, 4, true), persistent(3200, 5, 100_352));
    assert_eq!(plan(32, edge - 16, 5120, 4, true), tiles(324, 64));
}

#[test]
fn persistent_never_touches_the_8_row_arm_or_the_reverts() {
    for m in [1, 8] {
        assert_eq!(plan(m, 17408, 5120, 4, true), tiles(8, 16), "m={m}");
    }
    // 2026-09-25: NT=1 selects the one-tile kernels even with PS on.
    assert_eq!(plan(32, 17408, 5120, 1, true), tiles(32, 16));
    assert_eq!(plan(16, 17408, 5120, 1, true), tiles(16, 16));
    assert_eq!(plan(32, 17408, 5120, 4, false), tiles(324, 64));
    assert_eq!(plan(16, 17408, 5120, 4, false), tiles(162, 32));
}

#[test]
fn smem_contract_matches_the_kernel_layout() {
    // 2026-09-25: 8 warps x sst x mb x (32 lanes x 16 B + 8 g x 8 B) + 2 x 4 KiB
    // reduction.
    assert_eq!(ps_smem_bytes(4, 0), 8192);
    assert_eq!(ps_smem_bytes(4, 1), 8 * 4 * (32 * 16 + 8 * 8) + 8192);
    assert_eq!(ps_stripe_chunks(5120), 5);
    assert_eq!(ps_stripe_chunks(6144), 6);
    assert_eq!(ps_stripe_chunks(17408), 17);
    assert_eq!(ps_column_blocks(9), 2);
    assert_eq!(ps_column_blocks(17), 4);
}
