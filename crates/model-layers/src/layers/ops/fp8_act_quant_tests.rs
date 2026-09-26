// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Host tests of the Hopper FP8 activation quantizer's group and
//! element mapping, and of its width floor.
//!
//! Owner: model-layers ops.
//! Invariants: none beyond the types.
//!
//! `native_fp8_act_quant_hopper_microtest` compares the two kernels' bytes on
//! a device. These tests check that the launcher's grid and the kernel's span
//! cover every K-group of every token exactly once: a gap leaves a scale
//! unwritten, and an overlap has two CTAs writing one scale. The oracle is the
//! `.cu`'s arithmetic, restated in [`fp8_quant_hopper_span`] and
//! [`fp8_quant_hopper_lane`].

use super::*;
// 2026-09-25: The floor, the refusal reasons and the log slots are in the
// sibling module `fp8_act_quant_floor.rs`, which `super` does not include;
// `layers::ops` re-exports both modules.
use crate::layers::ops::{
    FP8_QUANT_LOG_SLOTS, FP8_QUANT_MIN_CTAS_PER_SM, FP8_QUANT_REJECTS, FP8_QUANT_TOO_FEW_CTAS,
    Fp8QuantLogSlot, fp8_quant_hopper_ctas, fp8_quant_hopper_min_m, fp8_quant_log_slot,
    fp8_quant_min_ctas,
};

/// 2026-09-25: The (M, K) shapes these tests use; every pair is a row of
/// [`MEASURED`].
const K_DIMS: [u32; 3] = [5120, 6144, 17408];
const M_DIMS: [u32; 5] = [16, 17, 25, 1168, 4576];

/// 2026-09-25: The Hopper grid and the kernel's span partition `0..K/128`.
#[test]
fn the_hopper_grid_covers_every_k_group_exactly_once() {
    for k in K_DIMS {
        let groups = k / 128;
        let [_, grid_y, _] = fp8_quant_grid(true, 1, k);
        let mut seen = vec![0u32; groups as usize];
        for by in 0..grid_y {
            let (g0, g1) = fp8_quant_hopper_span(groups, grid_y, by);
            for g in g0..g1 {
                seen[g as usize] += 1;
            }
        }
        assert!(
            seen.iter().all(|&c| c == 1),
            "K={k}: grid_y={grid_y} does not partition {groups} groups: {seen:?}"
        );
    }
}

/// 2026-09-25: The partition also holds for Y extents the launcher does not
/// pick, because the kernel derives its span from `gridDim.y`.
#[test]
fn any_grid_y_still_partitions_the_groups() {
    for k in K_DIMS {
        let groups = k / 128;
        for grid_y in [1, 2, 3, 5, 7, 8, 17, groups - 1, groups] {
            let mut seen = vec![0u32; groups as usize];
            for by in 0..grid_y {
                let (g0, g1) = fp8_quant_hopper_span(groups, grid_y, by);
                for g in g0..g1 {
                    seen[g as usize] += 1;
                }
            }
            assert!(
                seen.iter().all(|&c| c == 1),
                "K={k} grid_y={grid_y}: not a partition: {seen:?}"
            );
        }
    }
}

/// 2026-09-25: The 128 threads of one CTA cover the 8 groups' 1024 elements
/// exactly once: 16 lanes x 8 elements per group. A thread that covered an
/// element twice would store it twice; one that covered none would leave an
/// FP8 byte unwritten.
#[test]
fn the_hopper_lane_map_covers_a_full_tile_exactly_once() {
    let span = FP8_QUANT_HOPPER_GROUPS_PER_CTA;
    let mut seen = vec![0u32; (span * 128) as usize];
    for tid in 0..128u32 {
        let (sub, lo, hi) = fp8_quant_hopper_lane(tid, span).expect("full tile: every tid is live");
        assert_eq!(hi - lo, 8, "tid {tid} must own 8 elements (one uint4)");
        for e in lo..hi {
            seen[(sub * 128 + e) as usize] += 1;
        }
    }
    assert!(
        seen.iter().all(|&c| c == 1),
        "the 16x8 lane map is not a partition of the 8x128 tile"
    );
}

/// 2026-09-25: A CTA whose span is shorter than 8 groups: the threads past the
/// span are inert, and every element of the live groups is covered once.
#[test]
fn a_partial_tile_leaves_the_spare_threads_inert() {
    for span in 1..FP8_QUANT_HOPPER_GROUPS_PER_CTA {
        let mut seen = vec![0u32; (span * 128) as usize];
        let mut inert = 0;
        for tid in 0..128u32 {
            match fp8_quant_hopper_lane(tid, span) {
                None => inert += 1,
                Some((sub, lo, hi)) => {
                    for e in lo..hi {
                        seen[(sub * 128 + e) as usize] += 1;
                    }
                }
            }
        }
        assert_eq!(
            inert,
            (128 - span * 16) as usize,
            "span {span}: wrong number of inert threads"
        );
        assert!(
            seen.iter().all(|&c| c == 1),
            "span {span}: live groups not covered exactly once"
        );
    }
}

/// 2026-09-25: The shared arm's grid is `(M, K/128, 1)`, one CTA per group.
/// The Hopper arm has `ceil(K/128 / 8)` CTAs in Y, fewer than the shared arm.
#[test]
fn the_shared_grid_is_untouched_and_the_hopper_grid_is_an_eighth_of_it() {
    for k in K_DIMS {
        for m in M_DIMS {
            let shared = fp8_quant_grid(false, m, k);
            let hopper = fp8_quant_grid(true, m, k);
            assert_eq!(
                shared,
                [m, k / 128, 1],
                "shared grid changed for M={m} K={k}"
            );
            assert_eq!(hopper[0], m);
            assert_eq!(hopper[2], 1);
            assert_eq!(hopper[1], (k / 128).div_ceil(8));
            assert!(hopper[1] < shared[1], "M={m} K={k}: no CTA reduction");
        }
    }
}

/// 2026-09-25: `Fp8ActQuant` picks the entry point and the grid from the same
/// flag. The shared kernel on the Hopper grid would quantize only the first
/// `ceil(K/128 / 8)` groups of each row and leave the rest of the scratch
/// stale.
#[test]
fn the_pair_never_mixes_one_kernels_handle_with_the_others_grid() {
    let shared = Fp8ActQuant::shared_only(KernelHandle(0xA1));
    assert!(shared.available() && !shared.twin_present());
    assert_eq!(shared.kernel(1168, 5120).0, 0xA1);
    assert_eq!(shared.grid(1168, 5120), fp8_quant_grid(false, 1168, 5120));

    let twin = PAIR;
    assert!(twin.available() && twin.twin_present());
    for (m, k) in [(1168, 5120), (16, 5120)] {
        let pick = twin.pick_with(true, m, k, SM);
        assert_eq!(pick.kernel.0, if pick.twin { 0xB2 } else { 0xA1 });
        assert_eq!(pick.grid, fp8_quant_grid(pick.twin, m, k));
    }

    assert!(!Fp8ActQuant::default().available());
    assert!(!Fp8ActQuant::shared_only(KernelHandle(0)).available());
}

// 2026-09-25: The width floor (`fp8_act_quant_floor.rs`), graded at the points
// in `MEASURED`.

/// 2026-09-25: Hopper's `sm_count` (`kernels/hopper/HARDWARE.toml`).
const SM: u32 = 132;

/// 2026-09-25: A pair with both handles loaded, as a Hopper image resolves.
const PAIR: Fp8ActQuant = Fp8ActQuant {
    shared: KernelHandle(0xA1),
    hopper: KernelHandle(0xB2),
};

/// 2026-09-25: The (M, K) points and the arm expected to take each (`true` =
/// the Hopper twin). This table is the oracle: a floor change that moves a row
/// fails the test.
const MEASURED: [(u32, u32, bool); 15] = [
    (16, 5120, false),
    (17, 5120, false),
    (25, 5120, false),
    (1168, 5120, true),
    (4576, 5120, true),
    (16, 6144, false),
    (17, 6144, false),
    (25, 6144, false),
    (1168, 6144, true),
    (4576, 6144, true),
    // 2026-09-25: K = 17408: grid Y is 17, and 16 x 17 = 272 >= 264 CTAs, so
    // the twin takes M = 16.
    (16, 17408, true),
    (17, 17408, true),
    (25, 17408, true),
    (1168, 17408, true),
    (4576, 17408, true),
];

/// 2026-09-25: The floor routes each point in [`MEASURED`] to its expected arm,
/// with the handle and the grid from one decision.
#[test]
fn the_floor_routes_every_measured_arm_to_the_faster_kernel() {
    for (m, k, twin_won) in MEASURED {
        let pick = PAIR.pick_with(true, m, k, SM);
        assert_eq!(
            pick.twin,
            twin_won,
            "M={m} K={k}: routed to the {} but round 16 measured the {} faster \
             ({} CTAs against a floor of {})",
            if pick.twin { "twin" } else { "parent" },
            if twin_won { "twin" } else { "parent" },
            fp8_quant_hopper_ctas(m, k),
            fp8_quant_min_ctas(SM),
        );
        assert_eq!(pick.kernel.0, if twin_won { 0xB2 } else { 0xA1 });
        assert_eq!(pick.grid, fp8_quant_grid(twin_won, m, k));
        assert_eq!(pick.reject.is_none(), twin_won);
    }
}

/// 2026-09-25: The per-K thresholds on a 132-SM part: the threshold M takes
/// the twin and the M below it does not.
#[test]
fn the_per_k_thresholds_are_the_documented_ones() {
    for (k, min_m) in [(5120_u32, 53_u32), (6144, 44), (17408, 16)] {
        assert_eq!(
            fp8_quant_hopper_min_m(k, SM),
            min_m,
            "K={k}: grid Y is {}, floor {} CTAs",
            fp8_quant_grid(true, 1, k)[1],
            fp8_quant_min_ctas(SM),
        );
        assert!(PAIR.pick_with(true, min_m, k, SM).twin, "K={k} M={min_m}");
        assert!(
            !PAIR.pick_with(true, min_m - 1, k, SM).twin,
            "K={k} M={}: the threshold must be the smallest accepted M",
            min_m - 1
        );
    }
}

/// 2026-09-25: Each refusal reason has its own log slot, and an unrequested
/// pick logs nothing.
#[test]
fn every_reject_reason_has_its_own_log_slot() {
    let cases = [
        ("not requested", PAIR.pick_with(false, 4576, 5120, SM)),
        (
            "kernel absent from this image (kernels/hopper only)",
            Fp8ActQuant::shared_only(KernelHandle(0xA1)).pick_with(true, 4576, 5120, SM),
        ),
        (FP8_QUANT_TOO_FEW_CTAS, PAIR.pick_with(true, 16, 5120, SM)),
    ];
    for (why, pick) in cases {
        assert!(!pick.twin);
        assert_eq!(pick.reject, Some(why));
        assert!(
            FP8_QUANT_REJECTS.contains(&why),
            "{why:?} has no slot in FP8_QUANT_REJECTS"
        );
        let slot = fp8_quant_log_slot(&pick);
        if pick.requested {
            assert_eq!(
                slot,
                Some(Fp8QuantLogSlot::Reject(
                    FP8_QUANT_REJECTS.iter().position(|r| *r == why).unwrap()
                ))
            );
        } else {
            assert_eq!(slot, None, "the lever is off: the parent is the answer");
        }
    }
    assert_eq!(
        fp8_quant_log_slot(&PAIR.pick_with(true, 4576, 5120, SM)),
        Some(Fp8QuantLogSlot::Twin)
    );
}

/// 2026-09-25: Decode-width calls followed by prefill-width calls land in two
/// distinct slots, so a serve logs both lines.
#[test]
fn a_decode_then_prefill_serve_fills_two_distinct_slots() {
    let slots: Vec<_> = [(27, 5120), (16, 5120), (4576, 5120), (1168, 6144)]
        .into_iter()
        .filter_map(|(m, k)| fp8_quant_log_slot(&PAIR.pick_with(true, m, k, SM)))
        .collect();
    let reject = Fp8QuantLogSlot::Reject(
        FP8_QUANT_REJECTS
            .iter()
            .position(|r| *r == FP8_QUANT_TOO_FEW_CTAS)
            .unwrap(),
    );
    assert_eq!(
        slots,
        vec![reject, reject, Fp8QuantLogSlot::Twin, Fp8QuantLogSlot::Twin],
        "the negative and the positive must not share a once-flag"
    );
    assert_eq!(FP8_QUANT_LOG_SLOTS, FP8_QUANT_REJECTS.len() + 1);
}

/// 2026-09-25: A pair with no shared kernel runs the twin at every width.
/// `native_fp8_act_quant_hopper_microtest` builds this pair to launch the twin
/// at every M.
#[test]
fn a_pair_with_no_parent_keeps_the_twin_at_every_width() {
    let only_twin = Fp8ActQuant {
        shared: KernelHandle(0),
        hopper: KernelHandle(0xB2),
    };
    for (m, k, _) in MEASURED {
        let pick = only_twin.pick_with(true, m, k, SM);
        assert!(pick.twin && pick.reject.is_none(), "M={m} K={k}");
        assert_eq!(pick.kernel.0, 0xB2);
    }
    // 2026-09-25: An unrequested pick does not change that; refusing would
    // launch `KernelHandle(0)`.
    assert!(only_twin.pick_with(false, 4576, 5120, SM).twin);
    // 2026-09-25: With a shared kernel in the pair, the same unrequested pick
    // declines the twin.
    assert!(!PAIR.pick_with(false, 4576, 5120, SM).twin);
}

/// 2026-09-25: The floor scales with the SM count, and an SM count of 0 still
/// gives a defined floor.
#[test]
fn the_floor_follows_the_sm_count() {
    assert_eq!(fp8_quant_min_ctas(132), 264);
    assert_eq!(fp8_quant_min_ctas(148), 296);
    assert_eq!(fp8_quant_min_ctas(0), FP8_QUANT_MIN_CTAS_PER_SM);
    // 2026-09-25: 148 SMs (`kernels/b200/HARDWARE.toml`) raise the threshold.
    assert!(fp8_quant_hopper_min_m(5120, 148) > fp8_quant_hopper_min_m(5120, 132));
    // 2026-09-25: K below one group: grid Y is clamped to 1, so the CTA count
    // is M.
    assert_eq!(fp8_quant_hopper_ctas(64, 64), 64);
    assert_eq!(fp8_quant_hopper_min_m(64, 132), 264);
}
