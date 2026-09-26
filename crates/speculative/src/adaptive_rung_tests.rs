// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Unit tests for the adaptive rung: the token-ratio rule, the
//! hysteresis band, the adapted width band, probing, and the width-regime
//! counter. Each test builds its own `AdaptiveRung`.
//!
//! Owner: speculative.
//! Invariants: none beyond the types.

use super::*;

const P: RungParams = RungParams::DEFAULTS;

/// 2026-09-25: Telemetry points as `(label, p1, p2_cond, two drafts
/// expected)`: three prose workloads and one tool-shaped workload.
const W27: [(&str, f64, f64, bool); 4] = [
    ("decode_short L=64", 0.840, 0.542, false),
    ("decode_short L=128", 0.773, 0.539, false),
    ("decode_short L=1024", 0.732, 0.567, false),
    ("tool-shaped natural EOS", 0.917, 0.877, true),
];

#[test]
fn token_ratio_reproduces_wave_27() {
    assert!((token_ratio(0.917, 0.877) - 1.424).abs() < 0.01);
    assert!((token_ratio(0.732, 0.567) - 1.229).abs() < 0.015);
    assert_eq!(token_ratio(0.0, 0.9), 1.0);
    assert_eq!(token_ratio(f64::NAN, 0.9), 1.0);
    assert_eq!(token_ratio(0.9, 0.0), 1.0);
}

#[test]
fn p2_cond_inverts_mean_na() {
    let (p1, p2) = (0.917, 0.877);
    let mean_na = p1 + p1 * p2;
    assert!((p2_cond_from(p1, mean_na).unwrap() - p2).abs() < 1e-9);
    assert_eq!(p2_cond_from(p1, p1).unwrap(), 0.0);
    assert_eq!(p2_cond_from(0.5, 2.0).unwrap(), 1.0);
    assert_eq!(p2_cond_from(0.0, 0.5), None);
}

#[test]
fn rule_classifies_every_wave_27_point_the_way_the_box_did() {
    for (label, p1, p2, depth_wins) in W27 {
        let tr = token_ratio(p1, p2);
        // 2026-09-25: Checked from both states: for a point inside the dead
        // band the decision would depend on the current state.
        assert!(
            !(LEAVE..ENTER).contains(&tr),
            "{label}: token_ratio {tr:.4} lands INSIDE the dead band"
        );
        assert_eq!(next_state(false, tr, &P), depth_wins, "{label} from k=1");
        assert_eq!(next_state(true, tr, &P), depth_wins, "{label} from k=2");
    }
}

#[test]
fn dead_band_holds_state_and_is_asymmetric() {
    let mid = (ENTER + LEAVE) / 2.0;
    assert!(
        !next_state(false, mid, &P),
        "must not enter depth on weak evidence"
    );
    assert!(
        next_state(true, mid, &P),
        "must not leave depth on weak evidence"
    );
    const { assert!(ENTER > LEAVE) };
    assert!(next_state(false, ENTER, &P));
    assert!(!next_state(true, LEAVE - 1e-9, &P));
}

#[test]
fn scope_is_the_n16_rung_only() {
    let rung = AdaptiveRung::new(P);
    // 2026-09-25: Outside 9..=16 the count is the default ladder's
    // (`4:3,8:3,16:1,32:1`, last step beyond n=32).
    for n in [1usize, 4, 8] {
        assert_eq!(rung.drafts_for(n, 3), 3, "n={n} must keep the 8:3 rung");
    }
    for n in [17usize, 24, 32, 64] {
        assert_eq!(rung.drafts_for(n, 3), 1, "n={n} must keep the 32:1 rung");
    }
    // 2026-09-25: `num_drafts` stays the ceiling inside the band too.
    assert_eq!(rung.drafts_for(16, 1), 1);
    assert_eq!(rung.drafts_for(16, 0), 0);
}

#[test]
fn converges_and_holds_without_oscillating() {
    let rung = AdaptiveRung::new(P);
    // 2026-09-25: One `flush` stands for one accept-statistics flush of the
    // scheduler (`PERIOD` = 128 verifies in `mtp_accept_debug`).
    let flush = |p1: f64, p2: f64, k: usize| {
        let mean_na = if k >= 2 { p1 + p1 * p2 } else { p1 };
        rung.observe(16, k, p1, mean_na);
    };

    // 2026-09-25: Cold start: p2_cond has never been observed, so a probe is
    // due.
    assert_eq!(rung.drafts_for(16, 3), 2, "cold start must probe at depth");

    // 2026-09-25: The prose probe scores a ratio below `ENTER`; the state
    // stays at one draft and the probe ends.
    let (_, pp1, pp2, _) = W27[2];
    flush(pp1, pp2, 2);
    assert_eq!(
        rung.drafts_for(16, 3),
        1,
        "prose must settle at k=1 after one probe"
    );

    // 2026-09-25: Steady prose with per-flush noise on p1 must buy no probe.
    // The noise is the sum of three uniforms scaled to sigma 0.053, from a
    // fixed-seed xorshift, so the test is deterministic.
    let mut rng: u64 = 0x9E3779B97F4A7C15;
    let mut noise = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        let u = |b: u32| ((rng >> b) & 0xFFFF) as f64 / 65535.0 - 0.5;
        (u(0) + u(16) + u(32)) * 0.106
    };
    for t in 0..200 {
        flush((pp1 + noise()).clamp(0.0, 1.0), 0.0, 1);
        assert_eq!(
            rung.drafts_for(16, 3),
            1,
            "spurious probe at tick {t} on steady p1"
        );
    }

    // 2026-09-25: Tool-shaped traffic raises p1 from 0.732 to 0.917. The slow
    // EWMA crosses `P1_TRIGGER` within a few flushes, long before the
    // `PROBE_TICKS` backstop.
    let (_, tp1, tp2, _) = W27[3];
    let mut ticks_to_probe = 0;
    while rung.drafts_for(16, 3) != 2 && ticks_to_probe < 20 {
        flush(tp1, 0.0, 1);
        ticks_to_probe += 1;
    }
    assert!(
        ticks_to_probe <= 8,
        "p1 jump took {ticks_to_probe} flushes to trigger a probe"
    );

    // 2026-09-25: The probe's ratio reaches `ENTER`, so the state moves to two
    // drafts on that flush.
    flush(tp1, tp2, 2);
    assert!(rung.at_depth(), "tool traffic never reached k=2");

    // 2026-09-25: At two drafts every flush is a depth flush, so steady tool
    // traffic holds the state without a flip.
    let before = rung.flips();
    for t in 0..60 {
        assert_eq!(rung.drafts_for(16, 3), 2, "tool oscillated at tick {t}");
        flush(tp1, tp2, 2);
    }
    assert_eq!(rung.flips(), before, "rung oscillated");

    // 2026-09-25: Prose at two drafts takes the ratio below `LEAVE` within a
    // few flushes.
    let mut ticks = 0;
    while rung.at_depth() && ticks < 20 {
        flush(pp1, pp2, 2);
        ticks += 1;
    }
    assert!(!rung.at_depth(), "never left depth");
    assert!(ticks <= 4, "took {ticks} flushes to leave depth");
}

/// 2026-09-25: `note_width_regime` counts one flip per change of `engaged`
/// and none for a repeated value.
#[test]
fn width_regime_flips_once_per_transition() {
    let rung = AdaptiveRung::new(P);
    let base = rung.width_flips();
    for _ in 0..50 {
        rung.note_width_regime(16, true, 32);
    }
    assert_eq!(rung.width_flips(), base, "engaged->engaged flipped");
    for _ in 0..50 {
        rung.note_width_regime(64, false, 32);
    }
    assert_eq!(
        rung.width_flips(),
        base + 1,
        "crossing the cap must count exactly one flip"
    );
    rung.note_width_regime(32, true, 32);
    assert_eq!(rung.width_flips(), base + 2);
    assert!(rung.width_engaged());
}
