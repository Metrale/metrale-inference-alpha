// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for the per-rung instrument keys: the speculation arm,
//! both ITL clocks, the arrival-gap distribution and the GPU-rail energy
//! window. A child module of `concurrency_tests`, so it shares that module's
//! fixture builders.
//!
//! Owner: bench (concurrency).
//! Invariants: none beyond the types.

use super::*;

// 2026-09-26: The speculation arm is published (`c{C}_accept_len`,
// `non_mtp_arm_cells`) and never gated on, so a serial or mixed cell stays
// comparable.

#[test]
fn a_serial_arm_cell_is_reported_but_still_scored() {
    let serial = row(
        2,
        23.5,
        Some(2000.0),
        vec![evidence_with_arm(320, Some(0)); 2],
        320,
    );
    assert_eq!(serial.accept_len(), Some(1.0));
    assert!(serial.arm_is_not_mtp(), "accept_len 1.00 is the serial arm");
    assert!(
        serial.comparable(),
        "a serial cell is still comparable to a floor calibrated on serial"
    );
    assert!(!serial.vacuous);
    assert!(!serial.cache_uncontrolled);
}

#[test]
fn an_mtp_arm_cell_is_comparable() {
    let mtp = row(2, 30.6, Some(2000.0), vec![evidence(320); 2], 320);
    let a = mtp.accept_len().expect("accept_len derivable");
    assert!(
        a > 1.5,
        "the MTP arm sits well above the 1.5 threshold, got {a}"
    );
    assert!(!mtp.arm_is_not_mtp());
    assert!(mtp.comparable());
}

#[test]
fn a_mixed_cell_reports_the_minimum_arm_and_is_still_scored() {
    // 2026-09-26: The minimum governs, so a cell with any serial request
    // reports 1.0.
    let mixed = row(
        2,
        27.5,
        Some(2000.0),
        vec![evidence_with_arm(320, Some(0)), evidence(320)],
        320,
    );
    assert_eq!(mixed.accept_len(), Some(1.0), "the minimum, not the mean");
    assert!(
        mixed.comparable(),
        "reporting the arm must not remove the cell from scoring"
    );
}

#[test]
fn a_missing_accept_field_is_not_read_as_serial() {
    // 2026-09-26: `None` means the server did not report the field; reading it
    // as 1.0 would turn a missing field into a claim about the engine.
    let unknown = row(
        2,
        27.5,
        Some(2000.0),
        vec![evidence_with_arm(320, None); 2],
        320,
    );
    assert_eq!(unknown.accept_len(), None);
    assert!(
        !unknown.arm_is_not_mtp(),
        "an unreported arm is unknown, not serial"
    );
    assert!(
        unknown.comparable(),
        "a sweep from before this instrument existed must stay comparable"
    );
}

#[test]
fn a_corrupt_accept_count_does_not_divide_by_zero() {
    // 2026-09-26: `accepted >= completion` must give `None`, not a division by
    // zero or a negative depth.
    let corrupt = row(
        2,
        27.5,
        Some(2000.0),
        vec![evidence_with_arm(320, Some(320)); 2],
        320,
    );
    assert_eq!(corrupt.accept_len(), None);
    assert!(!corrupt.arm_is_not_mtp());
}

/// 2026-09-26: A clock the server did not report is absent, not zero.
#[test]
fn metrics_map_carries_both_itl_clocks_jitter_and_energy_per_rung() {
    let osl = 128;
    let mut b = configured(vec![4], vec![512]);
    b.osl = osl;
    let mut r = row(4, 100.0, Some(150.0), vec![evidence(128); 4], osl);
    r.tpot = Percentiles {
        p50: Some(31.0),
        p90: Some(35.0),
        p99: Some(40.0),
    };
    r.server_tpot = Percentiles {
        p50: Some(29.5),
        p90: Some(33.0),
        p99: Some(38.0),
    };
    let mut gaps = GapSample::default();
    for g in [30.0, 30.0, 30.0, 31.0, 30.0, 30.0, 30.0, 30.0, 30.0, 180.0] {
        gaps.push(g);
    }
    r.gaps = gaps.stats();
    r.energy = Some(EnergyWindow {
        window_s: 12.0,
        samples: 48,
        energy_j: 720.0,
        mean_power_w: 60.0,
        max_power_w: 66.0,
        sw_power_cap_frac: Some(1.0),
        hw_power_brake_frac: Some(0.0),
    });
    b.rows.push(r);
    let m = b.metrics();
    assert_eq!(m.get("c4_tpot_p50_ms"), Some(&31.0));
    assert_eq!(m.get("c4_tpot_p90_ms"), Some(&35.0));
    assert_eq!(m.get("c4_server_tpot_p50_ms"), Some(&29.5));
    assert_eq!(m.get("c4_server_tpot_p90_ms"), Some(&33.0));
    assert!(
        !m.keys().any(|k| k.contains("itl")),
        "no parallel itl_* key: {m:?}"
    );
    assert_eq!(m.get("c4_arrival_gap_count"), Some(&10.0));
    assert_eq!(m.get("c4_arrival_gap_max_ms"), Some(&180.0));
    assert_eq!(m.get("c4_arrival_gap_p50_ms"), Some(&30.0));
    assert_eq!(m.get("c4_arrival_gap_p99_ms"), Some(&180.0));
    assert_eq!(
        m.get("c4_stability"),
        Some(&5.0),
        "lower is better; a stall raises it"
    );
    assert!(m.contains_key("c4_arrival_gap_cv"));
    assert_eq!(m.get("c4_gpu_rail_energy_j"), Some(&720.0));
    assert_eq!(m.get("c4_gpu_rail_power_samples"), Some(&48.0));
    assert_eq!(m.get("c4_gpu_rail_energy_window_tokens"), Some(&512.0));
    // 2026-09-26: The cell's joules over its delivered tokens.
    assert_eq!(
        m.get("c4_gpu_rail_joules_per_token"),
        Some(&(720.0 / 512.0))
    );
    assert_eq!(m.get("c4_gpu_rail_sw_power_cap_frac"), Some(&1.0));
    // 2026-09-26: No idle baseline, so no above-idle key.
    assert!(!m.contains_key("c4_gpu_rail_energy_above_idle_j"));
    assert!(!m.contains_key("gpu_rail_idle_power_w"));

    // 2026-09-26: Without server-clock or instrument data, only the client
    // clock keys appear.
    let mut old = configured(vec![2], vec![512]);
    old.osl = osl;
    let mut r = row(2, 30.0, Some(120.0), vec![evidence(128); 2], osl);
    r.tpot = Percentiles {
        p50: Some(31.0),
        p90: Some(35.0),
        p99: Some(40.0),
    };
    old.rows.push(r);
    let m = old.metrics();
    assert_eq!(m.get("c2_tpot_p50_ms"), Some(&31.0));
    assert!(!m.contains_key("c2_server_tpot_p50_ms"));
    assert!(
        !m.keys()
            .any(|k| k.contains("arrival_gap") || k.contains("gpu_rail"))
    );
}
