// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for the state value type and for the delta between two
//! captures.
//!
//! Owner: bench hardware.
//! Invariants: none beyond the types.

use super::*;

/// 2026-09-26: A cool capture: the throttle counters of
/// `fixtures/gb10_performance.txt`, no thermal reason asserted, one compute app.
fn healthy() -> HardwareState {
    HardwareState {
        captured_at: 1_000,
        machine: MachineIdentity {
            hostname: Some("dgx1".into()),
            machine_id: Some("7af66f30966a49b6886e00e2fce4b42f".into()),
            gpu: Some("NVIDIA GB10".into()),
            driver: Some("580.126.09".into()),
        },
        gpu_temp_c: Some(55.0),
        chassis_temps_c: Some(
            [65.0, 65.0, 62.0, 59.0, 59.0, 58.0]
                .into_iter()
                .map(|t| ThermalZone {
                    name: "acpitz".into(),
                    temp_c: t,
                })
                .collect(),
        ),
        sm_clock_mhz: Some(2_990.0),
        sm_clock_max_mhz: Some(3_003.0),
        throttle_counters: ThrottleCounters {
            sw_power_cap_us: Some(16_130_111_127),
            sw_thermal_us: Some(502_088_297),
            hw_thermal_us: Some(704_014),
            hw_power_brake_us: Some(0),
            sync_boost_us: Some(0),
        },
        throttle_active: ThrottleActive {
            sw_power_cap: Some(false),
            sw_thermal: Some(false),
            hw_thermal: Some(false),
            hw_power_brake: Some(false),
        },
        mem_available_kb: Some(100_000_000),
        mem_total_kb: Some(127_601_452),
        page_cache_kb: Some(1_098_384),
        gpu_compute_apps: Some(vec![GpuComputeApp {
            pid: 2_392_883,
            name: "./target/release/met".into(),
            used_mib: Some(109_281),
        }]),
        cpu_governor: Some("performance".into()),
        persistence_mode: Some(true),
        sources: vec!["nvidia-smi".into(), "procfs".into(), "sysfs".into()],
    }
}

#[test]
fn hottest_chassis_zone_is_the_reading_that_separated_the_boxes() {
    assert_eq!(healthy().hottest_chassis_c(), Some(65.0));
    let hot = HardwareState {
        chassis_temps_c: Some(
            [89.0, 88.0, 82.0, 74.0, 71.0, 70.0]
                .into_iter()
                .map(|t| ThermalZone {
                    name: "acpitz".into(),
                    temp_c: t,
                })
                .collect(),
        ),
        ..HardwareState::default()
    };
    assert_eq!(hot.hottest_chassis_c(), Some(89.0));
}

/// 2026-09-26: Neither an unreadable thermal directory nor an empty zone list
/// yields a hottest zone.
#[test]
fn unreadable_thermal_sysfs_is_unknown_not_cool() {
    assert_eq!(HardwareState::default().hottest_chassis_c(), None);
    let none_present = HardwareState {
        chassis_temps_c: Some(Vec::new()),
        ..HardwareState::default()
    };
    assert_eq!(none_present.hottest_chassis_c(), None);
}

#[test]
fn foreign_compute_apps_excludes_this_process() {
    let mut s = healthy();
    assert_eq!(s.foreign_compute_apps(), Some(1));
    s.gpu_compute_apps = Some(vec![GpuComputeApp {
        pid: std::process::id(),
        name: "met".into(),
        used_mib: Some(1),
    }]);
    assert_eq!(
        s.foreign_compute_apps(),
        Some(0),
        "the TUI serves in-process; its own pid is not contention"
    );
}

/// 2026-09-26: An unread list stays `None` rather than reading as "no foreign
/// processes"; an empty list is a real zero.
#[test]
fn an_unlistable_gpu_leaves_foreign_apps_unknown() {
    let s = HardwareState {
        gpu_compute_apps: None,
        ..HardwareState::default()
    };
    assert_eq!(s.foreign_compute_apps(), None);
    let idle = HardwareState {
        gpu_compute_apps: Some(Vec::new()),
        ..HardwareState::default()
    };
    assert_eq!(
        idle.foreign_compute_apps(),
        Some(0),
        "idle is a real reading"
    );
}

#[test]
fn clock_headroom_needs_both_halves() {
    let s = HardwareState {
        sm_clock_mhz: Some(2_457.0),
        sm_clock_max_mhz: Some(3_003.0),
        ..HardwareState::default()
    };
    assert!((s.clock_headroom().unwrap() - 0.818).abs() < 0.001);
    assert_eq!(HardwareState::default().clock_headroom(), None);
    let no_max = HardwareState {
        sm_clock_mhz: Some(2_457.0),
        sm_clock_max_mhz: Some(0.0),
        ..HardwareState::default()
    };
    assert_eq!(no_max.clock_headroom(), None, "no divide by a zero ceiling");
}

#[test]
fn perf_class_pairs_the_silicon_class_with_the_box() {
    assert_eq!(healthy().machine.perf_class(), "gb10@dgx1");
    // 2026-09-26: Two records missing different halves never collapse onto the
    // same key.
    assert_eq!(MachineIdentity::default().perf_class(), "unknown@unknown");
    assert_eq!(
        MachineIdentity {
            gpu: Some("NVIDIA GB10".into()),
            ..MachineIdentity::default()
        }
        .perf_class(),
        "gb10@unknown"
    );
    assert_eq!(
        MachineIdentity {
            hostname: Some("dgx2".into()),
            ..MachineIdentity::default()
        }
        .perf_class(),
        "unknown@dgx2"
    );
}

/// 2026-09-26: A run that starts cool and throttles midway shows it only in the delta.
#[test]
fn a_run_that_throttled_midway_is_visible_only_in_the_delta() {
    let before = healthy();
    let mut after = healthy();
    after.captured_at = 1_000 + 692;
    after.throttle_counters.sw_thermal_us = Some(502_088_297 + 12_000_000);
    // 2026-09-26: The pre-state asserts no thermal reason; the delta shows the throttling.
    assert_eq!(before.throttle_active.thermal(), Some(false));
    let d = HardwareStateDelta::between(&before, &after);
    assert_eq!(d.elapsed_s, Some(692));
    assert_eq!(d.sw_thermal_us, Some(12_000_000));
    assert_eq!(d.thermal_throttle_advanced(), Some(true));
    let f = d.thermal_throttle_fraction().unwrap();
    assert!((f - 12.0 / 692.0).abs() < 1e-9);
}

#[test]
fn an_untroubled_run_reports_no_advance() {
    let before = healthy();
    let mut after = healthy();
    after.captured_at = 1_692;
    let d = HardwareStateDelta::between(&before, &after);
    assert_eq!(d.thermal_throttle_advanced(), Some(false));
    assert_eq!(d.thermal_throttle_fraction(), Some(0.0));
}

/// 2026-09-26: A counter that went backwards (a driver reload mid-run) is an
/// unusable reading, not zero throttling.
#[test]
fn a_counter_that_went_backwards_is_unknown_not_zero() {
    let before = healthy();
    let mut after = healthy();
    after.captured_at = 1_692;
    after.throttle_counters.sw_thermal_us = Some(1_000);
    after.throttle_counters.hw_thermal_us = Some(1);
    after.throttle_counters.hw_power_brake_us = None;
    let d = HardwareStateDelta::between(&before, &after);
    assert_eq!(d.sw_thermal_us, None);
    assert_eq!(d.thermal_throttle_advanced(), None);
}

/// 2026-09-26: SW power cap time alone does not make `thermal_throttle_advanced` true.
#[test]
fn sw_power_cap_advance_alone_is_not_a_thermal_event() {
    let before = healthy();
    let mut after = healthy();
    after.captured_at = 1_692;
    after.throttle_counters.sw_power_cap_us = Some(16_130_111_127 + 600_000_000);
    let d = HardwareStateDelta::between(&before, &after);
    assert_eq!(d.sw_power_cap_us, Some(600_000_000));
    assert_eq!(d.thermal_throttle_advanced(), Some(false));
}

#[test]
fn a_delta_between_two_unreadable_captures_is_unknown() {
    let d = HardwareStateDelta::between(&HardwareState::default(), &HardwareState::default());
    assert_eq!(d.thermal_throttle_advanced(), None);
    assert_eq!(d.thermal_throttle_fraction(), None);
}

#[test]
fn state_round_trips_through_json_with_every_field() {
    let s = healthy();
    let back: HardwareState = serde_json::from_str(&serde_json::to_string(&s).unwrap()).unwrap();
    assert_eq!(s, back);
}

/// 2026-09-26: An empty object deserializes to an all-unknown state rather than
/// failing, and an all-unknown state is not a healthy one.
#[test]
fn an_empty_object_deserializes_to_all_unknown() {
    let s: HardwareState = serde_json::from_str("{}").unwrap();
    assert_eq!(s, HardwareState::default());
    assert_eq!(s.foreign_compute_apps(), None);
    assert_eq!(s.throttle_active.thermal(), None);
}

#[test]
fn one_line_names_the_box_and_says_when_something_is_unknown() {
    let line = healthy().one_line();
    assert!(line.starts_with("gb10@dgx1"), "{line}");
    assert!(line.contains("2990/3003 MHz"), "{line}");
    assert!(line.contains("1 foreign gpu proc"), "{line}");
    let blind = HardwareState::default().one_line();
    assert!(blind.contains("foreign gpu proc unknown"), "{blind}");
}

/// 2026-09-26: A fraction over three unreadable counters is `None`, as
/// `thermal_throttle_advanced` is, never the 0.0 of a cool run.
#[test]
fn a_fraction_over_no_readable_counter_is_unknown_not_zero() {
    let d = HardwareStateDelta {
        elapsed_s: Some(692),
        sw_thermal_us: None,
        hw_thermal_us: None,
        hw_power_brake_us: None,
        ..HardwareStateDelta::default()
    };
    assert_eq!(d.thermal_throttle_advanced(), None);
    assert_eq!(d.thermal_throttle_fraction(), None);
    assert!(!d.thermal_counters_complete());
}

/// 2026-09-26: A partial sum is still reported, as a floor, and flagged as one:
/// every unreadable counter moves it downwards.
#[test]
fn a_partial_sum_is_a_floor_and_says_so() {
    let d = HardwareStateDelta {
        elapsed_s: Some(692),
        sw_thermal_us: None,
        hw_thermal_us: Some(12_000_000),
        hw_power_brake_us: Some(0),
        ..HardwareStateDelta::default()
    };
    let f = d
        .thermal_throttle_fraction()
        .expect("one counter was readable");
    assert!(
        (f - 12.0 / 692.0).abs() < 1e-9,
        "the readable counter counts"
    );
    assert!(
        !d.thermal_counters_complete(),
        "an unreadable counter must not pass as a complete sum"
    );
    // 2026-09-26: The fully readable case is not flagged, so the flag discriminates.
    let mut full = d.clone();
    full.sw_thermal_us = Some(0);
    assert!(full.thermal_counters_complete());
}
