// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for the verdict policy as a pure function of
//! `(sensitivity, state, options)`: no env, no GPU, no I/O.
//!
//! Owner: bench hardware.
//! Invariants: none beyond the types.
//!
//! Asserted from several directions: an unreadable throttle-reason list or
//! compute-process list never yields `Decision::Proceed`, and unreadable thermal
//! counters never yield `Validity::Valid`.

use super::*;
use crate::hardware::state::{GpuComputeApp, ThermalZone, ThrottleActive};

/// 2026-09-26: The GB10 ceilings (die 75, chassis 80), as the committed
/// HARDWARE.toml declares them.
fn options() -> PolicyOptions {
    PolicyOptions {
        ceilings: Some(TempCeilings {
            gpu_c: 75.0,
            chassis_c: 80.0,
        }),
        ..PolicyOptions::default()
    }
}

/// 2026-09-26: Cool, unthrottled, one model resident, SW power cap asserted.
fn healthy() -> HardwareState {
    HardwareState {
        gpu_temp_c: Some(55.0),
        chassis_temps_c: Some(vec![ThermalZone {
            name: "acpitz".into(),
            temp_c: 65.0,
        }]),
        throttle_active: ThrottleActive {
            sw_power_cap: Some(true),
            sw_thermal: Some(false),
            hw_thermal: Some(false),
            hw_power_brake: Some(false),
        },
        gpu_compute_apps: Some(vec![GpuComputeApp {
            pid: 2_392_883,
            name: "met".into(),
            used_mib: Some(109_281),
        }]),
        ..HardwareState::default()
    }
}

fn app(pid: u32) -> GpuComputeApp {
    GpuComputeApp {
        pid,
        name: "met".into(),
        used_mib: Some(87_014),
    }
}

fn delta(sw_us: Option<u64>, hw_us: Option<u64>) -> HardwareStateDelta {
    HardwareStateDelta {
        elapsed_s: Some(692),
        sw_thermal_us: sw_us,
        hw_thermal_us: hw_us,
        hw_power_brake_us: Some(0),
        ..HardwareStateDelta::default()
    }
}

#[test]
fn a_healthy_box_proceeds_with_nothing_to_say() {
    let p = precheck(Sensitivity::Speed, &healthy(), options());
    assert_eq!(p.decision, Decision::Proceed);
    assert!(p.concerns.is_empty(), "{:?}", p.concerns);
}

/// 2026-09-26: An asserted SW power cap alone does not refuse.
#[test]
fn an_active_power_cap_is_not_a_reason_to_refuse() {
    assert_eq!(healthy().throttle_active.sw_power_cap, Some(true));
    assert_eq!(
        precheck(Sensitivity::Speed, &healthy(), options()).decision,
        Decision::Proceed
    );
}

#[test]
fn a_speed_gate_refuses_a_box_that_is_thermally_throttling_right_now() {
    let mut s = healthy();
    s.throttle_active.hw_thermal = Some(true);
    let p = precheck(Sensitivity::Speed, &s, options());
    assert_eq!(p.decision, Decision::Refuse);
    assert!(p.concerns.iter().any(|c| c.contains("ACTIVE")), "{p:?}");
}

/// 2026-09-26: Two resident compute processes refuse a Speed run.
#[test]
fn a_speed_gate_refuses_a_box_with_a_second_gpu_process() {
    let mut s = healthy();
    s.gpu_compute_apps = Some(vec![app(2_392_883), app(2_118_440)]);
    let p = precheck(Sensitivity::Speed, &s, options());
    assert_eq!(p.decision, Decision::Refuse);
    assert!(
        p.concerns.iter().any(|c| c.contains("2 GPU compute")),
        "{p:?}"
    );
}

/// 2026-09-26: A Correctness run is recorded and proceeds (Warn), keeping every
/// concern the Speed run raised.
#[test]
fn a_correctness_gate_never_refuses_but_still_records_everything() {
    let mut s = healthy();
    s.throttle_active.hw_thermal = Some(true);
    s.gpu_compute_apps = Some(vec![app(1), app(2), app(3)]);
    let speed = precheck(Sensitivity::Speed, &s, options());
    let correctness = precheck(Sensitivity::Correctness, &s, options());
    assert_eq!(speed.decision, Decision::Refuse);
    assert_eq!(correctness.decision, Decision::Warn);
    for concern in &speed.concerns {
        assert!(
            correctness.concerns.contains(concern),
            "correctness dropped {concern:?}"
        );
    }
}

/// 2026-09-26: Unknown is not a pass: unreadable throttle reasons, an unlistable
/// compute-process list, or nothing readable at all, never yield Proceed.
#[test]
fn an_unreadable_field_never_yields_a_healthy_verdict() {
    let cases: [(&str, HardwareState); 3] = [
        ("nothing readable at all", HardwareState::default()),
        (
            "throttle reasons unreadable",
            HardwareState {
                throttle_active: ThrottleActive::default(),
                ..healthy()
            },
        ),
        (
            "compute apps unlistable",
            HardwareState {
                gpu_compute_apps: None,
                ..healthy()
            },
        ),
    ];
    for (label, state) in cases {
        for sensitivity in [Sensitivity::Speed, Sensitivity::Correctness] {
            let p = precheck(sensitivity, &state, options());
            assert_ne!(p.decision, Decision::Proceed, "{label} / {sensitivity:?}");
            assert!(!p.concerns.is_empty(), "{label} said nothing");
        }
    }
}

/// 2026-09-26: An unreadable box is warned about, not refused, so a benchmark stays
/// runnable on a machine without the reporting tools.
#[test]
fn an_unreadable_box_warns_rather_than_blocking_the_suite() {
    let p = precheck(Sensitivity::Speed, &HardwareState::default(), options());
    assert_eq!(p.decision, Decision::Warn);
}

/// 2026-09-26: Exceeded temperature ceilings warn by default and refuse only with
/// the opt-in, which changes the level and nothing else.
#[test]
fn absolute_temperature_is_recorded_by_default_and_gates_only_on_opt_in() {
    let mut s = healthy();
    s.gpu_temp_c = Some(89.0);
    s.chassis_temps_c = Some(vec![ThermalZone {
        name: "acpitz".into(),
        temp_c: 89.0,
    }]);

    let off = precheck(Sensitivity::Speed, &s, options());
    assert_eq!(off.decision, Decision::Warn);
    assert_eq!(off.concerns.len(), 2, "{:?}", off.concerns);

    let on = precheck(
        Sensitivity::Speed,
        &s,
        PolicyOptions {
            absolute_temp_gate: true,
            ..options()
        },
    );
    assert_eq!(on.decision, Decision::Refuse);
    assert_eq!(
        on.concerns, off.concerns,
        "the opt-in changes only the level"
    );
}

/// 2026-09-26: The GB10 ceilings are the declared ones; with no ceilings the
/// temperatures are recorded as not judged, never judged against another class's.
#[test]
fn the_ceilings_are_the_declared_ones_and_none_is_recorded_not_borrowed() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    let l = crate::hardware::limits::limits(root, "gb10")
        .unwrap()
        .unwrap();
    assert_eq!(
        TempCeilings::of(&l.thermal),
        TempCeilings {
            gpu_c: 75.0,
            chassis_c: 80.0
        }
    );
    let mut s = healthy();
    s.gpu_temp_c = Some(89.0);
    let none = precheck(
        Sensitivity::Speed,
        &s,
        PolicyOptions {
            ceilings: None,
            absolute_temp_gate: true,
            ..PolicyOptions::default()
        },
    );
    assert_eq!(none.decision, Decision::Warn, "{:?}", none.concerns);
    assert!(
        none.concerns
            .iter()
            .any(|c| c.contains("no temperature ceilings are declared")),
        "{:?}",
        none.concerns
    );
    assert!(
        !none.concerns.iter().any(|c| c.contains("above the")),
        "{:?}",
        none.concerns
    );
}

/// 2026-09-26: The kill switch suppresses the refusal and nothing else: it says it
/// was used and keeps every concern.
#[test]
fn the_kill_switch_downgrades_the_refusal_and_announces_itself() {
    let mut s = healthy();
    s.throttle_active.hw_thermal = Some(true);
    let blocked = precheck(Sensitivity::Speed, &s, options());
    let overridden = precheck(
        Sensitivity::Speed,
        &s,
        PolicyOptions {
            kill_switch: true,
            ..options()
        },
    );
    assert_eq!(blocked.decision, Decision::Refuse);
    assert_eq!(overridden.decision, Decision::Warn);
    for concern in &blocked.concerns {
        assert!(overridden.concerns.contains(concern));
    }
    assert!(
        overridden
            .concerns
            .iter()
            .any(|c| c.contains(KILL_SWITCH_ENV) && c.contains("SUPPRESSED")),
        "{overridden:?}"
    );
}

/// 2026-09-26: The kill switch cannot turn a warning into a pass.
#[test]
fn the_kill_switch_cannot_manufacture_a_proceed() {
    let opts = PolicyOptions {
        kill_switch: true,
        ..options()
    };
    let p = precheck(Sensitivity::Speed, &HardwareState::default(), opts);
    assert_eq!(p.decision, Decision::Warn);
}

#[test]
fn a_run_that_did_not_throttle_is_valid() {
    let p = postcheck(Sensitivity::Speed, &delta(Some(0), Some(0)), options());
    assert_eq!(p.validity, Validity::Valid);
}

/// 2026-09-26: A thermal counter that advanced during the run makes it invalid,
/// with the throttled share in the concern.
#[test]
fn a_run_during_which_a_thermal_counter_advanced_is_invalid() {
    let p = postcheck(
        Sensitivity::Speed,
        &delta(Some(12_000_000), Some(0)),
        options(),
    );
    assert_eq!(p.validity, Validity::Invalid);
    assert!(
        p.concerns.iter().any(|c| c.contains("not comparable")),
        "{p:?}"
    );
    assert!(p.concerns.iter().any(|c| c.contains("1.73%")), "{p:?}");
}

/// 2026-09-26: HW thermal time alone invalidates.
#[test]
fn hw_thermal_alone_invalidates() {
    let p = postcheck(Sensitivity::Speed, &delta(Some(0), Some(1)), options());
    assert_eq!(p.validity, Validity::Invalid);
}

/// 2026-09-26: Unreadable counters are Unknown, never Valid.
#[test]
fn unreadable_counters_leave_the_run_unknown_not_valid() {
    let p = postcheck(
        Sensitivity::Speed,
        &HardwareStateDelta::default(),
        options(),
    );
    assert_eq!(p.validity, Validity::Unknown);
    assert!(p.concerns.iter().any(|c| c.contains("not known")), "{p:?}");
}

#[test]
fn partially_unreadable_zero_counters_leave_the_run_unknown() {
    let p = postcheck(
        Sensitivity::Speed,
        &HardwareStateDelta {
            elapsed_s: Some(692),
            sw_thermal_us: Some(0),
            hw_thermal_us: None,
            hw_power_brake_us: None,
            ..HardwareStateDelta::default()
        },
        options(),
    );
    assert_eq!(p.validity, Validity::Unknown);
    assert_eq!(
        p.concerns,
        [
            "throttle counters were unreadable on at least one capture — this run is not known to have been unthrottled"
        ]
    );
}

/// 2026-09-26: A Correctness number is never invalidated by thermals, and the
/// concerns are still recorded beside it.
#[test]
fn a_correctness_run_is_never_invalidated_but_still_reports() {
    let d = delta(Some(12_000_000), Some(500_000));
    let p = postcheck(Sensitivity::Correctness, &d, options());
    assert_eq!(p.validity, Validity::NotApplicable);
    assert!(p.concerns.iter().any(|c| c.contains("throttled")), "{p:?}");
}

/// 2026-09-26: The kill switch lets a run start; it does not make a throttled run
/// valid.
#[test]
fn the_kill_switch_cannot_validate_a_throttled_run() {
    let p = postcheck(
        Sensitivity::Speed,
        &delta(Some(12_000_000), Some(0)),
        PolicyOptions {
            kill_switch: true,
            ..options()
        },
    );
    assert_eq!(p.validity, Validity::Invalid);
}

#[test]
fn verdicts_round_trip_through_json_for_the_record() {
    let p = precheck(Sensitivity::Speed, &healthy(), options());
    let back: Precheck = serde_json::from_str(&serde_json::to_string(&p).unwrap()).unwrap();
    assert_eq!(p, back);
    let q = postcheck(Sensitivity::Speed, &delta(Some(1), None), options());
    let back: Postcheck = serde_json::from_str(&serde_json::to_string(&q).unwrap()).unwrap();
    assert_eq!(q, back);
}

/// 2026-09-26: `PolicyOptions::default()` has both switches off and no ceilings.
#[test]
fn no_option_is_on_unless_it_was_asked_for() {
    assert_eq!(
        PolicyOptions::default(),
        PolicyOptions {
            kill_switch: false,
            absolute_temp_gate: false,
            ceilings: None,
        }
    );
}

/// 2026-09-26: Sensitivity lives on each registry descriptor. The answers for
/// these benchmarks are pinned, so changing one is a visible test change.
#[test]
fn the_registry_classifies_every_benchmark_the_incident_named() {
    let expected = [
        ("quick-speed-bench", Sensitivity::Speed),
        ("decode-floor", Sensitivity::Speed),
        ("concurrency-sweep", Sensitivity::Speed),
        ("ttft-warm-gate", Sensitivity::Speed),
        ("ttft-cold-gate", Sensitivity::Speed),
        // 2026-09-26: Mixed, classified by its `wall_budget_s` Σwall bound, the
        // half thermals can corrupt.
        ("agentic-webserver", Sensitivity::Speed),
        ("bfcl-subset", Sensitivity::Correctness),
        ("bfcl-subset-echolp", Sensitivity::Correctness),
        ("bfcl-full", Sensitivity::Correctness),
        ("ssm-state-poisoning-gate", Sensitivity::Correctness),
        ("vision-fidelity", Sensitivity::Correctness),
        ("video-fidelity", Sensitivity::Correctness),
        ("cross-contamination", Sensitivity::Correctness),
    ];
    for (id, sensitivity) in expected {
        let d = crate::registry::find(id).unwrap_or_else(|| panic!("{id} left the registry"));
        assert_eq!(d.sensitivity, sensitivity, "{id}");
    }
    // 2026-09-26: The field is not optional, so every registered benchmark has an
    // answer, pinned above or not.
    for d in crate::registry::all() {
        let _: Sensitivity = d.sensitivity;
    }
}

/// 2026-09-26: `hardware_state.postcheck.concerns` is persisted with the gate
/// record. With one counter unreadable the percentage is printed as a floor, and
/// the unread counter as `unreadable`, never as a measured `0 µs`.
#[test]
fn an_unreadable_counter_is_never_printed_as_a_measured_zero() {
    let p = postcheck(
        Sensitivity::Speed,
        &delta(None, Some(12_000_000)),
        options(),
    );
    assert_eq!(p.validity, Validity::Invalid);
    let joined = p.concerns.join("\n");
    assert!(
        joined.contains("sw unreadable"),
        "an unread counter must say so: {joined}"
    );
    assert!(
        !joined.contains("sw 0 µs"),
        "a never-read counter was recorded as a measured zero: {joined}"
    );
    assert!(
        joined.contains("AT LEAST 1.73%"),
        "a partial sum is a floor, not a figure: {joined}"
    );
    // 2026-09-26: The complete case keeps the plain wording, so the hedge carries
    // information.
    let full = postcheck(
        Sensitivity::Speed,
        &delta(Some(0), Some(12_000_000)),
        options(),
    );
    let full = full.concerns.join("\n");
    assert!(full.contains("throttled for 1.73% of the run"), "{full}");
    assert!(!full.contains("AT LEAST"), "{full}");
    assert!(
        full.contains("sw 0 µs"),
        "a real zero still reads as zero: {full}"
    );
}
