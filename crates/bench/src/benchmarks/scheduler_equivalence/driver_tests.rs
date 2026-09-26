// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for the scheduler-equivalence plan, lanes and defaults.
//!
//! Owner: bench, scheduler-equivalence gate.
//! Invariants: none beyond the types.

use super::compare::Pass;
use super::driver::{DESCRIPTOR, SchedulerEquivalence, Step, lanes_for, plan};
use super::host::{Lane, Router, ServeVariant};
use crate::benchmark::Benchmark;
use crate::hardware::Sensitivity;
use crate::params::ParamValue;

#[test]
fn the_gate_is_registered_and_findable_by_its_id() {
    let d = crate::registry::find("scheduler-equivalence").expect("registered");
    assert_eq!(d.id, DESCRIPTOR.id);
    assert_eq!(d.name, "Scheduler Equivalence Gate");
}

#[test]
fn byte_equality_is_a_correctness_finding_not_a_speed_one() {
    assert_eq!(DESCRIPTOR.sensitivity, Sensitivity::Correctness);
}

#[test]
fn the_plan_serves_each_variant_once_and_controls_on_the_sync_serve() {
    let steps = plan(&[Lane::SpecOff, Lane::MtpForce], &[1, 4], true);
    let sync = |lane| {
        Step::Serve(ServeVariant {
            router: Router::Sync,
            lane,
        })
    };
    let asy = |lane| {
        Step::Serve(ServeVariant {
            router: Router::Async,
            lane,
        })
    };
    let gen_ = |lane, pass, concurrency| Step::Generate {
        lane,
        pass,
        concurrency,
    };
    use Lane::*;
    assert_eq!(
        steps,
        vec![
            sync(SpecOff),
            gen_(SpecOff, Pass::Sync, 1),
            gen_(SpecOff, Pass::Sync, 4),
            gen_(SpecOff, Pass::Control, 1),
            gen_(SpecOff, Pass::Control, 4),
            asy(SpecOff),
            Step::Diagnostics {
                lane: SpecOff,
                before: true
            },
            gen_(SpecOff, Pass::Async, 1),
            gen_(SpecOff, Pass::Async, 4),
            Step::Diagnostics {
                lane: SpecOff,
                before: false
            },
            sync(MtpForce),
            gen_(MtpForce, Pass::Sync, 1),
            gen_(MtpForce, Pass::Sync, 4),
            gen_(MtpForce, Pass::Control, 1),
            gen_(MtpForce, Pass::Control, 4),
            asy(MtpForce),
            Step::Diagnostics {
                lane: MtpForce,
                before: true
            },
            gen_(MtpForce, Pass::Async, 1),
            gen_(MtpForce, Pass::Async, 4),
            Step::Diagnostics {
                lane: MtpForce,
                before: false
            },
        ]
    );
    // 2026-09-26: Four loads for two lanes; the sequence above takes each
    // reference before its candidate is served.
    assert_eq!(
        steps.iter().filter(|s| matches!(s, Step::Serve(_))).count(),
        4
    );
}

#[test]
fn without_the_control_no_second_sync_pass_is_planned() {
    let steps = plan(&[Lane::SpecOff], &[16], false);
    assert!(!steps.iter().any(|s| matches!(
        s,
        Step::Generate {
            pass: Pass::Control,
            ..
        }
    )));
    assert_eq!(
        steps
            .iter()
            .filter(|s| matches!(s, Step::Generate { .. }))
            .count(),
        2
    );
}

#[test]
fn the_lane_choice_maps_to_the_pins_and_refuses_anything_else() {
    assert_eq!(
        lanes_for("both").unwrap(),
        vec![Lane::SpecOff, Lane::MtpForce]
    );
    assert_eq!(lanes_for("spec-off").unwrap(), vec![Lane::SpecOff]);
    assert_eq!(lanes_for("mtp-force").unwrap(), vec![Lane::MtpForce]);
    assert!(lanes_for("auto").is_err());
}

#[test]
fn the_defaults_are_the_published_instrument() {
    let specs = SchedulerEquivalence::default().parameters();
    let get = |k: &str| {
        specs
            .iter()
            .find(|s| s.key == k)
            .unwrap_or_else(|| panic!("{k} declared"))
            .default
            .clone()
    };
    assert_eq!(get("concurrencies"), ParamValue::IntList(vec![1, 4, 16]));
    assert_eq!(get("lanes"), ParamValue::Text("both".into()));
    assert_eq!(get("control"), ParamValue::Bool(true));
    // 2026-09-26: BFCL's budget, the constant the KAT equality gate also uses.
    assert_eq!(
        get("max_new_tokens"),
        ParamValue::Int(crate::benchmarks::bfcl::MAX_NEW_TOKENS as i64)
    );
}
