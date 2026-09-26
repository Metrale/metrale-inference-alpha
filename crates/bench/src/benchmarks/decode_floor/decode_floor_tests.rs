// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for the decode-floor pins, verdict and instrument keys.
//! `evaluate` talks to no endpoint, and each vacuity pin has a test that fails
//! if the pin is removed.
//!
//! Owner: bench, decode_floor.
//! Invariants: none beyond the types.

use super::score::*;
use super::*;

/// 2026-09-26: A run that passes every vacuity pin: 1450 of 1500 tokens, a
/// server rate, and an accept length of 1450 / 750.
fn healthy(tps: f64) -> RunObs {
    RunObs {
        completion_tokens: 1450,
        server_tps: Some(tps),
        accepted_prediction_tokens: Some(700),
        e2e_ms: 50_000.0,
        ..Default::default()
    }
}

#[test]
fn run_observation_preserves_wire_evidence() {
    let outcome = crate::http::ChatOutcome {
        completion_tokens: 941,
        server_tps: Some(28.125),
        accepted_prediction_tokens: Some(417),
        e2e_ms: 33_456.75,
        ..Default::default()
    };

    assert_eq!(
        RunObs::from_outcome(&outcome),
        RunObs {
            completion_tokens: 941,
            server_tps: Some(28.125),
            accepted_prediction_tokens: Some(417),
            e2e_ms: 33_456.75,
            ..Default::default()
        }
    );
}

// 2026-09-26: Path A: the success path.

#[test]
fn three_healthy_runs_measure_the_median() {
    let samples = [healthy(31.5), healthy(29.6), healthy(30.5)];
    match evaluate(&samples) {
        Evaluation::Measured {
            median_decode_tok_s,
            min_output_tokens,
            accept_len_mean,
        } => {
            // 2026-09-26: 30.5 is the middle run. stats::percentile(_, 50)
            // would return 31.5 (the nearest-rank p50 of n=3 is the max).
            assert_eq!(median_decode_tok_s, 30.5);
            assert_eq!(min_output_tokens, 1450);
            assert!((accept_len_mean - 1450.0 / 750.0).abs() < 1e-9);
        }
        other => panic!("expected Measured, got {other:?}"),
    }
}

#[test]
fn min_output_tokens_is_the_worst_run_not_the_mean() {
    let mut samples = [healthy(30.0), healthy(30.0), healthy(30.0)];
    samples[1].completion_tokens = MIN_OUTPUT_TOKENS;
    match evaluate(&samples) {
        Evaluation::Measured {
            min_output_tokens, ..
        } => assert_eq!(min_output_tokens, MIN_OUTPUT_TOKENS),
        other => panic!("expected Measured, got {other:?}"),
    }
}

// 2026-09-26: Path B: the boundaries.

#[test]
fn one_below_the_output_floor_is_inconclusive() {
    let mut samples = [healthy(30.0), healthy(30.0), healthy(30.0)];
    samples[2].completion_tokens = MIN_OUTPUT_TOKENS - 1;
    match evaluate(&samples) {
        Evaluation::Inconclusive(why) => {
            assert!(why.contains("run 3"), "{why}");
            assert!(why.contains(&(MIN_OUTPUT_TOKENS - 1).to_string()), "{why}");
        }
        other => panic!("a one-below-the-floor run must be inconclusive, got {other:?}"),
    }
}

/// 2026-09-26: A run that stops naturally under the budget but above
/// `MIN_OUTPUT_TOKENS` (here 915 of 1500) measures; it is not inconclusive.
#[test]
fn the_calibration_instruments_915_token_stop_is_a_measurement() {
    let run = RunObs {
        completion_tokens: 915,
        server_tps: Some(28.0),
        accepted_prediction_tokens: Some(569),
        e2e_ms: 33_000.0,
        ..Default::default()
    };
    let samples = [run.clone(), run.clone(), run];
    match evaluate(&samples) {
        Evaluation::Measured {
            median_decode_tok_s,
            min_output_tokens,
            ..
        } => {
            assert_eq!(median_decode_tok_s, 28.0);
            assert_eq!(min_output_tokens, 915);
        }
        other => panic!("the calibration fingerprint must measure, got {other:?}"),
    }
}

#[test]
fn accept_len_floor_is_inclusive() {
    // 2026-09-26: completion 1500, accepted 500: 1500 / 1000 is exactly 1.5,
    // and `>=` passes it.
    let run = RunObs {
        completion_tokens: 1500,
        server_tps: Some(25.0),
        accepted_prediction_tokens: Some(500),
        e2e_ms: 60_000.0,
        ..Default::default()
    };
    let samples = [run.clone(), run.clone(), run];
    match evaluate(&samples) {
        Evaluation::Measured {
            accept_len_mean, ..
        } => assert!((accept_len_mean - 1.5).abs() < 1e-9),
        other => panic!("accept_len exactly 1.5 must measure, got {other:?}"),
    }
}

#[test]
fn a_disengaged_speculation_mean_is_inconclusive_not_a_floor() {
    // 2026-09-26: accepted 100 of 1400: 1400 / 1300 is about 1.077, under
    // `MIN_ACCEPT_LEN`. Speculation is nominally on but not at gate depth, and
    // the rate must not be recorded as the decode floor.
    let run = RunObs {
        completion_tokens: 1400,
        server_tps: Some(15.0),
        accepted_prediction_tokens: Some(100),
        e2e_ms: 90_000.0,
        ..Default::default()
    };
    let samples = [run.clone(), run.clone(), run];
    match evaluate(&samples) {
        Evaluation::Inconclusive(why) => {
            assert!(why.contains("not"), "{why}");
            assert!(why.contains("serial floor"), "{why}");
        }
        other => panic!("expected Inconclusive, got {other:?}"),
    }
}

#[test]
fn corrupt_accounting_is_inconclusive() {
    let mut samples = [healthy(30.0), healthy(30.0), healthy(30.0)];
    samples[0].accepted_prediction_tokens = Some(1450);
    match evaluate(&samples) {
        Evaluation::Inconclusive(why) => assert!(why.contains("corrupt"), "{why}"),
        other => panic!("expected Inconclusive, got {other:?}"),
    }
}

#[test]
fn fewer_than_the_pinned_runs_cannot_measure() {
    let samples = [healthy(30.0), healthy(30.0)];
    match evaluate(&samples) {
        Evaluation::Inconclusive(why) => assert!(why.contains("pinned count is 3"), "{why}"),
        other => panic!("expected Inconclusive, got {other:?}"),
    }
}

// 2026-09-26: Path C: the accept field and the server rate are missing or
// invalid.

#[test]
fn an_absent_accept_field_names_the_instrumentation_dependency() {
    let mut samples = [healthy(30.0), healthy(30.0), healthy(30.0)];
    samples[1].accepted_prediction_tokens = None;
    match evaluate(&samples) {
        Evaluation::Inconclusive(why) => {
            assert!(why.contains("accepted_prediction_tokens"), "{why}");
            assert!(why.contains("accept-stats instrumentation"), "{why}");
        }
        other => panic!("expected Inconclusive, got {other:?}"),
    }
}

#[test]
fn a_zero_accept_count_is_inconclusive_never_pass() {
    let mut samples = [healthy(30.0), healthy(30.0), healthy(30.0)];
    samples[0].accepted_prediction_tokens = Some(0);
    match evaluate(&samples) {
        Evaluation::Inconclusive(why) => {
            assert!(why.contains("accepted 0 draft tokens"), "{why}");
        }
        other => panic!("expected Inconclusive, got {other:?}"),
    }
}

#[test]
fn a_missing_server_rate_is_inconclusive() {
    let mut samples = [healthy(30.0), healthy(30.0), healthy(30.0)];
    samples[2].server_tps = None;
    match evaluate(&samples) {
        Evaluation::Inconclusive(why) => {
            assert!(why.contains("response_token/s"), "{why}");
        }
        other => panic!("expected Inconclusive, got {other:?}"),
    }
}

#[test]
fn a_nonpositive_or_nonfinite_server_rate_is_inconclusive() {
    for invalid in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let mut samples = [healthy(30.0), healthy(30.0), healthy(30.0)];
        samples[2].server_tps = Some(invalid);
        match evaluate(&samples) {
            Evaluation::Inconclusive(why) => {
                assert!(why.contains("run 3"), "{why}");
                assert!(why.contains("positive"), "{why}");
            }
            other => panic!("server rate {invalid} must be inconclusive, got {other:?}"),
        }
    }
}

// 2026-09-26: The run verdict against the floor parameter.

fn measured(median: f64) -> Evaluation {
    Evaluation::Measured {
        median_decode_tok_s: median,
        min_output_tokens: 1450,
        accept_len_mean: 1.93,
    }
}

/// 2026-09-26: With a floor set, a Measured run passes at or above it and
/// fails below it (the raw `median >= min_tok_s`, see `verdict_for`).
#[test]
fn a_measured_run_self_verdicts_against_the_floor_param() {
    use crate::result::VerdictKind;
    let v = verdict_for(&measured(30.5), 29.0);
    assert_eq!(v.kind, VerdictKind::Pass, "{}", v.reason);
    assert!(
        v.reason.contains("30.5") && v.reason.contains("29.0"),
        "{}",
        v.reason
    );

    let v = verdict_for(&measured(28.5), 29.0);
    assert_eq!(v.kind, VerdictKind::Fail, "{}", v.reason);
    assert!(v.reason.contains("BELOW THE DECODE FLOOR"), "{}", v.reason);
    assert!(
        v.reason.contains("28.5") && v.reason.contains("29.0"),
        "{}",
        v.reason
    );

    // 2026-09-26: Exactly on the floor passes: inclusive, like
    // `gate::scoring`'s `value + noise >= min`.
    assert_eq!(verdict_for(&measured(29.0), 29.0).kind, VerdictKind::Pass);
}

/// 2026-09-26: Floor 0, the schema default, gives an info verdict: a
/// standalone run has no committed floor to be judged against.
#[test]
fn no_floor_param_keeps_the_info_verdict() {
    use crate::result::VerdictKind;
    let v = verdict_for(&measured(30.5), 0.0);
    assert_eq!(v.kind, VerdictKind::Info, "{}", v.reason);
    assert!(v.reason.contains("--pull-request-gate"), "{}", v.reason);
}

/// 2026-09-26: An inconclusive evaluation is a failing verdict whatever the
/// floor parameter says.
#[test]
fn vacuous_runs_stay_inconclusive_regardless_of_the_floor_param() {
    use crate::result::VerdictKind;
    let eval = Evaluation::Inconclusive("accept_len_mean 1.10 < 1.5".to_string());
    for floor in [0.0, 29.0] {
        let v = verdict_for(&eval, floor);
        assert_eq!(v.kind, VerdictKind::Fail, "{}", v.reason);
        assert!(v.reason.contains("INCONCLUSIVE"), "{}", v.reason);
    }
}

/// 2026-09-26: The descriptor couples the floor parameter to the metric the
/// BENCH.toml bound is written on, and the schema default is the off state.
#[test]
fn the_floor_param_is_wired_to_the_gate() {
    assert_eq!(
        DESCRIPTOR.threshold_params,
        [("min_tok_s", "server_decode_tok_s")]
    );
    let b = DecodeFloor::default();
    let v = ParamValues::defaults(&b.parameters());
    assert_eq!(v.float("min_tok_s").unwrap(), 0.0);
}

// 2026-09-26: The pinned request, the accept-length rule and the parameters.

#[test]
fn the_pins_are_the_documented_fingerprint() {
    assert_eq!(RUNS, 3);
    assert_eq!(MAX_TOKENS, 1500);
    assert_eq!(MIN_OUTPUT_TOKENS, 750);
    assert_eq!(MIN_ACCEPT_LEN, 1.5);
    assert_eq!(
        DecodeFloor::request_body("fixture-model"),
        serde_json::json!({
            "model": "fixture-model",
            "stream": true,
            "temperature": 0.0,
            "seed": 0,
            "max_tokens": 1500,
            "reasoning_effort": "none",
            "messages": [{
                "role": "user",
                "content": "Implement a complete, production-quality MinHeap class in Python. Include the methods insert, extract_min, peek, heapify (bottom-up from an arbitrary list), decrease_key, delete_at_index, merge (with another MinHeap), __len__ and __iter__. Every method needs a full docstring with time-complexity analysis. Then write a comprehensive pytest test suite covering the empty heap, a single element, duplicate keys, and long interleaved insert/extract sequences. Finish with a line-by-line explanation of the sift_up and sift_down invariants. Be exhaustive and do not stop early."
            }],
        })
    );
}

#[test]
fn accept_len_derivation_matches_its_definition() {
    let r = RunObs {
        completion_tokens: 1200,
        server_tps: Some(30.0),
        accepted_prediction_tokens: Some(600),
        e2e_ms: 0.0,
        ..Default::default()
    };
    // 2026-09-26: 1200 tokens over 1200 - 600 = 600 decode steps is 2.0
    // tokens per step.
    assert_eq!(r.accept_len(), Some(2.0));
    let none = RunObs {
        accepted_prediction_tokens: None,
        ..r.clone()
    };
    assert_eq!(none.accept_len(), None);
}

#[test]
fn the_descriptor_is_registered_and_defaults_configure() {
    assert_eq!(
        crate::registry::find("decode-floor")
            .expect("registered")
            .name,
        "Decode Floor Gate"
    );
    let mut b = DecodeFloor::default();
    let v = ParamValues::defaults(&b.parameters());
    b.configure(&v).expect("defaults configure");
    assert_eq!(b.timeout, Duration::from_secs(300));
}

#[test]
fn reconfiguring_clears_collected_samples() {
    let mut b = DecodeFloor::default();
    let v = ParamValues::defaults(&b.parameters());
    b.configure(&v).unwrap();
    b.samples.push(healthy(30.0));
    b.probed = true;
    b.configure(&v).unwrap();
    assert!(b.samples.is_empty());
    assert!(!b.probed);
}

/// 2026-09-26: The client-clock ITL, the pooled jitter distribution and the
/// joules of the sampled windows ride beside the verdict metrics; the
/// server-clock ITL is `server_decode_tok_s` already and gets no second key.
#[test]
fn instrument_metrics_keep_both_clocks_and_store_joules_beside_tokens() {
    use crate::hardware::energy::EnergyWindow;
    use crate::http::GapSample;
    use std::collections::BTreeMap;

    let mut runs = [healthy(31.5), healthy(29.6), healthy(30.5)];
    for (i, r) in runs.iter_mut().enumerate() {
        r.client_tpot_ms = Some(34.0 + i as f64);
        r.server_tpot_ms = Some(33.0 + i as f64);
        let mut g = GapSample::default();
        for gap in [33.0, 33.0, 33.0, 34.0, 33.0] {
            g.push(gap);
        }
        r.arrival_gaps = g;
        r.energy = Some(EnergyWindow {
            window_s: 50.0,
            samples: 200,
            energy_j: 3000.0,
            mean_power_w: 60.0,
            max_power_w: 62.0,
            sw_power_cap_frac: Some(1.0),
            hw_power_brake_frac: Some(0.0),
        });
    }
    let idle = EnergyWindow {
        window_s: 2.0,
        samples: 8,
        energy_j: 10.0,
        mean_power_w: 5.0,
        ..Default::default()
    };
    let mut m = BTreeMap::new();
    instrument_metrics(&runs, Some(&idle), &mut m);
    assert_eq!(
        m.get("client_tpot_ms"),
        Some(&35.0),
        "median of the three runs"
    );
    assert!(
        !m.contains_key("server_tpot_ms"),
        "server ITL is server_decode_tok_s, no dup"
    );
    assert!(!m.keys().any(|k| k.contains("itl")), "{m:?}");
    assert_eq!(
        m.get("arrival_gap_count"),
        Some(&15.0),
        "pooled over the runs"
    );
    assert_eq!(m.get("arrival_gap_max_ms"), Some(&34.0));
    assert!(m.contains_key("stability"));
    assert_eq!(m.get("gpu_rail_energy_j"), Some(&9000.0), "joules add");
    assert_eq!(m.get("gpu_rail_energy_window_s"), Some(&150.0));
    assert_eq!(m.get("gpu_rail_power_samples"), Some(&600.0));
    assert_eq!(
        m.get("gpu_rail_energy_window_tokens"),
        Some(&(1450.0 * 3.0))
    );
    assert!((m["gpu_rail_energy_above_idle_j"] - (9000.0 - 5.0 * 150.0)).abs() < 1e-9);
    // 2026-09-26: J/token is the one stored ratio, and it is the summed pair's
    // quotient (joules add, tokens add, divide once), not a mean of per-run
    // ratios.
    assert!((m["gpu_rail_joules_per_token"] - 9000.0 / (1450.0 * 3.0)).abs() < 1e-12);
    assert_eq!(
        m.keys().filter(|k| k.contains("per_token")).count(),
        1,
        "no other ratio is stored: {m:?}"
    );

    // 2026-09-26: Nothing instrumented: nothing emitted, not zeros.
    let mut bare = BTreeMap::new();
    instrument_metrics(
        &[healthy(30.0), healthy(30.0), healthy(30.0)],
        None,
        &mut bare,
    );
    assert!(bare.is_empty(), "{bare:?}");
}
