// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for the concurrency sweep's run verdict: the floors, the
//! rules the floors cannot override (request errors, vacuous cells, an
//! uncontrolled cache), and the committed Qwen3.8-27B floors. The essay
//! fixture's tests are the child module `essay`.
//!
//! Owner: bench (concurrency).
//! Invariants: none beyond the types.

use super::verdict::{Exclusions, Floors, sweep_verdict};
use super::*;
use crate::result::VerdictKind;

fn floors(c1: f64, c4: f64, c8: f64, c16: f64, peak: f64) -> Floors {
    Floors {
        per_c: vec![(1, c1), (4, c4), (8, c8), (16, c16)],
        peak,
    }
}

fn ladder(entries: &[(&str, f64)]) -> BTreeMap<String, f64> {
    entries.iter().map(|(k, v)| (k.to_string(), *v)).collect()
}

fn committed_floors() -> Floors {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(std::path::Path::parent)
        .expect("workspace layout");
    let (_, entry) = crate::gate::bench::load_all(root)
        .expect("committed BENCH.toml files must load")
        .into_iter()
        .find(|(target, entry)| {
            target.hardware == "gb10"
                && target.model == "qwen3.8-27b"
                && entry.checkpoint == "unsloth/Qwen3.8-27B-NVFP4"
                && entry.gate == "concurrency-sweep"
        })
        .expect("the measured Qwen3.8 concurrency gate must be committed");
    let metrics = entry
        .metrics
        .expect("the measured concurrency gate must have bounds");
    let min = |metric: &str| {
        metrics[metric]
            .min
            .unwrap_or_else(|| panic!("{metric} must have a minimum"))
    };
    // 2026-09-26: Only rungs with a committed bound; an unbounded rung is not
    // turned into a 0.0 floor here.
    let floor = |metric: &str| metrics.get(metric).and_then(|b| b.min);
    Floors {
        per_c: [1usize, 2, 4, 8, 16, 32, 64, 128]
            .into_iter()
            .filter_map(|c| floor(&format!("c{c}_aggregate_tok_s")).map(|v| (c, v)))
            .collect(),
        peak: min("peak_aggregate_tok_s"),
    }
}

#[test]
fn a_clean_sweep_that_clears_every_floor_passes() {
    // 2026-09-26: The rung set is read from the committed BENCH.toml and the
    // floor values are pinned, so a floor change edits this test. The fixture
    // ladder clears every committed floor.
    //
    // Every floor must also be above zero: `Floors::gating` is false when all
    // floors are 0.0, which gives an info verdict, and
    // `GateRecord::verdict_passes` accepts only PASS, so an all-zero ladder
    // would make the gate unsatisfiable.
    let m = ladder(&[
        ("c1_aggregate_tok_s", 23.59),
        ("c2_aggregate_tok_s", 41.02),
        ("c4_aggregate_tok_s", 74.21),
        ("c8_aggregate_tok_s", 125.95),
        ("c16_aggregate_tok_s", 203.36),
        ("c32_aggregate_tok_s", 291.01),
        ("c64_aggregate_tok_s", 386.63),
        ("c128_aggregate_tok_s", 478.11),
        ("peak_aggregate_tok_s", 478.11),
    ]);
    let floors = committed_floors();
    assert_eq!(
        floors.per_c,
        vec![
            (1, 22.0),
            (2, 38.0),
            (4, 67.0),
            (8, 110.0),
            (16, 180.0),
            (32, 260.0),
            (64, 360.0),
            (128, 440.0)
        ]
    );
    assert_eq!(floors.peak, 440.0);
    assert!(
        floors.gating(),
        "an all-zero ladder is an INFO verdict, not an ungated one, and INFO is \
         not a PASS — the gate would be unsatisfiable"
    );
    let v = sweep_verdict(
        &m,
        8,
        0,
        Exclusions {
            vacuous: 0,
            cache_uncontrolled: 0,
            non_mtp_arm: 0,
        },
        80.0,
        &floors,
    );
    assert_eq!(v.kind, VerdictKind::Pass, "{}", v.reason);
    for rung in ["C1", "C2", "C4", "C8", "C16", "C32", "C64", "C128", "peak"] {
        assert!(v.reason.contains(rung), "{}", v.reason);
    }
}

#[test]
fn a_sweep_below_one_floor_fails_naming_the_cell() {
    let committed = committed_floors();
    // 2026-09-26: C=8 is found by value, not position, so an inserted rung
    // cannot retarget the test.
    let c8_floor = committed
        .per_c
        .iter()
        .find(|(c, _)| *c == 8)
        .expect("C=8 is a committed rung")
        .1;
    // 2026-09-26: C=8 sits just under its committed floor; the other rungs
    // clear theirs, so the fail has one cause.
    let c8_under = c8_floor - 0.1;
    let m = ladder(&[
        ("c1_aggregate_tok_s", 23.59),
        ("c2_aggregate_tok_s", 41.02),
        ("c4_aggregate_tok_s", 74.21),
        ("c8_aggregate_tok_s", c8_under),
        ("c16_aggregate_tok_s", 203.36),
        ("c32_aggregate_tok_s", 291.01),
        ("c64_aggregate_tok_s", 386.63),
        ("c128_aggregate_tok_s", 478.11),
        ("peak_aggregate_tok_s", 478.11),
    ]);
    let v = sweep_verdict(
        &m,
        8,
        0,
        Exclusions {
            vacuous: 0,
            cache_uncontrolled: 0,
            non_mtp_arm: 0,
        },
        80.0,
        &committed,
    );
    assert_eq!(v.kind, VerdictKind::Fail, "{}", v.reason);
    assert!(v.reason.contains("C=8"), "{}", v.reason);
    assert!(
        v.reason.contains(&format!("{c8_under:.1}"))
            && v.reason.contains(&format!("{c8_floor:.1}")),
        "{}",
        v.reason
    );
    for other in ["C=1", "C=2", "C=4", "C=16", "C=32", "C=64", "C=128"] {
        assert!(
            !v.reason.contains(&format!("{other} ")),
            "{other} should clear its floor: {}",
            v.reason
        );
    }
    let m = ladder(&[("c8_aggregate_tok_s", c8_floor)]);
    let v = sweep_verdict(
        &m,
        1,
        0,
        Exclusions {
            vacuous: 0,
            cache_uncontrolled: 0,
            non_mtp_arm: 0,
        },
        80.0,
        &floors(0.0, 0.0, c8_floor, 0.0, 0.0),
    );
    assert_eq!(v.kind, VerdictKind::Pass, "{}", v.reason);
}

#[test]
fn all_floors_zero_keeps_the_info_verdicts() {
    let m = ladder(&[("c1_aggregate_tok_s", 25.5)]);
    let clean = sweep_verdict(
        &m,
        4,
        0,
        Exclusions {
            vacuous: 0,
            cache_uncontrolled: 0,
            non_mtp_arm: 0,
        },
        80.0,
        &Floors::default(),
    );
    assert_eq!(clean.kind, VerdictKind::Info, "{}", clean.reason);
    assert!(
        clean.reason.contains("no request errors"),
        "{}",
        clean.reason
    );
    let vac = sweep_verdict(
        &m,
        4,
        0,
        Exclusions {
            vacuous: 2,
            cache_uncontrolled: 0,
            non_mtp_arm: 0,
        },
        80.0,
        &Floors::default(),
    );
    assert_eq!(vac.kind, VerdictKind::Info, "{}", vac.reason);
    assert!(vac.reason.contains("not comparable"), "{}", vac.reason);
}

#[test]
fn vacuous_cells_fail_a_gating_sweep_regardless_of_the_floors() {
    let m = ladder(&[
        ("c1_aggregate_tok_s", 999.0),
        ("c4_aggregate_tok_s", 999.0),
        ("c8_aggregate_tok_s", 999.0),
        ("c16_aggregate_tok_s", 999.0),
        ("peak_aggregate_tok_s", 999.0),
    ]);
    let v = sweep_verdict(
        &m,
        4,
        0,
        Exclusions {
            vacuous: 1,
            cache_uncontrolled: 0,
            non_mtp_arm: 0,
        },
        80.0,
        &floors(24.0, 43.0, 63.0, 94.0, 94.0),
    );
    assert_eq!(v.kind, VerdictKind::Fail, "{}", v.reason);
    assert!(v.reason.contains("INCONCLUSIVE"), "{}", v.reason);
    assert!(v.reason.contains("vacuity floor"), "{}", v.reason);
}

#[test]
fn a_gated_rung_with_no_comparable_cell_fails_as_inconclusive() {
    // 2026-09-26: C=16 is gated but has no comparable cell.
    let m = ladder(&[("c1_aggregate_tok_s", 25.5)]);
    let v = sweep_verdict(
        &m,
        4,
        0,
        Exclusions {
            vacuous: 0,
            cache_uncontrolled: 0,
            non_mtp_arm: 0,
        },
        80.0,
        &floors(0.0, 0.0, 0.0, 94.0, 0.0),
    );
    assert_eq!(v.kind, VerdictKind::Fail, "{}", v.reason);
    assert!(v.reason.contains("C=16"), "{}", v.reason);
    assert!(v.reason.contains("INCONCLUSIVE"), "{}", v.reason);
    let v = sweep_verdict(
        &m,
        4,
        0,
        Exclusions {
            vacuous: 0,
            cache_uncontrolled: 0,
            non_mtp_arm: 0,
        },
        80.0,
        &floors(0.0, 0.0, 0.0, 0.0, 94.0),
    );
    assert_eq!(v.kind, VerdictKind::Fail, "{}", v.reason);
    assert!(v.reason.contains("peak"), "{}", v.reason);
}

#[test]
fn request_errors_fail_the_sweep_in_both_modes() {
    let m = ladder(&[("c1_aggregate_tok_s", 999.0)]);
    for f in [Floors::default(), floors(24.0, 43.0, 63.0, 94.0, 94.0)] {
        let v = sweep_verdict(
            &m,
            4,
            2,
            Exclusions {
                vacuous: 0,
                cache_uncontrolled: 0,
                non_mtp_arm: 0,
            },
            80.0,
            &f,
        );
        assert_eq!(v.kind, VerdictKind::Fail, "{}", v.reason);
        assert!(v.reason.contains("2 request(s) failed"), "{}", v.reason);
    }
}

#[test]
fn an_unobserved_warm_cache_cannot_clear_the_gate() {
    let m = ladder(&[
        ("c1_aggregate_tok_s", 999.0),
        ("c4_aggregate_tok_s", 999.0),
        ("c8_aggregate_tok_s", 999.0),
        ("c16_aggregate_tok_s", 999.0),
        ("peak_aggregate_tok_s", 999.0),
    ]);
    let v = sweep_verdict(
        &m,
        4,
        0,
        Exclusions {
            vacuous: 0,
            cache_uncontrolled: 1,
            non_mtp_arm: 0,
        },
        80.0,
        &floors(24.0, 43.0, 63.0, 94.0, 94.0),
    );

    assert_eq!(v.kind, VerdictKind::Fail, "{}", v.reason);
    assert!(v.reason.contains("cached-prompt fraction"), "{}", v.reason);
}

/// 2026-09-26: Every `RUNGS` row must spell its own C. A row like
/// `(32, "min_c32", "c64_aggregate_tok_s", ..)` would fill the C=32 floor from
/// the C=64 bound, and both names are well formed, so nothing downstream would
/// notice.
#[test]
fn every_rung_names_itself_consistently() {
    for (c, param, metric, label) in RUNGS {
        assert_eq!(param, format!("min_c{c}"), "rung {c}: floor param");
        assert_eq!(
            metric,
            format!("c{c}_aggregate_tok_s"),
            "rung {c}: metric key"
        );
        assert!(
            label.contains(&format!("C={c} ")),
            "rung {c}: label {label:?}"
        );
    }
    let mut seen: Vec<usize> = RUNGS.iter().map(|(c, ..)| *c).collect();
    let sorted = {
        let mut v = seen.clone();
        v.sort_unstable();
        v
    };
    assert_eq!(seen, sorted, "rungs must be in ascending order");
    seen.dedup();
    assert_eq!(seen.len(), RUNGS.len(), "a rung is declared twice");
    assert_eq!(PEAK_FLOOR.1, "peak_aggregate_tok_s");
    assert!(
        RUNGS
            .iter()
            .all(|(_, p, m, _)| *p != PEAK_FLOOR.0 && *m != PEAK_FLOOR.1)
    );
}

#[test]
fn the_floor_params_are_wired_to_the_gate() {
    // 2026-09-26: Asserts the derivation from `RUNGS`, with the peak last.
    assert_eq!(
        DESCRIPTOR.threshold_params,
        [
            ("min_c1", "c1_aggregate_tok_s"),
            ("min_c2", "c2_aggregate_tok_s"),
            ("min_c4", "c4_aggregate_tok_s"),
            ("min_c8", "c8_aggregate_tok_s"),
            ("min_c16", "c16_aggregate_tok_s"),
            ("min_c32", "c32_aggregate_tok_s"),
            ("min_c64", "c64_aggregate_tok_s"),
            ("min_c128", "c128_aggregate_tok_s"),
            ("min_peak", "peak_aggregate_tok_s"),
        ]
    );
    assert_eq!(
        DFLASH2_DESCRIPTOR.threshold_params, DESCRIPTOR.threshold_params,
        "the two concurrency gates must gate on the same metric names"
    );
    let mut b = ConcurrencySweep::default();
    let specs = b.parameters();
    for (param, _) in DESCRIPTOR.threshold_params {
        assert!(
            specs.iter().any(|s| s.key == *param),
            "{param} declared but missing from the schema"
        );
    }
    let mut v = ParamValues::defaults(&specs);
    b.configure(&v).unwrap();
    assert!(!b.floors.gating(), "defaults must not gate");
    v.set("min_c8", ParamValue::Float(63.0));
    v.set("min_peak", ParamValue::Float(94.0));
    b.configure(&v).unwrap();
    assert!(b.floors.gating());
    assert_eq!(
        b.floors.per_c,
        vec![
            (1, 0.0),
            (2, 0.0),
            (4, 0.0),
            (8, 63.0),
            (16, 0.0),
            (32, 0.0),
            (64, 0.0),
            (128, 0.0)
        ],
        "an unbounded rung must carry the 0.0 OFF value, not be absent — \
         `sweep_verdict` fails a GATED rung that produced no comparable cell"
    );
    assert_eq!(b.floors.peak, 94.0);
}

#[path = "concurrency_essay_tests.rs"]
mod essay;
