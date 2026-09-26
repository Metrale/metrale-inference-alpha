// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for a per-rung J/token ceiling, a `max` bound that
//! `scoring::check_record` applies to `c8_gpu_rail_joules_per_token`, the key
//! `hardware::energy::EnergyWindow::metrics` writes for the concurrency sweep.
//!
//! Owner: bench gate.
//! Invariants: none beyond the types.

use super::tests::{MODEL, SHA, baseline_for, hw, run_record};
use super::*;
use crate::result::Verdict;
use std::collections::BTreeMap;

const KEY: &str = "c8_gpu_rail_joules_per_token";
/// 2026-09-26: The `c8_gpu_rail_joules_per_token` `max` in
/// `kernels/gb10/qwen3.8-27b/BENCH.toml`.
const CEILING: f64 = 0.44;

fn baseline(with_ceiling: bool) -> GateBaseline {
    let mut metrics = BTreeMap::new();
    // 2026-09-26: A second bound keeps the entry non-empty without the ceiling;
    // `check_record` refuses an entry with no bounds.
    metrics.insert(
        "c8_aggregate_tok_s".to_string(),
        Bound {
            min: Some(110.0),
            ..Bound::default()
        },
    );
    if with_ceiling {
        metrics.insert(
            KEY.to_string(),
            Bound {
                max: Some(CEILING),
                ..Bound::default()
            },
        );
    }
    baseline_for(MODEL, metrics)
}

fn record(j_per_tok: Option<f64>) -> GateRecord {
    let mut m = BTreeMap::new();
    m.insert("c8_aggregate_tok_s".to_string(), 127.8);
    if let Some(v) = j_per_tok {
        m.insert(KEY.to_string(), v);
    }
    GateRecord::from_run(
        &run_record(m, Verdict::pass("ok")),
        hw(),
        SHA.into(),
        Vec::new(),
        None,
    )
    .unwrap()
}

#[test]
fn a_rung_above_its_joule_ceiling_fails_and_names_the_ceiling() {
    let problems = check_record(&record(Some(0.50)), &baseline(true)).expect("must fail");
    assert_eq!(
        problems,
        vec![format!("{KEY} 0.50 is above the ceiling 0.44 (noise 0.00)")]
    );
}

#[test]
fn a_rung_at_or_below_its_joule_ceiling_passes() {
    assert_eq!(check_record(&record(Some(0.3499)), &baseline(true)), None);
    assert_eq!(check_record(&record(Some(CEILING)), &baseline(true)), None);
}

/// 2026-09-26: Control: the over-ceiling record passes once the ceiling is removed,
/// so the failure above comes from the `max` bound.
#[test]
fn without_the_ceiling_the_same_over_budget_record_passes() {
    assert_eq!(check_record(&record(Some(0.50)), &baseline(false)), None);
    assert!(check_record(&record(Some(0.50)), &baseline(true)).is_some());
}

/// 2026-09-26: When `joules_per_token` has no ratio (zero tokens, or joules not finite
/// and positive) the key is not written, and the ceiling fails it as missing.
#[test]
fn an_absent_joule_reading_fails_as_missing_never_passes() {
    assert_eq!(
        check_record(&record(None), &baseline(true)),
        Some(vec![format!("{KEY}: missing from the record")])
    );
}
