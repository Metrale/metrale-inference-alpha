// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for the equality report: the table rows and the metric
//! keys.
//!
//! Owner: bench, kat_equality.
//! Invariants: none beyond the types.

use super::*;
use crate::benchmarks::kat_equality::compare::{Observation, score};
use crate::benchmarks::transcript::{RequestOutcome, Transcript};

fn ok(text: &str) -> RequestOutcome {
    RequestOutcome::Ok(Box::new(Transcript {
        text: text.to_string(),
        completion_tokens: text.len(),
        ..Default::default()
    }))
}

fn run(label: &str, samples: &[(&str, &str)]) -> OrderRun {
    OrderRun {
        label: label.to_string(),
        observations: samples
            .iter()
            .map(|(id, t)| Observation {
                sample_id: (*id).to_string(),
                outcome: ok(t),
            })
            .collect(),
    }
}

/// 2026-09-26: Only the samples that changed get a row.
#[test]
fn the_table_lists_only_the_samples_that_changed() {
    let a = run("canonical", &[("s1", "x"), ("s2", "y"), ("s3", "z")]);
    let b = run("reversed", &[("s3", "z"), ("s2", "DIFFERENT"), ("s1", "x")]);
    let s = score(&[a.clone(), b.clone()]);
    let t = table(&s, &[a, b]);
    assert_eq!(t.rows.len(), 1, "only the diverged sample earns a row");
    assert_eq!(t.rows[0][0].text, "s2");
    assert_eq!(t.rows[0][1].text, "DIVERGED");
}

#[test]
fn an_all_equal_run_still_says_so_rather_than_rendering_an_empty_table() {
    let a = run("canonical", &[("s1", "x"), ("s2", "y")]);
    let b = run("reversed", &[("s2", "y"), ("s1", "x")]);
    let s = score(&[a.clone(), b.clone()]);
    let t = table(&s, &[a, b]);
    assert_eq!(t.rows.len(), 1);
    assert_eq!(t.rows[0][1].text, "identical");
}

/// 2026-09-26: A missing key and a zero must stay distinguishable to whatever
/// compares records, so every class is emitted even when it is empty.
#[test]
fn every_metric_key_is_present_even_at_zero() {
    let a = run("canonical", &[("s1", "x")]);
    let b = run("reversed", &[("s1", "x")]);
    let m = metrics(&score(&[a, b]));
    for k in [
        "orders",
        "samples",
        "identical",
        "diverged",
        "unmeasured",
        "empty_replies",
    ] {
        assert!(m.contains_key(k), "metric `{k}` missing: {m:?}");
    }
    assert_eq!(m["diverged"], 0.0);
    assert_eq!(m["identical"], 1.0);
}

/// 2026-09-26: `empty_replies` is the number a BENCH.toml bound can refuse. If
/// it were absent or always zero the bound would be inert.
#[test]
fn empty_replies_is_reported_so_a_bound_can_refuse_a_vacuous_run() {
    let a = run("canonical", &[("s1", ""), ("s2", "real")]);
    let b = run("reversed", &[("s2", "real"), ("s1", "")]);
    let m = metrics(&score(&[a, b]));
    assert_eq!(m["empty_replies"], 1.0);
}
