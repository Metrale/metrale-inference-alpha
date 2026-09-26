// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Scoring one record against one baseline, the pure half of the
//! gate check: `compare` judges one metric against one bound, and
//! `check_record` resolves the record's own (hardware, checkpoint) pair and
//! scores that entry's pins and bounds.
//!
//! Owner: bench gate (verdict).
//! Invariants:
//! - Nothing here reads the filesystem, git or the environment.
//! - `check_record` returns `None` only when the entry declares at least one
//!   bound and every pin and bound is met.

use super::record::{GateBaseline, GateRecord};

/// 2026-09-26: One metric's comparison, or why it cannot be judged.
pub enum Comparison {
    Pass,
    Fail(String),
    Skip(String),
}

/// 2026-09-26: Compare one recorded metric against its bound, widened by
/// `noise` (0 when absent) on the failing side. No bound at all is `Skip`.
pub fn compare(name: &str, value: f64, bound: &super::record::Bound) -> Comparison {
    let noise = bound.noise.unwrap_or(0.0);
    match (bound.min, bound.max) {
        (Some(min), None) if value + noise >= min => Comparison::Pass,
        (Some(min), None) => Comparison::Fail(format!(
            "{name} {value:.2} is below the floor {min:.2} (noise {noise:.2})"
        )),
        (None, Some(max)) if value - noise <= max => Comparison::Pass,
        (None, Some(max)) => Comparison::Fail(format!(
            "{name} {value:.2} is above the ceiling {max:.2} (noise {noise:.2})"
        )),
        // 2026-09-26: Both bounds are a range, and equal bounds an exact pin
        // with its own failure message.
        (Some(min), Some(max)) if value + noise >= min && value - noise <= max => Comparison::Pass,
        (Some(min), Some(max)) if (min - max).abs() < f64::EPSILON => Comparison::Fail(format!(
            "{name} is {value:.0}, but this gate is pinned to exactly {min:.0} — \
             the run measured something other than what the baseline describes"
        )),
        (Some(min), Some(max)) => Comparison::Fail(format!(
            "{name} {value:.2} is outside [{min:.2}, {max:.2}] (noise {noise:.2})"
        )),
        (None, None) => Comparison::Skip(format!("{name} has no bound")),
    }
}

/// 2026-09-26: Check one record against its baseline. `None` means every pin
/// and bound passed; `Some` carries the failures. A record whose (hardware,
/// model) pair has no baseline entry fails.
pub fn check_record(record: &GateRecord, baseline: &GateBaseline) -> Option<Vec<String>> {
    // 2026-09-26: Scored only against the record's own (hardware, model)
    // pair; a bound measured on another box or checkpoint does not apply.
    let hardware = record.hardware.gate_key();
    let entry = match baseline.resolve(&hardware, Some(&record.target_model)) {
        Ok((_, entry)) => entry,
        Err(e) => return Some(vec![format!("{e:#}")]),
    };
    // 2026-09-26: An entry with no bounds fails: the metric loop below would
    // otherwise pass any record.
    if entry.metrics.is_empty() {
        return Some(vec![format!(
            "the baseline entry for {} on {hardware} declares no thresholds — \
             there is nothing here for this run to have passed",
            record.target_model
        )]);
    }
    let mut problems = Vec::new();
    // 2026-09-26: Every baseline serve pin must be on the record at the
    // pinned value, and the record may carry no unpinned override. BENCH.toml
    // edits invalidate no record (`coverage::NON_COMPILED_KERNEL_FILES`), so
    // a new pin must not let an older record pass.
    for (k, want) in &entry.serve_overrides {
        match record.serve_overrides.get(k) {
            Some(got) if got == want => {}
            Some(got) => problems.push(format!(
                "serve override {k}={got} does not match the baseline pin {k}={want}"
            )),
            None => problems.push(format!(
                "serve override {k}={want} is pinned on the baseline but missing from the record"
            )),
        }
    }
    for (k, got) in &record.serve_overrides {
        if !entry.serve_overrides.contains_key(k) {
            problems.push(format!(
                "serve override {k}={got} is present on the record but not pinned by the baseline"
            ));
        }
    }
    // 2026-09-26: Every baseline param pin must be on the record at the pinned
    // value, for the same reason. Each comma-separated item is trimmed before
    // comparing, so `1, 4, 8, 16` matches the pin `1,4,8,16`.
    let normalize = |s: &str| s.split(',').map(str::trim).collect::<Vec<_>>().join(",");
    for (k, want) in &entry.param_overrides {
        match record.params.get(k) {
            Some(got) if normalize(got) == normalize(want) => {}
            Some(got) => problems.push(format!(
                "param {k}={got} does not match the baseline pin {k}={want} — the run \
                 measured a different instrument than the one these thresholds describe"
            )),
            None => problems.push(format!(
                "param {k}={want} is pinned on the baseline but missing from the record"
            )),
        }
    }
    for (name, bound) in &entry.metrics {
        let Some(value) = record.metrics.get(name) else {
            problems.push(format!("{name}: missing from the record"));
            continue;
        };
        match compare(name, *value, bound) {
            Comparison::Pass => {}
            Comparison::Fail(reason) => problems.push(reason),
            Comparison::Skip(reason) => problems.push(reason),
        }
    }
    if problems.is_empty() {
        None
    } else {
        Some(problems)
    }
}
