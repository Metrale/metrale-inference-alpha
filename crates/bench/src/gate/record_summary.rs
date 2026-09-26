// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The one-line summary a gate record carries, and the clock it
//! is stamped with.
//!
//! Owner: bench gate (records).
//! Invariants: none beyond the types.

use std::time::{SystemTime, UNIX_EPOCH};

use crate::history::RunRecord;

/// 2026-09-26: `<model> · <metric=value, …> · <verdict kind>: <reason>`, with
/// `no metrics` / `no verdict` for absent parts, plus the first Warn or Error
/// log line when there is one.
pub(super) fn summarize(record: &RunRecord) -> String {
    let frame = &record.frame;
    let numbers: Vec<String> = frame
        .metrics
        .iter()
        .map(|(k, v)| format!("{k}={v:.2}"))
        .collect();
    let numbers = if numbers.is_empty() {
        "no metrics".to_string()
    } else {
        numbers.join(", ")
    };
    let warning = frame
        .log
        .iter()
        .find(|l| {
            matches!(
                l.level,
                crate::result::LogLevel::Warn | crate::result::LogLevel::Error
            )
        })
        .map(|l| format!(" · warning: {}", l.text));
    let verdict = frame
        .verdict
        .as_ref()
        .map(|v| format!("{:?}: {}", v.kind, v.reason))
        .unwrap_or_else(|| "no verdict".into());
    format!(
        "{} · {} · {}{}",
        record.target_model,
        numbers,
        verdict,
        warning.unwrap_or_default()
    )
}

/// 2026-09-26: Unix seconds now; 0 when the clock reads before the epoch.
pub fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or_default()
}
