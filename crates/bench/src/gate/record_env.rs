// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: What a gate record discloses about the environment its server
//! ran under: the performance controls with their defaults filled in
//! (`perf_env`) and the `METRALE_*` levers the gate applied (`serve_env`).
//!
//! Owner: bench gate (records).
//! Invariants:
//! - [`resolve_perf_env`] returns every `PERF_CONTROLS` key.

use std::collections::BTreeMap;

use super::record::GateRecord;

/// 2026-09-26: The performance controls a gate record discloses, each with
/// the value recorded when it is unset.
///
/// - The two co-dispatch timing controls carry the defaults the scheduler
///   passes to `num` in `SchedLevers::from_env`. They are copied because
///   `metrale-bench` does not depend on `metrale-server`;
///   `perf_env_defaults_match_the_scheduler` reads the scheduler's source.
/// - `METRALE_NO_W4A16_TC`, `METRALE_NO_MTP_TC` and
///   `METRALE_NO_BORROW_STREAK_LIMIT` are presence kill switches: any
///   non-empty value, `0` included, turns the path off, so the default is the
///   literal `unset`, under which the path ran.
/// - `METRALE_W4A16_TC_WIDE` is a presence opt-in that raises
///   `narrow_gemv_max_rows` to `WIDE_MAX_ROWS`, so `unset` means it did not
///   run.
///
/// `record_env_tests` pins each switch rule against its lever's source. The
/// `--prefill-codispatch` and `--w4a4-downcast` flags are disclosed in
/// `serve_resolved` instead.
const PERF_CONTROLS: [(&str, &str); 6] = [
    ("METRALE_PREFILL_CODISPATCH_WINDOW_MS", "100"),
    ("METRALE_PREFILL_CODISPATCH_SETTLE_MS", "10"),
    ("METRALE_NO_W4A16_TC", "unset"),
    ("METRALE_W4A16_TC_WIDE", "unset"),
    ("METRALE_NO_MTP_TC", "unset"),
    ("METRALE_NO_BORROW_STREAK_LIMIT", "unset"),
];

/// 2026-09-26: Resolve `PERF_CONTROLS` through `lookup`, substituting each
/// default for an unset, empty or whitespace-only value. Takes `lookup` so
/// tests need not set process environment variables.
pub fn resolve_perf_env(lookup: impl Fn(&str) -> Option<String>) -> BTreeMap<String, String> {
    PERF_CONTROLS
        .iter()
        .map(|(key, default)| {
            let value = lookup(key).filter(|v| !v.trim().is_empty());
            (
                (*key).to_string(),
                value.unwrap_or_else(|| (*default).to_string()),
            )
        })
        .collect()
}

impl GateRecord {
    /// 2026-09-26: Attach the `METRALE_*` levers the gate applied to its
    /// server (`serve_env::Reconciled::env`) and re-resolve `perf_env` with
    /// the applied set first and this process's environment second. A leased
    /// server is a child given the declared levers this process lacks, so
    /// this process's environment alone can miss a value the server had.
    #[must_use]
    pub fn with_serve_env(mut self, env: BTreeMap<String, String>) -> Self {
        self.perf_env = resolve_perf_env(|k| env.get(k).cloned().or_else(|| std::env::var(k).ok()));
        self.serve_env = env;
        self
    }
}

#[cfg(test)]
#[path = "record_env_tests.rs"]
mod tests;
