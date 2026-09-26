// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The serve keys `met serve --hermetic` closes, as one table that the
//! bench and server crates both read.
//!
//! Owner: bench gate.
//! Invariants: none beyond the types.
//!
//! `metrale-server`'s `cli::hermetic` re-exports `CLOSED_KEYS` and expands a requested
//! `hermetic=true` into these keys. `gate::bench` refuses a BENCH.toml entry that pins
//! `hermetic=true` without them, because `scoring::check_record` also fails a record
//! whose serve overrides hold a key the baseline does not pin.

/// 2026-09-26: The serve keys `--hermetic` forces, and the values it forces them to.
pub const CLOSED_KEYS: &[(&str, &str)] =
    &[("enable_prefix_caching", "false"), ("mtp_gate", "force")];

/// 2026-09-26: Whether this override map requests the hermetic regime: only the exact
/// string `"true"` does. Any other value, including `"TRUE"` and `"1"`, reads as not
/// requested.
pub fn is_requested(overrides: &std::collections::BTreeMap<String, String>) -> bool {
    overrides.get("hermetic").map(String::as_str) == Some("true")
}

/// 2026-09-26: The `CLOSED_KEYS` entries that `overrides` lacks or holds at another
/// value. Empty means every closed key is pinned at its closed value.
pub fn missing_pins(
    overrides: &std::collections::BTreeMap<String, String>,
) -> Vec<(&'static str, &'static str)> {
    CLOSED_KEYS
        .iter()
        .filter(|(k, v)| overrides.get(*k).map(String::as_str) != Some(*v))
        .map(|(k, v)| (*k, *v))
        .collect()
}

#[cfg(test)]
#[path = "hermetic_tests.rs"]
mod hermetic_tests;
