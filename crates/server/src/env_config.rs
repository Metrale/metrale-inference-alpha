// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Strict parsing of the numeric process-scoped `METRALE_*`
//! settings: the rate limits (`rate_limiter`), the response store and the
//! conversation store.
//!
//! Owner: server config.
//! Invariants: `parse_min` never substitutes a default. Unset or blank is
//! `Ok(None)`; any other value that does not parse, or parses below `min`, is
//! `Err`.
//!
//! A lenient read like this one turns a typo into the default, and the
//! default rate limit, 0, means no limit:
//!
//! ```ignore
//! let rpm = std::env::var("METRALE_RATE_LIMIT_RPM")
//!     .ok()
//!     .and_then(|s| s.parse().ok())   // ← a typo lands here
//!     .unwrap_or(0);                  // ← and silently becomes "off"
//! ```
//!
//! [`parse_min`] takes the raw value rather than reading the environment, so
//! tests call it without `set_var`, which is process-global. Each caller's
//! `from_env` reads the variable. `Carried::from_env`
//! (`main_modules/serve_load.rs`) runs all three before the listener binds,
//! and `serve` returns its error, so a bad value stops startup.

use std::fmt::Display;
use std::str::FromStr;

/// 2026-09-26: Parse an optional numeric setting, trimmed at both ends.
///
/// Returns `Ok(None)` when `raw` is unset, empty or whitespace-only; the caller
/// applies its default. A value that does not parse, or parses below `min`, is
/// an `Err` whose message names `key`, quotes `raw`, and states `meaning` and
/// `min`.
pub fn parse_min<T>(
    key: &str,
    raw: Option<&str>,
    min: T,
    meaning: &str,
) -> Result<Option<T>, String>
where
    T: FromStr + PartialOrd + Display + Copy,
{
    let Some(raw) = raw else { return Ok(None) };
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    let parsed: T = trimmed
        .parse()
        .map_err(|_| describe(key, raw, min, meaning, "is not a whole number"))?;
    if parsed < min {
        return Err(describe(
            key,
            raw,
            min,
            meaning,
            "is below the smallest value this setting accepts",
        ));
    }
    Ok(Some(parsed))
}

/// 2026-09-26: The error text of `parse_min`, in the what / why / fix shape of
/// `cli::validate`'s `Violation`.
fn describe<T: Display>(key: &str, raw: &str, min: T, meaning: &str, problem: &str) -> String {
    format!(
        "{key}={raw:?} {problem}.\n      \
         why: {meaning} — expected a whole number >= {min}.\n      \
         fix: correct the value, or unset {key} to use the built-in default. \
         It is NOT ignored: the server refuses to start rather than serve a \
         configuration you did not ask for."
    )
}

#[cfg(test)]
#[path = "env_config_tests.rs"]
mod tests;
