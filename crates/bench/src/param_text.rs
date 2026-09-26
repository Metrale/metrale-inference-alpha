// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Text ↔ [`ParamValues`]: the conversions a run record
//! (`history.rs`) and a `--param` flag need.
//!
//! A stored run describes its whole configuration as text. Text goes back
//! through [`crate::params::ParamKind::parse`], the domain check the edit box
//! uses, so an old value that is no longer legal is reported, not accepted.
//!
//! Owner: bench (parameters).
//! Invariants:
//! - [`ParamValues::from_overrides`] refuses an unknown key and an
//!   out-of-domain value; it never returns a partial set.

use std::collections::BTreeMap;

use anyhow::{Result, bail};

use crate::params::{ParamSpec, ParamValue, ParamValues};

impl ParamValues {
    /// 2026-09-26: Every `(key, value)` pair, in key order (the map is a
    /// `BTreeMap`), so records built from the same values diff cleanly.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &ParamValue)> {
        self.0.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// 2026-09-26: The whole configuration as text, for persistence.
    pub fn to_strings(&self) -> BTreeMap<String, String> {
        self.iter()
            .map(|(k, v)| (k.to_string(), v.to_edit_string()))
            .collect()
    }

    /// 2026-09-26: Schema defaults with `overrides` applied on top.
    ///
    /// Every value is routed through [`ParamKind::parse`], so an out-of-domain
    /// override fails here rather than mid-run. An unknown key is an error
    /// naming the valid ones: an ignored `--param` typo would measure
    /// something other than what was asked for.
    ///
    /// [`ParamKind::parse`]: crate::params::ParamKind::parse
    pub fn from_overrides<'a>(
        specs: &[ParamSpec],
        overrides: impl IntoIterator<Item = (&'a str, &'a str)>,
    ) -> Result<Self> {
        let mut values = Self::defaults(specs);
        for (key, raw) in overrides {
            let Some(spec) = specs.iter().find(|s| s.key == key) else {
                let known: Vec<&str> = specs.iter().map(|s| s.key).collect();
                bail!(
                    "unknown parameter {key:?} — this benchmark takes: {}",
                    known.join(", ")
                );
            };
            let value = spec
                .kind
                .parse(raw)
                .map_err(|e| anyhow::anyhow!("{}: {e}", spec.label))?;
            values.set(key, value);
        }
        Ok(values)
    }
}

#[cfg(test)]
#[path = "param_text_tests.rs"]
mod tests;
