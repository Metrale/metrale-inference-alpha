// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The serve settings a gate record discloses about the server it
//! measured (`GateRecord::serve_resolved`).
//!
//! `served_by` names a recipe kept in another repository, and
//! `serve_overrides` only the keys changed for the run; neither states what
//! the server ran with. The keys are defined here, in the crate that owns the
//! record, and the server CLI fills them from its rendered serve flags.
//!
//! Owner: bench gate (records).
//! Invariants:
//! - A [`disclosure`] always carries [`SPECULATIVE`]; every other key is
//!   present only when resolved or on.

use std::collections::BTreeMap;

use super::record::GateRecord;

/// 2026-09-26: Key for the `--mtp-gate` regime in force: `force` or `auto`.
pub const MTP_GATE: &str = "mtp_gate";
/// 2026-09-26: Key for whether `--speculative` was on: `true` or `false`.
pub const SPECULATIVE: &str = "speculative";
/// 2026-09-26: Key for `--prefill-codispatch`, present (`true`) only when the
/// rendered serve gave the flag.
pub const PREFILL_CODISPATCH: &str = "prefill_codispatch";
/// 2026-09-26: Key for `--w4a4-downcast`, present (`true`) only when on. The
/// flag defaults to false and has no environment fallback, so an absent key
/// means off.
pub const W4A4_DOWNCAST: &str = "w4a4_downcast";

/// 2026-09-26: The disclosure for a server whose rendered flags resolved to
/// these values.
///
/// `mtp_gate_force` is the server's resolution of `--mtp-gate` and
/// `--hermetic`: `Some(true)` is written `force`, `Some(false)` `auto`.
/// `None` (no flag, so the server's `METRALE_MTP_GATE_FORCE` decides) writes
/// no key rather than a guessed default.
///
/// `prefill_codispatch` writes `true` when the flag was given and no key
/// otherwise; the server's `METRALE_PREFILL_CODISPATCH` then decides, and
/// `serve_env` discloses it when the recipe declares it.
pub fn disclosure(
    mtp_gate_force: Option<bool>,
    speculative: bool,
    prefill_codispatch: bool,
    w4a4_downcast: bool,
) -> BTreeMap<String, String> {
    let mut m = BTreeMap::new();
    m.insert(SPECULATIVE.to_string(), speculative.to_string());
    if let Some(force) = mtp_gate_force {
        m.insert(
            MTP_GATE.to_string(),
            if force { "force" } else { "auto" }.to_string(),
        );
    }
    if prefill_codispatch {
        m.insert(PREFILL_CODISPATCH.to_string(), "true".to_string());
    }
    if w4a4_downcast {
        m.insert(W4A4_DOWNCAST.to_string(), "true".to_string());
    }
    m
}

impl GateRecord {
    /// 2026-09-26: Attach what the gate's serve resolved; see [`disclosure`].
    #[must_use]
    pub fn with_serve_resolved(mut self, resolved: BTreeMap<String, String>) -> Self {
        self.serve_resolved = resolved;
        self
    }
}

#[cfg(test)]
#[path = "record_serve_tests.rs"]
mod record_serve_tests;
