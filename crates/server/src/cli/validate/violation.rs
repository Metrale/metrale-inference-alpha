// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The validator's vocabulary: one [`Violation`] per finding, the
//! closed-set check, and the operator-facing report.
//!
//! Owner: server CLI.
//! Invariants: none beyond the types.

/// 2026-09-26: One validation failure: what is wrong, why it is wrong, and how to fix it.
pub(super) struct Violation {
    what: String,
    why: String,
    fix: String,
}

impl Violation {
    pub(super) fn new(
        what: impl Into<String>,
        why: impl Into<String>,
        fix: impl Into<String>,
    ) -> Self {
        Self {
            what: what.into(),
            why: why.into(),
            fix: fix.into(),
        }
    }
}

/// 2026-09-26: Push a violation if `value` is not in `allowed`.
pub(super) fn check_enum(v: &mut Vec<Violation>, flag: &str, value: &str, allowed: &[&str]) {
    if !allowed.contains(&value) {
        v.push(Violation::new(
            format!("{flag} '{value}' is not a valid value."),
            format!("valid values are: {}.", allowed.join(", ")),
            format!("pick one of {}.", allowed.join(", ")),
        ));
    }
}

pub(super) fn format_violations(v: &[Violation]) -> String {
    let mut out = format!(
        "Metrale Engine CLI: {} invalid flag combination{} — fix before serving:\n",
        v.len(),
        if v.len() == 1 { "" } else { "s" }
    );
    for (i, vio) in v.iter().enumerate() {
        out.push_str(&format!(
            "\n  [{}] {}\n      why: {}\n      fix: {}\n",
            i + 1,
            vio.what,
            vio.why,
            vio.fix
        ));
    }
    out
}
