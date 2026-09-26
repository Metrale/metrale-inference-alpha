// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The `METRALE_*` environment a gate's server is measured under.
//!
//! A serve lever is a `METRALE_*` variable that is not a harness variable.
//! Levers are not recipe keys, so they do not appear in the argv a leased
//! server is fingerprinted by. Instead the recipe declares the levers it is
//! measured under (its `env:` block, with the gate's BENCH.toml
//! `[benchmarks.serve_env]` pin on top), and [`reconcile`] refuses a lever in
//! the harness's own environment that nothing declared, and a declared lever
//! it carries at another value. The set is fingerprinted into the serve
//! identity (`serve_identity`) and disclosed on the gate record
//! (`gate::GateRecord::serve_env`).
//!
//! It lives in this crate so both ends share one definition: the server
//! fingerprints its own environment for `GET /serve-config`, and the harness
//! computes what that fingerprint must be.
//!
//! Owner: bench (serve environment).
//! Invariants:
//! - [`declared`] returns only `Runtime`-class levers with non-empty values.
//! - [`reconcile`] succeeds only when every present lever is declared at the
//!   declared value; its `env` is then the declaration, whole.

use std::collections::BTreeMap;

use anyhow::{Result, bail};
use sha2::{Digest, Sha256};

/// 2026-09-26: The prefix every lever carries.
pub use metrale_config::levers::PREFIX;

/// 2026-09-26: Whether `name` is a serve lever: any `METRALE_*` name that is
/// not a harness variable.
///
/// The harness variables are the table's `Class::Harness` entries
/// (`metrale_config::levers`): they place, log or build for the harness.
/// Every other name under the prefix counts, including one the table does
/// not declare, so the harness names it in a refusal; `met` itself refuses
/// to start with an undeclared one set (`levers::check` in `main.rs`).
pub fn is_lever(name: &str) -> bool {
    name.starts_with(PREFIX) && !metrale_config::levers::is_harness(name)
}

/// 2026-09-26: The levers in an environment, from any `(name, value)`
/// iterator.
pub fn levers<I, K, V>(env: I) -> BTreeMap<String, String>
where
    I: IntoIterator<Item = (K, V)>,
    K: AsRef<str>,
    V: AsRef<str>,
{
    env.into_iter()
        .filter(|(k, _)| is_lever(k.as_ref()))
        .map(|(k, v)| (k.as_ref().to_string(), v.as_ref().to_string()))
        .collect()
}

/// 2026-09-26: The levers of this process: what the server fingerprints, and
/// what the harness reconciles against the recipe before it serves.
///
/// Lossy on a non-UTF-8 name or value rather than skipping it: a skipped
/// variable would be a lever the fingerprint omits.
pub fn process_levers() -> BTreeMap<String, String> {
    levers(std::env::vars_os().map(|(k, v)| {
        (
            k.to_string_lossy().into_owned(),
            v.to_string_lossy().into_owned(),
        )
    }))
}

/// 2026-09-26: The SHA-256 of a lever set: `name=value` in key order, each
/// NUL-terminated, so `A=1 B=2` and `A=1B=2` differ and the empty set has one
/// fixed digest.
#[must_use]
pub fn fingerprint(levers: &BTreeMap<String, String>) -> String {
    let mut h = Sha256::new();
    for (k, v) in levers {
        h.update(k.as_bytes());
        h.update(b"=");
        h.update(v.as_bytes());
        h.update([0u8]);
    }
    format!("{:x}", h.finalize())
}

/// 2026-09-26: Validate one declaration (a recipe's `env:` block or a
/// BENCH.toml `[benchmarks.serve_env]` table) into the lever set it means.
///
/// `owner` names the declaring file in every refusal.
///
/// # Errors
/// A key that is not a `Runtime` lever (no prefix; a harness, build, dev or
/// tool variable; or a name the lever table does not declare), or a value
/// that is empty after trimming (how a shell spells "unset").
pub fn declared(owner: &str, raw: &BTreeMap<String, String>) -> Result<BTreeMap<String, String>> {
    let mut out = BTreeMap::new();
    for (name, value) in raw {
        if !name.starts_with(PREFIX) {
            bail!(
                "{owner} declares env {name}, which is not a METRALE_* serve lever. The gate \
                 applies exactly the levers a recipe declares and nothing else, so a variable \
                 the server does not read cannot be declared here."
            );
        }
        match metrale_config::levers::lookup(name).map(|l| l.class) {
            Some(metrale_config::levers::Class::Runtime) => {}
            Some(metrale_config::levers::Class::Harness) => bail!(
                "{owner} declares env {name}, which places, logs or builds rather than \
                 measuring ({name} is a harness variable, not a serve lever) — it is the \
                 box's to set, never the recipe's."
            ),
            Some(class) => bail!(
                "{owner} declares env {name}, which the server never reads (it is a {} \
                 variable), so declaring it here would change nothing that is measured.",
                class.as_str()
            ),
            None => bail!(
                "{owner} declares env {name}, which is not a declared lever: nothing reads it, \
                 so it is a typo or a lever that no longer exists, and `met serve` refuses \
                 to start with it set. `met dump-serve-options` lists every declared lever."
            ),
        }
        if value.trim().is_empty() {
            bail!(
                "{owner} declares env {name} with an empty value. An exported-but-empty variable \
                 is how a shell spells \"unset\": declare a value or drop the key."
            );
        }
        out.insert(name.clone(), value.clone());
    }
    Ok(out)
}

/// 2026-09-26: The recipe's declaration under the gate entry's pin; the pin
/// wins on a clash.
#[must_use]
pub fn merge_declared(
    recipe: BTreeMap<String, String>,
    baseline: BTreeMap<String, String>,
) -> BTreeMap<String, String> {
    let mut out = recipe;
    out.extend(baseline);
    out
}

/// 2026-09-26: What the harness must do to serve `declared` from an
/// environment that already holds `present`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Reconciled {
    /// 2026-09-26: The lever set the server runs under: the declaration,
    /// whole.
    pub env: BTreeMap<String, String>,
    /// 2026-09-26: The declared levers the harness does not already carry:
    /// what a child serve is given on top of the inherited environment, and
    /// what an in-process serve refuses to start without.
    pub missing: BTreeMap<String, String>,
}

/// 2026-09-26: Reconcile the recipe's declaration with the harness's own
/// levers ([`process_levers`]).
///
/// # Errors
/// A present lever that `declared` does not name: refused, not inherited and
/// not stripped, because either would change what the gate measures without
/// a trace. A declared lever present at another value: refused, because
/// neither side may win silently.
pub fn reconcile(
    owner: &str,
    declared: &BTreeMap<String, String>,
    present: &BTreeMap<String, String>,
) -> Result<Reconciled> {
    let undeclared: Vec<String> = present
        .iter()
        .filter(|(k, _)| !declared.contains_key(*k))
        .map(|(k, v)| format!("{k}={v}"))
        .collect();
    if !undeclared.is_empty() {
        bail!(
            "the harness environment carries {} serve lever(s) that {owner} does not declare: \
             {}. A lever the recipe did not ask for changes what the gate measures \
             (METRALE_FP8_ROWWISE=1 builds FP8 twin weights that a 0.70-util serve cannot \
             boot with — #1242), so it is refused rather than inherited. Unset it — on a node \
             agent, remove it from bench.yaml `env:`, which reaches EVERY gate on the box — or \
             declare it in the recipe's `env:` (or the gate's [benchmarks.serve_env]) if the \
             gate is measured under it.",
            undeclared.len(),
            undeclared.join(", ")
        );
    }
    let mut missing = BTreeMap::new();
    let mut contradicted = Vec::new();
    for (k, want) in declared {
        match present.get(k) {
            Some(got) if got == want => {}
            Some(got) => {
                contradicted.push(format!("{k}: harness {got:?}, {owner} declares {want:?}"))
            }
            None => {
                missing.insert(k.clone(), want.clone());
            }
        }
    }
    if !contradicted.is_empty() {
        bail!(
            "the harness environment contradicts {owner} on {} serve lever(s): {}. Neither value \
             wins silently: unset the variable so the declaration applies, or change the \
             declaration.",
            contradicted.len(),
            contradicted.join("; ")
        );
    }
    Ok(Reconciled {
        env: declared.clone(),
        missing,
    })
}

#[cfg(test)]
#[path = "serve_env_tests.rs"]
mod tests;
