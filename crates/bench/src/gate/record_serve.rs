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
//! - A [`disclosure`] always carries [`SPECULATIVE`], [`WEIGHT_QUANTIZATION`] and
//!   [`ACTIVATION_QUANTIZATION`]; every other key is present only when resolved or on.

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
/// 2026-09-28: Key for `--weight-quantization`, always present on a record written since the
/// flag exists, with the tier's name (`declared`, `nvfp4`). A record without it predates the
/// flag, and its server ran what `nvfp4` names.
pub const WEIGHT_QUANTIZATION: &str = "weight_quantization";
/// 2026-09-27: Key for `--expert-quantization`, present only for a tier other than the default
/// `fp8`, with the tier's name as the value (`nvfp4-gate-up`, `nvfp4`). The flag has no
/// environment fallback, so an absent key means `fp8`.
pub const EXPERT_QUANTIZATION: &str = "expert_quantization";
/// 2026-09-30: Key for `--activation-quantization`, always present on a record written since the
/// flag exists, with the value's canonical form (`adaptive`, `declared`, a ladder). A record
/// without it predates the flag, and its server ran what `adaptive` names.
pub const ACTIVATION_QUANTIZATION: &str = "activation_quantization";

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
///
/// `expert_quantization` is the tier's name when it is not the default `fp8`, else `None`.
/// `weight_quantization` is the `--weight-quantization` tier's name, always written, and so is
/// `activation_quantization`, the `--activation-quantization` value's canonical form.
pub fn disclosure(
    mtp_gate_force: Option<bool>,
    speculative: bool,
    prefill_codispatch: bool,
    w4a4_downcast: bool,
    expert_quantization: Option<&str>,
    weight_quantization: &str,
    activation_quantization: &str,
) -> BTreeMap<String, String> {
    let mut m = BTreeMap::new();
    m.insert(
        ACTIVATION_QUANTIZATION.to_string(),
        activation_quantization.to_string(),
    );
    m.insert(SPECULATIVE.to_string(), speculative.to_string());
    m.insert(
        WEIGHT_QUANTIZATION.to_string(),
        weight_quantization.to_string(),
    );
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
    if let Some(tier) = expert_quantization {
        m.insert(EXPERT_QUANTIZATION.to_string(), tier.to_string());
    }
    m
}

/// 2026-09-28: Key for `--forward`, present only for a forward other than `legacy`.
pub const FORWARD: &str = "forward";
/// 2026-09-28: Key for the live decode plan's digest (`metrale_circuit::digest::plan_digest`),
/// present only when the server runs a circuit forward.
pub const PLAN_DIGEST: &str = "plan_digest";

/// 2026-09-28: What a server reports about its forward (`GET /forward`): the server fills it
/// from its model, the harness reads it into the record.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LiveForward {
    /// 2026-09-28: `legacy`, `circuit` or `circuit-reference`.
    pub forward: String,
    /// 2026-09-28: The decode plan's digest; `None` under `legacy`.
    #[serde(default)]
    pub plan_digest: Option<String>,
    /// 2026-10-01: The slot count `--max-batch-size auto` resolved to; `None` for an explicit
    /// count (which the rendered serve already states).
    #[serde(default)]
    pub auto_max_batch_size: Option<usize>,
    /// 2026-10-02: The MoE expert-table decision the serve's memory plan made before load
    /// (`build` or `skip`); `None` when the loader reads none.
    #[serde(default)]
    pub moe_expert_tables: Option<String>,
}

/// 2026-10-01: Key for the slot count `--max-batch-size auto` resolved to, written `auto:<n>`;
/// present only for `auto`.
pub const MAX_BATCH_SIZE: &str = "max_batch_size";
/// 2026-10-02: Key for the MoE expert-table decision, present (`skip`) only when the serve's
/// memory plan dropped the transposed MoE prefill tables. Absent means built, as every serve
/// before the decision existed did.
pub const MOE_EXPERT_TABLES: &str = "moe_expert_tables";

/// 2026-09-28: Add the live forward to `resolved`: [`FORWARD`] when it is not `legacy`, and
/// [`PLAN_DIGEST`]. `requested` is the forward the rendered serve asked for; a server running
/// another one, or a circuit without a digest, or legacy with one, is refused: the record would
/// state a configuration the measurement did not run.
pub fn merge_live_forward(
    resolved: &mut BTreeMap<String, String>,
    requested: &str,
    live: &LiveForward,
) -> Result<(), String> {
    let skipped = match live.moe_expert_tables.as_deref() {
        None | Some("build") => false,
        Some("skip") => true,
        Some(other) => return Err(format!("the server reports MoE expert tables `{other}`")),
    };
    if live.forward != requested {
        return Err(format!(
            "the server runs forward `{}`, the rendered serve asked for `{requested}`",
            live.forward
        ));
    }
    match (live.forward.as_str(), &live.plan_digest) {
        ("legacy", None) => {}
        ("legacy", Some(d)) => return Err(format!("a legacy forward reports plan digest {d}")),
        (other, Some(d)) => {
            resolved.insert(FORWARD.to_string(), other.to_string());
            resolved.insert(PLAN_DIGEST.to_string(), d.clone());
        }
        (other, None) => return Err(format!("forward `{other}` reports no plan digest")),
    }
    if let Some(n) = live.auto_max_batch_size {
        resolved.insert(MAX_BATCH_SIZE.to_string(), format!("auto:{n}"));
    }
    if skipped {
        resolved.insert(MOE_EXPERT_TABLES.to_string(), "skip".to_string());
    }
    Ok(())
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
