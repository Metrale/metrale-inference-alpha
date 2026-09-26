// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Startup gate that refuses an SSM snapshot tier on a model with no recurrent state.
//!
//! Owner: model-engine SSM tier.
//! Invariants:
//! - With none of `SSM_TIER_VARS` set, or on a model where
//!   [`ModelConfig::has_recurrent_state`] is true, the gate returns `Ok(())`.
//! - Otherwise it returns an error that names the model type and every var found set.

use anyhow::{Result, bail};
use metrale_config::ModelConfig;

/// 2026-09-25: The env vars whose presence counts as a tier request. The namespace
/// overrides `METRALE_SSM_SWAP_NS` and `METRALE_SSM_DECODE_NS` select no tier and are
/// not listed.
const SSM_TIER_VARS: [&str; 5] = [
    "METRALE_SSM_TIER",
    "METRALE_SSM_RDMA_TIER",
    "METRALE_SSM_SWAP",
    "METRALE_SSM_DECODE_TIER",
    "METRALE_SSM_DECODE_RING_ROLL",
];

/// 2026-09-25: Fail when any of `SSM_TIER_VARS` is present (by `var_os`, so any
/// value counts) and the model has no recurrent state. `TransformerModel::new` calls
/// it before it builds the SSM snapshot pool, so before any tier store or peer connect.
pub(crate) fn ensure_ssm_tier_capability(config: &ModelConfig) -> Result<()> {
    let set: Vec<&str> = SSM_TIER_VARS
        .iter()
        .copied()
        .filter(|v| std::env::var_os(v).is_some())
        .collect();
    ensure_ssm_tier_capability_from(config, &set)
}

/// 2026-09-25: Env-free core of [`ensure_ssm_tier_capability`]; `set_vars` is the
/// list of tier vars found set.
pub(crate) fn ensure_ssm_tier_capability_from(
    config: &ModelConfig,
    set_vars: &[&str],
) -> Result<()> {
    if set_vars.is_empty() || config.has_recurrent_state() {
        return Ok(());
    }
    bail!(
        "model '{}' has no recurrent state (num_ssm_layers=0) — the SSM snapshot \
         tier cannot be populated by this model. Unset {} for this serve (fleet-wide \
         env files now need per-model curation; a tier request on an incapable model \
         was previously a SILENT no-op and is now a startup error).",
        config.model_type,
        set_vars.join(", "),
    );
}

#[cfg(test)]
#[path = "capability_tests.rs"]
mod tests;
