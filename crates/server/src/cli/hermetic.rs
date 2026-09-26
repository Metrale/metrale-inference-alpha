// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `--hermetic`: serve so that no state produced for one request reaches another, for known-answer tests.
//!
//! Owner: server CLI.
//! Invariants:
//! - Outside `validate.rs`, production code reads the raw `enable_prefix_caching` and
//!   `mtp_gate` fields only through the resolvers here; the test
//!   `no_production_code_reads_the_raw_fields_behind_hermetic` scans `src/` for other reads.
//!
//! The resolvers close two channels: the radix KV prefix cache and the MTP gate's
//! probes. `serve_flags.rs` also publishes the flag to `metrale_gpu_runtime::set_hermetic`,
//! and under it the SSM snapshot lookup session-gates every entry
//! (`radix_tree/snapshot_session.rs`).

/// 2026-09-26: The serve keys `--hermetic` forces and the values it forces them to,
/// re-exported from `metrale_bench::gate::hermetic`. metrale-bench holds the table
/// because `gate::bench` also reads it, to refuse a BENCH.toml entry that pins
/// `hermetic=true` without these keys, and metrale-bench cannot depend on this crate.
/// `hermetic_closures_match_the_resolvers` checks the table against the resolvers.
pub(crate) use metrale_bench::gate::hermetic::CLOSED_KEYS;

/// 2026-09-26: When `requested` asks for `hermetic=true`, add each `CLOSED_KEYS` entry
/// the map does not already hold. A key already present keeps its value, so an explicit
/// contradiction still reaches `validate_serve_args` and is refused there.
/// `bench_serve_plan.rs` calls this before the recipe renders.
pub(crate) fn expand(
    mut requested: std::collections::BTreeMap<String, String>,
) -> std::collections::BTreeMap<String, String> {
    if !metrale_bench::gate::hermetic::is_requested(&requested) {
        return requested;
    }
    for (k, v) in CLOSED_KEYS {
        requested
            .entry((*k).to_string())
            .or_insert_with(|| (*v).to_string());
    }
    requested
}

/// 2026-09-26: Whether the radix KV prefix cache runs: as requested, and never under
/// `--hermetic`. Its KV blocks are keyed on token content and adapter, not on session,
/// so one request's blocks are reachable by any later request sharing a prefix.
pub(crate) fn prefix_caching_enabled(requested: bool, hermetic: bool) -> bool {
    requested && !hermetic
}

/// 2026-09-26: The MTP gate setting passed to `scheduler::levers::resolve_mtp_gate_force`.
///
/// `None` means no flag was given, so `METRALE_MTP_GATE_FORCE` decides. Under
/// `--hermetic` it is `Some(true)`, so the environment cannot re-arm the gate outside
/// the recorded regime. An armed gate probes the other arm whenever its token counter
/// reaches the event interval, and only a probe or a switch resets that counter, never a
/// request boundary (`metrale_speculative::mtp_gate`), so which request a probe lands on
/// depends on everything served before it. `force` leaves the scheduler with no gate at all
/// (`scheduler/core/mod.rs`).
pub(crate) fn mtp_gate_force(requested: Option<&str>, hermetic: bool) -> Option<bool> {
    if hermetic {
        return Some(true);
    }
    requested.map(|gate| gate == "force")
}

impl super::ServeArgs {
    /// 2026-09-26: The prefix-caching setting in force.
    pub(crate) fn prefix_caching_enabled(&self) -> bool {
        prefix_caching_enabled(self.enable_prefix_caching, self.hermetic)
    }

    /// 2026-09-26: The MTP gate setting in force.
    pub(crate) fn mtp_gate_force(&self) -> Option<bool> {
        mtp_gate_force(self.mtp_gate.as_deref(), self.hermetic)
    }

    /// 2026-09-26: Pin the speculation lane to the verify arm (`--speculative
    /// --mtp-gate force`). The dashboard's bench host calls it for its `MtpForce` lane;
    /// it is the only write to `mtp_gate` outside argument parsing.
    pub(crate) fn pin_mtp_force(&mut self) {
        self.speculative = true;
        self.mtp_gate = Some("force".to_string());
    }
}

#[cfg(test)]
#[path = "hermetic_tests.rs"]
mod tests;
