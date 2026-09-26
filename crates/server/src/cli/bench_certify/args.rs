// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Arguments for `met bench certify`.
//!
//! Owner: server CLI (`met benchmark certify`).
//! The `///` comments on `CertifyArgs` fields are the `--help` text and carry no date.
//! Invariants: none beyond the types.

use std::path::PathBuf;

/// 2026-09-26: `met bench certify`: run every required gate this commit still owes,
/// and say whether the tree is certified when they are done.
///
/// The plan comes from `gate::check_gates`, the same function
/// `--pull-request-gate-check` uses. On this box each unit is a child
/// `met benchmark run … --pull-request-gate` process; on a `--with-nodes` node
/// it is a metralectl job. The evidence is the record a unit writes, never an
/// exit code.
///
/// Exit codes: `0` certified (every required gate passes at the anchor and the
/// added records agree), or a dry run whose preflight is clean; `1` an error
/// returned before or around the units (usage, preflight refusal) or a dry run
/// with preflight findings; `2` a unit failed, timed out or could not be run,
/// or the final check is not certified; `3` aborted: a perf path moved on the
/// guarded branch, the guard could not answer, or Ctrl-C.
#[derive(clap::Args, Debug, Default)]
pub struct CertifyArgs {
    /// PR number, recorded in the campaign lockfile.
    #[arg(long)]
    pub pr: Option<u64>,
    /// Only these gates (comma-separated group/gate ids), each of which must
    /// still be open. A benchmark group is named by its group id; the shards
    /// it still owes run in any case.
    #[arg(long, value_delimiter = ',', value_name = "ID,...")]
    pub gates: Vec<String>,
    /// How many shards to cut each benchmark group's draw into. Default: two
    /// per box that will run (one box alone runs the whole draw). A partition
    /// already begun at this commit is finished at its own count regardless.
    #[arg(long, value_name = "N")]
    pub shards: Option<usize>,
    /// Print the plan and the preflight verdict, then stop before running.
    #[arg(long, visible_alias = "plan")]
    pub dry_run: bool,
    /// One JSON object per line on stdout instead of the human report.
    #[arg(long)]
    pub json: bool,
    /// Keep running the remaining gates after a verdict FAIL.
    #[arg(long)]
    pub keep_going: bool,
    /// Confirm the gates that need it (`agentic-webserver` executes
    /// model-authored shell in a sandbox). Refused rather than assumed.
    #[arg(long)]
    pub yes: bool,
    /// The box class the records are for (`gb10`). Probed when omitted; an
    /// unprobeable box is an error, never a guess.
    #[arg(long)]
    pub hardware: Option<String>,
    /// The commit the records must name. Must equal HEAD; default HEAD.
    #[arg(long, value_name = "SHA")]
    pub anchor: Option<String>,
    /// `REMOTE/BRANCH` to watch between gates: if a perf path moves there,
    /// the campaign aborts instead of measuring a dead tree. Defaults to the
    /// upstream of HEAD; with no upstream, name one or pass `--no-guard`.
    #[arg(long, value_name = "REMOTE/BRANCH")]
    pub guard_ref: Option<String>,
    /// Run with no drift guard. Only sensible on a tree nobody else pushes to.
    #[arg(long, conflicts_with = "guard_ref")]
    pub no_guard: bool,
    /// Per-gate deadline as a multiple of its expected duration.
    #[arg(long, default_value_t = 3.0)]
    pub timeout_factor: f64,
    /// Where the per-gate logs go. Default `<repo>/.certify/<anchor>/`.
    #[arg(long, value_name = "DIR")]
    pub out: Option<PathBuf>,
    /// metralectl-reachable nodes to run on, in parallel with this one:
    /// `ip[:port]`, `host.local[:port]` or `dns.name[:port]`, passed to
    /// metralectl as given (an omitted port is metralectl's choice).
    #[arg(long, value_delimiter = ',', value_name = "NODE,...")]
    pub with_nodes: Vec<String>,
    /// Do not run anything on this machine — dispatch everything to
    /// `--with-nodes` (a laptop that cannot serve a model).
    #[arg(long, requires = "with_nodes")]
    pub remote_only: bool,
    /// The `metralectl` binary to drive nodes with. Default: the one on PATH.
    #[arg(long, value_name = "PATH")]
    pub metralectl: Option<PathBuf>,
    /// Start a fresh server for every unit on this box instead of keeping
    /// one up across consecutive units that serve the same recipe the same
    /// way (`met benchmark run --serve-reuse`). Reuse is the default: a
    /// unit measures against the server it would have started (same
    /// binary, same rendering, verified) and pays for one model load
    /// instead of one per gate. The record's command line says which.
    #[arg(long)]
    pub no_serve_reuse: bool,
    /// Take no thermal-security action during the campaign: a box at or above
    /// its class's park temperature, or reporting a thermal slowdown, is not
    /// parked (cool-down), only warned about, and keeps taking units. The
    /// records it writes are still judged by the equivalence policy at the
    /// end, so a bad capture is refused there: this flag risks the hardware
    /// and the campaign's time, not the verdict.
    #[arg(long)]
    pub dangerous_ignore_thermals: bool,
}

impl CertifyArgs {
    /// 2026-09-26: Usage errors beyond clap's attributes. The `--remote-only` check
    /// repeats clap's `requires` for a `CertifyArgs` built without clap, as the
    /// tests below build it.
    pub fn validate(&self) -> Result<(), String> {
        if self.remote_only && self.with_nodes.is_empty() {
            return Err("--remote-only needs --with-nodes: nothing would run anywhere".into());
        }
        if !(self.timeout_factor.is_finite() && self.timeout_factor >= 1.0) {
            return Err(format!(
                "--timeout-factor must be a finite number >= 1, got {}",
                self.timeout_factor
            ));
        }
        if self.shards == Some(0) {
            return Err("--shards must be at least 1".into());
        }
        for g in &self.gates {
            if !metrale_bench::gate::REQUIRED_GATES.contains(&g.as_str()) {
                return Err(format!(
                    "--gates names {g}, which is not a required gate ({})",
                    metrale_bench::gate::REQUIRED_GATES.join(", ")
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args() -> CertifyArgs {
        CertifyArgs {
            timeout_factor: 3.0,
            ..CertifyArgs::default()
        }
    }

    #[test]
    fn defaults_validate() {
        assert_eq!(args().validate(), Ok(()));
    }

    #[test]
    fn remote_only_without_nodes_is_refused() {
        let mut a = args();
        a.remote_only = true;
        assert!(a.validate().unwrap_err().contains("--with-nodes"));
    }

    #[test]
    fn a_legacy_shard_id_is_not_a_gate() {
        let mut a = args();
        a.gates = vec!["bfcl-subset-a".into()];
        assert!(a.validate().unwrap_err().contains("not a required gate"));
    }

    #[test]
    fn zero_shards_is_refused_and_one_is_the_whole_draw() {
        let mut a = args();
        a.shards = Some(0);
        assert!(a.validate().unwrap_err().contains("--shards"));
        a.shards = Some(1);
        assert_eq!(a.validate(), Ok(()));
    }

    #[test]
    fn an_unknown_gate_is_refused_with_the_list() {
        let mut a = args();
        a.gates = vec!["nope".into()];
        assert!(a.validate().unwrap_err().contains("decode-floor"));
    }

    #[test]
    fn a_timeout_factor_below_one_is_refused() {
        let mut a = args();
        a.timeout_factor = 0.5;
        assert!(a.validate().is_err());
        a.timeout_factor = f64::NAN;
        assert!(a.validate().is_err());
    }
}
