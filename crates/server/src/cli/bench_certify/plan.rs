// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: What a campaign has to run, from the statuses `gate::check_gates` produced.
//!
//! Owner: server CLI (`met benchmark certify`).
//! Measured durations and owed shards come in as closures, so nothing here
//! reads the filesystem.
//! Invariants: none beyond the types.

use std::collections::BTreeMap;

use anyhow::{Result, bail};
use metrale_bench::gate::{self, GateStatus};
use metrale_bench::hardware::limits::TimingLimits;
use metrale_bench::hardware::policy::Sensitivity;
use metrale_bench::registry;

/// 2026-09-26: Where a unit's duration estimate came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Estimate {
    /// 2026-09-26: The descriptor's `expected_secs`.
    Declared(u64),
    /// 2026-09-26: The newest completed run in the run history (`<METRALE_HOME>/runs`).
    Measured { secs: u64, recorded_at: u64 },
}

/// 2026-09-26: One benchmark the campaign runs: a plain gate, or one shard of a group.
#[derive(Clone, Debug)]
pub struct Unit {
    /// 2026-09-26: The benchmark id the child runs; for a shard, the group's id.
    pub id: &'static str,
    /// 2026-09-26: The group this unit is a shard of, if any. The verdict belongs to the
    /// group; the unit only produces a record.
    pub group: Option<&'static str>,
    /// 2026-09-26: `(index, count)` of the slice this unit measures; `None` for a plain
    /// gate. Passed to the child as `--param shard=index/count`.
    pub shard: Option<(usize, usize)>,
    pub class: Sensitivity,
    pub estimate: Estimate,
    pub needs_confirmation: bool,
    /// 2026-09-26: The class's `serve_allowance_s` (`HARDWARE.toml`
    /// `[benchmarks.limits.timing]`), which `deadline` adds.
    pub serve_allowance_s: u64,
}

impl Unit {
    /// 2026-09-26: What the operator sees: `bfcl-subset[3/8]` for a shard, the id alone
    /// otherwise.
    pub fn label(&self) -> String {
        match self.shard {
            Some((i, n)) => format!("{}[{i}/{n}]", self.id),
            None => self.id.to_string(),
        }
    }

    /// 2026-09-26: The filename-safe spelling: `bfcl-subset-s3of8`, with the same shard
    /// tail as the record's file name (`gate::shard_suffix`). The unit's log file
    /// and remote job key are built from it.
    pub fn file_stem(&self) -> String {
        format!("{}{}", self.id, gate::shard_suffix(self.shard))
    }

    /// 2026-09-26: The `--param shard=i/n` the child needs, if this unit is a shard.
    pub fn shard_param(&self) -> Option<String> {
        self.shard.map(|(i, n)| format!("shard={i}/{n}"))
    }
}

/// 2026-09-26: How many shards a group's draw is cut into when `--shards` is not given:
/// two per box that will run, so the scheduler has slices to balance, and one
/// when a single box runs, since every shard pays `shard_overhead_s`. A
/// partition already begun at this commit is finished at its own count
/// regardless (`gate::shards_owed`).
pub fn shard_count(boxes: usize) -> usize {
    if boxes <= 1 { 1 } else { 2 * boxes }
}

impl Unit {
    /// 2026-09-26: When to give up on this unit: the class's serve allowance (server
    /// start and checkpoint load, which the measured run time does not include)
    /// plus the estimate scaled by `timeout_factor` (>= 1,
    /// `CertifyArgs::validate`). The local loop and every fleet worker call
    /// this; a remote node without the anchor built also gets
    /// `build_allowance_s` (`remote::runner::deadline_for`).
    pub fn deadline(&self, timeout_factor: f64) -> std::time::Duration {
        std::time::Duration::from_secs(self.serve_allowance_s)
            + std::time::Duration::from_secs((self.secs() as f64 * timeout_factor) as u64)
    }

    /// 2026-09-26: Seconds the scheduler plans with.
    pub fn secs(&self) -> u64 {
        match self.estimate {
            Estimate::Declared(s) | Estimate::Measured { secs: s, .. } => s,
        }
    }
}

/// 2026-09-26: The required gates that are not `Pass` at this commit, in `REQUIRED_GATES`
/// order, or the subset the caller named, each of which must be open.
pub fn remaining(
    statuses: &BTreeMap<String, GateStatus>,
    only: &[String],
) -> Result<Vec<&'static str>> {
    let open: Vec<&'static str> = gate::REQUIRED_GATES
        .iter()
        .copied()
        .filter(|id| !matches!(statuses.get(*id), Some(GateStatus::Pass)))
        .collect();
    if only.is_empty() {
        return Ok(open);
    }
    let mut chosen = Vec::new();
    for want in only {
        match open.iter().find(|id| **id == want.as_str()) {
            Some(id) => chosen.push(*id),
            None => bail!(
                "{want} already passes at this commit; pass no --gates to run everything \
                 that is still open ({})",
                if open.is_empty() {
                    "nothing".to_string()
                } else {
                    open.join(", ")
                }
            ),
        }
    }
    Ok(chosen)
}

/// 2026-09-26: Which shards of a group a certification still owes, as `(index, count)`.
/// Injected so the plan reads no files; the real one is [`gate::shards_owed`],
/// asked for the count the campaign wants.
pub type Owed<'a> = &'a dyn Fn(&'static gate::group::BenchmarkGroup) -> Vec<(usize, usize)>;

/// 2026-09-26: A fresh partition of `n` shards: every index owed.
pub fn fresh_partition(
    n: usize,
) -> impl Fn(&'static gate::group::BenchmarkGroup) -> Vec<(usize, usize)> {
    move |_| (0..n).map(|i| (i, n)).collect()
}

/// 2026-09-26: The runs that produce a gate's evidence: the shards a group still owes,
/// the gate itself otherwise.
///
/// `owed` decides both which shards run and at what count, so a shard the
/// gate would already accept at this commit is not re-measured.
pub fn expand(
    gate_id: &'static str,
    owed: Owed<'_>,
) -> Vec<(&'static str, Option<(usize, usize)>)> {
    match gate::group::find(gate_id) {
        Some(group) => owed(group)
            .into_iter()
            .map(|s| (gate_id, Some(s)))
            .collect(),
        None => vec![(gate_id, None)],
    }
}

/// 2026-09-26: Units for these gates. `measured(id)` returns `(secs, recorded_at)` of the
/// newest completed run of `id`, when there is one (a zero is ignored); `owed`
/// says which shards of a group still need a record. A shard's estimate is
/// [`shard_secs`] of the group's whole-draw estimate.
pub fn units(
    gates: &[&'static str],
    measured: &dyn Fn(&str) -> Option<(u64, u64)>,
    owed: Owed<'_>,
    timing: &TimingLimits,
) -> Result<Vec<Unit>> {
    let mut out = Vec::new();
    for gate_id in gates {
        for (id, shard) in expand(gate_id, owed) {
            let Some(d) = registry::find(id) else {
                bail!("{id} is a required gate but not a registered benchmark");
            };
            let mut estimate = match measured(id) {
                Some((secs, recorded_at)) if secs > 0 => Estimate::Measured { secs, recorded_at },
                _ => Estimate::Declared(d.expected_secs),
            };
            if let Some((_, n)) = shard {
                estimate = match estimate {
                    Estimate::Declared(s) => Estimate::Declared(shard_secs(s, n, timing)),
                    Estimate::Measured { secs, recorded_at } => Estimate::Measured {
                        secs: shard_secs(secs, n, timing),
                        recorded_at,
                    },
                };
            }
            out.push(Unit {
                id,
                group: gate::group::find(id).map(|g| g.id),
                shard,
                class: d.sensitivity,
                estimate,
                needs_confirmation: d.needs_confirmation,
                serve_allowance_s: timing.serve_allowance_s,
            });
        }
    }
    Ok(out)
}

/// 2026-09-26: A shard's planning estimate: its share of the whole draw plus
/// `shard_overhead_s`, floored at `shard_floor_s` (the class's
/// `[benchmarks.limits.timing]`; GB10's values and the measurement behind them
/// are in `kernels/gb10/HARDWARE.toml`).
pub fn shard_secs(whole: u64, n: usize, timing: &TimingLimits) -> u64 {
    (whole / n as u64 + timing.shard_overhead_s).max(timing.shard_floor_s)
}

/// 2026-09-26: The order one box runs its units in: group shards first, longest first;
/// then units of the Speed class shortest-first; then the rest shortest-first.
/// Ties break by id, then shard, so the same plan always prints the same list.
pub fn order_local(mut units: Vec<Unit>) -> Vec<Unit> {
    fn rank(u: &Unit) -> (u8, u64, &'static str, Option<(usize, usize)>) {
        let tier = match (u.group, u.class) {
            (Some(_), _) => 0,
            (None, Sensitivity::Speed) => 1,
            (None, Sensitivity::Correctness) => 2,
        };
        let secs = if tier == 0 {
            u64::MAX - u.secs()
        } else {
            u.secs()
        };
        (tier, secs, u.id, u.shard)
    }
    units.sort_by_key(rank);
    units
}

/// 2026-09-26: The wall-clock estimate of running `units` back to back.
pub fn serial_estimate_secs(units: &[Unit]) -> u64 {
    units.iter().map(Unit::secs).sum()
}

#[cfg(test)]
#[path = "plan_tests.rs"]
mod plan_tests;
