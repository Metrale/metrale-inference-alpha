// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Who runs what, and when: a pure scheduler over units and admitted nodes.
//!
//! Owner: server CLI (`met benchmark certify`).
//! - Speed mode: with more than one node, Speed-class units all go to one node
//!   ([`speed_mode`]). Correctness-class units go to any node.
//! - List scheduling: a free node takes the longest pending unit it may run,
//!   except that a shard of a group it already hosts yields to any other unit
//!   ([`next_for`]). The fleet estimate ([`simulate`]) and the run use the
//!   same `next_for`.
//!
//! Nothing here starts anything; the driver asks [`next_for`] and runs it.
//! Invariants: none beyond the types.

use metrale_bench::hardware::equivalence::{EquivalencePolicy, equivalent};
use metrale_bench::hardware::policy::Sensitivity;

use super::super::plan::Unit;
use super::node::Node;

/// 2026-09-26: Where Speed-class units may go.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SpeedMode {
    /// 2026-09-26: At most one node: Speed units may go to any node.
    Spread,
    /// 2026-09-26: More than one node: Speed units go to `node` only; `why` says why.
    Bundle { node: usize, why: Vec<String> },
}

impl SpeedMode {
    pub fn allows(&self, node: usize) -> bool {
        match self {
            Self::Spread => true,
            Self::Bundle { node: home, .. } => *home == node,
        }
    }
}

/// 2026-09-26: Decide the Speed mode for these nodes: one node (or none) is `Spread`;
/// more than one is always `Bundle`.
///
/// Bundle even when the boxes look equivalent now: this judges the nodes'
/// reports at plan time, while `gate::agreement` judges Speed records from
/// different signers by each record's own hardware capture, so boxes alike
/// now can still be refused at the verdict. `why` lists every pair
/// [`equivalent`] rejects now, or that no policy exists.
///
/// Bundling picks the node with the most free memory, then the coolest
/// chassis.
pub fn speed_mode(nodes: &[Node], policy: Option<EquivalencePolicy>) -> SpeedMode {
    if nodes.len() <= 1 {
        return SpeedMode::Spread;
    }
    let mut why = vec![
        "the Speed class runs on one box by default: equivalence at rest did not hold under \
         load on 2026-09-15, and the class never sets the makespan"
            .to_string(),
    ];
    let Some(policy) = policy else {
        why.push(
            "this class declares no thermal envelope, so no two of its boxes are one box \
             (--dangerous-ignore-thermals)"
                .to_string(),
        );
        return SpeedMode::Bundle {
            node: bundle_home(nodes),
            why,
        };
    };
    for (i, a) in nodes.iter().enumerate() {
        for b in &nodes[i + 1..] {
            if let Err(m) = equivalent(&a.hardware, &b.hardware, &policy) {
                why.push(format!(
                    "{} vs {}: {}",
                    a.addr,
                    b.addr,
                    m.iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }
        }
    }
    SpeedMode::Bundle {
        node: bundle_home(nodes),
        why,
    }
}

/// 2026-09-26: The box a bundled Speed set goes to: the most free memory, then the
/// coolest chassis (a missing reading ranks last).
fn bundle_home(nodes: &[Node]) -> usize {
    (0..nodes.len())
        .max_by(|&x, &y| {
            let fx = nodes[x].free_fraction.unwrap_or(0.0);
            let fy = nodes[y].free_fraction.unwrap_or(0.0);
            fx.partial_cmp(&fy)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    // 2026-09-26: Cooler is better, so compare reversed.
                    let cx = nodes[x].hardware.hottest_chassis_c.unwrap_or(f64::MAX);
                    let cy = nodes[y].hardware.hottest_chassis_c.unwrap_or(f64::MAX);
                    cy.partial_cmp(&cx).unwrap_or(std::cmp::Ordering::Equal)
                })
        })
        .unwrap_or(0)
}

/// 2026-09-26: The unit `node` should take next: the longest pending unit it may run,
/// shards of a group it already hosts yielding to every other unit.
///
/// `pending[i]` says unit `i` is still to run; `placed[i]` is the node a
/// running or finished unit went to (for anti-affinity).
pub fn next_for(
    node: usize,
    units: &[Unit],
    pending: &[bool],
    placed: &[Option<usize>],
    mode: &SpeedMode,
) -> Option<usize> {
    let hosts_shard_of = |group: &str| {
        units
            .iter()
            .enumerate()
            .any(|(j, u)| u.group == Some(group) && placed[j] == Some(node))
    };
    (0..units.len())
        .filter(|&i| pending[i])
        .filter(|&i| units[i].class != Sensitivity::Speed || mode.allows(node))
        .min_by_key(|&i| {
            let crowded = units[i].group.is_some_and(hosts_shard_of);
            // 2026-09-26: `Reverse` makes `min_by_key` pick the longest.
            (crowded, std::cmp::Reverse(units[i].secs()))
        })
}

/// 2026-09-26: A simulated run of the same rule, for the fleet estimate shown
/// before a campaign runs (and by `--dry-run`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plan {
    /// 2026-09-26: Per node, the units in the order it would take them.
    pub queues: Vec<Vec<usize>>,
    /// 2026-09-26: Per node, when it goes idle for good, seconds from start.
    pub finish_at: Vec<u64>,
    pub makespan_secs: u64,
}

/// 2026-09-26: Simulate list scheduling. `build_allowance` is added once to a node that
/// does not have the anchor built.
pub fn simulate(units: &[Unit], nodes: &[Node], mode: &SpeedMode, build_allowance: u64) -> Plan {
    let n = nodes.len();
    let mut pending = vec![true; units.len()];
    let mut placed = vec![None; units.len()];
    let mut queues = vec![Vec::new(); n];
    let mut free_at: Vec<u64> = nodes
        .iter()
        .map(|nd| if nd.built { 0 } else { build_allowance })
        .collect();
    let mut idle = vec![false; n];
    while pending.iter().any(|p| *p) {
        // 2026-09-26: The earliest-free node that can still take something.
        let Some(node) = (0..n).filter(|&k| !idle[k]).min_by_key(|&k| free_at[k]) else {
            break;
        };
        match next_for(node, units, &pending, &placed, mode) {
            Some(i) => {
                pending[i] = false;
                placed[i] = Some(node);
                queues[node].push(i);
                free_at[node] += units[i].secs();
            }
            None => idle[node] = true,
        }
    }
    let makespan_secs = free_at.iter().copied().max().unwrap_or(0);
    Plan {
        queues,
        finish_at: free_at,
        makespan_secs,
    }
}

#[cfg(test)]
#[path = "schedule_tests.rs"]
mod schedule_tests;
