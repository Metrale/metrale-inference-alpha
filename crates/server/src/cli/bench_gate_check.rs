// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `met benchmark --pull-request-gate-check`: the gate verdict
//! for the checked-out commit, read from the repository without an endpoint.
//!
//! Owner: server CLI (`met benchmark`).
//! Invariants:
//! - Once the statuses are read, the exit code depends only on them; the intent
//!   report is printed after them and does not feed the exit code.

use anyhow::Result;
use metrale_bench::gate;

use super::bench_run::repo_root;

/// 2026-09-26: Print one line per required gate, in `REQUIRED_GATES` order,
/// and return the ids that are not `Pass`. `met benchmark certify` prints its
/// final verdict with it too (`bench_certify/report.rs`).
pub(super) fn print_statuses(
    gates: &std::collections::BTreeMap<String, gate::GateStatus>,
) -> Vec<&'static str> {
    let mut open = Vec::new();
    for id in gate::REQUIRED_GATES {
        match &gates[id] {
            gate::GateStatus::Pass => println!("  PASS  {id}"),
            gate::GateStatus::Fail(reasons) => {
                println!("  FAIL  {id}");
                for reason in reasons {
                    println!("        - {reason}");
                }
                open.push(id);
            }
            gate::GateStatus::Missing(reason) => {
                println!("  NONE  {id} — {reason}");
                open.push(id);
            }
        }
    }
    open
}

/// 2026-09-26: `--pull-request-gate-check`: print every required gate's status
/// at the checked-out commit, then the advisory intent for `pr`. Returns 0
/// when every gate passes and 1 otherwise; the caller exits with it.
pub(super) fn gate_check_cmd(pr: Option<u64>) -> Result<i32> {
    let root = repo_root()?;
    let sha = gate::git_sha(&root)?;
    let gates = gate::check_gates(&root, &sha);
    println!("gate check for {sha} ({})", root.display());
    let open = print_statuses(&gates);
    // 2026-09-26: Advisory only: what the PR's classified intent would ask for.
    // The ledger is printed, never used for the verdict: the governance crate
    // doc keeps the gate independent of a file any job can append to.
    let roots = gate::pr_taxonomy::load(&root);
    let source = gate::required::intent_source(&root, pr);
    println!();
    match (&source, &roots) {
        (gate::required::IntentSource::NotRequested, _) => {
            println!("intent: not evaluated (no --pr)");
        }
        (gate::required::IntentSource::NotRecorded { ledger }, _) => {
            println!("intent: nothing recorded ({})", ledger.display());
        }
        (gate::required::IntentSource::Degraded { reason }, _) => {
            println!("intent: DEGRADED — {reason}");
        }
        (_, Err(e)) => println!("intent: taxonomy unreadable — {e:#}"),
        (
            gate::required::IntentSource::Recorded {
                categories,
                skipped,
            },
            Ok(roots),
        ) => {
            let report = gate::required::report(&[], source.clone(), roots);
            println!(
                "intent: {} classification(s){}",
                categories.len(),
                if *skipped > 0 {
                    format!(", {skipped} abstained/errored (not counted)")
                } else {
                    String::new()
                }
            );
            for c in categories {
                println!("        {}", c.join("/"));
            }
            let implied = report.set.by_intent;
            println!(
                "        implies: {}",
                if implied.is_empty() {
                    "(nothing)".to_string()
                } else {
                    implied.iter().cloned().collect::<Vec<_>>().join(", ")
                }
            );
            println!("        advisory — does not change the verdict above or the exit code");
        }
    }

    if open.is_empty() {
        println!("all {} required gates pass", gate::REQUIRED_GATES.len());
        Ok(0)
    } else {
        println!(
            "{} bench(es) still need a passing gate record: {}",
            open.len(),
            open.join(", ")
        );
        Ok(1)
    }
}
