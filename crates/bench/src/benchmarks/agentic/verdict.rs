// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The PASS/FAIL decision for an `agentic-webserver` tier, and the
//! turn aggregates it divides. `score.rs` measures an iteration; this judges
//! the tier.
//!
//! Owner: bench, agentic.
//! Invariants:
//! - A tier passes only if every iteration has `webserver_ok` and
//!   `followed_directions`.
//! - The s/turn bound divides the agent's own seconds, never the tier total:
//!   the scorer's build costs the same per iteration however many turns ran.
//!   It gates only when its budget is above 0.0.
//! - The Σwall bound is on the whole tier wall, scorer included.

/// 2026-09-26: `wall` is the total tier wall (scorer included) that
/// `wall_budget_s` bounds; `agent_wall` is the agent's own, which the per-turn
/// speed bound divides.
pub(super) fn verdict(
    rows: &[super::IterationRow],
    wall: f64,
    agent_wall: f64,
    wall_budget_s: f64,
    s_per_turn_budget: f64,
) -> crate::result::Verdict {
    use crate::result::Verdict;
    let n = rows.len();
    let ok = rows.iter().filter(|r| r.webserver_ok).count();
    let fd = rows.iter().filter(|r| r.directions.overall()).count();
    let turns = total_turns(rows);
    let mut failures = Vec::new();
    if ok < n {
        failures.push(format!("webserver_ok {ok}/{n}"));
    }
    if fd < n {
        failures.push(format!("followed_directions {fd}/{n}"));
    }
    // 2026-09-26: A budget of 0.0 is non-gating. Three decimals so a marginal
    // failure does not print two identical-looking numbers either side of `>`.
    let over_speed = seconds_per_turn(agent_wall, turns)
        .filter(|_| s_per_turn_budget > 0.0)
        .filter(|s| *s > s_per_turn_budget);
    if let Some(spt) = over_speed {
        failures.push(format!(
            "{spt:.3}s/turn > {s_per_turn_budget:.3}s/turn \
             ({agent_wall:.0}s agent / {turns} turns)"
        ));
    }
    // 2026-09-26: A blowup bound, not the speed bound: it catches a tier that
    // completes every task through far more turns than the work needs.
    if wall > wall_budget_s {
        failures.push(format!("Σwall {wall:.0}s > {wall_budget_s:.0}s"));
    }
    if failures.is_empty() {
        let spt = match (seconds_per_turn(agent_wall, turns), s_per_turn_budget > 0.0) {
            (Some(s), true) => format!("{s:.3}s/turn ≤ {s_per_turn_budget:.3}"),
            (Some(s), false) => format!("{s:.3}s/turn (unbounded)"),
            (None, _) => "no turns".to_string(),
        };
        Verdict::pass(format!(
            "{ok}/{n} webserver_ok · {fd}/{n} followed_directions · \
             {spt} · Σwall {wall:.0}s ≤ {wall_budget_s:.0}s"
        ))
    } else {
        Verdict::fail(failures.join(" · "))
    }
}

/// 2026-09-26: Agent turns summed across the tier, the speed bound's
/// denominator.
pub(super) fn total_turns(rows: &[super::IterationRow]) -> usize {
    rows.iter().map(|r| r.turns).sum()
}

/// 2026-09-26: Seconds of wall per agent turn, or `None` when the tier took no
/// turns. Not 0.0, which would pass the speed bound, and not infinity, which
/// would fail it a second time: an agent that never ran leaves no `Cargo.toml`,
/// so `webserver_ok` already fails.
pub(super) fn seconds_per_turn(wall: f64, turns: usize) -> Option<f64> {
    (turns > 0).then(|| wall / turns as f64)
}
