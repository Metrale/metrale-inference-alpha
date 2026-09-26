// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for the agentic benchmark's prompt, defaults, verdict and
//! metrics, and the tier fixtures the other agentic test files share.
//!
//! Owner: bench, agentic.
//! Invariants: none beyond the types.

use super::*;

#[test]
fn the_prompt_is_the_harness_prompt() {
    // 2026-09-26: A different prompt is a different benchmark, so the whole
    // shell assignment is compared.
    let harness = include_str!("../../../../../bench/fp8_dgx2_drift/harness/run_tier.sh");
    let assignment = harness
        .lines()
        .find(|line| line.starts_with("PROMPT='"))
        .expect("run_tier.sh must define PROMPT");
    let expected = assignment
        .strip_prefix("PROMPT='")
        .and_then(|prompt| prompt.strip_suffix('\''))
        .expect("PROMPT must remain a single-quoted shell assignment");
    assert_eq!(PROMPT, expected);
}

#[test]
fn it_requires_confirmation_because_it_runs_shell() {
    const { assert!(DESCRIPTOR.needs_confirmation) };
}

#[test]
fn defaults_are_the_gate_a_tier() {
    let b = AgenticWebserver::default();
    let v = ParamValues::defaults(&b.parameters());
    assert_eq!(v.usize("iterations").unwrap(), 10);
    assert_eq!(v.usize("max_turns").unwrap(), 40);
    assert_eq!(v.usize("command_timeout_s").unwrap(), 180);
    assert_eq!(v.usize("build_timeout_s").unwrap(), 600);
    assert_eq!(v.usize("serve_timeout_s").unwrap(), 30);
    assert_eq!(v.usize("max_tokens").unwrap(), 8192);
    assert_eq!(v.usize("request_timeout_s").unwrap(), 900);
    assert_eq!(v.float("wall_budget_s").unwrap(), 1000.0);
    // 2026-09-26: 0.0 means no speed bound; a variant gets one by committing an
    // `s_per_turn` bound in its BENCH.toml.
    assert_eq!(v.float("s_per_turn_budget").unwrap(), 0.0);
}

/// 2026-09-26: A benchmark holding `rows`, with wall budget `budget` and no
/// speed bound.
pub(super) fn with_rows(rows: Vec<IterationRow>, budget: f64) -> AgenticWebserver {
    with_budgets(rows, budget, 0.0)
}

pub(super) fn with_budgets(
    rows: Vec<IterationRow>,
    budget: f64,
    s_per_turn: f64,
) -> AgenticWebserver {
    AgenticWebserver {
        iterations: rows.len(),
        wall_budget_s: budget,
        s_per_turn_budget: s_per_turn,
        rows,
        ..Default::default()
    }
}

/// 2026-09-26: One row carrying a whole tier's totals. The two bounds read only
/// sums (Σwall, and agent Σwall over Σturns), so one row stands for a tier.
///
/// It is not a whole gate fixture: `metrics()["iterations"]` is `rows.len()`,
/// and the agentic BENCH.toml entries that carry metrics pin `iterations` to
/// exactly 10, so `check_record` refuses a one-row tier.
pub(super) fn tier(wall: f64, turns: usize) -> IterationRow {
    IterationRow {
        turns,
        ..row(true, true, wall)
    }
}

/// 2026-09-26: A tier whose agent-only wall differs from its total by the
/// scorer's share.
fn tier_split(total: f64, agent: f64, turns: usize) -> IterationRow {
    IterationRow {
        agent_wall_s: agent,
        ..tier(total, turns)
    }
}

fn row(ok: bool, steps_ok: bool, wall: f64) -> IterationRow {
    IterationRow {
        index: 0,
        wall_s: wall,
        // 2026-09-26: Equal by default; `tier_split` sets it apart.
        agent_wall_s: wall,
        webserver_ok: ok,
        directions: score::Directions {
            steps: score::REQUIRED_STEPS
                .iter()
                .map(|n| (*n, steps_ok))
                .collect(),
        },
        turns: 3,
        tool_calls: 9,
        completion_tokens: 300,
        // 2026-09-26: A clean trajectory; the diagnostics tests set these.
        hit_turn_cap: false,
        truncated_turns: 0,
        unparsed_call_turns: 0,
        note: String::new(),
    }
}

fn committed_agentic_max(model: &str, checkpoint: &str, metric: &str) -> f64 {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(std::path::Path::parent)
        .expect("workspace layout")
        .to_path_buf();
    let (_, committed) = crate::gate::bench::load_all(&root)
        .expect("committed BENCH.toml files must load")
        .into_iter()
        .find(|(target, entry)| {
            target.hardware == "gb10"
                && target.model == model
                && entry.gate == "agentic-webserver"
                && entry.checkpoint == checkpoint
        })
        .unwrap_or_else(|| panic!("the {model}/{checkpoint} agentic gate must be committed"));
    committed
        .metrics
        .expect("a measured agentic gate must declare bounds")[metric]
        .max
        .unwrap_or_else(|| panic!("{metric} must have a maximum"))
}

#[test]
fn all_three_conditions_must_hold_to_pass() {
    let pass = with_rows(vec![row(true, true, 100.0), row(true, true, 100.0)], 1300.0);
    assert_eq!(pass.verdict().kind, crate::result::VerdictKind::Pass);

    let ws = with_rows(vec![row(false, true, 100.0)], 1300.0);
    assert_eq!(ws.verdict().kind, crate::result::VerdictKind::Fail);
    assert!(ws.verdict().reason.contains("webserver_ok 0/1"));

    let fd = with_rows(vec![row(true, false, 100.0)], 1300.0);
    assert_eq!(fd.verdict().kind, crate::result::VerdictKind::Fail);
    assert!(fd.verdict().reason.contains("followed_directions 0/1"));

    let slow = with_rows(vec![row(true, true, 2000.0)], 1300.0);
    assert_eq!(slow.verdict().kind, crate::result::VerdictKind::Fail);
    assert!(slow.verdict().reason.contains("Σwall"));
}

/// 2026-09-26: A dense 27B tier measured 2026-08-14 (N=10, 10/10 on both
/// counts, the walls below, Σ 1925.1 s) fails a 1000 s wall budget on Σwall
/// alone and passes the qwen3.8-27b entry's committed 5000 s.
#[test]
fn the_measured_dense_tier_passes_its_own_budget_and_fails_the_35bs() {
    const DENSE_TIER_WALLS: [f64; 10] = [
        156.0, 187.3, 274.2, 144.7, 230.5, 243.3, 205.2, 117.3, 185.2, 181.4,
    ];
    let rows = || {
        DENSE_TIER_WALLS
            .iter()
            .map(|w| row(true, true, *w))
            .collect::<Vec<IterationRow>>()
    };

    let under_35b_budget = with_rows(rows(), 1000.0);
    let v = under_35b_budget.verdict();
    assert_eq!(v.kind, crate::result::VerdictKind::Fail);
    assert!(
        v.reason.contains("Σwall 1925s > 1000s"),
        "wall is the ONLY failure: {}",
        v.reason
    );
    assert!(
        !v.reason.contains("webserver_ok") && !v.reason.contains("followed_directions"),
        "correctness was perfect: {}",
        v.reason
    );

    let dense_wall_budget =
        committed_agentic_max("qwen3.8-27b", "unsloth/Qwen3.8-27B-NVFP4", "sum_wall_s");
    assert_eq!(
        dense_wall_budget, 5000.0,
        "the documented dense wall bound drifted"
    );
    let under_own_budget = with_rows(rows(), dense_wall_budget);
    assert_eq!(
        under_own_budget.verdict().kind,
        crate::result::VerdictKind::Pass
    );
}

#[test]
fn a_failing_verdict_lists_every_reason_not_just_the_first() {
    let bad = with_rows(vec![row(false, false, 9000.0)], 1300.0);
    let reason = bad.verdict().reason;
    assert!(reason.contains("webserver_ok") && reason.contains("followed_directions"));
    assert!(reason.contains("Σwall"), "{reason}");
}

/// 2026-09-26: A failed iteration names the directions it missed, in
/// declaration order.
#[test]
fn a_failed_iteration_names_the_directives_it_missed() {
    let d = super::score::Directions {
        steps: vec![
            ("built", true),
            ("ran", true),
            ("pinged", false),
            ("tore_down", false),
        ],
    };
    assert_eq!(d.met(), 2);
    assert!(!d.overall());
    assert_eq!(
        d.missing(),
        vec!["pinged", "tore_down"],
        "missing() must name them, in declaration order"
    );

    // 2026-09-26: A passing iteration names nothing.
    let ok = super::score::Directions {
        steps: vec![("built", true), ("ran", true)],
    };
    assert!(ok.missing().is_empty());
    assert!(ok.overall());
}

/// 2026-09-26: Five 35B tiers, 10/10 on both counts, measured 2026-08-17/18
/// (`MEASURED`: Σwall, Σturns; 6.14-7.22 s per turn). The committed 35B bounds
/// (sum_wall_s <= 700, s_per_turn <= 8.5) refuse each on Σwall alone, and each
/// passes under an 1800 s wall budget. A speed bound also orders the tiers by
/// seconds per turn rather than by Σwall.
#[test]
fn old_regime_tiers_are_refused_by_the_700_ceiling_but_still_rank_fairly() {
    let wall_budget =
        committed_agentic_max("qwen3.6-35b-a3b", "Qwen/Qwen3.6-35B-A3B-FP8", "sum_wall_s");
    let speed_budget =
        committed_agentic_max("qwen3.6-35b-a3b", "Qwen/Qwen3.6-35B-A3B-FP8", "s_per_turn");
    assert_eq!(wall_budget, 700.0, "the documented blowup bound drifted");
    assert_eq!(speed_budget, 8.5, "the documented speed bound drifted");

    const MEASURED: [(f64, usize); 5] = [
        (774.0, 113),
        (813.0, 115),
        (860.0, 126),
        (1039.0, 144),
        (1019.0, 166),
    ];
    for (wall, turns) in MEASURED {
        let v = with_budgets(vec![tier(wall, turns)], wall_budget, speed_budget).verdict();
        assert_eq!(
            v.kind,
            crate::result::VerdictKind::Fail,
            "old-regime tier {wall}s/{turns} turns is above the 700 s ceiling and must be \
             refused; if this passes, the ceiling moved: {}",
            v.reason
        );
        assert!(
            v.reason.contains("Σwall"),
            "the refusal must name Σwall: {}",
            v.reason
        );

        // 2026-09-26: Not on speed: under an 1800 s wall budget each passes.
        let generous_wall = with_budgets(vec![tier(wall, turns)], 1800.0, speed_budget).verdict();
        assert_eq!(
            generous_wall.kind,
            crate::result::VerdictKind::Pass,
            "tier {wall}s/{turns} turns is {:.2} s/turn and must clear the 8.5 speed bound \
             once the wall guard is set for ITS regime: {}",
            wall / turns as f64,
            generous_wall.reason
        );
    }

    // 2026-09-26: A 6.5 s/turn bound passes the tier with the larger Σwall
    // (1019 s / 166 turns = 6.14) and fails the one with the smaller (813 s /
    // 115 turns = 7.07).
    let between = 6.5;
    let faster = with_budgets(vec![tier(1019.0, 166)], 1800.0, between).verdict();
    let slower = with_budgets(vec![tier(813.0, 115)], 1800.0, between).verdict();
    assert_eq!(
        faster.kind,
        crate::result::VerdictKind::Pass,
        "6.14 s/turn must clear a {between} s/turn bound however long its Sigma-wall: {}",
        faster.reason
    );
    assert_eq!(
        slower.kind,
        crate::result::VerdictKind::Fail,
        "7.07 s/turn must not clear a {between} s/turn bound however short its Sigma-wall: {}",
        slower.reason
    );
    assert!(slower.reason.contains("s/turn"), "{}", slower.reason);
    // 2026-09-26: A 1000 s wall budget refuses the 1039 s and 1019 s tiers.
    for (wall, turns) in [(1039.0, 144), (1019.0, 166)] {
        let v = with_budgets(vec![tier(wall, turns)], 1000.0, speed_budget).verdict();
        assert_eq!(v.kind, crate::result::VerdictKind::Fail);
        assert!(v.reason.contains("Σwall"), "{}", v.reason);
    }
}

/// 2026-09-26: Why the schema default for `s_per_turn_budget` is 0.0 (no
/// bound): a passing dense tier at 1925 s over 107 turns (18 s/turn) fails the
/// 35B's 8.5, and the qwen3.8-27b entry commits no `s_per_turn` bound.
#[test]
fn the_35b_speed_bound_would_fail_the_healthy_dense_tier_hence_non_gating() {
    let dense = || vec![tier(1925.0, 107)];

    let with_the_35bs_bound = with_budgets(dense(), 5000.0, 8.5).verdict();
    assert_eq!(with_the_35bs_bound.kind, crate::result::VerdictKind::Fail);
    assert!(
        with_the_35bs_bound
            .reason
            .contains("17.991s/turn > 8.500s/turn"),
        "the healthy dense tier fails a bound drawn from another model: {}",
        with_the_35bs_bound.reason
    );

    // 2026-09-26: No committed bound, so 0.0, so speed is not checked.
    let v = with_budgets(dense(), 5000.0, 0.0).verdict();
    assert_eq!(v.kind, crate::result::VerdictKind::Pass, "{}", v.reason);
    assert!(
        v.reason.contains("(unbounded)"),
        "a pass must SAY speed went unchecked rather than imply it passed: {}",
        v.reason
    );
}

/// 2026-09-26: The speed bound divides the agent's seconds, not the tier's
/// total: the scorer's build is a per-iteration cost, and charging it to a
/// per-turn ratio makes a long trajectory look faster per turn.
#[test]
fn the_speed_bound_excludes_the_scorers_build_from_the_numerator() {
    let b = with_budgets(vec![tier_split(874.0, 774.0, 113)], 1800.0, 7.0);
    let m = b.metrics();
    assert_eq!(m["sum_wall_s"], 874.0);
    assert_eq!(m["sum_agent_wall_s"], 774.0);
    assert!((m["s_per_turn"] - 774.0 / 113.0).abs() < 1e-9);

    // 2026-09-26: 774 / 113 = 6.85 passes a 7.0 bound; 874 / 113 = 7.73 fails
    // it.
    assert_eq!(b.verdict().kind, crate::result::VerdictKind::Pass);
    let charged_the_scorer = with_budgets(vec![tier(874.0, 113)], 1800.0, 7.0);
    assert_eq!(
        charged_the_scorer.verdict().kind,
        crate::result::VerdictKind::Fail
    );
}

/// 2026-09-26: 1247 s over 144 turns is 8.66 s/turn, 20% above the slowest
/// measured tier's 7.22: it fails the 8.5 speed bound while Σwall stays under
/// 1800 s.
#[test]
fn a_real_per_turn_regression_fails_while_wall_stays_in_budget() {
    let regressed = with_budgets(vec![tier(1247.0, 144)], 1800.0, 8.5);
    let v = regressed.verdict();
    assert_eq!(v.kind, crate::result::VerdictKind::Fail);
    assert!(
        v.reason.contains("8.660s/turn > 8.500s/turn"),
        "speed must be the named failure: {}",
        v.reason
    );
    assert!(
        !v.reason.contains("Σwall"),
        "wall must NOT fire — that is the gap being closed: {}",
        v.reason
    );
}

/// 2026-09-26: Σwall still fails a tier that is fast per turn but takes too
/// many turns.
#[test]
fn wall_still_catches_turn_degeneracy_that_is_fast_per_turn() {
    // 2026-09-26: 220 turns at 8.30 s/turn = 1826 s, below the 400-turn cap of
    // ten iterations at the default `max_turns` of 40.
    let wandering = with_budgets(vec![tier(1826.0, 220)], 1800.0, 8.5);
    let v = wandering.verdict();
    assert_eq!(v.kind, crate::result::VerdictKind::Fail);
    assert!(v.reason.contains("Σwall 1826s > 1800s"), "{}", v.reason);
    assert!(
        !v.reason.contains("s/turn >"),
        "8.30 s/turn is inside the speed bound; only the wall is wrong: {}",
        v.reason
    );
}

/// 2026-09-26: A zero-turn tier has no `s_per_turn` metric and no speed
/// failure.
#[test]
fn a_zero_turn_tier_reports_no_speed_at_all() {
    let empty = with_budgets(vec![tier(120.0, 0)], 1300.0, 8.5);
    assert!(!empty.metrics().contains_key("s_per_turn"));
    assert_eq!(empty.metrics()["sum_turns"], 0.0);
    // 2026-09-26: Correctness is true here; its failures are covered by
    // `all_three_conditions_must_hold_to_pass`.
    assert!(!empty.verdict().reason.contains("s/turn >"));
}

/// 2026-09-26: The record carries `sum_turns` beside `sum_wall_s` and
/// `s_per_turn`.
#[test]
fn the_record_carries_turns_so_a_wall_anomaly_is_diagnosable_after_the_fact() {
    let m = with_budgets(vec![tier(500.0, 60), tier(274.0, 53)], 1300.0, 8.5).metrics();
    assert_eq!(m["sum_wall_s"], 774.0);
    assert_eq!(m["sum_turns"], 113.0);
    assert!((m["s_per_turn"] - 774.0 / 113.0).abs() < 1e-9);
}
