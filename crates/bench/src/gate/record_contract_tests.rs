// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for building gate records: file names, `from_run`'s
//! refusals and replay command, and the environment it discloses.
//!
//! Owner: bench gate (records).
//! Invariants: none beyond the types.

use super::record::resolve_perf_env;
use super::tests::{MODEL, SHA, frame, hw, run_record, tempdir};
use super::*;
use crate::history::RunRecord;
use crate::result::{RunStatus, Verdict};
use std::collections::BTreeMap;

#[test]
fn date_of_matches_the_utc_civil_calendar() {
    assert_eq!(
        [
            0,
            1_709_164_799,
            1_709_164_800,
            1_709_251_199,
            1_709_251_200,
            1_735_689_599,
            1_735_689_600,
        ]
        .map(date_of),
        [
            "1970-01-01",
            "2024-02-28",
            "2024-02-29",
            "2024-02-29",
            "2024-03-01",
            "2024-12-31",
            "2025-01-01",
        ]
    );
}

#[test]
fn the_record_path_is_date_and_sha_and_replaces_a_same_day_rerun() {
    let dir = tempdir::Dir::new();
    let p1 = record_path(dir.path(), "bfcl-subset", 1_785_891_382, SHA);
    assert_eq!(
        p1,
        dir.path()
            .join(".benchmarks/bfcl-subset/2026-08-05-b72dad1893.json")
    );
    let p2 = record_path(dir.path(), "bfcl-subset", 1_785_891_382 + 3_600, SHA);
    assert_eq!(p1, p2, "same sha + UTC day = same file");
    assert_eq!(
        record_path(dir.path(), "bfcl-subset", 1_785_974_400, SHA),
        dir.path()
            .join(".benchmarks/bfcl-subset/2026-08-06-b72dad1893.json")
    );
}

#[test]
fn from_run_rejects_a_missing_sha_and_a_non_terminal_frame() {
    let record = run_record(BTreeMap::new(), Verdict::pass("ok"));
    for missing in ["", " \t\n"] {
        assert_eq!(
            GateRecord::from_run(&record, hw(), missing.into(), Vec::new(), None,)
                .unwrap_err()
                .to_string(),
            "a gate record needs the commit sha it was measured from"
        );
    }

    let mut running = record.clone();
    running.frame.status = RunStatus::Running;
    assert_eq!(
        GateRecord::from_run(&running, hw(), SHA.into(), Vec::new(), None,)
            .unwrap_err()
            .to_string(),
        "the run never reached a terminal frame — nothing to gate"
    );
}

#[test]
fn from_run_reconstructs_the_exact_cli_command() {
    let mut metrics = BTreeMap::new();
    metrics.insert("overall_accuracy".to_string(), 87.74);
    let gate = GateRecord::from_run(
        &run_record(metrics, Verdict::pass("ok")),
        hw(),
        SHA.into(),
        Vec::new(),
        None,
    )
    .unwrap();
    assert_eq!(
        gate.command,
        [
            "met",
            "benchmark",
            "run",
            "bfcl-subset",
            "--url",
            "http://127.0.0.1:8888",
            "--model",
            MODEL,
            "--param",
            "repeats=12",
            "--pull-request-gate",
        ]
    );
    assert_eq!(gate.verdict.as_deref(), Some("PASS"));
    assert_eq!(gate.frame_status, RunStatus::Completed);
}

#[test]
fn a_self_provisioned_run_records_the_recipe_not_a_dead_url() {
    let mut metrics = BTreeMap::new();
    metrics.insert("overall_accuracy".to_string(), 87.74);
    let gate = GateRecord::from_run(
        &run_record(metrics, Verdict::pass("ok")),
        hw(),
        SHA.into(),
        Vec::new(),
        Some("qwen3.6/qwen3.6-27b-nvfp4-unsloth".to_string()),
    )
    .unwrap();
    assert_eq!(
        gate.command,
        [
            "met",
            "benchmark",
            "run",
            "bfcl-subset",
            "--param",
            "repeats=12",
            "--pull-request-gate",
        ]
    );
    assert_eq!(
        gate.served_by.as_deref(),
        Some("qwen3.6/qwen3.6-27b-nvfp4-unsloth")
    );
    assert_eq!(gate.target_model, MODEL);
}

#[test]
fn the_agentic_bench_needs_yes_in_its_command() {
    let mut record = run_record(BTreeMap::new(), Verdict::pass("ok"));
    record.benchmark_id = "agentic-webserver".to_string();
    let gate = GateRecord::from_run(&record, hw(), SHA.into(), Vec::new(), None).unwrap();
    assert_eq!(
        gate.command,
        [
            "met",
            "benchmark",
            "run",
            "agentic-webserver",
            "--url",
            "http://127.0.0.1:8888",
            "--model",
            MODEL,
            "--param",
            "repeats=12",
            "--yes",
            "--pull-request-gate",
        ]
    );
}

#[test]
fn a_failed_frame_is_recorded_but_never_passes() {
    let record = RunRecord {
        frame: frame(
            RunStatus::Failed,
            BTreeMap::new(),
            Verdict::fail("scoring crashed"),
        ),
        ..run_record(BTreeMap::new(), Verdict::fail("scoring crashed"))
    };
    let gate = GateRecord::from_run(&record, hw(), SHA.into(), Vec::new(), None).unwrap();
    assert_eq!(gate.frame_status, RunStatus::Failed);
    assert_eq!(gate.verdict.as_deref(), Some("FAIL"));
    assert_eq!(gate.verdict_reason, "scoring crashed");
    assert!(gate.frame_status_failed());
    assert!(!gate.verdict_passes());
}

#[test]
fn an_unset_control_is_recorded_as_the_default_the_scheduler_would_apply() {
    // 2026-09-26: Unset and set-to-default are the same run, so both record
    // the default.
    let resolved = resolve_perf_env(|_| None);
    assert_eq!(
        resolved
            .get("METRALE_PREFILL_CODISPATCH_WINDOW_MS")
            .map(String::as_str),
        Some("100")
    );
    assert_eq!(
        resolved
            .get("METRALE_PREFILL_CODISPATCH_SETTLE_MS")
            .map(String::as_str),
        Some("10")
    );
    assert_eq!(
        resolved.get("METRALE_NO_W4A16_TC").map(String::as_str),
        Some("unset")
    );
    assert_eq!(
        resolved.get("METRALE_NO_MTP_TC").map(String::as_str),
        Some("unset")
    );
    assert_eq!(
        resolved
            .get("METRALE_NO_BORROW_STREAK_LIMIT")
            .map(String::as_str),
        Some("unset")
    );
}

#[test]
fn a_set_control_wins_and_an_empty_one_does_not() {
    // 2026-09-26: An empty or whitespace-only value resolves to the default,
    // not to itself.
    let resolved = resolve_perf_env(|k| match k {
        "METRALE_PREFILL_CODISPATCH_SETTLE_MS" => Some("25".into()),
        "METRALE_PREFILL_CODISPATCH_WINDOW_MS" => Some("   ".into()),
        _ => None,
    });
    assert_eq!(
        resolved
            .get("METRALE_PREFILL_CODISPATCH_SETTLE_MS")
            .map(String::as_str),
        Some("25")
    );
    assert_eq!(
        resolved
            .get("METRALE_PREFILL_CODISPATCH_WINDOW_MS")
            .map(String::as_str),
        Some("100"),
        "an empty value must resolve to the default, not to the empty string"
    );
}

/// 2026-09-26: The co-dispatch defaults in `PERF_CONTROLS` are copied from
/// `crates/server/src/scheduler/levers.rs`, because `metrale-bench` does not
/// depend on `metrale-server`. This reads that source and fails when a
/// default moves there.
#[test]
fn perf_env_defaults_match_the_scheduler() {
    // 2026-09-26: `SchedLevers::from_env` reads both as `num(VAR, default)`.
    let path = repo_root().join("crates/server/src/scheduler/levers.rs");
    let src = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    // 2026-09-26: Anchored on the `num("VAR",` call, not the first mention,
    // because doc comments name each control before it is read.
    let resolution = |var: &str| -> String {
        let at = src
            .find(&format!("num(\"{var}\","))
            .unwrap_or_else(|| panic!("{var} is not resolved in levers.rs"));
        src[at..].chars().take(80).collect()
    };
    assert!(
        resolution("METRALE_PREFILL_CODISPATCH_WINDOW_MS").contains(", 100)"),
        "the scheduler's co-dispatch WINDOW default moved; PERF_CONTROLS in record_env.rs still \
         says 100 and every record would disclose a value the server never used"
    );
    assert!(
        resolution("METRALE_PREFILL_CODISPATCH_SETTLE_MS").contains(", 10)"),
        "the scheduler's co-dispatch SETTLE default moved; PERF_CONTROLS in record_env.rs still \
         says 10"
    );
    // 2026-09-26: The co-dispatch enable is `prefill_codispatch_enabled()` in
    // model-layers: `--prefill-codispatch` when given, else
    // `METRALE_PREFILL_CODISPATCH` through `bool_value_enabled`. The record
    // discloses the flag in `serve_resolved` and omits it when absent, which
    // reads as off only while an unset variable means off.
    let ssot = repo_root().join("crates/model-layers/src/layers/ops/dispatch_helpers.rs");
    let ssot_src =
        std::fs::read_to_string(&ssot).unwrap_or_else(|e| panic!("{}: {e}", ssot.display()));
    let at = ssot_src
        .find("pub fn prefill_codispatch_enabled()")
        .expect("the codispatch SSOT is gone; PERF_CONTROLS has nothing to agree with");
    let body: String = ssot_src[at..].chars().take(260).collect();
    assert!(
        body.contains("std::env::var(\"METRALE_PREFILL_CODISPATCH\")"),
        "the env FALLBACK is gone, so an operator setting the documented variable \
         gets nothing while the record still discloses it: {body}"
    );
    assert!(
        body.contains("bool_value_enabled"),
        "the truthiness rule changed; an absent disclosure assumes unset means off: {body}"
    );
    // 2026-09-26: The enable is not in `perf_env`: a flag never reaches the
    // environment, so an environment default there could contradict it.
    assert!(
        !crate::gate::record::resolve_perf_env(|_| None).contains_key("METRALE_PREFILL_CODISPATCH"),
        "perf_env re-discloses the codispatch enable from the environment; the flag is \
         disclosed in serve_resolved (record_serve::PREFILL_CODISPATCH)"
    );
    // 2026-09-26: `levers.rs` must not read the variable directly, bypassing
    // the flag.
    assert!(
        !src.contains("std::env::var(\"METRALE_PREFILL_CODISPATCH\")"),
        "mod_helpers.rs reads the codispatch variable directly again, bypassing the \
         flag: a --prefill-codispatch that the scheduler ignores is worse than no flag"
    );
}

fn repo_root() -> std::path::PathBuf {
    let mut d = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    while !d.join(".git").exists() {
        assert!(d.pop(), "no repo root above CARGO_MANIFEST_DIR");
    }
    d
}

/// 2026-09-26: `from_run` copies the run record's serve overrides onto the
/// gate record and into its replay command.
#[test]
fn the_gate_records_regime_is_the_runs_regime() {
    let mut record = run_record(BTreeMap::new(), Verdict::pass("ok"));
    record.serve_overrides = [
        ("hermetic".to_string(), "true".to_string()),
        ("ssm_cache_slots".to_string(), "0".to_string()),
    ]
    .into_iter()
    .collect();

    let gate = GateRecord::from_run(&record, hw(), SHA.to_string(), Vec::new(), None).unwrap();

    assert_eq!(
        gate.serve_overrides, record.serve_overrides,
        "the gate record must carry the regime the run was measured under"
    );
    let cmd = gate.command.join(" ");
    assert!(
        cmd.contains("--serve-override hermetic=true"),
        "the replay command must reproduce the regime: {cmd}"
    );
    assert!(cmd.contains("--serve-override ssm_cache_slots=0"), "{cmd}");
}

/// 2026-09-26: A run without serve overrides gives an empty map and no
/// `--serve-override` in the command.
#[test]
fn a_run_with_no_recorded_regime_claims_none() {
    let record = run_record(BTreeMap::new(), Verdict::pass("ok"));
    let gate = GateRecord::from_run(&record, hw(), SHA.to_string(), Vec::new(), None).unwrap();
    assert!(gate.serve_overrides.is_empty());
    assert!(!gate.command.join(" ").contains("--serve-override"));
}

/// 2026-09-26: `from_run` copies the frame's `hardware_state`, machine
/// identity included, and a frame without one gives a record without one.
#[test]
fn the_record_carries_the_box_identity_the_frame_captured() {
    use crate::hardware::policy::Sensitivity;
    use crate::hardware::state::MachineIdentity;
    use crate::hardware::{HardwareState, HardwareStateReport};
    let mut with_state = run_record(BTreeMap::new(), Verdict::pass("ok"));
    let before = HardwareState {
        captured_at: 1_000,
        machine: MachineIdentity {
            hostname: Some("spark-28c2".into()),
            machine_id: Some("7af66f30966a49b6886e00e2fce4b42f".into()),
            gpu: Some("NVIDIA GB10".into()),
            driver: Some("580.159.03".into()),
        },
        ..HardwareState::default()
    };
    with_state.frame.hardware_state = Some(HardwareStateReport::opened(
        Sensitivity::Correctness,
        before,
        None,
    ));
    let record = GateRecord::from_run(&with_state, hw(), SHA.into(), Vec::new(), None).unwrap();
    let machine = &record
        .hardware_state
        .as_ref()
        .expect("the captured state travels with the record")
        .before
        .machine;
    assert_eq!(
        machine.machine_id.as_deref(),
        Some("7af66f30966a49b6886e00e2fce4b42f")
    );
    assert_eq!(machine.hostname.as_deref(), Some("spark-28c2"));
    let json = serde_json::to_value(&record).unwrap();
    assert_eq!(
        json["hardware_state"]["before"]["machine"]["machine_id"],
        "7af66f30966a49b6886e00e2fce4b42f"
    );

    let without = GateRecord::from_run(
        &run_record(BTreeMap::new(), Verdict::pass("ok")),
        hw(),
        SHA.into(),
        Vec::new(),
        None,
    )
    .unwrap();
    assert!(without.hardware_state.is_none());
    assert!(
        serde_json::to_value(&without)
            .unwrap()
            .get("hardware_state")
            .is_none()
    );
}
