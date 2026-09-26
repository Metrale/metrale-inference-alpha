// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `met` refuses to start with an undeclared `METRALE_*`
//! variable set. It runs the real binary, so it checks that the process start
//! calls `metrale_config::levers::check` (`main.rs`), not only that the check
//! works.
//!
//! Owner: server tests.
//! Invariants: none beyond the types.

use std::process::Command;

fn met(extra_env: Option<(&str, &str)>) -> std::process::Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_met"));
    cmd.arg("dump-serve-options");
    if let Some((k, v)) = extra_env {
        cmd.env(k, v);
    }
    cmd.output().expect("met runs")
}

#[test]
fn an_undeclared_metrale_variable_stops_the_process_by_name() {
    // 2026-09-26: Control: the same command without the variable succeeds.
    let ok = met(None);
    assert!(
        ok.status.success(),
        "control failed: {}",
        String::from_utf8_lossy(&ok.stderr)
    );

    let refused = met(Some(("METRALE_FOO", "1")));
    assert!(!refused.status.success(), "METRALE_FOO=1 was accepted");
    let err = String::from_utf8_lossy(&refused.stderr);
    assert!(err.contains("METRALE_FOO"), "names the variable: {err}");
    assert!(err.contains("undeclared"), "says why: {err}");
    assert!(refused.stdout.is_empty(), "refused before any output");

    // 2026-09-26: A declared lever is not refused.
    let declared = met(Some(("METRALE_MTP_K_LADDER", "1:3")));
    assert!(declared.status.success());
}
