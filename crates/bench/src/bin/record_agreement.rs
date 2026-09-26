// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `record_agreement <record.json>...`: do the records a PR adds
//! agree? The verdict is [`metrale_bench::gate::agreement::check`].
//!
//! `.github/workflows/ci.yml` passes the added record files. Each record's
//! standing is judged against the head with `agreement::standing_at` (content,
//! not ancestry). The head is `RECORD_AGREEMENT_HEAD` when set, otherwise
//! `gate::git_sha` of the enclosing repository (`METRALE_GATE_SHA` when set,
//! else `HEAD`).
//!
//! Exit 0 when they agree or no record is given. Exit 1, with `::error`
//! annotations, on any disagreement, on a record that cannot be read, parsed
//! as JSON or attributed to a benchmark, or when the repository or the head
//! cannot be found.
//!
//! Owner: bench gate.
//! Invariants: none beyond the types.

use std::path::Path;

use metrale_bench::gate::Standing;
use metrale_bench::gate::agreement::{AddedRecord, check, standing_at};

fn field(v: &serde_json::Value, key: &str) -> Option<String> {
    v.get(key).and_then(|x| x.as_str()).map(str::to_owned)
}

/// 2026-09-26: The record's `benchmark_id`, or, when the field is absent, the
/// name of its directory (`.benchmarks/<id>/<date>-<sha>.json`).
fn benchmark_id_of(path: &Path, v: &serde_json::Value) -> Option<String> {
    field(v, "benchmark_id").or_else(|| {
        path.parent()?
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
    })
}

fn main() -> std::process::ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        println!("this PR adds no records — nothing to agree on.");
        return std::process::ExitCode::SUCCESS;
    }

    let root = match std::env::current_dir()
        .ok()
        .and_then(|d| metrale_bench::gate::git_rev_parse_toplevel(&d).ok())
    {
        Some(r) => r,
        None => {
            println!("::error title=No repository::record_agreement must run inside the checkout");
            return std::process::ExitCode::FAILURE;
        }
    };
    let head = match std::env::var("RECORD_AGREEMENT_HEAD")
        .ok()
        .or_else(|| metrale_bench::gate::git_sha(&root).ok())
    {
        Some(h) => h,
        None => {
            println!("::error title=No head::cannot resolve the head to judge standing against");
            return std::process::ExitCode::FAILURE;
        }
    };
    println!("head {head}");

    let mut added = Vec::new();
    for a in &args {
        let path = Path::new(a);
        let text = match std::fs::read_to_string(path) {
            Ok(t) => t,
            Err(e) => {
                println!("::error title=Unreadable record::{a}: {e}");
                return std::process::ExitCode::FAILURE;
            }
        };
        let v: serde_json::Value = match serde_json::from_str(&text) {
            Ok(v) => v,
            Err(e) => {
                println!("::error title=Malformed record::{a}: {e}");
                return std::process::ExitCode::FAILURE;
            }
        };
        let sig_path = format!("{a}.sig");
        let signer = std::fs::read_to_string(&sig_path)
            .ok()
            .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
            .and_then(|s| field(&s, "key"))
            .unwrap_or_else(|| "NO_SIDECAR".into());
        let Some(benchmark_id) = benchmark_id_of(path, &v) else {
            println!("::error title=Unattributable record::{a} names no benchmark.");
            return std::process::ExitCode::FAILURE;
        };
        let git_sha = field(&v, "git_sha").unwrap_or_else(|| "MISSING".into());
        // 2026-09-26: The hardware capture, when the record parses as a full
        // `GateRecord`; without one the record is equivalent to no other box.
        let parsed = metrale_bench::gate::read_record(path).ok();
        let hardware = parsed
            .as_ref()
            .map(metrale_bench::hardware::equivalence::HardwareFingerprint::from_record);
        let hardware_class = parsed
            .as_ref()
            .map_or_else(|| "unknown".to_string(), |r| r.hardware.gate_key());
        // 2026-09-26: A record that does not parse is `Standing::Unknown`.
        let standing = parsed
            .as_ref()
            .map_or(Standing::Unknown, |r| standing_at(&root, &head, r));
        println!(
            "  {a:<58} gate={benchmark_id} sha={git_sha} signer={signer} standing={standing:?}"
        );
        added.push(AddedRecord {
            path: a.clone(),
            benchmark_id,
            git_sha,
            signer,
            hardware,
            hardware_class,
            standing,
        });
    }

    let problems = check(&root, &added);
    if problems.is_empty() {
        println!(
            "all {} added record(s) agree: each stands at {head}, and signer agreement \
             holds for every speed-class gate.",
            added.len()
        );
        return std::process::ExitCode::SUCCESS;
    }
    for p in &problems {
        println!("::error title=Records do not agree::{p}");
    }
    std::process::ExitCode::FAILURE
}
