// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Turn a finished `--pull-request-gate` run into its gate
//! record: write it, sign it, and render the optional result card.
//!
//! Owner: server CLI (`met benchmark`).
//! Invariants:
//! - A gate record is written only for a run whose status is `Completed` and
//!   whose metrics are non-empty.
//! - The record is stamped with the sha captured before the run.

use super::bench_card::write_card;
use super::bench_run::repo_root;
use anyhow::{Result, bail};
use metrale_bench::TargetEndpoint;
use metrale_bench::gate;
use std::collections::BTreeMap;

/// 2026-09-26: Write this run as a gate record under the repo's
/// `.benchmarks/<id>/`, and sign it.
///
/// The hardware fingerprint is fetched from the endpoint that served the
/// run, not probed locally. Any failure to write or sign is an error, so a
/// run without a record does not report success.
pub(crate) async fn write_gate_record(
    record: &metrale_bench::RunRecord,
    url: &str,
    model: &str,
    recipe: Option<String>,
    // 2026-09-26: What that recipe resolved (`ServePlan::disclosed`); empty
    // for an operator's own endpoint.
    serve_resolved: BTreeMap<String, String>,
    // 2026-09-26: The `METRALE_*` levers the gate applied to its server
    // (`SelfServed::serve_env`); empty for an operator's own endpoint and
    // for a recipe that declares none.
    serve_env: BTreeMap<String, String>,
    sha_at_start: String,
    dirty_at_start: Vec<String>,
    // 2026-09-26: `--output-image` target plus its parsed `--output-image-args`.
    card: Option<(String, BTreeMap<String, String>)>,
) -> Result<()> {
    // 2026-09-26: An incomplete run must not become a gate record, although
    // `headless::run_blocking` returns a RunRecord for a failed or cancelled
    // run too.
    if record.frame.status != metrale_bench::RunStatus::Completed {
        bail!(
            "the run ended as {:?}, not Completed -- no gate record was written. \
             A record is evidence that a benchmark RAN; an interrupted one is not.",
            record.frame.status
        );
    }
    if record.frame.metrics.is_empty() {
        bail!(
            "the run produced no metrics -- no gate record was written. Every \
             threshold would read as \"missing from the record\", which blames the \
             baseline for a run that measured nothing."
        );
    }
    let root = repo_root()?;
    // 2026-09-26: The sha captured before the run, not HEAD now: HEAD may have
    // moved while the benchmark ran.
    let sha = sha_at_start;
    if let Ok(now) = gate::git_sha(&root)
        && now != sha
    {
        // 2026-09-26: Not fatal: the measurement belongs to `sha`; this only
        // warns that the working copy is no longer what was measured.
        eprintln!(
            "gate: HEAD moved during the run ({sha} -> {now}); the record is \
             stamped {sha}, the commit that was actually measured"
        );
    }
    let target = TargetEndpoint::new(url, model);
    let hardware = metrale_bench::http::fetch_hardware(&target, gate::HARDWARE_TIMEOUT).await;
    let dirty = dirty_at_start;
    let gate_record = gate::GateRecord::from_run(record, hardware, sha, dirty, recipe)?
        // 2026-09-26: What this binary's kernels were compiled from, baked at
        // build time rather than read from the tree now.
        .with_closure(metrale_kernels::TARGET_CLOSURES)
        .with_serve_resolved(serve_resolved)
        .with_serve_env(serve_env);
    let path = gate::write_record(&root, &gate_record)?;

    // 2026-09-26: Sign it and print both filenames, so the operator commits the
    // `.sig` beside the `.json`. Signing is kept out of `write_record` so that
    // function's tests mint no keys.
    let store = metrale_bench::artifacts::ArtifactStore::discover()?;
    let identity = gate::signing::load_or_create(store.root())?;
    let sig = gate::signing::sign_record(&identity, &path, &gate_record.git_sha)?;
    let fresh_signer = gate::signing::register(&root, &identity)?;
    eprintln!(
        "gate record written as {}\n                  and {}",
        path.display(),
        sig.display()
    );
    if let Some((target, card_args)) = &card {
        // 2026-09-26: After the record: the card is rendered from the record.
        match write_card(&root, &gate_record, target, card_args) {
            Ok(card) => eprintln!("result card written as {}", card.display()),
            // 2026-09-26: A card failure is reported, not fatal: the record is
            // already on disk.
            Err(e) => {
                eprintln!("gate: the run succeeded but the result card did not render: {e:#}")
            }
        }
    }
    if fresh_signer {
        // 2026-09-26: `register` wrote a new public key into the repo; it must
        // be committed with the record.
        eprintln!(
            "gate: this machine signed a record for the first time. Commit \
             {}/{}.pub alongside the record — it is how the gate learns to trust \
             records from this box.",
            gate::signing::REGISTRY_DIR,
            identity.fingerprint()
        );
    }
    // 2026-09-26: The record carries this verdict in `hardware_state.postcheck`;
    // it is also printed here, where the operator reads the run's numbers.
    if let Some(hw) = &gate_record.hardware_state
        && hw.invalidated()
    {
        eprintln!(
            "gate: ★ that record is marked INVALID — the box throttled while it was \
             measuring, so its SPEED numbers are not comparable and must not be quoted. \
             Concerns: {}",
            hw.concerns().join("; ")
        );
    }
    // 2026-09-26: The start-of-run dirty-tree warning, repeated beside the
    // record it applies to.
    if !gate_record.dirty_paths.is_empty() {
        eprintln!(
            "gate: that record is stamped {} but was measured from a tree with \
             {} uncommitted invalidation-set file(s); it records them, and \
             --pull-request-gate-check will reject it. Re-run from a clean tree.",
            gate_record.git_sha,
            gate_record.dirty_paths.len()
        );
    }
    Ok(())
}
