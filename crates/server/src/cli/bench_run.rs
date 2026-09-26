// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Dispatch for `met benchmark`, and its `run` subcommand.
//!
//! `run` drives the endpoint named by `--url`/`--model`. With
//! `--pull-request-gate` it instead serves the benchmark's recipe itself
//! (`bench_selfstart`) or reuses the leased server (`bench_lease`), then writes
//! the gate record. The run loop is `metrale_bench::headless::run_blocking`.
//!
//! Owner: server CLI (`met benchmark`).
//! Invariants:
//! - A server this run starts in-process (`bench_selfstart`) is shut down after
//!   the gate record is written, and also when the run errors; `SelfServed`'s
//!   `Drop` aborts it on the paths that reach neither. A leased server
//!   (`--serve-reuse`) is left running.

use anyhow::{Context, Result, bail};
use metrale_bench::headless::{HeadlessOptions, RunRequest, SilentReporter, run_blocking};
use metrale_bench::{
    ArtifactStore, BenchmarkDescriptor, BenchmarkExecutor, ParamValues, TargetEndpoint, gate,
    history, registry,
};

use super::bench_args::{BenchmarkArgs, BenchmarkCommand, HistoryArgs, OutputFormat, RunArgs};
use super::bench_print;

/// 2026-09-26: Look up a benchmark, naming the alternatives when it is not one.
pub fn find(id: &str) -> Result<&'static BenchmarkDescriptor> {
    registry::find(id).ok_or_else(|| {
        let known: Vec<&str> = registry::all().iter().map(|d| d.id).collect();
        anyhow::anyhow!(
            "unknown benchmark {id:?} — the suite is: {}",
            known.join(", ")
        )
    })
}

pub async fn dispatch(args: BenchmarkArgs) -> Result<()> {
    if let Err(msg) = args.reject_orphan_pr() {
        bail!("{msg}");
    }
    if args.pull_request_gate_check {
        let code = super::bench_gate_check::gate_check_cmd(args.pr)?;
        if code != 0 {
            std::process::exit(code);
        }
        return Ok(());
    }
    let command = args.command.expect("clap enforces a subcommand here");
    match command {
        BenchmarkCommand::List(a) => match a.id {
            Some(id) => bench_print::print_schema(&id, a.format),
            None => bench_print::print_suite(a.format),
        },
        BenchmarkCommand::History(a) => history_cmd(a),
        BenchmarkCommand::ServeRelease => {
            let code = super::bench_lease::release_cmd()?;
            std::process::exit(code);
        }
        BenchmarkCommand::Card(a) => super::bench_card::card_cmd(a),
        BenchmarkCommand::Certify(a) => {
            let code = super::bench_certify::certify_cmd(a).await?;
            if code != 0 {
                std::process::exit(code);
            }
            Ok(())
        }
        BenchmarkCommand::Aggregate(a) => {
            let code = super::bench_aggregate::aggregate_cmd(a)?;
            std::process::exit(code);
        }
        BenchmarkCommand::Run(a) => {
            let code = run(a).await?;
            // 2026-09-26: `run` reports its own outcome; this passes its exit
            // code to the shell. Both exits here, the `exit` below and the `?`
            // above, skip whatever follows (`exit` skips destructors too), so
            // server teardown lives inside `run` and `SelfServed::drop`.
            if code != 0 {
                std::process::exit(code);
            }
            Ok(())
        }
    }
}

/// 2026-09-26: `git rev-parse --show-toplevel`: the checkout that holds
/// `.benchmarks/`.
pub(crate) fn repo_root() -> Result<std::path::PathBuf> {
    let out = std::process::Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .stdin(std::process::Stdio::null())
        .output()
        .context("running git rev-parse")?;
    if !out.status.success() {
        bail!("not inside a git checkout — the gate records live in the repo's .benchmarks/");
    }
    let path = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if path.is_empty() {
        bail!("git rev-parse --show-toplevel printed nothing");
    }
    Ok(std::path::PathBuf::from(path))
}

/// 2026-09-26: The commit and the uncommitted perf-path files, read before
/// the run so the warning comes before the hours are spent. A failure to read
/// the dirty list is an error, before any model loads: an empty list on the
/// record must mean a clean tree, never that git could not answer.
fn capture_provenance() -> Result<(String, Vec<String>)> {
    let root = repo_root()?;
    capture_provenance_at(&root)
}

fn capture_provenance_at(root: &std::path::Path) -> Result<(String, Vec<String>)> {
    let sha = gate::git_sha(root)?;
    let dirty = gate::dirty_perf_paths(root)
        .context("reading the working tree state before the gate run")?;
    if !dirty.is_empty() {
        eprintln!(
            "gate: WARNING — {} uncommitted file(s) that change what a gate \
             measures are in this tree, so the record will be stamped {sha} \
             but the binary is not {sha}:",
            dirty.len()
        );
        for path in &dirty {
            eprintln!("gate:   {path}");
        }
        eprintln!(
            "gate: the record will disclose this and the gate check will \
             reject it. Commit (or stash) and rebuild first."
        );
    }
    warn_if_signer_is_not_committed(root);
    Ok((sha, dirty))
}

/// 2026-09-26: Before the run, print `signing::signer_notice` for this box's
/// signing identity (one per metrale home) against the committed
/// `.github/record-signers/`. `bench_record` registers the key only after the
/// run. Never fails the run: a first record from a new box is legitimate.
fn warn_if_signer_is_not_committed(root: &std::path::Path) {
    let Ok(store) = ArtifactStore::discover() else {
        return;
    };
    let Ok(identity) = gate::signing::load_or_create(store.root()) else {
        return;
    };
    let fp = identity.fingerprint();
    match gate::signing::committed_signers(root) {
        Ok(committed) => {
            if let Some(msg) = gate::signing::signer_notice(&committed, fp) {
                eprintln!("{msg}");
            }
        }
        // 2026-09-26: Cannot answer: say so rather than imply the signer is fine.
        Err(e) => eprintln!("gate: NOTE — could not read .github/record-signers/: {e:#}"),
    }
}

#[cfg(test)]
#[path = "bench_provenance_tests.rs"]
mod provenance_tests;

/// 2026-09-26: The box class's temperature ceilings for the hardware
/// pre-check, from `kernels/<hw>/HARDWARE.toml` `[benchmarks.limits.thermal]`,
/// for the class named by `--hardware`, else the probed one. `None` (no repo
/// root, no limits, or an unreadable file) makes the pre-check record the
/// temperatures without judging them.
fn temp_ceilings(hardware: Option<&str>) -> Option<metrale_bench::hardware::policy::TempCeilings> {
    let root = repo_root().ok()?;
    let class = match hardware {
        Some(h) => h.to_string(),
        None => metrale_bench::hardware::Hardware::probe().gate_key(),
    };
    metrale_bench::hardware::limits::limits(&root, &class)
        .ok()
        .flatten()
        .map(|l| metrale_bench::hardware::policy::TempCeilings::of(&l.thermal))
}

fn store() -> Result<ArtifactStore> {
    ArtifactStore::discover()
}

fn history_cmd(args: HistoryArgs) -> Result<()> {
    let store = store()?;
    if let Some(run_id) = &args.run {
        let Some(record) = history::find(&store, run_id) else {
            bail!("no run {run_id:?} under {}", store.root().display());
        };
        return bench_print::print_record(&record, args.format);
    }
    let mut records = match &args.id {
        Some(id) => {
            find(id)?; // 2026-09-26: a typo is an error, not an empty history
            history::load(&store, id)
        }
        None => history::load_all(&store),
    };
    records.truncate(args.limit);
    bench_print::print_history(&records, args.format)
}

async fn run(args: RunArgs) -> Result<i32> {
    if let Err(msg) = args.reject_orphan_checkpoint() {
        bail!("{msg}");
    }
    if let Err(msg) = args.reject_orphan_image_args() {
        bail!("{msg}");
    }
    let descriptor = find(&args.id)?;
    if descriptor.needs_confirmation && !args.yes {
        bail!(
            "{} has side effects beyond load on the endpoint — it executes \
             model-authored shell in a sandbox. Pass --yes to accept that.",
            descriptor.id
        );
    }

    let specs = descriptor.build().parameters();
    let pairs = args
        .params
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect::<Vec<_>>();
    let mut values = ParamValues::from_overrides(&specs, pairs)?;

    // 2026-09-26: Under --pull-request-gate, capture the commit and dirty
    // perf paths before the run, so the record names the commit that was
    // measured even if HEAD moves during it.
    let provenance = if args.pull_request_gate {
        Some(capture_provenance()?)
    } else {
        None
    };
    // 2026-09-26: Discovered before any server exists, so its failure needs no
    // teardown.
    let store = store()?;
    let served = if args.pull_request_gate && args.serve_reuse {
        let plan = super::bench_serve_plan::plan_serve(
            &args.id,
            args.hardware.as_deref(),
            args.checkpoint.as_deref(),
            super::bench_resolve::parse_serve_overrides(&args.serve_override)?,
        )?;
        Some(super::bench_lease::acquire(plan, args.serve_lease_owner).await?)
    } else if args.pull_request_gate {
        Some(
            super::bench_selfstart::serve_for(
                &args.id,
                args.hardware.as_deref(),
                args.checkpoint.as_deref(),
                super::bench_resolve::parse_serve_overrides(&args.serve_override)?,
            )
            .await?,
        )
    } else {
        None
    };
    let target = match &served {
        Some(s) => s.target.clone(),
        None => TargetEndpoint::new(&args.url, args.model.as_deref().unwrap_or_default()),
    };
    // 2026-09-26: The served variant's baseline entry sets its
    // `[benchmarks.param_overrides]` pins, then the threshold-coupled params;
    // an explicit --param wins over both (see `bench_resolve`).
    if let Some(s) = &served {
        for (param, value) in super::bench_resolve::apply_param_overrides(
            descriptor,
            &specs,
            &mut values,
            &s.baseline_entry,
            &args.params,
        )? {
            eprintln!(
                "gate: {param} = {value} pinned by the {} variant's baseline \
                 [benchmarks.param_overrides] (not the schema default)",
                target.model
            );
        }
        for (param, bound) in super::bench_resolve::apply_threshold_params(
            descriptor,
            &specs,
            &mut values,
            &s.baseline_entry,
            &args.params,
        )? {
            eprintln!(
                "gate: {param} = {bound} from the {} variant's baseline (not the schema default)",
                target.model
            );
        }
    }

    let executor = BenchmarkExecutor::new(tokio::runtime::Handle::current(), store);
    // 2026-09-26: The merged baseline pin and `--serve-override` set goes onto
    // the run's target, and so onto the RunRecord the gate record is built from.
    let serve_overrides = served
        .as_ref()
        .map(|s| s.overrides.clone())
        .unwrap_or_default();
    let request = RunRequest {
        descriptor,
        values,
        target: target.clone().with_serve_overrides(serve_overrides),
        options: HeadlessOptions {
            poll: std::time::Duration::from_millis(args.poll_ms),
            save: !args.no_save,
            source: metrale_bench::RunSource::Cli,
            metrale_version: super::METRALE_VERSION.to_string(),
            coherence: if args.skip_coherence_probe {
                metrale_bench::CoherencePolicy::Skip
            } else {
                metrale_bench::CoherencePolicy::Probe
            },
            temp_ceilings: temp_ceilings(args.hardware.as_deref()),
        },
    };

    // 2026-09-26: Ctrl-C goes through the server's shutdown latch. The startup
    // escape (exit on Ctrl-C during a model load) is disarmed first: this is
    // not a model load.
    crate::tui::shutdown::disarm_startup_escape();
    crate::tui::shutdown::install_signal_listeners();

    let quiet = args.quiet;
    let format = args.format;
    // 2026-09-26: `run_blocking` sleeps its thread, so it must not hold a runtime worker.
    let outcome = tokio::task::spawn_blocking(move || {
        let mut reporter = bench_print::StdoutReporter::new(quiet);
        let mut silent = SilentReporter;
        let reporter: &mut dyn metrale_bench::headless::RunReporter =
            if format == OutputFormat::Json {
                &mut silent // 2026-09-26: no progress beside JSON
            } else {
                &mut reporter
            };
        run_blocking(
            &executor,
            request,
            reporter,
            &crate::tui::shutdown::requested,
        )
    })
    .await;

    // 2026-09-26: A failed run shuts the server down before returning the
    // error. On success the gate record is written first and the server shut
    // down second (below).
    let outcome = match outcome {
        Ok(Ok(o)) => o,
        Ok(Err(e)) => {
            if let Some(s) = served {
                s.shutdown().await;
            }
            return Err(e);
        }
        Err(join) => {
            if let Some(s) = served {
                s.shutdown().await;
            }
            return Err(join.into());
        }
    };

    if args.pull_request_gate {
        // 2026-09-26: Write first, shut down second, and shut down even when
        // the write fails: `write_gate_record` fetches the hardware fingerprint
        // from the endpoint, and without a server it records an unknown box.
        let recipe = served.as_ref().map(|s| s.recipe_id.clone());
        let serve_resolved = served
            .as_ref()
            .map(|s| s.resolved.clone())
            .unwrap_or_default();
        let serve_env = served
            .as_ref()
            .map(|s| s.serve_env.clone())
            .unwrap_or_default();
        let (sha_at_start, dirty_at_start) = provenance.unwrap_or_default();
        let written = super::bench_record::write_gate_record(
            &outcome.record,
            &target.base_url,
            &target.model,
            recipe,
            serve_resolved,
            serve_env,
            sha_at_start,
            dirty_at_start,
            match &args.output_image {
                Some(target) => Some((
                    target.clone(),
                    args.output_image_args
                        .as_deref()
                        .map(metrale_bench::gate::card::parse_args)
                        .transpose()
                        .map_err(|e| anyhow::anyhow!("--output-image-args: {e}"))?
                        .unwrap_or_default(),
                )),
                None => None,
            },
        )
        .await;
        if let Some(s) = served {
            s.shutdown().await;
        }
        written?;
    }

    match args.format {
        OutputFormat::Json => bench_print::print_record(&outcome.record, OutputFormat::Json)?,
        OutputFormat::Text => {
            println!();
            bench_print::print_frame(&outcome.record.frame);
            if let Some(path) = &outcome.saved_to {
                eprintln!("\nrecorded as {}", path.display());
            }
        }
    }

    let code = outcome.exit_code();
    // 2026-09-26: `--no-fail-on-verdict` turns a failed verdict (code 2) into 0.
    if code == 2 && args.no_fail_on_verdict {
        return Ok(0);
    }
    Ok(code)
}
