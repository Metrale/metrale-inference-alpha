// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Serve a benchmark's own recipe in this process for the
//! duration of a `--pull-request-gate` run, so the record measures the config
//! the baseline names rather than a hand-started endpoint.
//!
//! Owner: server CLI (`met benchmark`).
//! Invariants:
//! - At most one self-start per process. Teardown trips the
//!   `tui::shutdown` latch, which has no reset, and a tripped latch stops
//!   both `run_blocking` (its cancel check) and `model_swap` (its load guard).
//!   `claim_start_slot` refuses a second start, or a start after any shutdown
//!   request.
//! - The server's task handle is owned by [`SelfServed`] from the spawn on;
//!   its `Drop` aborts the task on any path that skips `shutdown()`.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use metrale_bench::{TargetEndpoint, gate};

const POLL: Duration = Duration::from_millis(500);

static STARTED: AtomicBool = AtomicBool::new(false);

/// 2026-09-26: A server for a gate run, and the endpoint that reaches it:
/// started by this process, or the leased one ([`Self::external`]).
pub struct SelfServed {
    pub target: TargetEndpoint,
    /// 2026-09-26: The recipe that produced it, for the record's provenance.
    pub recipe_id: String,
    /// 2026-09-26: The recipe keys overridden for this run (the baseline's
    /// `serve_overrides` pin merged with `--serve-override`, after `--hermetic`
    /// expansion), for the record. Empty means the recipe ran as pinned.
    pub overrides: BTreeMap<String, String>,
    /// 2026-09-26: What the serve resolved, for the record's `serve_resolved`
    /// (`ServePlan::disclosed`).
    pub resolved: BTreeMap<String, String>,
    /// 2026-09-26: The `METRALE_*` serve levers the server runs under (the
    /// declaration after `serve_env::reconcile`), for the record's `serve_env`.
    /// Empty when none are declared.
    pub serve_env: BTreeMap<String, String>,
    /// 2026-09-26: The served variant's baseline entry. `bench_run` derives the
    /// run's pinned and threshold-coupled params from it
    /// (`bench_resolve::apply_param_overrides`, `apply_threshold_params`).
    pub baseline_entry: gate::ModelBaseline,
    /// 2026-09-26: `None` for an external server and once teardown has taken
    /// it. An `Option` so `Drop`, which cannot move out of `self`, can take it.
    server: Option<tokio::task::JoinHandle<Result<()>>>,
}

impl SelfServed {
    /// 2026-09-26: A server this process will not stop: the leased one from
    /// `bench_lease`. `shutdown()` still trips the shutdown latch, but there is
    /// no task to abort, and `Drop` does nothing.
    pub fn external(
        target: TargetEndpoint,
        recipe_id: String,
        overrides: BTreeMap<String, String>,
        resolved: BTreeMap<String, String>,
        serve_env: BTreeMap<String, String>,
        baseline_entry: gate::ModelBaseline,
    ) -> Self {
        Self {
            target,
            recipe_id,
            overrides,
            resolved,
            serve_env,
            baseline_entry,
            server: None,
        }
    }

    /// 2026-09-26: Request shutdown, then abort the server task and wait for it.
    ///
    /// Call this after the gate record is written. The record's hardware
    /// fingerprint is fetched from the endpoint, and that fetch returns
    /// `Hardware::unknown()` on failure without an error, so tearing down first
    /// would write a record that names no box.
    pub async fn shutdown(mut self) {
        crate::tui::shutdown::request("benchmark gate run finished");
        if let Some(server) = self.server.take() {
            server.abort();
            let _ = server.await;
        }
    }
}

impl Drop for SelfServed {
    /// 2026-09-26: Teardown for the paths that never reach [`Self::shutdown`].
    ///
    /// A dropped `JoinHandle` detaches its task and leaves the model resident,
    /// so this aborts the task. `Drop` cannot await, so it does not wait; after
    /// [`Self::shutdown`] `server` is `None` and this does nothing.
    ///
    /// It does not call `shutdown::request`: that latch is process-wide with no
    /// reset, and would stop every later `model_swap` and `run_blocking` in the
    /// process.
    fn drop(&mut self) {
        let Some(server) = self.server.take() else {
            return;
        };
        eprintln!("gate: tearing down the self-started server (no explicit shutdown ran)");
        server.abort();
    }
}

/// 2026-09-26: Resolve the recipe for `benchmark_id` and serve it in this
/// process on a free port.
///
/// The resolution is `bench_serve_plan::plan_serve`, shared with
/// `bench_lease`. `hardware: None` uses the baseline's only box class and
/// refuses when there are several. `overrides` (`--serve-override`) are merged
/// over the entry's `[benchmarks.serve_overrides]` pin, the operator winning a
/// clash, and the merged set is returned in [`SelfServed::overrides`] for the
/// record; the gate check requires the pins on the record (`gate::scoring`).
pub async fn serve_for(
    benchmark_id: &str,
    hardware: Option<&str>,
    checkpoint: Option<&str>,
    overrides: BTreeMap<String, String>,
) -> Result<SelfServed> {
    let plan = super::bench_serve_plan::plan_serve(benchmark_id, hardware, checkpoint, overrides)?;
    // 2026-09-26: The server is a task in this process, so its levers are this
    // process's environment: every declared lever must already be set, and no
    // undeclared one may be.
    let reconciled = plan.reconcile_env()?;
    refuse_unapplied_levers(&plan.recipe_id, &reconciled)?;
    let port = metrale_bench::benchmarks::agentic::score::free_port()?;
    let serve_args = plan.serve_args(port)?;
    let resolved = plan.disclosed(port)?;
    check_box_is_free_enough(
        serve_args.gpu_memory_utilization,
        &plan.recipe_id,
        plan.limits.memory.min_free_fraction,
    )?;

    // 2026-09-26: Claimed immediately before the spawn, not on entry: a failure
    // above leaves no server and no tripped latch, so it must not use up the
    // process's one slot.
    claim_start_slot(&STARTED, crate::tui::shutdown::requested())?;

    let model = plan.model;
    eprintln!(
        "gate: serving {model} from recipe {} on port {port}",
        plan.recipe_id
    );
    let server =
        tokio::spawn(async move { crate::main_modules::serve::serve(serve_args, None).await });

    // 2026-09-26: The handle goes into `SelfServed` before the wait, so an error
    // from the `?` below drops it through `Drop`, which aborts the task.
    let mut served = SelfServed {
        target: TargetEndpoint::local(port, &model),
        recipe_id: plan.recipe_id,
        overrides: plan.requested,
        resolved,
        serve_env: reconciled.env,
        baseline_entry: plan.entry,
        server: Some(server),
    };
    await_serving(
        &served.target,
        &model,
        served.server.as_mut().expect("just constructed as Some"),
        Duration::from_secs(plan.limits.timing.boot_timeout_s),
    )
    .await?;
    eprintln!("gate: endpoint is serving {model}");

    Ok(served)
}

/// 2026-09-26: Claim this process's one self-start slot. Takes the slot and
/// the latch state as arguments, so tests need neither a server nor the
/// process globals.
///
/// Refused, with distinct messages:
/// * a shutdown was already requested: a server started now would never
///   begin serving;
/// * a gate already started a server here (`started` was set).
fn claim_start_slot(started: &AtomicBool, shutdown_requested: bool) -> Result<()> {
    if shutdown_requested {
        bail!(
            "a shutdown has already been requested in this process, and that latch has no reset. \
             A server started now would return without ever serving, so the run is refused here \
             rather than after a fifteen-minute wait for a listener that is not coming."
        );
    }
    if started.swap(true, Ordering::SeqCst) {
        bail!(
            "a benchmark gate already started a server in this process; the shutdown latch is \
             one-way, so a second one cannot come up. Run one benchmark per invocation."
        );
    }
    Ok(())
}

/// 2026-09-26: Refuse an in-process serve when a declared lever is missing
/// from this process's environment, printing the `env` line to export them.
///
/// A child serve (`--serve-reuse`, `bench_lease`) is handed the missing
/// levers; an in-process serve cannot be, because setting the environment
/// after the runtime's threads exist races their reads. Pure over the
/// reconciliation.
fn refuse_unapplied_levers(
    recipe_id: &str,
    reconciled: &metrale_bench::serve_env::Reconciled,
) -> Result<()> {
    if reconciled.missing.is_empty() {
        return Ok(());
    }
    let exports = reconciled
        .missing
        .iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join(" ");
    bail!(
        "recipe {recipe_id:?} is measured under {} serve lever(s) this process does not carry: \
         {exports}. A gate that serves in-process cannot set them after it has started, so \
         export them first:\n    env {exports} met benchmark run …\nor run with \
         --serve-reuse, which starts the server as a child and hands it the recipe's set.",
        reconciled.missing.len()
    );
}

/// 2026-09-26: Refuse to start a server when the host's available memory is
/// below `min_free_fraction` of the total. The recipe's utilisation is served
/// as written; this checks only the live figure. When memory cannot be read,
/// it prints that and passes.
pub(super) fn check_box_is_free_enough(
    util: f64,
    recipe_id: &str,
    min_free_fraction: f64,
) -> Result<()> {
    let Some((total_gib, avail_gib)) = host_memory_gib() else {
        // 2026-09-26: No usable /proc/meminfo: warn and continue rather than
        // refuse every host that lacks it.
        eprintln!("gate: cannot read host memory; skipping the free-memory preflight");
        return Ok(());
    };
    eprintln!(
        "{}",
        headroom_verdict(total_gib, avail_gib, util, recipe_id, min_free_fraction)?
    );
    Ok(())
}

/// 2026-09-26: The preflight decision for one reading: the line to print, or
/// the refusal naming where to look for what holds the memory. Pure.
/// `total_gib` is positive: [`host_memory_gib`] returns `None` otherwise.
fn headroom_verdict(
    total_gib: f64,
    avail_gib: f64,
    util: f64,
    recipe_id: &str,
    min_free_fraction: f64,
) -> Result<String> {
    let free_fraction = avail_gib / total_gib;
    if free_fraction < min_free_fraction {
        bail!(
            "this box is not free enough to serve recipe {recipe_id:?}: only {avail_gib:.0} GiB \
             of {total_gib:.0} GiB is available ({:.0} %, below the {:.0} % a self-start \
             requires). Something else is holding memory — check `sudo docker ps` and \
             `nvidia-smi --query-compute-apps=pid,used_memory --format=csv`, free it, and \
             re-run. This is not a judgement on the recipe's --gpu-memory-utilization {util:.2}, \
             which is served exactly as written: co-tenancy is what turns a working \
             utilisation into an OOM freeze on unified memory, and it corrupts the measurement \
             well before that.",
            free_fraction * 100.0,
            min_free_fraction * 100.0,
        );
    }
    Ok(format!(
        "gate: {avail_gib:.0} GiB of {total_gib:.0} GiB free ({:.0} %); serving at {util:.2} as the recipe states",
        free_fraction * 100.0
    ))
}

/// 2026-09-26: `(MemTotal, MemAvailable)` in GiB from `/proc/meminfo`.
///
/// `MemAvailable`, not `MemFree`: page cache is reclaimable and `MemFree`
/// excludes it.
///
/// A non-positive `MemTotal` or a non-finite `MemAvailable` returns `None`:
/// the fraction would be NaN or infinite, and `NaN < min_free_fraction` is
/// false, so the preflight would pass.
fn host_memory_gib() -> Option<(f64, f64)> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    let field = |name: &str| -> Option<f64> {
        text.lines()
            .find(|l| l.starts_with(name))?
            .split_whitespace()
            .nth(1)?
            .parse::<f64>()
            .ok()
            .map(|kb| kb / 1024.0 / 1024.0)
    };
    let (total, avail) = (field("MemTotal:")?, field("MemAvailable:")?);
    (total > 0.0 && avail.is_finite()).then_some((total, avail))
}

/// 2026-09-26: Block until `/v1/models` names `model`, or fail at
/// `boot_timeout`. A server answering with another model does not count. A
/// server task that finishes first fails at once with its own error.
async fn await_serving(
    target: &TargetEndpoint,
    model: &str,
    server: &mut tokio::task::JoinHandle<Result<()>>,
    boot_timeout: Duration,
) -> Result<()> {
    let deadline = Instant::now() + boot_timeout;
    loop {
        if server.is_finished() {
            // 2026-09-26: Await the finished task for the reason it stopped.
            return match server.await {
                Ok(Err(e)) => Err(e).with_context(|| {
                    format!("the server failed before it began serving {model:?}")
                }),
                Ok(Ok(())) => bail!(
                    "the server returned before it began serving {model:?} — it stopped without \
                     an error, which should not happen while the accept loop is running"
                ),
                Err(join) => Err(anyhow::Error::new(join))
                    .with_context(|| format!("the server task died serving {model:?}")),
            };
        }
        // 2026-09-26: Kept for the timeout message, which then says whether
        // nothing answered or another model did.
        let last = match metrale_bench::http::list_models(target, Duration::from_secs(5)).await {
            Ok(models) if models.iter().any(|m| m == model) => return Ok(()),
            Ok(models) => format!("the endpoint is serving {models:?}"),
            Err(e) => format!("{e:#}"),
        };
        if Instant::now() >= deadline {
            bail!(
                "{model:?} did not come up within {}s — {last}",
                boot_timeout.as_secs()
            );
        }
        tokio::time::sleep(POLL).await;
    }
}

#[cfg(test)]
#[path = "bench_selfstart_tests.rs"]
mod tests;
