// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Replace the running model without restarting the process.
//!
//! The order in [`swap`]:
//! 1. Take the model out of the host: a new request that needs a model gets
//!    503 `model_not_loaded` (`CurrentModel`), and running requests keep the
//!    `Arc` they took.
//! 2. Drop the outgoing `AppState`, which closes `request_tx`.
//! 3. Join the scheduler. It stops on an idle tick once its inbox has closed
//!    (`scheduler/core/tick.rs`), and `Model::teardown` frees the model's
//!    pools as it finishes (`scheduler/core/finish.rs`).
//! 4. Load the new model with the process-scoped state carried over, and
//!    publish it.
//!
//! After step 3 the old model is gone. Before step 1, `swap` refuses what it
//! can check cheaply: flag validation, shutdown, multi-rank, missing kernels.
//! If the load fails, the previous argv is reloaded; if that fails too, no
//! model is loaded and the error names both failures.
//!
//! Owner: server (model hosting).
//! Invariants: `swap` returns `Ok` only after publishing the model `next`
//! describes, or on finding that argv already serving.

use std::sync::Arc;

use anyhow::Result;

use super::model_host::ModelHost;
use super::serve_load::{Carried, load_model};
use crate::cli;

#[derive(Debug)]
pub(crate) struct SwapOutcome {
    /// 2026-09-26: The argv of the model that was replaced. No caller reads it.
    pub previous: Option<cli::ServeArgs>,
}

/// 2026-09-26: Refuse to start a load once shutdown has been requested, before
/// the serving model is released. Takes the flag as an argument, so tests need
/// not set the process-wide latch, which has no reset.
fn refuse_if_shutting_down(shutting_down: bool) -> Result<()> {
    anyhow::ensure!(
        !shutting_down,
        "shutdown is in progress — not starting a model load"
    );
    Ok(())
}

/// 2026-09-26: Refuse a model this build has no kernels for, before anything
/// is released. The load runs the same check only after the old model is gone.
fn preflight_kernel_target(args: &cli::ServeArgs) -> Result<()> {
    let model_dir = super::serve_phases::resolve_model_dir(args)?;
    let (config, _) = super::serve_phases::load_model_config(&model_dir)?;
    // 2026-09-26: The same references and pin `load_model` passes to
    // `ptx_for_config`, so an ambiguous match is refused here too.
    let model_dir_str = model_dir.display().to_string();
    let model_refs: Vec<&str> = [
        args.model.as_deref(),
        args.model_name.as_deref(),
        Some(model_dir_str.as_str()),
    ]
    .into_iter()
    .flatten()
    .collect();
    let resolved = metrale_kernels::ptx_for_config(
        &config.model_type,
        config.hidden_size,
        &model_refs,
        args.kernel_target.as_deref(),
    )
    .map_err(|e| anyhow::anyhow!("{e} — the running model is untouched"))?;
    if resolved.is_none() {
        anyhow::bail!(
            "this build has no compiled kernels for model_type '{}' / hidden_size={} \
             (available: {:?}) — the running model is untouched",
            config.model_type,
            config.hidden_size,
            metrale_kernels::available_targets()
                .iter()
                .map(|t| &t.target.model)
                .collect::<Vec<_>>()
        );
    }
    Ok(())
}

/// 2026-09-26: Copy the process-scoped flags (`--auto-swap`, the bind address
/// and port, `--dump`) from the running argv onto the next one. The listener
/// is bound for the process lifetime, so the next argv's socket cannot apply.
fn carry_process_flags(next: &mut cli::ServeArgs, previous: &cli::ServeArgs) {
    next.auto_swap = previous.auto_swap;
    next.bind = previous.bind.clone();
    next.port = previous.port;
    // 2026-09-26: `load_model` opens the dump from the new argv, so without
    // this a swap would stop the dump.
    next.dump = previous.dump.clone();
}

/// 2026-09-26: How long in-flight requests get to release the outgoing model
/// before the swap gives up and republishes it.
const DRAIN_GRACE: std::time::Duration = std::time::Duration::from_secs(30);

/// 2026-09-26: Wait up to `grace` for `state` to have no other owners; returns
/// how many remain.
fn wait_for_sole_owner<T>(state: &Arc<T>, grace: std::time::Duration) -> usize {
    let deadline = std::time::Instant::now() + grace;
    while Arc::strong_count(state) > 1 && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    Arc::strong_count(state) - 1
}

/// 2026-09-26: Take the outgoing model out of the host and drop it, returning
/// the process-scoped state to carry forward.
///
/// A reference that outlives `grace` would keep `request_tx` open, so the
/// scheduler join would never return; the model is then republished and the
/// swap refused.
fn release_state(host: &Arc<ModelHost>, grace: std::time::Duration) -> Result<Carried> {
    let Some(state) = host.take() else {
        // 2026-09-26: `serve` installs the process state before the listener
        // binds, so `None` is not expected; the environment is read again then.
        return match host.process() {
            Some(carried) => Ok(carried),
            None => Carried::from_env().map_err(|e| anyhow::anyhow!("{e}")),
        };
    };
    let carried = Carried::from_previous(&state);
    let holders = wait_for_sole_owner(&state, grace);
    if holders > 0 {
        // 2026-09-26: Nothing has been freed, so republishing restores the
        // state from before the swap.
        host.publish(state);
        anyhow::bail!(
            "cannot swap: {holders} reference(s) to the running model outlived \
             the {}s drain window, so it can never be released. The model is \
             still serving. This is a leaked `Arc<AppState>` — most likely one \
             bound into a router layer or a spawned task rather than resolved \
             per request.",
            grace.as_secs()
        );
    }
    drop(state);
    Ok(carried)
}

/// 2026-09-26: Report the router and listening phases and readiness after a
/// swap or a restore. `load_model` does not, because the listener was bound at
/// boot.
fn signal_listener_phases(host: &Arc<ModelHost>) {
    if let Some((bind, port)) = host.bound() {
        metrale_telemetry::progress::phase(10, "router");
        metrale_telemetry::progress::phase(11, "listening");
        metrale_telemetry::progress::ready(port);
        // 2026-09-26: The ready line a boot prints (`serve_router.rs`), logged
        // after the model is published.
        tracing::info!(
            "{}",
            super::serve_router::ready_line(&bind, port, host.live_model().as_deref())
        );
    }
}

/// 2026-09-26: Replace the running model with the one `next` describes.
///
/// Blocking: it loads a model. Call it off the runtime.
pub(crate) fn swap(host: &Arc<ModelHost>, next: cli::ServeArgs) -> Result<SwapOutcome> {
    // 2026-09-26: Serialised here rather than at a call site, so every caller
    // is covered.
    let _swapping = host.swap_guard();

    let previous_args = host.args();

    let mut next = next;
    if let Some(previous) = previous_args.as_ref() {
        if next.port != previous.port || next.bind != previous.bind {
            tracing::warn!(
                "this recipe asks to bind {}:{}, but the listener is on {}:{} for the process \
                 lifetime — serving the new model there instead",
                next.bind,
                next.port,
                previous.bind,
                previous.port
            );
        }
        carry_process_flags(&mut next, previous);
    }

    // 2026-09-26: A caller that waited on the guard may ask for what the
    // previous holder just loaded. The whole argv is compared, after the
    // process flags were carried: two recipes for one checkpoint differ.
    if previous_args.as_ref() == Some(&next) && host.current().is_some() {
        return Ok(SwapOutcome {
            previous: previous_args,
        });
    }

    // 2026-09-26: The refusals below run before anything is torn down.
    cli::validate_serve_args(&next).map_err(|e| anyhow::anyhow!("{e}"))?;

    refuse_if_shutting_down(crate::tui::shutdown::requested())?;

    // 2026-09-26: The load spawns Tokio tasks. Entering here serves callers
    // inside the runtime and the TUI's plain thread alike.
    let runtime = host.runtime();
    let _entered = runtime.as_ref().map(|h| h.enter());

    // 2026-09-26: Single-node only: an EP worker owns its model inside
    // `maybe_run_ep_worker`, out of reach of a swap on the head.
    anyhow::ensure!(
        next.world_size <= 1 && next.rank == 0,
        "hot-swap is single-node only (world_size={}, rank={})",
        next.world_size,
        next.rank
    );

    // 2026-09-26: Reads the checkpoint's `config.json`, so it runs after the
    // checks that need only the argv.
    preflight_kernel_target(&next)?;

    let carried = release_state(host, DRAIN_GRACE)?;

    if let Some(handle) = host.take_scheduler() {
        handle
            .join()
            .map_err(|_| anyhow::anyhow!("the scheduler thread panicked while draining"))?;
    }

    let next_args = next.clone();
    let tui_handles_tx = host.tui_handles();
    let load_err = match load_model(next, tui_handles_tx.clone(), carried.clone()) {
        Ok(Some(prepared)) => {
            host.set_scheduler(prepared.scheduler);
            host.set_args(next_args);
            host.publish(prepared.state);
            signal_listener_phases(host);
            return Ok(SwapOutcome {
                previous: previous_args,
            });
        }
        Ok(None) => anyhow::anyhow!("hot-swap reached an EP-worker path on rank 0"),
        Err(e) => e,
    };

    // 2026-09-26: The old model is already torn down; reload its argv.
    let Some(previous) = previous_args else {
        return Err(load_err
            .context("the new model failed to load and there was no previous model to restore"));
    };
    tracing::warn!("load failed, restoring the previous model: {load_err:#}");
    match load_model(previous.clone(), tui_handles_tx, carried) {
        Ok(Some(prepared)) => {
            host.set_scheduler(prepared.scheduler);
            host.set_args(previous);
            host.publish(prepared.state);
            signal_listener_phases(host);
            // 2026-09-26: An `Err`: the requested swap did not happen.
            Err(load_err.context("the new model failed to load; the previous one was restored"))
        }
        Ok(None) => Err(load_err.context("restore reached an EP-worker path")),
        Err(restore_err) => Err(load_err.context(format!(
            "the new model failed to load AND the previous one could not be \
             restored ({restore_err:#}) — no model is loaded"
        ))),
    }
}

#[cfg(test)]
#[path = "model_swap_tests.rs"]
mod tests;

#[cfg(test)]
mod drain_tests {
    use super::wait_for_sole_owner;
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    #[test]
    fn a_holder_that_lets_go_is_waited_for_rather_than_refused() {
        let state = Arc::new(0u32);
        let borrowed = state.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(150));
            drop(borrowed);
        });
        assert_eq!(wait_for_sole_owner(&state, Duration::from_secs(5)), 0);
    }

    #[test]
    fn a_holder_that_never_lets_go_is_reported_not_waited_on_forever() {
        let state = Arc::new(0u32);
        let _leaked = state.clone();
        let began = Instant::now();
        assert_eq!(wait_for_sole_owner(&state, Duration::from_millis(200)), 1);
        assert!(began.elapsed() < Duration::from_secs(2), "bounded");
    }

    #[test]
    fn an_unshared_state_is_released_without_waiting() {
        let state = Arc::new(0u32);
        let began = Instant::now();
        assert_eq!(wait_for_sole_owner(&state, Duration::from_secs(30)), 0);
        assert!(
            began.elapsed() < Duration::from_millis(50),
            "no sleep at all"
        );
    }
}
