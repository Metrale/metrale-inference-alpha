// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The ratatui dashboard for `met serve`, on its own `metrale-tui` thread.
//!
//! Owner: server tui.
//! Invariants:
//! - `main.rs` starts nothing here unless [`plain_mode`] is false; otherwise it
//!   installs the plain `tracing_subscriber::fmt` layout.
//! - The render thread does not block on a future:
//!   `.github/scripts/check-no-block-on.sh` (run by `tui-threading.yml`)
//!   refuses calls to `block_on` and `block_in_place` in non-test files under `tui/`
//!   and `recipe/`.
//!
//! Work that is already async (the loopback chat client, the benchmark
//! executor) runs on the serving runtime through the `tokio::runtime::Handle`
//! captured in [`start`]. Blocking work runs on named `std::thread`s. Both
//! answer through a `std::sync::mpsc` channel that the render loop polls with
//! `try_recv`. The one-shot workers `metrale-recipes`, `metrale-recipe-date`,
//! `metrale-libscan` and `metrale-freshness` go through [`worker::spawn`],
//! which still answers when the thread cannot start. `metrale-download`
//! streams many progress messages, and `metrale-swap` sends only on failure,
//! so neither uses it.

pub mod capture_layer;
pub mod clipboard;
pub mod init;
pub mod log_ring;
pub mod selection;
pub mod shutdown;
pub mod terminal_guard;

pub mod app;
mod app_input;
pub mod app_library;
mod app_nav;
pub mod app_quit;
pub mod app_scroll;
mod app_types;
pub mod bench_host;
pub mod bench_keys;
pub mod bench_preflight;
pub mod bench_state;
pub mod bench_variants;
pub mod commands;
pub mod download_state;
pub mod events;
pub mod events_rules;
pub mod format;
pub mod help_keys;
pub mod help_state;
pub mod lib_borrow;
pub mod lib_config;
pub mod lib_dates;
pub mod lib_fields;
pub mod lib_keys;
pub mod lib_modal;
pub mod lib_scan;
pub mod lib_start;
pub mod lib_state;
pub mod logo;
pub mod progress;
pub mod redact;
pub mod report;
pub mod report_http;
pub mod section;
pub mod theme;
pub mod worker;

pub mod chat;
pub mod chat_stream;
pub mod chat_thinking;
pub mod data;
pub mod render;

use std::io::IsTerminal;

/// 2026-09-26: A live run's levers, so `/watchdog on|off` toggles that run's
/// flag. `load_model` sends them at boot and again on each swap.
pub type RunLevers = std::sync::Arc<crate::scheduler::levers::SchedLevers>;

/// 2026-09-26: The live run's handles: the levers the dashboard can toggle
/// and the scheduler snapshot cell. `load_model` publishes both in one send.
#[derive(Clone)]
pub struct RunHandles {
    pub levers: RunLevers,
    pub snapshot: std::sync::Arc<metrale_speculative::snapshot::SnapshotCell>,
}

/// 2026-09-26: Start the dashboard thread. `serve` calls this only on rank 0
/// and only when `plain_mode` was false. Captures the tokio runtime handle for
/// the chat client and the benchmark executor, and returns the sender for each
/// run's handles. If the spawn fails the receiver is dropped, so sends fail
/// and the callers ignore the error.
pub fn start(
    args: crate::cli::ServeArgs,
    progress_rx: std::sync::mpsc::Receiver<capture_layer::ProgressEvent>,
    host: std::sync::Arc<crate::main_modules::model_host::ModelHost>,
) -> std::sync::mpsc::Sender<RunHandles> {
    let (levers_tx, levers_rx) = std::sync::mpsc::channel::<RunHandles>();
    let runtime = tokio::runtime::Handle::current();
    // 2026-09-26: Claim the terminal on the caller's thread, before the TUI
    // thread exists. `SwitchableIo` writes logs to stdout while `TUI_ACTIVE`
    // is false, and a line written after the thread enters the alternate
    // screen scrolls it out of step with ratatui's diff.
    init::TUI_ACTIVE.store(true, std::sync::atomic::Ordering::SeqCst);
    match std::thread::Builder::new()
        .name("metrale-tui".into())
        .spawn(move || {
            let port = args.port;
            let model = args
                .model_name
                .clone()
                .or_else(|| args.model.clone())
                .unwrap_or_default();
            let cache_dir = args.cache_dir.clone();
            let mut app = app::App::new(args);
            // 2026-09-26: The serve matrix boots checkpoints through this
            // process; without an installed host it refuses to load with
            // `serve_matrix::host::NO_HOST`.
            metrale_bench::benchmarks::serve_matrix::host::install(std::sync::Arc::new(
                bench_host::TuiServeHost::new(host.clone(), cache_dir),
            ));
            app.host = Some(host);
            app.chat.set_runtime(runtime.clone());
            // 2026-09-26: Benchmarks target this server. When
            // `ArtifactStore::discover` fails (no usable `METRALE_HOME` or
            // `HOME`) the dashboard still starts, with a warning.
            match metrale_bench::ArtifactStore::discover() {
                Ok(store) => app.bench.attach(
                    metrale_bench::BenchmarkExecutor::new(runtime, store),
                    metrale_bench::TargetEndpoint::local(port, model),
                ),
                Err(e) => tracing::warn!("benchmarks unavailable: {e:#}"),
            }
            events::run(app, progress_rx, levers_rx);
        }) {
        Ok(handle) => {
            *THREAD.lock() = Some(handle);
        }
        Err(e) => tracing::warn!("TUI thread failed to start: {e}"),
    }
    levers_tx
}

/// 2026-09-26: The dashboard thread's handle, for the exit-path join. A static
/// because `stop_and_join` runs from `main`'s exit paths, which hold no server
/// state; it holds only a join handle.
static THREAD: parking_lot::Mutex<Option<std::thread::JoinHandle<()>>> =
    parking_lot::Mutex::new(None);

/// 2026-09-26: Exit path: request shutdown and wait up to `timeout` for the
/// TUI thread to finish and drop its `TerminalGuard`. Without the wait, a
/// `serve()` that fails before the thread enters raw mode would run the
/// caller's `terminal_guard::restore()` first, and the thread would then take
/// the terminal as the process exits.
pub fn stop_and_join(timeout: std::time::Duration) {
    let Some(handle) = THREAD.lock().take() else {
        return;
    };
    shutdown::request("process exit");
    let deadline = std::time::Instant::now() + timeout;
    while !handle.is_finished() && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    if handle.is_finished() {
        let _ = handle.join();
    }
}

/// 2026-09-26: True when the TUI must not start and `main.rs` installs the
/// plain fmt subscriber.
///
/// True for `--no-tui`, `METRALE_NO_TUI=1`, stdout or stdin not a terminal,
/// or `TERM` unset or `dumb`. `serve` additionally starts the TUI only on
/// rank 0.
pub fn plain_mode(no_tui_flag: bool) -> bool {
    if no_tui_flag {
        return true;
    }
    if std::env::var("METRALE_NO_TUI").as_deref() == Ok("1") {
        return true;
    }
    if !std::io::stdout().is_terminal() || !std::io::stdin().is_terminal() {
        return true;
    }
    matches!(std::env::var("TERM").as_deref(), Ok("dumb") | Err(_))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_flag_always_wins() {
        assert!(plain_mode(true));
    }

    #[test]
    fn piped_test_runner_is_plain() {
        // 2026-09-26: Under `cargo test` stdout is captured, so it is not a
        // terminal.
        assert!(plain_mode(false));
    }
}
