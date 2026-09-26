// SPDX-License-Identifier: MIT OR Apache-2.0

#![deny(warnings)]
#![deny(clippy::all)]
#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::large_enum_variant)]
#![allow(clippy::doc_lazy_continuation)]
#![allow(clippy::doc_overindented_list_items)]

//! 2026-09-26: Entry point of the `met` binary: parse the CLI, refuse undeclared `METRALE_*` variables, install logging, and run the subcommand.
//!
//! Owner: server.
//! Invariants:
//! - `dump-serve-options`, `sync-recipes` and `doctor` return or exit before any
//!   logging subscriber or dashboard is installed.
//! - A `serve` run that returns while the fault latch is set exits through
//!   `metrale_core::fault::exit_code`, never with status 0.

mod adaptive_sampler;
mod anthropic;
mod api;
mod auth;
mod citation;
mod citation_structured;
mod cli;
mod conversation_store;
mod disk_guard;
mod env_config;
mod error_hints;
pub mod grammar;
mod halluc_probe;
mod hint_injector;
mod identity;
mod ids;
mod ir;
mod llmlingua;
mod lookback_lens;
mod loop_detector;
mod loop_simhash;
mod lqer;
mod main_modules;
pub mod metrics;
mod model_download;
mod model_resolver;
mod moe_quality;
mod openai;
mod rate_limiter;
pub mod reasoning_parser;
pub mod recipe;
mod refusal;
mod request_dumper;
mod response_store;
mod retrieval_heads;
mod scheduler;
mod scheduling_policy;
mod session_manager;
mod symbol_trie;
mod tokenizer;
mod tool_arg_dedup;
pub mod tool_parser;
mod tool_rag;
mod tscg;
pub mod tui;

use anyhow::Result;
use clap::Parser;

use crate::cli::{Cli, Command};
use crate::main_modules::serve;

pub(crate) use crate::main_modules::AppState;

pub type ModelBehavior = metrale_kernels::ModelBehavior;

fn main() -> Result<()> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(serve_main())
}

async fn serve_main() -> Result<()> {
    // 2026-09-26: Parsed before the subscriber is installed, because the TUI
    // gate (`tui::plain_mode`) needs `--no-tui`.
    let cli = Cli::parse();

    // 2026-09-26: Every `METRALE_*` variable in the environment must be a
    // declared lever (`metrale_config::levers::check`). An undeclared one is a
    // typo or a removed lever, so the run is refused before anything starts
    // rather than served under a configuration that does not apply.
    metrale_config::levers::check(
        std::env::vars_os().map(|(name, _)| name.to_string_lossy().into_owned()),
    )?;

    // 2026-09-26: Prints the serve-option manifest as JSON and returns before
    // any subscriber, TUI or GPU exists; a dashboard would garble that output.
    if matches!(cli.command, Command::DumpServeOptions) {
        println!(
            "{}",
            serde_json::to_string_pretty(&cli::manifest::build())
                .expect("the manifest is plain data and always serialises")
        );
        return Ok(());
    }

    // 2026-09-26: Fetches the recipe index and prints two lines; no
    // subscriber, TUI or GPU.
    if matches!(cli.command, Command::SyncRecipes) {
        return cli::sync_recipes::run();
    }

    // 2026-09-26: `doctor` prints its findings and exits with status 1 when it
    // found a problem; no subscriber, TUI or GPU.
    if matches!(cli.command, Command::Doctor) {
        let code = cli::doctor::dispatch()?;
        std::process::exit(code);
    }

    let no_tui = match &cli.command {
        // 2026-09-26: `--check-kernels` prints a report and one JSON line on
        // stdout and exits, so it runs without a dashboard.
        Command::Serve(args) => args.no_tui || args.rank > 0 || args.check_kernels,
        // 2026-09-26: Benchmarks always log in plain mode.
        Command::Benchmark(_) => true,
        Command::DumpServeOptions | Command::SyncRecipes | Command::Doctor => true,
    };

    // 2026-09-26: `benchmark certify --json` writes JSON lines on stdout, so
    // the log moves to stderr for that command only.
    let logs_to_stderr = matches!(
        &cli.command,
        Command::Benchmark(b) if b.json_stdout()
    );
    let tui_channels = if tui::plain_mode(no_tui) {
        let fmt = tracing_subscriber::fmt().with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        );
        if logs_to_stderr {
            fmt.with_writer(std::io::stderr).init();
        } else {
            fmt.init();
        }
        None
    } else {
        let (progress_tx, progress_rx) = std::sync::mpsc::channel();
        tui::init::install_tty_subscriber(progress_tx);
        Some(progress_rx)
    };

    // 2026-09-26: Race the server against a shutdown during startup. Pinning
    // `serve()` without a spawn is enough, because its blocking startup runs
    // under `spawn_blocking` and the future yields while it does.
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<&'static str>();
    tui::shutdown::arm_startup_escape(shutdown_tx);
    let result = match cli.command {
        // 2026-09-26: Returned above. An explicit arm, not a wildcard, so a new
        // subcommand fails to compile here until it is routed.
        Command::DumpServeOptions | Command::SyncRecipes | Command::Doctor => {
            unreachable!("handled before initialisation")
        }
        Command::Benchmark(args) => {
            // 2026-09-26: Benchmarks load no model and do not use the startup
            // escape.
            drop(shutdown_rx);
            cli::bench_run::dispatch(args).await
        }
        Command::Serve(args) => {
            let serving = serve(args, tui_channels);
            tokio::pin!(serving);
            // 2026-09-26: Only a send means shutdown. The sender stays parked in
            // `tui::shutdown` for the life of the process, and a closed channel
            // waits forever instead of stopping a healthy server.
            let shutdown_signal = async {
                match shutdown_rx.await {
                    Ok(reason) => reason,
                    Err(_) => std::future::pending::<&'static str>().await,
                }
            };
            tokio::pin!(shutdown_signal);
            tokio::select! {
                res = &mut serving => res,
                reason = &mut shutdown_signal => {
                    // 2026-09-26: Cancelled before the server came up: nothing is
                    // in flight, and the startup task is abandoned.
                    tracing::info!(
                        "Shutdown requested ({reason}) during startup — exiting before the server came up"
                    );
                    // 2026-09-26: The same cleanup as the normal tail, then exit
                    // without returning: the startup task on the blocking pool
                    // cannot be aborted, and dropping the runtime would wait for it.
                    tui::stop_and_join(std::time::Duration::from_secs(2));
                    tui::terminal_guard::restore();
                    tui::init::flush_tee();
                    // 2026-09-26: A failed kernel launch or memset can latch the
                    // fault before the server is up, so this exit maps the status
                    // the same way as the normal tail.
                    std::process::exit(metrale_core::fault::exit_code(
                        true,
                        metrale_core::fault::global().fault(),
                    ));
                }
            }
        }
    };
    // 2026-09-26: Stop the dashboard thread and wait up to 2 s for it to drop
    // its terminal guard before an error prints. `restore()` then covers a
    // thread that had not entered raw mode yet or did not stop in time.
    tui::stop_and_join(std::time::Duration::from_secs(2));
    tui::terminal_guard::restore();
    tui::init::flush_tee();

    // 2026-09-26: A GPU fault requests the same drain as `SIGTERM`, so
    // `serve()` can return `Ok`; the latch is what makes the exit status
    // nonzero. A healthy run returns its own result.
    match metrale_core::fault::global().fault() {
        Some(reason) => {
            if let Err(e) = &result {
                tracing::error!("{e:#}");
            }
            tracing::error!(
                "Exiting after a fatal GPU fault ({reason}). The CUDA context is \
                 destroyed and cannot be recovered in-process; restart the server."
            );
            std::process::exit(metrale_core::fault::exit_code(result.is_ok(), Some(reason)));
        }
        None => result,
    }
}

#[cfg(test)]
#[path = "main_exit_tests.rs"]
mod main_exit_tests;
