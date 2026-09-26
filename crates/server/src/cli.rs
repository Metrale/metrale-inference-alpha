// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The `met` command line: `Cli` and its subcommands. The argument structs are
//! in the `cli/` modules.
//!
//! Owner: server CLI.
//! Invariants: none beyond the types.

use clap::Parser;

pub mod bench_aggregate;
mod bench_args;
pub mod bench_card;
pub(crate) mod bench_cause;
pub mod bench_certify;
mod bench_gate_check;
pub mod bench_lease;
mod bench_print;
pub mod bench_record;
mod bench_resolve;
pub mod bench_run;
mod bench_selfstart;
mod bench_serve_plan;
pub(crate) mod doctor;
pub(crate) mod flag_values;
pub(crate) mod hermetic;
pub(crate) mod manifest;
mod serve_args;
pub(crate) mod sync_recipes;
mod validate;
pub use bench_args::BenchmarkArgs;
pub use serve_args::{DEFAULT_KV_CACHE_DTYPE, DEFAULT_NUM_DRAFTS, ServeArgs};
pub use validate::validate_serve_args;

/// 2026-09-26: The release string, e.g. `1.0.0-beta-preview`: the package version from the
/// workspace `Cargo.toml`, which `met --version` prints. Code that records which engine
/// build produced an artifact should read this constant rather than derive its own.
pub const METRALE_VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Parser, Debug)]
#[command(
    name = "met",
    version = METRALE_VERSION,
    about = "Metrale Engine — pure Rust LLM inference server"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(clap::Subcommand, Debug)]
pub enum Command {
    /// Start the inference server.
    Serve(ServeArgs),
    /// Run and inspect the benchmark suite, without the dashboard.
    #[command(visible_alias = "bench")]
    Benchmark(BenchmarkArgs),
    /// Print the serve flag surface as JSON.
    ///
    /// Hidden because it is a build tool, not part of the supported CLI: it
    /// exists so downstream tooling can be generated from clap rather than
    /// transcribed from it. `ServeArgs` has no `Serialize` derive, and this
    /// does not promise that any flag keeps its name: a rename shows up as a
    /// diff in whatever consumes the output.
    #[command(hide = true)]
    DumpServeOptions,
    /// Populate the local recipe index from the recipe repository.
    ///
    /// `benchmark run` resolves a recipe id against this index. The TUI Library
    /// also fills it, but a CI runner, a container or a machine reached over
    /// ssh cannot open the TUI; this command fills it without one.
    ///
    /// It is a separate command rather than an automatic fetch inside
    /// `benchmark run`, so a benchmark never reaches the network mid-run and
    /// its result depends only on what was declared.
    SyncRecipes,
    /// Report whether this box can run a benchmark, and say what to fix.
    ///
    /// Each check covers one cause of the same symptom, `recipe "..." is not in
    /// the local index (0 cached)`: an `~/.metrale` owned by another uid, a
    /// `sync-recipes` that was never run, or a signing identity created in a
    /// scratch METRALE_HOME whose key was never committed.
    ///
    /// Exits non-zero when anything is wrong, so a provisioning script can gate
    /// on it.
    Doctor,
}

#[cfg(test)]
mod bool_surface_tests;

#[cfg(test)]
mod version_tests {
    use super::*;

    #[test]
    fn version_flag_reports_the_packaged_version() {
        // 2026-09-26: `--version` ends parsing early, so clap returns it as an error whose
        // kind is DisplayVersion and whose rendering is the output.
        let err = Cli::try_parse_from(["met", "--version"]).expect_err("exits early");
        assert_eq!(err.kind(), clap::error::ErrorKind::DisplayVersion);
        assert!(
            err.to_string().contains(METRALE_VERSION),
            "`--version` printed {:?}, which does not carry {METRALE_VERSION}",
            err.to_string()
        );
    }

    #[test]
    fn the_reported_version_is_the_cargo_version() {
        // 2026-09-26: The constant is the package version itself, not a literal copy.
        assert_eq!(METRALE_VERSION, env!("CARGO_PKG_VERSION"));
        assert!(!METRALE_VERSION.is_empty());
    }
}
