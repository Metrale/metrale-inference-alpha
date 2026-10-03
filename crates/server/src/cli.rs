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
pub(crate) mod circuit;
mod circuit_diff;
pub(crate) mod circuit_hw;
pub(crate) mod circuit_hw_tree;
mod circuit_memory;
pub(crate) mod circuit_memory_point;
mod circuit_memory_serve;
pub(crate) mod circuit_memory_tables;
mod circuit_memory_weights;
mod circuit_paint;
mod circuit_precision;
mod circuit_venn;
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
    /// Show a recipe's architecture circuit and the fused kernel plan the engine runs.
    ///
    /// The circuits, precision tables and fusion rules are the ones this binary was built
    /// with (`kernels/circuits/`, `kernels/<hw>/common/FUSIONS.toml`).
    Circuit(CircuitArgs),
}

/// `met circuit`: inspect an architecture circuit.
#[derive(clap::Args, Debug)]
pub struct CircuitArgs {
    #[command(subcommand)]
    pub action: CircuitAction,
}

/// The `met circuit` views.
#[derive(clap::Subcommand, Debug)]
pub enum CircuitAction {
    /// Print the plan as stable text, one kernel group per line (the format of the checked-in
    /// plans under kernels/circuits/plans/).
    Show(CircuitPlanArgs),
    /// Draw the architecture for a terminal: the layer strip, one diagram per distinct layer
    /// plan with fused groups framed, and the per-step totals.
    Display(CircuitDisplayArgs),
    /// Load a model as `met serve` would and compare its decode logits, byte for byte, under
    /// the legacy forward and the circuit forward (reference rules only, then every rule).
    /// Set METRALE_DEBUG_NO_GRAPH=1 for the eager comparison; without it decode is graphed.
    Diff(Box<CircuitDiffArgs>),
    /// Classify a new model's ops against the kernel families of supported models (shared,
    /// parameterization opportunity, policy variant, novel), ranked by estimated step share,
    /// and write or check the Markdown report (kernels/circuits/venn/).
    Venn(Box<CircuitVennArgs>),
    /// Plan a checkpoint on a target device (kernels/DEVICES.toml): the fused plan its kernel
    /// class can run, the gap report against that class ranked by the device's roofline,
    /// decode and prefill estimates, and the memory fit. `--matrix` writes or checks every
    /// report of the roadmap matrix instead.
    Plan(Box<CircuitHwArgs>),
    /// The memory a checkpoint needs on a device at a serve's settings: per node (weights and
    /// their load-time copies, activations, workspace, state), per state and cache (KV, SSM,
    /// snapshots, speculative caches) as functions of ISL, OSL, concurrency and slots, the total
    /// against the util budget, and the largest concurrency and prompt that fit.
    Memory(Box<CircuitMemoryArgs>),
    /// List the precision each node needs: its input formats, the steps inside it (activation
    /// and weight preparation, multiply operands, accumulation, scales, compute, cache and state
    /// precision) and its output formats, as the plan on a device requires them and its kernels
    /// declare them (KERNEL_FAMILIES.toml `pipeline`).
    Precision(Box<CircuitPrecisionArgs>),
}

/// `met circuit memory` options. Serve settings come from `--recipe` (`recipes/<id>.yaml`) and the
/// `met serve` flags after `--`, parsed exactly as `met serve` parses them.
#[derive(clap::Args, Debug, Clone)]
pub struct CircuitMemoryArgs {
    /// The checkpoint: an id (org/name) or a checkpoint directory.
    #[arg(long)]
    pub checkpoint: String,
    /// Target device id from kernels/DEVICES.toml.
    #[arg(long)]
    pub hardware: String,
    /// A recipe (`family/stem`, e.g. qwen3.6/qwen3.6-35b-a3b-nvfp4) whose serve settings apply.
    #[arg(long)]
    pub recipe: Option<String>,
    /// Prompt tokens per sequence.
    #[arg(long)]
    pub isl: u64,
    /// Generated tokens per sequence.
    #[arg(long)]
    pub osl: u64,
    /// Sequences in flight.
    #[arg(long)]
    pub concurrency: u64,
    /// Sequence slots the serve is built with: a count, or `auto` for exactly the concurrency.
    /// Default: the serve's `--max-batch-size`.
    #[arg(long)]
    pub slots: Option<String>,
    /// Hidden rows the drafter-prefill capture holds (the serve log's "MTP drafter context"
    /// line); 0 when the serve does not capture.
    #[arg(long, default_value_t = 0)]
    pub capture_rows: u64,
    /// Model a prompt-lookup n-gram index per sequence (host memory).
    #[arg(long)]
    pub prompt_lookup: bool,
    /// Model a token tree (planned DFlash2) of this many nodes per verify slot: its attention
    /// mask and drafted ids, every node's recurrent state (a uniform K = nodes verify) and the
    /// nodes' KV.
    #[arg(long)]
    pub tree_nodes: Option<u64>,
    /// Print every node instead of one row per block-template node.
    #[arg(long)]
    pub per_node: bool,
    /// Print JSON instead of the tables.
    #[arg(long)]
    pub json: bool,
    /// Fetch config.json / hf_quant_config.json from huggingface.co when the checkpoint is not
    /// local.
    #[arg(long)]
    pub allow_network: bool,
    /// Repository root; by default the nearest directory above the working directory that has
    /// kernels/circuits/INSTANCES.toml.
    #[arg(long)]
    pub root: Option<std::path::PathBuf>,
    /// `met serve` flags, after `--` (e.g. `-- --max-batch-size 16 --kv-cache-dtype fp8`).
    #[arg(last = true)]
    pub serve: Vec<String>,
}

/// `met circuit precision` options.
#[derive(clap::Args, Debug, Clone)]
pub struct CircuitPrecisionArgs {
    /// The model: a checkpoint id (org/name), a checkpoint directory, or a recipe id.
    #[arg(long)]
    pub checkpoint: String,
    /// Formats served: the checkpoint's declared ones, or a recipe's pinned ones.
    #[arg(long, value_enum, default_value_t = CircuitPrecision::Declared)]
    pub precision: CircuitPrecision,
    /// Node ids to list, as a glob (`*` matches any run, dots included), e.g. `l3.moe_ffn.*`;
    /// every planned node when absent.
    #[arg(long)]
    pub node: Option<String>,
    /// Target device id from kernels/DEVICES.toml whose class plans the model.
    #[arg(long, default_value = "gb10")]
    pub hardware: String,
    /// The forward to plan.
    #[arg(long, value_enum, default_value_t = CircuitMode::Decode)]
    pub mode: CircuitMode,
    /// Padded rows (required for multi_seq and verify).
    #[arg(long)]
    pub rows: Option<u64>,
    /// Fetch config.json / hf_quant_config.json from huggingface.co when the checkpoint is not
    /// in the local cache.
    #[arg(long)]
    pub allow_network: bool,
    /// Repository root; by default the nearest directory above the working directory that has
    /// kernels/circuits/INSTANCES.toml.
    #[arg(long)]
    pub root: Option<std::path::PathBuf>,
}

/// `met circuit plan` options.
#[derive(clap::Args, Debug, Clone)]
pub struct CircuitHwArgs {
    /// The model: a checkpoint id (org/name), a checkpoint directory, or a recipe id.
    #[arg(long, required_unless_present = "matrix")]
    pub checkpoint: Option<String>,
    /// Target device id from kernels/DEVICES.toml (h100-sxm, h200-sxm, b200, gb300, gb10, ...).
    #[arg(long, required_unless_present = "matrix")]
    pub hardware: Option<String>,
    /// Formats served: the recipe's pinned formats or the checkpoint's declared ones.
    #[arg(long, value_enum, required_unless_present = "matrix")]
    pub precision: Option<CircuitPrecision>,
    /// What to print: the Markdown report, or one fused plan as stable text.
    #[arg(long, value_enum, default_value_t = CircuitHwFormat::Report)]
    pub format: CircuitHwFormat,
    /// With `--format plan`: the forward.
    #[arg(long, value_enum, default_value_t = CircuitMode::Decode)]
    pub mode: CircuitMode,
    /// With `--format plan`: padded rows (required for multi_seq and verify).
    #[arg(long)]
    pub rows: Option<u64>,
    /// Write the output here (relative to the repository root) instead of stdout.
    #[arg(long, conflicts_with = "matrix")]
    pub out: Option<String>,
    /// Write every report of the roadmap matrix into this directory (relative to the root).
    #[arg(long)]
    pub matrix: Option<String>,
    /// With `--matrix`: also write the summary table to this path.
    #[arg(long, requires = "matrix")]
    pub summary: Option<std::path::PathBuf>,
    /// Compare with the file(s) on disk instead of writing; a stale or missing one is an error.
    #[arg(long)]
    pub check: bool,
    /// Fetch config.json / hf_quant_config.json from huggingface.co when the checkpoint is not
    /// in the local cache (`curl -sfL .../resolve/main/<file>`).
    #[arg(long)]
    pub allow_network: bool,
    /// Repository root; by default the nearest directory above the working directory that has
    /// kernels/circuits/INSTANCES.toml.
    #[arg(long)]
    pub root: Option<std::path::PathBuf>,
}

/// `met circuit plan --precision`.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum CircuitPrecision {
    /// The recipe's pinned formats (the golden plans').
    #[value(name = "recipe")]
    Recipe,
    /// The checkpoint's declared formats.
    #[value(name = "declared")]
    Declared,
}

/// `met circuit plan --format`.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum CircuitHwFormat {
    /// The Markdown report.
    #[value(name = "report")]
    Report,
    /// One fused plan (`--mode`, `--rows`) as the stable text of `met circuit show`.
    #[value(name = "plan")]
    Plan,
}

/// `met circuit diff` options.
#[derive(clap::Args, Debug)]
pub struct CircuitDiffArgs {
    /// Decode steps compared per prompt.
    #[arg(long)]
    pub steps: usize,
    /// Prompts: synthetic token sequences from a fixed generator, each a different length.
    #[arg(long)]
    pub prompts: usize,
    /// Where to write the JSON report.
    #[arg(long)]
    pub out: std::path::PathBuf,
    /// Batch widths (comma-separated): diff multi-sequence decode at each width instead of
    /// single-sequence decode. `--prompts` is unused then; a width of `n` decodes `n` prompts.
    #[arg(long, value_delimiter = ',')]
    pub batch: Vec<usize>,
    /// With --batch: hold a spare sequence between rows n/2 - 1 and n/2, so the rows' state
    /// slots are not contiguous and each step takes the batched GDN arm's per-sequence fallback
    /// (the circuit's `gdn_state_slots_fragmented` route).
    #[arg(long)]
    pub fragment_slots: bool,
    /// MTP verify widths K (comma-separated, 2..=4): diff the single-sequence verify at each
    /// K instead. Needs a serve with speculative decoding (the rollback slots).
    #[arg(long, value_delimiter = ',')]
    pub verify: Vec<usize>,
    /// With --verify: the drafts come from the MTP draft head, as the speculative loop runs it,
    /// and every run also runs the draft head and compares its logits and drafts. With
    /// --verify-batch: after each step every run also runs the batched propose over the batch
    /// (the drafts still come from the greedy continuations) and compares its last logits,
    /// drafts and confidences.
    #[arg(long)]
    pub mtp: bool,
    /// Batched MTP verify widths `n` (comma-separated, sequences per verify): diff the batched
    /// verify with write-on-accept, as the scheduler runs it, instead. Needs a serve with
    /// speculative decoding and `--max-batch-size` at least the widest `n`.
    #[arg(long, value_delimiter = ',', requires = "verify_batch_ks")]
    pub verify_batch: Vec<usize>,
    /// With --verify-batch: the verify widths K (comma-separated, 2..=4). Each `n` runs every K
    /// uniform, then one ragged mix of them deepest first.
    #[arg(long, value_delimiter = ',', requires = "verify_batch")]
    pub verify_batch_ks: Vec<usize>,
    /// The serve the model is built with. `--forward` is ignored: the diff runs every forward.
    #[command(flatten)]
    pub serve: ServeArgs,
}

/// `met circuit venn` options.
#[derive(clap::Args, Debug, Clone)]
pub struct CircuitVennArgs {
    /// The model being added: a recipe id, checkpoint id or arch from
    /// kernels/circuits/INSTANCES.toml, or a checkpoint directory (its config.json and
    /// hf_quant_config.json are checked against the instance).
    #[arg(long)]
    pub target: String,
    /// Supported recipes (or checkpoint ids, or arches) to compare with, comma-separated.
    #[arg(long, value_delimiter = ',', required = true)]
    pub against: Vec<String>,
    /// Forwards to classify, comma-separated.
    #[arg(long, value_delimiter = ',', value_enum, default_values_t = [CircuitMode::Decode, CircuitMode::MultiSeq, CircuitMode::Verify, CircuitMode::Draft])]
    pub mode: Vec<CircuitMode>,
    /// Concurrency rungs: multi_seq plans every one above 1; decode and draft plan one row.
    #[arg(long, value_delimiter = ',', default_values_t = [1u64, 16, 128])]
    pub rows: Vec<u64>,
    /// MTP verify widths K (1 + drafts), comma-separated.
    #[arg(long, value_delimiter = ',', default_values_t = [2u64])]
    pub verify_rows: Vec<u64>,
    /// Report path, relative to the repository root
    /// (kernels/circuits/venn/TARGET-vs-ARCH.md).
    #[arg(long)]
    pub out: String,
    /// Verify the report at --out is what this command produces now; write nothing.
    #[arg(long)]
    pub check: bool,
    /// Repository root; by default the nearest directory above the working directory that has
    /// kernels/circuits/INSTANCES.toml.
    #[arg(long)]
    pub root: Option<std::path::PathBuf>,
}

/// Which plan to show.
#[derive(clap::Args, Debug, Clone)]
pub struct CircuitPlanArgs {
    /// Recipe id, e.g. qwen3.8/qwen3.8-27b-nvfp4-unsloth.
    #[arg(long)]
    pub recipe: String,
    /// The forward to plan.
    #[arg(long, value_enum, default_value_t = CircuitMode::Decode)]
    pub mode: CircuitMode,
    /// Padded rows: the batch rung for multi_seq, K for verify. Required for those two
    /// modes; decode and draft plan one row.
    #[arg(long)]
    pub rows: Option<u64>,
}

/// `met circuit display` options.
#[derive(clap::Args, Debug, Clone)]
pub struct CircuitDisplayArgs {
    #[command(flatten)]
    pub plan: CircuitPlanArgs,
    /// Expand one layer, with its module bindings.
    #[arg(long, conflicts_with = "all_layers")]
    pub layer: Option<usize>,
    /// Draw every layer instead of one diagram per distinct layer plan.
    #[arg(long)]
    pub all_layers: bool,
    /// Draw with 7-bit ASCII only.
    #[arg(long)]
    pub ascii: bool,
    /// When to colour: auto colours a terminal only; NO_COLOR always wins.
    #[arg(long, value_enum, default_value_t = ColorChoice::Auto)]
    pub color: ColorChoice,
    /// Draw the plan for this device of kernels/DEVICES.toml (its class's rules and kernels;
    /// ops no rule covers there are drawn as `novel` groups) instead of the recipe's own
    /// hardware. Reads the repository from the working directory.
    #[arg(long)]
    pub hardware: Option<String>,
}

/// A forward `met circuit` can plan.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum CircuitMode {
    /// One sequence, one row.
    #[value(name = "decode")]
    Decode,
    /// Many sequences, one row each, padded to the batch ladder.
    #[value(name = "multi_seq")]
    MultiSeq,
    /// K draft rows of one sequence.
    #[value(name = "verify")]
    Verify,
    /// The MTP draft head.
    #[value(name = "draft")]
    Draft,
}

/// `--color`.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorChoice {
    /// Colour when stdout is a terminal.
    Auto,
    /// Colour even when piped.
    Always,
    /// Never colour.
    Never,
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
