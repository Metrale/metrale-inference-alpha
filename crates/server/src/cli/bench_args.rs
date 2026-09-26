// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Arguments for `met benchmark` (alias `bench`).
//!
//! Owner: server CLI (`met benchmark`).
//! The `///` comments on clap items below are the `--help` text and carry no date.
//! Invariants: none beyond the types.

/// 2026-09-26: `met benchmark <command>`, or `--pull-request-gate-check` on its own,
/// which needs no subcommand.
#[derive(clap::Args, Debug)]
#[command(arg_required_else_help = true)]
pub struct BenchmarkArgs {
    /// Check the committed `.benchmarks/` records for this commit: every
    /// required gate must have a passing record. Prints what is missing or
    /// failing and exits non-zero when the branch is not fully gated.
    /// Runs without a subcommand (and without an endpoint).
    #[arg(long = "pull-request-gate-check")]
    pub pull_request_gate_check: bool,

    /// PR number, for the advisory intent half of `--pull-request-gate-check`.
    ///
    /// The journey ledger is keyed by PR (`governance/pr-<n>.jsonl`), and the
    /// gate otherwise has only a sha, so without this the classified intent
    /// cannot be found at all.
    ///
    /// There is no default: guessing a PR number would attribute another PR's
    /// classification to this one. Absent means `NotRequested`, and the
    /// verdict is unchanged either way: `gate::exit_code` takes only the
    /// verdicts.
    ///
    /// Without `--pull-request-gate-check` it is refused
    /// (`BenchmarkArgs::reject_orphan_pr`).
    #[arg(long)]
    pub pr: Option<u64>,
    #[command(subcommand)]
    pub command: Option<BenchmarkCommand>,
}

impl BenchmarkArgs {
    /// 2026-09-26: Whether this invocation is `certify --json`, whose stdout carries
    /// only JSON event lines; `main` sends the log to stderr for it.
    pub fn json_stdout(&self) -> bool {
        matches!(&self.command, Some(BenchmarkCommand::Certify(c)) if c.json)
    }

    /// 2026-09-26: Refuse `--pr` without `--pull-request-gate-check`: `--pr` only keys
    /// the gate check's advisory intent lookup. `bench_run::dispatch` calls this
    /// before it unwraps the subcommand, so `met benchmark --pr N` alone gets
    /// this usage error. The `Err` text is the message shown.
    pub fn reject_orphan_pr(&self) -> Result<(), String> {
        if self.pr.is_some() && !self.pull_request_gate_check {
            return Err(
                "--pr keys the advisory intent lookup of --pull-request-gate-check and does \
                 nothing anywhere else; pass --pull-request-gate-check with it, or drop --pr."
                    .to_string(),
            );
        }
        Ok(())
    }
}

#[derive(clap::Subcommand, Debug)]
pub enum BenchmarkCommand {
    /// List the suite, or one benchmark's parameter schema.
    List(ListArgs),
    /// Run one benchmark against a served endpoint.
    Run(RunArgs),
    /// Show a benchmark group's aggregate over its committed shard records.
    ///
    /// GPU-free: it reads what is already in `.benchmarks/` and applies the
    /// same partition rule and aggregation the gate does, so an operator can
    /// see the group's number, and which shard is missing, without waiting for
    /// CI to say so.
    Aggregate(AggregateArgs),
    /// Past runs, from `~/.metrale/runs`.
    History(HistoryArgs),
    /// Stop the server a `run --pull-request-gate --serve-reuse` left running
    /// on this box, if any.
    ServeRelease,
    /// Run every required gate this commit still owes, and say whether the
    /// tree is certified when they are done. See `certify --help`.
    Certify(super::bench_certify::args::CertifyArgs),
    /// Render a shareable result card from a committed gate record.
    ///
    /// Separate from `run --output-image`: a card can be regenerated from any
    /// past record, with different attribution, without re-measuring. It reads
    /// a gate record because the record carries the hardware and the commit
    /// the number belongs to, which the card prints.
    Card(CardArgs),
}

#[derive(clap::Args, Debug)]
pub struct CardArgs {
    /// A benchmark ID (`decode-floor`) or a path to a record.
    ///
    /// An existing file is taken as the record. Otherwise the ID takes the
    /// record in `.benchmarks/<ID>/` whose file name sorts last, which is the
    /// newest by day.
    pub record: String,
    /// Where to write. A name becomes `./<name>.svg`; a value containing `/` or
    /// ending in `.svg` is a path and is taken literally.
    #[arg(long = "output-image", value_name = "NAME|PATH")]
    pub output_image: Option<String>,
    /// `author=Ada,handle=@ada,website=ada.dev`
    #[arg(long = "output-image-args", value_name = "K=V,...")]
    pub output_image_args: Option<String>,
}

#[derive(clap::Args, Debug)]
pub struct ListArgs {
    /// Benchmark id. Omit for the whole suite.
    pub id: Option<String>,
    #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
    pub format: OutputFormat,
}

#[derive(clap::Args, Debug)]
pub struct RunArgs {
    /// Benchmark id — `met benchmark list` prints them.
    pub id: String,
    /// The endpoint to drive.
    ///
    /// This does not start a server. Under `--pull-request-gate` the run serves
    /// the benchmark's own recipe on a port it picks, so the two conflict and
    /// passing both is rejected.
    #[arg(
        long,
        default_value = "http://127.0.0.1:8888",
        conflicts_with = "pull_request_gate"
    )]
    pub url: String,
    /// The `model` field sent in every request.
    ///
    /// Required rather than defaulted, because it is recorded with the run.
    /// Under `--pull-request-gate` the benchmark's recipe supplies it, and
    /// passing it is rejected rather than ignored.
    #[arg(
        long,
        required_unless_present = "pull_request_gate",
        conflicts_with = "pull_request_gate"
    )]
    pub model: Option<String>,

    /// Which box class the run is for, e.g. `gb10`.
    ///
    /// Under `--pull-request-gate` it picks the baseline entry when a
    /// benchmark has thresholds for more than one box class. With a single
    /// entry it is inferred; with several, omitting it is an error rather than
    /// a guess. On every run it also names the class whose temperature
    /// ceilings the hardware pre-check applies (the probed class when omitted).
    ///
    /// A value the baseline does not carry is refused. The refusal says
    /// whether it is a registered box class
    /// (`metrale_bench::hardware::ids::KNOWN_HARDWARE_IDS`) with no baseline
    /// yet, or an id that is not registered.
    #[arg(long)]
    pub hardware: Option<String>,
    /// Which model variant of the benchmark to run, as the checkpoint id its
    /// `BENCH.toml` entry names, e.g. `unsloth/Qwen3.8-27B-NVFP4`.
    ///
    /// Gate-only. Omitted, the run takes the one checkpoint the benchmark's
    /// baseline marks `default = true` (two defaults, or none, refuse to
    /// assemble at all). A checkpoint the baseline does not carry is an error
    /// naming what exists. Distinct from `--model`: `--model` names a request
    /// field against a server someone else started, while this selects which
    /// (thresholds, serve recipe) pair the gate provisions and is recorded as
    /// the record's `target_model`.
    ///
    /// Without `--pull-request-gate` it is refused
    /// (`RunArgs::reject_orphan_checkpoint`).
    #[arg(long)]
    pub checkpoint: Option<String>,
    /// Override one parameter, e.g. `--param osl=8`. Repeatable.
    ///
    /// Anything not overridden takes the schema default and is still recorded.
    #[arg(long = "param", value_name = "KEY=VALUE", value_parser = parse_kv)]
    pub params: Vec<(String, String)>,
    #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
    pub format: OutputFormat,
    /// How often to drain the run's channels, in milliseconds.
    #[arg(long, default_value_t = 250)]
    pub poll_ms: u64,
    /// Do not write the run to `~/.metrale/runs`.
    #[arg(long)]
    pub no_save: bool,
    /// Confirm a benchmark with side effects beyond load on the endpoint.
    ///
    /// Required for `agentic-webserver`, which executes model-authored shell.
    #[arg(long)]
    pub yes: bool,
    /// Print only the final report, not per-phase progress.
    #[arg(long)]
    pub quiet: bool,
    /// Exit 0 even when the gate verdict is FAIL.
    #[arg(long)]
    pub no_fail_on_verdict: bool,
    /// Do not ask the endpoint two known-answer questions before measuring.
    ///
    /// The probe only warns and never refuses to start, so this skips the two
    /// extra completions; it does not silence a veto.
    #[arg(long)]
    pub skip_coherence_probe: bool,
    /// Write this run as a gate record under the repo's `.benchmarks/<id>/`.
    ///
    /// The record carries the metrics, verdict, hardware fingerprint, the
    /// exact command and the current commit sha, so the branch itself can
    /// answer "did this pass" with no `~/.metrale` state.
    #[arg(long)]
    pub pull_request_gate: bool,
    /// Override one serve key from the benchmark's recipe, e.g.
    /// `--serve-override kv_cache_dtype=fp8`. Repeatable; the last value for a
    /// key wins.
    ///
    /// Distinct from `--param`, which sets the benchmark's own knobs
    /// (iterations, max_tokens). This one reaches the recipe that starts the
    /// server, so it is how you exercise a code path the recipe's pinned
    /// config never reaches.
    ///
    /// A key the recipe's `defaults` lists is replaced; any other key is added
    /// if it is a serve flag, and otherwise refused when the rendered command
    /// line is parsed.
    /// `port` is refused: the gate binds a port itself.
    ///
    /// Every override is written into the gate record (`serve_overrides`), so
    /// a record whose numbers came from a config other than its recipe's says
    /// so.
    #[arg(long = "serve-override", value_name = "KEY=VALUE")]
    pub serve_override: Vec<String>,
    /// Reuse the server an earlier `--serve-reuse` run left on this box, if it
    /// is the one this run would start (same binary, same recipe rendering,
    /// same checkpoint, `GET /serve-config`); otherwise start one as a
    /// separate process and leave it running for the next run. A campaign
    /// running several gates on one recipe pays for one model load instead of
    /// one per gate. `met benchmark serve-release` stops it.
    #[arg(long, requires = "pull_request_gate")]
    pub serve_reuse: bool,
    /// The process the leased server belongs to (the campaign driver). A
    /// lease whose owner is gone is released by the next run rather than
    /// kept warm for nobody. Default: this run.
    #[arg(long, value_name = "PID", requires = "serve_reuse")]
    pub serve_lease_owner: Option<u32>,
    /// Write a shareable result card beside the run.
    ///
    /// Takes a name (`my-run` -> `./my-run.svg`) or a path (`/tmp/x.svg`,
    /// `cards/run.svg`). A value containing `/` or ending in `.svg` is a path
    /// and is taken literally; anything else is a name.
    ///
    /// The card carries the model, quantization, recipe and hardware beside the
    /// number.
    #[arg(long = "output-image", value_name = "NAME|PATH")]
    pub output_image: Option<String>,

    /// Attribution for the card: `author=Ada Lovelace,handle=@ada,website=ada.dev`.
    ///
    /// Comma-separated `key=value`. Unknown keys are accepted and ignored, so a
    /// future card field does not break an old command line. Requires
    /// `--output-image`; without it the run is refused
    /// (`RunArgs::reject_orphan_image_args`).
    #[arg(long = "output-image-args", value_name = "K=V,...")]
    pub output_image_args: Option<String>,
}

impl RunArgs {
    /// 2026-09-26: Refuse `--output-image-args` without `--output-image`, and
    /// `--output-image-args` that `gate::card::parse_args` rejects. The `Err`
    /// text is the message shown.
    pub fn reject_orphan_image_args(&self) -> Result<(), String> {
        if self.output_image_args.is_some() && self.output_image.is_none() {
            return Err(
                "--output-image-args needs --output-image: there is no card to put them on"
                    .to_string(),
            );
        }
        if let Some(raw) = &self.output_image_args {
            metrale_bench::gate::card::parse_args(raw)
                .map_err(|e| format!("--output-image-args: {e}"))?;
        }
        Ok(())
    }

    /// 2026-09-26: Refuse `--checkpoint` without `--pull-request-gate`: outside the
    /// gate the serve config is whatever the operator started, so a variant
    /// selector would do nothing. The `Err` text is the message shown.
    pub fn reject_orphan_checkpoint(&self) -> Result<(), String> {
        if self.checkpoint.is_some() && !self.pull_request_gate {
            return Err(
                "--checkpoint selects a model variant for a GATE run, which serves that \
                 variant's own recipe; without --pull-request-gate the endpoint is whatever \
                 you started, so pass --model to name what it serves instead."
                    .to_string(),
            );
        }
        Ok(())
    }
}

#[derive(clap::Args, Debug)]
pub struct HistoryArgs {
    /// Restrict to one benchmark id.
    #[arg(long)]
    pub id: Option<String>,
    /// Print the whole record for one run id.
    #[arg(long)]
    pub run: Option<String>,
    #[arg(long, default_value_t = 20)]
    pub limit: usize,
    #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
    pub format: OutputFormat,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum OutputFormat {
    Text,
    Json,
}

/// 2026-09-26: Split `KEY=VALUE` on the first `=` only, so a value may itself contain
/// `=`. An empty key is refused.
fn parse_kv(s: &str) -> Result<(String, String), String> {
    match s.split_once('=') {
        Some((k, v)) if !k.is_empty() => Ok((k.to_string(), v.to_string())),
        _ => Err(format!(
            "expected KEY=VALUE, got {s:?} — e.g. --param osl=8 or --param isls=128,512"
        )),
    }
}

#[cfg(test)]
#[path = "bench_args_tests.rs"]
mod tests;

/// 2026-09-26: `met benchmark aggregate <group>`.
#[derive(clap::Args, Debug)]
pub struct AggregateArgs {
    /// The group id, e.g. `bfcl-subset`.
    pub id: String,
    /// The commit the records must cover. Defaults to HEAD.
    #[arg(long)]
    pub sha: Option<String>,
    #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
    pub format: OutputFormat,
}
