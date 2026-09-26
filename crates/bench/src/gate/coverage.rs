// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Which changed paths invalidate which gate's records.
//!
//! Owner: bench gate.
//! Invariants:
//! - A path under a [`BOUNDARY_FILES`] entry invalidates every gate; [`invalidates`] asks that
//!   first, so no exclusion can exempt the files that define the rules.
//! - Any other path under [`PERF_PATHS`] invalidates a gate unless it is a registered
//!   test-only module, a `kernels/**/BENCH.toml`, or under one of that gate's [`Exclusion`]s.
//! - [`invalidated_by`] and [`promotion_debt`] read only their paths and the constants here.
//!
//! Gates subtract with exclusions rather than claim what they cover, so a path nobody
//! classified re-runs a gate instead of escaping every gate.

/// 2026-09-26: A path prefix that does not invalidate one gate, with the reason it cannot move
/// that gate's numbers (`every_exclusion_states_why` rejects an empty or short reason).
#[derive(Debug, Clone, Copy)]
pub struct Exclusion {
    pub prefix: &'static str,
    pub rationale: &'static str,
}

/// 2026-09-26: One gate id and the exclusions that do not invalidate it.
#[derive(Debug, Clone, Copy)]
pub struct GateCoverage {
    pub id: &'static str,
    pub excludes: &'static [Exclusion],
}

/// 2026-09-26: Paths whose contents can change what the engine computes. `.benchmarks` is not
/// listed because the records are the verdict, not its subject.
pub const PERF_PATHS: [&str; 7] = [
    "crates",
    "kernels",
    "Cargo.toml",
    "Cargo.lock",
    "vendor",
    "jinja-templates",
    "rust-toolchain.toml",
];

/// 2026-09-26: Files that define the boundary or decide a verdict, so a change to any of them
/// invalidates every gate. Were one excludable, a PR changing the rules would be judged by
/// its own new rules.
///
/// Every other file under `crates/bench/src/gate` is in `GATE_MACHINERY_FILES` and falls under
/// every gate's `GATE_MACHINERY` exclusion. `every_verdict_symbol_is_defined_inside_the_boundary`
/// fails if a verdict function it names is defined in a gate file outside this list.
pub const BOUNDARY_FILES: [&str; 14] = [
    "crates/bench/src/gate/coverage.rs",
    // 2026-09-26: `required_for`, `union`, `intent_only`: what the intent half adds to the
    // path-derived floor.
    "crates/bench/src/gate/required.rs",
    // 2026-09-26: The intent half's input (its `_benches` lists). It is outside `PERF_PATHS`,
    // and still invalidates every gate because `invalidates` checks this list before
    // `on_boundary` (`the_taxonomy_and_the_union_are_on_the_boundary`). The cost is a full
    // re-certification per taxonomy edit; without the entry, deleting a `_benches` line
    // would shrink coverage and invalidate nothing.
    ".github/pr-taxonomy.json",
    // 2026-09-26: `record_covers`, `record_standing`, `check_one`: whether a record stands
    // against the changed paths and passes.
    "crates/bench/src/gate/check.rs",
    // 2026-09-26: `invalidating_paths`: which changed paths invalidate a record.
    "crates/bench/src/gate/check_paths.rs",
    // 2026-09-26: `check_group`: one verdict from a group's shard records (a complete
    // partition at one commit, no shard with transport errors).
    "crates/bench/src/gate/check_group.rs",
    // 2026-09-26: `check_record`, `compare`: whether a record's numbers pass.
    "crates/bench/src/gate/scoring.rs",
    // 2026-09-26: `excuses`, `changed_targets`: which invalidating paths the closure hash
    // forgives.
    "crates/bench/src/gate/closure.rs",
    // 2026-09-26: `sources`, `configs`, `affected`: which targets a kernel edit reaches, the
    // input to `excuses`.
    "crates/bench/src/gate/taxon.rs",
    // 2026-09-26: `baseline_for`: which thresholds a record is judged against.
    "crates/bench/src/gate/bench.rs",
    // 2026-09-26: `check`: whether a set of added records hangs together (every record stands
    // at head; Speed-class records agree on signer or box).
    "crates/bench/src/gate/agreement.rs",
    // 2026-09-26: `CLOSED_KEYS`, `missing_pins`: what `--hermetic` closes. `bench` refuses an
    // entry that pins `hermetic=true` without these keys, and `check_record` matches a
    // record's serve overrides against the entry's pins in both directions.
    "crates/bench/src/gate/hermetic.rs",
    // 2026-09-26: `excused`: whether an invalidating path is excused. A PR widening the table
    // would otherwise excuse itself.
    "crates/bench/src/gate/amnesty.rs",
    // 2026-09-26: `select_partition`, `held_by`: whether a set of shard records forms a
    // complete partition at one commit.
    "crates/bench/src/gate/group.rs",
];

/// 2026-09-26: Gate sources reviewed and found not to decide a verdict. Every non-test `.rs`
/// file directly under `src/gate` is in exactly one of this list and `BOUNDARY_FILES`;
/// `gate_sources_are_all_classified` fails on a file in neither or both, so a new file must
/// be classified before the tests pass. The symbol walk in
/// `every_verdict_symbol_is_defined_inside_the_boundary` cannot catch a new verdict function,
/// because it only knows the names it lists.
#[cfg(test)]
pub(super) fn gate_machinery_files() -> &'static [&'static str] {
    GATE_MACHINERY_FILES
}

#[cfg(test)]
const GATE_MACHINERY_FILES: &[&str] = &[
    // 2026-09-26: Record reading and paths: where a record is read from, not whether it
    // passes; the record is still judged by `scoring.rs`.
    "crates/bench/src/gate/record.rs",
    "crates/bench/src/gate/record_path.rs",
    // 2026-09-26: Where a record is written; a failing record is kept and the newcomer written
    // beside it. `records_newest_first` still orders what it leaves, and `scoring.rs` judges.
    "crates/bench/src/gate/record_write.rs",
    // 2026-09-26: What a record discloses about its serve (`serve_resolved`). `check_record`
    // never reads it (`serve_resolved_never_reaches_check_record`).
    "crates/bench/src/gate/record_serve.rs",
    // 2026-09-26: A record's disclosed environment (`perf_env`, `serve_env`) and its one-line
    // summary; `scoring.rs` reads neither.
    "crates/bench/src/gate/record_env.rs",
    "crates/bench/src/gate/record_summary.rs",
    // 2026-09-26: Rendering and reporting only.
    "crates/bench/src/gate/card.rs",
    "crates/bench/src/gate/check_fmt.rs",
    "crates/bench/src/gate/telemetry.rs",
    "crates/bench/src/gate/telemetry_order.rs",
    // 2026-09-26: Owners for the telemetry table; only `telemetry.rs` calls it.
    "crates/bench/src/gate/codeowners.rs",
    // 2026-09-26: Parses `.github/pr-taxonomy.json` (a boundary file); `required.rs` decides
    // what the intent half adds.
    "crates/bench/src/gate/pr_taxonomy.rs",
    // 2026-09-26: Signature minting, the signer registry and `verify_record`, whose error
    // `check_one` and `check_group` report as a failure.
    "crates/bench/src/gate/signing.rs",
    // 2026-09-26: Test-only baseline fixtures (`#[cfg(test)]` in mod.rs).
    "crates/bench/src/gate/fixture_baseline.rs",
    "crates/bench/src/gate/mod.rs",
];

/// 2026-09-26: File names under `kernels/` that the gate reads and nothing compiles.
///
/// `BENCH.toml` holds the thresholds records are judged against; if editing it invalidated
/// every record, ratcheting a bar would discard the record that justified it. Exempting it
/// is safe because `taxon::configs` never returns it, so no closure hash contains it
/// (`bench_toml_is_not_a_closure_input`). Matched on the exact file name, so a path that
/// merely ends with the same characters is unaffected.
const NON_COMPILED_KERNEL_FILES: [&str; 1] = ["BENCH.toml"];

/// 2026-09-26: A Rust source whose only module edge is a `#[cfg(test)]` declaration in
/// `parent`, so it is absent from release builds and invalidates no gate.
///
/// An exact registry, not a naming rule: `registered_test_modules_are_proven_cfg_test_only`
/// fails if the declaration loses its guard or another `#[path]`/`include!` edge reaches the
/// file, and an unregistered test file invalidates like any other path.
#[derive(Debug, Clone, Copy)]
pub struct TestOnlyRustModule {
    pub path: &'static str,
    pub parent: &'static str,
    pub name: &'static str,
    pub declared_path: Option<&'static str>,
}

pub const TEST_ONLY_RUST_MODULES: &[TestOnlyRustModule] = &[
    TestOnlyRustModule {
        path: "crates/model-arch/src/seq_state_reserve_tests.rs",
        parent: "crates/model-arch/src/seq_state_reserve.rs",
        name: "tests",
        declared_path: Some("seq_state_reserve_tests.rs"),
    },
    TestOnlyRustModule {
        path: "crates/model-layers/src/layer/release_contract_tests.rs",
        parent: "crates/model-layers/src/layer.rs",
        name: "release_contract_tests",
        declared_path: Some("layer/release_contract_tests.rs"),
    },
    TestOnlyRustModule {
        path: "crates/server/src/main_modules/serve_phases/preflight/per_sequence_state_tests.rs",
        parent: "crates/server/src/main_modules/serve_phases/preflight/per_sequence_state.rs",
        name: "tests",
        declared_path: Some("per_sequence_state_tests.rs"),
    },
    TestOnlyRustModule {
        path: "crates/config/src/tests.rs",
        parent: "crates/config/src/lib.rs",
        name: "tests",
        declared_path: None,
    },
    TestOnlyRustModule {
        path: "crates/config/src/gguf/tests.rs",
        parent: "crates/config/src/gguf.rs",
        name: "tests",
        declared_path: None,
    },
    TestOnlyRustModule {
        path: "crates/config/src/parsers/lora_tests.rs",
        parent: "crates/config/src/parsers/lora.rs",
        name: "tests",
        declared_path: Some("lora_tests.rs"),
    },
    TestOnlyRustModule {
        path: "crates/bench/src/benchmarks/concurrency_tests.rs",
        parent: "crates/bench/src/benchmarks/concurrency.rs",
        name: "concurrency_tests",
        declared_path: Some("concurrency_tests.rs"),
    },
    TestOnlyRustModule {
        path: "crates/bench/src/benchmarks/concurrency_verdict_tests.rs",
        parent: "crates/bench/src/benchmarks/concurrency.rs",
        name: "concurrency_verdict_tests",
        declared_path: Some("concurrency_verdict_tests.rs"),
    },
    TestOnlyRustModule {
        path: "crates/bench/src/benchmarks/concurrency_moe_tests.rs",
        parent: "crates/bench/src/benchmarks/concurrency.rs",
        name: "concurrency_moe_tests",
        declared_path: Some("concurrency_moe_tests.rs"),
    },
];

fn is_test_only_rust_module(path: &str) -> bool {
    TEST_ONLY_RUST_MODULES
        .iter()
        .any(|entry| path == entry.path)
}

/// 2026-09-26: Whether `path` is under `kernels/` and named in `NON_COMPILED_KERNEL_FILES`.
fn is_non_compiled_kernel_file(path: &str) -> bool {
    let Some(rest) = path.strip_prefix("kernels/") else {
        return false;
    };
    rest.rsplit('/')
        .next()
        .is_some_and(|name| NON_COMPILED_KERNEL_FILES.contains(&name))
}

/// 2026-09-26: Gate machinery reads records and prints verdicts; it never runs a model, so it
/// cannot change an inference number. Its pass/fail logic is covered by `cargo test`. A file
/// in `BOUNDARY_FILES` still invalidates, because `invalidates` checks that list first.
const GATE_MACHINERY: Exclusion = Exclusion {
    prefix: "crates/bench/src/gate",
    rationale: "gate bookkeeping never runs a model; its correctness is covered by cargo test",
};

/// 2026-09-26: Excludes another benchmark's driver from a gate: a change to the BFCL driver can
/// move BFCL numbers but not what the TTFT probe measures.
///
/// This holds only while the excluded driver is not imported by the gate's own driver.
/// `benchmark_drivers_do_not_import_each_other` (coverage_driver_tests.rs) checks that for
/// the drivers it lists; `kat_equality` and `scheduler_equivalence` import `bfcl`, which is
/// why neither gate excludes it.
const fn other_driver(prefix: &'static str, mine: &'static str) -> Exclusion {
    Exclusion {
        prefix,
        rationale: mine,
    }
}

/// 2026-09-26: The concurrency driver is flat files, not a directory, so each excluded file is
/// named exactly: `under` matches per path component, so a `benchmarks/concurrency` prefix
/// matches no file, and excluding `benchmarks` would also hide shared code such as `stats.rs`.
const fn concurrency_driver(prefix: &'static str, rationale: &'static str) -> Exclusion {
    other_driver(prefix, rationale)
}

const TTFT_EXCLUDES: &[Exclusion] = &[
    GATE_MACHINERY,
    other_driver(
        "crates/bench/src/benchmarks/bfcl",
        "the BFCL driver cannot change what a first-token latency probe measures",
    ),
    other_driver(
        "crates/bench/src/benchmarks/agentic",
        "the agentic driver cannot change what a first-token latency probe measures",
    ),
    other_driver(
        "crates/bench/src/benchmarks/contamination",
        "the contamination driver cannot change what a first-token latency probe measures",
    ),
    other_driver(
        "crates/bench/src/benchmarks/ssm_poison",
        "the SSM poison driver cannot change what a first-token latency probe measures",
    ),
    concurrency_driver(
        "crates/bench/src/benchmarks/concurrency.rs",
        "the concurrency request planner cannot change what a first-token latency probe measures",
    ),
    concurrency_driver(
        "crates/bench/src/benchmarks/concurrency_verdict.rs",
        "the concurrency verdict cannot change what a first-token latency probe measures",
    ),
    concurrency_driver(
        "crates/bench/src/benchmarks/concurrency_descriptors.rs",
        "the concurrency request planner cannot change what a first-token latency probe measures",
    ),
    concurrency_driver(
        "crates/bench/src/benchmarks/concurrency_prompt.rs",
        "the concurrency request planner cannot change what a first-token latency probe measures",
    ),
    concurrency_driver(
        "crates/bench/src/benchmarks/concurrency_cache.rs",
        "the concurrency request planner cannot change what a first-token latency probe measures",
    ),
    concurrency_driver(
        "crates/bench/src/benchmarks/concurrency_cell.rs",
        "the concurrency request planner cannot change what a first-token latency probe measures",
    ),
    concurrency_driver(
        "crates/bench/src/benchmarks/concurrency_report.rs",
        "the concurrency request planner cannot change what a first-token latency probe measures",
    ),
    other_driver(
        "crates/bench/src/benchmarks/kat_equality",
        "the equality driver cannot change what a first-token latency probe measures",
    ),
    other_driver(
        "crates/bench/src/benchmarks/scheduler_equivalence",
        "the equivalence driver cannot change what a first-token latency probe measures",
    ),
];

const BFCL_EXCLUDES: &[Exclusion] = &[
    GATE_MACHINERY,
    other_driver(
        "crates/bench/src/benchmarks/ttft",
        "the TTFT driver cannot change a tool-calling accuracy score",
    ),
    other_driver(
        "crates/bench/src/benchmarks/agentic",
        "the agentic driver cannot change a tool-calling accuracy score",
    ),
    other_driver(
        "crates/bench/src/benchmarks/contamination",
        "the contamination driver cannot change a tool-calling accuracy score",
    ),
    other_driver(
        "crates/bench/src/benchmarks/ssm_poison",
        "the SSM poison driver cannot change a tool-calling accuracy score",
    ),
    concurrency_driver(
        "crates/bench/src/benchmarks/concurrency.rs",
        "the concurrency request planner cannot change a tool-calling accuracy score",
    ),
    concurrency_driver(
        "crates/bench/src/benchmarks/concurrency_verdict.rs",
        "the concurrency verdict cannot change a tool-calling accuracy score",
    ),
    concurrency_driver(
        "crates/bench/src/benchmarks/concurrency_descriptors.rs",
        "the concurrency request planner cannot change a tool-calling accuracy score",
    ),
    concurrency_driver(
        "crates/bench/src/benchmarks/concurrency_prompt.rs",
        "the concurrency request planner cannot change a tool-calling accuracy score",
    ),
    concurrency_driver(
        "crates/bench/src/benchmarks/concurrency_cache.rs",
        "the concurrency request planner cannot change a tool-calling accuracy score",
    ),
    concurrency_driver(
        "crates/bench/src/benchmarks/concurrency_cell.rs",
        "the concurrency request planner cannot change a tool-calling accuracy score",
    ),
    concurrency_driver(
        "crates/bench/src/benchmarks/concurrency_report.rs",
        "the concurrency request planner cannot change a tool-calling accuracy score",
    ),
    other_driver(
        "crates/bench/src/benchmarks/kat_equality",
        "the equality driver cannot change a tool-calling accuracy score",
    ),
    other_driver(
        "crates/bench/src/benchmarks/scheduler_equivalence",
        "the equivalence driver cannot change a BFCL score",
    ),
];

const AGENTIC_EXCLUDES: &[Exclusion] = &[
    GATE_MACHINERY,
    other_driver(
        "crates/bench/src/benchmarks/ttft",
        "the TTFT driver cannot change whether the agent's webserver task succeeds",
    ),
    other_driver(
        "crates/bench/src/benchmarks/bfcl",
        "the BFCL driver cannot change whether the agent's webserver task succeeds",
    ),
    other_driver(
        "crates/bench/src/benchmarks/contamination",
        "the contamination driver cannot change whether the agent's webserver task succeeds",
    ),
    other_driver(
        "crates/bench/src/benchmarks/ssm_poison",
        "the SSM poison driver cannot change whether the agent's webserver task succeeds",
    ),
    concurrency_driver(
        "crates/bench/src/benchmarks/concurrency.rs",
        "the concurrency request planner cannot change whether the agent's webserver task succeeds",
    ),
    concurrency_driver(
        "crates/bench/src/benchmarks/concurrency_verdict.rs",
        "the concurrency verdict cannot change whether the agent's webserver task succeeds",
    ),
    concurrency_driver(
        "crates/bench/src/benchmarks/concurrency_descriptors.rs",
        "the concurrency request planner cannot change whether the agent's webserver task succeeds",
    ),
    concurrency_driver(
        "crates/bench/src/benchmarks/concurrency_prompt.rs",
        "the concurrency request planner cannot change whether the agent's webserver task succeeds",
    ),
    concurrency_driver(
        "crates/bench/src/benchmarks/concurrency_cache.rs",
        "the concurrency request planner cannot change whether the agent's webserver task succeeds",
    ),
    concurrency_driver(
        "crates/bench/src/benchmarks/concurrency_cell.rs",
        "the concurrency request planner cannot change whether the agent's webserver task succeeds",
    ),
    concurrency_driver(
        "crates/bench/src/benchmarks/concurrency_report.rs",
        "the concurrency request planner cannot change whether the agent's webserver task succeeds",
    ),
    other_driver(
        "crates/bench/src/benchmarks/kat_equality",
        "the equality driver cannot change whether the agent's webserver task succeeds",
    ),
    other_driver(
        "crates/bench/src/benchmarks/scheduler_equivalence",
        "the equivalence driver cannot change what the agentic harness measures",
    ),
];

/// 2026-09-26: The SSM state poisoning gate excludes gate bookkeeping and the drivers below;
/// its own driver directory is not excluded, so a change to the detector re-opens it. The
/// gate replays identical requests, and a driver that only issues requests cannot change
/// whether a replay returns identical bytes.
const SSM_POISON_EXCLUDES: &[Exclusion] = &[
    GATE_MACHINERY,
    other_driver(
        "crates/bench/src/benchmarks/ttft",
        "the TTFT driver cannot change whether an identical replay returns identical bytes",
    ),
    other_driver(
        "crates/bench/src/benchmarks/bfcl",
        "the BFCL driver cannot change whether an identical replay returns identical bytes",
    ),
    other_driver(
        "crates/bench/src/benchmarks/agentic",
        "the agentic driver cannot change whether an identical replay returns identical bytes",
    ),
    other_driver(
        "crates/bench/src/benchmarks/contamination",
        "the contamination driver cannot change whether an identical replay returns identical bytes",
    ),
    concurrency_driver(
        "crates/bench/src/benchmarks/concurrency.rs",
        "the concurrency request planner cannot change whether an identical replay returns identical bytes",
    ),
    concurrency_driver(
        "crates/bench/src/benchmarks/concurrency_verdict.rs",
        "the concurrency verdict cannot change whether an identical replay returns identical bytes",
    ),
    concurrency_driver(
        "crates/bench/src/benchmarks/concurrency_descriptors.rs",
        "the concurrency request planner cannot change whether an identical replay returns identical bytes",
    ),
    concurrency_driver(
        "crates/bench/src/benchmarks/concurrency_prompt.rs",
        "the concurrency request planner cannot change whether an identical replay returns identical bytes",
    ),
    concurrency_driver(
        "crates/bench/src/benchmarks/concurrency_cache.rs",
        "the concurrency request planner cannot change whether an identical replay returns identical bytes",
    ),
    concurrency_driver(
        "crates/bench/src/benchmarks/concurrency_cell.rs",
        "the concurrency request planner cannot change whether an identical replay returns identical bytes",
    ),
    concurrency_driver(
        "crates/bench/src/benchmarks/concurrency_report.rs",
        "the concurrency request planner cannot change whether an identical replay returns identical bytes",
    ),
    other_driver(
        "crates/bench/src/benchmarks/kat_equality",
        "the equality driver cannot change whether an identical replay returns identical bytes",
    ),
    other_driver(
        "crates/bench/src/benchmarks/scheduler_equivalence",
        "the equivalence driver cannot change whether a replay poisons SSM state",
    ),
];

/// 2026-09-26: The concurrency ladders measure serving latency and throughput, so every engine
/// path invalidates them; only gate bookkeeping and the drivers below, which issue requests
/// client-side, are excluded. The three concurrency gates share this list.
const CONCURRENCY_EXCLUDES: &[Exclusion] = &[
    GATE_MACHINERY,
    other_driver(
        "crates/bench/src/benchmarks/bfcl",
        "the BFCL driver cannot change the server's latency/throughput curve",
    ),
    other_driver(
        "crates/bench/src/benchmarks/agentic",
        "the agentic driver cannot change the server's latency/throughput curve",
    ),
    other_driver(
        "crates/bench/src/benchmarks/contamination",
        "the contamination driver cannot change the server's latency/throughput curve",
    ),
    other_driver(
        "crates/bench/src/benchmarks/ttft",
        "the TTFT driver issues single requests client-side; it cannot change how fast the \
         server answers a batch of 32",
    ),
    other_driver(
        "crates/bench/src/benchmarks/kat_equality",
        "the equality driver cannot change the server's latency/throughput curve",
    ),
    other_driver(
        "crates/bench/src/benchmarks/scheduler_equivalence",
        "the equivalence driver cannot change a concurrency ladder's numbers",
    ),
];

/// 2026-09-26: The decode-floor gate measures single-user decode throughput, so every engine path
/// invalidates it; only gate bookkeeping and the drivers below are excluded. Its own driver
/// directory is not excluded.
const DECODE_FLOOR_EXCLUDES: &[Exclusion] = &[
    GATE_MACHINERY,
    other_driver(
        "crates/bench/src/benchmarks/bfcl",
        "the BFCL driver cannot change the server's single-user decode rate",
    ),
    other_driver(
        "crates/bench/src/benchmarks/agentic",
        "the agentic driver cannot change the server's single-user decode rate",
    ),
    other_driver(
        "crates/bench/src/benchmarks/contamination",
        "the contamination driver cannot change the server's single-user decode rate",
    ),
    other_driver(
        "crates/bench/src/benchmarks/ttft",
        "the TTFT driver cannot change the server's single-user decode rate",
    ),
    other_driver(
        "crates/bench/src/benchmarks/ssm_poison",
        "the SSM poison driver cannot change the server's single-user decode rate",
    ),
    concurrency_driver(
        "crates/bench/src/benchmarks/concurrency.rs",
        "the concurrency request planner cannot change the server's single-user decode rate",
    ),
    concurrency_driver(
        "crates/bench/src/benchmarks/concurrency_verdict.rs",
        "the concurrency verdict cannot change the server's single-user decode rate",
    ),
    concurrency_driver(
        "crates/bench/src/benchmarks/concurrency_descriptors.rs",
        "the concurrency request planner cannot change the server's single-user decode rate",
    ),
    concurrency_driver(
        "crates/bench/src/benchmarks/concurrency_prompt.rs",
        "the concurrency request planner cannot change the server's single-user decode rate",
    ),
    concurrency_driver(
        "crates/bench/src/benchmarks/concurrency_cache.rs",
        "the concurrency request planner cannot change the server's single-user decode rate",
    ),
    concurrency_driver(
        "crates/bench/src/benchmarks/concurrency_cell.rs",
        "the concurrency request planner cannot change the server's single-user decode rate",
    ),
    concurrency_driver(
        "crates/bench/src/benchmarks/concurrency_report.rs",
        "the concurrency request planner cannot change the server's single-user decode rate",
    ),
    other_driver(
        "crates/bench/src/benchmarks/kat_equality",
        "the equality driver cannot change the server's single-user decode rate",
    ),
    other_driver(
        "crates/bench/src/benchmarks/scheduler_equivalence",
        "the equivalence driver cannot change the decode floor",
    ),
];

/// 2026-09-26: What the KAT-equality gate ignores. The BFCL driver is not excluded: this gate's
/// driver imports `bfcl::dataset` and `bfcl::draw`, which decide which samples it compares.
/// Its own directory is not excluded either.
const KAT_EQUALITY_EXCLUDES: &[Exclusion] = &[
    GATE_MACHINERY,
    other_driver(
        "crates/bench/src/benchmarks/ttft",
        "the TTFT driver cannot change whether a reply depends on what ran before it",
    ),
    other_driver(
        "crates/bench/src/benchmarks/agentic",
        "the agentic driver cannot change whether a reply depends on what ran before it",
    ),
    other_driver(
        "crates/bench/src/benchmarks/ssm_poison",
        "the SSM poison driver cannot change whether a reply depends on what ran before it",
    ),
    other_driver(
        "crates/bench/src/benchmarks/contamination",
        "the contamination driver cannot change whether a reply depends on what ran before it",
    ),
    concurrency_driver(
        "crates/bench/src/benchmarks/concurrency.rs",
        "the concurrency request planner cannot change whether a reply depends on what ran before it",
    ),
    concurrency_driver(
        "crates/bench/src/benchmarks/concurrency_verdict.rs",
        "the concurrency verdict cannot change whether a reply depends on what ran before it",
    ),
    concurrency_driver(
        "crates/bench/src/benchmarks/concurrency_descriptors.rs",
        "the concurrency request planner cannot change whether a reply depends on what ran before it",
    ),
    concurrency_driver(
        "crates/bench/src/benchmarks/concurrency_prompt.rs",
        "the concurrency request planner cannot change whether a reply depends on what ran before it",
    ),
    concurrency_driver(
        "crates/bench/src/benchmarks/concurrency_cache.rs",
        "the concurrency request planner cannot change whether a reply depends on what ran before it",
    ),
    concurrency_driver(
        "crates/bench/src/benchmarks/concurrency_cell.rs",
        "the concurrency request planner cannot change whether a reply depends on what ran before it",
    ),
    concurrency_driver(
        "crates/bench/src/benchmarks/concurrency_report.rs",
        "the concurrency request planner cannot change whether a reply depends on what ran before it",
    ),
    other_driver(
        "crates/bench/src/benchmarks/scheduler_equivalence",
        "the equivalence driver cannot change whether a reply depends on what ran before it",
    ),
];

/// 2026-09-26: What the scheduler-equivalence candidate ignores. Neither its own driver
/// directory nor `bfcl` (its driver imports `bfcl::dataset`) is excluded.
const SCHEDULER_EQUIVALENCE_EXCLUDES: &[Exclusion] = &[
    GATE_MACHINERY,
    other_driver(
        "crates/bench/src/benchmarks/ttft",
        "the TTFT driver cannot change whether two routers answer a sample alike",
    ),
    other_driver(
        "crates/bench/src/benchmarks/agentic",
        "the agentic driver cannot change whether two routers answer a sample alike",
    ),
    other_driver(
        "crates/bench/src/benchmarks/ssm_poison",
        "the SSM poison driver cannot change whether two routers answer a sample alike",
    ),
    other_driver(
        "crates/bench/src/benchmarks/contamination",
        "the contamination driver cannot change whether two routers answer a sample alike",
    ),
    other_driver(
        "crates/bench/src/benchmarks/kat_equality",
        "the order-independence driver cannot change whether two routers answer a sample alike",
    ),
    concurrency_driver(
        "crates/bench/src/benchmarks/concurrency.rs",
        "the concurrency request planner cannot change whether two routers answer a sample alike",
    ),
    concurrency_driver(
        "crates/bench/src/benchmarks/concurrency_verdict.rs",
        "the concurrency verdict cannot change whether two routers answer a sample alike",
    ),
    concurrency_driver(
        "crates/bench/src/benchmarks/concurrency_descriptors.rs",
        "the concurrency request planner cannot change whether two routers answer a sample alike",
    ),
    concurrency_driver(
        "crates/bench/src/benchmarks/concurrency_prompt.rs",
        "the concurrency request planner cannot change whether two routers answer a sample alike",
    ),
    concurrency_driver(
        "crates/bench/src/benchmarks/concurrency_cache.rs",
        "the concurrency request planner cannot change whether two routers answer a sample alike",
    ),
    concurrency_driver(
        "crates/bench/src/benchmarks/concurrency_cell.rs",
        "the concurrency request planner cannot change whether two routers answer a sample alike",
    ),
    concurrency_driver(
        "crates/bench/src/benchmarks/concurrency_report.rs",
        "the concurrency request planner cannot change whether two routers answer a sample alike",
    ),
];

/// 2026-09-26: What the cross-contamination candidate ignores. Its own driver directory is not
/// excluded, so a change to the detector re-opens it.
const CONTAMINATION_EXCLUDES: &[Exclusion] = &[
    GATE_MACHINERY,
    other_driver(
        "crates/bench/src/benchmarks/ttft",
        "the TTFT driver cannot change whether one request's state leaks into another's output",
    ),
    other_driver(
        "crates/bench/src/benchmarks/bfcl",
        "the BFCL driver cannot change whether one request's state leaks into another's output",
    ),
    other_driver(
        "crates/bench/src/benchmarks/agentic",
        "the agentic driver cannot change whether one request's state leaks into another's output",
    ),
    concurrency_driver(
        "crates/bench/src/benchmarks/concurrency.rs",
        "the concurrency request planner cannot change whether one request's state leaks into another",
    ),
    concurrency_driver(
        "crates/bench/src/benchmarks/concurrency_verdict.rs",
        "the concurrency verdict cannot change whether one request's state leaks into another",
    ),
    concurrency_driver(
        "crates/bench/src/benchmarks/concurrency_descriptors.rs",
        "the concurrency request planner cannot change whether one request's state leaks into another",
    ),
    concurrency_driver(
        "crates/bench/src/benchmarks/concurrency_prompt.rs",
        "the concurrency request planner cannot change whether one request's state leaks into another",
    ),
    concurrency_driver(
        "crates/bench/src/benchmarks/concurrency_cache.rs",
        "the concurrency request planner cannot change whether one request's state leaks into another",
    ),
    concurrency_driver(
        "crates/bench/src/benchmarks/concurrency_cell.rs",
        "the concurrency request planner cannot change whether one request's state leaks into another",
    ),
    concurrency_driver(
        "crates/bench/src/benchmarks/concurrency_report.rs",
        "the concurrency request planner cannot change whether one request's state leaks into another",
    ),
    other_driver(
        "crates/bench/src/benchmarks/kat_equality",
        "the equality driver cannot change whether one request's state leaks into \
         another's output",
    ),
    other_driver(
        "crates/bench/src/benchmarks/scheduler_equivalence",
        "the equivalence driver cannot change whether one request's state leaks into another's output",
    ),
];

/// 2026-09-26: What the vision and video fidelity gates ignore: gate bookkeeping and the drivers
/// below, none of which can change how an image is preprocessed or tokenized.
const VISION_EXCLUDES: &[Exclusion] = &[
    GATE_MACHINERY,
    other_driver(
        "crates/bench/src/benchmarks/ttft",
        "the TTFT driver cannot change how an image is patched or how many tokens it becomes",
    ),
    other_driver(
        "crates/bench/src/benchmarks/bfcl",
        "the BFCL driver cannot change how an image is patched or how many tokens it becomes",
    ),
    other_driver(
        "crates/bench/src/benchmarks/agentic",
        "the agentic driver cannot change how an image is patched or how many tokens it becomes",
    ),
    other_driver(
        "crates/bench/src/benchmarks/contamination",
        "the contamination driver cannot change how an image is patched or how many tokens it becomes",
    ),
    concurrency_driver(
        "crates/bench/src/benchmarks/concurrency.rs",
        "the concurrency request planner cannot change image preprocessing or encoder tokens",
    ),
    concurrency_driver(
        "crates/bench/src/benchmarks/concurrency_verdict.rs",
        "the concurrency verdict cannot change image preprocessing or encoder tokens",
    ),
    concurrency_driver(
        "crates/bench/src/benchmarks/concurrency_descriptors.rs",
        "the concurrency request planner cannot change image preprocessing or encoder tokens",
    ),
    concurrency_driver(
        "crates/bench/src/benchmarks/concurrency_prompt.rs",
        "the concurrency request planner cannot change image preprocessing or encoder tokens",
    ),
    concurrency_driver(
        "crates/bench/src/benchmarks/concurrency_cache.rs",
        "the concurrency request planner cannot change image preprocessing or encoder tokens",
    ),
    concurrency_driver(
        "crates/bench/src/benchmarks/concurrency_cell.rs",
        "the concurrency request planner cannot change image preprocessing or encoder tokens",
    ),
    concurrency_driver(
        "crates/bench/src/benchmarks/concurrency_report.rs",
        "the concurrency request planner cannot change image preprocessing or encoder tokens",
    ),
    other_driver(
        "crates/bench/src/benchmarks/kat_equality",
        "the equality driver cannot change how an image is patched or how many tokens \
         it becomes",
    ),
    other_driver(
        "crates/bench/src/benchmarks/scheduler_equivalence",
        "the equivalence driver cannot change how an image is patched or how many tokens it becomes",
    ),
];

/// 2026-09-26: The gates whose records must pass, and what each one ignores. `REQUIRED_GATES`
/// in mod.rs is built from these ids (`required_gates_is_derived_from_the_coverage_table`).
pub const REQUIRED: [GateCoverage; 13] = [
    GateCoverage {
        id: "agentic-webserver",
        excludes: AGENTIC_EXCLUDES,
    },
    // 2026-09-26: Coverage is per path, with no per-model dimension; which checkpoint a gate
    // measures comes from its `[[benchmarks]]` entries in BENCH.toml.
    GateCoverage {
        id: "vision-fidelity",
        excludes: VISION_EXCLUDES,
    },
    GateCoverage {
        id: "video-fidelity",
        excludes: VISION_EXCLUDES,
    },
    GateCoverage {
        id: "ttft-warm-gate",
        excludes: TTFT_EXCLUDES,
    },
    GateCoverage {
        id: "ttft-cold-gate",
        excludes: TTFT_EXCLUDES,
    },
    GateCoverage {
        id: "bfcl-subset",
        excludes: BFCL_EXCLUDES,
    },
    GateCoverage {
        id: "bfcl-subset-echolp",
        excludes: BFCL_EXCLUDES,
    },
    GateCoverage {
        id: "ssm-state-poisoning-gate",
        excludes: SSM_POISON_EXCLUDES,
    },
    GateCoverage {
        id: "decode-floor",
        excludes: DECODE_FLOOR_EXCLUDES,
    },
    GateCoverage {
        id: "concurrency-sweep",
        excludes: CONCURRENCY_EXCLUDES,
    },
    GateCoverage {
        id: "concurrency-sweep-dflash2",
        excludes: CONCURRENCY_EXCLUDES,
    },
    GateCoverage {
        id: "kat-equality-gate",
        excludes: KAT_EQUALITY_EXCLUDES,
    },
    GateCoverage {
        id: "concurrency-sweep-moe",
        excludes: CONCURRENCY_EXCLUDES,
    },
];

/// 2026-09-26: Gates not required yet whose invalidation is tracked as debt.
///
/// Each candidate carries a full [`GateCoverage`], and [`promotion_debt`] names the candidates
/// a set of changed paths invalidates; the telemetry renders them, so a merge that skipped
/// a candidate says so instead of reading as unaffected. A candidate must be a registered
/// benchmark and not in [`REQUIRED`] (`every_promotion_candidate_is_a_registered_benchmark`).
pub const PROMOTION_CANDIDATES: &[GateCoverage] = &[
    GateCoverage {
        id: "cross-contamination",
        excludes: CONTAMINATION_EXCLUDES,
    },
    // 2026-09-26: Certifies the asynchronous device router (`--scheduler-config async`); the
    // serve default is `sync`.
    GateCoverage {
        id: "scheduler-equivalence",
        excludes: SCHEDULER_EQUIVALENCE_EXCLUDES,
    },
];

/// 2026-09-26: Registered benchmarks that are not required gates, each with the reason
/// (`every_excusal_names_a_real_benchmark_and_a_reason`).
pub const NOT_REQUIRED: [(&str, &str); 6] = [
    (
        "quick-speed-bench",
        "a single-user speed probe with no thresholds and no baseline — a MEASUREMENT tool, \
         deliberately never a gate: the required gates already cost hours per PR, and \
         its warm-path numbers (primed prefix cache + SSM snapshot) are not regression evidence",
    ),
    (
        "bfcl-full",
        "the unsampled ~3600-sample draw; the two subset gates cover the same code at a \
         fraction of the GPU time, and a full run would dominate every PR",
    ),
    (
        "serve-matrix",
        "a multi-checkpoint survey used for release notes; it measures breadth, not regression",
    ),
    (
        "cross-contamination",
        "not required YET: a promotion candidate (see PROMOTION_CANDIDATES) run on release cuts \
         and recorded as debt until it has proven itself; a fresh gate that fails on day one \
         would train people to override it",
    ),
    (
        "scheduler-equivalence",
        "not required YET: a promotion candidate (see PROMOTION_CANDIDATES). It certifies the \
         asynchronous device router against the synchronous one, and `--scheduler-config \
         async` is not the default; it becomes required when that changes, on a green run \
         on every served model",
    ),
    (
        "mlperf-agentic-subset",
        "not RUNNABLE yet: the official MLPerf Agentic Inference dataset is unpublished \
         upstream (mlcommons/endpoints@7935df4: \"MLCommons storage (link TBD)\"), so the leg \
         cannot be run, scored, or timed, and it refuses proxy datasets on purpose. Not a \
         promotion candidate either — a candidate accrues debt rows, and debt nobody can \
         discharge is worse than no row. Promote only after the dataset ships and a \
         calibration run sizes a <2 h draw",
    ),
];

/// 2026-09-26: True when `path` is `entry` or lies beneath it, compared per component:
/// `"Cargo.toml.orig"` and `"crates2/x"` start with `"Cargo.toml"` and `"crates"` but are
/// not under them.
fn under(path: &str, entry: &str) -> bool {
    path == entry
        || path
            .strip_prefix(entry)
            .is_some_and(|rest| rest.starts_with('/'))
}

/// 2026-09-26: Whether `path` is under a [`PERF_PATHS`] entry.
pub fn on_boundary(path: &str) -> bool {
    PERF_PATHS.iter().any(|entry| under(path, entry))
}

/// 2026-09-26: Whether changing `path` invalidates `gate`'s existing records. The checks run in
/// this order, and the first that applies decides:
///
/// 1. under a [`BOUNDARY_FILES`] entry: invalidates;
/// 2. a registered test-only module: does not;
/// 3. a non-compiled kernel file (`BENCH.toml`): does not;
/// 4. not under [`PERF_PATHS`]: does not;
/// 5. otherwise invalidates unless one of `gate`'s exclusions covers it, so a path nobody
///    classified invalidates.
pub fn invalidates(gate: &GateCoverage, path: &str) -> bool {
    if BOUNDARY_FILES.iter().any(|f| under(path, f)) {
        return true;
    }
    if is_test_only_rust_module(path) {
        return false;
    }
    if is_non_compiled_kernel_file(path) {
        return false;
    }
    if !on_boundary(path) {
        return false;
    }
    !gate.excludes.iter().any(|e| under(path, e.prefix))
}

/// 2026-09-26: The required gates invalidated by a set of changed paths: a function of the paths
/// alone, with no network, environment or clock input, so it reproduces offline.
pub fn invalidated_by<'a, I>(paths: I) -> Vec<&'static str>
where
    I: IntoIterator<Item = &'a str>,
{
    let paths: Vec<&str> = paths.into_iter().collect();
    REQUIRED
        .iter()
        .filter(|gate| paths.iter().any(|p| invalidates(gate, p)))
        .map(|gate| gate.id)
        .collect()
}

/// 2026-09-26: A required gate's coverage by id; promotion candidates are not searched.
pub fn find(id: &str) -> Option<&'static GateCoverage> {
    REQUIRED.iter().find(|g| g.id == id)
}

/// 2026-09-26: Which [`PROMOTION_CANDIDATES`] these changed paths invalidate: the debt a merge
/// takes on, one id per candidate that was owed a run and not required to have one.
pub fn promotion_debt<'a, I>(paths: I) -> Vec<&'static str>
where
    I: IntoIterator<Item = &'a str>,
{
    let paths: Vec<&str> = paths.into_iter().collect();
    PROMOTION_CANDIDATES
        .iter()
        .filter(|gate| paths.iter().any(|p| invalidates(gate, p)))
        .map(|gate| gate.id)
        .collect()
}
