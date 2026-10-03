// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: The circuit memory model against real boots: each fixture under
//! `tests/fixtures/memory_ledger/` is one serve's allocation ledger ("by file" rollup) and
//! preflight reserve terms (PR #72's named terms), with the serve's settings and its pool sizes.
//! The model predicts every term the ledger itemizes from the checkpoint's config and those
//! settings alone, and must land within each term's stated tolerance; #72's reserve terms (the
//! SSM pool, the Marconi snapshots, the carry stash, the driver) must agree with the model's.
//!
//! Owner: server CLI.
//! Invariants:
//! - A term is compared only when the boot's ledger lists one of its files (the rollup shows the
//!   twelve largest); an absent term is reported, never treated as zero.
//! - The tolerances are the model's stated accuracy (`kernels/circuits/MEMORY-DESIGN.md`).

use std::collections::BTreeMap;
use std::path::PathBuf;

use metrale_circuit::memory::MemoryReport;
use metrale_circuit::memory::copies::COPY_SETTINGS;
use metrale_circuit::state::{Holding, StateKind};
use serde::Deserialize;

use super::super::CircuitMemoryArgs;
use super::super::circuit_hw::CheckpointTexts;
use super::super::circuit_memory_point::{BootPool, Point, prepare};
use super::super::circuit_memory_serve::EngineFacts;
use super::load;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Fixture {
    checkpoint: String,
    recipe: Option<String>,
    serve: Vec<String>,
    kv_blocks: u64,
    draft_kv_blocks: u64,
    capture_rows: u64,
    chunk_slack_gb: f64,
    ledger_total_gb: f64,
    ledger_sites: u64,
    outside_bytes: u64,
    ledger: BTreeMap<String, (f64, u64)>,
    reserve: BTreeMap<String, u64>,
}

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

const MIB: f64 = (1u64 << 20) as f64;

/// 2026-10-02: A term: its name, the ledger files that hold it, and its tolerance (fraction).
const TERMS: [(&str, &[&str], f64); 6] = [
    (
        "weights",
        &[
            "crates/model-weights/src/fast_weights/mod.rs",
            "crates/model-layers/src/layers/dense_ffn_load.rs",
            "crates/model-layers/src/weight_map/quant_helpers.rs",
            "crates/model-layers/src/weight_map/loaders_fp8.rs",
            "crates/model-arch/src/weight_loader/qwen35_dense/gdn_dequant.rs",
            "crates/model-layers/src/weight_map/quantized/transpose.rs",
            "crates/model-layers/src/weight_map/quantized.rs",
            "crates/model-layers/src/weight_map/fp8_lut.rs",
            "crates/model-arch/src/weight_loader/qwen35/load_layers/linear_attn_arms.rs",
            "crates/model-arch/src/weight_loader/nemotron.rs",
        ],
        0.01,
    ),
    ("kv", &["crates/cache/src/kv_cache/paged_impl.rs"], 0.001),
    (
        "ssm_pool",
        &["crates/model-engine/src/model/ssm_pool.rs"],
        0.001,
    ),
    (
        "snapshots",
        &["crates/model-engine/src/model/ssm_snapshot_init.rs"],
        0.005,
    ),
    ("arena", &["crates/gpu-runtime/src/buffers.rs"], 0.01),
    (
        "gdn_two_phase",
        &["crates/model-engine/src/model/impl_a1_init.rs"],
        0.001,
    ),
];

/// 2026-10-02: The model's bytes of `term`; `f` gives the engine's legacy arena.
fn model_term(r: &MemoryReport, f: &EngineFacts, term: &str) -> u64 {
    let state = |keep: &dyn Fn(Holding) -> bool| r.states.bytes_where(|t| keep(t.holding));
    let cache = |keep: &dyn Fn(StateKind) -> bool| {
        r.caches
            .iter()
            .filter(|c| keep(c.kind))
            .map(|c| c.bytes)
            .sum::<u64>()
    };
    match term {
        "weights" => r.totals.weights_stored + r.totals.weights_derived,
        "kv" => state(&|h| h == Holding::Blocks),
        "ssm_pool" => state(&|h| h != Holding::Blocks),
        "snapshots" => cache(&|k| matches!(k, StateKind::PrefixSnapshot | StateKind::RingSnapshot)),
        "arena" => f.legacy_arena - f.gdn_two_phase,
        "gdn_two_phase" => f.gdn_two_phase,
        "carry" => cache(&|k| matches!(k, StateKind::CarryStash | StateKind::CarryTable)),
        "marconi" => r
            .caches
            .iter()
            .filter(|c| c.kind == StateKind::PrefixSnapshot && !c.state.starts_with("head."))
            .map(|c| c.bytes)
            .sum(),
        "ring" => cache(&|k| k == StateKind::RingSnapshot),
        "driver" => r.totals.driver,
        other => panic!("no term {other}"),
    }
}

/// 2026-10-02: One fixture's model, evaluated at the boot's own pools.
/// 2026-10-02: The prepared point of `checkpoint` (its in-tree config fixture) under `recipe` and
/// `serve`; every setting a copy rule may read is set.
pub(crate) fn point(
    name: &str,
    checkpoint: &str,
    recipe: Option<String>,
    serve: Vec<String>,
    capture_rows: u64,
) -> Point<'static> {
    let dir = root()
        .join("crates/circuit/tests/fixtures/checkpoints")
        .join(checkpoint.replacen('/', "--", 1));
    let read = |f: &str| std::fs::read_to_string(dir.join(f)).ok();
    let texts = CheckpointTexts {
        id: checkpoint.to_string(),
        config: read("config.json"),
        hf_quant: read("hf_quant_config.json"),
        dir: None,
    };
    // 2026-10-02: A test-lifetime point borrows its args, tree and registry; leaked, not freed.
    let a: &'static CircuitMemoryArgs = Box::leak(Box::new(CircuitMemoryArgs {
        checkpoint: checkpoint.to_string(),
        hardware: "gb10".into(),
        recipe,
        isl: 1,
        osl: 1,
        concurrency: 1,
        slots: None,
        capture_rows,
        prompt_lookup: false,
        tree_nodes: None,
        per_node: false,
        json: false,
        allow_network: false,
        root: Some(root()),
        serve,
    }));
    let (tree, reg) = load(a, &root()).unwrap();
    let (tree, reg) = (Box::leak(Box::new(tree)), Box::leak(Box::new(reg)));
    let p = prepare(a, &texts, &root(), tree, reg).unwrap_or_else(|e| panic!("{name}: {e:#}"));
    for k in COPY_SETTINGS {
        assert!(
            p.settings.contains_key(k),
            "{name}: copy setting `{k}` unset"
        );
    }
    p
}

fn evaluate(name: &str, fx: &Fixture) -> (MemoryReport, EngineFacts) {
    let p = point(
        name,
        &fx.checkpoint,
        fx.recipe.clone(),
        fx.serve.clone(),
        fx.capture_rows,
    );
    let boot = BootPool {
        kv_blocks: fx.kv_blocks,
        draft_kv_blocks: fx.draft_kv_blocks,
    };
    let (r, f) = p
        .eval_with(1, 1, 1, Some(boot))
        .unwrap_or_else(|e| panic!("{name}: {e:#}"));
    (r.with_outside(fx.outside_bytes), f)
}

fn fixtures() -> Vec<(String, Fixture)> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/memory_ledger");
    let mut out: Vec<(String, Fixture)> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "toml"))
        .map(|p| {
            let name = p.file_stem().unwrap().to_string_lossy().into_owned();
            let fx = toml::from_str(&std::fs::read_to_string(&p).unwrap())
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            (name, fx)
        })
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// 2026-10-02: Every itemized term of every boot within its tolerance; the table prints with
/// `--nocapture`.
#[test]
fn the_model_predicts_each_boot_ledger_within_its_tolerance() {
    let mut failures = Vec::new();
    println!("| boot | term | model MiB | ledger MiB | error |\n|---|---|---:|---:|---:|");
    for (name, fx) in fixtures() {
        let (r, f) = evaluate(&name, &fx);
        let mut listed = 0.0;
        for (term, files, tol) in TERMS {
            let present: Vec<f64> = files
                .iter()
                .filter_map(|f| fx.ledger.get(*f).map(|x| x.0))
                .collect();
            let model = model_term(&r, &f, term) as f64 / MIB;
            if present.is_empty() {
                println!("| {name} | {term} | {model:.1} | not itemized | - |");
                continue;
            }
            let ledger: f64 = present.iter().sum();
            listed += ledger;
            let err = (model - ledger) / ledger;
            println!(
                "| {name} | {term} | {model:.1} | {ledger:.1} | {:+.2}% |",
                err * 100.0
            );
            if err.abs() > tol {
                failures.push(format!("{name} {term}: {err:+.4} beyond ±{tol}"));
            }
        }
        let total_mib = fx.ledger_total_gb * 1e9 / MIB;
        println!(
            "| {name} | (not attributed: smaller files and sites; {} sites, slack {} GB) | - | {:.1} | - |",
            fx.ledger_sites,
            fx.chunk_slack_gb,
            total_mib - fx.ledger.values().map(|x| x.0).sum::<f64>()
        );
        assert!(listed > 0.0, "{name}: no term itemized");
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// 2026-10-02: PR #72's named reserve terms equal the model's terms of the same name (#72 logs
/// whole MB, truncated): the SSM pool, the Marconi snapshots (h and conv; the reserve leaves out
/// the last-hidden row), the decode ring, the carry stash and the driver.
#[test]
fn pr72_reserve_terms_agree_with_the_model() {
    for (name, fx) in fixtures() {
        if fx.reserve.is_empty() {
            continue;
        }
        let (r, f) = evaluate(&name, &fx);
        for (term, keys) in [
            ("ssm_pool", &["ssm_pool"][..]),
            ("marconi", &["marconi"][..]),
            ("ring", &["decode_ring"][..]),
            ("carry", &["gdn_carry_stash"][..]),
            ("driver", &["driver_fixed", "driver_bookkeeping"][..]),
        ] {
            let reserve: u64 = keys.iter().map(|k| fx.reserve[*k]).sum();
            let model = model_term(&r, &f, term) / (1 << 20);
            assert!(
                model.abs_diff(reserve) <= keys.len() as u64,
                "{name} {term}: model {model} MiB, #72 reserve {reserve} MB"
            );
        }
    }
}

/// 2026-10-02: Path B: the routed experts' transposed NVFP4 tables are built only without a
/// latent MoE (`prefill_weights.rs`): Nemotron-3-Nano carries them, Nemotron-3-Super
/// (`moe_latent_size` 1024) has none.
#[test]
fn the_nemotron_expert_twins_follow_the_latent_moe() {
    for (checkpoint, latent) in [
        ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4", "off"),
        ("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4", "on"),
    ] {
        let p = point(checkpoint, checkpoint, None, vec![], 0);
        assert_eq!(p.settings["latent_moe"], latent, "{checkpoint}");
        let (r, _) = p.eval_with(1, 1, 1, None).unwrap();
        let twins: u64 = r
            .weights
            .iter()
            .flat_map(|w| &w.derived)
            .filter(|d| d.rule == "nemotron-expert-twin")
            .map(|d| d.bytes)
            .sum();
        assert_eq!(
            twins > 0,
            latent == "off",
            "{checkpoint}: {twins} twin bytes"
        );
    }
}
