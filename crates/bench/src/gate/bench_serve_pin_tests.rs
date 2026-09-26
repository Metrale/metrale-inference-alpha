// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for `[benchmarks.serve_overrides]`: how pins parse, and the committed
//! tree's pins per gate.
//!
//! `check_record` matches a record's serve overrides against its entry's pins in both
//! directions, so a changed pin refuses existing records; asserting every committed pin here
//! makes such a change an edit to this file as well.
//!
//! Owner: bench gate.
//! Invariants: none beyond the types.

use super::bench_tests::{fixture, repo_root};
use super::*;

/// 2026-09-26: A `[benchmarks.serve_overrides]` table reaches the resolved baseline entry.
#[test]
fn serve_overrides_are_assembled_into_the_baseline() {
    let root = fixture(
        "serve-overrides",
        r#"
[[benchmarks]]
quant = "nvfp4"
checkpoint = "org/A"
gate = "bfcl-subset"
default = true
status = "measured"
[benchmarks.serve_overrides]
ssm_cache_slots = "256"
[benchmarks.metrics.overall_accuracy]
min = 85.0
"#,
    );
    let baseline = baseline_for(&root, "bfcl-subset").unwrap();
    let (checkpoint, entry) = baseline.resolve("gb10", None).unwrap();
    assert_eq!(checkpoint, "org/A");
    assert_eq!(
        entry.serve_overrides,
        std::collections::BTreeMap::from([("ssm_cache_slots".to_string(), "256".to_string())])
    );
}

/// 2026-09-26: A `port` pin is refused at load: self-start binds a free port itself.
#[test]
fn a_port_serve_override_is_refused() {
    let root = fixture(
        "port-pin",
        r#"
[[benchmarks]]
quant = "nvfp4"
checkpoint = "org/A"
gate = "bfcl-subset"
default = true
status = "measured"
[benchmarks.serve_overrides]
port = "8888"
[benchmarks.metrics.overall_accuracy]
min = 85.0
"#,
    );
    let err = load_all(&root).unwrap_err().to_string();
    assert_eq!(
        err,
        format!(
            "{}: bfcl-subset / org/A serve_overrides cannot set `port`: self-start binds a free port itself, so a pin here would name a listener that is not there",
            root.join("kernels/gb10/modelA/BENCH.toml").display()
        )
    );
}

/// 2026-09-26: The committed tree's serve pins, gate by gate, plus the `bfcl-subset-echolp`
/// floors, asserted by value so a change to any of them is an edit here too.
#[test]
fn the_trees_serve_pins_sit_on_the_gates_that_need_them() {
    let root = repo_root();

    let echolp = baseline_for(&root, "bfcl-subset-echolp").unwrap();
    let (_, e) = echolp.resolve("gb10", None).unwrap();
    assert_eq!(e.metrics["overall_accuracy"].min, Some(84.56));
    assert_eq!(e.metrics["normalized_single_turn_score"].min, Some(85.77));
    assert_eq!(e.metrics["samples"].min, Some(1004.0));
    assert_eq!(e.metrics["samples"].max, Some(1004.0));
    assert_eq!(
        e.serve_overrides.get("ssm_cache_slots").map(String::as_str),
        Some("256")
    );
    assert_eq!(
        e.serve_overrides
            .get("gpu_memory_utilization")
            .map(String::as_str),
        Some(GB10_UTIL_CEILING)
    );
    assert_eq!(e.serve_overrides.len(), 2, "{:?}", e.serve_overrides);

    // 2026-09-26: The poison gate pins the two serve settings its driver documents
    // (`ssm_cache_slots=256` in ssm_poison/driver.rs, `disable_thinking=true` in
    // ssm_poison/probe.rs) plus the GB10 util ceiling.
    let poison = baseline_for(&root, "ssm-state-poisoning-gate").unwrap();
    let (_, p) = poison.resolve("gb10", None).unwrap();
    assert_eq!(
        p.serve_overrides.get("ssm_cache_slots").map(String::as_str),
        Some("256")
    );
    assert_eq!(
        p.serve_overrides
            .get("disable_thinking")
            .map(String::as_str),
        Some("true")
    );
    assert_eq!(
        p.serve_overrides
            .get("gpu_memory_utilization")
            .map(String::as_str),
        Some(GB10_UTIL_CEILING)
    );
    assert_eq!(p.serve_overrides.len(), 3, "{:?}", p.serve_overrides);

    let sweep = baseline_for(&root, "concurrency-sweep").unwrap();
    let (_, c) = sweep.resolve("gb10", None).unwrap();
    assert_eq!(
        c.recipe.as_deref(),
        Some("qwen3.8/qwen3.8-27b-nvfp4-throughput"),
        "the concurrency ladder must serve the published leg's own recipe, not the \
         agentic profile it used to override key by key"
    );
    for (key, want) in [
        ("max_batch_size", "128"),
        ("kv_cache_dtype", "fp8"),
        ("max_model_len", "2048"),
        // 2026-09-26: `--prefill-codispatch` is a serve flag, so it is pinned per gate; the
        // TTFT gates below do not pin it.
        ("prefill_codispatch", "true"),
        // 2026-09-26: `--w4a4-downcast` defaults to false; this ladder pins it on.
        ("w4a4_downcast", "true"),
        // 2026-09-26: `--w4a4-downcast-wide` extends the same numerics to 33..=64 rows and
        // needs `--w4a4-downcast`; both ladders pin both.
        ("w4a4_downcast_wide", "true"),
    ] {
        assert_eq!(
            c.serve_overrides.get(key).map(String::as_str),
            Some(want),
            "concurrency-sweep serve pin {key}: {:?}",
            c.serve_overrides
        );
    }
    assert_eq!(c.serve_overrides.len(), 6, "{:?}", c.serve_overrides);
    assert!(
        !c.serve_overrides.contains_key("lm_head_dtype"),
        "the throughput recipe leaves the head at the checkpoint's native NVFP4; pinning \
         bf16 here is what cost 41-53% across the ladder"
    );
    assert!(
        !c.serve_overrides.contains_key("ssm_cache_slots"),
        "the recipe's 8 is also the published leg's `--ssm-cache-slots 8`; overriding it \
         back to 32 re-opens the last disagreement with that leg"
    );

    // 2026-09-26: The DFlash2 ladder pins the drafter explicitly: `dflash`, `draft_model` and
    // `dflash_gamma` (unset, the gamma is derived from the drafter's block size).
    let dflash2 = baseline_for(&root, "concurrency-sweep-dflash2").unwrap();
    let (_, d) = dflash2.resolve("gb10", None).unwrap();
    for (key, want) in [
        ("max_batch_size", "16"),
        ("kv_cache_dtype", "fp8"),
        ("ssm_cache_slots", "32"),
        ("max_model_len", "4096"),
        ("dflash", "true"),
        ("draft_model", "incoai/Qwen3.8-27B-DFlash2"),
        ("dflash_gamma", "8"),
        ("w4a4_downcast", "true"),
        ("w4a4_downcast_wide", "true"),
    ] {
        assert_eq!(
            d.serve_overrides.get(key).map(String::as_str),
            Some(want),
            "concurrency-sweep-dflash2 serve pin {key}: {:?}",
            d.serve_overrides
        );
    }
    assert_eq!(d.serve_overrides.len(), 9, "{:?}", d.serve_overrides);
    assert!(
        !d.serve_overrides.contains_key("speculative"),
        "--dflash conflicts with --speculative at the CLI: pinning both would not start"
    );
    // 2026-09-26: Every key the plain ladder pins must be pinned identically on the DFlash2
    // ladder, except the keys in the three lists below, which must differ; a listed key that
    // comes back into agreement fails, so an exception cannot outlive its cause. The two
    // ladders name different recipes, so this compares only the plain ladder's pins.
    //
    // `max_batch_size`: 16 on the DFlash2 ladder; its BENCH.toml note records the memory
    // measurements behind the drafter's batch limit.
    const FORCED_BY_THE_DRAFTER: [&str; 1] = ["max_batch_size"];
    // 2026-09-26: `max_model_len`: the plain ladder runs ISL 128 / OSL 1024 at ctx 2048; the
    // DFlash2 ladder runs ISL 512 / OSL 200 at ctx 4096, where its floors were cut, and
    // moving the pin would refuse its records.
    const FORCED_BY_THE_REPOINT: [&str; 1] = ["max_model_len"];
    // 2026-09-26: `prefill_codispatch`: pinned on the plain ladder only.
    const FORCED_BY_THE_LEVER_PROMOTION: [&str; 1] = ["prefill_codispatch"];
    for (key, want) in &c.serve_overrides {
        if FORCED_BY_THE_DRAFTER.contains(&key.as_str())
            || FORCED_BY_THE_REPOINT.contains(&key.as_str())
            || FORCED_BY_THE_LEVER_PROMOTION.contains(&key.as_str())
        {
            assert_ne!(
                d.serve_overrides.get(key),
                Some(want),
                "{key} is listed as forced apart but the two gates agree on it — drop it \
                 from the exception list rather than leaving a stale excuse"
            );
            continue;
        }
        assert_eq!(
            d.serve_overrides.get(key),
            Some(want),
            "the two concurrency ladders may differ only where an exception list says so \
             and says why, but {key} differs"
        );
    }

    // 2026-09-26: `decode-floor` is checked because a serve table placed above the next
    // `[[benchmarks]]` header in its BENCH.toml would attach to it.
    for id in ["bfcl-subset", "decode-floor"] {
        let b = baseline_for(&root, id).unwrap();
        let (_, entry) = b.resolve("gb10", None).unwrap();
        assert!(
            entry.serve_overrides.is_empty(),
            "{id} keeps the recipe's own config: {:?}",
            entry.serve_overrides
        );
    }
    for id in ["ttft-warm-gate", "ttft-cold-gate", "agentic-webserver"] {
        let b = baseline_for(&root, id).unwrap();
        let (checkpoint, entry) = b.resolve("gb10", None).unwrap();
        assert_eq!(checkpoint, "Qwen/Qwen3.6-35B-A3B-FP8", "{id}");
        assert_eq!(
            entry.serve_overrides,
            std::collections::BTreeMap::from([(
                "gpu_memory_utilization".to_string(),
                GB10_UTIL_CEILING.to_string()
            )]),
            "{id} pins only the GB10 util ceiling"
        );
    }
}

/// 2026-09-26: The `gpu_memory_utilization` every 35B FP8 entry on GB10 pins
/// (qwen3.6-35b-a3b/BENCH.toml).
const GB10_UTIL_CEILING: &str = "0.85";

/// 2026-09-26: Every GB10 entry for the 35B FP8 checkpoint pins the util ceiling. The list of
/// gates seen is asserted, so the test cannot pass on zero entries.
#[test]
fn every_gb10_fp8_moe_entry_pins_the_util_ceiling() {
    let root = repo_root();
    let mut seen = Vec::new();
    for (target, entry) in load_all(&root).expect("tree loads") {
        if target.hardware != "gb10" || entry.checkpoint != "Qwen/Qwen3.6-35B-A3B-FP8" {
            continue;
        }
        assert_eq!(
            entry
                .serve_overrides
                .get("gpu_memory_utilization")
                .map(String::as_str),
            Some(GB10_UTIL_CEILING),
            "{} ({:?}) must pin the GB10 util ceiling",
            entry.gate,
            entry.recipe
        );
        seen.push(entry.gate.clone());
    }
    seen.sort();
    assert_eq!(
        seen,
        [
            "agentic-webserver",
            "bfcl-subset-echolp",
            "concurrency-sweep-moe",
            "mlperf-agentic-subset",
            "ssm-state-poisoning-gate",
            "ttft-cold-gate",
            "ttft-warm-gate",
            "video-fidelity",
            "vision-fidelity",
        ],
        "the check must not pass vacuously"
    );
}

#[path = "bench_override_tree_tests.rs"]
mod bench_override_tree_tests;

/// 2026-09-26: An entry pinning `hermetic=true` without every key `--hermetic` closes is refused
/// at load. Such a run's record would carry the closed keys, and `check_record`, which
/// compares serve overrides in both directions, would refuse it after the run.
#[test]
fn a_hermetic_gate_that_does_not_pin_what_hermetic_closes_is_refused() {
    let root = fixture(
        "hermetic-underpinned",
        r#"
[[benchmarks]]
quant = "nvfp4"
checkpoint = "org/A"
gate = "bfcl-subset"
default = true
status = "measured"
[benchmarks.serve_overrides]
hermetic = "true"
[benchmarks.metrics.overall_accuracy]
min = 85.0
"#,
    );
    let err = format!("{:#}", baseline_for(&root, "bfcl-subset").unwrap_err());
    assert!(
        err.contains("pins hermetic=true but not"),
        "must name the omission: {err}"
    );
    assert!(
        err.contains("enable_prefix_caching=false") && err.contains("mtp_gate=force"),
        "and must name every missing key at the value it needs: {err}"
    );
}

/// 2026-09-26: An entry pinning `hermetic` and every key it closes loads.
#[test]
fn a_hermetic_gate_that_pins_the_whole_set_is_accepted() {
    let root = fixture(
        "hermetic-complete",
        r#"
[[benchmarks]]
quant = "nvfp4"
checkpoint = "org/A"
gate = "bfcl-subset"
default = true
status = "measured"
[benchmarks.serve_overrides]
hermetic = "true"
enable_prefix_caching = "false"
mtp_gate = "force"
[benchmarks.metrics.overall_accuracy]
min = 85.0
"#,
    );
    let baseline = baseline_for(&root, "bfcl-subset").expect("the complete set must parse");
    let (_, entry) = baseline.resolve("gb10", None).unwrap();
    assert_eq!(entry.serve_overrides.len(), 3);
    assert!(crate::gate::hermetic::missing_pins(&entry.serve_overrides).is_empty());
}

/// 2026-09-26: An entry without a `hermetic` pin is not affected by the rule.
#[test]
fn a_gate_with_no_hermetic_pin_is_unaffected() {
    let root = fixture(
        "hermetic-absent",
        r#"
[[benchmarks]]
quant = "nvfp4"
checkpoint = "org/A"
gate = "bfcl-subset"
default = true
status = "measured"
[benchmarks.serve_overrides]
ssm_cache_slots = "256"
[benchmarks.metrics.overall_accuracy]
min = 85.0
"#,
    );
    assert!(baseline_for(&root, "bfcl-subset").is_ok());
}
