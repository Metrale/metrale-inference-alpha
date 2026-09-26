// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests that the committed tree's `[benchmarks.param_overrides]` and the concurrency
//! entries match their gates' schemas and pinned values.
//!
//! Owner: bench gate.
//! Invariants: none beyond the types.

use super::*;

/// 2026-09-26: Every committed `[benchmarks.param_overrides]` pin names a registered benchmark
/// and one of its parameters, parses through that parameter's kind, and is not a
/// `threshold_params` key. The full list of pins is asserted, so none is skipped.
#[test]
fn every_committed_param_override_parses_against_its_gates_schema() {
    let root = repo_root();
    let mut observed = Vec::new();
    for (target, entry) in load_all(&root).expect("tree loads") {
        if entry.param_overrides.is_empty() {
            continue;
        }
        let descriptor = crate::registry::find(&entry.gate).unwrap_or_else(|| {
            panic!(
                "{}/{}: param_overrides on unregistered benchmark {:?}",
                target.hardware, target.model, entry.gate
            )
        });
        let specs = descriptor.build().parameters();
        for (key, raw) in &entry.param_overrides {
            observed.push((
                target.hardware.clone(),
                target.model.clone(),
                entry.gate.clone(),
                key.clone(),
                raw.clone(),
            ));
            assert!(
                !descriptor.threshold_params.iter().any(|(p, _)| p == key),
                "{}/{}/{}: pin {key:?} names a threshold-coupled param",
                target.hardware,
                target.model,
                entry.gate
            );
            let spec = specs
                .iter()
                .find(|s| s.key == key.as_str())
                .unwrap_or_else(|| {
                    panic!(
                        "{}/{}/{}: pin {key:?} names no schema parameter",
                        target.hardware, target.model, entry.gate
                    )
                });
            spec.kind.parse(raw).unwrap_or_else(|e| {
                panic!(
                    "{}/{}/{}: pin {key}={raw} does not parse: {e:#}",
                    target.hardware, target.model, entry.gate
                )
            });
        }
    }
    // 2026-09-26: `load_all` walks `kernels/<hw>/<model>` in sorted order, so qwen3.6-35b-a3b's
    // pins come before qwen3.8-27b's.
    let moe = |key: &str, value: &str| {
        (
            "gb10".to_string(),
            "qwen3.6-35b-a3b".to_string(),
            "concurrency-sweep-moe".to_string(),
            key.to_string(),
            value.to_string(),
        )
    };
    assert_eq!(
        observed,
        vec![
            moe("concurrencies", "1,2,4,8,16"),
            moe("isls", "128"),
            moe("osl", "1024"),
            moe("prompt_mode", "essay"),
            (
                "gb10".into(),
                "qwen3.8-27b".into(),
                "concurrency-sweep".into(),
                "concurrencies".into(),
                "1,2,4,8,16,32,64,128".into(),
            ),
            (
                "gb10".into(),
                "qwen3.8-27b".into(),
                "concurrency-sweep".into(),
                "isls".into(),
                "128".into(),
            ),
            (
                "gb10".into(),
                "qwen3.8-27b".into(),
                "concurrency-sweep".into(),
                "osl".into(),
                "1024".into(),
            ),
            (
                "gb10".into(),
                "qwen3.8-27b".into(),
                "concurrency-sweep".into(),
                "prompt_mode".into(),
                "essay".into(),
            ),
            // 2026-09-26: The DFlash2 ladder stops at its batch cap of 16: a wider rung would
            // measure the cap. It keeps ISL 512 / OSL 200, where its floors were cut.
            (
                "gb10".into(),
                "qwen3.8-27b".into(),
                "concurrency-sweep-dflash2".into(),
                "concurrencies".into(),
                "1,2,4,8,16".into(),
            ),
            (
                "gb10".into(),
                "qwen3.8-27b".into(),
                "concurrency-sweep-dflash2".into(),
                "isls".into(),
                "512".into(),
            ),
            (
                "gb10".into(),
                "qwen3.8-27b".into(),
                "concurrency-sweep-dflash2".into(),
                "osl".into(),
                "200".into(),
            ),
            // 2026-09-26: kat-equality-gate: `orders` and `sample_cap` fix what its `orders` and
            // `samples` metric pins describe, and `max_new_tokens` matches BFCL's
            // `MAX_NEW_TOKENS`, since the gate replays BFCL's requests.
            (
                "gb10".into(),
                "qwen3.8-27b".into(),
                "kat-equality-gate".into(),
                "max_new_tokens".into(),
                "1024".into(),
            ),
            (
                "gb10".into(),
                "qwen3.8-27b".into(),
                "kat-equality-gate".into(),
                "orders".into(),
                "2".into(),
            ),
            (
                "gb10".into(),
                "qwen3.8-27b".into(),
                "kat-equality-gate".into(),
                "sample_cap".into(),
                "257".into(),
            ),
            // 2026-09-26: scheduler-equivalence: at max_batch=1 a C=16 request can wait behind 15
            // others, longer than the driver's 300 s default.
            (
                "gb10".into(),
                "qwen3.8-27b".into(),
                "scheduler-equivalence".into(),
                "request_timeout_s".into(),
                "3600".into(),
            ),
        ],
        "the committed override validation must not pass vacuously or skip a pin"
    );
}

/// 2026-09-26: The MoE concurrency entry, pinned by value: the only entry for its gate id, its
/// declared subject, measured, with its floors, recipe, param pins and serve pins asserted
/// exactly, so re-cutting a floor or moving a pin is an edit here too.
#[test]
fn the_moe_concurrency_entry_is_the_published_instrument_with_its_bootstrap_floors() {
    use std::collections::BTreeMap;
    let root = repo_root();
    let all = load_all(&root).expect("tree loads");
    let moe: Vec<_> = all
        .iter()
        .filter(|(target, entry)| {
            target.hardware == "gb10"
                && target.model == "qwen3.6-35b-a3b"
                && entry.checkpoint == "Qwen/Qwen3.6-35B-A3B-FP8"
                && entry.gate.starts_with("concurrency-sweep")
        })
        .collect();
    assert_eq!(
        moe.iter().map(|(_, e)| e.gate.as_str()).collect::<Vec<_>>(),
        ["concurrency-sweep-moe"],
        "the MoE ladder has exactly one gate id — a second entry under \
         `concurrency-sweep` would split its records across two directories \
         and the site would file them under two subjects"
    );
    let (_, entry) = moe[0];
    assert!(
        entry.default,
        "the only checkpoint on this gate must declare itself its subject"
    );
    assert_eq!(entry.status, "measured");
    let floors: BTreeMap<String, (Option<f64>, Option<f64>)> = entry
        .metrics
        .as_ref()
        .expect("the MoE ladder carries its bootstrap floors")
        .iter()
        .map(|(k, b)| (k.clone(), (b.min, b.max)))
        .collect();
    assert_eq!(
        floors,
        [
            ("c1_aggregate_tok_s", (Some(69.47), None)),
            ("c2_aggregate_tok_s", (Some(80.88), None)),
            ("c4_aggregate_tok_s", (Some(92.72), None)),
            ("c8_aggregate_tok_s", (Some(101.71), None)),
            ("c16_aggregate_tok_s", (Some(102.63), None)),
            ("peak_aggregate_tok_s", (Some(102.63), None)),
            ("min_completion_tokens", (Some(820.0), None)),
            ("vacuous_cells", (None, Some(0.0))),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect::<BTreeMap<_, _>>()
    );
    assert!(
        entry
            .metrics
            .as_ref()
            .unwrap()
            .values()
            .all(|b| b.noise.is_none()),
        "no noise allowance: the 5% band is already in each bar, and the entry has no \
         run-to-run history to size one from"
    );
    assert_eq!(
        entry.recipe.as_deref(),
        Some("qwen3.6/qwen3.6-35b-a3b-fp8-nvfp4head")
    );
    let pins = |kv: &[(&str, &str)]| {
        kv.iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect::<BTreeMap<_, _>>()
    };
    assert_eq!(
        entry.param_overrides,
        pins(&[
            ("concurrencies", "1,2,4,8,16"),
            ("isls", "128"),
            ("osl", "1024"),
            ("prompt_mode", "essay"),
        ])
    );
    assert_eq!(
        entry.serve_overrides,
        pins(&[
            ("disable_thinking", "true"),
            ("gpu_memory_utilization", "0.85"),
            ("kv_cache_dtype", "bf16"),
            ("max_batch_size", "128"),
            ("max_model_len", "2048"),
            ("num_drafts", "1"),
            ("scheduler", "fifo"),
            ("ssm_cache_slots", "32"),
        ])
    );
    assert!(
        !entry.serve_overrides.contains_key("lm_head_dtype"),
        "the head dtype is the nvfp4head recipe's own precision choice, not a gate pin"
    );
}

/// 2026-09-26: The dense ladder carries a J/token ceiling (`max` only, no noise) on each of its
/// eight rungs, on the key `hardware::energy` writes (`c{C}_gpu_rail_joules_per_token`).
/// The values are pinned so a re-cut edits this test.
#[test]
fn the_dense_concurrency_entry_carries_a_joule_ceiling_on_every_rung() {
    use crate::gate::record::Bound;
    use std::collections::BTreeMap;
    let root = repo_root();
    let all = load_all(&root).expect("tree loads");
    let dense: Vec<_> = all
        .iter()
        .filter(|(target, entry)| {
            target.hardware == "gb10"
                && target.model == "qwen3.8-27b"
                && entry.gate == "concurrency-sweep"
                && entry.default
        })
        .collect();
    assert_eq!(dense.len(), 1, "one default subject on the dense ladder");
    let metrics = dense[0]
        .1
        .metrics
        .as_ref()
        .expect("the dense ladder is bounded");
    let ceilings: BTreeMap<&str, &Bound> = metrics
        .iter()
        .filter(|(k, _)| k.ends_with("_gpu_rail_joules_per_token"))
        .map(|(k, b)| (k.as_str(), b))
        .collect();
    let want = [
        (1, 1.7),
        (2, 1.0),
        (4, 0.62),
        (8, 0.44),
        (16, 0.30),
        (32, 0.23),
        (64, 0.19),
        (128, 0.16),
    ];
    assert_eq!(ceilings.len(), want.len(), "{:?}", ceilings.keys());
    for (c, max) in want {
        let key = format!("c{c}_gpu_rail_joules_per_token");
        let b = ceilings
            .get(key.as_str())
            .unwrap_or_else(|| panic!("{key} missing"));
        assert_eq!((b.min, b.max, b.noise), (None, Some(max), None), "{key}");
    }
    // 2026-09-26: Every ceiling is on a rung that also has a tok/s floor.
    for k in ceilings.keys() {
        let rung = k.trim_end_matches("_gpu_rail_joules_per_token");
        assert!(
            metrics.contains_key(&format!("{rung}_aggregate_tok_s")),
            "{k} has no matching tok/s floor"
        );
    }
}
