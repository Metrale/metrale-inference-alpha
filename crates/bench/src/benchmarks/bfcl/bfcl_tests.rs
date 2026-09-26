// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for the BFCL benchmark's parameters, draw defaults,
//! verdicts (MLPerf floor, per-checkpoint baseline mins, shard info), metrics,
//! and the committed draw-size pins.
//!
//! Owner: bench, BFCL benchmark.
//! Invariants: none beyond the types.

use super::*;
use crate::result::VerdictKind;

fn configured(variant: Variant) -> Bfcl {
    let mut b = Bfcl::new(variant);
    let v = ParamValues::defaults(&b.parameters());
    b.configure(&v).unwrap();
    b
}

#[test]
fn all_three_variants_are_distinct_benchmarks() {
    assert_eq!(Bfcl::new(Variant::Subset).descriptor().id, "bfcl-subset");
    assert_eq!(
        Bfcl::new(Variant::SubsetEcholp).descriptor().id,
        "bfcl-subset-echolp"
    );
    assert_eq!(Bfcl::new(Variant::Full).descriptor().id, "bfcl-full");
    let ids = [
        SUBSET_DESCRIPTOR.id,
        SUBSET_ECHOLP_DESCRIPTOR.id,
        FULL_DESCRIPTOR.id,
    ];
    assert_eq!(
        ids.into_iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        3
    );
}

#[test]
fn subset_defaults_reproduce_the_golden_draw() {
    let b = configured(Variant::Subset);
    assert_eq!(b.spec, DrawSpec::golden());
}

#[test]
fn full_defaults_take_every_sample_of_the_scored_categories() {
    let b = configured(Variant::Full);
    // 2026-09-26: 100% with no floor selects the same rows as `full()`.
    assert_eq!(b.spec.subset_floor, None);
    assert_eq!(b.spec.take_count("simple_python", 400), 400);
    assert_eq!(b.spec.take_count("live_relevance", 16), 0);
}

#[test]
fn defaults_are_the_mlperf_generation_config() {
    let b = Bfcl::new(Variant::Subset);
    let v = ParamValues::defaults(&b.parameters());
    assert_eq!(v.usize("max_new_tokens").unwrap(), 1024);
    assert_eq!(v.float("temperature").unwrap(), 0.0);
    assert_eq!(v.usize("subset_floor").unwrap(), 25);
}

#[test]
fn a_changed_percentage_changes_the_draw() {
    let mut b = Bfcl::new(Variant::Subset);
    let mut v = ParamValues::defaults(&b.parameters());
    v.set("non_live_pct", ParamValue::Float(20.0));
    b.configure(&v).unwrap();
    assert_eq!(b.spec.take_count("simple_python", 400), 80);
    assert_ne!(b.spec, DrawSpec::golden());
}

fn scores(overall: f64, normalized: f64) -> Scores {
    Scores {
        overall_accuracy: overall,
        normalized_single_turn_score: normalized,
        category_scores: BTreeMap::new(),
        subset_scores: BTreeMap::new(),
        total_samples: 995,
        unmatched_responses: 0,
        subset_totals: BTreeMap::new(),
    }
}

/// 2026-09-26: A configured run on `model`. The verdict reads the model
/// captured at `load()`, which unit tests set directly.
fn on_model(variant: Variant, model: &str) -> Bfcl {
    let mut b = configured(variant);
    b.target_model = Some(model.to_string());
    b
}

#[test]
fn the_verdict_gates_on_both_mlperf_floors() {
    let mut b = on_model(Variant::Subset, "unsloth/Qwen3.6-27B-NVFP4");

    b.scores = Some(scores(87.44, 88.53));
    assert_eq!(b.verdict().kind, VerdictKind::Pass);

    b.scores = Some(scores(83.63, 90.0));
    let v = b.verdict();
    assert_eq!(v.kind, VerdictKind::Fail);
    assert!(
        v.reason.contains("BELOW THE MLPERF-EDGE FLOOR"),
        "{}",
        v.reason
    );

    b.scores = Some(scores(90.0, 85.31));
    assert_eq!(b.verdict().kind, VerdictKind::Fail);

    // 2026-09-26: Exactly on both floors passes; the floors are inclusive.
    b.scores = Some(scores(MLPERF_FLOOR_OVERALL, MLPERF_FLOOR_NORMALIZED));
    assert_eq!(b.verdict().kind, VerdictKind::Pass);
}

/// 2026-09-26: A shard is not judged against a whole-draw floor: scores that
/// fail a whole run, and scores that would pass it, are both `Info` on a shard,
/// naming the slice and deferring to the group. The same object without a
/// shard still fails.
#[test]
fn a_shards_verdict_is_info_and_defers_to_the_group() {
    let mut b = on_model(Variant::Subset, "unsloth/Qwen3.6-27B-NVFP4");
    b.shard = Some(dataset::Shard { index: 3, count: 6 });
    b.scores = Some(scores(80.12, 82.0));
    let v = b.verdict();
    assert_eq!(v.kind, VerdictKind::Info, "{}", v.reason);
    assert!(v.reason.contains("shard 3/6"), "{}", v.reason);
    assert!(v.reason.contains("80.12"), "{}", v.reason);
    assert!(v.reason.contains("group's verdict"), "{}", v.reason);
    b.scores = Some(scores(90.0, 90.0));
    assert_eq!(b.verdict().kind, VerdictKind::Info);
    b.shard = None;
    b.scores = Some(scores(80.12, 82.0));
    assert_eq!(b.verdict().kind, VerdictKind::Fail);
}

#[test]
fn the_verdict_always_states_the_measured_values_and_the_floor() {
    let mut b = on_model(Variant::Subset, "centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf");
    b.scores = Some(scores(87.44, 88.53));
    let reason = b.verdict().reason;
    for value in ["87.44", "83.64", "88.53", "85.32"] {
        assert!(reason.contains(value), "missing {value}: {reason}");
    }
    assert!(reason.contains("n=995"));
}

#[test]
fn an_unscored_run_is_info_not_a_pass() {
    let b = configured(Variant::Subset);
    assert_eq!(b.verdict().kind, VerdictKind::Info);
}

/// 2026-09-26: A run on weights that are not an MLPerf submission checkpoint
/// is not failed by the MLPerf floor, and the verdict says it is judged by
/// baseline thresholds instead.
#[test]
fn a_qwen38_run_below_the_36_floor_is_not_failed_by_it() {
    let mut b = on_model(Variant::Subset, "unsloth/Qwen3.8-27B-NVFP4");
    b.scores = Some(scores(84.22, 84.12));
    let v = b.verdict();
    assert_eq!(v.kind, VerdictKind::Info, "{}", v.reason);
    assert!(
        v.reason.contains("judged by baseline thresholds"),
        "{}",
        v.reason
    );
    assert!(
        !v.reason.contains("BELOW THE MLPERF-EDGE FLOOR"),
        "{}",
        v.reason
    );
    assert!(
        v.reason.contains("84.22") && v.reason.contains("85.32"),
        "{}",
        v.reason
    );
}

/// 2026-09-26: A Qwen3.8 run with its own bars. `min_overall` and
/// `min_normalized` are threshold params (`descriptors::GATE_THRESHOLD_PARAMS`)
/// that the gate fills from BENCH.toml into the values passed to `configure`;
/// here they are set the same way.
fn with_mins(overall: f64, normalized: f64) -> Bfcl {
    let mut b = Bfcl::new(Variant::Subset);
    let mut v = ParamValues::defaults(&b.parameters());
    v.set("min_overall", ParamValue::Float(overall));
    v.set("min_normalized", ParamValue::Float(normalized));
    b.configure(&v).unwrap();
    b.target_model = Some("unsloth/Qwen3.8-27B-NVFP4".to_string());
    b
}

fn qwen38_committed_mins() -> (f64, f64) {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(std::path::Path::parent)
        .expect("workspace layout");
    let (_, entry) = crate::gate::bench::load_all(root)
        .expect("committed BENCH.toml files must load")
        .into_iter()
        .find(|(target, entry)| {
            target.hardware == "gb10"
                && target.model == "qwen3.8-27b"
                && entry.checkpoint == "unsloth/Qwen3.8-27B-NVFP4"
                && entry.gate == "bfcl-subset"
        })
        .expect("the Qwen3.8 BFCL gate must be committed");
    let metrics = entry
        .metrics
        .expect("the measured BFCL gate must have bounds");
    (
        metrics["overall_accuracy"]
            .min
            .expect("overall_accuracy must have a minimum"),
        metrics["normalized_single_turn_score"]
            .min
            .expect("normalized score must have a minimum"),
    )
}

/// 2026-09-26: A run on a non-MLPerf checkpoint that clears its committed
/// BENCH.toml bars gets a PASS run verdict; the gate refuses a record whose run
/// verdict is not PASS (`GateRecord::verdict_passes` in `gate::check`).
#[test]
fn a_qwen38_run_clearing_its_own_bars_passes() {
    let (overall_min, normalized_min) = qwen38_committed_mins();
    assert_eq!((overall_min, normalized_min), (83.00, 82.96));
    let mut b = with_mins(overall_min, normalized_min);
    b.scores = Some(scores(84.22, 84.12));
    let v = b.verdict();
    assert_eq!(v.kind, VerdictKind::Pass, "{}", v.reason);
    for needle in ["84.22", "83.00", "84.12", "82.96"] {
        assert!(v.reason.contains(needle), "{}", v.reason);
    }

    // 2026-09-26: Exactly on both bars passes: the comparison is the raw
    // `>=` against the bars as handed in. Under the gate those bars are
    // already the BENCH.toml min less its noise (`apply_threshold_params`).
    b.scores = Some(scores(83.00, 82.96));
    assert_eq!(b.verdict().kind, VerdictKind::Pass);
}

#[test]
fn a_qwen38_run_below_its_own_bars_fails() {
    let (overall_min, normalized_min) = qwen38_committed_mins();
    let mut b = with_mins(overall_min, normalized_min);
    b.scores = Some(scores(82.90, 84.12));
    let v = b.verdict();
    assert_eq!(v.kind, VerdictKind::Fail, "{}", v.reason);
    assert!(
        v.reason.contains("BELOW THE BASELINE THRESHOLDS"),
        "{}",
        v.reason
    );
    for needle in ["82.90", "83.00", "84.12", "82.96"] {
        assert!(v.reason.contains(needle), "{}", v.reason);
    }

    // 2026-09-26: Either bar alone fails the run.
    b.scores = Some(scores(84.22, 82.50));
    assert_eq!(b.verdict().kind, VerdictKind::Fail);
}

/// 2026-09-26: The MLPerf submission checkpoints keep the floor verdict; the
/// baseline mins do not change it.
#[test]
fn baseline_mins_do_not_touch_the_mlperf_family() {
    let mut b = with_mins(80.0, 80.0);
    b.target_model = Some("unsloth/Qwen3.6-27B-NVFP4".to_string());
    // 2026-09-26: Above the mins but below the MLPerf floor: still a floor fail.
    b.scores = Some(scores(83.63, 90.0));
    let v = b.verdict();
    assert_eq!(v.kind, VerdictKind::Fail, "{}", v.reason);
    assert!(
        v.reason.contains("BELOW THE MLPERF-EDGE FLOOR"),
        "{}",
        v.reason
    );
}

/// 2026-09-26: Both gated draws couple the same params to the same metrics
/// (the echolp entry in kernels/gb10/qwen3.6-35b-a3b/BENCH.toml bounds both
/// metric names); bfcl-full declares none.
#[test]
fn the_gated_descriptors_declare_the_verdict_threshold_params() {
    let wired = [
        ("min_overall", "overall_accuracy"),
        ("min_normalized", "normalized_single_turn_score"),
    ];
    assert_eq!(SUBSET_DESCRIPTOR.threshold_params, wired);
    assert_eq!(SUBSET_ECHOLP_DESCRIPTOR.threshold_params, wired);
    assert!(FULL_DESCRIPTOR.threshold_params.is_empty());
    // 2026-09-26: The schema must carry the params (`apply_threshold_params`
    // errors otherwise), and their defaults must be non-gating.
    for variant in [Variant::Subset, Variant::SubsetEcholp, Variant::Full] {
        let v = ParamValues::defaults(&Bfcl::new(variant).parameters());
        assert_eq!(v.float("min_overall").unwrap(), 0.0);
        assert_eq!(v.float("min_normalized").unwrap(), 0.0);
    }
}

/// 2026-09-26: Both submission checkpoints, in either letter case, still fail
/// below the 85.32 normalized floor.
#[test]
fn a_36_submission_run_below_the_floor_still_fails() {
    for model in [
        "unsloth/Qwen3.6-27B-NVFP4",
        "centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf",
        "UNSLOTH/QWEN3.6-27B-NVFP4",
    ] {
        let mut b = on_model(Variant::Subset, model);
        b.scores = Some(scores(90.0, 85.31));
        let v = b.verdict();
        assert_eq!(v.kind, VerdictKind::Fail, "{model}: {}", v.reason);
        assert!(
            v.reason.contains("BELOW THE MLPERF-EDGE FLOOR"),
            "{}",
            v.reason
        );
    }
}

/// 2026-09-26: An unknown served model is not failed by the MLPerf floor.
#[test]
fn an_unknown_model_is_not_failed_by_the_floor() {
    let mut b = configured(Variant::Subset);
    b.scores = Some(scores(80.0, 80.0));
    let v = b.verdict();
    assert_eq!(v.kind, VerdictKind::Info, "{}", v.reason);
    assert!(
        v.reason.contains("judged by baseline thresholds"),
        "{}",
        v.reason
    );
}

/// 2026-09-26: The summary tiles style against the MLPerf floor for every
/// model, as a visual reference, even where the verdict does not gate on it.
#[test]
fn floor_styling_is_kept_for_non_submission_checkpoints() {
    use crate::result::CellStyle;
    let mut b = on_model(Variant::Subset, "unsloth/Qwen3.8-27B-NVFP4");
    b.scores = Some(scores(84.22, 84.12));
    let stats = b.summary();
    let overall = &stats[0];
    let normalized = &stats[1];
    assert_eq!(overall.style, CellStyle::Good, "84.22 >= floor 83.64");
    assert_eq!(normalized.style, CellStyle::Bad, "84.12 < floor 85.32");
}

#[test]
fn reconfiguring_clears_generated_responses() {
    let mut b = configured(Variant::Subset);
    b.responses.push(serde_json::json!({"sample_id": "x"}));
    b.cursor = 7;
    b.tool_call_samples = 3;
    let v = ParamValues::defaults(&b.parameters());
    b.configure(&v).unwrap();
    assert!(b.responses.is_empty() && b.cursor == 0 && b.tool_call_samples == 0);
}

/// 2026-09-26: The metrics always carry how many known partition-sensitive
/// samples the run scored, zero included, so an absent key cannot be mistaken
/// for a clean run.
#[test]
fn the_known_partition_sensitive_count_is_always_a_metric() {
    let mut b = configured(Variant::Subset);
    b.scores = Some(scores(84.0, 84.0));
    assert_eq!(b.metrics().get("known_partition_sensitive"), Some(&0.0));
    b.known_sensitive_seen = 2;
    assert_eq!(b.metrics().get("known_partition_sensitive"), Some(&2.0));
    // 2026-09-26: Reconfiguring resets the count with the responses.
    let v = ParamValues::defaults(&b.parameters());
    b.configure(&v).unwrap();
    assert_eq!(b.known_sensitive_seen, 0);
}

/// 2026-09-26: Every committed baseline of a gated variant pins the `samples`
/// bound to the variant's `expected_samples`, exactly (`min == max`, no
/// noise). The bounds come from each model's BENCH.toml (`gate::read_baseline`).
/// A one-sided `min` would accept the full 3625-sample draw against subset
/// thresholds.
#[test]
fn the_committed_baselines_pin_the_draw_each_variant_makes() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root is two levels above the crate");

    for variant in [Variant::Subset, Variant::SubsetEcholp] {
        let id = variant.descriptor().id;
        let want = variant
            .expected_samples()
            .expect("a gated variant has a pinned draw") as f64;
        let baseline = crate::gate::read_baseline(root, id)
            .unwrap_or_else(|e| panic!("{id}: baseline does not load: {e:#}"));
        for (hw, entry) in &baseline.hardware {
            for (model, mb) in &entry.models {
                let bound = mb.metrics.get("samples").unwrap_or_else(|| {
                    panic!(
                        "{id}/{hw}/{model}: no `samples` bound — a score from any draw \
                         would be accepted against these thresholds"
                    )
                });
                assert_eq!(
                    (bound.min, bound.max),
                    (Some(want), Some(want)),
                    "{id}/{hw}/{model}: the draw must be pinned EXACTLY at {want}"
                );
                assert!(
                    bound.noise.is_none(),
                    "{id}/{hw}/{model}: a sample count is exact; noise would widen the pin"
                );
            }
        }
    }
}

#[test]
fn the_mlperf_floors_are_the_recorded_thresholds() {
    // 2026-09-26: 86.23 / 87.96 x 0.97, the derivation `report.rs` states.
    let rounded = |value: f64| (value * 100.0).round() / 100.0;
    assert_eq!(MLPERF_FLOOR_OVERALL, rounded(86.23 * 0.97));
    assert_eq!(MLPERF_FLOOR_NORMALIZED, rounded(87.96 * 0.97));
}
