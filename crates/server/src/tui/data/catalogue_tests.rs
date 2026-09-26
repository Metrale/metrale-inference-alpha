// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests of the Library row model: `join`, the row badges, sizes,
//! the filter and the subtitle.
//!
//! Owner: server tui.
//! Invariants: none beyond the types.

use super::*;
use std::collections::BTreeMap;

fn recipe(id: &str, model: &str, runtime: &str) -> Recipe {
    Recipe {
        id: id.into(),
        version: "2".into(),
        model: model.into(),
        runtime: Some(runtime.into()),
        container: "c".into(),
        min_nodes: 1,
        description: "d".into(),
        maintainer: "metrale".into(),
        category: "agent".into(),
        model_params: "27B".into(),
        quantization: "nvfp4".into(),
        kv_dtype: "bf16".into(),
        updated: "2026-08-01".into(),
        defaults: BTreeMap::from([("port".into(), "8888".into())]),
        env: BTreeMap::new(),
        starting_point: None,
    }
}

fn local(id: &str, has_weights: bool, optimized: bool) -> LibraryEntry {
    LibraryEntry {
        id: id.into(),
        snapshot_dir: Default::default(),
        size_bytes: 20 * 1024 * 1024 * 1024,
        has_weights,
        model_type: "qwen3_5".into(),
        quant: "nvfp4".into(),
        layers: 64,
        hidden: 5120,
        heads: 40,
        experts: 0,
        context: 262144,
        optimized,
    }
}

#[test]
fn a_recipe_with_local_weights_is_runnable_and_sorts_first() {
    let rows = join(
        &[
            recipe("a/no-weights", "org/absent", "metrale"),
            recipe("b/here", "org/present", "metrale"),
        ],
        &[local("org/present", true, true)],
    );
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].model, "org/present", "runnable rows come first");
    assert!(rows[0].runnable_now());
    assert!(!rows[1].runnable_now());
    assert!(rows[1].has_recipe(), "still listed, just not runnable yet");
}

#[test]
fn a_local_checkpoint_without_a_recipe_is_still_listed() {
    let rows = join(&[], &[local("org/orphan", true, false)]);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].model, "org/orphan");
    assert!(!rows[0].has_recipe());
    assert!(rows[0].has_weights());
}

#[test]
fn local_only_rows_sort_last() {
    let rows = join(
        &[recipe("a/x", "org/with-recipe", "metrale")],
        &[
            local("org/orphan", true, false),
            local("org/with-recipe", true, false),
        ],
    );
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].model, "org/with-recipe");
    assert_eq!(rows[1].model, "org/orphan");
}

#[test]
fn optimized_is_independent_of_having_a_recipe() {
    let rows = join(
        &[recipe("a/x", "org/m", "metrale")],
        &[local("org/m", true, false)],
    );
    assert!(rows[0].has_recipe());
    assert!(!rows[0].optimized(), "recipe does not imply optimized");

    let rows = join(&[], &[local("org/m", true, true)]);
    assert!(!rows[0].has_recipe());
    assert!(rows[0].optimized(), "optimized does not imply a recipe");
}

#[test]
fn a_vllm_recipe_is_listed_but_not_runnable() {
    let rows = join(
        &[recipe("d/vllm", "org/m", "vllm")],
        &[local("org/m", true, true)],
    );
    assert_eq!(rows.len(), 1);
    assert!(rows[0].has_recipe());
    assert!(rows[0].has_weights());
    assert!(
        !rows[0].runnable_now(),
        "weights alone do not make a vLLM recipe servable here"
    );
}

#[test]
fn several_recipes_for_one_model_become_one_row() {
    let rows = join(
        &[
            recipe("q/nvfp4", "org/m", "metrale"),
            recipe("q/fp8", "org/m", "metrale"),
        ],
        &[local("org/m", true, true)],
    );
    assert_eq!(rows.len(), 1, "one model, one row");
    assert_eq!(rows[0].recipes.len(), 2, "both recipes are still reachable");
    // 2026-09-26: Sorted by recipe id, whatever the input order.
    assert_eq!(rows[0].recipes[0].id, "q/fp8");
    assert_eq!(rows[0].recipes[1].id, "q/nvfp4");
}

#[test]
fn the_primary_recipe_prefers_one_that_can_actually_run() {
    let rows = join(
        &[
            recipe("a/vllm", "org/m", "vllm"),
            recipe("b/metrale", "org/m", "metrale"),
        ],
        &[local("org/m", true, true)],
    );
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].primary().expect("a primary").id, "b/metrale");
    assert!(
        rows[0].runnable_now(),
        "the metrale sibling makes it runnable"
    );
}

#[test]
fn a_model_whose_only_recipe_is_vllm_is_described_by_it_but_not_runnable() {
    let rows = join(
        &[recipe("a/vllm", "org/m", "vllm")],
        &[local("org/m", true, true)],
    );
    assert_eq!(rows[0].primary().expect("a primary").id, "a/vllm");
    assert!(!rows[0].runnable_now());
}

#[test]
fn partial_weights_are_named_rather_than_shown_as_a_size() {
    let rows = join(&[], &[local("org/m", false, false)]);
    assert_eq!(rows[0].size_text(), "partial");
    assert!(!rows[0].has_weights());

    let rows = join(&[recipe("a/x", "org/absent", "metrale")], &[]);
    assert_eq!(rows[0].size_text(), "—", "absent weights are not 0 B");
}

#[test]
fn the_filter_matches_id_recipe_and_architecture() {
    let rows = join(
        &[recipe("qwen3.6/flagship", "Qwen/Qwen3.6-27B", "metrale")],
        &[local("Qwen/Qwen3.6-27B", true, true)],
    );
    let row = &rows[0];
    assert!(row.matches(""), "an empty filter matches everything");
    assert!(
        row.matches("qwen3.6-27b"),
        "the model id, case-insensitively"
    );
    assert!(
        row.matches("flagship"),
        "a recipe id the row now hides must still be findable"
    );
    assert!(row.matches("qwen3_5"), "the architecture");
    assert!(!row.matches("llama"));
}

#[test]
fn the_subtitle_does_not_repeat_itself() {
    let rows = join(
        &[recipe("a/x", "org/m", "metrale")],
        &[local("org/m", true, true)],
    );
    let subtitle = rows[0].subtitle();
    assert!(subtitle.contains("27B"), "{subtitle}");
    assert!(subtitle.contains("qwen3_5"), "{subtitle}");
    assert!(subtitle.contains("64L"), "{subtitle}");
}
