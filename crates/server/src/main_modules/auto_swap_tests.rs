// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for `auto_swap::decide` and the `--auto-swap` switch.
//!
//! Owner: server (model hosting).
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
        maintainer: "m".into(),
        category: "agent".into(),
        model_params: "27B".into(),
        quantization: "nvfp4".into(),
        kv_dtype: "bf16".into(),
        updated: "2026-08-01".into(),
        defaults: BTreeMap::new(),
        env: BTreeMap::new(),
        starting_point: None,
    }
}

fn catalogue() -> Vec<Recipe> {
    vec![
        recipe("q/27b", "nvidia/Qwen3.6-27B-NVFP4", "metrale"),
        recipe("q/35b", "Qwen/Qwen3.6-35B-A3B-FP8", "metrale"),
        recipe("d/vllm", "org/vllm-only", "vllm"),
    ]
}

const LIVE: &str = "nvidia/Qwen3.6-27B-NVFP4";

#[test]
fn a_different_known_model_swaps() {
    assert_eq!(
        decide("Qwen/Qwen3.6-35B-A3B-FP8", LIVE, &catalogue()),
        Decision::SwapTo("q/35b".into())
    );
}

#[test]
fn the_live_model_does_not_swap() {
    assert_eq!(decide(LIVE, LIVE, &catalogue()), Decision::ServeCurrent);
}

#[test]
fn an_unknown_model_is_ignored_not_an_error() {
    assert_eq!(
        decide("does/not-exist", LIVE, &catalogue()),
        Decision::ServeCurrent
    );
}

#[test]
fn an_absent_or_blank_model_is_ignored() {
    assert_eq!(decide("", LIVE, &catalogue()), Decision::ServeCurrent);
    assert_eq!(decide("   ", LIVE, &catalogue()), Decision::ServeCurrent);
}

/// 2026-09-26: A recipe for another runtime cannot be launched here, so it is
/// never a swap target.
#[test]
fn a_vllm_recipe_is_never_a_swap_target() {
    assert_eq!(
        decide("org/vllm-only", LIVE, &catalogue()),
        Decision::ServeCurrent
    );
}

#[test]
fn matching_is_exact_not_fuzzy() {
    assert_eq!(
        decide("Qwen/Qwen3.6-35B", LIVE, &catalogue()),
        Decision::ServeCurrent,
        "a prefix of a known id is not that id"
    );
    assert_eq!(
        decide("nvidia/Qwen3.6-27B-NVFP4-extra", LIVE, &catalogue()),
        Decision::ServeCurrent
    );
}

#[test]
fn an_empty_catalogue_never_swaps() {
    assert_eq!(decide("anything", LIVE, &[]), Decision::ServeCurrent);
}

mod policy {
    use super::super::enabled;
    use crate::cli;
    use clap::Parser as _;

    fn args(extra: &[&str]) -> cli::ServeArgs {
        let mut argv = vec!["met", "serve", "m"];
        argv.extend_from_slice(extra);
        match cli::Cli::parse_from(argv).command {
            cli::Command::Serve(a) => a,
            cli::Command::Benchmark(_)
            | cli::Command::DumpServeOptions
            | cli::Command::SyncRecipes
            | cli::Command::Doctor => unreachable!(),
        }
    }

    #[test]
    fn request_swapping_is_off_unless_asked_for() {
        assert!(!enabled(&args(&[])));
        assert!(enabled(&args(&["--auto-swap"])));
    }

    #[test]
    fn there_is_no_deny_twin() {
        let err = cli::Cli::try_parse_from(["met", "serve", "m", "--no-auto-swap"])
            .expect_err("--no-auto-swap no longer exists");
        assert_eq!(err.kind(), clap::error::ErrorKind::UnknownArgument);
    }
}
