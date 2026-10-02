// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: `met circuit precision` on the checked-out tree: one MoE layer of each
//! Qwen3.6-35B-A3B checkpoint lists the pipeline each node needs and what runs it (the NVFP4
//! experts and shared expert on the grouped tensor-core NVFP4 point, the FP8 recipe's fused
//! expert kernels), the NVFP4 recipe's plans have no gap at any planned width, and a glob that
//! matches no node is an error.
//!
//! Owner: server CLI tests.
//! Invariants: the tests read the repository at the workspace root and write nothing.

use std::path::PathBuf;

use super::listing;
use crate::cli::{CircuitMode, CircuitPrecision, CircuitPrecisionArgs};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("workspace root")
}

fn args(checkpoint: String, precision: CircuitPrecision, node: &str) -> CircuitPrecisionArgs {
    CircuitPrecisionArgs {
        checkpoint,
        precision,
        node: Some(node.to_string()),
        hardware: "gb10".into(),
        mode: CircuitMode::Decode,
        rows: None,
        allow_network: false,
        root: None,
    }
}

fn fixture(name: &str) -> String {
    root()
        .join(super::super::circuit_hw::MATRIX_CONFIGS)
        .join(name)
        .display()
        .to_string()
}

/// 2026-10-02: The node lines of a listing, without the header and the `<-` lines.
fn nodes(text: &str) -> Vec<&str> {
    text.lines()
        .filter(|l| !l.starts_with('#') && !l.starts_with("    <- "))
        .collect()
}

const W4A16: &str =
    "act bf16 | weight nvfp4/g16->bf16 | mma bf16*bf16 | accumulate f32 | scale f32";
const BF16: &str = "act bf16 | weight bf16->bf16 | mma bf16*bf16 | accumulate f32 | scale none";

#[test]
fn the_nvfp4_35b_moe_layer_runs_its_w4a16_experts_on_the_grouped_tensor_core_point() {
    let a = args(
        fixture("nvidia--Qwen3.6-35B-A3B-NVFP4"),
        CircuitPrecision::Declared,
        "l3.moe_ffn.*",
    );
    let text = listing(&root(), &a).unwrap();
    let want = [
        "l3.moe_ffn.post_norm rms_norm: f32 (rule) -> [compute f32] -> bf16".to_string(),
        format!("l3.moe_ffn.router router: bf16 -> [{BF16}] -> bf16"),
        "l3.moe_ffn.top_k top_k: bf16 -> [score f32] -> f32,i32".into(),
        format!(
            "l3.moe_ffn.experts_gate_up expert_gate_up: bf16,i32 -> [gather bf16 | {W4A16}] -> bf16"
        ),
        "l3.moe_ffn.experts_act silu_mul: bf16 -> [compute f32] -> bf16".into(),
        format!("l3.moe_ffn.shared_gate_up linear:shared_gate_up: bf16 -> [{W4A16}] -> bf16"),
        "l3.moe_ffn.shared_act silu_mul: bf16 -> [compute f32] -> bf16".into(),
        format!("l3.moe_ffn.experts_down expert_down: bf16,i32 -> [{W4A16}] -> bf16"),
        format!("l3.moe_ffn.shared_down linear:shared_down: bf16 -> [{W4A16}] -> bf16"),
        format!("l3.moe_ffn.shared_gate linear:shared_gate: bf16 -> [{BF16}] -> f32 (rule)"),
        "l3.moe_ffn.blend blend: bf16,f32,bf16,f32 (rule) -> [scatter f32 | combine f32] -> bf16"
            .into(),
        "l3.moe_ffn.add residual_add: bf16,bf16 -> [compute f32] -> bf16".into(),
    ];
    // 2026-10-02: Plan order: the gate+up launch group (routed and shared, with both SiLUs),
    // then the down group.
    assert_eq!(nodes(&text), want);
    assert_eq!(text.matches("<- gap:").count(), 0, "{text}");
    for kernel in [
        "<- moe_gate_up_act_grouped_nvfp4_tc: moe_nvfp4_grouped_tc::moe_expert_gate_up_act_nvfp4_grouped_tc",
        "<- moe_down_act_grouped_nvfp4_tc: moe_nvfp4_grouped_tc::moe_expert_down_act_nvfp4_grouped_tc",
    ] {
        assert!(text.contains(kernel), "{kernel} missing:\n{text}");
    }
}

/// 2026-10-02: Every node of the NVFP4 35B recipe (`qwen3.6-35b-a3b-nvfp4-declared`) has a kernel
/// whose declared pipeline is the required one, at every mode and a spread of the instance's
/// planned widths: the listing errors on a refused pipeline, and a gap prints a `gap:` line.
#[test]
fn the_nvfp4_35b_recipe_plans_have_no_gap() {
    let cases = [
        (CircuitMode::Decode, 1),
        (CircuitMode::Draft, 1),
        (CircuitMode::Verify, 2),
        (CircuitMode::Verify, 4),
        (CircuitMode::MultiSeq, 2),
        (CircuitMode::MultiSeq, 64),
        (CircuitMode::MultiSeq, 128),
    ];
    for (mode, rows) in cases {
        let mut a = args(
            "nvidia/Qwen3.6-35B-A3B-NVFP4".into(),
            CircuitPrecision::Recipe,
            "*",
        );
        a.mode = mode;
        a.rows = Some(rows);
        let text = listing(&root(), &a).unwrap_or_else(|e| panic!("{mode:?} {rows}: {e:#}"));
        assert!(
            text.contains("lm_head_dtype=nvfp4"),
            "{mode:?} {rows}: not the recipe"
        );
        assert_eq!(
            text.matches("<- gap:").count(),
            0,
            "{mode:?} {rows}:\n{text}"
        );
    }
}

#[test]
fn the_fp8_35b_recipe_moe_layer_lists_its_fused_expert_kernels() {
    let a = args(
        "Qwen/Qwen3.6-35B-A3B-FP8".into(),
        CircuitPrecision::Recipe,
        "l3.moe_ffn.experts_*",
    );
    let text = listing(&root(), &a).unwrap();
    let w8 = "weight fp8/block128x128";
    assert_eq!(
        nodes(&text),
        [
            format!(
                "l3.moe_ffn.experts_gate_up expert_gate_up: bf16,i32 -> [gather bf16 | act bf16 | {w8}->bf16 | mma bf16*bf16 | accumulate f32 | scale f32] -> bf16"
            ),
            "l3.moe_ffn.experts_act silu_mul: bf16 -> [compute f32] -> f32".into(),
            format!(
                "l3.moe_ffn.experts_down expert_down: f32,i32 -> [act f32 | {w8}->f32 | mma f32*f32 | accumulate f32 | scale f32] -> bf16"
            ),
        ]
    );
    assert!(text.contains(
        "    <- moe_silu_down_shared_fp8: moe_shared_expert_fused_fp8::moe_expert_silu_down_shared_fp8\n"
    ));
    assert!(text.contains("    <- moe_silu_down_shared_fp8: the launch group above\n"));
}

#[test]
fn a_glob_that_matches_no_node_is_an_error() {
    let a = args(
        "Qwen/Qwen3.6-35B-A3B-FP8".into(),
        CircuitPrecision::Recipe,
        "l3.mamba.*",
    );
    let e = listing(&root(), &a).unwrap_err().to_string();
    assert!(
        e.contains("no node of the decode n=1 plan matches `l3.mamba.*`"),
        "{e}"
    );
}

// 2026-10-02: The command line: the checkpoint is required; the declared formats, gb10 and one
// decode row are what an unqualified request lists.
#[test]
fn the_command_line_parses_with_its_stated_defaults() {
    use clap::Parser;
    let parse = |argv: &[&str]| {
        let mut all = vec!["met", "circuit", "precision"];
        all.extend_from_slice(argv);
        crate::cli::Cli::try_parse_from(all)
    };
    let cli = parse(&[
        "--checkpoint",
        "nvidia/Qwen3.6-35B-A3B-NVFP4",
        "--node",
        "l3.moe_ffn.*",
    ])
    .unwrap();
    let crate::cli::Command::Circuit(c) = cli.command else {
        panic!("not a circuit command");
    };
    let crate::cli::CircuitAction::Precision(p) = c.action else {
        panic!("not precision");
    };
    assert_eq!(p.precision, CircuitPrecision::Declared);
    assert_eq!(
        (p.hardware.as_str(), p.mode, p.rows),
        ("gb10", CircuitMode::Decode, None)
    );
    assert_eq!(p.node.as_deref(), Some("l3.moe_ffn.*"));
    assert!(parse(&["--node", "l3.*"]).is_err());
}
