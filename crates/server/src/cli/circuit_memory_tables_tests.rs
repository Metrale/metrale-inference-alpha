// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: The expert-table decision on the in-tree checkpoint fixtures: the NVFP4 35B recipe
//! keeps its tables, the 122B recipe at util 0.85 drops them and then fits, a checkpoint without
//! such tables decides nothing, and a CLI what-if never moves the decision.
//!
//! Owner: server CLI.
//! Invariants: none beyond the types.

use metrale_model_layers::layers::MoeExpertTables;

use super::super::circuit_memory::circuit_memory_ledger_tests::point;
use super::{SETTING, TablesDecision};

const NVFP4_35B: (&str, &str) = (
    "nvidia/Qwen3.6-35B-A3B-NVFP4",
    "qwen3.6/qwen3.6-35b-a3b-nvfp4",
);
const Q122B: (&str, &str) = (
    "Sehyo/Qwen3.5-122B-A10B-NVFP4",
    "qwen3.5/qwen3.5-122b-a10b-nvfp4-single",
);

fn util(u: &str) -> Vec<String> {
    vec!["--gpu-memory-utilization".into(), u.into()]
}

fn decision(ck: (&str, &str), serve: Vec<String>) -> (Option<TablesDecision>, String) {
    let p = point(ck.0, ck.0, Some(ck.1.into()), serve, 0);
    (p.tables, p.settings[SETTING].clone())
}

/// 2026-10-02: Path A: the NVFP4 35B recipe builds its tables, as it always has.
#[test]
fn the_nvfp4_35b_recipe_keeps_its_tables() {
    let (d, setting) = decision(NVFP4_35B, util("0.85"));
    let d = d.expect("the NVFP4 35B plan has expert tables");
    assert_eq!(d.tables, MoeExpertTables::Build, "{}", d.describe());
    assert!(d.bytes > 16 << 30, "{} bytes of tables", d.bytes);
    assert!(d.headroom_with >= 0 && d.headroom_without.is_none());
    assert_eq!(setting, "build");
}

/// 2026-10-02: Path B: the 122B single-GB10 recipe at util 0.85 does not fit with its tables;
/// the decision skips them, and the plan without them fits.
#[test]
fn the_122b_recipe_skips_its_tables_and_fits() {
    let (d, setting) = decision(Q122B, util("0.85"));
    let d = d.expect("the 122B plan has expert tables");
    assert_eq!(d.tables, MoeExpertTables::Skip, "{}", d.describe());
    assert!(d.headroom_with < 0, "{}", d.describe());
    assert!(
        d.headroom_without.is_some_and(|h| h >= 0),
        "{}",
        d.describe()
    );
    assert_eq!(setting, "skip");
    // 2026-10-02: Every later evaluation of the point is the skipped plan.
    let p = point(Q122B.0, Q122B.0, Some(Q122B.1.into()), util("0.85"), 0);
    let (r, _) = p.eval(1, 1000, 1).unwrap();
    assert!(
        r.weights
            .iter()
            .flat_map(|w| &w.derived)
            .all(|c| c.rule != "moe-nvfp4-expert-twin")
    );
}

/// 2026-10-02: A checkpoint whose plan has no tables (an FP8 MoE, a Nemotron) decides nothing.
#[test]
fn no_tables_no_decision() {
    for ck in [
        (
            "Qwen/Qwen3.6-35B-A3B-FP8",
            "qwen3.6/qwen3.6-35b-a3b-fp8-mtp",
        ),
        (
            "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4",
            "nemotron-3-nano/nemotron-3-nano-30b-a3b-nvfp4",
        ),
    ] {
        assert_eq!(decision(ck, util("0.85")).0, None, "{}", ck.0);
    }
}

/// 2026-10-02: The decision reads only the serve's flags: a CLI what-if (a token tree, explicit
/// slots, drafter capture rows) leaves it as the plain serve's.
#[test]
fn a_cli_what_if_never_moves_the_decision() {
    let plain = decision(Q122B, util("0.85")).0;
    let tree = point(Q122B.0, Q122B.0, Some(Q122B.1.into()), util("0.85"), 4096).tables;
    assert_eq!(plain, tree);
}
