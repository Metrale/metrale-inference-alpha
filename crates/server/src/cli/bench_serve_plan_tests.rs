// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for `disclosed_from`: the serve knobs a gate record
//! discloses, read off a rendered fixture recipe.
//!
//! Owner: server CLI (`met benchmark`).
//! Invariants: none beyond the types.

use super::disclosed_from;
use crate::recipe::Recipe;
use std::collections::BTreeMap;

fn recipe(defaults: &str) -> Recipe {
    let text = format!(
        "recipe_version: \"2\"\nmodel: org/model\ncontainer: metrale\nruntime: metrale\n\
         metadata:\n  updated: \"2026-08-28\"\ndefaults:\n{defaults}"
    );
    Recipe::parse("fam/stem", &text).expect("the fixture recipe parses")
}

fn disclosed(defaults: &str, overrides: &[(&str, &str)]) -> Vec<(String, String)> {
    let overrides: BTreeMap<String, String> = overrides
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    let args = recipe(defaults)
        .serve_args(&overrides)
        .unwrap_or_else(|e| panic!("the fixture renders: {e:#}"));
    disclosed_from(&args).into_iter().collect()
}

fn pairs(v: &[(&str, &str)]) -> Vec<(String, String)> {
    v.iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

/// 2026-09-26: The disclosure is what the rendered command resolved: the
/// recipe as pinned, an override on top, and `--hermetic` forcing the gate.
#[test]
fn the_disclosure_is_read_off_the_rendered_serve() {
    // 2026-09-26: A recipe that pins `mtp_gate: force` discloses it.
    assert_eq!(
        disclosed("  speculative: \"true\"\n  mtp_gate: force\n", &[]),
        pairs(&[("mtp_gate", "force"), ("speculative", "true")])
    );
    // 2026-09-26: Speculation on and no `mtp_gate` pinned: no `mtp_gate` is
    // disclosed, rather than `auto`.
    assert_eq!(
        disclosed("  speculative: \"true\"\n", &[]),
        pairs(&[("speculative", "true")])
    );
    // 2026-09-26: An explicit `auto` is disclosed.
    assert_eq!(
        disclosed("  speculative: \"true\"\n  mtp_gate: auto\n", &[]),
        pairs(&[("mtp_gate", "auto"), ("speculative", "true")])
    );
    // 2026-09-26: A `--serve-override mtp_gate=force` on an `auto` recipe wins.
    assert_eq!(
        disclosed(
            "  speculative: \"true\"\n  mtp_gate: auto\n",
            &[("mtp_gate", "force")]
        ),
        pairs(&[("mtp_gate", "force"), ("speculative", "true")])
    );
    // 2026-09-26: `--hermetic` forces the gate where the recipe pinned nothing;
    // the disclosure follows `ServeArgs::mtp_gate_force`, not the raw flag.
    assert_eq!(
        disclosed("  speculative: \"true\"\n", &[("hermetic", "true")]),
        pairs(&[("mtp_gate", "force"), ("speculative", "true")])
    );
    // 2026-09-26: No speculation: only `speculative = false` is disclosed.
    assert_eq!(
        disclosed("  max_batch_size: \"8\"\n", &[]),
        pairs(&[("speculative", "false")])
    );
}

/// 2026-09-26: A `prefill_codispatch = "true"` serve override renders
/// `--prefill-codispatch`, and the disclosure carries `true`.
#[test]
fn the_codispatch_flag_is_disclosed_off_the_rendered_serve() {
    assert_eq!(
        disclosed(
            "  max_batch_size: \"8\"\n",
            &[("prefill_codispatch", "true")]
        ),
        pairs(&[("prefill_codispatch", "true"), ("speculative", "false")])
    );
    // 2026-09-26: An override's `false` removes the recipe's `true`: the flag is
    // not rendered, so nothing is disclosed.
    assert_eq!(
        disclosed(
            "  prefill_codispatch: \"true\"\n",
            &[("prefill_codispatch", "false")]
        ),
        pairs(&[("speculative", "false")])
    );
}

/// 2026-09-26: `--w4a4-downcast` is disclosed when on; off discloses nothing.
#[test]
fn the_w4a4_downcast_flag_is_disclosed_off_the_rendered_serve() {
    assert_eq!(
        disclosed("  max_batch_size: \"8\"\n", &[("w4a4_downcast", "true")]),
        pairs(&[("speculative", "false"), ("w4a4_downcast", "true")])
    );
    assert_eq!(
        disclosed("  w4a4_downcast: \"true\"\n", &[("w4a4_downcast", "false")]),
        pairs(&[("speculative", "false")])
    );
}
