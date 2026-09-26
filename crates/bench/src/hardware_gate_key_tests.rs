// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests of `Hardware::gate_key`, the key that picks which
//! hardware entry of a gate baseline a record is scored against.
//!
//! Owner: bench (hardware).
//! Invariants: none beyond the types.

use super::Hardware;

fn gpu(name: &str) -> Hardware {
    Hardware {
        gpu: name.to_string(),
        ..Hardware::default()
    }
}

#[test]
fn a_gb10_normalises_to_the_key_the_baselines_use() {
    // 2026-09-26: `gate::bench` keys a baseline by the hardware directory of
    // its `kernels/<hw>/<model>/BENCH.toml`, here `gb10`. Another key makes
    // `GateBaseline::resolve` fail with "no baseline for hardware ...".
    assert_eq!(gpu("NVIDIA GB10").gate_key(), "gb10");
    assert_eq!(gpu("GB10").gate_key(), "gb10");
    assert_eq!(gpu("nvidia gb10").gate_key(), "gb10");
}

#[test]
fn a_box_that_reports_nothing_is_named_unknown_not_empty() {
    // 2026-09-26: `fetch_hardware` returns an unknown `Hardware` on every
    // error path without surfacing the error.
    assert_eq!(Hardware::default().gate_key(), "unknown");
    assert_eq!(gpu("").gate_key(), "unknown");
    assert_eq!(gpu("- / .").gate_key(), "unknown");
}

#[test]
fn parts_that_differ_must_not_collapse_onto_one_key() {
    // 2026-09-26: Distinct generations and capacities get distinct keys, and
    // none is "unknown".
    let keys = [
        gpu("NVIDIA GB10").gate_key(),
        gpu("NVIDIA GB200").gate_key(),
        gpu("NVIDIA A100-SXM4-40GB").gate_key(),
        gpu("NVIDIA A100-SXM4-80GB").gate_key(),
        gpu("AMD Radeon 8060S (gfx1151)").gate_key(),
    ];
    for (i, a) in keys.iter().enumerate() {
        for b in &keys[i + 1..] {
            assert_ne!(a, b, "distinct parts must not share a gate key");
        }
        assert_ne!(a, "unknown", "a named part must not read as unknown");
    }
}

/// 2026-09-26: Marketing SKU names map through the SKU table
/// (`ids::hardware_id_from_gpu_name`) onto the registered ids `h100`, `h200`
/// and `gh200`; GB10 stays `gb10`.
#[test]
fn a_hopper_sku_keys_onto_the_slot_the_resolver_accepts() {
    assert_eq!(gpu("NVIDIA H100 80GB HBM3").gate_key(), "h100");
    assert_eq!(gpu("NVIDIA H100 PCIe").gate_key(), "h100");
    assert_eq!(gpu("NVIDIA H200 NVL").gate_key(), "h200");
    assert_eq!(gpu("NVIDIA GH200 480GB").gate_key(), "gh200");
    assert_eq!(gpu("NVIDIA GB10").gate_key(), "gb10");
}
