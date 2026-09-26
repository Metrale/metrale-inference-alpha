// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Holds `metrale_core::arch::target_hint`, a hand-written `match`
//! from device compute capability to the `METRALE_TARGET_HW` to build, to the
//! `compute_capability` each NVIDIA `kernels/<hw>/HARDWARE.toml` declares.
//!
//! Owner: metrale-kernels tests.
//! Invariants: none beyond the types.
//!
//! metrale-core cannot read `kernels/`, so this test is what keeps the two in
//! agreement.

use std::path::{Path, PathBuf};

fn kernels_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/kernels is two levels below the workspace root")
        .join("kernels")
}

/// 2026-09-25: `(directory name, [hardware] table)` for every `kernels/<hw>/`
/// that has a HARDWARE.toml, sorted by name.
fn hardware_sets() -> Vec<(String, toml::Value)> {
    let mut sets: Vec<(String, toml::Value)> = std::fs::read_dir(kernels_root())
        .expect("kernels/ is in the tree")
        .flatten()
        .filter_map(|entry| {
            let path = entry.path().join("HARDWARE.toml");
            if !path.is_file() {
                return None;
            }
            let text = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            let toml: toml::Value = toml::from_str(&text)
                .unwrap_or_else(|e| panic!("bad TOML in {}: {e}", path.display()));
            let hw = toml
                .get("hardware")
                .unwrap_or_else(|| panic!("{}: no [hardware] table", path.display()))
                .clone();
            Some((entry.file_name().to_string_lossy().to_string(), hw))
        })
        .collect();
    sets.sort_by(|a, b| a.0.cmp(&b.0));
    sets
}

/// 2026-09-25: `"12.1"` -> `(12, 1)`; panics on anything else. The strix
/// trees' `"11.5.1"` never reaches it: their vendor is not `nvidia`.
fn parse_cc(text: &str, hw: &str) -> (u32, u32) {
    let (major, minor) = text.split_once('.').unwrap_or_else(|| {
        panic!("kernels/{hw}: compute_capability {text:?} is not `major.minor`")
    });
    (
        major
            .parse()
            .unwrap_or_else(|e| panic!("kernels/{hw}: compute_capability major {major:?}: {e}")),
        minor
            .parse()
            .unwrap_or_else(|e| panic!("kernels/{hw}: compute_capability minor {minor:?}: {e}")),
    )
}

/// 2026-09-25: Every NVIDIA hardware set declares a compute capability, and
/// `target_hint` maps it back to that set's directory name.
#[test]
fn every_nvidia_hardware_set_is_reachable_from_its_declared_compute_capability() {
    let mut checked = Vec::new();
    for (dir, hw) in hardware_sets() {
        if hw.get("vendor").and_then(|v| v.as_str()) != Some("nvidia") {
            continue;
        }
        let cc_text = hw
            .get("compute_capability")
            .and_then(|v| v.as_str())
            .unwrap_or_else(|| {
                panic!(
                    "kernels/{dir}/HARDWARE.toml is an NVIDIA target with no \
                     compute_capability; metrale_core::arch::target_hint cannot be \
                     held to a value that is not declared"
                )
            });
        let cc = parse_cc(cc_text, &dir);
        assert_eq!(
            metrale_core::arch::target_hint(cc),
            Some(dir.as_str()),
            "kernels/{dir}/HARDWARE.toml declares compute_capability {cc_text:?}, \
             but metrale_core::arch::target_hint({cc:?}) does not name {dir:?} — \
             add the arm in crates/core/src/arch.rs"
        );
        checked.push(dir);
    }
    assert!(
        checked.len() >= 2,
        "found {checked:?} — fewer NVIDIA hardware sets than the tree has, so the \
         walk is broken rather than the hints"
    );
}
/// 2026-09-25: Each listed `match` arm names a hardware directory that exists.
#[test]
fn every_target_hint_names_a_hardware_set_that_exists() {
    // 2026-09-25: The CCs the `match` arms answer for, listed by hand. One that
    // stops being hinted panics here.
    for cc in [(9, 0), (10, 0), (10, 3), (12, 1)] {
        let hw = metrale_core::arch::target_hint(cc)
            .unwrap_or_else(|| panic!("target_hint({cc:?}) answered None"));
        assert!(
            kernels_root().join(hw).join("HARDWARE.toml").is_file(),
            "target_hint({cc:?}) says to build METRALE_TARGET_HW={hw}, but \
             kernels/{hw}/HARDWARE.toml does not exist"
        );
    }
}
