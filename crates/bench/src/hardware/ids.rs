// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Box-class ids, the strings a gate baseline is keyed by, and the map
//! from a GPU's reported name to its id.
//!
//! Owner: bench hardware.
//! Invariants:
//! - Every id [`hardware_id_from_gpu_name`] returns is in [`KNOWN_HARDWARE_IDS`]
//!   (`every_id_the_sku_table_produces_is_registered`).
//!
//! A benchmark's thresholds live under `kernels/<hw>/<model>/BENCH.toml` and are
//! assembled into `GateBaseline.hardware`. A run's key comes from `--hardware` or
//! from a record's fingerprint via [`super::Hardware::gate_key`]. The baseline
//! holds only what has been measured, so it cannot tell a registered class with no
//! record yet from a misspelling; [`KNOWN_HARDWARE_IDS`] can, and the resolver
//! gives the two different refusals.

/// 2026-09-26: Every box class the engine recognises, whether or not any gate has
/// a record on it yet. `--hardware` is validated against it (`bench_resolve.rs`).
///
/// Two kinds of id live here:
///
/// * **architecture classes**: `gb10`, `hopper`, `b200`, `b300`, `metal`, `strix`,
///   `strix-hip`, the `kernels/<hw>/` directories that hold a `HARDWARE.toml`
///   (`every_kernel_hardware_dir_is_registered`);
/// * **bench SKUs**: the ids [`hardware_id_from_gpu_name`] maps a GPU name onto,
///   which a record's fingerprint keys by.
///
/// They coincide for `gb10`, `b200` and `b300`.
pub const KNOWN_HARDWARE_IDS: [&str; 12] = [
    // 2026-09-26: NVIDIA GB10: architecture class and SKU at once.
    "gb10",
    // 2026-09-26: NVIDIA Hopper, the architecture class: `kernels/hopper/` (sm_90a).
    "hopper",
    // 2026-09-26: NVIDIA H100 and H200: separate baseline keys.
    "h100",
    "h200",
    // 2026-09-26: NVIDIA GH200: its own baseline key.
    "gh200",
    // 2026-09-26: `b200` is both the `kernels/b200/` class (sm_100a) and the B200
    // SKU. GB200 has its own baseline key.
    "b200",
    "gb200",
    // 2026-09-26: `b300` is both the `kernels/b300/` class (sm_103a) and the B300
    // SKU. GB300 is not registered and maps to no id.
    "b300",
    // 2026-09-26: Apple Silicon, via the Metal backend (`kernels/metal/`).
    "metal",
    // 2026-09-26: AMD Strix Halo (gfx1151) through two backends, each its own
    // class: `strix` builds the CUDA sources with SCALE (`kernels/strix/`), and
    // `strix-hip` builds them with hipcc (`kernels/strix-hip/`).
    "strix",
    "strix-hip",
    // 2026-09-26: AMD Instinct MI300X. No `kernels/` directory; registered because
    // `hardware_id_from_gpu_name` maps the SKU onto it.
    "mi300x",
];

/// 2026-09-26: Every GPU-name token that names a box class, and the class it names.
///
/// Matched as a whole token, never a substring: `gh200` contains `h200`, and a
/// substring match would file every Grace-Hopper run under the H200 baseline.
///
/// A name with no listed token falls through to [`super::Hardware::gate_key`]'s
/// normalisation, which keeps capacity and generation in the key
/// (`NVIDIA A100-SXM4-80GB` → `a100sxm480gb`).
const SKU_TOKENS: [(&str, &str); 8] = [
    ("gb10", "gb10"),
    ("h100", "h100"),
    ("h200", "h200"),
    ("gh200", "gh200"),
    ("b200", "b200"),
    ("gb200", "gb200"),
    ("b300", "b300"),
    ("mi300x", "mi300x"),
];

/// 2026-09-26: Map a GPU's reported name onto the box class its numbers belong to:
/// the first whole alphanumeric token, lowercased, that `SKU_TOKENS` lists.
///
/// The name is free text with SKU detail: `"NVIDIA H100 80GB HBM3"`,
/// `"NVIDIA H100 PCIe"`, `"NVIDIA H200 NVL"`. The fallback normalisation alone
/// would key those as `h10080gbhbm3`, `h100pcie` and `h200nvl`, three keys no
/// baseline uses.
///
/// `None` means "no opinion", not "unknown box": the caller keeps its own
/// normalisation. GB300 and the A100 capacities are deliberately absent, so they
/// keep separate fallback keys rather than sharing a family id.
///
/// One SKU family, one id: `H100 PCIe` and `H100 SXM` share `h100`. The exact name
/// stays verbatim in `Hardware::gpu`; only the baseline key is coarse.
pub fn hardware_id_from_gpu_name(name: &str) -> Option<&'static str> {
    name.split(|c: char| !c.is_ascii_alphanumeric())
        .map(str::to_ascii_lowercase)
        .find_map(|token| {
            SKU_TOKENS
                .iter()
                .find(|(sku, _)| *sku == token)
                .map(|(_, id)| *id)
        })
}

/// 2026-09-26: True when `id` is in [`KNOWN_HARDWARE_IDS`].
///
/// Case-sensitive on purpose: the id is a directory name and a baseline key, so
/// `H100` is not `h100`.
pub fn is_known_hardware_id(id: &str) -> bool {
    KNOWN_HARDWARE_IDS.contains(&id)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-09-26: Registered ids are known whether or not a record exists, so the
    /// resolver can tell "go measure it" from a misspelling.
    #[test]
    fn the_hopper_slots_are_registered_before_they_are_measured() {
        assert!(is_known_hardware_id("h100"));
        assert!(is_known_hardware_id("h200"));
        assert!(is_known_hardware_id("gb10"));
    }

    /// 2026-09-26: The negative side: `h800` is not registered, and `H100` is the
    /// right part in the wrong case.
    #[test]
    fn an_unregistered_or_miscased_id_is_not_known() {
        assert!(!is_known_hardware_id("h800"));
        assert!(!is_known_hardware_id("H100"));
        assert!(!is_known_hardware_id(""));
        // 2026-09-26: GB300 has no registered id.
        assert!(!is_known_hardware_id("gb300"));
    }

    /// 2026-09-26: Both kinds of id are registered: architecture classes
    /// (`kernels/<hw>/` directories) and bench SKUs. `b200` and `b300` are both.
    #[test]
    fn the_kernel_arch_classes_and_the_bench_skus_are_both_registered() {
        for arch_class in ["gb10", "hopper", "b200", "b300"] {
            assert!(is_known_hardware_id(arch_class), "{arch_class}");
        }
        for sku in ["h100", "h200", "b200", "gb200", "b300"] {
            assert!(is_known_hardware_id(sku), "{sku}");
        }
    }

    /// 2026-09-26: Spellings of each SKU land on its id, whatever the case or the
    /// trailing detail.
    #[test]
    fn each_sku_spelling_lands_on_its_box_class() {
        for name in ["NVIDIA GB10", "GB10", "nvidia gb10"] {
            assert_eq!(hardware_id_from_gpu_name(name), Some("gb10"), "{name}");
        }
        for name in [
            "NVIDIA H100 80GB HBM3",
            "NVIDIA H100 PCIe",
            "NVIDIA H100-SXM5-80GB",
        ] {
            assert_eq!(hardware_id_from_gpu_name(name), Some("h100"), "{name}");
        }
        for name in ["NVIDIA H200", "NVIDIA H200 NVL"] {
            assert_eq!(hardware_id_from_gpu_name(name), Some("h200"), "{name}");
        }
        assert_eq!(
            hardware_id_from_gpu_name("AMD Instinct MI300X"),
            Some("mi300x")
        );
    }

    /// 2026-09-26: GH200 is its own id, and survives the substring trap: `gh200`
    /// contains `h200`.
    #[test]
    fn a_grace_hopper_superchip_is_neither_h100_nor_h200() {
        for name in ["NVIDIA GH200 480GB", "NVIDIA GH200 120GB"] {
            assert_eq!(hardware_id_from_gpu_name(name), Some("gh200"), "{name}");
        }
    }

    /// 2026-09-26: A part the table has no entry for answers `None`, never the
    /// nearest-looking id. The A100 capacities differ only in memory; `None` hands
    /// them back to the normalisation that keeps the capacity in the key.
    #[test]
    fn an_unlisted_part_answers_none_rather_than_the_nearest_id() {
        for name in [
            // 2026-09-26: GB300's tokens are `nvidia` and `gb300`, neither listed.
            "NVIDIA GB300",
            "NVIDIA A100-SXM4-40GB",
            "NVIDIA A100-SXM4-80GB",
            "NVIDIA L40S",
            "AMD Radeon 8060S (gfx1151)",
            "",
        ] {
            let got = hardware_id_from_gpu_name(name);
            assert!(got.is_none(), "{name:?} must not map anywhere, got {got:?}");
        }
    }

    /// 2026-09-26: B200 and GB200 spellings land on separate ids, and `gb200`
    /// survives the substring trap: it contains `b200`.
    #[test]
    fn each_blackwell_datacentre_spelling_lands_on_its_box_class() {
        for name in [
            "NVIDIA B200",
            "NVIDIA B200 180GB HBM3e",
            "NVIDIA B200-SXM-180GB",
        ] {
            assert_eq!(hardware_id_from_gpu_name(name), Some("b200"), "{name}");
        }
        for name in ["NVIDIA GB200", "NVIDIA GB200 NVL72"] {
            assert_eq!(hardware_id_from_gpu_name(name), Some("gb200"), "{name}");
        }
    }

    /// 2026-09-26: A Blackwell datacentre name never lands on a Hopper id, where it
    /// would be scored against Hopper thresholds.
    #[test]
    fn a_blackwell_datacentre_part_is_never_filed_under_hopper() {
        for name in [
            "NVIDIA B200",
            "NVIDIA B200 180GB HBM3e",
            "NVIDIA GB200",
            "NVIDIA GB200 NVL72",
        ] {
            let got = hardware_id_from_gpu_name(name);
            assert_ne!(got, Some("h100"), "{name}");
            assert_ne!(got, Some("h200"), "{name}");
            assert_ne!(got, Some("gh200"), "{name}");
        }
    }

    #[test]
    fn b300_has_its_own_key_and_gb300_is_not_misfiled() {
        assert!(is_known_hardware_id("b300"));
        for name in ["NVIDIA B300", "NVIDIA B300 SXM6 AC", "nvidia b300"] {
            assert_eq!(hardware_id_from_gpu_name(name), Some("b300"));
        }
        // 2026-09-26: GB300 maps to no id, so it keeps its fallback key.
        for name in ["NVIDIA GB300", "NVIDIA GB300 NVL72"] {
            assert_eq!(hardware_id_from_gpu_name(name), None);
        }
        assert!(!is_known_hardware_id("gb300"));
    }

    /// 2026-09-26: Every id the SKU table produces is registered; otherwise a box
    /// could write records that `--hardware` refuses to ask for.
    #[test]
    fn every_id_the_sku_table_produces_is_registered() {
        for (sku, id) in SKU_TOKENS {
            assert!(
                is_known_hardware_id(id),
                "{sku} maps to {id:?}, which is not in KNOWN_HARDWARE_IDS"
            );
        }
    }

    /// 2026-09-26: Every `kernels/<hw>/` directory with a `HARDWARE.toml` is
    /// registered, so its `--hardware` is never refused as unknown.
    #[test]
    fn every_kernel_hardware_dir_is_registered() {
        let kernels = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../kernels");
        let mut found = Vec::new();
        for entry in std::fs::read_dir(&kernels)
            .expect("kernels/ is in the tree")
            .flatten()
        {
            if !entry.path().join("HARDWARE.toml").exists() {
                continue;
            }
            let name = entry.file_name().to_string_lossy().to_string();
            assert!(
                is_known_hardware_id(&name),
                "kernels/{name}/ declares a HARDWARE.toml but {name:?} is not in \
                 KNOWN_HARDWARE_IDS; add it there or --hardware {name} reads as a typo"
            );
            found.push(name);
        }
        assert!(
            !found.is_empty(),
            "read no hardware dirs at all — the walk is broken, not the registry"
        );
    }
}
