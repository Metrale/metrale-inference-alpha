// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests `build_flags.rs`: the extra compiler flags a target is
//! built with, from three layers (HARDWARE.toml `[build]`, the common-role
//! KERNEL.tomls, the leaf-role KERNEL.tomls), their merge order and the
//! per-vendor key.
//!
//! Owner: metrale-kernels tests.
//! Invariants: none beyond the types.
//!
//! The HARDWARE.toml layer holds flags about the architecture, so hopper, b200
//! and b300 define `METRALE_NO_WARP_BLOCKSCALE_MMA` once for every model;
//! hopper and b200 read gb10's KERNEL.tomls. This is an integration test
//! because cargo does not run a build script's own unit tests.

#[path = "../build_flags.rs"]
mod build_flags;

use build_flags::{flag_key, hardware_extra_flags, merge_extra_flags};

use std::path::{Path, PathBuf};

fn s(v: &[&str]) -> Vec<String> {
    v.iter().map(|x| x.to_string()).collect()
}

/// 2026-09-25: Hardware flags come first, then common's, then the model's,
/// the order `resolve_targets` in build.rs passes them.
#[test]
fn the_layers_merge_least_specific_first() {
    assert_eq!(
        merge_extra_flags(
            &s(&["-DMETRALE_NO_WARP_BLOCKSCALE_MMA"]),
            &s(&["--fmad=false", "-DTQ_PLUS_SIGNS"]),
            &s(&["--fmad=false"]),
        ),
        s(&[
            "-DMETRALE_NO_WARP_BLOCKSCALE_MMA",
            "--fmad=false",
            "-DTQ_PLUS_SIGNS"
        ]),
    );
}

/// 2026-09-25: A repeated flag appears once, at the position its first
/// declaring layer gave it.
#[test]
fn a_flag_declared_twice_is_emitted_once() {
    let merged = merge_extra_flags(
        &s(&["--fmad=false"]),
        &s(&["--fmad=false", "-DTQ_PLUS_SIGNS"]),
        &s(&["--fmad=false"]),
    );
    assert_eq!(merged, s(&["--fmad=false", "-DTQ_PLUS_SIGNS"]));
    assert_eq!(
        merged.iter().filter(|f| *f == "--fmad=false").count(),
        1,
        "a flag declared by every layer must still be passed once"
    );
}

/// 2026-09-25: With no hardware layer, the result is common's flags then the
/// model's, deduped. Every HARDWARE.toml but hopper's, b200's and b300's has
/// no `[build]` table.
#[test]
fn no_hardware_layer_leaves_the_old_two_layer_result() {
    assert_eq!(
        merge_extra_flags(
            &[],
            &s(&["--fmad=false", "-DTQ_PLUS_SIGNS"]),
            &s(&["--fmad=false"]),
        ),
        s(&["--fmad=false", "-DTQ_PLUS_SIGNS"]),
    );
}

/// 2026-09-25: `extra_metal_flags` for apple and metal, `extra_nvcc_flags`
/// otherwise. `build_parse::parse_kernel_toml` reads KERNEL.toml flags with the
/// same `flag_key`.
#[test]
fn the_flag_key_follows_the_vendor() {
    assert_eq!(flag_key("nvidia"), "extra_nvcc_flags");
    assert_eq!(flag_key("amd"), "extra_nvcc_flags");
    assert_eq!(flag_key("apple"), "extra_metal_flags");
    assert_eq!(flag_key("metal"), "extra_metal_flags");
}

/// 2026-09-25: A HARDWARE.toml key for another vendor contributes nothing.
#[test]
fn the_wrong_vendors_key_is_not_read() {
    let toml: toml::Value = toml::from_str("[build]\nextra_nvcc_flags = [\"-DX\"]\n").unwrap();
    assert_eq!(hardware_extra_flags(&toml, "nvidia"), s(&["-DX"]));
    assert!(hardware_extra_flags(&toml, "apple").is_empty());
}

/// 2026-09-25: No `[build]` table gives no flags, not a panic.
#[test]
fn a_hardware_toml_with_no_build_table_declares_no_flags() {
    let toml: toml::Value =
        toml::from_str("[hardware]\nname = \"gb10\"\narch = \"sm_121f\"\n").unwrap();
    assert!(hardware_extra_flags(&toml, "nvidia").is_empty());
}

/// 2026-09-25: A non-string entry panics with a message naming the key.
#[test]
#[should_panic(expected = "extra_nvcc_flags")]
fn a_non_string_flag_is_refused() {
    let toml: toml::Value = toml::from_str("[build]\nextra_nvcc_flags = [1]\n").unwrap();
    let _ = hardware_extra_flags(&toml, "nvidia");
}

fn kernels_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/kernels is two levels below the workspace root")
        .join("kernels")
}

fn hardware_toml(hw: &str) -> toml::Value {
    let path = kernels_root().join(hw).join("HARDWARE.toml");
    toml::from_str(
        &std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display())),
    )
    .unwrap_or_else(|e| panic!("bad TOML in {}: {e}", path.display()))
}

/// 2026-09-25: gb10's HARDWARE.toml declares no hardware-level flags.
#[test]
fn gb10_declares_no_hardware_level_flags() {
    assert!(
        hardware_extra_flags(&hardware_toml("gb10"), "nvidia").is_empty(),
        "kernels/gb10/HARDWARE.toml must add no flags — the guarded W4A4 path \
         is compiled IN on GB10 and its PTX may not move"
    );
}
