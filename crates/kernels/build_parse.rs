// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: KERNEL.toml and MODEL.toml parse helpers for build.rs.
//!
//! Owner: metrale-kernels build.
//! Invariants: none beyond the types.
//!
//! Included via `#[path = "build_parse.rs"] mod build_parse;`, so types
//! defined in build.rs (`SamplingCat`, `ModelTypeMatch`, `DflashRaw`) are
//! reachable via `super::`.

use std::collections::HashMap;

use super::{DflashRaw, ModelTypeMatch, SamplingCat};

#[path = "build_parse_behavior.rs"]
mod behavior;
pub(super) use behavior::*;

pub(super) fn parse_kernel_toml(
    kernel_dir: &std::path::Path,
    vendor: &str,
) -> (Vec<String>, HashMap<String, String>) {
    let kernel_toml_path = kernel_dir.join("KERNEL.toml");
    let kernel_toml: toml::Value = toml::from_str(
        &std::fs::read_to_string(&kernel_toml_path)
            .unwrap_or_else(|e| panic!("{}: {e}", kernel_toml_path.display())),
    )
    .unwrap_or_else(|e| panic!("Bad TOML in {}: {e}", kernel_toml_path.display()));
    println!("cargo:rerun-if-changed={}", kernel_toml_path.display());

    // 2026-09-25: Per-vendor extra flag keys — SSOT in `build_flags::flag_key`, shared
    // with HARDWARE.toml's layer so the two files can never disagree about
    // which key a vendor reads.
    let flag_key = super::build_flags::flag_key(vendor);
    let extra_flags: Vec<String> = kernel_toml
        .get("build")
        .and_then(|b| b.get(flag_key))
        .and_then(|f| f.as_array())
        .map(|arr| {
            arr.iter()
                .map(|v| v.as_str().unwrap().to_string())
                .collect()
        })
        .unwrap_or_default();

    let module_overrides: HashMap<String, String> = kernel_toml
        .get("modules")
        .and_then(|m| m.as_table())
        .map(|t| {
            t.iter()
                .map(|(k, v)| (k.clone(), v.as_str().unwrap().to_string()))
                .collect()
        })
        .unwrap_or_default();

    (extra_flags, module_overrides)
}

/// 2026-09-25: Parse `[shadow_exempt]` from a KERNEL.toml: `module = ["kernel", ...]`.
///
/// Declares `(module, kernel)` pairs a model shadow may omit without that
/// being drift. Each needs a stated reason in the TOML comment.
///
/// The caller uses this only to filter the build warning; the pairs stay in
/// `TargetPtxSet::shadowed_dropped`. The boot audit fails on an unresolved
/// lookup unless `[expected_absent]` declares it, so an exemption cannot turn
/// a missing kernel into a silent pass.
///
/// A missing KERNEL.toml yields an empty list; malformed TOML or a non-string
/// entry panics.
pub(super) fn parse_shadow_exempt(kernel_dir: &std::path::Path) -> Vec<(String, String)> {
    let path = kernel_dir.join("KERNEL.toml");
    let Ok(text) = std::fs::read_to_string(&path) else {
        return Vec::new();
    };
    let toml: toml::Value =
        toml::from_str(&text).unwrap_or_else(|e| panic!("Bad TOML in {}: {e}", path.display()));
    let Some(table) = toml.get("shadow_exempt").and_then(|v| v.as_table()) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for (module, kernels) in table {
        let list = kernels.as_array().unwrap_or_else(|| {
            panic!(
                "{}: [shadow_exempt] {module} must be an array of kernel names",
                path.display()
            )
        });
        for k in list {
            let name = k.as_str().unwrap_or_else(|| {
                panic!(
                    "{}: [shadow_exempt] {module} entries must be strings",
                    path.display()
                )
            });
            out.push((module.clone(), name.to_string()));
        }
    }
    out.sort();
    out
}

/// 2026-09-25: Parse `[expected_absent]` from a MODEL.toml.
///
/// ```toml
/// [expected_absent.mla_absorbed]
/// mla_batched_gemv = "no MLA: this checkpoint is standard GQA (kv_lora_rank 0)"
/// ```
///
/// Each entry names a `(module, kernel)` lookup this model's dispatch may issue
/// and fail to resolve without that being an error. The value is a mandatory,
/// non-empty reason. The preferred fix is to gate the lookup on config so it is
/// never issued (as `qwen3_attention::init_arch_gates` does). Anything neither
/// gated nor listed here fails the boot audit unless
/// `--dangerously-allow-unresolved-kernel-lookups` is set.
pub(super) fn parse_expected_absent(model_dir: &std::path::Path) -> Vec<(String, String)> {
    let path = model_dir.join("MODEL.toml");
    let Ok(text) = std::fs::read_to_string(&path) else {
        return Vec::new();
    };
    let toml: toml::Value =
        toml::from_str(&text).unwrap_or_else(|e| panic!("Bad TOML in {}: {e}", path.display()));
    let Some(table) = toml.get("expected_absent").and_then(|v| v.as_table()) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for (module, kernels) in table {
        let entries = kernels.as_table().unwrap_or_else(|| {
            panic!(
                "{}: [expected_absent.{module}] must be a table of `kernel = \"reason\"` \
                 — every entry needs a stated reason",
                path.display()
            )
        });
        for (kernel, reason) in entries {
            let reason = reason.as_str().unwrap_or_else(|| {
                panic!(
                    "{}: [expected_absent.{module}] {kernel} must be a reason string",
                    path.display()
                )
            });
            assert!(
                !reason.trim().is_empty(),
                "{}: [expected_absent.{module}] {kernel} needs a stated reason",
                path.display()
            );
            out.push((module.clone(), kernel.clone()));
        }
    }
    out.sort();
    out
}

/// 2026-09-25: Parse the four sampling presets (`thinking_text`, `thinking_coding`,
/// `non_thinking`, `tools`) from MODEL.toml `[sampling.*]`. A missing file or
/// section gives `SamplingCat::default()`, and a missing key the same default
/// value as that impl.
pub(super) fn parse_sampling_presets(
    model_dir: &std::path::Path,
) -> (SamplingCat, SamplingCat, SamplingCat, SamplingCat) {
    let model_toml_path = model_dir.join("MODEL.toml");
    if !model_toml_path.exists() {
        return (
            SamplingCat::default(),
            SamplingCat::default(),
            SamplingCat::default(),
            SamplingCat::default(),
        );
    }
    println!("cargo:rerun-if-changed={}", model_toml_path.display());
    let content = std::fs::read_to_string(&model_toml_path)
        .unwrap_or_else(|e| panic!("{}: {e}", model_toml_path.display()));
    let toml: toml::Value = toml::from_str(&content)
        .unwrap_or_else(|e| panic!("Bad TOML in {}: {e}", model_toml_path.display()));

    let parse_cat = |key: &str| -> SamplingCat {
        let section = toml.get("sampling").and_then(|s| s.get(key));
        match section {
            Some(v) => SamplingCat {
                temperature: v
                    .get("temperature")
                    .and_then(|t| t.as_float())
                    .unwrap_or(0.7) as f32,
                top_p: v.get("top_p").and_then(|t| t.as_float()).unwrap_or(0.95) as f32,
                top_k: v.get("top_k").and_then(|t| t.as_integer()).unwrap_or(20) as u32,
                presence_penalty: v
                    .get("presence_penalty")
                    .and_then(|t| t.as_float())
                    .unwrap_or(0.0) as f32,
                frequency_penalty: v
                    .get("frequency_penalty")
                    .and_then(|t| t.as_float())
                    .unwrap_or(0.0) as f32,
                repetition_penalty: v
                    .get("repetition_penalty")
                    .and_then(|t| t.as_float())
                    .unwrap_or(1.0) as f32,
                dry_multiplier: v
                    .get("dry_multiplier")
                    .and_then(|t| t.as_float())
                    .unwrap_or(0.0) as f32,
                dry_base: v.get("dry_base").and_then(|t| t.as_float()).unwrap_or(1.75) as f32,
                dry_allowed_length: v
                    .get("dry_allowed_length")
                    .and_then(|t| t.as_integer())
                    .unwrap_or(2) as u32,
                lz_penalty: v
                    .get("lz_penalty")
                    .and_then(|t| t.as_float())
                    .unwrap_or(0.0) as f32,
                // 2026-09-25: No unwrap_or: an absent min_p must stay None so the
                // server's --default-min-p keeps owning the field.
                min_p: v.get("min_p").and_then(|t| t.as_float()).map(|p| p as f32),
                // 2026-09-25: Same rule as min_p: absent stays None so --default-top-n-sigma
                // keeps owning the field for every model that does not declare it.
                top_n_sigma: v
                    .get("top_n_sigma")
                    .and_then(|t| t.as_float())
                    .map(|p| p as f32),
            },
            None => SamplingCat::default(),
        }
    };

    (
        parse_cat("thinking_text"),
        parse_cat("thinking_coding"),
        parse_cat("non_thinking"),
        parse_cat("tools"),
    )
}

/// 2026-09-25: Parse `[[model_types]]` from MODEL.toml.
///
/// Each entry maps a `(model_type, optional hidden_size)` pair to this kernel
/// target; a missing `hidden_size` is a wildcard. An entry without a string
/// `model_type`, a missing file or unparsable TOML is skipped.
pub(super) fn parse_model_types(model_dir: &std::path::Path) -> Vec<ModelTypeMatch> {
    let model_toml_path = model_dir.join("MODEL.toml");
    if !model_toml_path.exists() {
        return Vec::new();
    }
    let content = std::fs::read_to_string(&model_toml_path).unwrap_or_default();
    let toml: toml::Value = match toml::from_str(&content) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };
    let Some(entries) = toml.get("model_types").and_then(|v| v.as_array()) else {
        return Vec::new();
    };
    entries
        .iter()
        .filter_map(|entry| {
            let mt = entry.get("model_type")?.as_str()?.to_string();
            let hs = entry
                .get("hidden_size")
                .and_then(|v| v.as_integer())
                .map(|v| v as usize);
            Some(ModelTypeMatch {
                model_type: mt,
                hidden_size: hs,
            })
        })
        .collect()
}

/// 2026-09-25: Parse `[model] match_names` from MODEL.toml.
///
/// Checkpoint-reference needles (case-insensitive substrings of the checkpoint
/// reference) that identify checkpoints this target serves; `src/resolve.rs`
/// uses them to break a tie between targets. Empty entries are rejected: an
/// empty needle would match every reference.
pub(super) fn parse_match_names(model_dir: &std::path::Path) -> Vec<String> {
    let path = model_dir.join("MODEL.toml");
    let Ok(text) = std::fs::read_to_string(&path) else {
        return Vec::new();
    };
    let toml: toml::Value =
        toml::from_str(&text).unwrap_or_else(|e| panic!("Bad TOML in {}: {e}", path.display()));
    let Some(arr) = toml
        .get("model")
        .and_then(|m| m.get("match_names"))
        .and_then(|v| v.as_array())
    else {
        return Vec::new();
    };
    arr.iter()
        .map(|v| {
            let s = v.as_str().unwrap_or_else(|| {
                panic!(
                    "{}: [model] match_names entries must be strings",
                    path.display()
                )
            });
            assert!(
                !s.trim().is_empty(),
                "{}: [model] match_names entries must be non-empty — an empty needle \
                 would match every checkpoint reference",
                path.display()
            );
            // 2026-09-25: The needles are emitted into generated Rust as
            // `"{needle}"` string literals (build_codegen.rs) with no escaping;
            // a quote or backslash would produce an uncompilable target_ptx.rs
            // with an error pointing nowhere near this file. Reject here, where
            // the operator can see which TOML entry to fix.
            assert!(
                !s.contains('"') && !s.contains('\\'),
                "{}: [model] match_names entry {s:?} contains a quote or backslash — \
                 needles are emitted verbatim into generated Rust string literals \
                 and checkpoint references never contain these characters",
                path.display()
            );
            s.to_string()
        })
        .collect()
}

/// 2026-09-25: Parse `[dflash]` from MODEL.toml into `TargetPtxSet::dflash`.
/// Returns `None` when the file, the section or its `draft_model` string is
/// missing, or the TOML does not parse. Missing numeric keys default to
/// `gamma` 16, `window_size` 0 and `mask_token_id` 0.
pub(super) fn parse_dflash(model_dir: &std::path::Path) -> Option<DflashRaw> {
    let model_toml_path = model_dir.join("MODEL.toml");
    if !model_toml_path.exists() {
        return None;
    }
    let content = std::fs::read_to_string(&model_toml_path).unwrap_or_default();
    let toml: toml::Value = toml::from_str(&content).ok()?;
    let dflash = toml.get("dflash")?;
    let draft_model = dflash.get("draft_model")?.as_str()?.to_string();
    let gamma = dflash
        .get("gamma")
        .and_then(|v| v.as_integer())
        .map(|v| v as usize)
        .unwrap_or(16);
    let window_size = dflash
        .get("window_size")
        .and_then(|v| v.as_integer())
        .map(|v| v as usize)
        .unwrap_or(0);
    let mask_token_id = dflash
        .get("mask_token_id")
        .and_then(|v| v.as_integer())
        .map(|v| v as u32)
        .unwrap_or(0);
    let target_layer_ids: Vec<usize> = dflash
        .get("target_layer_ids")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_integer().map(|x| x as usize))
                .collect()
        })
        .unwrap_or_default();
    Some(DflashRaw {
        draft_model,
        gamma,
        window_size,
        mask_token_id,
        target_layer_ids,
    })
}
