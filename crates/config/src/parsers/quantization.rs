// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Parser for the `quantization_config` block (compressed-tensors, or a ModelOpt
//! `hf_quant_config.json` placed in that slot) into `QuantizationConfig`.
//!
//! Owner: config (quantization).
//! Invariants: none beyond the types.

#![allow(unused_imports)]

use anyhow::{Context, Result};
use serde_json::Value;

use super::super::{ModelConfig, QuantizationConfig};

pub fn parse_quantization_config(raw: &serde_json::Value) -> Option<QuantizationConfig> {
    let qc_raw = raw.get("quantization_config")?;
    // 2026-09-26: A ModelOpt `hf_quant_config.json`, which `merge_sidecar_quant_config` places
    // in this slot, nests the fields under `"quantization"` and has no `quant_method`; the
    // scheme is named by `producer.name == "modelopt"` (the Nemotron-3 Nano and Super
    // checkpoints in the local HF cache):
    //
    //   { "producer": { "name": "modelopt", ... },
    //     "quantization": { "quant_algo": "NVFP4",
    //                       "exclude_modules": ["lm_head", ...] } }
    //
    // Read as is, it would give `None`. `normalize_modelopt_sidecar` turns it into the flat
    // shape; a flat block passes through unchanged.
    let canonical = normalize_modelopt_sidecar(qc_raw);
    let qc = &canonical;

    let quant_method = qc
        .get("quant_method")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("")
        .to_string();
    let quant_algo = qc
        .get("quant_algo")
        .and_then(serde_json::Value::as_str)
        .or_else(|| {
            // 2026-09-26: Without `quant_algo`, the label comes from
            // `config_groups.group_0.weights`: a 4-bit float is NVFP4, an 8-bit float FP8.
            let group = qc.get("config_groups")?.get("group_0")?;
            let weights = group.get("weights")?;
            let bits = weights.get("num_bits")?.as_u64()?;
            let ty = weights.get("type")?.as_str()?;
            match (bits, ty) {
                (4, "float") => Some("NVFP4"),
                (8, "float") => Some("FP8"),
                _ => None,
            }
        })
        .unwrap_or("")
        .to_string();
    let format = qc
        .get("format")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("")
        .to_string();

    // 2026-09-26: The ignore list is `ignore` followed by the `exclude_modules` entries not
    // already in it.
    let mut ignore_modules: Vec<String> = Vec::new();
    if let Some(arr) = qc.get("ignore").and_then(serde_json::Value::as_array) {
        for v in arr {
            if let Some(s) = v.as_str() {
                ignore_modules.push(s.to_string());
            }
        }
    }
    if let Some(arr) = qc
        .get("exclude_modules")
        .and_then(serde_json::Value::as_array)
    {
        for v in arr {
            if let Some(s) = v.as_str()
                && !ignore_modules.contains(&s.to_string())
            {
                ignore_modules.push(s.to_string());
            }
        }
    }

    // 2026-09-26: No method, algorithm or ignore list means no quantization config.
    if quant_method.is_empty() && quant_algo.is_empty() && ignore_modules.is_empty() {
        return None;
    }

    Some(QuantizationConfig {
        quant_method,
        quant_algo,
        format,
        ignore_modules,
    })
}

/// 2026-09-26: Flatten a ModelOpt `hf_quant_config.json` payload into the shape
/// [`parse_quantization_config`] reads. Both steps leave a flat block unchanged:
///   1. A nested `"quantization"` object replaces the payload.
///   2. Without a non-empty `quant_method`, `producer.name == "modelopt"` (any case) adds
///      `quant_method = "modelopt"`, the value `detect_quant_format` and
///      `detect_nvfp4_variant` match on.
fn normalize_modelopt_sidecar(qc_raw: &serde_json::Value) -> serde_json::Value {
    let mut canonical = match qc_raw.get("quantization") {
        Some(serde_json::Value::Object(inner)) => serde_json::Value::Object(inner.clone()),
        _ => qc_raw.clone(),
    };

    let Some(obj) = canonical.as_object_mut() else {
        return canonical;
    };

    let has_method = obj
        .get("quant_method")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|s| !s.is_empty());
    if !has_method {
        let producer_modelopt = qc_raw
            .get("producer")
            .and_then(|p| p.get("name"))
            .and_then(serde_json::Value::as_str)
            .is_some_and(|n| n.eq_ignore_ascii_case("modelopt"));
        if producer_modelopt {
            obj.insert(
                "quant_method".to_string(),
                serde_json::Value::String("modelopt".to_string()),
            );
        }
    }

    canonical
}

#[cfg(test)]
mod tests {
    use super::{normalize_modelopt_sidecar, parse_quantization_config};

    /// 2026-09-26: The `hf_quant_config.json` shape of the local
    /// `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` checkpoint (exclusion list cut to three),
    /// placed in the `quantization_config` slot as `merge_sidecar_quant_config` does.
    #[test]
    fn modelopt_sidecar_nested_schema_preserves_exclusions() {
        let raw = serde_json::json!({
            "quantization_config": {
                "producer": { "name": "modelopt", "version": "0.29.0" },
                "quantization": {
                    "quant_algo": "NVFP4",
                    "kv_cache_quant_algo": "FP8",
                    "group_size": 16,
                    "exclude_modules": [
                        "lm_head",
                        "backbone.layers.4.mixer.in_proj",
                        "backbone.layers.0.mixer.conv1d"
                    ]
                }
            }
        });
        let qc = parse_quantization_config(&raw)
            .expect("ModelOpt nested sidecar must yield a QuantizationConfig");
        assert_eq!(qc.quant_method, "modelopt");
        assert_eq!(qc.quant_algo, "NVFP4");
        assert_eq!(qc.ignore_modules.len(), 3);
        assert!(qc.ignore_modules.iter().any(|m| m == "lm_head"));
        assert!(
            qc.ignore_modules
                .iter()
                .any(|m| m == "backbone.layers.4.mixer.in_proj")
        );
        assert!(
            qc.ignore_modules
                .iter()
                .any(|m| m == "backbone.layers.0.mixer.conv1d")
        );
    }

    /// 2026-09-26: A flat compressed-tensors block is unchanged by the ModelOpt normalization.
    #[test]
    fn flat_compressed_tensors_block_is_not_normalized() {
        let raw = serde_json::json!({
            "quantization_config": {
                "quant_method": "compressed-tensors",
                "format": "nvfp4-pack-quantized",
                "ignore": ["lm_head"]
            }
        });
        let flat = &raw["quantization_config"];
        assert_eq!(normalize_modelopt_sidecar(flat), flat.clone());

        let qc = parse_quantization_config(&raw).expect("flat block must still parse");
        assert_eq!(qc.quant_method, "compressed-tensors");
        assert_eq!(qc.format, "nvfp4-pack-quantized");
        assert_eq!(qc.ignore_modules, vec!["lm_head".to_string()]);
    }

    /// 2026-09-26: The ModelOpt mixed-precision shape of the local Nemotron-3 Super 120B
    /// checkpoint (nested, no `exclude_modules`) resolves `quant_method = "modelopt"`.
    #[test]
    fn modelopt_mixed_precision_sidecar_parses_without_exclusions() {
        let raw = serde_json::json!({
            "quantization_config": {
                "producer": { "name": "modelopt", "version": "0.43.0" },
                "quantization": {
                    "quant_algo": "MIXED_PRECISION",
                    "kv_cache_quant_algo": "FP8"
                }
            }
        });
        let qc = parse_quantization_config(&raw)
            .expect("mixed-precision sidecar must yield a QuantizationConfig");
        assert_eq!(qc.quant_method, "modelopt");
        assert_eq!(qc.quant_algo, "MIXED_PRECISION");
        assert!(qc.ignore_modules.is_empty());
    }
}
