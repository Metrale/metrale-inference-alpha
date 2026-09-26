// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The lever parsers and layer selectors `load_layers` reads: the
//! `METRALE_FP8_DEQUANT_LAYERS` layer list, the Holo fast-MoE mode and layer list, and the
//! Holo ModelOpt mixed-precision checkpoint test.
//!
//! Owner: model-arch weight loader (Qwen3.5).
//! Invariants: none beyond the types.

use metrale_config::ModelConfig;

/// 2026-09-25: Whether layer `layer` is selected by `METRALE_FP8_DEQUANT_LAYERS`, a
/// comma-separated list of indices and inclusive ranges (`"31-39"`, `"31,35,39"`). Unset
/// selects every layer; parts that do not parse are skipped.
pub(super) fn layer_dequant_selected(layer: usize) -> bool {
    // 2026-09-25: Read per call, so a later model load sees the variable's current value.
    let spec: Option<Vec<(usize, usize)>> = (|| -> Option<Vec<(usize, usize)>> {
        let s = std::env::var("METRALE_FP8_DEQUANT_LAYERS").ok()?;
        let mut ranges: Vec<(usize, usize)> = Vec::new();
        for part in s.split(',') {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            if let Some((a, b)) = part.split_once('-') {
                if let (Ok(a), Ok(b)) = (a.trim().parse::<usize>(), b.trim().parse::<usize>()) {
                    ranges.push((a.min(b), a.max(b)));
                }
            } else if let Ok(a) = part.parse::<usize>() {
                ranges.push((a, a));
            }
        }
        Some(ranges)
    })();
    match spec {
        None => true,
        Some(ranges) => ranges.iter().any(|&(a, b)| layer >= a && layer <= b),
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) enum HoloFastMoeMode {
    GateUp,
    Full,
    Unified,
}

/// 2026-09-25: `ModelLevers::holo_moe_gateup_fp4` (`METRALE_HOLO_MOE_GATEUP_FP4=1` or `true`,
/// any case), resolved once per process. Read here only for the layer-0 warning.
pub(super) fn holo_moe_gateup_fp4() -> bool {
    metrale_model_layers::layers::ops::ModelLevers::get().holo_moe_gateup_fp4
}

/// 2026-09-25: `ModelLevers::holo_moe_down_fp4` (`METRALE_HOLO_MOE_DOWN_FP4=1` or `true`, any
/// case), resolved once per process. Read here only for the layer-0 warning.
pub(super) fn holo_moe_down_fp4() -> bool {
    metrale_model_layers::layers::ops::ModelLevers::get().holo_moe_down_fp4
}

pub(super) fn holo_fast_moe_mode() -> Option<HoloFastMoeMode> {
    // 2026-09-25: Read per call, like `layer_dequant_selected`.
    (|| -> Option<HoloFastMoeMode> {
        let Ok(mode) = std::env::var("METRALE_HOLO_FAST_MOE_MODE") else {
            return None;
        };
        match mode.trim() {
            "gate_up" | "gate-up" => Some(HoloFastMoeMode::GateUp),
            "full" => Some(HoloFastMoeMode::Full),
            "unified" => {
                let unified_layout = std::env::var("METRALE_UNIFIED_MOE_LAYOUT")
                    .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                    .unwrap_or(false);
                if unified_layout {
                    Some(HoloFastMoeMode::Unified)
                } else {
                    tracing::warn!(
                        target: "metrale_model_arch::weight_loader::qwen35::load_layers",
                        "Ignoring METRALE_HOLO_FAST_MOE_MODE=unified; set METRALE_UNIFIED_MOE_LAYOUT=1 so decode uses transposed experts"
                    );
                    None
                }
            }
            other => {
                tracing::warn!(
                    target: "metrale_model_arch::weight_loader::qwen35::load_layers",
                    "Ignoring METRALE_HOLO_FAST_MOE_MODE={other:?}; expected gate_up, full, or unified"
                );
                None
            }
        }
    })()
}

/// 2026-09-25: Whether `layer` falls in `spec`, the layer list the caller resolved
/// (`holo_fast_moe_spec`); the low-memory layout and this selection read the same value.
pub(super) fn holo_fast_moe_layer_selected(spec: &str, layer: usize) -> bool {
    parse_layer_ranges(spec)
        .iter()
        .any(|&(a, b)| layer >= a && layer <= b)
}

fn parse_layer_ranges(spec: &str) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    for part in spec.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Some((a, b)) = part.split_once('-') {
            if let (Ok(a), Ok(b)) = (a.trim().parse::<usize>(), b.trim().parse::<usize>()) {
                ranges.push((a.min(b), a.max(b)));
            }
        } else if let Ok(a) = part.parse::<usize>() {
            ranges.push((a, a));
        }
    }
    ranges
}

pub(super) fn is_holo_modelopt_mixed_precision(config: &ModelConfig) -> bool {
    config.model_type == "holo3_1_moe"
        && config.quantization_config.as_ref().is_some_and(|qc| {
            qc.quant_method == "modelopt" && qc.quant_algo.eq_ignore_ascii_case("MIXED_PRECISION")
        })
}
