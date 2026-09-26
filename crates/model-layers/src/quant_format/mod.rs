// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The quantized-weight layout of a checkpoint, and its ignore
//! list: [`CompressedTensorsFormat`], [`ModeloptFormat`] or
//! [`Fp8BlockScaledFormat`], chosen by [`detect_quant_format`].
//!
//! Owner: model-layers (weight loading).
//! Invariants: none beyond the types.

use metrale_config::ModelConfig;
use metrale_model_weights::weights::WeightStore;

use crate::weight_map::Nvfp4Variant;

mod compressed_tensors;
mod fp8_blockscaled;
mod modelopt;

pub use compressed_tensors::CompressedTensorsFormat;
pub use fp8_blockscaled::Fp8BlockScaledFormat;
pub use modelopt::ModeloptFormat;

/// 2026-09-25: A quantized-weight layout: its log name, the [`Nvfp4Variant`]
/// it maps to, and the module globs its ignore list keeps unquantized. The
/// weight loaders pick their variant through `weight_map::detect_nvfp4_variant`,
/// not through this trait.
pub trait QuantFormat: Send + Sync + std::fmt::Debug {
    /// 2026-09-25: Name for logs.
    fn name(&self) -> &'static str;

    /// 2026-09-25: The [`Nvfp4Variant`] this layout maps to.
    fn base_variant(&self) -> Nvfp4Variant;

    /// 2026-09-25: Whether `module_path` matches a glob of the ignore list
    /// (`module_matches_pattern`).
    fn is_ignored(&self, module_path: &str) -> bool;

    /// 2026-09-25: `Bf16Raw` for an ignored module, else [`Self::base_variant`].
    fn variant_for(&self, module_path: &str) -> Nvfp4Variant {
        if self.is_ignored(module_path) {
            Nvfp4Variant::Bf16Raw
        } else {
            self.base_variant()
        }
    }
}

/// 2026-09-25: Pick the [`QuantFormat`] for a checkpoint.
///
/// 1. A `quantization_config` whose `quant_method` is `modelopt`,
///    `compressed-tensors` or `fp8` selects that layout directly. The
///    `quant_algo` and `format` are logged or stored, not used to choose.
/// 2. Otherwise the layout comes from `weight_map::detect_nvfp4_variant`
///    (an unrecognised method is warned about first).
/// 3. When that finds no quantized weights (`Bf16Raw`), it warns and returns
///    a [`ModeloptFormat`].
///
/// The ignore list is the config's `ignore_modules`, or empty without a
/// config.
pub fn detect_quant_format(config: &ModelConfig, store: &WeightStore) -> Box<dyn QuantFormat> {
    if let Some(qc) = &config.quantization_config {
        let method = qc.quant_method.as_str();
        let algo = qc.quant_algo.as_str();
        let format = qc.format.as_str();
        let ignore = qc.ignore_modules.clone();

        match method {
            "modelopt" => {
                tracing::info!(
                    "QuantFormat: modelopt (algo={algo:?}), {} ignored module(s)",
                    ignore.len(),
                );
                return Box::new(ModeloptFormat::new(algo.to_string(), ignore));
            }
            "compressed-tensors" => {
                tracing::info!(
                    "QuantFormat: compressed-tensors (format={format:?}), {} ignored module(s)",
                    ignore.len(),
                );
                return Box::new(CompressedTensorsFormat::new(format.to_string(), ignore));
            }
            "fp8" => {
                tracing::info!(
                    "QuantFormat: fp8 (block-scaled), {} ignored module(s)",
                    ignore.len(),
                );
                return Box::new(Fp8BlockScaledFormat::new(ignore));
            }
            other if !other.is_empty() => {
                tracing::warn!(
                    "QuantFormat: config declares unrecognized quant_method={other:?}; \
                     falling back to tensor-name heuristic. Metrale Engine currently understands \
                     {{compressed-tensors, modelopt, fp8}}. Checkpoint load may fail."
                );
            }
            _ => {
                // 2026-09-25: An empty `quant_method` falls through to the
                // tensor-name detection below.
            }
        }
    }

    let variant = crate::weight_map::detect_nvfp4_variant(store, config);
    let ignore = config
        .quantization_config
        .as_ref()
        .map(|qc| qc.ignore_modules.clone())
        .unwrap_or_default();
    match variant {
        Nvfp4Variant::CompressedTensors => {
            tracing::info!("QuantFormat: compressed-tensors (detected from tensor names)");
            Box::new(CompressedTensorsFormat::new(String::new(), ignore))
        }
        Nvfp4Variant::Fp8Dequanted => {
            tracing::info!("QuantFormat: fp8-blockscaled (detected from tensor names)");
            Box::new(Fp8BlockScaledFormat::new(ignore))
        }
        Nvfp4Variant::Standard => {
            tracing::info!("QuantFormat: modelopt-style NVFP4 (detected from tensor names)");
            Box::new(ModeloptFormat::new(String::new(), ignore))
        }
        Nvfp4Variant::Bf16Raw => {
            tracing::warn!(
                "QuantFormat: no quantization declared and no pre-quantized weights found; \
                 treating checkpoint as BF16 raw (weights will be runtime-quantized). \
                 Quality will be inferior to a calibrated NVFP4 release."
            );
            Box::new(ModeloptFormat::new(String::new(), ignore)) as Box<dyn QuantFormat>
        }
    }
}

/// 2026-09-25: Glob match for ignore-list entries, shared by the three
/// [`QuantFormat::is_ignored`] impls. `*` matches any run of characters,
/// including none; every other character matches itself. A pattern without
/// `*` must equal the path, so `lm_head` does not match `lm_head_norm`; a
/// pattern ending in `*` matches by prefix.
pub(crate) fn module_matches_pattern(path: &str, pattern: &str) -> bool {
    let segments: Vec<&str> = pattern.split('*').collect();
    if segments.len() == 1 {
        return path == pattern;
    }
    let mut rest = path;
    // 2026-09-25: The first segment anchors at the start unless the pattern
    // begins with `*`.
    let first = segments[0];
    if !first.is_empty() {
        if !rest.starts_with(first) {
            return false;
        }
        rest = &rest[first.len()..];
    }
    // 2026-09-25: Middle segments must appear in order; each takes its
    // leftmost match.
    for seg in &segments[1..segments.len() - 1] {
        if seg.is_empty() {
            continue;
        }
        match rest.find(seg) {
            Some(pos) => rest = &rest[pos + seg.len()..],
            None => return false,
        }
    }
    // 2026-09-25: An empty last segment means the pattern ended in `*`;
    // otherwise the remainder must end with it.
    let last = segments[segments.len() - 1];
    last.is_empty() || rest.ends_with(last)
}

#[cfg(test)]
mod tests {
    use super::{
        CompressedTensorsFormat, Fp8BlockScaledFormat, ModeloptFormat, QuantFormat,
        module_matches_pattern as m,
    };
    use crate::weight_map::Nvfp4Variant;

    #[test]
    fn exact_match() {
        assert!(m("lm_head", "lm_head"));
        assert!(!m("lm_head_norm", "lm_head"));
    }

    #[test]
    fn prefix_star() {
        assert!(m(
            "model.layers.5.self_attn.q_proj",
            "model.layers.*.self_attn*"
        ));
        assert!(m(
            "model.layers.62.self_attn.out",
            "model.layers.*.self_attn*"
        ));
        assert!(!m(
            "model.layers.5.mlp.gate_proj",
            "model.layers.*.self_attn*"
        ));
    }

    #[test]
    fn trailing_star_matches_a_prefix() {
        assert!(m("lm_head.weight", "lm_head*"));
        assert!(!m("model.lm_head.weight", "lm_head*"));
    }

    #[test]
    fn leading_and_all_star_patterns() {
        assert!(m("model.layers.7.lm_head.weight", "*.lm_head.weight"));
        assert!(!m("model.layers.7.lm_head.bias", "*.lm_head.weight"));
        assert!(m("anything.at.all", "*"));
    }

    #[test]
    fn middle_star() {
        assert!(m("model.layers.0.mlp.gate", "model.layers.*.mlp.gate"));
        assert!(!m("model.layers.0.attn.gate", "model.layers.*.mlp.gate"));
        assert!(m("a.left.middle.right.z", "a.*.middle.*.z"));
        assert!(!m("a.right.middle.left.z", "a.*.left.*.z"));
    }

    #[test]
    fn ignore_patterns_drive_effective_variants_for_every_format() {
        let formats: Vec<(Box<dyn QuantFormat>, Nvfp4Variant)> = vec![
            (
                Box::new(ModeloptFormat::new(
                    "NVFP4".into(),
                    vec!["model.layers.*.self_attn*".into()],
                )),
                Nvfp4Variant::Standard,
            ),
            (
                Box::new(CompressedTensorsFormat::new(
                    "nvfp4-pack-quantized".into(),
                    vec!["model.layers.*.self_attn*".into()],
                )),
                Nvfp4Variant::CompressedTensors,
            ),
            (
                Box::new(Fp8BlockScaledFormat::new(vec![
                    "model.layers.*.self_attn*".into(),
                ])),
                Nvfp4Variant::Fp8Dequanted,
            ),
        ];
        for (format, base) in formats {
            assert_eq!(format.base_variant(), base);
            assert_eq!(
                format.variant_for("model.layers.5.self_attn.q_proj"),
                Nvfp4Variant::Bf16Raw
            );
            assert_eq!(format.variant_for("model.layers.5.mlp.gate_proj"), base);
        }
    }
}
