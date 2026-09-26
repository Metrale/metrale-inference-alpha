// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The compressed-tensors NVFP4 layout, mapped to
//! [`Nvfp4Variant::CompressedTensors`]. Its packed weights are
//! `.weight_packed` with a per-tensor `.weight_global_scale`, which the loader
//! inverts into `weight_scale_2` (`weight_map::quantized_v2`).
//!
//! Owner: model-layers (weight loading).
//! Invariants: none beyond the types.

use crate::quant_format::{QuantFormat, module_matches_pattern};
use crate::weight_map::Nvfp4Variant;

/// 2026-09-25: A compressed-tensors checkpoint.
#[derive(Debug)]
pub struct CompressedTensorsFormat {
    /// 2026-09-25: The config's `format` string; nothing in this module reads
    /// it.
    pub format: String,
    /// 2026-09-25: Module globs that ship unquantized.
    pub ignore_modules: Vec<String>,
}

impl CompressedTensorsFormat {
    pub fn new(format: String, ignore_modules: Vec<String>) -> Self {
        Self {
            format,
            ignore_modules,
        }
    }
}

impl QuantFormat for CompressedTensorsFormat {
    fn name(&self) -> &'static str {
        "compressed-tensors"
    }

    fn base_variant(&self) -> Nvfp4Variant {
        Nvfp4Variant::CompressedTensors
    }

    fn is_ignored(&self, module_path: &str) -> bool {
        self.ignore_modules
            .iter()
            .any(|pat| module_matches_pattern(module_path, pat))
    }
}
