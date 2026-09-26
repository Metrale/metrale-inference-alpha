// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The FP8 E4M3 block-scaled layout (`.weight` with
//! `.weight_scale_inv`), mapped to [`Nvfp4Variant::Fp8Dequanted`].
//!
//! Owner: model-layers (weight loading).
//! Invariants: none beyond the types.

use crate::quant_format::{QuantFormat, module_matches_pattern};
use crate::weight_map::Nvfp4Variant;

/// 2026-09-25: An FP8 block-scaled checkpoint.
#[derive(Debug)]
pub struct Fp8BlockScaledFormat {
    pub ignore_modules: Vec<String>,
}

impl Fp8BlockScaledFormat {
    pub fn new(ignore_modules: Vec<String>) -> Self {
        Self { ignore_modules }
    }
}

impl QuantFormat for Fp8BlockScaledFormat {
    fn name(&self) -> &'static str {
        "fp8-blockscaled"
    }

    fn base_variant(&self) -> Nvfp4Variant {
        Nvfp4Variant::Fp8Dequanted
    }

    fn is_ignored(&self, module_path: &str) -> bool {
        self.ignore_modules
            .iter()
            .any(|pat| module_matches_pattern(module_path, pat))
    }
}
