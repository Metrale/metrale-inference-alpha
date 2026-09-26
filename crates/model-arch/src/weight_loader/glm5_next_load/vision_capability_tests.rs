// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The vision-tower bind decision of `Glm5NextWeightLoader` and of the trait default.
//!
//! Owner: model-arch weight loader (GLM-5.3).
//! Invariants: none beyond the types.

use super::Glm5NextWeightLoader;
use crate::weight_loader::ModelWeightLoader;

/// 2026-09-25: The loader binds the tower exactly when
/// `metrale_config::glm_vision_enabled()` says so, and an unset
/// `METRALE_GLM_VISION` means off.
///
/// Both states are tested through `glm_vision_enabled_from`, because setting
/// the variable here would race the other tests in the binary.
#[test]
fn the_vision_tower_is_off_unless_the_operator_asks() {
    assert_eq!(
        Glm5NextWeightLoader.binds_vision_encoder(),
        metrale_config::glm_vision_enabled(),
        "the loader's withhold decision must be the SAME gate the config \
             parser reads, or the two disagree and the loader asks for tensors \
             that were never uploaded"
    );
    assert!(
        !metrale_config::glm_vision_enabled_from(None),
        "unset must mean off: binding the tower costs 1.05 GiB/rank"
    );
    assert!(metrale_config::glm_vision_enabled_from(Some("1")));
}

#[test]
fn a_multimodal_loader_still_declares_true_by_default() {
    // 2026-09-25: The trait default is `true`, and `Qwen35WeightLoader` does
    // not override it.
    assert!(crate::weight_loader::qwen35::Qwen35WeightLoader.binds_vision_encoder());
}
