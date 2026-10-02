// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: What `--weight-quantization declared` changes in `load_layers` for a checkpoint
//! that declares mixed formats per layer (ModelOpt MIXED_PRECISION, e.g.
//! nvidia/Qwen3.6-35B-A3B-NVFP4): which projections stay at their declared FP8 weights, whether
//! the router stays BF16, and whether a layer's experts are the checkpoint's own NVFP4.
//!
//! Owner: model-arch weight loader (Qwen3.5).
//! Invariants: under the `nvfp4` tier (`WeightQuantPolicy::follows_plan` false) every answer is
//! the pre-plan one: nothing stays FP8 here, the router follows the variant, no layer is marked.

use anyhow::Result;
use metrale_config::{ModelConfig, WeightQuantPolicy};
use metrale_model_layers::layers::MoeLayer;
use metrale_model_layers::weight_map::Nvfp4Variant;
use metrale_model_weights::weights::WeightStore;

/// 2026-10-02: The declared-precision answers for one model load.
pub(super) struct Declared<'a> {
    policy: WeightQuantPolicy<'a>,
    store: &'a WeightStore,
    /// 2026-10-02: Whether the per-layer FP8 arms below may apply: not a native-FP8 checkpoint
    /// (which takes them already), not the Holo checkpoint (its own arms), TP = 1 (the FP8 SSM
    /// arm has no TP form).
    fp8_arms_open: bool,
}

impl<'a> Declared<'a> {
    pub(super) fn new(
        config: &'a ModelConfig,
        store: &'a WeightStore,
        native_fp8: bool,
        modelopt_mixed_precision: bool,
    ) -> Self {
        Self {
            policy: WeightQuantPolicy::for_checkpoint(
                metrale_model_layers::layers::weight_quantization(),
                config.quantization_config.as_ref(),
                metrale_model_layers::layers::kernel_caps(),
            ),
            store,
            fp8_arms_open: !native_fp8
                && !modelopt_mixed_precision
                && config.tp_world_size.max(1) == 1,
        }
    }

    /// 2026-10-02: Whether every module `{prefix}.{name}` is declared FP8 and stored as FP8
    /// E4M3 with a per-tensor or 128x128 block scale, so it is served at its FP8 weights through
    /// the native FP8 arms (the scalar scale is repeated over the block grid,
    /// `load_fp8_block_scaled_as_fp8weight`) instead of being requantized to NVFP4.
    pub(super) fn fp8(&self, prefix: &str, names: &[&str]) -> bool {
        self.fp8_arms_open
            && names.iter().all(|n| {
                let m = format!("{prefix}.{n}");
                self.policy.wants_fp8_weights(&m)
                    && super::super::super::qwen35_dense::proj_is_fp8_any_scale(self.store, &m)
            })
    }

    /// 2026-10-02: Whether layer `lp`'s router stays BF16 because the checkpoint leaves it
    /// unquantized.
    pub(super) fn router_bf16(&self, lp: &str) -> bool {
        self.policy.follows_plan()
            && self
                .policy
                .declared(&format!("{lp}.mlp.gate"))
                .weight
                .is_none()
    }

    /// 2026-10-02: Mark layer `lp`'s experts as the checkpoint's own NVFP4 (W4A16) when they are:
    /// the `Standard` variant, no FP8 experts loaded, a BF16 router, experts declared FP4. The
    /// grouped NVFP4 decode then serves them at every width.
    pub(super) fn mark_nvfp4_experts(
        &self,
        moe: &mut MoeLayer,
        lp: &str,
        i: usize,
        variant: Nvfp4Variant,
        fp8_experts: bool,
    ) -> Result<()> {
        let declared_fp4 = self
            .policy
            .declared(&format!("{lp}.mlp.experts"))
            .weight
            .is_some_and(|w| w.is_fp4());
        if !(variant == Nvfp4Variant::Standard
            && !fp8_experts
            && self.router_bf16(lp)
            && declared_fp4)
        {
            return Ok(());
        }
        moe.set_declared_nvfp4_experts()?;
        if i == 0 {
            tracing::info!(
                "--weight-quantization declared: routed and shared experts decode at the \
                 checkpoint's NVFP4 (W4A16) through the grouped NVFP4 path; router BF16"
            );
        }
        Ok(())
    }
}
