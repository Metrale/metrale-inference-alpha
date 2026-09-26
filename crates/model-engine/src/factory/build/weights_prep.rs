// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Load-time steps of `build_model` that adjust the config or the
//! weight store: the DFlash capture layers and γ, and the release of a vision
//! tower that no encoder bound.
//!
//! Owner: metrale-model-engine.
//! Invariants:
//! - `release_unbound_vision_tower` frees only store tensors whose names start
//!   with a vision-tower prefix.

use anyhow::Result;
use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_model_weights::weights::WeightStore;

use crate::factory::DflashBuildArgs;

/// 2026-09-26: With DFlash, sets `config.dflash_capture_layers` and
/// `config.dflash_gamma` from the drafter; without it, leaves `config` as is.
pub(super) fn apply_dflash_capture_config(
    config: &mut ModelConfig,
    dflash_args: &Option<DflashBuildArgs<'_>>,
) {
    if let Some(args) = dflash_args
        && let Some(ref sub) = args.drafter_config.dflash_config
    {
        config.dflash_capture_layers = sub.target_layer_ids.clone();
        // 2026-09-25: γ is `--dflash-gamma` if given, else
        // `default_dflash_gamma` of the drafter's `effective_block_size`. The
        // DFlash head resolves its γ through the same helper
        // (dflash_head/from_weights.rs), so the pools sized from this value
        // match the head.
        config.dflash_gamma = Some(args.gamma.unwrap_or_else(|| {
            metrale_model_layers::layers::qwen3_ssm::default_dflash_gamma(
                args.drafter_config.effective_block_size(),
            )
        }));
        tracing::info!(target: "metrale_model_engine::factory::build", "DFlash: target layer capture indices = {:?} (drafter target_layer_ids, \
             used directly), γ = {:?}",
            config.dflash_capture_layers,
            config.dflash_gamma,
        );
    }
}

/// 2026-09-26: Frees the store's vision-tower tensors and logs what was freed.
pub(super) fn release_unbound_vision_tower(
    store: &mut WeightStore,
    gpu: &dyn GpuBackend,
    config: &ModelConfig,
) -> Result<()> {
    let (n, bytes) = store.free_matching(gpu, |name| {
        name.starts_with("model.visual.")
            || name.starts_with("model.vision")
            || name.starts_with("visual.")
    })?;
    if n > 0 {
        tracing::info!(target: "metrale_model_engine::factory::build", "Vision tower: {n} tensors ({:.2} GiB) released — this build binds no vision \
             encoder for model_type '{}', so the tower was resident and unreachable. \
             Text capability is unchanged; image input was already unsupported here.",
            bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            config.model_type,
        );
    }
    Ok(())
}
