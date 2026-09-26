// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The MTP modules `build_model` loads besides the generic
//! `MtpWeights` (GLM-5.3, DeepSeek-V4), and the MTP head's effective
//! quantization.
//!
//! Owner: metrale-model-engine.
//! Invariants:
//! - A module load error is logged and yields `None`; it is not returned.

use metrale_cache::kv_cache::KvCacheDtype;
use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_model_arch::weight_loader::deepseek_v4::mtp::DeepseekV4MtpModule;
use metrale_model_arch::weight_loader::glm5_next_mtp::Glm5NextMtpModule;
use metrale_model_layers::layers::MtpQuantization;
use metrale_model_layers::weight_map::MtpWeights;
use metrale_model_weights::weights::WeightStore;

/// 2026-09-26: The GLM-5.3 MTP module, for `glm5_next` with `--speculative`.
pub(super) fn load_glm_mtp_module(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    use_speculative: bool,
) -> Option<Glm5NextMtpModule> {
    if config.model_type == "glm5_next" && use_speculative {
        match metrale_model_arch::weight_loader::glm5_next_mtp::load_glm5next_mtp_module(
            store, config, gpu,
        ) {
            Ok(Some(m)) => {
                tracing::info!(target: "metrale_model_engine::factory::build", "GLM-5.3 MTP draft module loaded (layers.{})",
                    config.num_hidden_layers
                );
                Some(m)
            }
            Ok(None) => {
                tracing::info!(target: "metrale_model_engine::factory::build", "GLM-5.3: no MTP block in checkpoint (MTP off)");
                None
            }
            Err(e) => {
                tracing::error!(target: "metrale_model_engine::factory::build", "GLM-5.3 MTP module load FAILED: {e:#}");
                None
            }
        }
    } else {
        None
    }
}

/// 2026-09-26: The DeepSeek-V4 MTP module, for `deepseek_v4` with
/// `--speculative` on EP rank 0.
pub(super) fn load_v4_mtp_module(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    attn_layer_dtypes: &[KvCacheDtype],
    use_speculative: bool,
) -> Option<DeepseekV4MtpModule> {
    if config.model_type == "deepseek_v4" && use_speculative && config.ep_rank == 0 {
        match metrale_model_arch::weight_loader::deepseek_v4::mtp::load_v4_mtp_module(
            store,
            config,
            gpu,
            attn_layer_dtypes,
        ) {
            Ok(Some(m)) => {
                tracing::info!(target: "metrale_model_engine::factory::build", "DeepSeek-V4 MTP draft module loaded OK (num_mtp_modules={})",
                    config.num_mtp_modules
                );
                Some(m)
            }
            Ok(None) => {
                tracing::info!(target: "metrale_model_engine::factory::build", "DeepSeek-V4: no MTP module in checkpoint (MTP off)");
                None
            }
            Err(e) => {
                tracing::error!(target: "metrale_model_engine::factory::build", "DeepSeek-V4 MTP module load FAILED: {e:#}");
                None
            }
        }
    } else {
        None
    }
}

/// 2026-09-26: The MTP head's quantization: `mtp_quant`, or BF16 when the
/// checkpoint's quantization config ignores the MTP weights.
pub(super) fn effective_mtp_quantization(
    mtp_weights: &[MtpWeights],
    config: &ModelConfig,
    store: &WeightStore,
    mtp_quant: MtpQuantization,
) -> MtpQuantization {
    if !mtp_weights.is_empty() {
        let quant_fmt = metrale_model_layers::quant_format::detect_quant_format(config, store);
        if quant_fmt.is_ignored("mtp.fc.weight")
            || quant_fmt.is_ignored("mtp.layers.0.self_attn.q_proj.weight")
        {
            if mtp_quant != MtpQuantization::Bf16 {
                tracing::info!(target: "metrale_model_engine::factory::build", "MTP head listed in checkpoint ignore_modules — overriding \
                     --mtp-quantization {:?} → Bf16 to preserve precision",
                    mtp_quant,
                );
            }
            MtpQuantization::Bf16
        } else {
            mtp_quant
        }
    } else {
        mtp_quant
    }
}

/// 2026-09-26: Logs why `--speculative` found no MTP module: a warning when
/// the checkpoint ships no MTP head, an error when it ships one no loader bound.
pub(super) fn warn_unbound_mtp_head(store: &WeightStore, config: &ModelConfig) {
    match metrale_model_weights::mtp_layout::detect_in_store(store, config) {
        None => {
            tracing::warn!(target: "metrale_model_engine::factory::build", "`--speculative` was requested but this checkpoint ships no MTP head — \
                 speculative decoding will be disabled. Either drop `--speculative` or \
                 use a checkpoint that ships one (e.g. `mtp.safetensors`)."
            )
        }
        Some(layout) => {
            tracing::error!(target: "metrale_model_engine::factory::build", "`--speculative` was requested and this checkpoint DOES ship MTP weights \
                 ({layout:?}), but no loader bound them for model_type '{}' — speculative \
                 decoding will be disabled. This is a Metrale Engine capability gap, not a \
                 checkpoint problem.",
                config.model_type,
            )
        }
    }
}
