// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Feature flags derived from a parsed `ModelConfig` by `ModelConfig::capabilities`.
//!
//! Owner: config.
//! Invariants: none beyond the types.

/// 2026-09-26: Recurrent-layer family, chosen in `ModelCapabilities::from_config`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SsmArchitecture {
    None,
    /// 2026-09-26: Some layer is `LayerType::LinearAttention` and the Mamba-2 head dims are unset.
    Gdn,
    /// 2026-09-26: `mamba_num_heads` and `mamba_head_dim` are both non-zero; this wins over `Gdn`.
    Mamba2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionType {
    Standard,
    /// 2026-09-26: Multi-head latent attention, chosen when `kv_lora_rank > 0`.
    Mla,
}

#[derive(Debug, Clone)]
pub struct ModelCapabilities {
    /// 2026-09-26: Some layer is `LinearAttention`, or the Mamba-2 head dims are set.
    pub has_ssm_layers: bool,
    /// 2026-09-26: Some layer is `FullAttention`.
    pub has_attention_layers: bool,
    /// 2026-09-26: `num_experts > 0`.
    pub has_moe_layers: bool,
    /// 2026-09-26: Equal to `has_ssm_layers`: it follows the architecture, not the tokenizer
    /// vocabulary.
    pub supports_thinking: bool,
    /// 2026-09-26: `ModelConfig::vision` is set.
    pub supports_vision: bool,
    /// 2026-09-26: `mtp_num_hidden_layers > 0`.
    pub has_mtp: bool,
    pub ssm_architecture: SsmArchitecture,
    pub attention_type: AttentionType,
    /// 2026-09-26: Copied from `ModelConfig::nested_config`, which a parser sets when the text
    /// model's fields sit under a nested key such as `text_config`.
    pub has_nested_config: bool,
}

impl ModelCapabilities {
    pub fn from_config(config: &crate::ModelConfig) -> Self {
        use crate::LayerType;

        let has_ssm = config
            .layer_types
            .iter()
            .any(|t| matches!(t, LayerType::LinearAttention));
        let has_mamba2 = config.mamba_num_heads > 0 && config.mamba_head_dim > 0;
        let has_attention = config
            .layer_types
            .iter()
            .any(|t| matches!(t, LayerType::FullAttention));
        let has_moe = config.num_experts > 0;
        let has_vision = config.vision.is_some();
        let has_mtp = config.mtp_num_hidden_layers > 0;
        let has_nested = config.nested_config;

        let ssm_arch = if has_mamba2 {
            SsmArchitecture::Mamba2
        } else if has_ssm {
            SsmArchitecture::Gdn
        } else {
            SsmArchitecture::None
        };

        Self {
            has_ssm_layers: has_ssm || has_mamba2,
            has_attention_layers: has_attention,
            has_moe_layers: has_moe,
            supports_thinking: has_ssm || has_mamba2,
            supports_vision: has_vision,
            has_mtp,
            ssm_architecture: ssm_arch,
            has_nested_config: has_nested,
            attention_type: if config.kv_lora_rank > 0 {
                AttentionType::Mla
            } else {
                AttentionType::Standard
            },
        }
    }
}
