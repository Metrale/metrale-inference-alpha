// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Model directory and config for serve: resolve the directory,
//! parse its config, merge a sibling quant config, cap the vocab to the
//! tokenizer, and resolve `num_drafts`.
//!
//! Owner: server startup (`met serve`).
//! Invariants: none beyond the types.

use std::path::Path;

use anyhow::{Context, Result};

use metrale_config::ModelConfig;

use crate::cli;

pub(crate) fn merge_sidecar_quant_config(model_dir: &Path, config: &mut ModelConfig) {
    if config.quantization_config.is_some() {
        return;
    }
    let hf_quant_path = model_dir.join("hf_quant_config.json");
    if !hf_quant_path.exists() {
        return;
    }
    match std::fs::read_to_string(&hf_quant_path) {
        Ok(raw_hq) => {
            let wrapped = format!(r#"{{"quantization_config":{raw_hq}}}"#);
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&wrapped) {
                config.quantization_config = metrale_config::parse_quantization_config(&v);
            }
        }
        Err(e) => tracing::warn!("Failed to read sibling hf_quant_config.json: {e}"),
    }
}

pub(crate) fn load_model_config(model_dir: &Path) -> Result<(ModelConfig, String)> {
    let config_path = model_dir.join("config.json");
    let params_path = model_dir.join("params.json");

    // 2026-09-26: A directory with no config.json or params.json but a GGUF
    // file: build the config from the GGUF metadata.
    if !config_path.exists()
        && !params_path.exists()
        && metrale_model_weights::weights::find_gguf(model_dir).is_some()
    {
        let config = metrale_model_weights::weights::config_from_gguf_dir(model_dir)
            .context("Failed to build ModelConfig from GGUF metadata")?;
        tracing::info!(
            "Built ModelConfig from GGUF metadata (model_type={}, layers={}, hidden={})",
            config.model_type,
            config.num_hidden_layers,
            config.hidden_size,
        );
        // 2026-09-26: No config text. `resolve_model_name`, its only reader,
        // then skips the `_name_or_path` step.
        return Ok((config, String::new()));
    }

    let config_json = if config_path.exists() {
        std::fs::read_to_string(&config_path)
            .with_context(|| format!("Failed to read {}", config_path.display()))?
    } else if params_path.exists() {
        std::fs::read_to_string(&params_path)
            .with_context(|| format!("Failed to read {}", params_path.display()))?
    } else {
        anyhow::bail!(
            "No config.json, params.json, or .gguf found in {}",
            model_dir.display()
        );
    };
    let config = if params_path.exists() && !config_path.exists() {
        metrale_config::parse_mistral_params(&config_json)
            .context("Failed to parse params.json (Mistral format)")?
    } else {
        metrale_config::parse_config(&config_json).context("Failed to parse config.json")?
    };
    Ok((config, config_json))
}

pub(crate) fn resolve_model_dir(args: &cli::ServeArgs) -> Result<std::path::PathBuf> {
    use crate::model_resolver;
    if let Some(ref path) = args.model_from_path {
        model_resolver::resolve_model_dir(
            path.to_str().context("Invalid model path")?,
            args.cache_dir.as_deref(),
        )
    } else {
        let model_spec = args
            .model
            .as_deref()
            .context("Either MODEL or --model-from-path is required")?;
        model_resolver::resolve_model_dir(model_spec, args.cache_dir.as_deref())
    }
}

pub(crate) fn cap_vocab_size_to_tokenizer(model_dir: &Path, config: &mut ModelConfig) {
    let tok_path = model_dir.join("tokenizer.json");
    if tok_path.exists()
        && let Ok(tok) = tokenizers::Tokenizer::from_file(&tok_path)
    {
        let tok_vocab = tok.get_vocab_size(true);
        if tok_vocab > 0 && tok_vocab < config.vocab_size {
            tracing::info!(
                "Capping vocab_size from {} to {} (tokenizer)",
                config.vocab_size,
                tok_vocab,
            );
            config.vocab_size = tok_vocab;
        }
    }
}

/// 2026-09-26: Where the effective `num_drafts` came from: `--num-drafts`,
/// MODEL.toml `[behavior].default_num_drafts`, or `cli::DEFAULT_NUM_DRAFTS`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NumDraftsSource {
    Cli,
    ModelDefault,
    EngineDefault,
}

/// 2026-09-26: Resolve the effective draft count. An explicit `--num-drafts`
/// wins, including a value equal to the engine default. An omitted flag uses
/// MODEL.toml `default_num_drafts` when it is above 0, else
/// `cli::DEFAULT_NUM_DRAFTS`.
pub(crate) fn resolve_num_drafts(
    cli_num_drafts: Option<usize>,
    model_default_num_drafts: u32,
) -> (usize, NumDraftsSource) {
    let model_default = (model_default_num_drafts > 0).then_some(model_default_num_drafts as usize);
    match (cli_num_drafts, model_default) {
        (Some(v), _) => (v, NumDraftsSource::Cli),
        (None, Some(md)) => (md, NumDraftsSource::ModelDefault),
        (None, None) => (cli::DEFAULT_NUM_DRAFTS, NumDraftsSource::EngineDefault),
    }
}

pub(crate) fn apply_model_default_num_drafts(
    args: &mut cli::ServeArgs,
    ptx_set: &metrale_kernels::TargetPtxSet,
) {
    let (effective, source) =
        resolve_num_drafts(args.num_drafts, ptx_set.behavior.default_num_drafts);
    match source {
        NumDraftsSource::Cli => {
            let model_default = ptx_set.behavior.default_num_drafts as usize;
            if ptx_set.behavior.default_num_drafts > 0 && model_default != effective {
                tracing::info!(
                    "num_drafts: {} (K={}) from --num-drafts, overriding MODEL.toml default_num_drafts={}",
                    effective,
                    effective + 1,
                    model_default,
                );
            }
        }
        NumDraftsSource::ModelDefault => {
            tracing::info!(
                "num_drafts: using MODEL.toml default_num_drafts={} (K={}) — pass --num-drafts to override",
                effective,
                effective + 1,
            );
        }
        NumDraftsSource::EngineDefault => {}
    }
    args.num_drafts = Some(effective);
}

#[cfg(test)]
mod tests {
    use super::{NumDraftsSource, resolve_num_drafts};

    /// 2026-09-26: `--num-drafts 1` (equal to `cli::DEFAULT_NUM_DRAFTS`) on a
    /// model with `default_num_drafts = 3` serves 1, not 3.
    #[test]
    fn explicit_cli_value_equal_to_engine_default_beats_model_default() {
        assert_eq!(resolve_num_drafts(Some(1), 3), (1, NumDraftsSource::Cli));
    }

    #[test]
    fn explicit_cli_value_beats_model_default() {
        assert_eq!(resolve_num_drafts(Some(2), 3), (2, NumDraftsSource::Cli));
        assert_eq!(resolve_num_drafts(Some(3), 1), (3, NumDraftsSource::Cli));
    }

    #[test]
    fn omitted_flag_falls_back_to_model_default() {
        assert_eq!(
            resolve_num_drafts(None, 3),
            (3, NumDraftsSource::ModelDefault)
        );
    }

    #[test]
    fn omitted_flag_without_model_default_uses_engine_default() {
        assert_eq!(
            resolve_num_drafts(None, 0),
            (
                crate::cli::DEFAULT_NUM_DRAFTS,
                NumDraftsSource::EngineDefault
            )
        );
    }
}
