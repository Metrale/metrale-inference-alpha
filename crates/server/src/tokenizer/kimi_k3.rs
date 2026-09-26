// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Official Kimi K3 checkpoints: detection (`kimi_k3` model type with a
//! `tiktoken.model`, or a `TikTokenTokenizer` tokenizer config) and the chat refusal. Their
//! XTML chat encoding is not implemented, so every chat-template apply path returns an
//! error rather than rendering ChatML; `/v1/completions` still works.
//!
//! Owner: server (tokenizer).
//! Invariants: none beyond the types.

use anyhow::{Context, Result, bail};
use std::path::Path;

pub(super) fn uses_xtml(model_dir: &Path, model_type: &str) -> Result<bool> {
    if model_type != "kimi_k3" {
        return Ok(false);
    }
    if model_dir.join("tiktoken.model").exists() {
        return Ok(true);
    }
    let config = model_dir.join("tokenizer_config.json");
    if !config.exists() {
        return Ok(false);
    }
    let value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(config).context("Read Kimi tokenizer config")?)
            .context("Parse Kimi tokenizer config")?;
    Ok(value["tokenizer_class"].as_str() == Some("TikTokenTokenizer"))
}

pub(super) fn require_chat_support(encoding: super::ChatEncoding) -> Result<()> {
    if encoding == super::ChatEncoding::KimiK3XtmlUnsupported {
        bail!(
            "Kimi K3 XTML chat/reasoning/tool parsing is not implemented; \
             use /v1/completions with independently prepared token IDs for bring-up. \
             Generic ChatML and K2 tool parsers are not compatible."
        );
    }
    Ok(())
}

impl super::ChatTokenizer {
    /// 2026-09-26: True for an official Kimi K3 checkpoint; `/v1/completions` then skips
    /// `<think>` stripping on the output.
    pub(crate) fn uses_kimi_k3_xtml(&self) -> bool {
        self.chat_encoding == super::ChatEncoding::KimiK3XtmlUnsupported
    }
}
