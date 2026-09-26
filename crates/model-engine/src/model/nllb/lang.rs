// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Translation language tokens for the served NLLB model.
//!
//! The encoder input is `[src_lang] + tokens + </s>`, and decoding is seeded with the
//! target-language token (`forced_bos`). The server's tokenizer resolves language names
//! to ids, so this struct carries only ids.
//!
//! Owner: model-engine.
//! Invariants: none beyond the types.

/// 2026-09-25: Resolved translation tokens for the deployment. `src_lang_id` and
/// `tgt_lang_id` are the defaults from `--src-lang`/`--tgt-lang`; a request's own
/// non-zero language ids override them. The factory sets `decoder_start_id` and
/// `eos_id` to the config's `eos_token_id` and `pad_id` to 1.
#[derive(Debug, Clone, Copy)]
pub struct NllbLang {
    /// 2026-09-25: Source-language token prepended to the encoder input.
    pub src_lang_id: u32,
    /// 2026-09-25: Target-language token forced as the first decoded token (`forced_bos`).
    pub tgt_lang_id: u32,
    /// 2026-09-25: Decoder start token, the first decoder input.
    pub decoder_start_id: u32,
    /// 2026-09-25: End-of-sequence token: appended to the encoder input, and it
    /// ends a beam hypothesis.
    pub eos_id: u32,
    /// 2026-09-25: Padding token id; encoder positions skip it.
    pub pad_id: u32,
}

impl NllbLang {
    /// 2026-09-25: Format the raw source subword ids into the encoder input
    /// `[src_lang] + tokens + </s>` using the deployment-default source language.
    pub(super) fn encoder_input(&self, tokens: &[u32]) -> Vec<u32> {
        self.encoder_input_with(self.src_lang_id, tokens)
    }

    /// 2026-09-25: Encoder input with an explicit (per-request) source-language token.
    pub(super) fn encoder_input_with(&self, src_lang_id: u32, tokens: &[u32]) -> Vec<u32> {
        let mut ids = Vec::with_capacity(tokens.len() + 2);
        ids.push(src_lang_id);
        ids.extend_from_slice(tokens);
        ids.push(self.eos_id);
        ids
    }
}
