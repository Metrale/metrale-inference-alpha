// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Resolve the `/v1/completions` `prompt` field into token lists.
//!
//! Owner: server completions API.
//! Invariants: none beyond the types.

use axum::http::StatusCode;

use crate::AppState;
use crate::openai::PromptInput;

/// 2026-09-26: Resolve an OpenAI `prompt` field into one token list per prompt.
/// Text is tokenized without special tokens (no BOS); token-ID forms are used
/// verbatim after [`validate_token_ids`] (out of range → 400).
pub(super) fn resolve_prompts(
    state: &AppState,
    prompt: &PromptInput,
) -> Result<Vec<Vec<u32>>, (StatusCode, String)> {
    match prompt {
        PromptInput::Text(s) => {
            Ok(vec![state.tokenizer.encode(s).map_err(|e| {
                (StatusCode::BAD_REQUEST, format!("Tokenization error: {e}"))
            })?])
        }
        PromptInput::TextArray(parts) => {
            // 2026-09-26: Each array element is a separate prompt with its
            // own choices (`completions_exec::run_blocking`).
            parts
                .iter()
                .map(|part| {
                    state
                        .tokenizer
                        .encode(part)
                        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Tokenization error: {e}")))
                })
                .collect()
        }
        PromptInput::TokenIds(ids) => {
            validate_token_ids(state, ids)?;
            Ok(vec![ids.clone()])
        }
        PromptInput::TokenIdBatch(batch) => {
            // 2026-09-26: One prompt per sub-array.
            for ids in batch {
                validate_token_ids(state, ids)?;
            }
            Ok(batch.clone())
        }
    }
}

/// 2026-09-26: Reject (400) a token ID at or above the tokenizer's vocabulary
/// size, added tokens included.
fn validate_token_ids(state: &AppState, ids: &[u32]) -> Result<(), (StatusCode, String)> {
    let vocab_size = state.tokenizer.inner().get_vocab_size(true) as u32;
    if let Some(&bad) = ids.iter().find(|&&id| id >= vocab_size) {
        return Err((
            StatusCode::BAD_REQUEST,
            format!("Token ID {bad} out of range: vocab_size is {vocab_size}"),
        ));
    }
    Ok(())
}
