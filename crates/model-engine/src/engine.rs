// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: A greedy generate loop for one request over the [`Model`]
//! trait: prefill, then decode, taking each token with `argmax_on_device`.
//!
//! Owner: metrale-model-engine.
//! Invariants:
//! - The engine holds no state: [`generate`] and [`generate_streaming`]
//!   allocate a fresh sequence per call and call `free_sequence` on it whether
//!   generation returned `Ok` or `Err`.

use anyhow::Result;
use metrale_sampling::SamplingParams;

use crate::traits::Model;

/// 2026-09-25: Result of a generate call.
pub struct GenerateResult {
    /// 2026-09-25: Output tokens, without the prompt tokens.
    pub output_tokens: Vec<u32>,
    /// 2026-09-25: Why generation stopped: `"stop"` (a token in
    /// `params.stop_token_ids`) or `"length"`.
    pub finish_reason: String,
}

/// 2026-09-25: Generate response tokens from a prompt.
///
/// Runs prefill, takes its token, then decodes until `params.max_tokens`
/// tokens are out or a token in `params.stop_token_ids` is produced. The
/// prefill token is always returned, so the output has at least one token.
pub fn generate(
    model: &dyn Model,
    prompt_tokens: &[u32],
    params: &SamplingParams,
) -> Result<GenerateResult> {
    let mut seq = model.alloc_sequence()?;
    let stream = 0u64;

    let result = generate_inner(model, prompt_tokens, params, &mut seq, stream);

    // 2026-09-25: Free the sequence whether or not generation failed. An
    // error from `free_sequence` replaces the result.
    model.free_sequence(&mut seq)?;

    result
}

fn generate_inner(
    model: &dyn Model,
    prompt_tokens: &[u32],
    params: &SamplingParams,
    seq: &mut crate::traits::SequenceState,
    stream: u64,
) -> Result<GenerateResult> {
    let logits_ptr = model.prefill(prompt_tokens, seq, stream)?;
    let first_token = model.argmax_on_device(logits_ptr, stream)?;

    let mut output_tokens = Vec::with_capacity(params.max_tokens);
    output_tokens.push(first_token);

    if params.stop_token_ids.contains(&first_token) {
        return Ok(GenerateResult {
            output_tokens,
            finish_reason: "stop".to_string(),
        });
    }

    for _step in 1..params.max_tokens {
        let last_token = *output_tokens.last().unwrap();
        let logits_ptr = model.decode(last_token, seq, stream)?;
        let token = model.argmax_on_device(logits_ptr, stream)?;

        output_tokens.push(token);

        if params.stop_token_ids.contains(&token) {
            return Ok(GenerateResult {
                output_tokens,
                finish_reason: "stop".to_string(),
            });
        }
    }

    Ok(GenerateResult {
        output_tokens,
        finish_reason: "length".to_string(),
    })
}

/// 2026-09-25: Same as [`generate`], but calls `on_token(token_id)`
/// synchronously after each token, the prefill token included.
pub fn generate_streaming<F>(
    model: &dyn Model,
    prompt_tokens: &[u32],
    params: &SamplingParams,
    mut on_token: F,
) -> Result<GenerateResult>
where
    F: FnMut(u32),
{
    let mut seq = model.alloc_sequence()?;
    let stream = 0u64;

    let result = generate_streaming_inner(
        model,
        prompt_tokens,
        params,
        &mut on_token,
        &mut seq,
        stream,
    );

    model.free_sequence(&mut seq)?;

    result
}

fn generate_streaming_inner<F>(
    model: &dyn Model,
    prompt_tokens: &[u32],
    params: &SamplingParams,
    on_token: &mut F,
    seq: &mut crate::traits::SequenceState,
    stream: u64,
) -> Result<GenerateResult>
where
    F: FnMut(u32),
{
    let logits_ptr = model.prefill(prompt_tokens, seq, stream)?;
    let first_token = model.argmax_on_device(logits_ptr, stream)?;

    let mut output_tokens = Vec::with_capacity(params.max_tokens);
    output_tokens.push(first_token);
    on_token(first_token);

    if params.stop_token_ids.contains(&first_token) {
        return Ok(GenerateResult {
            output_tokens,
            finish_reason: "stop".to_string(),
        });
    }

    for _step in 1..params.max_tokens {
        let last_token = *output_tokens.last().unwrap();
        let logits_ptr = model.decode(last_token, seq, stream)?;
        let token = model.argmax_on_device(logits_ptr, stream)?;

        output_tokens.push(token);
        on_token(token);

        if params.stop_token_ids.contains(&token) {
            return Ok(GenerateResult {
                output_tokens,
                finish_reason: "stop".to_string(),
            });
        }
    }

    Ok(GenerateResult {
        output_tokens,
        finish_reason: "length".to_string(),
    })
}

/// 2026-09-25: Generate with MTP speculative decoding, delegated to
/// `Model::generate_speculative`.
pub fn generate_speculative(
    model: &dyn Model,
    prompt_tokens: &[u32],
    params: &SamplingParams,
    num_drafts: usize,
) -> Result<GenerateResult> {
    model.generate_speculative(prompt_tokens, params, num_drafts)
}

#[cfg(test)]
mod tests;
