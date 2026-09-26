// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Logprobs extraction helpers.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::*;

/// 2026-09-25: Logprobs for one position from FP32 logits: the sampled token's
/// log-softmax value and the top-`k` alternatives, descending by logprob.
pub fn extract_logprobs_from_f32(
    f32_logits: &[f32],
    sampled_token: u32,
    k: usize,
) -> crate::api::TokenLogprobs {
    // 2026-09-25: The math is `metrale_model_engine::traits::logprob_of`, which the
    // prompt-logprob path (`traits::logprobs::extract_bf16`) also uses.
    let (logprob, top) = metrale_model_engine::traits::logprob_of(f32_logits, sampled_token, k);
    crate::api::TokenLogprobs {
        token_id: sampled_token,
        logprob,
        top,
    }
}

/// 2026-09-25: Logprobs for `tokens.len()` verify positions, read from the model's
/// BF16 logits buffer: copies the rows to the host, converts them to FP32 and
/// calls [`extract_logprobs_from_f32`] per row. Returns an empty `Vec` when the
/// copy fails.
///
/// `row_base` is the first logits row of this sequence's verify span in the
/// shared buffer. The single-sequence verify steps pass 0; the batched verify
/// (`verify_k4_batch_step`) passes the sequence's row offset.
pub fn extract_verify_logprobs(
    model: &dyn Model,
    tokens: &[u32],
    k_logprobs: u8,
    row_base: usize,
) -> Vec<crate::api::TokenLogprobs> {
    let k = tokens.len();
    let vocab = model.vocab_size();
    let mut buf = vec![0u8; k * vocab * 2];
    if model
        .copy_logits_to_host(
            model.logits_buffer_ptr().offset(row_base * vocab * 2),
            &mut buf,
        )
        .is_err()
    {
        return Vec::new();
    }
    tokens
        .iter()
        .enumerate()
        .map(|(i, &tok)| {
            let slice = &buf[i * vocab * 2..(i + 1) * vocab * 2];
            let f32_logits: Vec<f32> = (0..vocab)
                .map(|j| {
                    let lo = slice[j * 2];
                    let hi = slice[j * 2 + 1];
                    bf16_to_f32(lo, hi)
                })
                .collect();
            extract_logprobs_from_f32(&f32_logits, tok, k_logprobs as usize)
        })
        .collect()
}

/// 2026-09-25: Logprobs for one token from a BF16 logits row on the device: copies
/// it to the host, converts it to FP32 and calls [`extract_logprobs_from_f32`].
/// Returns `None` when the copy fails.
pub fn extract_single_logprobs(
    model: &dyn Model,
    logits: DevicePtr,
    sampled_token: u32,
    k_logprobs: u8,
) -> Option<crate::api::TokenLogprobs> {
    let vocab = model.vocab_size();
    let mut buf = vec![0u8; vocab * 2];
    if model.copy_logits_to_host(logits, &mut buf).is_err() {
        return None;
    }
    let f32_logits: Vec<f32> = (0..vocab)
        .map(|j| {
            let lo = buf[j * 2];
            let hi = buf[j * 2 + 1];
            bf16_to_f32(lo, hi)
        })
        .collect();
    Some(extract_logprobs_from_f32(
        &f32_logits,
        sampled_token,
        k_logprobs as usize,
    ))
}
