// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `/v1/completions` logprobs: the four parallel arrays `tokens`,
//! `token_logprobs`, `top_logprobs` and `text_offset`, with echoed prompt
//! tokens first.
//!
//! `text_offset` is the cumulative byte length of the per-token decoded
//! strings. Per-token decodes need not concatenate to the full-sequence decode
//! in `text` (a multibyte character split across tokens, for one), so an
//! offset can differ from the position in `text` near such a token.
//!
//! Owner: server completions API.
//! Invariants: the four arrays have equal length.

use crate::api::inference_types::TokenLogprobs;
use crate::openai::CompletionLogprobs;

/// 2026-09-26: Assemble the logprobs block.
///
/// * `decode`: per-token-ID detokenizer, passed in so unit tests need no
///   tokenizer; the handlers pass `state.tokenizer.decode(&[id])`.
/// * `prompt_tokens`/`prompt_lps`: read only when `echo`. `prompt_lps[i]` is
///   attached to `prompt_tokens[i + 1]`; the first prompt token gets `null`.
/// * `gen_tokens`/`gen_lps`: generated tokens and their logprobs. When
///   `gen_lps` is shorter, the missing entries are taken to be the first ones,
///   and those tokens get `null`.
pub(super) fn build_completion_logprobs(
    decode: &dyn Fn(u32) -> String,
    echo: bool,
    prompt_tokens: &[u32],
    prompt_lps: &[TokenLogprobs],
    gen_tokens: &[u32],
    gen_lps: &[TokenLogprobs],
) -> CompletionLogprobs {
    let cap = if echo { prompt_tokens.len() } else { 0 } + gen_tokens.len();
    let mut tokens: Vec<String> = Vec::with_capacity(cap);
    let mut token_logprobs: Vec<Option<f32>> = Vec::with_capacity(cap);
    let mut top_logprobs = Vec::with_capacity(cap);
    let mut text_offset: Vec<usize> = Vec::with_capacity(cap);
    let mut offset = 0usize;

    let mut push = |tok_id: u32, lp: Option<&TokenLogprobs>| {
        let piece = decode(tok_id);
        text_offset.push(offset);
        offset += piece.len();
        tokens.push(piece);
        token_logprobs.push(lp.map(|l| l.logprob));
        top_logprobs.push(lp.map(|l| {
            l.top
                .iter()
                .map(|&(id, p)| (decode(id), p))
                .collect::<std::collections::HashMap<String, f32>>()
        }));
    };

    if echo {
        for (i, &tok) in prompt_tokens.iter().enumerate() {
            push(tok, i.checked_sub(1).and_then(|j| prompt_lps.get(j)));
        }
    }
    let gen_pad = gen_tokens.len().saturating_sub(gen_lps.len());
    for (i, &tok) in gen_tokens.iter().enumerate() {
        push(tok, i.checked_sub(gen_pad).and_then(|j| gen_lps.get(j)));
    }

    CompletionLogprobs {
        tokens,
        token_logprobs,
        top_logprobs,
        text_offset,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lp(token_id: u32, logprob: f32) -> TokenLogprobs {
        TokenLogprobs {
            token_id,
            logprob,
            top: vec![(token_id, logprob)],
        }
    }

    fn dec(id: u32) -> String {
        format!("t{id} ")
    }

    #[test]
    fn echo_null_first_then_prompt_scores() {
        let out = build_completion_logprobs(
            &dec,
            true,
            &[10, 11, 12],
            &[lp(11, -0.1), lp(12, -0.2)],
            &[],
            &[],
        );
        assert_eq!(out.tokens, vec!["t10 ", "t11 ", "t12 "]);
        assert_eq!(out.token_logprobs, vec![None, Some(-0.1), Some(-0.2)]);
        assert!(out.top_logprobs[0].is_none());
        assert!(out.top_logprobs[1].as_ref().unwrap().contains_key("t11 "));
    }

    #[test]
    fn text_offset_is_cumulative_decoded_length() {
        let out =
            build_completion_logprobs(&dec, true, &[7, 8], &[lp(8, -0.3)], &[9], &[lp(9, -0.4)]);
        assert_eq!(out.text_offset, vec![0, 3, 6]);
    }

    #[test]
    fn echo_concatenates_prompt_then_generated() {
        let out = build_completion_logprobs(
            &dec,
            true,
            &[1, 2],
            &[lp(2, -0.1)],
            &[3, 4],
            &[lp(3, -0.2), lp(4, -0.3)],
        );
        assert_eq!(out.tokens.len(), 4);
        assert_eq!(
            out.token_logprobs,
            vec![None, Some(-0.1), Some(-0.2), Some(-0.3)]
        );
    }

    #[test]
    fn no_echo_generated_only_with_first_token_pad() {
        let out = build_completion_logprobs(
            &dec,
            false,
            &[1, 2, 3],
            &[],
            &[5, 6, 7],
            &[lp(6, -0.1), lp(7, -0.2)],
        );
        assert_eq!(out.tokens.len(), 3);
        assert_eq!(out.token_logprobs, vec![None, Some(-0.1), Some(-0.2)]);
    }

    #[test]
    fn scoring_only_prompt_len_entries() {
        let prompt = [10u32, 11, 12, 13];
        let lps = [lp(11, -1.0), lp(12, -2.0), lp(13, -3.0)];
        let out = build_completion_logprobs(&dec, true, &prompt, &lps, &[], &[]);
        assert_eq!(out.tokens.len(), prompt.len());
        assert_eq!(out.token_logprobs[0], None);
        assert!(out.token_logprobs[1..].iter().all(Option::is_some));
    }
}
