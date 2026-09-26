// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: [`process_seq_logits`], the per-sequence half of `process_decode_logits`:
//! dequantise one row of host logits, run the logits pipeline, sample.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::*;

/// 2026-09-25: Process logits for a single active sequence: dequant, adjust, sample, return token + optional logprobs.
#[allow(clippy::too_many_arguments)]
pub fn process_seq_logits(
    _model: &dyn Model,
    a: &mut ActiveSeq,
    buf: &[u8],
    i: usize,
    vocab_size: usize,
    elem_bytes: usize,
    logits_fp32: bool,
    ctx: &crate::scheduler::logit_processors::LogitsContext,
    adaptive_sampling: bool,
) -> (u32, Option<crate::api::TokenLogprobs>) {
    let slice = &buf[i * vocab_size * elem_bytes..(i + 1) * vocab_size * elem_bytes];
    // 2026-09-25: Reuse the run's scratch buffer (put back before both returns below).
    // `clear` + `extend` of exactly `vocab_size` items keeps its capacity and
    // leaves `len() == vocab_size`.
    let mut f32_logits = ctx.scratch.seq_f32.borrow_mut().split_off(0);
    f32_logits.clear();
    if logits_fp32 {
        f32_logits.extend((0..vocab_size).map(|j| {
            let off = j * 4;
            f32::from_le_bytes([slice[off], slice[off + 1], slice[off + 2], slice[off + 3]])
        }));
    } else {
        f32_logits.extend((0..vocab_size).map(|j| {
            let lo = slice[j * 2];
            let hi = slice[j * 2 + 1];
            bf16_to_f32(lo, hi)
        }));
    };

    // 2026-09-25: Raw-logits dump (`METRALE_DUMP_LOGITS_PATH=dir`): append the
    // dequantised row, before the pipeline and penalties, to `logits_seq.bin`.
    if let Some(dir) = ctx.tel.dumps().raw_logits_dir.as_deref() {
        use std::io::Write;
        let path = dir.join("logits_seq.bin");
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
        {
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(f32_logits.as_ptr() as *const u8, vocab_size * 4)
            };
            let _ = f.write_all(bytes);
        }
    }

    // 2026-09-25: Adaptive sampling (`adaptive_sampling`): update the zone, observe the
    // entropy of the dequantised logits (before the pipeline masks) and check
    // the greedy gate. When off, `greedy_gate` is false and the temperature is
    // the request's.
    let greedy_gate = if adaptive_sampling {
        a.adaptive.update_zone(
            a.tool_call_opened,
            a.inside_thinking,
            a.grammar_state.is_some(),
        );
        a.adaptive.observe_entropy(&f32_logits);
        a.adaptive.update_lz_ratio(&a.output_tokens);
        a.adaptive.should_use_greedy(&f32_logits)
    } else {
        false
    };
    let effective_temp = if adaptive_sampling {
        a.adaptive.effective_temperature()
    } else {
        a.temperature
    };

    // 2026-09-25: The greedy gate samples at temperature 0.
    let sampling_temp = if greedy_gate { 0.0 } else { effective_temp };
    // 2026-09-25: Advance seed per token for deterministic but varying randomness.
    let step_seed = a.seed.map(|s| s.wrapping_add(a.output_tokens.len() as u64));

    // 2026-09-25: This position's sampling, penalty and bias parameters, from
    // `penalty_params_for`, the builder the MTP verify and bootstrap paths
    // also use.
    let params = crate::scheduler::sample_step::penalty_params_for(
        a,
        crate::scheduler::sample_step::PositionKind::FinalDecode,
        sampling_temp,
        step_seed,
        a.logit_bias.clone(),
        ctx.watchdog.min_reasoning_floor,
    );

    // 2026-09-25: The logits pipeline, then penalties and bias applied in place
    // (`process_position_logits`). `Some(tok)` is a token to emit without
    // sampling: the `METRALE_FORCE_TEMP_ZERO` argmax or a forced token. The
    // call never advances the grammar matcher; `process_decode_logits` feeds
    // it the chosen token.
    if let Some(tok) = crate::scheduler::logit_processors::process_position_logits(
        &mut f32_logits,
        a,
        ctx,
        &params,
        crate::scheduler::sample_step::PositionKind::FinalDecode,
    ) {
        let logprobs = a
            .top_logprobs
            .map(|k| extract_logprobs_from_f32(&f32_logits, tok, k as usize));
        *ctx.scratch.seq_f32.borrow_mut() = std::mem::take(&mut f32_logits);
        return (tok, logprobs);
    }

    // 2026-09-25: The penalties and bias are already in `f32_logits`: sample with
    // neutral penalty parameters (no bias, no history) so they are not applied
    // twice.
    let f32_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(f32_logits.as_ptr() as *const u8, vocab_size * 4) };
    let sampler_shape = SamplingParams {
        temperature: params.temperature,
        top_k: params.top_k,
        top_p: params.top_p,
        top_n_sigma: params.top_n_sigma,
        min_p: params.min_p,
        logit_bias: Vec::new(),
        repetition_penalty: 1.0,
        repetition_penalty_window: 0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        lz_penalty: 0.0,
        dry_multiplier: 0.0,
        dry_base: params.dry_base,
        dry_allowed_length: params.dry_allowed_length,
        dry_sequence_breakers: Vec::new(),
        max_tokens: 0,
        stop_token_ids: Vec::new(),
        seed: params.seed,
    };
    let sampled = sample_with_params_history(f32_bytes, &sampler_shape, &[]);

    // 2026-09-25: Per-step logit dump (`METRALE_LOGIT_DUMP=file`): the masked and
    // penalised row, the bias applied and the sampled token.
    if let Some(sink) = ctx.tel.dumps().logits.as_ref() {
        super::logit_dump::record(
            sink,
            a.output_tokens.len(),
            a.inside_parameter_body,
            a.param_body_chars_emitted as usize,
            &f32_logits,
            &params.logit_bias,
            sampled,
        );
    }

    let logprobs = a
        .top_logprobs
        .map(|k| extract_logprobs_from_f32(&f32_logits, sampled, k as usize));
    *ctx.scratch.seq_f32.borrow_mut() = std::mem::take(&mut f32_logits);
    (sampled, logprobs)
}
