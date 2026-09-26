// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Integration tests that load real model weights on a GPU and
//! run inference, plus a teardown leak check. Every test is `#[ignore]`d and
//! needs a GPU and a HuggingFace snapshot directory, named by
//! `METRALE_INTEGRATION_MODEL_DIR`:
//!
//!   METRALE_INTEGRATION_MODEL_DIR=/path/to/snapshot \
//!     cargo test -p metrale-server --release -- --ignored
//!
//! When the directory does not exist, each test prints a skip message and
//! passes.
//!
//! Owner: server tests.
//! Invariants: none beyond the types.

use anyhow::Result;
use std::path::Path;

#[path = "integration/helpers.rs"]
mod helpers;
use helpers::{chat_tokenizer, free_device_memory_bytes, generate, model_dir_path, setup_model};

/// 2026-09-26: Smoke test: load the model, decode one BOS step, then 200 more
/// timed decode steps.
#[test]
#[ignore]
fn smoke_test_single_decode() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .try_init()
        .ok();

    let model_dir_buf = model_dir_path();
    if !model_dir_buf.exists() {
        eprintln!(
            "SKIP: model directory not found: {}\n      \
             Set METRALE_INTEGRATION_MODEL_DIR to a HuggingFace snapshot path.",
            model_dir_buf.display()
        );
        return Ok(());
    }
    let model_dir: &Path = model_dir_buf.as_path();

    let (model, config) = setup_model(model_dir)?;
    tracing::info!("Model built successfully");

    let mut seq = model.alloc_sequence()?;
    let bos_token = config.bos_token_id;
    let logits_ptr = model.decode(bos_token, &mut seq, 0)?;
    assert!(!logits_ptr.is_null(), "Logits pointer should not be null");
    assert_eq!(seq.seq_len, 1);

    let vocab_size = model.vocab_size();
    let best_idx = model.argmax_on_device(logits_ptr, 0)?;
    tracing::info!("GPU argmax token: {}, vocab_size: {}", best_idx, vocab_size);
    assert!((best_idx as usize) < vocab_size);

    // 2026-09-26: The best logit, read back as bf16, is finite.
    let mut logits_host = vec![0u8; vocab_size * 2];
    model.copy_logits_to_host(logits_ptr, &mut logits_host)?;
    let idx = best_idx as usize;
    let lo = logits_host[idx * 2];
    let hi = logits_host[idx * 2 + 1];
    let best_val = f32::from_bits(((lo as u32) | ((hi as u32) << 8)) << 16);
    tracing::info!("Best logit value: {:.4}", best_val);
    assert!(best_val.is_finite());

    let num_steps = 200;
    let mut tokens = vec![best_idx];
    let mut step_times = Vec::with_capacity(num_steps);
    for step in 0..num_steps {
        let last_token = *tokens.last().unwrap();
        let t0 = std::time::Instant::now();
        let logits_ptr = model.decode(last_token, &mut seq, 0)?;
        let token = model.argmax_on_device(logits_ptr, 0)?;
        let dt = t0.elapsed();
        step_times.push(dt);
        tokens.push(token);
        if step < 10 || step % 50 == 49 {
            tracing::info!(
                "Step {}: token {} (seq_len={}, {:.1}ms)",
                step + 1,
                token,
                seq.seq_len,
                dt.as_secs_f64() * 1000.0,
            );
        }
    }
    // 2026-09-26: The first timed step is reported apart, as `capture`.
    let replay_times: Vec<f64> = step_times[1..]
        .iter()
        .map(|t| t.as_secs_f64() * 1000.0)
        .collect();
    let avg_ms = replay_times.iter().sum::<f64>() / replay_times.len() as f64;
    let min_ms = replay_times.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_ms = replay_times.iter().cloned().fold(0.0f64, f64::max);
    let mut sorted = replay_times.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50_ms = sorted[sorted.len() / 2];
    let p95_ms = sorted[(sorted.len() as f64 * 0.95) as usize];
    let capture_ms = step_times[0].as_secs_f64() * 1000.0;
    let total_elapsed: f64 = step_times.iter().map(|t| t.as_secs_f64()).sum();
    let tok_per_sec = num_steps as f64 / total_elapsed;
    tracing::info!(
        "Generated {} tokens in {:.2}s ({:.1} tok/s)",
        num_steps,
        total_elapsed,
        tok_per_sec,
    );
    tracing::info!(
        "Step timing: capture={:.1}ms, replay avg={:.1}ms p50={:.1}ms p95={:.1}ms min={:.1}ms max={:.1}ms",
        capture_ms,
        avg_ms,
        p50_ms,
        p95_ms,
        min_ms,
        max_ms,
    );
    assert_eq!(seq.seq_len, num_steps + 1);

    tracing::info!("Integration smoke test passed!");
    Ok(())
}

/// 2026-09-26: Encode "What is the capital of France?" with the chat
/// template and check that the answer contains "Paris".
#[test]
#[ignore]
fn coherence_test_capital_of_france() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .try_init()
        .ok();

    let model_dir_buf = model_dir_path();
    if !model_dir_buf.exists() {
        eprintln!(
            "SKIP: model directory not found: {}\n      \
             Set METRALE_INTEGRATION_MODEL_DIR to a HuggingFace snapshot path.",
            model_dir_buf.display()
        );
        return Ok(());
    }
    let model_dir: &Path = model_dir_buf.as_path();

    let (model, config) = setup_model(model_dir)?;

    let tokenizer = chat_tokenizer(model_dir, &config)?;

    let messages = vec![(
        "user".to_string(),
        "What is the capital of France?".to_string(),
    )];
    let prompt_tokens = tokenizer.apply_chat_template(&messages, false, &[])?;
    tracing::info!(
        "Prompt: {} tokens: {:?}",
        prompt_tokens.len(),
        prompt_tokens
    );

    let (gen_tokens, tok_per_sec) = generate(model.as_ref(), &config, &prompt_tokens, 200)?;
    let output_text = tokenizer.decode(&gen_tokens)?;
    tracing::info!(
        "Generated {} tokens ({:.1} tok/s):\n  {}",
        gen_tokens.len(),
        tok_per_sec,
        output_text,
    );

    let output_lower = output_text.to_lowercase();
    assert!(
        output_lower.contains("paris"),
        "Expected output to contain 'Paris', got: {output_text}"
    );

    // 2026-09-26: An answer that ends in EOS is not checked for repetition;
    // one that does not must hold at least 10 distinct token ids.
    let unique_tokens: std::collections::HashSet<u32> = gen_tokens.iter().copied().collect();
    let hit_eos = gen_tokens.last().copied() == Some(config.eos_token_id);
    if !hit_eos {
        assert!(
            unique_tokens.len() >= 10,
            "Output is degenerate (only {} unique tokens, no EOS): {:?}",
            unique_tokens.len(),
            gen_tokens,
        );
    }

    tracing::info!("Coherence test PASSED: output contains 'Paris'");
    Ok(())
}

/// 2026-09-26: `generate_streaming` hands the callback exactly the tokens it
/// returns, and the answer contains "Paris".
#[test]
#[ignore]
fn streaming_coherence_test() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .try_init()
        .ok();

    let model_dir_buf = model_dir_path();
    if !model_dir_buf.exists() {
        eprintln!(
            "SKIP: model directory not found: {}\n      \
             Set METRALE_INTEGRATION_MODEL_DIR to a HuggingFace snapshot path.",
            model_dir_buf.display()
        );
        return Ok(());
    }
    let model_dir: &Path = model_dir_buf.as_path();

    let (model, config) = setup_model(model_dir)?;

    let tokenizer = chat_tokenizer(model_dir, &config)?;

    let messages = vec![(
        "user".to_string(),
        "What is the capital of France?".to_string(),
    )];
    let prompt_tokens = tokenizer.apply_chat_template(&messages, false, &[])?;
    tracing::info!("Prompt: {} tokens", prompt_tokens.len());

    let params = metrale_sampling::SamplingParams {
        stop_token_ids: vec![config.eos_token_id],
        ..metrale_sampling::SamplingParams::greedy(200)
    };

    let mut streamed_tokens = Vec::new();
    let start = std::time::Instant::now();
    let result = metrale_model_engine::engine::generate_streaming(
        model.as_ref(),
        &prompt_tokens,
        &params,
        |token| {
            streamed_tokens.push(token);
        },
    )?;
    let elapsed = start.elapsed();

    let output_text = tokenizer.decode(&result.output_tokens)?;
    let tok_per_sec = result.output_tokens.len() as f64 / elapsed.as_secs_f64();
    tracing::info!(
        "Streaming: {} tokens in {:.2}s ({:.1} tok/s):\n  {}",
        result.output_tokens.len(),
        elapsed.as_secs_f64(),
        tok_per_sec,
        output_text,
    );

    assert_eq!(
        streamed_tokens, result.output_tokens,
        "Streamed tokens should match final output tokens"
    );

    let output_lower = output_text.to_lowercase();
    assert!(
        output_lower.contains("paris"),
        "Expected output to contain 'Paris', got: {output_text}"
    );

    tracing::info!(
        "Streaming coherence test PASSED: {} tokens streamed, output contains 'Paris'",
        streamed_tokens.len()
    );
    Ok(())
}

/// 2026-09-26: `generate_speculative` with 2 drafts on a model that has a
/// proposer; the answer contains "Paris".
#[test]
#[ignore]
fn speculative_decode_coherence() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .try_init()
        .ok();

    let model_dir_buf = model_dir_path();
    if !model_dir_buf.exists() {
        eprintln!(
            "SKIP: model directory not found: {}\n      \
             Set METRALE_INTEGRATION_MODEL_DIR to a HuggingFace snapshot path.",
            model_dir_buf.display()
        );
        return Ok(());
    }
    let model_dir: &Path = model_dir_buf.as_path();

    let (model, config) = setup_model(model_dir)?;
    assert!(model.has_proposer(), "MTP proposer should be enabled");

    let tokenizer = chat_tokenizer(model_dir, &config)?;

    let messages = vec![(
        "user".to_string(),
        "What is the capital of France?".to_string(),
    )];
    let prompt_tokens = tokenizer.apply_chat_template(&messages, false, &[])?;
    tracing::info!("Prompt: {} tokens", prompt_tokens.len());

    let params = metrale_sampling::SamplingParams {
        stop_token_ids: vec![config.eos_token_id],
        ..metrale_sampling::SamplingParams::greedy(200)
    };

    let start = std::time::Instant::now();
    let result = metrale_model_engine::engine::generate_speculative(
        model.as_ref(),
        &prompt_tokens,
        &params,
        2,
    )?;
    let elapsed = start.elapsed();

    let output_text = tokenizer.decode(&result.output_tokens)?;
    let tok_per_sec = result.output_tokens.len() as f64 / elapsed.as_secs_f64();
    tracing::info!(
        "Speculative: {} tokens in {:.2}s ({:.1} tok/s), reason={}:\n  {}",
        result.output_tokens.len(),
        elapsed.as_secs_f64(),
        tok_per_sec,
        result.finish_reason,
        output_text,
    );

    let output_lower = output_text.to_lowercase();
    assert!(
        output_lower.contains("paris"),
        "Expected output to contain 'Paris', got: {output_text}"
    );

    tracing::info!("Speculative decode coherence test PASSED");
    Ok(())
}

/// 2026-09-26: Prompt-token logprobs collected during prefill, as
/// `/v1/completions` with `echo` requests them (`api/completions.rs`).
///
/// Checks on a real model:
/// 1. there are prompt_len - 1 entries: position i scores token i+1, and the
///    last position, which would score the first generated token, is left
///    out;
/// 2. every logprob is finite and <= 0;
/// 3. each `token_id` is the next prompt token;
/// 4. the top-k alternatives are sorted descending, and the target's logprob
///    does not exceed the top-1 alternative;
/// 5. a control sequence without `collect_prompt_logprobs` collects nothing.
#[test]
#[ignore]
fn prompt_logprobs_collection_during_prefill() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .try_init()
        .ok();
    let model_dir_buf = model_dir_path();
    if !model_dir_buf.exists() {
        eprintln!(
            "SKIP: model directory not found: {}\n      \
             Set METRALE_INTEGRATION_MODEL_DIR to a HuggingFace snapshot path.",
            model_dir_buf.display()
        );
        return Ok(());
    }
    let model_dir: &Path = model_dir_buf.as_path();
    let (model, config) = setup_model(model_dir)?;
    let tokenizer = chat_tokenizer(model_dir, &config)?;

    let prompt = "The capital of France is Paris. The capital of Germany is";
    let prompt_tokens = tokenizer.encode(prompt)?;
    let n = prompt_tokens.len();
    assert!(n >= 4, "prompt too short to exercise scoring");

    // 2026-09-26: The collecting sequence asks for 2 alternatives.
    let mut seq = model.alloc_sequence()?;
    seq.collect_prompt_logprobs = Some(2);
    let _ = model.prefill_chunk(&prompt_tokens, &mut seq, 0, n, true, 0)?;

    assert_eq!(
        seq.prompt_logprobs.len(),
        n - 1,
        "one entry per prompt position scoring the NEXT prompt token"
    );
    for (i, lp) in seq.prompt_logprobs.iter().enumerate() {
        assert_eq!(lp.token_id, prompt_tokens[i + 1], "target at position {i}");
        assert!(lp.logprob.is_finite(), "logprob finite at {i}");
        assert!(lp.logprob <= 0.0, "logprob <= 0 at {i}: {}", lp.logprob);
        assert_eq!(lp.top.len(), 2, "top-k size at {i}");
        assert!(lp.top[0].1 >= lp.top[1].1, "top sorted desc at {i}");
        assert!(
            lp.logprob <= lp.top[0].1 + 1e-4,
            "target logprob cannot exceed top-1 at {i}"
        );
    }
    // 2026-09-26: The prompt's summed log-likelihood is negative and above
    // -200.
    let total: f32 = seq.prompt_logprobs.iter().map(|l| l.logprob).sum();
    assert!(total < 0.0 && total > -200.0, "plausible total ll: {total}");
    model.free_sequence(&mut seq)?;

    // 2026-09-26: Control: without the flag nothing is collected.
    let mut seq2 = model.alloc_sequence()?;
    let _ = model.prefill_chunk(&prompt_tokens, &mut seq2, 0, n, true, 0)?;
    assert!(
        seq2.prompt_logprobs.is_empty(),
        "no collection without the flag"
    );
    model.free_sequence(&mut seq2)?;
    Ok(())
}

/// 2026-09-26: Whether a real load followed by `teardown()` gives the device
/// memory back, over several load/teardown cycles (3 unless
/// `METRALE_TEARDOWN_CYCLES` says otherwise): a per-cycle leak shows only once
/// it accumulates past allocator noise.
///
/// Run it deliberately:
/// ```text
/// METRALE_INTEGRATION_MODEL_DIR=<snapshot> \
///   cargo test -p metrale-server --test integration teardown_returns -- --ignored --nocapture
/// ```
/// Before trusting any self-relative number, check nothing else is on the GPU:
/// `nvidia-smi --query-compute-apps=pid,used_memory --format=csv`.
#[test]
#[ignore]
fn teardown_returns_the_vram_it_took() -> Result<()> {
    let model_dir = model_dir_path();
    if !model_dir.exists() {
        eprintln!("skipping: {} does not exist", model_dir.display());
        return Ok(());
    }

    let cycles: usize = std::env::var("METRALE_TEARDOWN_CYCLES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);

    let mut readings: Vec<usize> = Vec::new();
    for cycle in 1..=cycles {
        let (mut model, _config) = setup_model(&model_dir)?;
        model.teardown()?;
        drop(model);

        let free = free_device_memory_bytes()?;
        eprintln!(
            "cycle {cycle}/{cycles}: {:.2} GB free after teardown",
            free as f64 / (1024.0 * 1024.0 * 1024.0)
        );
        readings.push(free);
    }

    let gb = |b: usize| b as f64 / (1024.0 * 1024.0 * 1024.0);
    eprintln!(
        "readings (GB): {:?}",
        readings
            .iter()
            .map(|b| (gb(*b) * 100.0).round() / 100.0)
            .collect::<Vec<_>>()
    );

    // 2026-09-26: A leak costs the same every cycle, while memory the driver
    // retains and reuses stops costing after the first cycles. So the last
    // step between readings must be at most half the first step.
    let first_step = readings
        .first()
        .zip(readings.get(1))
        .map(|(a, b)| a.saturating_sub(*b))
        .unwrap_or(0);
    let last_step = readings
        .iter()
        .rev()
        .nth(1)
        .zip(readings.last())
        .map(|(a, b)| a.saturating_sub(*b))
        .unwrap_or(0);
    eprintln!(
        "first step {:.2} GB, last step {:.2} GB",
        gb(first_step),
        gb(last_step)
    );
    assert!(
        last_step * 2 <= first_step.max(1),
        "memory use is not plateauing: the last cycle still cost {:.2} GB \
         against the first cycle's {:.2} GB. That is a per-cycle LEAK — a \
         device-memory owner is not registered for teardown.",
        gb(last_step),
        gb(first_step)
    );
    Ok(())
}
