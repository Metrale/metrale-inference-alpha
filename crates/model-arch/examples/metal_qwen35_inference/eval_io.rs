// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Evaluation IO for the Metal Qwen3.5 inference example: per-step logit dumps
//! (`METRALE_LOGITS_OUT`), argmax-sequence dumps (`METRALE_DUMP_TOKENS`) and teacher-forced
//! token lists (`METRALE_FORCE_TOKENS_FILE`). `tests/metal_kv_kld_compare.py` compares two
//! logit dumps, for example from runs with different KV dtypes.
//!
//! Owner: model-arch examples (Metal Qwen3.5 driver).
//! Invariants: none beyond the types.

use anyhow::Result;

/// 2026-09-25: Open the per-step logits file when `METRALE_LOGITS_OUT` is set. The driver
/// appends each step's raw little-endian BF16 logits, so the file is `[n_steps, vocab]`.
pub fn logits_writer() -> Option<std::cell::RefCell<std::io::BufWriter<std::fs::File>>> {
    std::env::var("METRALE_LOGITS_OUT").ok().map(|p| {
        std::cell::RefCell::new(std::io::BufWriter::new(
            std::fs::File::create(p).expect("create METRALE_LOGITS_OUT"),
        ))
    })
}

/// 2026-09-25: Load the teacher-forcing list from `METRALE_FORCE_TOKENS_FILE`: token ids, one
/// per line; entry 0 replaces the first sampled token. Two runs fed the same list see the same
/// context at every position, so their logit dumps compare step by step.
pub fn forced_tokens() -> Option<Vec<u32>> {
    std::env::var("METRALE_FORCE_TOKENS_FILE").ok().map(|p| {
        std::fs::read_to_string(&p)
            .unwrap_or_else(|e| panic!("read METRALE_FORCE_TOKENS_FILE {p}: {e}"))
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| l.trim().parse().expect("token id"))
            .collect()
    })
}

/// 2026-09-25: Write the generated ids, one per line, when `METRALE_DUMP_TOKENS` is set; the
/// file is in the format `METRALE_FORCE_TOKENS_FILE` reads.
pub fn maybe_dump_tokens(generated_ids: &[u32]) -> Result<()> {
    if let Ok(path) = std::env::var("METRALE_DUMP_TOKENS") {
        let body: String = generated_ids.iter().map(|t| format!("{t}\n")).collect();
        std::fs::write(&path, body)?;
        println!("  (dumped {} token ids to {path})", generated_ids.len());
    }
    Ok(())
}
