// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The pre-load reserve refusal: the error text, a suggested
//! flag change, and the decode-ring formula.
//!
//! Owner: server startup (`met serve`).
//! Invariants: none beyond the types.

use metrale_config::ModelConfig;

use crate::cli;

use super::decode_ring;

/// 2026-09-26: The refusal's inputs that `args` and `config` do not give.
pub(super) struct Refusal {
    /// 2026-09-26: `inference_reserve + buffer_arena_bytes`.
    pub(super) total_reserve: usize,
    pub(super) free_mem: usize,
    /// 2026-09-26: The reserve terms that do not scale with `--max-seq-len`:
    /// SSM pools, h stage, both snapshot regions and CUDA headroom. The
    /// suggested `--max-seq-len` is priced from half of what `free_mem`
    /// leaves after them.
    pub(super) seq_len_independent: usize,
    /// 2026-09-26: The ring depth requested before the fit
    /// (`decode_ring::requested_slots`: flags and environment), and the depth
    /// reserved after the fit.
    pub(super) ring_requested: usize,
    pub(super) ring_slots: usize,
    pub(super) per_seq_blob: usize,
    /// 2026-09-26: `published_decode_ring_slots().is_some()`, read by the
    /// caller after the fit. Passed in so the text depends only on the inputs
    /// and tests need not write the process-global cell.
    pub(super) ring_pinned: bool,
}

/// 2026-09-26: Build the refusal error from already-computed bytes.
pub(super) fn reserve_refusal(
    args: &cli::ServeArgs,
    config: &ModelConfig,
    r: Refusal,
) -> anyhow::Error {
    let need_gb = r.total_reserve as f64 / (1024.0 * 1024.0 * 1024.0);
    let free_gb = r.free_mem as f64 / (1024.0 * 1024.0 * 1024.0);
    let budget_for_seq_term = r.free_mem.saturating_sub(r.seq_len_independent) / 2;
    let per_tok_bytes = {
        let key_dim = config.linear_num_key_heads * config.linear_key_head_dim;
        let value_dim = config.linear_num_value_heads * config.linear_value_head_dim;
        let nv = config.linear_num_value_heads;
        let conv_dim = key_dim * 2 + value_dim;
        if conv_dim > 0 && config.num_ssm_layers() > 0 {
            (conv_dim * 2) + (nv * 2 * 4) + (value_dim * 2) + (value_dim * 2)
        } else {
            0
        }
    };
    let suggested = budget_for_seq_term
        .checked_div(per_tok_bytes)
        .map(|q| q.max(2048))
        .unwrap_or(0);
    let hint = if suggested > 0 && suggested < args.max_seq_len {
        format!(
            " Try --max-seq-len {} (or lower --max-batch-size / --num-drafts).",
            suggested
        )
    } else if args.max_batch_size > 1 {
        " Reduce --max-batch-size.".to_string()
    } else {
        " Use a smaller model or a GPU with more memory.".to_string()
    };
    anyhow::anyhow!(
        "Preflight failed: inference buffers alone need {:.2} GB but only {:.2} GB is free on the GPU \
         (before weights load). SSM pool + GDN chunked prefill scales with --max-seq-len={} × --max-batch-size={}.{}{}",
        need_gb,
        free_gb,
        args.max_seq_len,
        args.max_batch_size,
        hint,
        ring_note(args, &r),
    )
}

/// 2026-09-26: The ring formula at the reserved depth and why the ring was
/// not lowered further; empty when no ring was requested.
fn ring_note(args: &cli::ServeArgs, r: &Refusal) -> String {
    if r.ring_requested == 0 {
        return String::new();
    }
    format!(
        " {} — {}.",
        decode_ring::formula(r.ring_slots, args.max_batch_size, r.per_seq_blob),
        if r.ring_pinned {
            "an explicit --ssm-decode-ring-slots pins the depth, so it was not shrunk"
        } else {
            "even 0 ring slots does not fit, so shrinking it cannot rescue this boot"
        },
    )
}

#[cfg(test)]
#[path = "refusal_tests.rs"]
mod tests;
