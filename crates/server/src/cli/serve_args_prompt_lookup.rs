// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: The `--prompt-lookup-*` serve flags (flattened into `ServeSchedulingArgs`
//! after `--ngram-speculative`) and the settings `ServeArgs` derives from them.
//!
//! Owner: server CLI.
//! Invariants: the `///` text on the struct's fields is the `--help` output and
//! carries no date.

use clap::Args;

use super::serve_args::ServeArgs;

#[derive(Args, Debug, Clone, PartialEq)]
pub struct ServePromptLookupArgs {
    /// Prompt-lookup decoding (default: false): when a sequence's last
    /// `--prompt-lookup-ngram` tokens occurred earlier in its prompt or output,
    /// verify the tokens that followed that occurrence as this round's drafts,
    /// in place of the MTP drafter's chain; with no match the round is MTP's.
    /// Per sequence, inside batched verify, up to `--prompt-lookup-max-seqs`
    /// active sequences. Requires `--speculative`.
    ///
    /// Only verified tokens are emitted, so greedy output does not depend on the
    /// flag wherever verify rows are row-invariant (on Qwen3.6-35B-A3B-FP8 under
    /// `--exact-verify`). The copy window starts at 2 tokens (capped at
    /// `--prompt-lookup-max-drafts`), doubles after a fully accepted copy and
    /// halves after a broken one.
    /// Measured on GB10 at C1: code edits 1.50x, JSON rewrites 1.25x decode
    /// tok/s; prose and tool echoes 1-4% slower (wrong copies cost verify rows).
    #[arg(long, default_value_t = false, conflicts_with_all = ["dflash", "ngram_speculative", "self_speculative"])]
    pub prompt_lookup_decoding: bool,

    /// n-gram length a prompt-lookup match needs (default: 4). Longer matches
    /// propose less often and are wrong less often.
    #[arg(long, default_value_t = 4, value_parser = clap::value_parser!(u32).range(1..=64))]
    pub prompt_lookup_ngram: u32,

    /// Most tokens one prompt-lookup copy proposes, the copy window's ceiling
    /// (default: 3). 1..=16. The first `--prompt-lookup-max-seqs` verify-pool
    /// slots are sized to hold this many drafts (about 64 MB per slot per draft
    /// above the MTP chain on Qwen3.6-35B-A3B, 152 MB on Qwen3.8-27B). A lone
    /// sequence verifies a copy of up to this length in one pass; inside a
    /// batch a copy is cut to 3 drafts, the batched verify's widest row count.
    #[arg(long, default_value_t = 3, value_parser = clap::value_parser!(u32).range(1..=16))]
    pub prompt_lookup_max_drafts: u32,

    /// Tokens a prompt-lookup match must span, counted back from the end of the
    /// sequence (default: 4, at least `--prompt-lookup-ngram`). Longer matches
    /// copy less often on prose, where most short matches are coincidences.
    #[arg(long, default_value_t = 4, value_parser = clap::value_parser!(u32).range(1..=64))]
    pub prompt_lookup_min_match: u32,

    /// Most rounds a sequence skips copying after copies that matched nothing
    /// (default: 0, never). The skip doubles with each miss in a row, up to this
    /// cap, and clears on any accepted copied token.
    #[arg(long, default_value_t = 0, value_parser = clap::value_parser!(u32).range(0..=1024))]
    pub prompt_lookup_miss_backoff: u32,

    /// Widest batch that proposes prompt-lookup copies (default: 8). Wider
    /// batches run the MTP drafter only: a copy widens the verify, which costs
    /// most where the batch is compute-bound.
    #[arg(long, default_value_t = 8, value_parser = clap::value_parser!(u32).range(1..=128))]
    pub prompt_lookup_max_seqs: u32,
}

impl ServeArgs {
    /// 2026-10-02: The prompt-lookup settings, `None` without
    /// `--prompt-lookup-decoding`.
    pub fn prompt_lookup_config(
        &self,
    ) -> Option<metrale_speculative::prompt_lookup::PromptLookupConfig> {
        self.prompt_lookup.prompt_lookup_decoding.then(|| {
            metrale_speculative::prompt_lookup::PromptLookupConfig {
                ngram: self.prompt_lookup.prompt_lookup_ngram as usize,
                max_drafts: self.prompt_lookup.prompt_lookup_max_drafts as usize,
                max_seqs: self.prompt_lookup.prompt_lookup_max_seqs as usize,
                min_match: self.prompt_lookup.prompt_lookup_min_match as usize,
                miss_backoff: self.prompt_lookup.prompt_lookup_miss_backoff as usize,
            }
        })
    }

    /// 2026-10-02: Drafts the widest speculative verify can carry: the drafter's
    /// [`Self::resolved_num_drafts`], raised to `--prompt-lookup-max-drafts` when prompt
    /// lookup is on. It sizes the per-sequence verify rows the prefill budget reserves; the
    /// SSM pools take the copy depth from the published copy tier (`ssm_reserve::copy_tier`).
    pub fn verify_pool_drafts(&self) -> usize {
        let copy = self.prompt_lookup_config().map_or(0, |pl| pl.max_drafts);
        self.resolved_num_drafts().max(copy)
    }
}
