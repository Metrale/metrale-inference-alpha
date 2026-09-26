// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The warm-cache rule for a concurrency cell: whether the
//! server's snapshot pool can hold the cell's warmed prompts, and whether its
//! measured requests agree about being warm.
//!
//! Owner: bench (concurrency).
//! Invariants: with no warm-up rounds a cell is never cache-uncontrolled.

use super::RequestEvidence;

// 2026-09-26: `stats::make_prompt` puts the prompt's tag at byte zero of the
// message content, so a different prompt shares at most the chat-template
// prefix with a warmed one. 80% cached separates reuse of the warmed prompt
// from that.
pub(super) const WARM_CACHE_FLOOR: f64 = 0.8;

/// 2026-09-26: SSM snapshot slots one warm request holds through a cell: its
/// prompt-tail checkpoint and its finish leaf (`finish_leaf_snapshot` in
/// model-engine's `decode_checkpoint.rs`). The prefill-end leaf is skipped
/// when a tail checkpoint lies within two blocks of the prompt end
/// (`prefill_b::exact_leaf`).
const LIVE_SLOTS_PER_WARM_REQUEST: usize = 2;
/// 2026-09-26: Stale finish leaves each earlier cell of the sweep leaves behind
/// for a prompt it also ran. A finish leaf covers prompt plus generated text,
/// so a warm-up round and a measured round that generate different text leave
/// two. The snapshot pool's eviction is session-aware (`evict_lru` in the
/// cache crate's `radix_tree/snapshot.rs`).
///
/// Measured 2026-09-14 on a 32-slot pool, twice on one node: the conc 8 cell
/// after the 1/2/4 cells lost two, then three, of its eight tails. With one
/// stale leaf per earlier cell [`slots_needed`] gives 31, which fits; with two
/// it gives 38, which does not.
const STALE_LEAVES_PER_EARLIER_CELL: usize = 2;
/// 2026-09-26: Slots the pool must hold for a `conc`-way cell to keep every
/// warmed prompt through its measured round: each prompt's live slots, the
/// stale leaves of every earlier cell that ran the same prompt (`earlier` are
/// those cells' concurrencies; prompt `i` ran in each one wider than `i`),
/// and one spare per in-flight request, because a save that re-homes a prefix
/// takes a new slot before it frees the old one.
///
/// It errs toward "cold": a cell judged cold that hits warm anyway is still
/// measured (only the warm rule is skipped), while a cell judged warm that
/// eviction turns cold would fail the gate with nothing wrong in the engine.
pub(super) fn slots_needed(conc: usize, earlier: &[usize]) -> usize {
    (0..conc)
        .map(|i| {
            let stale_cells = earlier.iter().filter(|&&c| c > i).count();
            LIVE_SLOTS_PER_WARM_REQUEST + STALE_LEAVES_PER_EARLIER_CELL * stale_cells
        })
        .sum::<usize>()
        + conc
}
pub(super) const SSM_CACHE_SLOTS_KEY: &str = "ssm_cache_slots";

/// 2026-09-26: Whether the server's snapshot pool can hold every warmed prompt
/// of a `conc`-way cell. `slots` is the pool size from the serve overrides;
/// `None` (not stated) answers `true`, so the warm rule applies in full.
/// Strictly greater, because the last re-home still needs its spare slot.
pub(super) fn warm_cache_capable(conc: usize, slots: Option<usize>, earlier: &[usize]) -> bool {
    slots.is_none_or(|slots| slots > slots_needed(conc, earlier))
}

/// 2026-09-26: A cell's cache state is uncontrolled when warm-up was requested
/// and its measured requests disagree about it: some are warm (at least
/// `WARM_CACHE_FLOOR` of the prompt cached) and some are not, or a request
/// reported no prompt usage. A uniformly cold cell is controlled: below the
/// server's restore threshold (`--marconi-min-tokens`, default
/// `DEFAULT_MARCONI_MIN_TOKENS` = 256 in model-layers `mtp_carry.rs`) no
/// snapshot is restored, so every request reports 0 cached.
/// A mixture is two measurements reported as one.
///
/// The record still carries `min_cached_prompt_pct` and
/// `min_cached_prompt_tokens`, and the evidence line prints each request's
/// cached count.
pub(super) fn cache_is_uncontrolled(requests: &[RequestEvidence], warmup: usize) -> bool {
    if warmup == 0 {
        return false;
    }
    if requests.iter().any(|request| request.prompt_tokens == 0) {
        return true;
    }
    let warm = |request: &RequestEvidence| {
        (request.cached_prompt_tokens as f64) >= WARM_CACHE_FLOOR * request.prompt_tokens as f64
    };
    requests.iter().any(&warm) != requests.iter().all(&warm)
}
