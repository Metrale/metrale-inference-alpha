// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: Prompt-lookup decoding: copy proposals from a sequence's own
//! history (prompt and generated tokens). An index maps every n-gram of the
//! history to the position just after its latest occurrence, so a proposal
//! costs one hash lookup however long the history is. The proposal is the
//! continuation of that occurrence, up to a window that doubles after a fully
//! accepted copy and halves after a broken one. Host code only.
//!
//! Owner: speculative.
//! Invariants:
//! - A proposal is only ever a slice of the history it was asked about: the
//!   `n` tokens before its start equal the history's last `n` tokens, it is
//!   non-empty, and it never reaches past the history's end.
//! - The index never holds the history's own final n-gram (it has no
//!   continuation yet), so a proposal never points at the suffix itself.
//! - A stale entry (left by a rewind of the history) can never produce a
//!   proposal: every hit is re-checked token by token before it is used.
//! - [`CopyWindow::current`] stays within `[min, max]`.

use std::collections::HashMap;

/// 2026-10-02: Per-sequence n-gram index over a token history.
#[derive(Debug, Clone)]
pub struct PromptLookupIndex {
    /// 2026-10-02: n-gram length that must match before a copy is proposed.
    n: usize,
    /// 2026-10-02: n-gram key -> position of the token after its latest
    /// indexed occurrence.
    latest: HashMap<u64, u32>,
    /// 2026-10-02: History length up to which n-grams are indexed: every
    /// n-gram ending before `indexed_len - 1` is in `latest`.
    indexed_len: usize,
}

/// 2026-10-02: Key of an n-gram. Collisions are harmless (every hit is
/// re-checked against the history); the mix only keeps them rare.
fn ngram_key(gram: &[u32]) -> u64 {
    gram.iter().fold(0xcbf2_9ce4_8422_2325_u64, |h, &t| {
        (h ^ u64::from(t))
            .wrapping_mul(0x0000_0100_0000_01b3)
            .rotate_left(29)
    })
}

impl PromptLookupIndex {
    /// 2026-10-02: An empty index matching n-grams of length `n` (`n >= 1`).
    pub fn new(n: usize) -> Self {
        assert!(n >= 1, "prompt-lookup n-gram length must be at least 1");
        Self {
            n,
            latest: HashMap::new(),
            indexed_len: 0,
        }
    }

    /// 2026-10-02: The n-gram length this index matches.
    pub fn ngram_len(&self) -> usize {
        self.n
    }

    /// 2026-10-02: Indexes every n-gram of `history` that has a following
    /// token and is not indexed yet. Amortised O(n) per new token. A history
    /// shorter than the indexed length (a rewind) moves the mark back; the
    /// entries past it are left in place and rejected on use.
    pub fn observe(&mut self, history: &[u32]) {
        if history.len() < self.indexed_len {
            self.indexed_len = history.len();
        }
        // 2026-10-02: n-gram ending at e-1 (tokens [e-n, e)) is indexable when
        // a token follows it, i.e. e < len.
        let first_end = self.indexed_len.max(self.n);
        for end in first_end..history.len() {
            let key = ngram_key(&history[end - self.n..end]);
            self.latest.insert(key, end as u32);
        }
        self.indexed_len = self.indexed_len.max(history.len());
    }

    /// 2026-10-02: The continuation of the latest earlier occurrence of
    /// `history`'s final n-gram, at most `max_len` tokens; `None` when there
    /// is no earlier occurrence, `max_len` is 0, or the history is too short.
    /// Call [`observe`](Self::observe) with the same history first.
    pub fn propose<'h>(&self, history: &'h [u32], max_len: usize) -> Option<&'h [u32]> {
        let start = self.continuation_start(history)?;
        if max_len == 0 {
            return None;
        }
        let end = history.len().min(start.saturating_add(max_len));
        Some(&history[start..end])
    }

    /// 2026-10-02: Where the continuation [`propose`](Self::propose) copies
    /// from starts: the position after the latest earlier occurrence of the
    /// history's final n-gram, re-checked token by token.
    pub fn continuation_start(&self, history: &[u32]) -> Option<usize> {
        let len = history.len();
        if len <= self.n {
            return None;
        }
        let suffix = &history[len - self.n..];
        let start = *self.latest.get(&ngram_key(suffix))? as usize;
        if start < self.n || start >= len || history[start - self.n..start] != *suffix {
            return None;
        }
        Some(start)
    }

    /// 2026-10-02: How many tokens before `start` equal the tokens before the
    /// history's end, counted back from both, up to `cap`: the length of the
    /// match a copy starting at `start` rests on. At least `n` for a start
    /// [`propose`](Self::propose) returned.
    pub fn match_len(history: &[u32], start: usize, cap: usize) -> usize {
        let len = history.len();
        (1..=cap.min(start))
            .take_while(|&k| history[start - k] == history[len - k])
            .count()
    }
}

/// 2026-10-02: Copy window: how many tokens a copy may propose. It doubles
/// after a copy is accepted in full and halves after one is cut short.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CopyWindow {
    current: usize,
    min: usize,
    max: usize,
}

impl CopyWindow {
    /// 2026-10-02: A window starting at `start`, kept within `[min, max]`
    /// (`1 <= min <= max`).
    pub fn new(start: usize, min: usize, max: usize) -> Self {
        assert!(
            min >= 1 && min <= max,
            "copy window bounds must satisfy 1 <= min <= max"
        );
        Self {
            current: start.clamp(min, max),
            min,
            max,
        }
    }

    /// 2026-10-02: Tokens the next copy may propose.
    pub fn current(&self) -> usize {
        self.current
    }

    /// 2026-10-02: Updates the window after a copy of `proposed` tokens of
    /// which the first `accepted` matched the target.
    pub fn record(&mut self, proposed: usize, accepted: usize) {
        if proposed == 0 {
            return;
        }
        self.current = if accepted >= proposed {
            self.current.saturating_mul(2).min(self.max)
        } else {
            (self.current / 2).max(self.min)
        };
    }
}

/// 2026-10-02: Where a sequence's copy window starts (capped at
/// `--prompt-lookup-max-drafts`). A short first copy bounds the verify rows an
/// untested match can waste; a fully accepted copy doubles the window.
pub const WINDOW_START: usize = 2;

/// 2026-10-02: Rounds to skip after `misses` copies in a row that matched
/// nothing: `2^(misses-1)`, capped at `cap` (0 for no backoff).
pub fn backoff_rounds(misses: u32, cap: usize) -> usize {
    if cap == 0 || misses == 0 {
        return 0;
    }
    1usize
        .checked_shl(misses - 1)
        .unwrap_or(usize::MAX)
        .min(cap)
}

/// 2026-10-02: Serve-wide prompt-lookup settings (`--prompt-lookup-*`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PromptLookupConfig {
    /// 2026-10-02: n-gram length that must match (`--prompt-lookup-ngram`).
    pub ngram: usize,
    /// 2026-10-02: Most tokens one copy proposes (`--prompt-lookup-max-drafts`);
    /// the copy window's ceiling.
    pub max_drafts: usize,
    /// 2026-10-02: Widest batch that proposes copies (`--prompt-lookup-max-seqs`).
    pub max_seqs: usize,
    /// 2026-10-02: Tokens a copy's match must span, counted back from the
    /// history's end (`--prompt-lookup-min-match`, at least `ngram`).
    pub min_match: usize,
    /// 2026-10-02: Most rounds a sequence skips copying after copies that
    /// matched nothing (`--prompt-lookup-miss-backoff`); 0 never skips.
    pub miss_backoff: usize,
}

/// 2026-10-02: One sequence's prompt-lookup state: its index, its copy window,
/// and the length of the copy now awaiting verification (0 when the pending
/// drafts are the model drafter's).
#[derive(Debug, Clone)]
pub struct PromptLookupSeq {
    index: PromptLookupIndex,
    window: CopyWindow,
    in_flight: usize,
    min_match: usize,
    miss_backoff: usize,
    /// 2026-10-02: Copies in a row that matched nothing.
    misses: u32,
    /// 2026-10-02: Rounds left to skip copying.
    cooldown: usize,
}

impl PromptLookupSeq {
    /// 2026-10-02: Fresh state. The window starts at [`WINDOW_START`] (or the
    /// ceiling, if lower), so an untested sequence's first copies are short and
    /// a wrong one costs few verify rows.
    pub fn new(cfg: &PromptLookupConfig) -> Self {
        Self {
            index: PromptLookupIndex::new(cfg.ngram),
            window: CopyWindow::new(WINDOW_START, 1, cfg.max_drafts),
            in_flight: 0,
            min_match: cfg.min_match.max(cfg.ngram),
            miss_backoff: cfg.miss_backoff,
            misses: 0,
            cooldown: 0,
        }
    }

    /// 2026-10-02: Proposes a copy of at most `max_len` tokens for `history`
    /// (prompt, generated tokens and the token about to be verified), and marks
    /// it in flight. `None` (and nothing in flight) when there is no match or
    /// `max_len` is 0. The caller bounds `max_len`, normally by
    /// [`window`](Self::window).
    ///
    /// No copy is proposed while a miss backoff is cooling down (each call
    /// counts one round), or when the match spans fewer than `min_match`
    /// tokens.
    pub fn propose(&mut self, history: &[u32], max_len: usize) -> Option<Vec<u32>> {
        self.in_flight = 0;
        self.index.observe(history);
        if self.cooldown > 0 {
            self.cooldown -= 1;
            return None;
        }
        let start = self.index.continuation_start(history)?;
        if max_len == 0
            || PromptLookupIndex::match_len(history, start, self.min_match) < self.min_match
        {
            return None;
        }
        let copy = history[start..history.len().min(start + max_len)].to_vec();
        self.in_flight = copy.len();
        Some(copy)
    }

    /// 2026-10-02: Length of the copy awaiting verification; 0 when none.
    pub fn in_flight(&self) -> usize {
        self.in_flight
    }

    /// 2026-10-02: Closes the in-flight copy: `accepted` of its tokens matched.
    /// Returns the copy's length (0 when none was in flight, which changes
    /// nothing).
    ///
    /// A copy that matched nothing starts (or doubles) a backoff of up to
    /// `miss_backoff` rounds; any accepted token clears it.
    pub fn settle(&mut self, accepted: usize) -> usize {
        let proposed = std::mem::take(&mut self.in_flight);
        if proposed == 0 {
            return 0;
        }
        self.window.record(proposed, accepted.min(proposed));
        if accepted == 0 {
            self.misses = self.misses.saturating_add(1);
            self.cooldown = backoff_rounds(self.misses, self.miss_backoff);
        } else {
            self.misses = 0;
        }
        proposed
    }

    /// 2026-10-02: Drops an in-flight copy without judging it (its drafts were
    /// discarded unverified).
    pub fn abandon(&mut self) {
        self.in_flight = 0;
    }

    /// 2026-10-02: The current copy window.
    pub fn window(&self) -> usize {
        self.window.current()
    }
}

#[cfg(test)]
#[path = "prompt_lookup_tests.rs"]
mod tests;
