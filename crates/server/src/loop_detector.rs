// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Repetition detection over the recent assistant turns of a
//! conversation, for `api/chat/loop_detect.rs`: a shingle-similarity verdict
//! ([`detect`]) and the tool-call outcome checks that decide what the caller
//! does with it.
//!
//! Owner: server chat API.
//! Invariants: a `Hint` or `Suppress` verdict has `run_length >= 2` turns, and
//! `detect` returns `LoopState::None` for fewer than 3 non-empty signatures.
//!
//! Each assistant message becomes a [`Signature`]: sets of hashed
//! `SHINGLE_ORDER`-token shingles over its text, over its tool-call names and
//! arguments, and over both together. Only text and argument strings are
//! read; nothing is parsed.

use std::collections::HashSet;
use std::hash::{Hash, Hasher};

/// 2026-09-26: Tokens per shingle.
const SHINGLE_ORDER: usize = 4;

/// 2026-09-26: Most non-empty signatures `detect` compares, newest first.
const RECENT_WINDOW: usize = 5;

/// 2026-09-26: A channel with fewer tokens than this gets an empty set.
const MIN_CHANNEL_TOKENS: usize = 8;

/// 2026-09-26: Jaccard at or above which a pair counts as high.
const HIGH_SIMILARITY: f64 = 0.65;

/// 2026-09-26: Jaccard at or above which a pair extends a run.
const MODERATE_SIMILARITY: f64 = 0.50;

/// 2026-09-26: Verdict of [`detect`]. A run is the leading streak of adjacent
/// newest-first pairs at or above `MODERATE_SIMILARITY` in one channel;
/// `run_length` counts its turns (pairs + 1), and `score` is its highest pair
/// similarity.
#[derive(Debug, Clone, PartialEq)]
pub enum LoopState {
    None,
    /// 2026-09-26: A run of 2 turns whose pair is at or above
    /// `HIGH_SIMILARITY`, or of 3 turns with no pair that high.
    /// `api/chat/loop_detect.rs` only turns it into a `<tool_call>` logit
    /// bias.
    Hint {
        score: f64,
        run_length: usize,
        /// 2026-09-26: The run's channel, logged and used as a metric label.
        channel: SimilarityChannel,
    },
    /// 2026-09-26: A run of 4 or more turns, or of 3 turns with a pair at or
    /// above `HIGH_SIMILARITY`. `api/chat/loop_detect.rs` also hard-masks
    /// `<tool_call>` for the turn unless the repeated calls all failed, their
    /// results still change, or `METRALE_LOOP_NO_SUPPRESS=1`.
    Suppress {
        score: f64,
        run_length: usize,
        channel: SimilarityChannel,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimilarityChannel {
    Text,
    Tools,
    Combined,
}

impl SimilarityChannel {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Tools => "tools",
            Self::Combined => "combined",
        }
    }
}

/// 2026-09-26: The shingle-hash sets of one assistant message: `text`,
/// `tools` (names and arguments) and `combined` (both token streams in a
/// row).
#[derive(Debug, Clone, Default)]
pub struct Signature {
    text: HashSet<u64>,
    tools: HashSet<u64>,
    combined: HashSet<u64>,
}

impl Signature {
    /// 2026-09-26: Build from the message text and its tool calls as
    /// `(name, arguments)` strings, which are tokenised, not parsed. A
    /// channel under `MIN_CHANNEL_TOKENS` tokens gets an empty set.
    pub fn build<'a, I>(text: &str, tool_calls: I) -> Self
    where
        I: IntoIterator<Item = (&'a str, &'a str)>,
    {
        let text_tokens = tokenise(text);
        let mut tool_string = String::new();
        for (name, args) in tool_calls {
            if !tool_string.is_empty() {
                tool_string.push('\n');
            }
            tool_string.push_str(name);
            tool_string.push(' ');
            tool_string.push_str(args);
        }
        let tool_tokens = tokenise(&tool_string);

        let text = if text_tokens.len() >= MIN_CHANNEL_TOKENS {
            shingles(&text_tokens, SHINGLE_ORDER)
        } else {
            HashSet::new()
        };
        let tools = if tool_tokens.len() >= MIN_CHANNEL_TOKENS {
            shingles(&tool_tokens, SHINGLE_ORDER)
        } else {
            HashSet::new()
        };
        let combined_tokens: Vec<&str> = text_tokens
            .iter()
            .chain(tool_tokens.iter())
            .copied()
            .collect();
        let combined = if combined_tokens.len() >= MIN_CHANNEL_TOKENS {
            shingles(&combined_tokens, SHINGLE_ORDER)
        } else {
            HashSet::new()
        };
        Self {
            text,
            tools,
            combined,
        }
    }

    /// 2026-09-26: True when every channel is empty; `detect` skips such
    /// signatures.
    pub fn is_empty(&self) -> bool {
        self.text.is_empty() && self.tools.is_empty() && self.combined.is_empty()
    }
}

/// 2026-09-26: The verdict for assistant-message signatures given newest
/// first; every input is treated as an assistant turn. Channels are tried in
/// the order combined, text, tools. A run needs 2 or more pairs, or one pair
/// at or above `HIGH_SIMILARITY`. A later channel's run replaces the best so
/// far when it is longer, equally long with a higher score, or high-band
/// (at least two high pairs, or one in a run of 3 or more pairs) while the
/// best is not.
pub fn detect(recent_newest_first: &[Signature]) -> LoopState {
    let recent: Vec<&Signature> = recent_newest_first
        .iter()
        .filter(|s| !s.is_empty())
        .take(RECENT_WINDOW)
        .collect();
    if recent.len() < 3 {
        return LoopState::None;
    }

    let channels = [
        SimilarityChannel::Combined,
        SimilarityChannel::Text,
        SimilarityChannel::Tools,
    ];

    let mut best: Option<(SimilarityChannel, f64, usize, bool)> = None;
    for ch in channels {
        let sims: Vec<f64> = (0..recent.len() - 1)
            .map(|i| {
                let (a, b) = (recent[i], recent[i + 1]);
                let (sa, sb) = match ch {
                    SimilarityChannel::Text => (&a.text, &b.text),
                    SimilarityChannel::Tools => (&a.tools, &b.tools),
                    SimilarityChannel::Combined => (&a.combined, &b.combined),
                };
                jaccard(sa, sb)
            })
            .collect();
        let mut run_length = 0;
        let mut max_score = 0.0_f64;
        for &s in &sims {
            if s >= MODERATE_SIMILARITY {
                run_length += 1;
                if s > max_score {
                    max_score = s;
                }
            } else {
                break;
            }
        }
        // 2026-09-26: Here `run_length` counts pairs; `n` pairs span `n + 1`
        // turns.
        if run_length == 0 {
            continue;
        }
        let high_pairs = sims
            .iter()
            .take(run_length)
            .filter(|&&s| s >= HIGH_SIMILARITY)
            .count();
        let high_band = high_pairs >= 2 || (high_pairs >= 1 && run_length >= 3);
        let qualifies = run_length >= 2 || max_score >= HIGH_SIMILARITY;
        if !qualifies {
            continue;
        }
        match best {
            Some((_, prev_score, prev_run, prev_high)) => {
                let prefer_new = run_length > prev_run
                    || (run_length == prev_run && max_score > prev_score)
                    || (high_band && !prev_high);
                if prefer_new {
                    best = Some((ch, max_score, run_length, high_band));
                }
            }
            None => best = Some((ch, max_score, run_length, high_band)),
        }
    }

    match best {
        None => LoopState::None,
        Some((channel, score, run_length, high_band)) => {
            let turns = run_length + 1;
            if high_band || (score >= HIGH_SIMILARITY && turns >= 3) || turns >= 4 {
                LoopState::Suppress {
                    score,
                    run_length: turns,
                    channel,
                }
            } else {
                LoopState::Hint {
                    score,
                    run_length: turns,
                    channel,
                }
            }
        }
    }
}

/// 2026-09-26: One assistant turn's tool calls and what the tool results
/// after it looked like. `api/chat/loop_detect.rs` builds these from the
/// tool messages, which a `Signature` never sees.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CallOutcome {
    /// 2026-09-26: The turn's tool calls as one string (names and serialized
    /// arguments); `None` when the turn made no tool call.
    pub call_unit: Option<String>,
    /// 2026-09-26: True when at least one tool result followed and every one
    /// was error-shaped (`crate::hint_injector::looks_like_error`).
    pub failing: bool,
    /// 2026-09-26: The text of the tool results that followed, as
    /// `api/chat/loop_detect.rs` truncates it; `None` when none followed.
    /// `recent_results_progressing` compares these.
    pub result_unit: Option<String>,
}

/// 2026-09-26: True when `n >= 2`, each of the newest `n` turns has results,
/// and some adjacent pair of them differs: not equal after trimming, and a
/// shingle Jaccard below 0.9. A turn without results makes it false.
/// `api/chat/loop_detect.rs` skips the `<tool_call>` hard mask while this
/// holds, because a fix-and-rebuild cycle repeats its calls but not its
/// results.
pub fn recent_results_progressing(newest_first: &[CallOutcome], n: usize) -> bool {
    if n < 2 || newest_first.len() < n {
        return false;
    }
    let window = &newest_first[..n];
    if window.iter().any(|c| c.result_unit.is_none()) {
        return false;
    }
    window.windows(2).any(|pair| {
        let a = pair[0].result_unit.as_deref().unwrap_or("");
        let b = pair[1].result_unit.as_deref().unwrap_or("");
        if a.trim() == b.trim() {
            return false;
        }
        jaccard(&shingle_set(a), &shingle_set(b)) < 0.9
    })
}

/// 2026-09-26: `Some(run)` when the newest `run >= 3` turns have the same
/// `call_unit` and each is `failing`. It catches calls too short for
/// [`detect`] (`MIN_CHANNEL_TOKENS`). `api/chat/loop_detect.rs` uses it only
/// to raise `tool_call_repeat_count`, never for the hard mask: the repeated
/// call is failing, and `<tool_call>` is the model's way out.
pub fn detect_exact_failing_repeat(newest_first: &[CallOutcome]) -> Option<usize> {
    let newest_unit = newest_first.first().and_then(|c| c.call_unit.as_deref())?;
    let run = newest_first
        .iter()
        .take_while(|c| c.failing && c.call_unit.as_deref() == Some(newest_unit))
        .count();
    if run >= 3 { Some(run) } else { None }
}

/// 2026-09-26: True when `n >= 1` and each of the newest `n` turns made a
/// tool call and is `failing`. `api/chat/loop_detect.rs` then skips the
/// Suppress hard mask, since `<tool_call>` is the way out of a failing loop.
pub fn recent_calls_all_failing(newest_first: &[CallOutcome], n: usize) -> bool {
    n >= 1
        && newest_first.len() >= n
        && newest_first[..n]
            .iter()
            .all(|c| c.call_unit.is_some() && c.failing)
}

/// 2026-09-26: The shingle set of `text`; the duplicate-error masking in
/// `api/chat/msg_entry.rs` uses it with [`jaccard`].
pub(crate) fn shingle_set(text: &str) -> HashSet<u64> {
    shingles(&tokenise(text), SHINGLE_ORDER)
}

fn tokenise(s: &str) -> Vec<&str> {
    // 2026-09-26: Split at every non-alphanumeric char. Tokens stay borrowed;
    // `shingles` lowercases while hashing.
    s.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .collect()
}

fn shingles(tokens: &[&str], order: usize) -> HashSet<u64> {
    if tokens.len() < order {
        return HashSet::new();
    }
    let mut out = HashSet::with_capacity(tokens.len());
    for window in tokens.windows(order) {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        for tok in window {
            for ch in tok.chars() {
                for lc in ch.to_lowercase() {
                    lc.hash(&mut h);
                }
            }
            // 2026-09-26: A separator, so "ab cd" and "abcd" hash differently.
            0u8.hash(&mut h);
        }
        out.insert(h.finish());
    }
    out
}

/// 2026-09-26: `|A ∩ B| / |A ∪ B|`, or 0 when either set is empty.
pub(crate) fn jaccard(a: &HashSet<u64>, b: &HashSet<u64>) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let intersection = a.intersection(b).count() as f64;
    let union = a.union(b).count() as f64;
    if union == 0.0 {
        0.0
    } else {
        intersection / union
    }
}

#[cfg(test)]
mod tests;
