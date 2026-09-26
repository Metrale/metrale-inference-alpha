// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The concurrency leg shared by the video and vision benchmarks:
//! for each of `LEVELS`, that many copies of one request are sent at once and
//! scored for how many returned, how many were correct, and whether they
//! agree on one prompt-token count.
//!
//! It checks correctness, not throughput: `wall_ms` is recorded and never
//! read by `ok`, `ok_against` or `sweep_ok`. The copies are identical, so this
//! leg cannot see one request answered with another's input;
//! `media_integrity::heterogeneous_concurrency` covers that.
//!
//! Owner: bench, video and vision.
//! Invariants: none beyond the types.

use std::time::{Duration, Instant};

use crate::http::{self, ChatOutcome};
use crate::plugin::PluginHandle;

/// 2026-09-26: The levels swept, in order. `sweep_ok` needs one result per
/// level, and the first (C=1) sets the prompt-token baseline.
pub const LEVELS: &[usize] = &[1, 2, 4];

/// 2026-09-26: One level's outcome.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LevelResult {
    pub conc: usize,
    /// 2026-09-26: Replies that came back without an error.
    pub returned: usize,
    /// 2026-09-26: Returned replies for which `is_correct` held.
    pub correct: usize,
    /// 2026-09-26: Distinct prompt-token counts among the returned replies;
    /// `ok` requires exactly one.
    pub distinct_token_counts: usize,
    /// 2026-09-26: The agreed prompt-token count when there was exactly one,
    /// kept so later levels can be compared with the C=1 level.
    pub prompt_tokens: Option<usize>,
    pub wall_ms: u128,
    pub errors: Vec<String>,
}

impl LevelResult {
    pub fn ok(&self) -> bool {
        self.correct == self.conc && self.distinct_token_counts == 1 && self.prompt_tokens.is_some()
    }

    /// 2026-09-26: `ok`, and the agreed prompt-token count equals the C=1
    /// level's.
    pub fn ok_against(&self, baseline_prompt_tokens: usize) -> bool {
        self.ok() && self.prompt_tokens == Some(baseline_prompt_tokens)
    }

    pub fn geometry_detail(&self, baseline_prompt_tokens: Option<usize>) -> String {
        match (self.prompt_tokens, baseline_prompt_tokens) {
            (Some(got), Some(want)) if got != want => {
                format!("{got} prompt tokens, C=1 baseline {want}")
            }
            (Some(got), _) => format!("one geometry ({got} prompt tokens)"),
            (None, _) => format!("{} distinct token counts", self.distinct_token_counts),
        }
    }
}

/// 2026-09-26: True when there is one result per `LEVELS` entry, in order,
/// and each is `ok_against` the first result's prompt-token count.
pub fn sweep_ok(results: &[LevelResult]) -> bool {
    let Some(baseline_prompt_tokens) = results.first().and_then(|r| r.prompt_tokens) else {
        return false;
    };
    results.len() == LEVELS.len()
        && results
            .iter()
            .zip(LEVELS)
            .all(|(r, &level)| r.conc == level && r.ok_against(baseline_prompt_tokens))
}

/// 2026-09-26: Send `conc` copies of `body` at once and score them with
/// `is_correct`.
pub async fn run_level(
    handle: &PluginHandle,
    body: &serde_json::Value,
    conc: usize,
    timeout: Duration,
    // 2026-09-26: `Sync` as well as `Fn`: the reference is held across an
    // await inside a `Send` benchmark future, and `&dyn Fn` is `Send` only
    // when the closure type is `Sync`.
    is_correct: &(dyn Fn(&str) -> bool + Sync),
) -> LevelResult {
    let start = Instant::now();
    let futures: Vec<_> = (0..conc)
        .map(|_| http::chat_stream(handle.target(), body, timeout))
        .collect();
    let outcomes: Vec<anyhow::Result<ChatOutcome>> = futures::future::join_all(futures).await;
    let wall_ms = start.elapsed().as_millis();

    let mut returned = 0usize;
    let mut correct = 0usize;
    let mut counts: Vec<usize> = Vec::new();
    let mut errors = Vec::new();
    for o in outcomes {
        match o {
            Ok(out) => {
                returned += 1;
                counts.push(out.prompt_tokens);
                if is_correct(out.text.trim()) {
                    correct += 1;
                }
            }
            Err(e) => errors.push(crate::benchmarks::one_line(format!("{e:#}"))),
        }
    }
    counts.sort_unstable();
    counts.dedup();
    let prompt_tokens = (counts.len() == 1).then(|| counts[0]);
    LevelResult {
        conc,
        returned,
        correct,
        distinct_token_counts: counts.len(),
        prompt_tokens,
        wall_ms,
        errors,
    }
}

#[cfg(test)]
#[path = "concurrency_tests.rs"]
mod concurrency_tests;
