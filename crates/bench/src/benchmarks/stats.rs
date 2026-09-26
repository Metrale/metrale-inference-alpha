// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Shared measurement helpers: accept length, prompt synthesis,
//! percentiles, median and formatting.
//!
//! Owner: bench.
//! Invariants:
//! - `percentile` uses the nearest-rank rule of `pct` in
//!   `bench/bench_concurrency.py`, over the finite values only.
//! - `make_prompt` with an empty tag uses that script's word list and word
//!   count. Checked 2026-09-26 by running both: the texts are equal up to an
//!   ISL of 1104; above it the script emits fewer words than it asks for.

use std::fmt::Write as _;

/// 2026-09-26: Emitted tokens per decode step, `completion / (completion -
/// accepted)`: 1.0 when no draft token was accepted. `decode_floor` and
/// `concurrency` both read this one rule.
///
/// `None` when it cannot be derived: no accept count on the wire, or
/// `accepted >= completion`. A `None` is not 1.0, which would claim the
/// server accepted nothing.
pub fn accept_len(
    completion_tokens: usize,
    accepted_prediction_tokens: Option<usize>,
) -> Option<f64> {
    let accepted = accepted_prediction_tokens?;
    (accepted < completion_tokens && completion_tokens > 0)
        .then(|| completion_tokens as f64 / (completion_tokens - accepted) as f64)
}

/// 2026-09-26: The filler sentences, the word list of `bench/bench_concurrency.py`'s
/// `make_prompt`.
const FILLER: &str = concat!(
    "The quick brown fox jumped over the lazy dog near a river bank. ",
    "Mountains rise above the clouds while birds sing their morning songs. ",
    "Science explores the universe through careful observation and experiment. ",
    "Ancient civilizations built remarkable structures that still stand today. ",
    "Music fills the air with rhythm and harmony across every culture. ",
    "Technology advances rapidly changing how people communicate and work. ",
    "Forests provide shelter for countless species of plants and animals. ",
    "Ocean waves crash upon the shore under the light of the moon. ",
);

/// 2026-09-26: Should the prompt push the model to fill the output budget?
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum PromptMode {
    /// 2026-09-26: No instruction appended; the model stops when it chooses.
    Natural,
    /// 2026-09-26: Append a counting instruction, so replies run toward the
    /// output budget. The default.
    #[default]
    Count,
}

impl PromptMode {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "natural" | "hello" => Some(PromptMode::Natural),
            "count" => Some(PromptMode::Count),
            _ => None,
        }
    }
}

/// 2026-09-26: Build a prompt of roughly `isl_tokens` tokens. A non-empty
/// `prefix_tag` opens the prompt as `[tag] `: a unique tag gives a prompt no
/// earlier request shares, and a constant tag gives the same prompt every time.
pub fn make_prompt(isl_tokens: usize, mode: PromptMode, prefix_tag: &str) -> String {
    // 2026-09-26: 12 tokens are left for the chat template, as in
    // `bench/bench_concurrency.py`.
    let needed = isl_tokens.saturating_sub(12).max(1);
    let words: Vec<&str> = FILLER.split_whitespace().collect();
    let mut out = String::with_capacity(needed * 6 + prefix_tag.len() + 80);
    if !prefix_tag.is_empty() {
        let _ = write!(out, "[{prefix_tag}] ");
    }
    for i in 0..needed {
        if i > 0 {
            out.push(' ');
        }
        out.push_str(words[i % words.len()]);
    }
    if mode == PromptMode::Count {
        out.push_str(" Count from 1 upward, one number per line, until told to stop.");
    }
    out
}

/// 2026-09-26: `p`-th percentile (0–100) of `values`, nearest rank:
/// `idx = min(int(n*p/100 + 0.5), n-1)` over the sorted finite values.
pub fn percentile(values: &[f64], p: u32) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if sorted.is_empty() {
        return None;
    }
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("filtered to finite"));
    let n = sorted.len();
    let idx = ((n as f64 * p as f64 / 100.0) + 0.5) as usize;
    Some(sorted[idx.min(n - 1)])
}

/// 2026-09-26: True median of the finite values: the middle element for odd `n`,
/// the mean of the two middle elements for even `n`. Not `percentile(values,
/// 50)`, whose index for n=3 is 2, the maximum.
pub fn median(values: &[f64]) -> Option<f64> {
    let mut sorted: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if sorted.is_empty() {
        return None;
    }
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("filtered to finite"));
    let n = sorted.len();
    Some(if n % 2 == 1 {
        sorted[n / 2]
    } else {
        sorted[n / 2 - 1].midpoint(sorted[n / 2])
    })
}

/// 2026-09-26: p50 / p90 / p99, each from `percentile`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Percentiles {
    pub p50: Option<f64>,
    pub p90: Option<f64>,
    pub p99: Option<f64>,
}

impl Percentiles {
    pub fn of(values: &[f64]) -> Self {
        Self {
            p50: percentile(values, 50),
            p90: percentile(values, 90),
            p99: percentile(values, 99),
        }
    }
}

/// 2026-09-26: Format an optional millisecond value for a table cell: seconds
/// from 10 s up, and `—` for none, negative or non-finite.
pub fn fmt_ms(v: Option<f64>) -> String {
    match v {
        Some(ms) if !ms.is_finite() || ms < 0.0 => "—".into(),
        Some(ms) if ms >= 10_000.0 => format!("{:.1}s", ms / 1000.0),
        Some(ms) => format!("{ms:.1}"),
        None => "—".into(),
    }
}

/// 2026-09-26: Relative change `new` vs `base`, in percent. `None` when either is
/// non-finite, `base` is not positive, or the result is not finite.
pub fn pct_delta(new: Option<f64>, base: Option<f64>) -> Option<f64> {
    match (new, base) {
        (Some(n), Some(b)) if n.is_finite() && b.is_finite() && b > f64::EPSILON => {
            let delta = (n - b) / b * 100.0;
            delta.is_finite().then_some(delta)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    fn sha256(text: &str) -> String {
        format!("{:x}", Sha256::digest(text.as_bytes()))
    }

    #[test]
    fn percentile_matches_the_python_nearest_rank_rule() {
        let v: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        assert_eq!(percentile(&v, 50), Some(6.0));
        assert_eq!(percentile(&v, 90), Some(10.0));
        assert_eq!(percentile(&v, 100), Some(10.0));
        assert_eq!(percentile(&[], 50), None);
    }

    #[test]
    fn percentiles_ignore_non_finite_samples() {
        let v = vec![1.0, f64::NAN, 3.0, f64::INFINITY];
        assert_eq!(
            Percentiles::of(&v),
            Percentiles {
                p50: Some(3.0),
                p90: Some(3.0),
                p99: Some(3.0),
            }
        );
        assert_eq!(percentile(&[f64::NAN], 50), None);
    }

    #[test]
    fn median_handles_empty_odd_even_and_large_finite_samples() {
        assert_eq!(median(&[]), None);
        assert_eq!(median(&[f64::NAN, f64::INFINITY]), None);
        assert_eq!(median(&[3.0, 1.0, 2.0]), Some(2.0));
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), Some(2.5));
        assert_eq!(median(&[f64::MAX, f64::MAX]), Some(f64::MAX));
    }

    #[test]
    fn prompt_length_tracks_the_request_and_the_tag_changes_the_text() {
        let a = make_prompt(256, PromptMode::Natural, "");
        let b = make_prompt(1024, PromptMode::Natural, "");
        assert!(b.len() > a.len());
        assert_eq!(a.split_whitespace().count(), 256 - 12);
        assert!(
            make_prompt(256, PromptMode::Natural, "s1").starts_with("[s1] The quick brown fox")
        );
        assert_eq!(
            make_prompt(256, PromptMode::Natural, "s1"),
            make_prompt(256, PromptMode::Natural, "s1")
        );
        assert_ne!(
            make_prompt(256, PromptMode::Natural, "s1"),
            make_prompt(256, PromptMode::Natural, "s2")
        );
    }

    #[test]
    fn prompt_modes_pin_the_complete_natural_and_forced_bytes() {
        assert_eq!(
            sha256(&make_prompt(64, PromptMode::Natural, "fixture")),
            "1d34b0e6f6c8b13f614be586f7f29ab9fcf918ed11416a3c62702d4a05bf7a54"
        );
        assert_eq!(
            sha256(&make_prompt(64, PromptMode::Count, "fixture")),
            "619330d2a0c46380c7c3e815226610d5cdea78b760736f496f64a6ec8a6efe96"
        );
    }

    #[test]
    fn millisecond_formatting_rejects_invalid_durations() {
        assert_eq!(fmt_ms(None), "—");
        assert_eq!(fmt_ms(Some(12.25)), "12.2");
        assert_eq!(fmt_ms(Some(10_000.0)), "10.0s");
        for invalid in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(fmt_ms(Some(invalid)), "—", "value={invalid:?}");
        }
    }

    #[test]
    fn pct_delta_is_none_when_there_is_no_usable_baseline() {
        assert_eq!(pct_delta(Some(110.0), Some(100.0)), Some(10.0));
        assert_eq!(pct_delta(Some(110.0), None), None);
        assert_eq!(pct_delta(Some(110.0), Some(0.0)), None);
        assert_eq!(pct_delta(Some(110.0), Some(-100.0)), None);
        assert_eq!(pct_delta(Some(110.0), Some(f64::INFINITY)), None);
        assert_eq!(pct_delta(Some(f64::INFINITY), Some(100.0)), None);
        assert_eq!(pct_delta(Some(f64::NAN), Some(100.0)), None);
        assert_eq!(pct_delta(Some(f64::MAX), Some(1.0)), None);
    }
}
