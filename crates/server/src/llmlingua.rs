// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Prompt compression by per-token keep probabilities. `compress`
//! keeps the most probable whitespace-separated tokens up to a target ratio;
//! `compress_with_anchors` also keeps a head and a tail verbatim. The
//! probabilities come from a [`KeepClassifier`]; [`HeuristicClassifier`] is
//! the only implementation, and nothing outside this module calls either.
//!
//! Owner: server.
//! Invariants: `compress` keeps at least one token of a non-empty input, and
//! kept tokens stay in document order.

/// 2026-09-26: Scores tokens for `compress`.
pub trait KeepClassifier {
    /// 2026-09-26: A keep probability in [0, 1] per input token, one output
    /// per input (`compress` checks the length only with `debug_assert_eq!`).
    fn keep_probs(&self, tokens: &[&str]) -> Vec<f32>;
}

/// 2026-09-26: Blank tokens score lowest, `STOPWORDS` (any case) next, and
/// every other token highest.
pub struct HeuristicClassifier;

impl KeepClassifier for HeuristicClassifier {
    fn keep_probs(&self, tokens: &[&str]) -> Vec<f32> {
        const STOPWORDS: &[&str] = &[
            "the", "a", "an", "and", "or", "but", "of", "in", "on", "at", "to", "for", "with",
            "by", "as", "is", "was", "are", "were", "be", "been",
        ];
        tokens
            .iter()
            .map(|t| {
                let trimmed = t.trim();
                if trimmed.is_empty() {
                    0.0
                } else if STOPWORDS.contains(&trimmed.to_ascii_lowercase().as_str()) {
                    0.4
                } else {
                    0.9
                }
            })
            .collect()
    }
}

/// 2026-09-26: Keep the `round(n × target_ratio)` most probable of the `n`
/// whitespace-separated tokens (ratio clamped to [0.05, 1], at least one
/// token), ties at the cut filled in document order, and join them with
/// single spaces. Returns the text and the kept fraction. When every token
/// would be kept, `text` comes back unchanged with ratio 1.0; an empty input
/// gives `("", 1.0)`.
pub fn compress<C: KeepClassifier>(text: &str, target_ratio: f32, classifier: &C) -> (String, f32) {
    let target_ratio = target_ratio.clamp(0.05, 1.0);
    let tokens: Vec<&str> = text.split_whitespace().collect();
    if tokens.is_empty() {
        return (String::new(), 1.0);
    }
    let probs = classifier.keep_probs(&tokens);
    debug_assert_eq!(probs.len(), tokens.len());

    let n_total = tokens.len();
    let n_keep = ((n_total as f32 * target_ratio).round() as usize).max(1);
    if n_keep >= n_total {
        return (text.to_string(), 1.0);
    }

    let mut sorted_probs: Vec<f32> = probs.clone();
    sorted_probs.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let cutoff = sorted_probs[n_keep.saturating_sub(1)];

    // 2026-09-26: Two passes so exactly `n_keep` tokens are kept when several
    // share the cutoff probability: first every token above it, then tokens
    // equal to it, in document order, until `n_keep` is reached.
    let mut keep_mask = vec![false; n_total];
    let mut kept_count = 0usize;
    for (i, &p) in probs.iter().enumerate() {
        if p > cutoff && kept_count < n_keep {
            keep_mask[i] = true;
            kept_count += 1;
        }
    }
    for (i, &p) in probs.iter().enumerate() {
        if kept_count >= n_keep {
            break;
        }
        if p == cutoff && !keep_mask[i] {
            keep_mask[i] = true;
            kept_count += 1;
        }
    }
    let out: Vec<&str> = tokens
        .iter()
        .enumerate()
        .filter(|(i, _)| keep_mask[*i])
        .map(|(_, &t)| t)
        .collect();
    let achieved = kept_count as f32 / n_total as f32;
    (out.join(" "), achieved)
}

/// 2026-09-26: Keep the first `n_preserve_head` and last `n_preserve_tail`
/// tokens verbatim and `compress` the middle at the ratio that would bring
/// the whole text to `target_ratio` (clamped to [0.05, 1]). An input no
/// longer than head plus tail comes back unchanged with ratio 1.0.
pub fn compress_with_anchors<C: KeepClassifier>(
    text: &str,
    target_ratio: f32,
    n_preserve_head: usize,
    n_preserve_tail: usize,
    classifier: &C,
) -> (String, f32) {
    let tokens: Vec<&str> = text.split_whitespace().collect();
    if tokens.len() <= n_preserve_head + n_preserve_tail {
        return (text.to_string(), 1.0);
    }
    let head: Vec<&str> = tokens.iter().take(n_preserve_head).copied().collect();
    let tail: Vec<&str> = tokens
        .iter()
        .skip(tokens.len() - n_preserve_tail)
        .copied()
        .collect();
    let middle_text = tokens[n_preserve_head..tokens.len() - n_preserve_tail].join(" ");
    let middle_target = if target_ratio >= 1.0 {
        1.0
    } else {
        let total = tokens.len() as f32;
        let preserved = (n_preserve_head + n_preserve_tail) as f32;
        let target_total = total * target_ratio;
        ((target_total - preserved).max(1.0) / (total - preserved).max(1.0)).clamp(0.05, 1.0)
    };
    let (middle_compressed, _achieved) = compress(&middle_text, middle_target, classifier);
    let mut out = String::new();
    out.push_str(&head.join(" "));
    if !middle_compressed.is_empty() {
        out.push(' ');
        out.push_str(&middle_compressed);
    }
    if !tail.is_empty() {
        out.push(' ');
        out.push_str(&tail.join(" "));
    }
    let achieved = (out.split_whitespace().count() as f32) / (tokens.len() as f32);
    (out, achieved)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compress_at_full_ratio_returns_input() {
        let text = "the quick brown fox jumps over the lazy dog";
        let (out, r) = compress(text, 1.0, &HeuristicClassifier);
        assert_eq!(out, text);
        assert!((r - 1.0).abs() < 1e-3);
    }

    #[test]
    fn compress_at_half_ratio_drops_stopwords_first() {
        let text = "the alpha and beta is in the gamma";
        let (out, r) = compress(text, 0.5, &HeuristicClassifier);
        assert!(out.contains("alpha"));
        assert!(out.contains("beta"));
        assert!(out.contains("gamma"));
        assert!(r <= 0.6);
    }

    #[test]
    fn compress_empty_input_returns_empty() {
        let (out, r) = compress("", 0.5, &HeuristicClassifier);
        assert_eq!(out, "");
        assert_eq!(r, 1.0);
    }

    #[test]
    fn compress_with_anchors_preserves_head_and_tail() {
        let text = "head1 head2 mid1 mid2 mid3 mid4 mid5 tail1 tail2";
        let (out, _) = compress_with_anchors(text, 0.6, 2, 2, &HeuristicClassifier);
        assert!(out.starts_with("head1 head2"), "head preserved: {out}");
        assert!(out.ends_with("tail1 tail2"), "tail preserved: {out}");
    }

    #[test]
    fn compress_below_min_ratio_clamps_to_at_least_one_token() {
        let text = "alpha beta gamma";
        let (out, _) = compress(text, 0.01, &HeuristicClassifier);
        let kept = out.split_whitespace().count();
        assert!(kept >= 1, "must keep at least one token");
    }
}
