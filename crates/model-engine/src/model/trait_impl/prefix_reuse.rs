// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The prompt-token count stored in `SequenceState::reused_prefix_tokens`, which
//! the scheduler reports as `usage.prompt_tokens_details.cached_tokens`: the tokens whose KV
//! the prefill read from the prefix cache instead of recomputing.
//!
//! A lookup can match, take refs on the matched blocks, and the prefill can still recompute
//! all of their KV, for example a hybrid-SSM model with no usable SSM snapshot, the
//! exact-leaf snapshot bypass (`METRALE_MARCONI_EXACT` unset), or a restore below
//! `marconi_min_tokens()`. The lookup's count stays in `cached_prefix_tokens`, which block-ref
//! accounting uses.
//!
//! Owner: model-engine.
//! Invariants: the reported count never exceeds the lookup's matched tokens.

/// 2026-09-25: Prompt tokens served from the prefix cache.
///
/// `matched`: `PrefixMatch::matched_tokens` from the lookup.
/// `kv_write_start`: the position the prefill starts writing KV at, 0 when the
///   match was found but discarded.
/// `skip`: whether the prefill took the skip path.
///
/// Capped at `matched`: the SSM paths can set `kv_write_start` from a snapshot
/// depth, and a snapshot deeper than the match still reuses only the match.
pub(crate) fn reused_prefix_tokens(matched: usize, kv_write_start: usize, skip: bool) -> usize {
    if !skip && kv_write_start == 0 {
        return 0;
    }
    kv_write_start.min(matched)
}

#[cfg(test)]
mod tests {
    use super::reused_prefix_tokens;

    /// 2026-09-25: A match the SSM arm discards (`kv_write_start = 0`) reports 0.
    #[test]
    fn a_discarded_match_reports_zero_not_the_lookup_length() {
        assert_eq!(reused_prefix_tokens(48, 0, false), 0);
    }

    /// 2026-09-25: The exact-leaf bypass matches the whole prompt and recomputes it all.
    #[test]
    fn the_exact_leaf_bypass_reports_zero() {
        assert_eq!(reused_prefix_tokens(4593, 0, false), 0);
    }

    #[test]
    fn a_real_skip_reports_the_reused_prefix() {
        assert_eq!(reused_prefix_tokens(48, 48, true), 48);
    }

    #[test]
    fn an_intermediate_snapshot_reports_only_what_it_skipped() {
        assert_eq!(reused_prefix_tokens(512, 256, true), 256);
    }

    #[test]
    fn the_report_never_exceeds_the_matched_prefix() {
        assert_eq!(reused_prefix_tokens(256, 512, true), 256);
    }

    #[test]
    fn no_match_reports_zero() {
        assert_eq!(reused_prefix_tokens(0, 0, false), 0);
    }
}
