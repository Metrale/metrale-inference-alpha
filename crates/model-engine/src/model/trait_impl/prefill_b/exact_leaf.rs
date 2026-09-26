// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Decides whether the prefill-end ("exact leaf") SSM snapshot gets a pool slot.
//!
//! The exact leaf holds the state after the whole N-token prompt. Restoring it for
//! an identical prompt is refused unless `METRALE_MARCONI_EXACT=1`
//! (`snap_agree::local_proposal`). When this prefill already has a checkpoint at
//! most two blocks below N, a prompt that extends this one restores from that
//! checkpoint instead. Measured 2026-09-13 on an
//! 8-slot snapshot pool: with the leaf also saved, the tail checkpoint was evicted
//! first, and from two concurrent prompts upward every warm request recomputed its
//! SSM state.
//!
//! Owner: model-engine prefill (SSM prefix cache).
//! Invariants: none beyond the types.

/// 2026-09-25: Whether `finalize_last` saves the exact leaf.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExactLeaf {
    /// 2026-09-25: Save it: this prefill has no checkpoint below the prompt end and within two
    /// blocks of it.
    Save,
    /// 2026-09-25: Save it: the exact shortcut is enabled, and it needs the last-token hidden
    /// that only the leaf save stashes (`finalize_last`, `save_hidden`).
    SaveForShortcut,
    /// 2026-09-25: Skip it: a checkpoint at token `tail` lies `replay` tokens (at most two
    /// blocks) below the prompt end.
    Redundant { tail: usize, replay: usize },
}

/// 2026-09-25: Decide from the checkpoint this prefill already has.
///
/// `tail_checkpoint` is `seq.tail_checkpoint_tokens`: the depth of a prompt-tail
/// checkpoint saved during this prefill (`save_checkpoint.rs`) or of the snapshot
/// this prefill restored from (`prefix_lookup.rs`), and `None` when there is neither.
pub fn exact_leaf(
    tail_checkpoint: Option<usize>,
    total: usize,
    block_size: usize,
    exact_shortcut_enabled: bool,
) -> ExactLeaf {
    if exact_shortcut_enabled {
        return ExactLeaf::SaveForShortcut;
    }
    match tail_checkpoint {
        // 2026-09-25: The tail split cuts one block below the last block boundary under
        // `total` (`prefill_chunk_dispatch`), so its replay is at most two blocks. A
        // checkpoint farther back does not stand in for the leaf.
        Some(tail) if tail < total && total - tail <= 2 * block_size => ExactLeaf::Redundant {
            tail,
            replay: total - tail,
        },
        _ => ExactLeaf::Save,
    }
}

/// 2026-09-25: `METRALE_MARCONI_EXACT=1` enables the exact full-prompt snapshot shortcut.
/// This is the only reader of that variable.
pub fn marconi_exact_enabled() -> bool {
    std::env::var("METRALE_MARCONI_EXACT").as_deref() == Ok("1")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_leaf_is_redundant_exactly_when_the_tail_split_fired_for_this_prompt() {
        // 2026-09-25: 657 tokens, bs 16: tail cut at 640, replay 17.
        assert_eq!(
            exact_leaf(Some(640), 657, 16, false),
            ExactLeaf::Redundant {
                tail: 640,
                replay: 17
            }
        );
        // 2026-09-25: Block-aligned prompt: cut at total - 2bs, replay 32, still inside.
        assert_eq!(
            exact_leaf(Some(608), 640, 16, false),
            ExactLeaf::Redundant {
                tail: 608,
                replay: 32
            }
        );
        // 2026-09-25: Negative controls: no checkpoint; one far below the prompt end;
        // one at the end.
        assert_eq!(exact_leaf(None, 657, 16, false), ExactLeaf::Save);
        assert_eq!(exact_leaf(Some(256), 657, 16, false), ExactLeaf::Save);
        assert_eq!(exact_leaf(Some(657), 657, 16, false), ExactLeaf::Save);
        // 2026-09-25: With the shortcut enabled the leaf is always saved.
        assert_eq!(
            exact_leaf(Some(640), 657, 16, true),
            ExactLeaf::SaveForShortcut
        );
    }
}
