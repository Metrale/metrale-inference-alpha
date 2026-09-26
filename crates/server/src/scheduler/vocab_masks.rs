// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: per-token classification masks, derived from the resolved
//! tokenizer.
//!
//! Owner: scheduler.
//! Invariants:
//! - Each mask is indexed by token id, so it is valid only for the tokenizer
//!   it was built from. `serve_phases/tokenizer_runtime.rs` builds all three
//!   and carries them in `TokenizerRuntime` beside that tokenizer's other
//!   derived values.
//! - A `None` mask disables what reads it; no reader treats it as an error.

use std::sync::Arc;

/// 2026-09-25: the three token-classification masks for one vocabulary.
/// Cloning copies three `Arc`s.
#[derive(Clone, Default)]
pub struct VocabMasks {
    /// 2026-09-25: `mask[id]` iff token `id` decodes to a non-empty run of
    /// ASCII digits, with at most one leading space. Read by the
    /// digit-normalised content-loop check; `None` turns that check off.
    pub numeric: Option<Arc<[bool]>>,
    /// 2026-09-25: `mask[id]` iff token `id` decodes to text ending in a
    /// newline, or in `.`, `!` or `?` followed only by closing quotes,
    /// brackets or whitespace. Read by rollback-to-boundary (`None` makes it
    /// decline with `RollbackFallback::NoBoundary`) and by the forced
    /// `</think>` sentence-boundary check.
    pub boundary: Option<Arc<[bool]>>,
    /// 2026-09-25: `mask[id]` iff token `id` decodes to text whose last
    /// character is alphanumeric, so `</think>` right after it would split a
    /// word. Read by `MidWordThinkEndMask`; `None` turns it off.
    pub mid_word: Option<Arc<[bool]>>,
}

impl VocabMasks {
    /// 2026-09-25: an absent mask or an out-of-range id reads as false.
    pub fn is_numeric(&self, id: u32) -> bool {
        Self::at(&self.numeric, id)
    }

    pub fn is_boundary(&self, id: u32) -> bool {
        Self::at(&self.boundary, id)
    }

    pub fn is_mid_word(&self, id: u32) -> bool {
        Self::at(&self.mid_word, id)
    }

    fn at(mask: &Option<Arc<[bool]>>, id: u32) -> bool {
        mask.as_deref()
            .and_then(|m| m.get(id as usize))
            .copied()
            .unwrap_or(false)
    }
}

impl std::fmt::Debug for VocabMasks {
    /// 2026-09-25: prints each mask's population, not its contents.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let count = |m: &Option<Arc<[bool]>>| match m.as_deref() {
            Some(m) => format!("{}/{}", m.iter().filter(|b| **b).count(), m.len()),
            None => "none".to_string(),
        };
        f.debug_struct("VocabMasks")
            .field("numeric", &count(&self.numeric))
            .field("boundary", &count(&self.boundary))
            .field("mid_word", &count(&self.mid_word))
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mask(bits: &[bool]) -> Option<Arc<[bool]>> {
        Some(Arc::from(bits.to_vec()))
    }

    #[test]
    fn an_absent_mask_classifies_nothing() {
        let m = VocabMasks::default();
        assert!(!m.is_numeric(0) && !m.is_boundary(7) && !m.is_mid_word(u32::MAX));
    }

    #[test]
    fn an_id_past_the_end_is_unclassified_rather_than_a_panic() {
        // 2026-09-25: a mask shorter than the vocabulary reads as false
        // past its end instead of panicking.
        let m = VocabMasks {
            numeric: mask(&[true, false]),
            ..Default::default()
        };
        assert!(m.is_numeric(0));
        assert!(!m.is_numeric(1));
        assert!(!m.is_numeric(2), "past the end of a 2-token vocabulary");
        assert!(!m.is_numeric(50_000));
    }

    #[test]
    fn the_three_masks_are_independent() {
        let m = VocabMasks {
            numeric: mask(&[true, false]),
            boundary: mask(&[false, true]),
            mid_word: None,
        };
        assert!(m.is_numeric(0) && !m.is_boundary(0) && !m.is_mid_word(0));
        assert!(!m.is_numeric(1) && m.is_boundary(1) && !m.is_mid_word(1));
    }

    #[test]
    fn debug_reports_populations_not_contents() {
        let m = VocabMasks {
            numeric: mask(&[true, false, true]),
            ..Default::default()
        };
        let s = format!("{m:?}");
        assert!(s.contains("2/3"), "{s}");
        assert!(s.contains("none"), "{s}");
        assert!(!s.contains("true"), "must not print the bools: {s}");
    }
}
