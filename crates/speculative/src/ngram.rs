// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Prompt-lookup draft proposer: finds the longest suffix of the
//! token history (prompt and generated tokens) that occurred earlier, and
//! proposes the token that followed it. Host code only.
//!
//! Owner: speculative.
//! Invariants: none beyond the types.

/// 2026-09-25: The proposer. It keeps no index: `propose` searches the
/// history it is given.
pub struct NgramProposer {
    /// 2026-09-25: Shortest suffix length tried.
    min_match: usize,
    /// 2026-09-25: Longest suffix length tried.
    max_match: usize,
    /// 2026-09-25: Counts added to by `record_accept` and `record_reject`.
    pub accepts: u64,
    pub rejects: u64,
}

#[allow(clippy::len_without_is_empty)]
impl NgramProposer {
    pub fn new(_order: usize) -> Self {
        Self {
            min_match: 2,
            max_match: 16,
            accepts: 0,
            rejects: 0,
        }
    }

    /// 2026-09-25: Counts an accepted draft in `accepts`.
    pub fn record_accept(&mut self) {
        self.accepts += 1;
    }

    /// 2026-09-25: Counts a rejected draft in `rejects`.
    pub fn record_reject(&mut self) {
        self.rejects += 1;
    }

    /// 2026-09-25: For the longest suffix of `all_tokens` (from `min_match`
    /// to `max_match` tokens) that also occurs earlier, returns the token after
    /// its earliest occurrence; `None` when no suffix repeats.
    pub fn propose(&self, all_tokens: &[u32]) -> Option<u32> {
        let len = all_tokens.len();
        if len < self.min_match + 1 {
            return None;
        }

        let max_n = self.max_match.min(len - 1);
        let mut best_next: Option<u32> = None;
        let mut best_match_len: usize = 0;

        // 2026-09-25: Suffix = all_tokens[len-n..len], searched at positions
        // 0..=len-n-1 so that a token follows the match.
        for n in self.min_match..=max_n {
            let suffix = &all_tokens[len - n..len];
            for start in 0..=(len - n - 1) {
                if all_tokens[start..start + n] == *suffix && n > best_match_len {
                    best_match_len = n;
                    best_next = Some(all_tokens[start + n]);
                    break;
                }
            }
        }

        best_next
    }

    /// 2026-09-25: No-op: the proposer keeps no index.
    pub fn observe(&mut self, _history: &[u32], _next: u32) {}

    pub fn len(&self) -> usize {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prompt_lookup_basic() {
        let p = NgramProposer::new(4);

        // 2026-09-25: The suffix [1, 2, 3] occurs at position 0, followed by 4.
        let tokens = vec![1, 2, 3, 4, 5, 1, 2, 3];
        assert_eq!(p.propose(&tokens), Some(4));
    }

    #[test]
    fn test_prompt_lookup_no_match() {
        let p = NgramProposer::new(4);

        let tokens = vec![1, 2, 3, 4, 5, 6, 7, 8];
        assert_eq!(p.propose(&tokens), None);
    }

    #[test]
    fn test_prompt_lookup_short() {
        let p = NgramProposer::new(4);

        let tokens = vec![1, 2];
        assert_eq!(p.propose(&tokens), None);
    }

    #[test]
    fn test_prompt_lookup_repetitive() {
        let p = NgramProposer::new(4);

        // 2026-09-25: [A, B, C, A, B, C, A, B]: the longest repeated suffix is
        // [A, B, C, A, B], at position 0, followed by C.
        let tokens = vec![10, 20, 30, 10, 20, 30, 10, 20];
        assert_eq!(p.propose(&tokens), Some(30));
    }
}
