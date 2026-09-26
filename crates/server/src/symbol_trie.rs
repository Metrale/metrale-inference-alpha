// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Token-level trie over a closed set of symbol strings, giving the tokens
//! allowed next at each position.
//!
//! Owner: server.
//! Invariants:
//! - The trie is over token IDs; each string is tokenised once, in `build`.
//! - Node 0 is the root.
//! - `terminal` marks where a complete symbol ends, so a symbol that is a
//!   prefix of another (`read` and `read file`) stays matchable.
//! - `build` returns `None` rather than a trie that allows nothing.
//!
//! Nothing outside this module uses it.

use std::collections::HashMap;

/// 2026-09-26: A token-level trie over a closed set of symbol strings.
#[derive(Debug, Clone)]
pub struct SymbolTrie {
    nodes: Vec<TrieNode>,
}

#[derive(Debug, Clone)]
struct TrieNode {
    /// 2026-09-26: Children by next token ID.
    children: HashMap<u32, usize>,
    /// 2026-09-26: True if a complete symbol ends at this node.
    terminal: bool,
}

/// 2026-09-26: The tokeniser the caller supplies to `build`.
pub trait Tokeniser {
    /// 2026-09-26: Encode `s` to a sequence of token IDs. This must match the model's
    /// tokeniser exactly (BPE merges, BOS/EOS rules): a mismatch lets the model
    /// emit allowed tokens that do not decode to a symbol in the set.
    fn encode(&self, s: &str) -> Vec<u32>;
}

impl SymbolTrie {
    /// 2026-09-26: Build a trie from the given symbol strings. Returns `None` when the
    /// list is empty or every symbol encodes to no tokens; a symbol that encodes
    /// to no tokens is skipped.
    pub fn build<T: Tokeniser>(symbols: &[&str], tokeniser: &T) -> Option<Self> {
        if symbols.is_empty() {
            return None;
        }
        let mut nodes = vec![TrieNode {
            children: HashMap::new(),
            terminal: false,
        }];
        for sym in symbols {
            let toks = tokeniser.encode(sym);
            if toks.is_empty() {
                continue;
            }
            let mut cur = 0usize;
            for t in &toks {
                let next = nodes[cur].children.get(t).copied();
                cur = match next {
                    Some(idx) => idx,
                    None => {
                        let new_idx = nodes.len();
                        nodes.push(TrieNode {
                            children: HashMap::new(),
                            terminal: false,
                        });
                        nodes[cur].children.insert(*t, new_idx);
                        new_idx
                    }
                };
            }
            nodes[cur].terminal = true;
        }
        if nodes.len() == 1 {
            // 2026-09-26: Every symbol encoded to no tokens.
            return None;
        }
        Some(Self { nodes })
    }

    /// 2026-09-26: Return `Some(node_id)` after walking `prefix_tokens` from the root,
    /// or `None` once the prefix leaves the trie.
    pub fn walk_prefix(&self, prefix_tokens: &[u32]) -> Option<usize> {
        let mut cur = 0usize;
        for &t in prefix_tokens {
            cur = *self.nodes[cur].children.get(&t)?;
        }
        Some(cur)
    }

    /// 2026-09-26: The tokens allowed next at `node_id`, sorted ascending and without
    /// duplicates; empty at a leaf. Panics if `node_id` is not a node.
    pub fn allowed_next_tokens(&self, node_id: usize) -> Vec<u32> {
        let node = &self.nodes[node_id];
        let mut out: Vec<u32> = node.children.keys().copied().collect();
        out.sort_unstable();
        out
    }

    /// 2026-09-26: Whether a complete symbol ends at `node_id`. Panics if `node_id` is
    /// not a node.
    pub fn is_terminal(&self, node_id: usize) -> bool {
        self.nodes[node_id].terminal
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-09-26: Test tokeniser: each whitespace-separated word maps to one nonzero
    /// id hashed from its bytes, so equal words share an id.
    struct WordTokeniser;
    impl Tokeniser for WordTokeniser {
        fn encode(&self, s: &str) -> Vec<u32> {
            s.split_whitespace()
                .map(|w| {
                    let mut h: u32 = 1;
                    for b in w.bytes() {
                        h = h.wrapping_mul(31).wrapping_add(b as u32);
                    }
                    h.max(1)
                })
                .collect()
        }
    }

    #[test]
    fn empty_symbols_returns_none() {
        let trie = SymbolTrie::build(&[], &WordTokeniser);
        assert!(trie.is_none());
    }

    #[test]
    fn single_symbol_walks_to_terminal() {
        let trie = SymbolTrie::build(&["foo bar baz"], &WordTokeniser).unwrap();
        let toks = WordTokeniser.encode("foo bar baz");
        let node = trie.walk_prefix(&toks).expect("walk hits terminal");
        assert!(trie.is_terminal(node));
        let next = trie.allowed_next_tokens(node);
        assert!(next.is_empty(), "no extensions past terminal");
    }

    #[test]
    fn two_symbols_share_prefix() {
        let trie = SymbolTrie::build(&["read file", "read dir"], &WordTokeniser).unwrap();
        let read_toks = WordTokeniser.encode("read");
        let read_node = trie.walk_prefix(&read_toks).expect("walk hits 'read'");
        // 2026-09-26: Not terminal: both symbols extend.
        assert!(!trie.is_terminal(read_node));
        let allowed = trie.allowed_next_tokens(read_node);
        assert_eq!(allowed.len(), 2, "two extensions: file + dir");
    }

    #[test]
    fn prefix_overlap_keeps_terminal_at_inner_node() {
        // 2026-09-26: "read" is a symbol and also a prefix of "read file".
        let trie = SymbolTrie::build(&["read", "read file"], &WordTokeniser).unwrap();
        let read_toks = WordTokeniser.encode("read");
        let read_node = trie.walk_prefix(&read_toks).expect("walk hits 'read'");
        assert!(trie.is_terminal(read_node), "'read' alone is terminal");
        let next = trie.allowed_next_tokens(read_node);
        assert_eq!(next.len(), 1, "only 'file' extends");
    }

    #[test]
    fn off_trie_prefix_returns_none() {
        let trie = SymbolTrie::build(&["foo bar"], &WordTokeniser).unwrap();
        let bogus = WordTokeniser.encode("foo wrong");
        let res = trie.walk_prefix(&bogus);
        assert!(res.is_none(), "off-trie prefix has no node");
    }

    #[test]
    fn allowed_next_tokens_is_sorted() {
        let trie = SymbolTrie::build(&["zebra", "alpha", "delta"], &WordTokeniser).unwrap();
        let root_allowed = trie.allowed_next_tokens(0);
        for window in root_allowed.windows(2) {
            assert!(window[0] < window[1], "tokens must be sorted ascending");
        }
    }
}
