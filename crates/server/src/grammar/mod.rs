// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Grammar-constrained decoding on top of `metrale_grammar`.
//!
//! - [`GrammarEngine`] is built from the tokenizer on each model load (at
//!   startup and on a model swap) and compiles grammars from tool
//!   definitions, JSON schemas and EBNF.
//! - [`GrammarState`] is per request: it wraps a
//!   [`metrale_grammar::GrammarMatcher`], fills the token bitmask, accepts
//!   tokens and rolls back rejected speculative drafts.
//!
//! Owner: server (grammar).
//! Invariants: none beyond the types.

mod compile_misc;
mod compile_tools;
mod engine;
mod mask_cache;
mod prewarm;
mod schema;
mod state;

#[cfg(test)]
pub(crate) mod tests;

pub use engine::{GrammarEngine, GrammarError};
pub use mask_cache::PrewarmHook;
pub use schema::augment_schema_with_tafc_think;
pub use state::{GrammarState, grammar_blocks_stop};

/// 2026-09-26: The tokenizer's vocabulary (added tokens included), indexed
/// by token id: `vocab[i]` is the string of token `i`. The length is the
/// larger of the entry count and the highest id + 1; ids with no entry are
/// empty strings.
pub fn extract_ordered_vocab(tokenizer: &tokenizers::Tokenizer) -> Vec<String> {
    let vocab = tokenizer.get_vocab(true);
    let max_id = vocab.values().copied().max().unwrap_or(0) as usize;
    let size = vocab.len().max(max_id + 1);
    let mut ordered = vec![String::new(); size];
    for (token, id) in vocab {
        let idx = id as usize;
        if idx < size {
            ordered[idx] = token;
        }
    }
    ordered
}
