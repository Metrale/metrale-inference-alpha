// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Grammar compiler: turns a grammar, a JSON schema or a structural tag,
//! together with a tokenizer, into a `CompiledGrammar`, which holds the grammar and
//! computes, per parser state, the token masks the matcher uses to fill the logit bitmask.
//!
//! `GrammarCompiler` caches compiled grammars in `GrammarCache`, and, when caching is
//! enabled, shares one `RuleLevelCache` of per-rule masks across every grammar it builds.
//!
//! Owner: grammar.
//! Invariants: none beyond the types.

mod coalesce;
mod compile;
mod compiled_grammar;
mod compiler;
mod decompose;
mod grammar_cache;
mod mask;
mod mask_gen;
pub mod mask_snapshot;
mod prewarm;
mod rule_cache;

pub use coalesce::{Forced, analyze_bitmask};
pub use compiled_grammar::{CompiledGrammar, CompiledGrammarImpl};
pub use compiler::{CompileError, GrammarCompiler};
pub use decompose::{GrammarDecomposition, RuleDecomposition, Segment, decompose_static_regions};
pub use mask::{AdaptiveTokenMask, StoreType, USE_BITSET_THRESHOLD};
pub use mask_snapshot::{SnapshotError, SnapshotIdentity};
pub use rule_cache::{RuleLevelCache, RuleMaskKey, UNLIMITED_SIZE};

#[cfg(test)]
mod tests;
