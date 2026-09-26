// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The grammar representation and its front end: the expression and rule
//! data (`GrammarExpr`, `GrammarData`), `GrammarBuilder`, the EBNF lexer and parser
//! (`parse_ebnf`), the normalising and optimising passes (`functor`), and the EBNF
//! printer.
//!
//! Owner: grammar.
//! Invariants: none beyond the types.

pub mod builder;
pub mod data;
pub mod expr;
pub mod functor;
pub mod lexer;
pub mod parser;
pub mod printer;

pub use builder::GrammarBuilder;
pub use data::{GrammarData, Rule, TagDispatch};
pub use expr::{GrammarExpr, GrammarExprType};
pub use functor::{
    GrammarConcat, GrammarFsmBuilder, GrammarFsmHasher, GrammarNormalizer, GrammarOptimizer,
    GrammarUnion,
};
pub use parser::{ParseError, parse_ebnf, parse_ebnf_default};
pub use printer::{GrammarPrinter, print_grammar};
