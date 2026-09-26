// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Support code for the grammar crate: UTF-8, Latin-1 and hex decoding
//! (`encoding`), escape handling for grammar literals (`escape`), union and intersection
//! of sorted integer sets (`int_set`), the CSR `Compact2DArray`, `UnionFindSet`, and
//! hash-combine helpers (`hash`).
//!
//! Owner: grammar.
//! Invariants: none beyond the types.

pub mod compact_2d_array;
pub mod encoding;
pub mod escape;
pub mod hash;
pub mod int_set;
pub mod union_find;

pub use compact_2d_array::Compact2DArray;
pub use encoding::{
    TCodepoint, byte_to_latin1, char_handling_error, char_to_utf8, handle_utf8_first_byte,
    hex_char_to_int, latin1_to_bytes, parse_next_utf8, parse_utf8,
};
pub use escape::{
    parse_next_escaped, parse_next_utf8_or_escaped, print_as_escaped, print_byte_as_escaped,
    print_str_as_escaped, unescape_string,
};
pub use hash::{hash_combine, hash_combine_binary};
pub use int_set::{intset_intersection, intset_union};
pub use union_find::UnionFindSet;
