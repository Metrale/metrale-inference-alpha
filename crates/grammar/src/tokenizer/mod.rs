// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tokenizer metadata for the grammar matcher: `TokenizerInfo` (the decoded
//! vocabulary), `VocabType` (`Raw`, `ByteFallback`, `ByteLevel`), `decode_token` and the
//! byte-level character maps, and `detect_metadata_from_hf`, which reads the vocabulary
//! type and prefix-space flag from the text of a `tokenizer.json`.
//!
//! Owner: grammar.
//! Invariants: none beyond the types.

pub mod decoder;
pub mod hf_metadata;
pub mod info;
pub mod vocab_type;

pub use decoder::{byte_to_char_map, char_to_byte_map, decode_token};
pub use hf_metadata::{HfMetadata, detect_metadata_from_hf};
pub use info::TokenizerInfo;
pub use vocab_type::VocabType;
