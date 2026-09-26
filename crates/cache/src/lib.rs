// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The paged KV block pool, host-side KV dequantization, the KV
//! spill-file manager and the radix-tree prefix cache.
//!
//! Owner: cache.
//! Invariants: none beyond the types.

pub mod kv_cache;
pub mod kv_dequant;
pub mod kv_spill;
pub mod radix_tree;
