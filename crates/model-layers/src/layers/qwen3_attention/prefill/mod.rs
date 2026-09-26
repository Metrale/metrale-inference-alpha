// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Full-attention prefill: Q/K/V projection, attention and O
//! projection, one file per path (cache-skip, paged, MLA, DeepSeek-V4) and per
//! attention-kernel family.
//!
//! Owner: model-layers (attention).
//! Invariants: none beyond the types.

// 2026-09-25: Tests that the cache-skip Q/K/V chain and the paged O
// projection allocate no device memory with the cuBLAS `attn` scope armed.
#[cfg(test)]
mod alloc_tests;
mod cache_skip;
mod cache_skip_attn_gates;
mod cache_skip_mla;
mod cache_skip_norm_rope;
mod cache_skip_qkv;
mod cache_skip_v4;
mod cache_skip_v4_attn;
mod cache_skip_v4_q_rope;
mod paged;
mod paged_attn;
mod paged_attn_batched;
mod paged_attn_fp8k;
mod paged_attn_turbok;
mod paged_mla;
mod paged_oproj;
mod paged_qkv;
mod paged_v4;
