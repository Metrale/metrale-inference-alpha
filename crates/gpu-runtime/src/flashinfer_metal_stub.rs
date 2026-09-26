// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The `flashinfer` module of a build without the `cuda` feature. It
//! keeps the entry points metrale-model-layers calls without a `cfg`, so that
//! build compiles.
//!
//! Owner: gpu-runtime.
//! Invariants:
//! - `available()` is false, and the only caller checks it before a prefill
//!   call, so the `unreachable!` prefill bodies are not reached.

use anyhow::Result;

/// 2026-09-25: Always false in a build without the `cuda` feature.
pub fn available() -> bool {
    false
}

#[allow(clippy::too_many_arguments)]
pub fn ragged_prefill_bf16_hd256(
    _q: u64,
    _k: u64,
    _v: u64,
    _o: u64,
    _qo_indptr_h: &[i32],
    _kv_indptr_h: &[i32],
    _qo_indptr_d: u64,
    _kv_indptr_d: u64,
    _batch: u32,
    _total_qo_rows: u32,
    _total_kv_rows: u32,
    _num_qo_heads: u32,
    _num_kv_heads: u32,
    _head_dim: u32,
    _sm_scale: f32,
    _causal: bool,
    _stream: u64,
) -> Result<()> {
    unreachable!("flashinfer::ragged_prefill_bf16_hd256 is cuda-only (not built for metal)")
}

#[allow(clippy::too_many_arguments)]
pub fn ragged_prefill_bf16_hd128(
    _q: u64,
    _k: u64,
    _v: u64,
    _o: u64,
    _qo_indptr_h: &[i32],
    _kv_indptr_h: &[i32],
    _qo_indptr_d: u64,
    _kv_indptr_d: u64,
    _batch: u32,
    _total_qo_rows: u32,
    _total_kv_rows: u32,
    _num_qo_heads: u32,
    _num_kv_heads: u32,
    _head_dim: u32,
    _sm_scale: f32,
    _causal: bool,
    _sliding_window: Option<u32>,
    _stream: u64,
) -> Result<()> {
    unreachable!("flashinfer::ragged_prefill_bf16_hd128 is cuda-only (not built for metal)")
}
