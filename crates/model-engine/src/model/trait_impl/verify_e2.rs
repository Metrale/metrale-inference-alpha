// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Support for the batched verify in `verify_e.rs`: WY pointer-table staging, its cache, the graph key and the verify env switches.
//!
//! Owner: model-engine (speculative verify).
//! Invariants:
//! - `upload_verify_wy_tables` sets `verify_wy_cache` only after the H2D of
//!   the tables that key describes was enqueued without error. Its `NULL`
//!   and cache-hit returns leave both the cache and the device tables
//!   unchanged.

#![allow(dead_code)]

use anyhow::Result;
use metrale_config::LayerType;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::super::types::TransformerModel;
use crate::traits::SequenceState;
use metrale_model_layers::layer::{
    SsmLayerState, VERIFY_WY_LAYER_STRIDE_BYTES, VERIFY_WY_TABLE_SEQS,
};

/// 2026-09-25: Most batched-verify graphs kept at once. At the cap,
/// `verify_e.rs` destroys the least recently used graph before inserting a new
/// one, so the cap bounds graph memory without making the path eager.
pub(super) const VERIFY_BATCHED_GRAPH_CAP: usize = 32;

/// 2026-09-25: Most rows (`R = Σ ks`) one batched verify may run. The
/// `VMETA_*` metadata offsets in `verify_e.rs` are derived from it. Three
/// other values repeat 160 and must change with it: `VERIFY_ROW_BUDGET` in
/// the scheduler's `mtp_dcut.rs`, and `bt_rows` and the `logits_tokens` floor
/// in `gpu-runtime`'s `sizes.rs`. The sequence count is bounded separately,
/// by `VERIFY_WY_TABLE_SEQS`.
pub(in crate::model) const VERIFY_ROW_CAP: usize = 160;

/// 2026-09-25: Batched-verify CUDA graphs are on unless
/// `METRALE_NO_MTP_VERIFY_GRAPHS` is present, whatever its value. Read once
/// per process.
pub(super) fn verify_graphs_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("METRALE_NO_MTP_VERIFY_GRAPHS").is_none())
}

/// 2026-09-25: A value switch is armed by the exact value `1` and nothing
/// else; `0`, `true`, an empty value and absence leave it off. Used by
/// `k4_diag_enabled`, `verify_d2h_default_stream` and `verify_d2h_no_pinned`.
fn value_switch_armed(raw: Option<&str>) -> bool {
    raw == Some("1")
}

fn read_value_switch(name: &str) -> bool {
    value_switch_armed(std::env::var(name).ok().as_deref())
}

/// 2026-09-25: `METRALE_K4_DIAG=1`: synchronise after every layer of the
/// batched verify. Read once per process.
pub(super) fn k4_diag_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| read_value_switch("METRALE_K4_DIAG"))
}

/// 2026-09-25: `METRALE_VERIFY_D2H_DEFAULT_STREAM=1`: read the batched-verify
/// argmax with `copy_d2h`. Read once per process.
pub(super) fn verify_d2h_default_stream() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| read_value_switch("METRALE_VERIFY_D2H_DEFAULT_STREAM"))
}

/// 2026-09-25: `METRALE_NO_PINNED_VERIFY_D2H=1`: read the batched-verify argmax
/// with an on-stream copy into pageable memory. Read once per process.
pub(super) fn verify_d2h_no_pinned() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| read_value_switch("METRALE_NO_PINNED_VERIFY_D2H"))
}

/// 2026-09-25: The WY-table staging cache is on unless
/// `METRALE_NO_VERIFY_WY_CACHE` is present, whatever its value. Off, every call
/// of `upload_verify_wy_tables` rebuilds and uploads the tables. Read once per
/// process.
pub(super) fn verify_wy_cache_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("METRALE_NO_VERIFY_WY_CACHE").is_none())
}

/// 2026-09-25: Encode every input the tables staged by
/// `upload_verify_wy_tables` depend on, so equal keys mean identical tables.
///
/// Every entry the fill loops write is a pool address: a batch sequence's
/// `SsmLayerState::h_state` is set only to `ssm_pool.h_state(layer, slot)`
/// (`meta.rs` at allocation, `sequence_compact.rs` at compaction) or to 0 on
/// free (`sequence.rs`), and its `h_state_intermediates[t]` only to
/// `ssm_pool.h_intermediate(layer, slot, t)`; a ghost's entries are read from
/// the same two accessors. `ssm_pool.h_state` is `h_state_pools[layer]
/// .offset(slot * h_stored_bytes)`, fixed from model construction. So an entry
/// is a function of `(layer, slot, t)`, the layer set is fixed, and the inputs
/// that vary per step are the ones encoded here:
///   1. `k`: how many tables per layer are filled.
///   2. `slots.len()`: how many batch entries are filled.
///   3. `slots`: each sequence's SSM pool slot, in batch order (entry `i` of
///      every table is sequence `i`).
///   4. `ghosts`: the borrow's `(slot, depth)` pairs, in order, after the
///      batch entries.
///
/// The encoding `[k, n, slots[0..n], (slot, depth) * g]` is injective: `n`
/// separates the slot run from the ghost pairs, and the ghost count follows
/// from the remaining length.
pub(super) fn verify_wy_cache_key(slots: &[u32], k: usize, ghosts: &[(u32, u32)]) -> Vec<u64> {
    let mut key = Vec::with_capacity(2 + slots.len() + 2 * ghosts.len());
    key.push(k as u64);
    key.push(slots.len() as u64);
    key.extend(slots.iter().map(|&s| s as u64));
    for &(slot, depth) in ghosts {
        key.push(slot as u64);
        key.push(depth as u64);
    }
    key
}

/// 2026-09-25: What the batched verify did with its CUDA graph on one step.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum VerifyGraphOutcome {
    /// 2026-09-25: Exact key hit: the forward replayed.
    Replay,
    /// 2026-09-25: A wider captured key replayed with ghost rows.
    Borrow,
    /// 2026-09-25: Miss: the forward ran under capture and the new graph was
    /// inserted.
    Capture,
    /// 2026-09-25: No graph ran: `METRALE_NO_MTP_VERIFY_GRAPHS`,
    /// `METRALE_K4_DIAG`, a batch with a sequence that has no SSM slot, or a
    /// capture that produced no graph.
    Eager,
}

/// 2026-09-25: Count batched-verify graph outcomes and log one INFO summary
/// per 200 steps: the count of each outcome, the capture fraction and the
/// live key count. Does nothing unless `METRALE_MTP_ACCEPT_DEBUG` is present,
/// which is checked before any counter is touched.
pub(super) fn record_verify_graph_outcome(n: usize, live_keys: usize, outcome: VerifyGraphOutcome) {
    use std::sync::atomic::{AtomicU64, Ordering};
    const PERIOD: u64 = 200;
    static STEPS: AtomicU64 = AtomicU64::new(0);
    static REPLAY: AtomicU64 = AtomicU64::new(0);
    static BORROW: AtomicU64 = AtomicU64::new(0);
    static CAPTURE: AtomicU64 = AtomicU64::new(0);
    static EAGER: AtomicU64 = AtomicU64::new(0);
    if !metrale_model_layers::speculative::mtp_accept_debug() {
        return;
    }
    match outcome {
        VerifyGraphOutcome::Replay => &REPLAY,
        VerifyGraphOutcome::Borrow => &BORROW,
        VerifyGraphOutcome::Capture => &CAPTURE,
        VerifyGraphOutcome::Eager => &EAGER,
    }
    .fetch_add(1, Ordering::Relaxed);
    if STEPS.fetch_add(1, Ordering::Relaxed) + 1 >= PERIOD {
        let steps = STEPS.swap(0, Ordering::Relaxed).max(1);
        let (replay, borrow) = (
            REPLAY.swap(0, Ordering::Relaxed),
            BORROW.swap(0, Ordering::Relaxed),
        );
        let (capture, eager) = (
            CAPTURE.swap(0, Ordering::Relaxed),
            EAGER.swap(0, Ordering::Relaxed),
        );
        tracing::info!(
            "batched-verify graphs [{steps} steps, last n={n}]: replay={replay} \
             borrow={borrow} CAPTURE={capture} eager={eager} capture_frac={:.3} \
             live_keys={live_keys}/{}",
            capture as f64 / steps as f64,
            VERIFY_BATCHED_GRAPH_CAP,
        );
    }
}

impl TransformerModel {
    /// 2026-09-25: The batched-verify graph key: each sequence's
    /// `(SSM pool slot, row count)` in batch order, then one sentinel word for
    /// the WY-tables-present and write-on-accept bits
    /// (`speculative::verify_key::verify_graph_key`). The SSM pointers a graph
    /// bakes are functions of the slots, and its launch shapes of the row
    /// counts. `None`, meaning no graph, when a sequence has no SSM slot.
    pub(super) fn verify_batched_graph_key(
        &self,
        seqs: &[&mut SequenceState],
        ks: &[usize],
        wy_tables_null: bool,
        write_on_accept: bool,
    ) -> Option<Vec<u32>> {
        let mut pairs: Vec<(u32, u32)> = Vec::with_capacity(seqs.len());
        for (i, s) in seqs.iter().enumerate() {
            pairs.push((s.ssm_slot_idx()? as u32, *ks.get(i)? as u32));
        }
        Some(
            metrale_model_layers::speculative::verify_key::verify_graph_key(
                &pairs,
                wy_tables_null,
                write_on_accept,
            ),
        )
    }

    /// 2026-09-25: Stage the WY pointer tables for this batch into the fixed
    /// `verify_wy_tables` buffer and return its address. Per GDN layer there
    /// are `VERIFY_WY_TABLES_PER_LAYER` tables of `VERIFY_WY_TABLE_SEQS` u64
    /// entries; table 0 holds each sequence's `h_state` and table `t + 1` its
    /// `h_state_intermediates[t]`, for the first `k` tables. Unfilled entries
    /// are zero. Table strides do not depend on `k`.
    ///
    /// `ghosts` are the borrowed graph's extra `(slot, k)` pairs
    /// (`graph_borrow.rs`), appended after the batch entries and read from the
    /// SSM pool; empty when not borrowing.
    ///
    /// When `verify_wy_cache_key` matches `verify_wy_cache`, the device buffer
    /// already holds these bytes: allocation zeroes it and this function is
    /// its only writer. Both the build and the H2D are then skipped.
    ///
    /// Returns NULL, uploading nothing, when the buffer is NULL, when
    /// `n + ghosts.len() > VERIFY_WY_TABLE_SEQS`, when `k` is outside
    /// 2..=`VERIFY_WY_TABLES_PER_LAYER`, when the model has no SSM layers, or
    /// when some sequence's GDN layer state is not an `SsmLayerState` or lacks
    /// an `h_state` or `k - 1` intermediates. The per-layer batched GDN arm re-checks the
    /// intermediate count before it reads a table
    /// (`trait_decode_batched_conv_gdn_multi.rs`).
    pub(super) fn upload_verify_wy_tables(
        &self,
        seqs: &[&mut SequenceState],
        k: usize,
        ghosts: &[(u32, u32)],
        stream: u64,
    ) -> Result<DevicePtr> {
        let n = seqs.len();
        if self.verify_wy_tables.is_null()
            || n + ghosts.len() > VERIFY_WY_TABLE_SEQS
            || !(2..=metrale_model_layers::layer::VERIFY_WY_TABLES_PER_LAYER).contains(&k)
        {
            return Ok(DevicePtr::NULL);
        }
        let num_ssm = self.config.num_ssm_layers();
        if num_ssm == 0 {
            return Ok(DevicePtr::NULL);
        }
        // 2026-09-25: A batch with a sequence that has no SSM slot has no
        // key; it is staged without the cache.
        let cache_key: Option<Vec<u64>> = if verify_wy_cache_enabled() {
            seqs.iter()
                .map(|s| s.ssm_slot_idx().map(|v| v as u32))
                .collect::<Option<Vec<u32>>>()
                .map(|slots| verify_wy_cache_key(&slots, k, ghosts))
        } else {
            None
        };
        if let Some(key) = cache_key.as_deref()
            && self.verify_wy_cache.lock().as_deref() == Some(key)
        {
            return Ok(self.verify_wy_tables);
        }
        let entries_per_layer = VERIFY_WY_LAYER_STRIDE_BYTES / 8;
        let mut host = vec![0u64; num_ssm * entries_per_layer];
        let mut ssm_idx = 0usize;
        for layer_idx in 0..self.layers.len() {
            if self.config.layer_type(layer_idx) != LayerType::LinearAttention {
                continue;
            }
            let base = ssm_idx * entries_per_layer;
            for (i, seq) in seqs.iter().enumerate() {
                let Some(st) = seq.layer_states[layer_idx]
                    .as_any()
                    .downcast_ref::<SsmLayerState>()
                else {
                    return Ok(DevicePtr::NULL);
                };
                if st.h_state.is_null() || st.h_state_intermediates.len() < k - 1 {
                    return Ok(DevicePtr::NULL);
                }
                host[base + i] = st.h_state.0;
                for t in 0..k - 1 {
                    host[base + (t + 1) * VERIFY_WY_TABLE_SEQS + i] = st.h_state_intermediates[t].0;
                }
            }
            for (gi, &(slot, gk)) in ghosts.iter().enumerate() {
                let i = n + gi;
                let s = slot as usize;
                host[base + i] = self.ssm_pool.h_state(ssm_idx, s).0;
                for t in 0..(gk as usize).saturating_sub(1) {
                    host[base + (t + 1) * VERIFY_WY_TABLE_SEQS + i] =
                        self.ssm_pool.h_intermediate(ssm_idx, s, t).0;
                }
            }
            ssm_idx += 1;
        }
        // 2026-09-25: SAFETY: `host.len() * 8 == size_of_val(&host[..])`, so
        // the slice stops at the vector's length. `host` is
        // `vec![0u64; num_ssm * entries_per_layer]`, so the entries the fill
        // loops skip are zeros. `u64` is POD. `copy_h2d_async` lets the source
        // drop once it returns.
        let bytes =
            unsafe { std::slice::from_raw_parts(host.as_ptr() as *const u8, host.len() * 8) };
        self.gpu
            .copy_h2d_async(bytes, self.verify_wy_tables, stream)?;
        // 2026-09-25: Recorded only after the copy is enqueued. Every early
        // return above leaves both the device buffer and the cache as the
        // last successful stage left them.
        if let Some(key) = cache_key {
            *self.verify_wy_cache.lock() = Some(key);
        }
        Ok(self.verify_wy_tables)
    }
}

#[cfg(test)]
#[path = "verify_e2_tests.rs"]
mod tests;
