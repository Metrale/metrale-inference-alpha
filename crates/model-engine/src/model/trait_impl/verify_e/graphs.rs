// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The batched verify's CUDA-graph cache steps: the per-step replay, borrow or
//! capture decision, and the insert (with LRU eviction) after a capture.
//!
//! Owner: model-engine (speculative verify).
//! Invariants:
//! - Both run with the `verify_batched_graphs` lock held by the caller.

use std::collections::HashMap;

use metrale_gpu_runtime::gpu::{DevicePtr, GraphHandle};

use super::super::super::types::TransformerModel;

/// 2026-09-26: The locked batched-verify graph cache: `(key -> (graph, last_use_tick), tick)`.
pub(super) type VerifyGraphs<'a> =
    parking_lot::MutexGuard<'a, (HashMap<Vec<u32>, (GraphHandle, u64)>, u64)>;

impl TransformerModel {
    /// 2026-09-26: Returns `(replay, ghosts, outcome)`: the graph to replay (exact or
    /// borrowed key), the borrowed key's ghost `(slot, k)` pairs, and what was decided.
    pub(super) fn pick_verify_graph(
        &self,
        graphs: &mut Option<VerifyGraphs<'_>>,
        graph_key: &Option<Vec<u32>>,
        wy_tables_base: DevicePtr,
        r_total: usize,
        n: usize,
    ) -> (
        Option<GraphHandle>,
        Vec<(u32, u32)>,
        super::super::verify_e2::VerifyGraphOutcome,
    ) {
        // 2026-09-25: LRU touch on hit: bump the tick so eviction removes the
        // least recently replayed key.
        let mut replay: Option<metrale_gpu_runtime::gpu::GraphHandle> = None;
        let mut ghosts: Vec<(u32, u32)> = Vec::new();
        // 2026-09-25: Reported to `record_verify_graph_outcome`. Stays `Eager`
        // unless a replay, borrow or capture happens.
        let mut outcome = super::super::verify_e2::VerifyGraphOutcome::Eager;
        if let (Some(g), Some(key)) = (graphs, graph_key) {
            g.1 += 1;
            let tick = g.1;
            if let Some(e) = g.0.get_mut(key) {
                e.1 = tick;
                replay = Some(e.0);
                outcome = super::super::verify_e2::VerifyGraphOutcome::Replay;
            } else if super::super::graph_borrow::graph_borrow_enabled() {
                let wy_present = !wy_tables_base.is_null();
                let borrowed = super::super::graph_borrow::find_borrowable_verify_key(
                    key,
                    g.0.keys(),
                    |s, k| {
                        self.ssm_pool.slot_is_free(s as usize)
                            && (!wy_present
                                || self.ssm_pool.h_inter_count(s as usize) + 1 >= k as usize)
                    },
                );
                // 2026-09-25: The borrowed row total is re-checked against
                // VERIFY_ROW_CAP because it bounds the `unsafe` metadata
                // uploads in `stage_verify_metadata`. `VERIFY_BORROW_STREAK` declines a key that
                // keeps borrowing, so it gets captured (`borrow_streak.rs`).
                if let Some(b) = borrowed
                    && r_total + b.ghosts.iter().map(|&(_, k)| k as usize).sum::<usize>()
                        <= super::super::verify_e2::VERIFY_ROW_CAP
                    && super::super::borrow_streak::VERIFY_BORROW_STREAK.allow(key)
                {
                    let e =
                        g.0.get_mut(&b.key)
                            .expect("borrowed key comes from this cache");
                    e.1 = tick;
                    replay = Some(e.0);
                    ghosts = b.ghosts;
                    outcome = super::super::verify_e2::VerifyGraphOutcome::Borrow;
                    // 2026-09-25: Logged only when the (exact, borrowed) key
                    // pair differs from the last one logged.
                    if super::super::graph_borrow::VERIFY_BORROW_LOG.should_log(key, &b.key) {
                        tracing::info!(target: "metrale_model_engine::model::trait_impl::verify_e", "verify graph borrow: n={n} R={r_total} -> replaying captured \
                             {}-seq key with {} ghost pairs",
                            (b.key.len() - 1) / 2,
                            ghosts.len()
                        );
                    }
                }
            }
        }
        (replay, ghosts, outcome)
    }

    /// 2026-09-26: End the capture and, when a graph was captured, insert it under
    /// `graph_key` (evicting the least recently used entry at the cap) and launch it.
    pub(super) fn finish_verify_capture(
        &self,
        graphs: &mut Option<VerifyGraphs<'_>>,
        graph_key: Option<Vec<u32>>,
        outcome: &mut super::super::verify_e2::VerifyGraphOutcome,
        ks: &[usize],
        n: usize,
        stream: u64,
    ) -> anyhow::Result<()> {
        let graph = self.gpu.end_capture(stream)?;
        if graph.0 != 0 {
            tracing::info!(target: "metrale_model_engine::model::trait_impl::verify_e", "Captured CUDA graph for batched verify ks={ks:?} (n={n}, key={:?})",
                graph_key
            );
            if let (Some(ref mut g), Some(key)) = (graphs.as_mut(), graph_key) {
                if g.0.len() >= super::super::verify_e2::VERIFY_BATCHED_GRAPH_CAP {
                    // 2026-09-25: Evict the least recently used
                    // graph. A call that returned `Ok` waited for this
                    // stream before returning, so a graph last
                    // replayed by such a call is no longer running.
                    if let Some(evict) =
                        g.0.iter()
                            .min_by_key(|(_, entry)| entry.1)
                            .map(|(key, _)| key.clone())
                        && let Some((old, _)) = g.0.remove(&evict)
                        && let Err(e) = self.gpu.destroy_graph(old)
                    {
                        tracing::warn!(target: "metrale_model_engine::model::trait_impl::verify_e", "batched-verify graph evict: {e:#}");
                    }
                }
                g.1 += 1;
                let tick = g.1;
                g.0.insert(key, (graph, tick));
                *outcome = super::super::verify_e2::VerifyGraphOutcome::Capture;
            }
            self.gpu.launch_graph(graph, stream)?;
        }
        Ok(())
    }
}
