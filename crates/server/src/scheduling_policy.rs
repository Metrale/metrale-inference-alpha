// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: the scheduling policy trait and its FIFO and SLAI
//! implementations.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types. A policy decides whether new prefills
//! run this tick, which pending requests are prefilled and in what order,
//! and how many prefill tokens a fused decode step carries. It performs no
//! I/O: the caller passes the clock reading.

use std::time::{Duration, Instant};

/// 2026-09-25: metadata about a pending request for selection decisions.
pub struct PendingRequestInfo {
    /// 2026-09-25: number of prompt tokens.
    pub prompt_len: usize,
    /// 2026-09-25: position in the pending queue.
    pub index: usize,
}

/// 2026-09-25: per-sequence timing for decode urgency decisions.
pub struct ActiveSeqTiming {
    /// 2026-09-25: when this sequence last emitted a token.
    pub last_token_time: Instant,
}

pub trait SchedulingPolicy: Send {
    /// 2026-09-25: whether new prefills may run this tick; `false` sends the
    /// scheduler straight to decode.
    fn should_prefill(&self, now: Instant, active_timings: &[ActiveSeqTiming]) -> bool;

    /// 2026-09-25: up to `capacity` indices into `requests`, in the order to
    /// prefill them.
    fn select_prefills(&self, requests: &[PendingRequestInfo], capacity: usize) -> Vec<usize>;

    /// 2026-09-25: prefill tokens to fuse into this tick's decode step on the
    /// mixed path, in `[0, full_chunk]`. 0 makes the caller skip prefill
    /// this tick. The default returns `full_chunk`.
    fn prefill_slice_budget(
        &self,
        now: Instant,
        active_timings: &[ActiveSeqTiming],
        full_chunk: usize,
    ) -> usize {
        let _ = (now, active_timings);
        full_chunk
    }

    fn name(&self) -> &str;
}

/// 2026-09-25: always prefills, and takes the first `capacity` requests in
/// queue order.
pub struct FifoPolicy;

impl SchedulingPolicy for FifoPolicy {
    fn should_prefill(&self, _now: Instant, _active_timings: &[ActiveSeqTiming]) -> bool {
        true
    }

    fn select_prefills(&self, requests: &[PendingRequestInfo], capacity: usize) -> Vec<usize> {
        (0..requests.len().min(capacity)).collect()
    }

    fn name(&self) -> &str {
        "fifo"
    }
}

/// 2026-09-25: deadline-aware scheduling (SLAI).
///
/// - Skips prefills while any active sequence has waited at least 80% of
///   `tbt_deadline` since its last token.
/// - Prefills the front of the queue, then the shortest prompts.
pub struct SlaiPolicy {
    tbt_deadline: Duration,
}

impl SlaiPolicy {
    pub fn new(tbt_deadline_ms: u64) -> Self {
        Self {
            tbt_deadline: Duration::from_millis(tbt_deadline_ms),
        }
    }
}

impl SchedulingPolicy for SlaiPolicy {
    fn should_prefill(&self, now: Instant, active_timings: &[ActiveSeqTiming]) -> bool {
        if active_timings.is_empty() {
            return true;
        }
        let margin = self.tbt_deadline.mul_f64(0.8);
        for timing in active_timings {
            if now.duration_since(timing.last_token_time) >= margin {
                return false;
            }
        }
        true
    }

    /// 2026-09-25: seat 0 goes to `requests[0]`, the front of the pending
    /// queue; the other `capacity - 1` seats go to the shortest of the rest.
    ///
    /// Shortest-first alone has no wait-time term: the list is rebuilt from
    /// the queue every tick, so a long prompt can lose to shorter arrivals
    /// on every tick. Reserving seat 0 means the front of the queue is
    /// always selected. New requests join at the back; requests that
    /// admission defers go back to the front, in their selection order
    /// (`admission.rs`).
    fn select_prefills(&self, requests: &[PendingRequestInfo], capacity: usize) -> Vec<usize> {
        if capacity == 0 || requests.is_empty() {
            return Vec::new();
        }
        let mut indices: Vec<usize> = vec![0];
        let mut rest: Vec<usize> = (1..requests.len()).collect();
        rest.sort_by_key(|&i| requests[i].prompt_len);
        rest.truncate(capacity - 1);
        indices.extend(rest);
        indices
    }

    fn prefill_slice_budget(
        &self,
        now: Instant,
        active_timings: &[ActiveSeqTiming],
        full_chunk: usize,
    ) -> usize {
        if active_timings.is_empty() {
            return full_chunk;
        }

        // 2026-09-25: a decode already past `tbt_deadline` gets a
        // decode-only tick.
        let worst = active_timings
            .iter()
            .map(|t| now.duration_since(t.last_token_time))
            .max()
            .unwrap_or_default();
        if worst >= self.tbt_deadline {
            return 0;
        }

        // 2026-09-25: otherwise the full chunk, never a smaller slice.
        // Measured 2026-06-24 (varied-load burst A/B): the fused step has a
        // ~250ms full-forward floor, so a smaller slice did not lower decode
        // TBT and slowed prefill (slice=32: 4.6x slower prefill, no TBT
        // gain); full chunks halved the decode-freeze p99 (2529->1285ms).
        full_chunk
    }

    fn name(&self) -> &str {
        "slai"
    }
}

#[cfg(test)]
#[path = "scheduling_policy_tests.rs"]
mod tests;
