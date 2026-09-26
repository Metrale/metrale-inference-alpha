// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The run mailbox: process-global counters read by the
//! `/metrics` handler and the TUI, which hold no handle to the scheduler's
//! context, and [`reset_for_new_run`], which starts a new run's accounting.
//!
//! `MetraleCudaBackend::new` calls [`reset_for_new_run`] before its first
//! kernel lookup, so the kernel audit lists only that backend's lookups.
//!
//! Owner: telemetry.
//! Invariants: the prefix-cache counters never decrease; a run's own counts
//! are the difference from the snapshot `reset_for_new_run` takes.

use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{LazyLock, Mutex};

/// 2026-09-26: The process's run mailbox.
#[derive(Debug, Default)]
pub struct RunMetrics {
    pub cache_hits: AtomicU64,
    pub cache_misses: AtomicU64,
    pub cache_hit_tokens: AtomicU64,

    /// 2026-09-26: Most recent per-token entropy, as f32 bits.
    pub last_entropy: AtomicU32,
    pub low_entropy_tokens: AtomicU64,
    pub total_sampled_tokens: AtomicU64,

    /// 2026-09-26: The free-memory baseline `set_baseline_free_bytes` records
    /// after the backend is built. KV sizing uses it only when neither
    /// `METRALE_KV_EXTERNAL_RESERVE_GB` nor the allocation ledger applies: it
    /// measures this process's use as `baseline - free_now` and charges the
    /// smaller of that and the known footprint (`factory/build/kv_budget.rs`).
    /// `0` means unset.
    pub baseline_free_bytes: AtomicUsize,
    /// 2026-09-26: `(module, func, loaded, dispatch site)` for every lookup
    /// [`crate::kernel_audit::record`] received this run. The site is the
    /// caller's `Location`, carried through `#[track_caller]` on
    /// `GpuBackend::kernel`.
    pub kernel_audit: Mutex<Vec<(String, String, bool, &'static std::panic::Location<'static>)>>,

    // 2026-09-26: The prefix-cache counters at the start of this run.
    // `/metrics` exports the counters as `metrale_prefix_cache_*_total`, type
    // counter, so they are never reset; `cache_counts_this_run` subtracts
    // these instead.
    run_base_cache_hits: AtomicU64,
    run_base_cache_misses: AtomicU64,
    run_base_cache_hit_tokens: AtomicU64,
}

static METRICS: LazyLock<RunMetrics> = LazyLock::new(RunMetrics::default);

/// 2026-09-26: Read the mailbox.
pub fn metrics() -> &'static RunMetrics {
    &METRICS
}

/// 2026-09-26: Begin a new run's accounting: snapshot the prefix-cache
/// counters as this run's baseline, zero the entropy counters, the last
/// entropy and the free-memory baseline, clear the kernel audit (unless its
/// lock is poisoned) and unseal it.
pub fn reset_for_new_run() {
    let m = metrics();
    for (counter, base) in [
        (&m.cache_hits, &m.run_base_cache_hits),
        (&m.cache_misses, &m.run_base_cache_misses),
        (&m.cache_hit_tokens, &m.run_base_cache_hit_tokens),
    ] {
        base.store(counter.load(Ordering::Relaxed), Ordering::Relaxed);
    }
    for c in [&m.low_entropy_tokens, &m.total_sampled_tokens] {
        c.store(0, Ordering::Relaxed);
    }
    m.baseline_free_bytes.store(0, Ordering::Relaxed);
    m.last_entropy.store(0, Ordering::Relaxed);
    if let Ok(mut v) = m.kernel_audit.lock() {
        v.clear();
    }
    // 2026-09-26: The incoming model's lookups run before its own gate seals;
    // under the outgoing model's seal a failed one would count as late.
    crate::kernel_audit::unseal();
}

/// 2026-09-26: Prefix-cache `(hits, misses, hit tokens)` since the last
/// [`reset_for_new_run`]. The TUI shows these; `/metrics` exports the
/// cumulative counters.
pub fn cache_counts_this_run() -> (u64, u64, u64) {
    let m = metrics();
    let sub = |c: &AtomicU64, b: &AtomicU64| {
        c.load(Ordering::Relaxed)
            .saturating_sub(b.load(Ordering::Relaxed))
    };
    (
        sub(&m.cache_hits, &m.run_base_cache_hits),
        sub(&m.cache_misses, &m.run_base_cache_misses),
        sub(&m.cache_hit_tokens, &m.run_base_cache_hit_tokens),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-09-26: After a reset the exported counter has not gone back, and
    /// the run-relative count starts near 0. The checks are bounds, because
    /// another test in this binary records into the same global mailbox
    /// concurrently.
    #[test]
    fn a_new_run_starts_from_the_bottom() {
        const RUN: u64 = 10_000;
        for _ in 0..RUN {
            crate::prefix_cache::record_cache_hit(1);
        }
        let cumulative = crate::prefix_cache::cache_hit_count();
        assert!(cumulative >= RUN, "the run accumulated");

        reset_for_new_run();

        assert!(
            crate::prefix_cache::cache_hit_count() >= cumulative,
            "the exported counter must never go backwards"
        );
        let (hits, _, tokens) = cache_counts_this_run();
        assert!(
            hits < RUN / 10,
            "but the next run does not inherit the previous run's hits"
        );
        assert!(tokens < RUN / 10);
    }
}

#[cfg(test)]
mod swap_counter_tests {
    use super::*;

    /// 2026-09-26: A lower bound against a value read at the start: other tests
    /// in this binary can only add to the same global counter concurrently.
    #[test]
    fn a_new_run_does_not_move_the_counters_prometheus_exports() {
        let m = metrics();
        let before = m.cache_hits.load(Ordering::Relaxed);
        m.cache_hits.fetch_add(7, Ordering::Relaxed);

        reset_for_new_run();

        assert!(
            m.cache_hits.load(Ordering::Relaxed) >= before + 7,
            "the cumulative counter must not go backwards across a swap"
        );
    }
}
