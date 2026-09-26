// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: MTP acceptance statistics per batch width, and the per-request Done-line figures.
//!
//! [`AcceptBuckets`] counts verify outcomes per batch-width bucket. Every
//! `PERIOD` verifies a bucket flushes: a fresh flush goes to the run's
//! `AdaptiveRung::observe`, and when `METRALE_MTP_ACCEPT_DEBUG` is set a line
//! logs `p1` (share of verifies whose first draft matched), `mean_na` (mean
//! accepted drafts) and `tok_step = 1 + mean_na`. [`RequestAccept`] holds the
//! same figures for one request.
//!
//! Owner: scheduler.
//! Invariants:
//! - The counters are relaxed atomics; nothing here copies from the device or
//!   synchronises a stream.

use std::sync::atomic::{AtomicU64, Ordering};

/// 2026-09-25: Number of width buckets. Widths `0..MAX_N - 1` each have their own
/// bucket; wider ones share the last.
///
/// It must exceed the MTP dispatch cap
/// ([`metrale_model_layers::speculative::mtp_max_seqs`], 32 by default), or
/// distinct widths share a bucket. A shared
/// bucket mixes their statistics, and its flush carries the width of whichever
/// caller tripped [`PERIOD`]; `AdaptiveRung::observe` ignores widths outside
/// `9..=16`, so a flush tripped by a wider caller is lost to the rung.
///
/// `METRALE_MTP_ACCEPT_FOLD_AT_16` (set, whatever its value) folds every width
/// of 16 or more into bucket 16 instead.
const MAX_N: usize = 33;

/// 2026-09-25: The bucket index for batch width `n`, used by
/// [`AcceptBuckets::record`] and the aliasing test.
fn bucket_idx(n: usize, fold_at_16: bool) -> usize {
    if fold_at_16 {
        n.min(16)
    } else {
        n.min(MAX_N - 1)
    }
}
/// 2026-09-25: Verifies per flush. A fresh flush is one `AdaptiveRung::observe`
/// call, so this sets how often the rung is updated; at n=16 it is 8 verify
/// steps.
const PERIOD: u64 = 128;

/// 2026-09-25: A flush whose `PERIOD` verifies took longer than this to accumulate
/// is not passed to the rung: it describes older traffic. The clock is read
/// when a bucket's window opens and at its flush, not per verify.
const MAX_SAMPLE_SPAN_MS: u64 = 5_000;

struct Bucket {
    steps: AtomicU64,
    d1: AtomicU64,
    na: AtomicU64,
    k: AtomicU64,
    /// 2026-09-25: Millis since [`AcceptBuckets::epoch`] when this bucket's
    /// current window opened (stamped when `steps` goes 0 -> 1), so the flush's
    /// span covers exactly this window, the first one included.
    window_start_ms: AtomicU64,
}

const fn new_bucket() -> Bucket {
    Bucket {
        steps: AtomicU64::new(0),
        d1: AtomicU64::new(0),
        na: AtomicU64::new(0),
        k: AtomicU64::new(0),
        window_start_ms: AtomicU64::new(0),
    }
}

#[allow(clippy::declare_interior_mutable_const)]
const INIT: Bucket = new_bucket();

/// 2026-09-25: The per-width accept table of one scheduler run, built by
/// `SchedCtx::new`.
pub struct AcceptBuckets {
    fold_at_16: bool,
    /// 2026-09-25: Creation time: the monotonic reference for `window_start_ms`.
    epoch: std::time::Instant,
    buckets: [Bucket; MAX_N],
}

impl AcceptBuckets {
    pub fn new(fold_at_16: bool) -> Self {
        Self {
            fold_at_16,
            epoch: std::time::Instant::now(),
            buckets: [INIT; MAX_N],
        }
    }
}

/// 2026-09-25: Record one sequence's verify outcome at batch width `n`.
///
/// `d1_match` is whether the first draft matched the target
/// (`drafts[0] == verified[0]`), so `p1` is not conditional on the accept
/// chain.
///
/// `k_drafts` is this sequence's retained draft depth, which D-Cut can make
/// differ within one batch; a flush reports the largest depth recorded in its
/// window.
///
/// Counting does not depend on `METRALE_MTP_ACCEPT_DEBUG`, which gates only
/// the log line: the rung is fed from these counters, so its behaviour does
/// not depend on whether the log is on.
impl AcceptBuckets {
    /// 2026-09-25: `rung` is the run's rung controller, which receives every fresh
    /// flush; `debug` is the `METRALE_MTP_ACCEPT_DEBUG` lever, which gates
    /// only the log line.
    pub(super) fn record(
        &self,
        rung: &metrale_speculative::adaptive_rung::AdaptiveRung,
        debug: bool,
        n: usize,
        k_drafts: usize,
        d1_match: bool,
        num_accepted: usize,
    ) {
        let b = &self.buckets[bucket_idx(n, self.fold_at_16)];
        b.k.fetch_max(k_drafts as u64, Ordering::Relaxed);
        b.na.fetch_add(num_accepted as u64, Ordering::Relaxed);
        if d1_match {
            b.d1.fetch_add(1, Ordering::Relaxed);
        }
        let prev_steps = b.steps.fetch_add(1, Ordering::Relaxed);
        if prev_steps == 0 {
            b.window_start_ms
                .store(self.epoch.elapsed().as_millis() as u64, Ordering::Relaxed);
        }
        if prev_steps + 1 >= PERIOD {
            let steps = b.steps.swap(0, Ordering::Relaxed).max(1);
            let d1 = b.d1.swap(0, Ordering::Relaxed);
            let na = b.na.swap(0, Ordering::Relaxed);
            let k = b.k.swap(0, Ordering::Relaxed);
            let mean_na = na as f64 / steps as f64;
            let p1 = d1 as f64 / steps as f64;
            let span_ms = (self.epoch.elapsed().as_millis() as u64)
                .saturating_sub(b.window_start_ms.load(Ordering::Relaxed));
            let fresh = span_ms <= MAX_SAMPLE_SPAN_MS;
            // 2026-09-25: The rung has no counters of its own; it is fed from
            // this flush. A stale flush (MAX_SAMPLE_SPAN_MS) is not passed on.
            if fresh {
                rung.observe(n, k as usize, p1, mean_na);
            }
            if debug {
                tracing::info!(
                    "MTP accept n={n} k_drafts={k} verifies={steps} p1={p1:.3} \
                 mean_na={mean_na:.3} tok_step={:.3} token_ratio={:.4} \
                 span_ms={span_ms} fresh={fresh}",
                    1.0 + mean_na,
                    metrale_speculative::adaptive_rung::token_ratio(
                        p1,
                        metrale_speculative::adaptive_rung::p2_cond_from(p1, mean_na)
                            .unwrap_or(0.0),
                    ),
                );
            }
        }
    }
}

/// 2026-09-25: One request's `p1`, `mean_na` and `tok_step = 1 + mean_na`, plus its
/// serial and MTP step shares and depth-regime re-probe count. `log_done`
/// prints them on the request's Done line; `accepted_total` feeds the usage
/// block.
#[derive(Debug, Default, Clone, Copy)]
pub struct RequestAccept {
    serial_steps: u64,
    mtp_steps: u64,
    d1: u64,
    na: u64,
    pub regime_reprobes: u64,
}

impl RequestAccept {
    pub fn record_serial(&mut self) {
        self.serial_steps = self.serial_steps.saturating_add(1);
    }

    /// 2026-09-25: `emitted` is tokens committed this verify (1 + accepted drafts).
    /// The first draft counts as matched when `emitted > 1`.
    pub fn record_verify_emitted(&mut self, emitted: usize) {
        self.mtp_steps = self.mtp_steps.saturating_add(1);
        let accepted = emitted.saturating_sub(1) as u64;
        self.na = self.na.saturating_add(accepted);
        if accepted > 0 {
            self.d1 = self.d1.saturating_add(1);
        }
    }

    /// 2026-09-25: Total draft tokens accepted for this request, reported as
    /// `usage.completion_tokens_details.accepted_prediction_tokens`. A count,
    /// not a rate.
    pub fn accepted_total(&self) -> u64 {
        self.na
    }

    pub fn note_regime_reprobe(&mut self) {
        self.regime_reprobes = self.regime_reprobes.saturating_add(1);
    }

    fn total_steps(&self) -> u64 {
        self.serial_steps.saturating_add(self.mtp_steps)
    }

    pub fn serial_frac(&self) -> f64 {
        let n = self.total_steps();
        if n == 0 {
            0.0
        } else {
            self.serial_steps as f64 / n as f64
        }
    }

    pub fn mtp_frac(&self) -> f64 {
        let n = self.total_steps();
        if n == 0 {
            0.0
        } else {
            self.mtp_steps as f64 / n as f64
        }
    }

    pub fn p1(&self) -> f64 {
        if self.mtp_steps == 0 {
            0.0
        } else {
            self.d1 as f64 / self.mtp_steps as f64
        }
    }

    pub fn mean_na(&self) -> f64 {
        if self.mtp_steps == 0 {
            0.0
        } else {
            self.na as f64 / self.mtp_steps as f64
        }
    }

    pub fn tok_step(&self) -> f64 {
        1.0 + self.mean_na()
    }

    pub fn done_suffix(&self) -> String {
        format!(
            "serial={:.2} mtp={:.2} p1={:.3} mean_na={:.3} tok_step={:.3} regime_reprobes={}",
            self.serial_frac(),
            self.mtp_frac(),
            self.p1(),
            self.mean_na(),
            self.tok_step(),
            self.regime_reprobes
        )
    }

    pub fn log_done(n: usize, reason: &str, tps: f64, ttft_ms: f64, acct: &Self) {
        tracing::info!(
            "Done: {n} tokens ({reason}) {tps:.1} tok/s, TTFT={ttft_ms:.1}ms, {}",
            acct.done_suffix()
        );
    }
}

#[cfg(test)]
mod tests {

    // 2026-09-25: The bucket table must exceed the MTP dispatch cap (see the
    // MAX_N doc).
    #[test]
    fn bucket_table_covers_the_mtp_dispatch_cap() {
        assert_eq!(AcceptBuckets::new(false).buckets.len(), MAX_N);
        // 2026-09-25: 32 is the default cap in `mtp_max_seqs`.
        const { assert!(MAX_N > 32) };
        // 2026-09-25: Checked against the live cap only when the override is unset.
        if std::env::var_os("METRALE_MTP_MAX_SEQS").is_none() {
            assert!(
                MAX_N > metrale_model_layers::speculative::mtp_max_seqs(),
                "MAX_N {MAX_N} does not cover dispatch cap {}",
                metrale_model_layers::speculative::mtp_max_seqs()
            );
        }
    }

    // 2026-09-25: No two widths up to the dispatch cap share a bucket, checked
    // on `bucket_idx`. Skipped when either override is set.
    #[test]
    fn widths_up_to_the_cap_do_not_alias() {
        if std::env::var_os("METRALE_MTP_ACCEPT_FOLD_AT_16").is_some()
            || std::env::var_os("METRALE_MTP_MAX_SEQS").is_some()
        {
            return;
        }
        for n in 0..=metrale_model_layers::speculative::mtp_max_seqs() {
            assert_eq!(
                bucket_idx(n, false),
                n,
                "width {n} aliases onto another bucket"
            );
        }
        // 2026-09-25: Beyond the table widths fold onto the last bucket, inside the
        // array.
        assert_eq!(bucket_idx(1_000, false), MAX_N - 1);
        assert!(bucket_idx(usize::MAX, false) < MAX_N);
        assert_eq!(
            bucket_idx(1_000, true),
            16,
            "the kill switch restores the fold"
        );
    }

    use super::*;

    #[test]
    fn empty_suffix_is_zeros() {
        let a = RequestAccept::default();
        assert_eq!(
            a.done_suffix(),
            "serial=0.00 mtp=0.00 p1=0.000 mean_na=0.000 tok_step=1.000 regime_reprobes=0"
        );
    }

    #[test]
    fn thinking_serial_run_is_all_serial() {
        let mut a = RequestAccept::default();
        for _ in 0..300 {
            a.record_serial();
        }
        assert!((a.serial_frac() - 1.0).abs() < 1e-9);
        assert_eq!(a.mean_na(), 0.0);
        assert_eq!(a.tok_step(), 1.0);
        assert!(a.done_suffix().contains("serial=1.00"));
        assert!(a.done_suffix().contains("mtp=0.00"));
    }

    #[test]
    fn mtp_run_reports_p1_mean_na_tok_step() {
        let mut a = RequestAccept::default();
        for _ in 0..7 {
            a.record_verify_emitted(2);
        }
        for _ in 0..3 {
            a.record_verify_emitted(1);
        }
        assert!((a.mtp_frac() - 1.0).abs() < 1e-9);
        assert!((a.p1() - 0.7).abs() < 1e-9);
        assert!((a.mean_na() - 0.7).abs() < 1e-9);
        // 2026-09-25: 7 verifies each accepting 1 draft: the per-request
        // total the usage field reports is the raw sum, not a rate.
        assert_eq!(a.accepted_total(), 7);
        assert!((a.tok_step() - 1.7).abs() < 1e-9);
        a.note_regime_reprobe();
        assert!(a.done_suffix().contains("mean_na=0.700"));
        assert!(a.done_suffix().contains("tok_step=1.700"));
        assert!(a.done_suffix().contains("regime_reprobes=1"));
    }
}
