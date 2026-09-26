// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Arrival-gap accumulator: the within-response jitter instrument.
//!
//! An arrival is one socket read that carried at least one token delta
//! (`chat_stream_inner` in `http.rs`); a gap is the time between two
//! consecutive arrivals of one response. Gaps are taken per read, not per
//! token: several deltas that arrive in one read are one arrival.
//!
//! Count, mean and variance are kept online (Welford in `push`, Chan's
//! update in `merge`) with a running max. Percentiles need the samples, so
//! gaps are retained as `f32` up to [`RETAIN_CAP`] per response; past that
//! the online statistics keep counting and [`GapStats::retained`] says how
//! many gaps the percentiles cover. A cell pools its requests with
//! [`GapSample::merge`]. `stability` and the coefficient of variation are
//! derived from the primitives by [`GapStats`]; for both, lower is better.
//!
//! Owner: bench (HTTP client).
//! Invariants:
//! - A `GapSample` never retains more gaps than it has counted.
//! - [`GapSample::stats`] is `None` exactly when no gap was recorded.

use std::collections::BTreeMap;

use crate::benchmarks::stats;

/// 2026-09-26: Gaps retained per response for the percentiles. Equal to the
/// concurrency driver's largest `osl`, which is its `max_tokens`, so no
/// concurrency response is truncated; a longer response keeps counting
/// online and reports the truncation through [`GapStats::retained`].
pub const RETAIN_CAP: usize = 8192;

/// 2026-09-26: Online accumulator of one response's arrival gaps (or a pool of them).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GapSample {
    count: u64,
    mean: f64,
    m2: f64,
    max: f64,
    retained: Vec<f32>,
}

impl GapSample {
    /// 2026-09-26: Record one gap, in milliseconds.
    pub fn push(&mut self, gap_ms: f64) {
        self.count += 1;
        let delta = gap_ms - self.mean;
        self.mean += delta / self.count as f64;
        self.m2 += delta * (gap_ms - self.mean);
        if gap_ms > self.max {
            self.max = gap_ms;
        }
        if self.retained.len() < RETAIN_CAP {
            self.retained.push(gap_ms as f32);
        }
    }

    /// 2026-09-26: Pool another sample into this one (Chan's parallel update
    /// for the moments; the retained gaps concatenate, so a pool of `k`
    /// responses holds at most `k × RETAIN_CAP` gaps).
    pub fn merge(&mut self, other: &GapSample) {
        if other.count == 0 {
            return;
        }
        let na = self.count as f64;
        let nb = other.count as f64;
        let n = na + nb;
        let delta = other.mean - self.mean;
        self.mean += delta * nb / n;
        self.m2 += other.m2 + delta * delta * na * nb / n;
        self.count += other.count;
        self.max = self.max.max(other.max);
        self.retained.extend_from_slice(&other.retained);
    }

    pub fn count(&self) -> u64 {
        self.count
    }

    /// 2026-09-26: The distribution summary, or `None` when no gap was
    /// recorded (fewer than two arrivals): a zero would read as "perfectly
    /// smooth".
    pub fn stats(&self) -> Option<GapStats> {
        if self.count == 0 {
            return None;
        }
        let retained: Vec<f64> = self.retained.iter().map(|g| f64::from(*g)).collect();
        let pct = |p| stats::percentile(&retained, p).unwrap_or(f64::NAN);
        Some(GapStats {
            count: self.count,
            retained: self.retained.len(),
            mean_ms: self.mean,
            // 2026-09-26: Population deviation: the gaps are the whole
            // population of this response, not a sample of a larger one.
            stddev_ms: (self.m2 / self.count as f64).sqrt(),
            max_ms: self.max,
            p50_ms: pct(50),
            p90_ms: pct(90),
            p99_ms: pct(99),
        })
    }
}

/// 2026-09-26: One response's (or one pool's) arrival-gap distribution, in ms.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GapStats {
    /// 2026-09-26: Gaps observed.
    pub count: u64,
    /// 2026-09-26: Gaps the percentiles were computed over (`< count` only
    /// when a response ran past [`RETAIN_CAP`]).
    pub retained: usize,
    pub mean_ms: f64,
    pub stddev_ms: f64,
    /// 2026-09-26: A stall is a tail event; the mean hides it and this does
    /// not.
    pub max_ms: f64,
    pub p50_ms: f64,
    pub p90_ms: f64,
    pub p99_ms: f64,
}

impl GapStats {
    /// 2026-09-26: `stability = (p99 − p50) / p50`: the arrival-gap tail
    /// spread relative to its median. Dimensionless, so it compares runs
    /// whose absolute step times differ.
    ///
    /// Lower is better: it is a dispersion measure, and 0 means no tail at
    /// all. `None` unless p50 is positive and finite and p99 is finite.
    pub fn stability(&self) -> Option<f64> {
        (self.p50_ms.is_finite() && self.p50_ms > 0.0 && self.p99_ms.is_finite())
            .then(|| (self.p99_ms - self.p50_ms) / self.p50_ms)
    }

    /// 2026-09-26: Coefficient of variation, `σ / mean`. Also a dispersion
    /// measure: lower is better. `None` unless the mean is positive.
    pub fn cv(&self) -> Option<f64> {
        (self.mean_ms > 0.0).then(|| self.stddev_ms / self.mean_ms)
    }

    /// 2026-09-26: The record keys, under `prefix` (e.g. `"c8_"` or `""`).
    /// The primitives always; the two derived ratios only when defined.
    pub fn metrics(&self, prefix: &str, m: &mut BTreeMap<String, f64>) {
        let put = |m: &mut BTreeMap<String, f64>, k: &str, v: f64| {
            m.insert(format!("{prefix}{k}"), v);
        };
        put(m, "arrival_gap_count", self.count as f64);
        put(m, "arrival_gap_retained", self.retained as f64);
        put(m, "arrival_gap_mean_ms", self.mean_ms);
        put(m, "arrival_gap_stddev_ms", self.stddev_ms);
        put(m, "arrival_gap_max_ms", self.max_ms);
        put(m, "arrival_gap_p50_ms", self.p50_ms);
        put(m, "arrival_gap_p90_ms", self.p90_ms);
        put(m, "arrival_gap_p99_ms", self.p99_ms);
        if let Some(s) = self.stability() {
            put(m, "stability", s);
        }
        if let Some(cv) = self.cv() {
            put(m, "arrival_gap_cv", cv);
        }
    }
}

/// 2026-09-26: Inter-token latency: `decode_window_ms / (output_tokens − 1)`.
/// The one definition both clocks use: the client passes its first-delta →
/// stream-end window, and `ChatOutcome::server_tpot_ms` passes the
/// endpoint's `usage.decode_time_ms`.
///
/// `None`, never 0, below two output tokens and for a window that is not
/// positive and finite: a zero would read as an infinitely fast decode.
pub fn itl_ms(decode_window_ms: f64, output_tokens: usize) -> Option<f64> {
    (output_tokens >= 2 && decode_window_ms.is_finite() && decode_window_ms > 0.0)
        .then(|| decode_window_ms / (output_tokens - 1) as f64)
}

#[cfg(test)]
#[path = "gaps_tests.rs"]
mod tests;
