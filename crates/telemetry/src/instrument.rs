// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Lock-free instruments: counters, gauges and power-of-two
//! histograms. All are `const`-constructible, so a whole instrument set can
//! live in a `static` without a lazy initialiser.
//!
//! Every instrument is a set of independent relaxed atomics, so a reader may
//! see a histogram's `count` and buckets disagree by the observations in
//! flight. Values that must be read together (a device sample) go through
//! [`crate::seqcell::SeqCell`] instead. Recording is not level-gated here;
//! [`crate::hub::Telemetry`] does that.
//!
//! Owner: telemetry.
//! Invariants: recording never allocates.

use std::sync::atomic::{AtomicU64, Ordering};

/// 2026-09-26: Monotonic count.
#[derive(Debug, Default)]
pub struct Counter(AtomicU64);

impl Counter {
    pub const fn new() -> Self {
        Self(AtomicU64::new(0))
    }
    #[inline]
    pub fn add(&self, n: u64) {
        self.0.fetch_add(n, Ordering::Relaxed);
    }
    pub fn get(&self) -> u64 {
        self.0.load(Ordering::Relaxed)
    }
}

/// 2026-09-26: Last-value integer gauge.
#[derive(Debug, Default)]
pub struct Gauge(AtomicU64);

impl Gauge {
    pub const fn new() -> Self {
        Self(AtomicU64::new(0))
    }
    #[inline]
    pub fn set(&self, v: u64) {
        self.0.store(v, Ordering::Relaxed);
    }
    pub fn get(&self) -> u64 {
        self.0.load(Ordering::Relaxed)
    }
}

/// 2026-09-26: Last-value float gauge: the f64's bits in one atomic word.
/// `None` is stored as NaN, so an unmeasured ratio reads as `None`, not 0.
#[derive(Debug)]
pub struct GaugeF64(AtomicU64);

const UNSET: u64 = f64::NAN.to_bits();

impl Default for GaugeF64 {
    fn default() -> Self {
        Self::new()
    }
}

impl GaugeF64 {
    pub const fn new() -> Self {
        Self(AtomicU64::new(UNSET))
    }
    #[inline]
    pub fn set(&self, v: Option<f64>) {
        self.0
            .store(v.map_or(UNSET, f64::to_bits), Ordering::Relaxed);
    }
    pub fn get(&self) -> Option<f64> {
        let v = f64::from_bits(self.0.load(Ordering::Relaxed));
        (!v.is_nan()).then_some(v)
    }
}

/// 2026-09-26: A histogram with `N` finite power-of-two buckets plus `+Inf`.
///
/// A value lands in the first bucket `i` whose bound `2^(SHIFT + i)` covers
/// it, or in the overflow slot. Placing it is a leading-zeros count.
#[derive(Debug)]
pub struct Histogram<const N: usize, const SHIFT: u32> {
    buckets: [AtomicU64; N],
    overflow: AtomicU64,
    sum: AtomicU64,
    count: AtomicU64,
}

impl<const N: usize, const SHIFT: u32> Default for Histogram<N, SHIFT> {
    fn default() -> Self {
        Self::new()
    }
}

/// 2026-09-26: A copy of a histogram's state, cumulative: `cumulative[i]`
/// counts values `<= upper_bounds[i]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HistogramSnapshot {
    pub upper_bounds: Vec<u64>,
    pub cumulative: Vec<u64>,
    pub sum: u64,
    pub count: u64,
}

impl<const N: usize, const SHIFT: u32> Histogram<N, SHIFT> {
    pub const fn new() -> Self {
        Self {
            buckets: [const { AtomicU64::new(0) }; N],
            overflow: AtomicU64::new(0),
            sum: AtomicU64::new(0),
            count: AtomicU64::new(0),
        }
    }

    /// 2026-09-26: The bucket index for `v`, `N` meaning `+Inf`.
    pub const fn bucket_of(v: u64) -> usize {
        if v <= (1u64 << SHIFT) {
            return 0;
        }
        // 2026-09-26: ceil(log2(v)); here v > 2^SHIFT >= 1.
        let ceil_log2 = 64 - (v - 1).leading_zeros();
        let i = (ceil_log2 - SHIFT) as usize;
        if i < N { i } else { N }
    }

    pub const fn upper_bound(i: usize) -> u64 {
        1u64 << (SHIFT + i as u32)
    }

    #[inline]
    pub fn observe(&self, v: u64) {
        let i = Self::bucket_of(v);
        let slot = if i < N {
            &self.buckets[i]
        } else {
            &self.overflow
        };
        slot.fetch_add(1, Ordering::Relaxed);
        self.sum.fetch_add(v, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }

    pub fn snapshot(&self) -> HistogramSnapshot {
        let mut running = 0;
        let cumulative = self
            .buckets
            .iter()
            .map(|b| {
                running += b.load(Ordering::Relaxed);
                running
            })
            .collect();
        HistogramSnapshot {
            upper_bounds: (0..N).map(Self::upper_bound).collect(),
            cumulative,
            sum: self.sum.load(Ordering::Relaxed),
            count: self.count.load(Ordering::Relaxed),
        }
    }
}

/// 2026-09-26: Nanosecond durations, bounds 1.024 µs to ~68.7 s.
pub type NanosHistogram = Histogram<27, 10>;
/// 2026-09-26: Small counts (batch rows, prefilling sequences), bounds 1 to 1024.
pub type CountHistogram = Histogram<11, 0>;
/// 2026-09-26: Millijoules, bounds 1 mJ to ~16.8 kJ.
pub type MillijouleHistogram = Histogram<25, 0>;

#[cfg(test)]
mod tests {
    use super::*;

    type H = Histogram<4, 2>;

    #[test]
    fn values_land_in_the_first_bucket_whose_bound_covers_them() {
        // 2026-09-26: Bounds: 4, 8, 16, 32, then +Inf.
        assert_eq!(H::bucket_of(0), 0);
        assert_eq!(H::bucket_of(4), 0);
        assert_eq!(H::bucket_of(5), 1);
        assert_eq!(H::bucket_of(8), 1);
        assert_eq!(H::bucket_of(9), 2);
        assert_eq!(H::bucket_of(32), 3);
        assert_eq!(H::bucket_of(33), 4, "past the last bound is +Inf");
        assert_eq!(H::bucket_of(u64::MAX), 4);
    }

    #[test]
    fn the_snapshot_is_cumulative_and_keeps_sum_and_count() {
        let h = H::new();
        for v in [1, 6, 7, 30, 1000] {
            h.observe(v);
        }
        let s = h.snapshot();
        assert_eq!(s.upper_bounds, vec![4, 8, 16, 32]);
        assert_eq!(s.cumulative, vec![1, 3, 3, 4]);
        assert_eq!(s.count, 5, "the +Inf observation counts in the total");
        assert_eq!(s.sum, 1044);
    }

    #[test]
    fn an_unset_float_gauge_is_none_not_zero() {
        let g = GaugeF64::new();
        assert_eq!(g.get(), None);
        g.set(Some(0.25));
        assert_eq!(g.get(), Some(0.25));
        g.set(None);
        assert_eq!(g.get(), None);
    }

    #[test]
    fn the_nanos_histogram_spans_a_microsecond_to_a_minute() {
        assert_eq!(NanosHistogram::upper_bound(0), 1024);
        assert!(NanosHistogram::upper_bound(26) > 60_000_000_000);
    }
}
