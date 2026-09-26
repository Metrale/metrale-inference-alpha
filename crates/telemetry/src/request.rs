// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: L5: per-request latency (TTFT, TPOT, end-to-end) and energy.
//!
//! Owner: telemetry.
//! Invariants: a non-finite or negative time records as 0 ns.

use crate::instrument::{Counter, MillijouleHistogram, NanosHistogram};

/// 2026-09-26: A finished request's timing, as the scheduler reports it with
/// the request's terminal frame. A request that ends in an error frame
/// reports none.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RequestTiming {
    /// 2026-09-26: Request start to decode start.
    pub ttft_ms: f64,
    /// 2026-09-26: Decode start to the terminal frame.
    pub decode_ms: f64,
    pub output_tokens: u64,
}

impl RequestTiming {
    /// 2026-09-26: Request start to the terminal frame: `ttft_ms + decode_ms`, in ns.
    pub fn e2e_ns(&self) -> u64 {
        ms_to_ns(self.ttft_ms + self.decode_ms)
    }

    /// 2026-09-26: Mean time per output token after the first; `None` for a
    /// request that produced fewer than two tokens.
    pub fn tpot_ns(&self) -> Option<u64> {
        (self.output_tokens >= 2)
            .then(|| ms_to_ns(self.decode_ms / (self.output_tokens - 1) as f64))
    }
}

fn ms_to_ns(ms: f64) -> u64 {
    if ms.is_finite() && ms > 0.0 {
        (ms * 1e6) as u64
    } else {
        0
    }
}

#[derive(Debug)]
pub struct RequestInstruments {
    pub finished: Counter,
    /// 2026-09-26: Output tokens of finished requests, as their terminal frames report.
    pub output_tokens: Counter,
    pub ttft: NanosHistogram,
    pub tpot: NanosHistogram,
    pub e2e: NanosHistogram,
    pub energy: MillijouleHistogram,
    /// 2026-09-26: Finished requests charged no energy: the device layer was
    /// not available, or their window had no ring point or no tokens.
    pub energy_unattributed: Counter,
}

impl Default for RequestInstruments {
    fn default() -> Self {
        Self::new()
    }
}

impl RequestInstruments {
    pub const fn new() -> Self {
        Self {
            finished: Counter::new(),
            output_tokens: Counter::new(),
            ttft: NanosHistogram::new(),
            tpot: NanosHistogram::new(),
            e2e: NanosHistogram::new(),
            energy: MillijouleHistogram::new(),
            energy_unattributed: Counter::new(),
        }
    }

    pub(crate) fn record(&self, r: &RequestTiming, energy_mj: Option<u64>) {
        self.finished.add(1);
        self.output_tokens.add(r.output_tokens);
        self.ttft.observe(ms_to_ns(r.ttft_ms));
        if let Some(t) = r.tpot_ns() {
            self.tpot.observe(t);
        }
        self.e2e.observe(r.e2e_ns());
        match energy_mj {
            Some(mj) => self.energy.observe(mj),
            None => self.energy_unattributed.add(1),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tpot_excludes_the_first_token_and_needs_two() {
        let r = RequestTiming {
            ttft_ms: 100.0,
            decode_ms: 90.0,
            output_tokens: 10,
        };
        assert_eq!(r.tpot_ns(), Some(10_000_000));
        assert_eq!(r.e2e_ns(), 190_000_000);
        let one = RequestTiming {
            output_tokens: 1,
            ..r
        };
        assert_eq!(one.tpot_ns(), None);
    }

    #[test]
    fn a_non_finite_time_records_as_zero_not_a_wrapped_huge_value() {
        assert_eq!(ms_to_ns(f64::NAN), 0);
        assert_eq!(ms_to_ns(-5.0), 0);
        assert_eq!(ms_to_ns(f64::INFINITY), 0);
    }
}
