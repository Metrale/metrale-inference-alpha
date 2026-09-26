// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The clock [`crate::hub::Telemetry`] reads, injected so a test
//! can count its reads (`tests/zero_cost_off.rs`).
//!
//! Owner: telemetry.
//! Invariants: none beyond the types.

use std::sync::OnceLock;
use std::time::Instant;

pub trait Clock: Send + Sync {
    /// 2026-09-26: Nanoseconds on a monotonic timeline private to this clock.
    fn now_ns(&self) -> u64;
    /// 2026-09-26: Nanoseconds since `since`.
    fn elapsed_ns(&self, since: Instant) -> u64;
}

/// 2026-09-26: The process's monotonic clock. Its origin is the first
/// `now_ns` of any `MonotonicClock` in the process.
#[derive(Debug, Default)]
pub struct MonotonicClock;

fn origin() -> Instant {
    static ORIGIN: OnceLock<Instant> = OnceLock::new();
    *ORIGIN.get_or_init(Instant::now)
}

fn nanos(d: std::time::Duration) -> u64 {
    u64::try_from(d.as_nanos()).unwrap_or(u64::MAX)
}

impl Clock for MonotonicClock {
    fn now_ns(&self) -> u64 {
        nanos(origin().elapsed())
    }
    fn elapsed_ns(&self, since: Instant) -> u64 {
        nanos(since.elapsed())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_monotonic_clock_never_goes_backwards() {
        let c = MonotonicClock;
        let a = c.now_ns();
        let b = c.now_ns();
        assert!(b >= a);
        assert!(c.elapsed_ns(Instant::now()) < 1_000_000_000);
    }
}
