// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Energy attribution: the NVML counter's deltas, paired with the
//! tokens produced over the same interval.
//!
//! Two cumulative totals are kept, GPU-rail millijoules and tokens, and
//! sampled together into [`EnergyRing`]. Every ratio is formed from
//! differences of those totals: J/token over the last
//! [`crate::hub::LIVE_WINDOW`] samples, J/token since the first sample, and a
//! request's energy (its token share of its lifetime window). No ratio is
//! accumulated.
//!
//! The sampler thread reads the counter at the device cadence, not per
//! scheduler step, so a window's ΔE/Δtokens is exact up to one sample period
//! at each edge. A backwards step is a wrap or a reset;
//! [`CounterTracker::advance`] tells them apart.
//!
//! Owner: telemetry.
//! Invariants: none beyond the types.

use crate::seqcell::SeqCell;
use std::sync::atomic::{AtomicU64, Ordering};

/// 2026-09-26: What one counter reading did to the running total.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Advance {
    /// 2026-09-26: The first reading: it anchors the series and contributes nothing.
    First,
    /// 2026-09-26: The counter moved forward by this many mJ.
    Delta(u64),
    /// 2026-09-26: The counter wrapped past `u64::MAX`; this many mJ elapsed.
    Wrapped(u64),
    /// 2026-09-26: The counter restarted; the new reading, the mJ since the
    /// restart, is charged.
    Reset(u64),
}

impl Advance {
    pub fn millijoules(self) -> u64 {
        match self {
            Self::First => 0,
            Self::Delta(d) | Self::Wrapped(d) | Self::Reset(d) => d,
        }
    }
}

/// 2026-09-26: Turns raw counter readings into non-negative deltas.
#[derive(Clone, Copy, Debug, Default)]
pub struct CounterTracker {
    prev: Option<u64>,
}

/// 2026-09-26: A backwards step is a wrap only when the previous reading sat
/// in the top quarter of the range and the new one in the bottom quarter;
/// any other backwards step is a restart.
const QUARTER: u64 = u64::MAX / 4;

impl CounterTracker {
    pub fn advance(&mut self, raw: u64) -> Advance {
        let prev = self.prev.replace(raw);
        match prev {
            None => Advance::First,
            Some(p) if raw >= p => Advance::Delta(raw - p),
            Some(p) if p > u64::MAX - QUARTER && raw < QUARTER => {
                Advance::Wrapped(raw.wrapping_sub(p))
            }
            Some(_) => Advance::Reset(raw),
        }
    }
}

/// 2026-09-26: GPU-rail joules per token for a (ΔmJ, Δtokens) pair. `None`
/// when either is 0, so the result is never 0 or ∞.
pub fn joules_per_token(delta_mj: u64, tokens: u64) -> Option<f64> {
    (tokens > 0 && delta_mj > 0).then(|| delta_mj as f64 / 1000.0 / tokens as f64)
}

/// 2026-09-26: One ring point: cumulative energy and tokens at one sample instant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EnergyPoint {
    pub at_ns: u64,
    pub energy_mj: u64,
    pub tokens: u64,
}

/// 2026-09-26: Ring capacity: 8192 samples, ~13.6 minutes at the 10 Hz
/// serving cadence. A request older than the ring is attributed from the
/// oldest point it still holds.
pub const RING_SLOTS: usize = 8192;

/// 2026-09-26: The sampled (time, energy, tokens) series. `push` assumes one
/// writer (the sampler thread); any number of readers read without a lock.
pub struct EnergyRing {
    slots: Box<[SeqCell<3>]>,
    written: AtomicU64,
}

impl std::fmt::Debug for EnergyRing {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EnergyRing")
            .field("written", &self.written.load(Ordering::Relaxed))
            .finish_non_exhaustive()
    }
}

impl EnergyRing {
    pub fn with_capacity(slots: usize) -> Self {
        assert!(slots >= 2, "a ring needs two points to form a window");
        Self {
            slots: (0..slots).map(|_| SeqCell::new()).collect(),
            written: AtomicU64::new(0),
        }
    }

    pub fn push(&self, p: EnergyPoint) {
        let n = self.written.load(Ordering::Relaxed);
        let i = (n % self.slots.len() as u64) as usize;
        self.slots[i].store([p.at_ns, p.energy_mj, p.tokens]);
        self.written.store(n + 1, Ordering::Release);
    }

    fn point(&self, logical: u64) -> EnergyPoint {
        let (_, [at_ns, energy_mj, tokens]) =
            self.slots[(logical % self.slots.len() as u64) as usize].load();
        EnergyPoint {
            at_ns,
            energy_mj,
            tokens,
        }
    }

    /// 2026-09-26: The live logical index range `[lo, hi)`.
    fn live(&self) -> (u64, u64) {
        let hi = self.written.load(Ordering::Acquire);
        // 2026-09-26: One slot of slack: the writer may be overwriting the oldest.
        let lo = hi.saturating_sub(self.slots.len() as u64 - 1);
        (lo, hi)
    }

    /// 2026-09-26: The point `back` samples before the newest (`0` = the newest).
    pub fn back(&self, back: u64) -> Option<EnergyPoint> {
        let (lo, hi) = self.live();
        let i = hi.checked_sub(1 + back)?;
        (i >= lo).then(|| self.point(i))
    }

    /// 2026-09-26: The newest point at or before `t_ns`; the oldest point when
    /// every live point is later; `None` for an empty ring.
    pub fn at_or_before(&self, t_ns: u64) -> Option<EnergyPoint> {
        let (lo, hi) = self.live();
        if lo == hi {
            return None;
        }
        // 2026-09-26: Binary search for the first point after `t_ns`, in [lo, hi).
        let (mut a, mut b) = (lo, hi);
        while a < b {
            let mid = a + (b - a) / 2;
            if self.point(mid).at_ns <= t_ns {
                a = mid + 1;
            } else {
                b = mid;
            }
        }
        Some(self.point(if a == lo { lo } else { a - 1 }))
    }
}

/// 2026-09-26: Millijoules attributed to a request that emitted `tokens` over
/// `[start_ns, end_ns]`: its share, by tokens, of the energy counted over that
/// window. `None` when the ring is empty or the window saw no tokens.
pub fn request_millijoules(
    ring: &EnergyRing,
    start_ns: u64,
    end_ns: u64,
    tokens: u64,
) -> Option<u64> {
    let p0 = ring.at_or_before(start_ns)?;
    let p1 = ring.at_or_before(end_ns)?;
    let de = p1.energy_mj.checked_sub(p0.energy_mj)?;
    let dt = p1.tokens.checked_sub(p0.tokens).filter(|t| *t > 0)?;
    Some((de as u128 * tokens as u128 / dt as u128) as u64)
}

#[cfg(test)]
#[path = "energy_tests.rs"]
mod tests;
