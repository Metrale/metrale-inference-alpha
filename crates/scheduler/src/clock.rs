// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: [`ClockIo`], the injectable clock, and [`SystemClock`], its
//! implementation over [`Instant::now`].
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use std::time::Instant;

pub trait ClockIo: Send + Sync + std::fmt::Debug {
    fn now(&self) -> Instant;
}

/// 2026-09-25: The monotonic system clock ([`Instant`]).
#[derive(Debug, Default)]
pub struct SystemClock;

impl ClockIo for SystemClock {
    fn now(&self) -> Instant {
        Instant::now()
    }
}
