// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The commit ledger of the RDMA memory blades (`cache_peer`,
//! `expert_peer`): a running total of reserved bytes against a ceiling. Each
//! `serve` call creates one and shares it with its connection threads, which
//! reserve their bytes before they map or register them. The module is not behind
//! `cfg(metrale_rdma_verbs)`, so its tests run without RDMA.
//!
//! Owner: metrale-storage peers.
//! Invariants:
//! - `committed` equals the sum of `bytes` over the live `Reservation`s.
//! - With a non-zero cap, `committed` never exceeds it.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result, bail};

/// 2026-09-25: Reserved bytes against a fixed ceiling; `cap == 0` means no
/// ceiling.
#[derive(Debug)]
pub struct CommitLedger {
    cap: u64,
    committed: AtomicU64,
}

/// 2026-09-25: A claim on `bytes` of the ledger, created only by a successful
/// `try_reserve`. Dropping it returns the bytes.
#[derive(Debug)]
pub struct Reservation {
    ledger: Arc<CommitLedger>,
    bytes: u64,
}

impl CommitLedger {
    pub fn new(cap_bytes: u64) -> Self {
        Self {
            cap: cap_bytes,
            committed: AtomicU64::new(0),
        }
    }

    pub fn cap(&self) -> u64 {
        self.cap
    }

    pub fn committed(&self) -> u64 {
        self.committed.load(Ordering::Acquire)
    }

    /// 2026-09-25: Add `bytes` if the total stays within the cap. The check and
    /// the add form one compare-exchange loop, so two concurrent callers cannot
    /// both pass a nearly full ledger. On overflow or when the cap would be
    /// crossed it fails and reserves nothing.
    pub fn try_reserve(self: &Arc<Self>, bytes: u64) -> Result<Reservation> {
        let mut cur = self.committed.load(Ordering::Acquire);
        loop {
            let new = cur
                .checked_add(bytes)
                .context("blade commit total overflow")?;
            if self.cap != 0 && new > self.cap {
                bail!(
                    "blade cap exceeded: request {bytes} B + {cur} B committed > cap {} B",
                    self.cap,
                );
            }
            match self.committed.compare_exchange_weak(
                cur,
                new,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    return Ok(Reservation {
                        ledger: self.clone(),
                        bytes,
                    });
                }
                Err(observed) => cur = observed,
            }
        }
    }
}

impl Reservation {
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        self.ledger
            .committed
            .fetch_sub(self.bytes, Ordering::AcqRel);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reserve_within_cap_raises_committed() {
        let l = Arc::new(CommitLedger::new(1000));
        let r = l.try_reserve(400).expect("within cap");
        assert_eq!(l.committed(), 400);
        assert_eq!(r.bytes(), 400);
    }

    #[test]
    fn reserve_over_cap_bails_and_leaves_committed_unchanged() {
        let l = Arc::new(CommitLedger::new(1000));
        let _r = l.try_reserve(600).expect("within cap");
        assert_eq!(l.committed(), 600);
        let err = l.try_reserve(500);
        assert!(err.is_err(), "600 + 500 > 1000 must be rejected");
        assert_eq!(l.committed(), 600);
    }

    #[test]
    fn dropping_a_reservation_restores_committed() {
        let l = Arc::new(CommitLedger::new(1000));
        {
            let _r = l.try_reserve(700).expect("within cap");
            assert_eq!(l.committed(), 700);
        }
        assert_eq!(l.committed(), 0);
        let _r2 = l.try_reserve(1000).expect("fits after release");
        assert_eq!(l.committed(), 1000);
    }

    #[test]
    fn two_reservations_aggregate_reject_then_accept_after_drop() {
        let l = Arc::new(CommitLedger::new(1000));
        let r1 = l.try_reserve(700).expect("first fits");
        assert!(l.try_reserve(400).is_err(), "700 + 400 > 1000");
        assert_eq!(l.committed(), 700);
        drop(r1);
        let _r2 = l.try_reserve(400).expect("fits after first drops");
        assert_eq!(l.committed(), 400);
    }

    #[test]
    fn cap_zero_accepts_arbitrarily_large() {
        let l = Arc::new(CommitLedger::new(0));
        let _r = l.try_reserve(u64::MAX / 2).expect("unlimited");
        let _r2 = l.try_reserve(1 << 40).expect("still unlimited");
        assert_eq!(l.committed(), (u64::MAX / 2) + (1 << 40));
    }

    #[test]
    fn exact_fit_is_accepted() {
        let l = Arc::new(CommitLedger::new(1000));
        let _r = l.try_reserve(1000).expect("new == cap is allowed");
        assert_eq!(l.committed(), 1000);
        assert!(l.try_reserve(1).is_err());
    }

    #[test]
    fn overflow_is_rejected_not_wrapped() {
        let l = Arc::new(CommitLedger::new(0));
        let _r = l.try_reserve(u64::MAX - 10).expect("fits");
        assert!(
            l.try_reserve(100).is_err(),
            "checked_add must reject wraparound"
        );
    }
}
