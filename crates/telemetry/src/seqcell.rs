// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: A sequence-locked cell of `W` words: readers take no lock and
//! never see a mix of two stores.
//!
//! A device sample is [`crate::device::SAMPLE_WORDS`] words describing one
//! instant; one atomic per word would let a reader pair one sample's power
//! with another sample's energy counter. A writer makes the sequence word
//! odd, stores the words and makes it even; a reader retries until it saw the
//! same even value before and after reading. Every access is an atomic.
//!
//! Owner: telemetry.
//! Invariants: `load` returns only a value that a single `store` wrote in full.

use std::sync::atomic::{AtomicU64, Ordering, fence};

#[derive(Debug)]
pub struct SeqCell<const W: usize> {
    seq: AtomicU64,
    words: [AtomicU64; W],
}

impl<const W: usize> Default for SeqCell<W> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const W: usize> SeqCell<W> {
    pub const fn new() -> Self {
        Self {
            seq: AtomicU64::new(0),
            words: [const { AtomicU64::new(0) }; W],
        }
    }

    /// 2026-09-26: Replace the value. Concurrent writers serialise on the
    /// sequence word (a CAS from even to odd), so two writers cannot
    /// interleave words.
    pub fn store(&self, value: [u64; W]) {
        let mut s = self.seq.load(Ordering::Relaxed);
        loop {
            if s & 1 == 1 {
                std::hint::spin_loop();
                s = self.seq.load(Ordering::Relaxed);
                continue;
            }
            match self
                .seq
                .compare_exchange_weak(s, s + 1, Ordering::Acquire, Ordering::Relaxed)
            {
                Ok(_) => break,
                Err(now) => s = now,
            }
        }
        fence(Ordering::Release);
        for (w, v) in self.words.iter().zip(value) {
            w.store(v, Ordering::Relaxed);
        }
        self.seq.store(s + 2, Ordering::Release);
    }

    /// 2026-09-26: A consistent copy, and the number of stores it reflects
    /// (0 = never written).
    pub fn load(&self) -> (u64, [u64; W]) {
        loop {
            let s1 = self.seq.load(Ordering::Acquire);
            if s1 & 1 == 1 {
                std::hint::spin_loop();
                continue;
            }
            let mut out = [0u64; W];
            for (o, w) in out.iter_mut().zip(&self.words) {
                *o = w.load(Ordering::Relaxed);
            }
            fence(Ordering::Acquire);
            if self.seq.load(Ordering::Relaxed) == s1 {
                return (s1 / 2, out);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;

    #[test]
    fn a_fresh_cell_reports_no_stores() {
        let c = SeqCell::<3>::new();
        assert_eq!(c.load(), (0, [0; 3]));
        c.store([1, 2, 3]);
        assert_eq!(c.load(), (1, [1, 2, 3]));
    }

    /// 2026-09-26: Every value written has all words equal, so any mixture of
    /// two values shows as unequal words. Two writers and four readers race
    /// over 200k stores; a single torn read fails the test.
    #[test]
    fn concurrent_readers_never_see_a_torn_value() {
        const W: usize = 15;
        let cell = Arc::new(SeqCell::<W>::new());
        let stop = Arc::new(AtomicBool::new(false));
        let writers: Vec<_> = (0..2u64)
            .map(|id| {
                let cell = cell.clone();
                std::thread::spawn(move || {
                    for k in 0..100_000u64 {
                        let v = k * 2 + id + 1;
                        cell.store([v; W]);
                    }
                })
            })
            .collect();
        let readers: Vec<_> = (0..4)
            .map(|_| {
                let (cell, stop) = (cell.clone(), stop.clone());
                std::thread::spawn(move || {
                    let mut reads = 0u64;
                    while !stop.load(Ordering::Relaxed) {
                        let (_, w) = cell.load();
                        assert!(w.iter().all(|x| *x == w[0]), "torn read: {w:?}");
                        reads += 1;
                    }
                    reads
                })
            })
            .collect();
        for w in writers {
            w.join().unwrap();
        }
        stop.store(true, Ordering::Relaxed);
        let reads: u64 = readers.into_iter().map(|r| r.join().unwrap()).sum();
        assert!(reads > 0, "the readers ran");
        assert_eq!(cell.load().0, 200_000, "every store is counted once");
    }
}
