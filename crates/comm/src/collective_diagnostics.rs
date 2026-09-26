// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Per-submission checks and optional logging for `NcclBackend`,
//! switched by `METRALE_COMM_DIAGNOSTICS` (`0`, `1` or unset). The sequence
//! number counts host submissions on this rank; it is not exchanged between
//! ranks and counts neither completions nor CUDA graph replays.
//!
//! Owner: metrale-comm.
//! Invariants:
//! - `submit` validates whether or not logging is on.
use anyhow::{Result, ensure};
use std::sync::atomic::{AtomicU64, Ordering};

pub const ENV: &str = "METRALE_COMM_DIAGNOSTICS";

#[derive(Clone, Copy, Debug)]
pub enum Dtype {
    Bf16,
    U8,
    F32,
}

impl Dtype {
    pub fn width(self) -> usize {
        match self {
            Self::Bf16 => 2,
            Self::U8 => 1,
            Self::F32 => 4,
        }
    }
}

pub struct Diagnostics {
    enabled: bool,
    sequence: AtomicU64,
    rank: usize,
    world: usize,
}

impl Diagnostics {
    pub fn new(value: Option<&str>, rank: usize, world: usize) -> Result<Self> {
        ensure!(
            world > 0 && world <= i32::MAX as usize && rank < world,
            "NCCL invalid group: rank={rank} world_size={world}"
        );
        let enabled = match value {
            None | Some("0") => false,
            Some("1") => true,
            Some(other) => anyhow::bail!("{ENV} must be 0 or 1, got {other:?}"),
        };
        Ok(Self {
            enabled,
            sequence: AtomicU64::new(0),
            rank,
            world,
        })
    }

    /// 2026-09-26: Refuse a byte count that is not a whole number of `dtype`
    /// elements, or a peer outside the group; then log when enabled.
    pub fn submit(
        &self,
        op: &str,
        dtype: Dtype,
        bytes: usize,
        stream: u64,
        peer: Option<usize>,
    ) -> Result<()> {
        ensure!(
            bytes.is_multiple_of(dtype.width()),
            "NCCL rank={} world_size={} op={op} dtype={dtype:?} bytes={bytes}: not a whole element (width={})",
            self.rank,
            self.world,
            dtype.width()
        );
        if let Some(peer) = peer {
            ensure!(
                peer < self.world,
                "NCCL rank={} world_size={} op={op}: peer/root={peer} outside group",
                self.rank,
                self.world
            );
        }
        if self.enabled {
            let sequence = self.sequence.fetch_add(1, Ordering::Relaxed);
            tracing::info!(target: "metrale::comm", rank = self.rank, world_size = self.world,
                sequence, op, ?dtype, count = bytes / dtype.width(), bytes, stream, ?peer,
                phase = "submit", "NCCL host submission (not completion)");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn malformed_groups_and_diagnostic_switch_fail_before_nccl() {
        for (rank, world) in [(0, 0), (8, 8), (0, i32::MAX as usize + 1)] {
            assert!(Diagnostics::new(None, rank, world).is_err());
        }
        assert!(Diagnostics::new(Some("yes"), 0, 8).is_err());
    }

    #[test]
    fn partial_bf16_element_is_refused_even_with_diagnostics_off() {
        let d = Diagnostics::new(Some("0"), 3, 8).unwrap();
        let err = d
            .submit("all_reduce", Dtype::Bf16, 3, 0, None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("rank=3") && err.contains("bytes=3") && err.contains("Bf16"));
        assert!(d.submit("all_reduce", Dtype::Bf16, 4, 0, None).is_ok());
    }

    #[test]
    fn peer_bounds_and_submission_sequence_are_local() {
        let d = Diagnostics::new(Some("1"), 0, 8).unwrap();
        assert!(d.submit("broadcast", Dtype::U8, 3, 0, Some(8)).is_err());
        d.submit("broadcast", Dtype::U8, 3, 0, Some(7)).unwrap();
        d.submit("barrier", Dtype::F32, 0, 0, None).unwrap();
        assert_eq!(d.sequence.load(Ordering::Relaxed), 2);
    }
}
