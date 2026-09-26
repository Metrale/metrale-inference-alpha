// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `HighSpeedSwapConfig`, the settings of the high-speed swap, and
//! their range checks. `HighSpeedSwap::new_on_stream` runs
//! `validate_and_prepare` before it builds anything.
//!
//! Owner: metrale-storage high-speed swap.
//! Invariants: none beyond the types.

use anyhow::{Result, bail};
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Clone, Debug, Deserialize)]
pub struct HighSpeedSwapConfig {
    /// 2026-09-25: Directory of the per-layer KV files.
    pub dir: PathBuf,
    /// 2026-09-25: Disk budget in bytes; the server passes
    /// `--high-speed-swap-gb` × 2^30. `validate` requires it to be non-zero,
    /// and nothing in this crate reads it otherwise.
    pub bytes: u64,
    /// 2026-09-25: Scratch-pool slot count; also the attention tile capacity
    /// and the eviction policy's capacity.
    pub resident_blocks: u32,
    /// 2026-09-25: Predictor low-rank dimension r.
    pub rank: u32,
    /// 2026-09-25: io_uring queue depth; the portable backend ignores it.
    pub qd: u32,
    /// 2026-09-25: Capture the per-layer body in a CUDA graph and replay it.
    /// The server sets it unless `--no-high-speed-swap-graph` is given.
    pub graph: bool,
    /// 2026-09-25: Seed of the predictor's projection matrix.
    pub projection_seed: u64,
}

impl HighSpeedSwapConfig {
    /// 2026-09-25: Range checks: `bytes` and `resident_blocks` non-zero, `rank`
    /// in 1..=128, `qd` in 1..=64. The directory is not checked here.
    pub fn validate(&self) -> Result<()> {
        if self.bytes == 0 {
            bail!("--high-speed-swap-bytes must be > 0");
        }
        if self.resident_blocks == 0 {
            bail!("--high-speed-swap-resident-blocks must be > 0");
        }
        if self.rank == 0 || self.rank > 128 {
            bail!(
                "--high-speed-swap-rank must be in 1..=128, got {}",
                self.rank
            );
        }
        if self.qd == 0 || self.qd > 64 {
            bail!("--high-speed-swap-qd must be in 1..=64, got {}", self.qd);
        }
        Ok(())
    }

    /// 2026-09-25: `validate`, then create the directory if it is missing.
    pub fn validate_and_prepare(&self) -> Result<()> {
        self.validate()?;
        std::fs::create_dir_all(&self.dir)
            .map_err(|e| anyhow::anyhow!("create {}: {e}", self.dir.display()))?;
        // 2026-09-25: The server checks the directory against the
        // `--swap-space-gb` path (`validate_head_high_speed_swap`).
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> HighSpeedSwapConfig {
        HighSpeedSwapConfig {
            dir: PathBuf::from("/tmp/metrale-hss-cfg"),
            bytes: 64 << 30,
            resident_blocks: 8192,
            rank: 32,
            qd: 8,
            graph: true,
            projection_seed: 0xCAFE_F00D,
        }
    }

    #[test]
    fn happy_path() {
        cfg().validate().unwrap();
    }

    #[test]
    fn rejects_zero_bytes() {
        let mut c = cfg();
        c.bytes = 0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn rejects_zero_resident_blocks() {
        let mut c = cfg();
        c.resident_blocks = 0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn rejects_out_of_range_rank() {
        let mut c = cfg();
        c.rank = 0;
        assert!(c.validate().is_err());
        c.rank = 129;
        assert!(c.validate().is_err());
    }

    #[test]
    fn rejects_out_of_range_qd() {
        let mut c = cfg();
        c.qd = 0;
        assert!(c.validate().is_err());
        c.qd = 65;
        assert!(c.validate().is_err());
    }
}
