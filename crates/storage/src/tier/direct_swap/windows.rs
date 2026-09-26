// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Windows `DirectSwapFile`: buffered positional I/O on the caller's
//! buffer, with no alignment requirement and no bounce.
//!
//! Owner: storage, tiered-cache core.
//! Invariants:
//! - A read or write moves exactly `record_bytes`, or returns an error.

use std::fs::{File, OpenOptions};
use std::os::windows::fs::FileExt;
use std::path::Path;

use anyhow::{Result, bail};

use crate::tier::direct_swap::validate_record_bytes;
use crate::tier::traits::SwapStore;

pub struct DirectSwapFile {
    file: File,
    record_bytes: usize,
}

impl DirectSwapFile {
    pub fn create(path: &Path, record_bytes: usize) -> Result<Self> {
        validate_record_bytes(record_bytes)?;
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .map_err(|e| anyhow::anyhow!("open {}: {e}", path.display()))?;
        Ok(Self { file, record_bytes })
    }

    fn offset(&self, disk_slot: usize) -> u64 {
        disk_slot as u64 * self.record_bytes as u64
    }
}

impl SwapStore for DirectSwapFile {
    fn record_bytes(&self) -> usize {
        self.record_bytes
    }

    fn write_record(&mut self, disk_slot: usize, bytes: &[u8]) -> Result<()> {
        if bytes.len() != self.record_bytes {
            bail!(
                "write_record: {} bytes, expected {}",
                bytes.len(),
                self.record_bytes
            );
        }
        // 2026-09-25: `seek_write` may write short, so loop until the record is
        // written.
        let mut off = self.offset(disk_slot);
        let mut written = 0usize;
        while written < bytes.len() {
            let n = self
                .file
                .seek_write(&bytes[written..], off)
                .map_err(|e| anyhow::anyhow!("seek_write record {disk_slot}: {e}"))?;
            if n == 0 {
                bail!("seek_write record {disk_slot} wrote 0 bytes at offset {off}");
            }
            written += n;
            off += n as u64;
        }
        Ok(())
    }

    fn read_record(&self, disk_slot: usize, out: &mut [u8]) -> Result<()> {
        if out.len() != self.record_bytes {
            bail!(
                "read_record: {} bytes, expected {}",
                out.len(),
                self.record_bytes
            );
        }
        let mut off = self.offset(disk_slot);
        let mut read = 0usize;
        while read < out.len() {
            let n = self
                .file
                .seek_read(&mut out[read..], off)
                .map_err(|e| anyhow::anyhow!("seek_read record {disk_slot}: {e}"))?;
            // 2026-09-25: EOF before the record is complete means the slot was never
            // written; that is an error, not a partial buffer.
            if n == 0 {
                bail!(
                    "seek_read record {disk_slot} hit EOF after {read} of {} bytes",
                    out.len()
                );
            }
            read += n;
            off += n as u64;
        }
        Ok(())
    }
}
