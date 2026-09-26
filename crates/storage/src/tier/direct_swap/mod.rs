// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `DirectSwapFile`, the NVMe cold tier, one implementation per platform.
//!
//! Owner: storage, tiered-cache core.
//! Invariants:
//! - Record `disk_slot` is `record_bytes` long at offset `disk_slot * record_bytes`,
//!   and `record_bytes` is a non-zero multiple of 4096 (`validate_record_bytes`).
//!
//! `unix`: `pread`/`pwrite` on a file opened `O_DIRECT` on Linux and buffered on other
//! unix, staging through an aligned bounce when the caller's buffer is not 4 KiB
//! aligned. `windows`: buffered `seek_read`/`seek_write`.

#[cfg(unix)]
mod unix;
#[cfg(unix)]
pub use unix::DirectSwapFile;

#[cfg(windows)]
mod windows;
#[cfg(windows)]
pub use windows::DirectSwapFile;

/// 2026-09-25: `record_bytes` must be a non-zero multiple of 4096; both
/// implementations check it with this one message.
#[allow(dead_code)]
pub(crate) fn validate_record_bytes(record_bytes: usize) -> anyhow::Result<()> {
    if record_bytes == 0 || !record_bytes.is_multiple_of(4096) {
        anyhow::bail!(
            "DirectSwapFile: record_bytes ({record_bytes}) must be a non-zero 4 KiB multiple"
        );
    }
    Ok(())
}
