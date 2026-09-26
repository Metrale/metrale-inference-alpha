// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Unix `DirectSwapFile`: `pread`/`pwrite` on a file opened `O_DIRECT` on
//! Linux, with an aligned bounce for unaligned buffers.
//!
//! Owner: storage, tiered-cache core.
//! Invariants:
//! - A read or write moves exactly `record_bytes`, or returns an error.

use std::fs::OpenOptions;
use std::os::fd::{AsRawFd, OwnedFd};
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;

use anyhow::{Result, bail};

use crate::tier::direct_swap::validate_record_bytes;
use crate::tier::traits::SwapStore;

/// 2026-09-25: `O_DIRECT` on Linux; other unix targets open the file buffered.
#[cfg(target_os = "linux")]
const DIRECT_FLAGS: i32 = libc::O_DIRECT;
#[cfg(not(target_os = "linux"))]
const DIRECT_FLAGS: i32 = 0;

/// 2026-09-25: Fixed-stride swap file: record `disk_slot` at
/// `disk_slot * record_bytes`, with `record_bytes` a non-zero 4 KiB multiple.
///
/// A buffer that is not 4 KiB-aligned is staged through an internal aligned bounce,
/// one extra copy of the record. `Residency` passes its page-aligned scratch.
///
/// An empty swap file does not mean a write failed: `Residency` writes a record only
/// when its arena has no free slot. The file then reaches
/// `(highest disk_slot + 1) * record_bytes` and never shrinks, since this type keeps
/// the no-op `discard_record` default.
pub struct DirectSwapFile {
    fd: OwnedFd,
    record_bytes: usize,
    /// 2026-09-25: Aligned bounce for callers whose buffer is not 4 KiB-aligned.
    bounce: AlignedBuf,
}

impl DirectSwapFile {
    pub fn create(path: &Path, record_bytes: usize) -> Result<Self> {
        validate_record_bytes(record_bytes)?;
        let f = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .custom_flags(DIRECT_FLAGS)
            .open(path)
            .map_err(|e| anyhow::anyhow!("open O_DIRECT {}: {e}", path.display()))?;
        Ok(Self {
            fd: OwnedFd::from(f),
            record_bytes,
            bounce: AlignedBuf::new(record_bytes),
        })
    }

    fn offset(&self, disk_slot: usize) -> libc::off_t {
        (disk_slot as u64 * self.record_bytes as u64) as libc::off_t
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
        let off = self.offset(disk_slot);
        let src = if is_aligned(bytes.as_ptr()) {
            bytes.as_ptr()
        } else {
            self.bounce.as_mut_slice().copy_from_slice(bytes);
            self.bounce.ptr()
        };
        let n = unsafe {
            libc::pwrite(
                self.fd.as_raw_fd(),
                src as *const libc::c_void,
                self.record_bytes,
                off,
            )
        };
        if n != self.record_bytes as isize {
            bail!("pwrite record {disk_slot} returned {n}, errno {}", errno());
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
        let off = self.offset(disk_slot);
        if is_aligned(out.as_ptr()) {
            let n = unsafe {
                libc::pread(
                    self.fd.as_raw_fd(),
                    out.as_mut_ptr() as *mut libc::c_void,
                    self.record_bytes,
                    off,
                )
            };
            if n != self.record_bytes as isize {
                bail!("pread record {disk_slot} returned {n}, errno {}", errno());
            }
        } else {
            // 2026-09-25: Stage through the bounce, written through a raw pointer from
            // `&self`. `AlignedBuf` is not `Sync`, so neither is this type, and no two
            // `read_record` calls run at once.
            let bp = self.bounce.ptr();
            let n = unsafe {
                libc::pread(
                    self.fd.as_raw_fd(),
                    bp as *mut libc::c_void,
                    self.record_bytes,
                    off,
                )
            };
            if n != self.record_bytes as isize {
                bail!(
                    "pread(bounce) record {disk_slot} returned {n}, errno {}",
                    errno()
                );
            }
            unsafe {
                std::ptr::copy_nonoverlapping(bp, out.as_mut_ptr(), self.record_bytes);
            }
        }
        Ok(())
    }
}

fn errno() -> i32 {
    std::io::Error::last_os_error().raw_os_error().unwrap_or(0)
}

fn is_aligned(p: *const u8) -> bool {
    (p as usize) & 0xfff == 0
}

/// 2026-09-25: A 4 KiB-aligned heap buffer from `posix_memalign`, for O_DIRECT
/// staging; a failed allocation panics.
struct AlignedBuf {
    ptr: *mut u8,
    len: usize,
}
unsafe impl Send for AlignedBuf {}
impl AlignedBuf {
    fn new(len: usize) -> Self {
        let mut p: *mut libc::c_void = std::ptr::null_mut();
        let rc = unsafe { libc::posix_memalign(&mut p, 4096, len) };
        assert!(
            rc == 0 && !p.is_null(),
            "posix_memalign({len}) failed rc={rc}"
        );
        Self {
            ptr: p as *mut u8,
            len,
        }
    }
    fn ptr(&self) -> *mut u8 {
        self.ptr
    }
    fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}
impl Drop for AlignedBuf {
    fn drop(&mut self) {
        unsafe { libc::free(self.ptr as *mut libc::c_void) }
    }
}
