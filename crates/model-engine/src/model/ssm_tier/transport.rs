// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The transports under the SSM spill stores: offset-addressed
//! [`SnapshotTransport`] and key-addressed [`PagingTransport`], with their
//! in-process, RDMA and local-file implementations.
//!
//! Transports move host bytes only. `SsmSnapshotPool::spill_slot` gathers the
//! device state and synchronises before `put`; `fault_in_slot` scatters and
//! synchronises after `get`.
//!
//! Owner: model-engine (SSM snapshot tier).
//! Invariants:
//! - `FileSnapshotArena` never reads or writes past its `capacity`.

use anyhow::Result;
use parking_lot::Mutex;

/// 2026-09-25: A flat byte arena addressed by absolute offset. Implemented by
/// `metrale_storage::RdmaSnapshotArena` (peer RAM over RDMA verbs),
/// [`FileSnapshotArena`] (a local file) and [`MockSnapshotTransport`] (tests).
#[allow(dead_code)]
pub(crate) trait SnapshotTransport: Send + Sync {
    /// 2026-09-25: Write `bytes` at absolute `offset`. Callers keep
    /// `offset + bytes.len()` within the arena.
    fn write_blob(&self, offset: u64, bytes: &[u8]) -> Result<()>;
    /// 2026-09-25: Read `out.len()` bytes at absolute `offset` into `out`.
    fn read_blob(&self, offset: u64, out: &mut [u8]) -> Result<()>;
}

/// 2026-09-25: An in-process `Vec<u8>` arena for tests of the offset-addressed stores.
#[allow(dead_code)]
pub(crate) struct MockSnapshotTransport {
    arena: Mutex<Vec<u8>>,
}

#[allow(dead_code)]
impl MockSnapshotTransport {
    pub(crate) fn new(capacity_bytes: usize) -> Self {
        Self {
            arena: Mutex::new(vec![0u8; capacity_bytes]),
        }
    }
}

impl SnapshotTransport for MockSnapshotTransport {
    fn write_blob(&self, offset: u64, bytes: &[u8]) -> Result<()> {
        let mut a = self.arena.lock();
        let off = offset as usize;
        a[off..off + bytes.len()].copy_from_slice(bytes);
        Ok(())
    }
    fn read_blob(&self, offset: u64, out: &mut [u8]) -> Result<()> {
        let a = self.arena.lock();
        let off = offset as usize;
        out.copy_from_slice(&a[off..off + out.len()]);
        Ok(())
    }
}

/// 2026-09-25: A key-addressed store on a peer that owns residency, used by
/// `PagingSnapshotStore`. Implemented by `metrale_storage::RdmaSnapshotArena`
/// and, in `paging_isolation_tests`, by a shared in-process mock peer.
pub(crate) trait PagingTransport: Send + Sync {
    /// 2026-09-25: Store `bytes` under `key` on the peer. There is no refusal
    /// result; a failure is an `Err`.
    fn paging_put(&self, key: u64, bytes: &[u8]) -> Result<()>;
    /// 2026-09-25: Fetch `key` into `out`. `Ok(false)` is a miss.
    fn paging_get(&self, key: u64, out: &mut [u8]) -> Result<bool>;
    /// 2026-09-25: Drop `key` from the peer.
    fn paging_remove(&self, key: u64) -> Result<()>;
}

// 2026-09-25: Delegation to the arena's inherent paging methods, called by their
// full path so the inherent method, not this trait method, runs. A put/get
// transposition here would pass every mock-backed test.
impl PagingTransport for metrale_storage::RdmaSnapshotArena {
    fn paging_put(&self, key: u64, bytes: &[u8]) -> Result<()> {
        metrale_storage::RdmaSnapshotArena::paging_put(self, key, bytes)
    }
    fn paging_get(&self, key: u64, out: &mut [u8]) -> Result<bool> {
        metrale_storage::RdmaSnapshotArena::paging_get(self, key, out)
    }
    fn paging_remove(&self, key: u64) -> Result<()> {
        metrale_storage::RdmaSnapshotArena::paging_remove(self, key)
    }
}

// 2026-09-25: Without RDMA verbs, `RdmaSnapshotArena` is a stub whose `connect`
// always errors, so this impl is never reached in such a build.
impl SnapshotTransport for metrale_storage::RdmaSnapshotArena {
    fn write_blob(&self, offset: u64, bytes: &[u8]) -> Result<()> {
        self.write(offset, bytes)
    }
    fn read_blob(&self, offset: u64, out: &mut [u8]) -> Result<()> {
        self.read(offset, out)
    }
}

/// 2026-09-25: A fixed-size local file used as an offset-addressed arena, with
/// buffered (not O_DIRECT) I/O. Reads and writes are positional and share no
/// file cursor, so concurrent calls need no lock of their own.
#[allow(dead_code)]
pub(crate) struct FileSnapshotArena {
    file: std::fs::File,
    capacity: u64,
}

#[allow(dead_code)]
impl FileSnapshotArena {
    /// 2026-09-25: Create or truncate `dir/metrale-decode-ring.<pid>.arena` at
    /// exactly `capacity` bytes. The pid keeps two servers on one box apart, and
    /// truncation means nothing is carried over from an earlier run.
    pub(crate) fn create(dir: &str, capacity: u64) -> Result<Self> {
        std::fs::create_dir_all(dir)?;
        let path = std::path::Path::new(dir)
            .join(format!("metrale-decode-ring.{}.arena", std::process::id()));
        let file = std::fs::OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(true)
            .open(&path)?;
        file.set_len(capacity)?;
        Ok(Self { file, capacity })
    }
}

impl SnapshotTransport for FileSnapshotArena {
    fn write_blob(&self, offset: u64, bytes: &[u8]) -> Result<()> {
        if offset + bytes.len() as u64 > self.capacity {
            anyhow::bail!(
                "FileSnapshotArena write {offset}+{} exceeds capacity {}",
                bytes.len(),
                self.capacity
            );
        }
        write_all_at(&self.file, bytes, offset)
    }
    fn read_blob(&self, offset: u64, out: &mut [u8]) -> Result<()> {
        if offset + out.len() as u64 > self.capacity {
            anyhow::bail!(
                "FileSnapshotArena read {offset}+{} exceeds capacity {}",
                out.len(),
                self.capacity
            );
        }
        read_exact_at(&self.file, out, offset)
    }
}

// 2026-09-25: Positional file I/O, one implementation per platform. The bounds
// checks stay above in shared code. `pread`/`pwrite` and `seek_read`/`seek_write`
// are positional and leave the file cursor alone, which the arena relies on.
#[cfg(unix)]
fn write_all_at(f: &std::fs::File, bytes: &[u8], offset: u64) -> Result<()> {
    use std::os::unix::fs::FileExt;
    f.write_all_at(bytes, offset)?;
    Ok(())
}

#[cfg(unix)]
fn read_exact_at(f: &std::fs::File, out: &mut [u8], offset: u64) -> Result<()> {
    use std::os::unix::fs::FileExt;
    f.read_exact_at(out, offset)?;
    Ok(())
}

// 2026-09-25: Windows has no `write_all_at`/`read_exact_at`. `seek_write`/`seek_read`
// may transfer short, so loop.
#[cfg(windows)]
fn write_all_at(f: &std::fs::File, bytes: &[u8], offset: u64) -> Result<()> {
    use std::os::windows::fs::FileExt;
    let (mut off, mut done) = (offset, 0usize);
    while done < bytes.len() {
        let n = f.seek_write(&bytes[done..], off)?;
        if n == 0 {
            anyhow::bail!("seek_write wrote 0 bytes at offset {off}");
        }
        done += n;
        off += n as u64;
    }
    Ok(())
}

#[cfg(windows)]
fn read_exact_at(f: &std::fs::File, out: &mut [u8], offset: u64) -> Result<()> {
    use std::os::windows::fs::FileExt;
    let (mut off, mut done) = (offset, 0usize);
    let total = out.len();
    while done < total {
        let n = f.seek_read(&mut out[done..], off)?;
        if n == 0 {
            anyhow::bail!("seek_read hit EOF after {done} of {total} bytes at offset {off}");
        }
        done += n;
        off += n as u64;
    }
    Ok(())
}
