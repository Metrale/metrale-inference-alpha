// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Swap files for the `--swap-space-gb` spill of sequence state.
//!
//! Callers write and read the file contents; this module creates, opens and
//! removes the `swap_<id>.bin` files in one directory and counts the bytes
//! recorded against a budget. `has_space` reports the budget; nothing here
//! enforces it.
//!
//! Owner: cache.
//! Invariants: file ids are handed out in increasing order from 0, so a
//! manager never reuses one.

use anyhow::Result;
use std::fs;
use std::io::{BufReader, BufWriter};
use std::path::PathBuf;

/// 2026-09-25: The swap files of one directory and their recorded size.
pub struct KvSpillManager {
    spill_dir: PathBuf,
    next_id: u64,
    max_bytes: u64,
    used_bytes: u64,
}

impl KvSpillManager {
    /// 2026-09-25: A manager for `spill_dir` with a `max_bytes` budget.
    /// Creates the directory if it is missing; otherwise deletes the `swap_*`
    /// files already in it, ignoring failures to delete.
    pub fn new(spill_dir: PathBuf, max_bytes: u64) -> Result<Self> {
        if spill_dir.exists() {
            for entry in fs::read_dir(&spill_dir)? {
                let entry = entry?;
                if entry.file_name().to_string_lossy().starts_with("swap_") {
                    let _ = fs::remove_file(entry.path());
                }
            }
        } else {
            fs::create_dir_all(&spill_dir)?;
        }
        Ok(Self {
            spill_dir,
            next_id: 0,
            max_bytes,
            used_bytes: 0,
        })
    }

    /// 2026-09-25: Create the next swap file; returns its id and a buffered
    /// writer.
    pub fn create_file(&mut self) -> Result<(u64, BufWriter<fs::File>)> {
        let id = self.next_id;
        self.next_id += 1;
        let path = self.file_path(id);
        let file = fs::File::create(&path)?;
        Ok((id, BufWriter::new(file)))
    }

    /// 2026-09-25: Open swap file `id` for buffered reading.
    pub fn open_file(&self, id: u64) -> Result<BufReader<fs::File>> {
        let path = self.file_path(id);
        let file = fs::File::open(&path)?;
        Ok(BufReader::new(file))
    }

    /// 2026-09-25: Delete swap file `id` and subtract its size from the
    /// recorded usage.
    pub fn remove_file(&mut self, id: u64) -> Result<()> {
        let path = self.file_path(id);
        let size = fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        fs::remove_file(&path)?;
        self.used_bytes = self.used_bytes.saturating_sub(size);
        Ok(())
    }

    /// 2026-09-25: Add file `id`'s current size to the recorded usage. Call
    /// it once, after the writer is flushed; a missing file adds nothing.
    pub fn record_usage(&mut self, id: u64) {
        let path = self.file_path(id);
        if let Ok(meta) = fs::metadata(&path) {
            self.used_bytes += meta.len();
        }
    }

    /// 2026-09-25: Whether `estimated_bytes` more would stay within the budget.
    pub fn has_space(&self, estimated_bytes: u64) -> bool {
        self.used_bytes + estimated_bytes <= self.max_bytes
    }

    /// 2026-09-25: Recorded usage in bytes.
    pub fn used_bytes(&self) -> u64 {
        self.used_bytes
    }

    fn file_path(&self, id: u64) -> PathBuf {
        self.spill_dir.join(format!("swap_{id}.bin"))
    }
}

impl Drop for KvSpillManager {
    fn drop(&mut self) {
        // 2026-09-25: Best effort: delete the `swap_*` files, then the
        // directory.
        if let Ok(entries) = fs::read_dir(&self.spill_dir) {
            for entry in entries.flatten() {
                if entry.file_name().to_string_lossy().starts_with("swap_") {
                    let _ = fs::remove_file(entry.path());
                }
            }
        }
        // 2026-09-25: The server names the directory per process
        // (`metrale-swap-<pid>`), so leaving it would leave one per run.
        // `remove_dir` refuses a non-empty directory, so a file that is not
        // ours survives.
        let _ = fs::remove_dir(&self.spill_dir);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn temp_dir() -> PathBuf {
        std::env::temp_dir().join(format!(
            "metrale_spill_test_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    #[test]
    fn test_create_open_remove_lifecycle() {
        let dir = temp_dir();
        let mut mgr = KvSpillManager::new(dir.clone(), 1024 * 1024).unwrap();

        let (id, mut writer) = mgr.create_file().unwrap();
        writer.write_all(&[1u8; 256]).unwrap();
        writer.flush().unwrap();
        drop(writer);
        mgr.record_usage(id);
        assert_eq!(mgr.used_bytes(), 256);

        let mut reader = mgr.open_file(id).unwrap();
        let mut buf = Vec::new();
        std::io::Read::read_to_end(&mut reader, &mut buf).unwrap();
        assert_eq!(buf.len(), 256);
        assert!(buf.iter().all(|&b| b == 1));

        mgr.remove_file(id).unwrap();
        assert_eq!(mgr.used_bytes(), 0);
        assert!(!dir.join("swap_0.bin").exists());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_has_space() {
        let dir = temp_dir();
        let mut mgr = KvSpillManager::new(dir.clone(), 512).unwrap();

        assert!(mgr.has_space(512));
        assert!(!mgr.has_space(513));

        let (id, mut writer) = mgr.create_file().unwrap();
        writer.write_all(&[0u8; 256]).unwrap();
        writer.flush().unwrap();
        drop(writer);
        mgr.record_usage(id);

        assert!(mgr.has_space(256));
        assert!(!mgr.has_space(257));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_stale_cleanup_on_new() {
        let dir = temp_dir();
        fs::create_dir_all(&dir).unwrap();

        fs::write(dir.join("swap_99.bin"), [0u8; 64]).unwrap();
        assert!(dir.join("swap_99.bin").exists());

        let _mgr = KvSpillManager::new(dir.clone(), 1024).unwrap();
        assert!(!dir.join("swap_99.bin").exists());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_drop_cleanup() {
        let dir = temp_dir();
        {
            let mut mgr = KvSpillManager::new(dir.clone(), 1024).unwrap();
            let (_, mut writer) = mgr.create_file().unwrap();
            writer.write_all(&[0u8; 32]).unwrap();
            writer.flush().unwrap();
            drop(writer);
        }
        assert!(!dir.join("swap_0.bin").exists());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_sequential_ids() {
        let dir = temp_dir();
        let mut mgr = KvSpillManager::new(dir.clone(), 1024 * 1024).unwrap();

        let (id0, w0) = mgr.create_file().unwrap();
        drop(w0);
        let (id1, w1) = mgr.create_file().unwrap();
        drop(w1);

        assert_eq!(id0, 0);
        assert_eq!(id1, 1);

        let _ = fs::remove_dir_all(&dir);
    }
}

#[cfg(test)]
mod dir_cleanup_tests {
    use super::*;

    #[test]
    fn dropping_the_manager_leaves_no_directory_behind() {
        // 2026-09-25: Drop removes the directory it created.
        let dir = std::env::temp_dir().join(format!(
            "metrale_spill_cleanup_{}_{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        {
            let mgr = KvSpillManager::new(dir.clone(), 1024).expect("constructs");
            assert!(dir.exists(), "the manager creates its directory");
            drop(mgr);
        }
        assert!(!dir.exists(), "and removes it again");
    }

    #[test]
    fn a_directory_holding_someone_elses_file_is_left_alone() {
        // 2026-09-25: A directory holding a file that is not `swap_*` survives
        // the drop, and so does the file.
        let dir = std::env::temp_dir().join(format!(
            "metrale_spill_shared_{}_{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&dir).expect("mkdir");
        std::fs::write(dir.join("not-ours.txt"), b"keep me").expect("write");
        {
            let _mgr = KvSpillManager::new(dir.clone(), 1024).expect("constructs");
        }
        assert!(dir.exists(), "a directory with a foreign file survives");
        assert!(dir.join("not-ours.txt").exists(), "and so does the file");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
