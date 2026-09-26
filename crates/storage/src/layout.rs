// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: On-disk layout of the high-speed-swap KV tier: one pre-allocated file
//! per layer, `layer_{:05}.kv` under `HighSpeedSwapConfig::dir` (`--high-speed-swap-dir`).
//!
//! Owner: storage, high-speed swap.
//! Invariants:
//! - A `Layout` holds one open file per layer of `spec`, each at least
//!   `spec.bytes_per_layer()` long; `create` and `open` return an error otherwise.
//!
//! On Linux the files are opened `O_DIRECT`. Space is reserved with
//! `posix_fallocate` on unix and `set_len` on Windows.

use anyhow::{Context, Result};
use std::fs::{File, OpenOptions};
#[cfg(unix)]
use std::os::fd::{AsRawFd, RawFd};
use std::path::{Path, PathBuf};

use crate::group::{GroupKey, GroupLayout};

pub struct Layout {
    pub dir: PathBuf,
    pub spec: GroupLayout,
    /// 2026-09-25: One file per layer. A `File`, because `PosixBackend` does
    /// positional I/O on it (`tier::pio`) on every platform.
    files: Vec<File>,
}

impl Layout {
    pub fn create(dir: &Path, spec: GroupLayout) -> Result<Self> {
        std::fs::create_dir_all(dir).with_context(|| format!("mkdir {}", dir.display()))?;
        let mut files = Vec::with_capacity(spec.num_layers as usize);
        for layer in 0..spec.num_layers {
            let p = dir.join(format!("layer_{layer:05}.kv"));
            let mut opts = OpenOptions::new();
            opts.read(true).write(true).create(true).truncate(false);
            set_direct_flag(&mut opts);
            let f = opts
                .open(&p)
                .with_context(|| format!("open {}", p.display()))?;
            preallocate(&f, spec.bytes_per_layer())
                .with_context(|| format!("preallocate {}", p.display()))?;
            files.push(f);
        }
        Ok(Self {
            dir: dir.to_path_buf(),
            spec,
            files,
        })
    }

    /// 2026-09-25: Open an existing layout. A missing or undersized layer file is an
    /// error.
    pub fn open(dir: &Path, spec: GroupLayout) -> Result<Self> {
        let mut files = Vec::with_capacity(spec.num_layers as usize);
        for layer in 0..spec.num_layers {
            let p = dir.join(format!("layer_{layer:05}.kv"));
            let mut opts = OpenOptions::new();
            opts.read(true).write(true);
            set_direct_flag(&mut opts);
            let f = opts
                .open(&p)
                .with_context(|| format!("open {}", p.display()))?;
            let len = f.metadata()?.len();
            if len < spec.bytes_per_layer() {
                anyhow::bail!(
                    "layer file {} is undersized: {} < {}",
                    p.display(),
                    len,
                    spec.bytes_per_layer()
                );
            }
            files.push(f);
        }
        Ok(Self {
            dir: dir.to_path_buf(),
            spec,
            files,
        })
    }

    /// 2026-09-25: Raw fd of a layer file, for `IoUringBackend` and
    /// `PosixBackend::drop_pagecache`.
    #[cfg(unix)]
    pub fn fd(&self, layer: u32) -> RawFd {
        self.files[layer as usize].as_raw_fd()
    }

    /// 2026-09-25: The layer file, for `PosixBackend`'s positional I/O; the only
    /// accessor on Windows.
    pub fn file(&self, layer: u32) -> &File {
        &self.files[layer as usize]
    }

    pub fn offset(&self, key: GroupKey) -> u64 {
        self.spec.file_offset(key)
    }

    pub fn group_bytes(&self) -> u64 {
        self.spec.group_bytes()
    }
}

#[cfg(target_os = "linux")]
fn set_direct_flag(opts: &mut OpenOptions) {
    use std::os::unix::fs::OpenOptionsExt;
    opts.custom_flags(libc::O_DIRECT);
}

#[cfg(not(target_os = "linux"))]
fn set_direct_flag(_opts: &mut OpenOptions) {}

#[cfg(unix)]
fn preallocate(file: &File, size: u64) -> Result<()> {
    // 2026-09-25: This grows the file to `size`; the round-trip test checks the
    // length.
    let fd = file.as_raw_fd();
    let res = unsafe { libc::posix_fallocate(fd, 0, size as libc::off_t) };
    if res != 0 {
        anyhow::bail!("posix_fallocate({size}) failed: {res}");
    }
    Ok(())
}

#[cfg(windows)]
fn preallocate(file: &File, size: u64) -> Result<()> {
    file.set_len(size)
        .with_context(|| format!("set_len({size})"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::group::{GroupKey, KvKind};

    #[test]
    fn create_open_round_trip() {
        let tmp = tempdir();
        let spec = GroupLayout::new(2, 4, 2, 16, 128, 2, 4096);
        {
            let l = Layout::create(&tmp, spec).unwrap();
            assert_eq!(l.spec.num_layers, 2);
            let p = tmp.join("layer_00000.kv");
            let len = std::fs::metadata(&p).unwrap().len();
            assert_eq!(len, spec.bytes_per_layer());
        }
        {
            let l = Layout::open(&tmp, spec).unwrap();
            let off = l.offset(GroupKey::new(0, 1, 1, KvKind::V));
            assert_eq!(off, spec.file_offset(GroupKey::new(0, 1, 1, KvKind::V)));
        }
        std::fs::remove_dir_all(&tmp).ok();
    }

    fn tempdir() -> PathBuf {
        let p = std::env::temp_dir().join(format!("metrale-storage-test-{}", std::process::id()));
        std::fs::create_dir_all(&p).unwrap();
        p
    }
}
