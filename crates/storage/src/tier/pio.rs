// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Positional file reads and writes at an explicit offset:
//! `FileExt::read_at`/`write_at` on unix, `seek_read`/`seek_write` on Windows.
//! Bounds checks and record layout stay with the callers (`backend/posix.rs`,
//! `expert_pack_fs.rs`, `expert_tier.rs`, `ngram_cache_fault.rs`).
//!
//! Owner: storage (tier).
//! Invariants: a transfer shorter than requested is retried from where it
//! stopped; a zero-byte transfer ends the loop.

use std::fs::File;
use std::io;

/// 2026-09-25: Write all of `buf` at `offset`. A write of zero bytes is a
/// `WriteZero` error.
pub fn write_all_at(f: &File, buf: &[u8], offset: u64) -> io::Result<()> {
    let (mut off, mut done) = (offset, 0usize);
    while done < buf.len() {
        let n = write_at(f, &buf[done..], off)?;
        if n == 0 {
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                format!("positional write returned 0 bytes at offset {off}"),
            ));
        }
        done += n;
        off += n as u64;
    }
    Ok(())
}

/// 2026-09-25: Fill `buf` from `offset`. A zero-byte read before `buf` is full
/// is an `UnexpectedEof` error, so the tail of `buf` is never left stale.
pub fn read_exact_at(f: &File, buf: &mut [u8], offset: u64) -> io::Result<()> {
    let (mut off, mut done) = (offset, 0usize);
    let total = buf.len();
    while done < total {
        let n = read_at(f, &mut buf[done..], off)?;
        if n == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("positional read hit EOF after {done} of {total} bytes at offset {off}"),
            ));
        }
        done += n;
        off += n as u64;
    }
    Ok(())
}

/// 2026-09-25: Fill as much of `buf` as the file holds from `offset`, and
/// succeed if at least the first `needed` bytes arrived. Returns the byte count
/// read. Fewer than `needed` is an `UnexpectedEof` error.
///
/// For whole-block reads at the end of a file (`ngram_cache_fault.rs`): a
/// safetensors file is `8 + header + tensors` long and nothing pads it to a
/// block, so the block holding its last rows runs past EOF, where
/// [`read_exact_at`] would fail although every requested row is in the file.
pub fn read_at_least_at(f: &File, buf: &mut [u8], offset: u64, needed: usize) -> io::Result<usize> {
    debug_assert!(needed <= buf.len());
    let (mut off, mut done) = (offset, 0usize);
    while done < buf.len() {
        let n = read_at(f, &mut buf[done..], off)?;
        if n == 0 {
            break;
        }
        done += n;
        off += n as u64;
    }
    if done < needed {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            format!(
                "positional read got {done} bytes at offset {offset}, needed {needed} \
                 — the file is shorter than the data it is supposed to hold"
            ),
        ));
    }
    Ok(done)
}

#[cfg(unix)]
fn write_at(f: &File, buf: &[u8], offset: u64) -> io::Result<usize> {
    use std::os::unix::fs::FileExt;
    f.write_at(buf, offset)
}

#[cfg(unix)]
fn read_at(f: &File, buf: &mut [u8], offset: u64) -> io::Result<usize> {
    use std::os::unix::fs::FileExt;
    f.read_at(buf, offset)
}

#[cfg(windows)]
fn write_at(f: &File, buf: &[u8], offset: u64) -> io::Result<usize> {
    use std::os::windows::fs::FileExt;
    f.seek_write(buf, offset)
}

#[cfg(windows)]
fn read_at(f: &File, buf: &mut [u8], offset: u64) -> io::Result<usize> {
    use std::os::windows::fs::FileExt;
    f.seek_read(buf, offset)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-09-25: A file whose length is not a multiple of the block size,
    /// read in whole blocks: `read_at_least_at` succeeds when the needed bytes
    /// exist and fails when they do not, and `read_exact_at` fails.
    #[test]
    fn a_block_read_past_eof_succeeds_for_the_bytes_that_exist() {
        let dir = std::env::temp_dir().join(format!("metrale_pio_tail_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tail.bin");
        let len = 4096usize + 288;
        std::fs::write(&path, vec![7u8; len]).unwrap();
        let f = std::fs::File::open(&path).unwrap();

        let mut buf = vec![0u8; 8192];
        let n = read_at_least_at(&f, &mut buf, 4096, 288).unwrap();
        assert_eq!(n, 288, "should read to EOF and no further");
        assert!(
            buf[..288].iter().all(|&b| b == 7),
            "the row's bytes must arrive"
        );

        let e = read_at_least_at(&f, &mut buf, 4096, 289).unwrap_err();
        assert_eq!(e.kind(), io::ErrorKind::UnexpectedEof);

        assert!(read_exact_at(&f, &mut buf[..8192], 4096).is_err());

        drop(f);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // 2026-09-25: Writes and reads at non-zero offsets, and checks after each
    // call that the file cursor is still where `seek` put it.
    #[test]
    fn round_trip_at_offset() {
        use std::io::{Seek, SeekFrom};

        let dir = std::env::temp_dir().join(format!("metrale_pio_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("pio.bin");
        let mut f = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)
            .unwrap();
        f.seek(SeekFrom::Start(137)).unwrap();

        write_all_at(&f, &[0u8; 4096], 0).unwrap();
        assert_eq!(f.stream_position().unwrap(), 137);
        let payload: Vec<u8> = (0..1024u32).map(|i| (i % 251) as u8).collect();
        write_all_at(&f, &payload, 2048).unwrap();
        assert_eq!(f.stream_position().unwrap(), 137);

        let mut out = vec![0u8; payload.len()];
        read_exact_at(&f, &mut out, 2048).unwrap();
        assert_eq!(out, payload);
        assert_eq!(f.stream_position().unwrap(), 137);

        let mut past = vec![0u8; 8192];
        assert!(read_exact_at(&f, &mut past, 4096).is_err());
        assert_eq!(f.stream_position().unwrap(), 137);

        drop(f);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
