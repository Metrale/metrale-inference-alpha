// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: OS entropy for per-process key salts; its caller is the SSM decode
//! tier's client salt (`decode_client_salt` in metrale-model-engine).
//!
//! Owner: storage, tiered-cache core.
//! Invariants:
//! - `random_u64` returns OS entropy or an error, never a fallback value.
//!
//! Kept apart from `tier::hash`, which must return the same value for the same input
//! forever; a salt must differ per process, so two clients of one model sharing a
//! paging peer derive different keys.

use anyhow::Result;

/// 2026-09-25: 8 bytes of OS entropy as a `u64`. Linux: `libc::getrandom` with no
/// flags, retried on `EINTR` and partial reads. Other unix: `/dev/urandom`. Windows:
/// the `getrandom` crate. A failure is an error.
pub fn random_u64() -> Result<u64> {
    let mut buf = [0u8; 8];
    fill(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

#[cfg(target_os = "linux")]
fn fill(buf: &mut [u8]) -> Result<()> {
    let mut got = 0usize;
    while got < buf.len() {
        let r = unsafe { libc::getrandom(buf[got..].as_mut_ptr().cast(), buf.len() - got, 0) };
        if r < 0 {
            let e = std::io::Error::last_os_error();
            if e.raw_os_error() == Some(libc::EINTR) {
                continue;
            }
            anyhow::bail!("getrandom failed (refusing a degraded zero-entropy salt): {e}");
        }
        got += r as usize;
    }
    Ok(())
}

#[cfg(all(unix, not(target_os = "linux")))]
fn fill(buf: &mut [u8]) -> Result<()> {
    use std::io::Read;
    let mut f = std::fs::File::open("/dev/urandom")
        .map_err(|e| anyhow::anyhow!("open /dev/urandom failed (refusing a degraded salt): {e}"))?;
    f.read_exact(buf)
        .map_err(|e| anyhow::anyhow!("read /dev/urandom failed (refusing a degraded salt): {e}"))
}

/// 2026-09-25: Windows: the `getrandom` crate, a direct dependency of this crate.
#[cfg(windows)]
fn fill(buf: &mut [u8]) -> Result<()> {
    getrandom::fill(buf)
        .map_err(|e| anyhow::anyhow!("BCryptGenRandom failed (refusing a degraded salt): {e}"))
}

#[cfg(test)]
#[path = "entropy_tests.rs"]
mod tests;
