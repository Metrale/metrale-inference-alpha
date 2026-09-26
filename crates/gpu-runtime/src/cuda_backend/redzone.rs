// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Red-zone diagnostic: an opt-in guard band after each `alloc`, the
//! `METRALE_REDZONE*` variables that configure it, and the passes that
//! re-fill and check the bands.
//!
//! Owner: gpu-runtime (CUDA backend).
//! Invariants:
//! - Each `METRALE_REDZONE*` variable this module reads is read once per
//!   process (one `OnceLock` each).

use super::*;

/// 2026-09-25: One allocation's trailing guard band: `pad_bytes` after
/// `user_ptr + user_bytes`, for the allocation with creation index `idx`.
#[derive(Clone, Copy)]
pub(crate) struct RedZone {
    user_ptr: u64,
    user_bytes: usize,
    pad_bytes: usize,
    idx: usize,
}

/// 2026-09-25: `METRALE_REDZONE=<bytes>`: guard-band size, rounded up to a
/// multiple of 16. Unset, unparsable or 0 disables the red zones.
pub(crate) fn redzone_bytes() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("METRALE_REDZONE")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .map(|n| if n == 0 { 0 } else { n.next_multiple_of(16) })
            .unwrap_or(0)
    })
}

/// 2026-09-25: `METRALE_REDZONE_MIN_IDX=<n>`: pad only allocations whose
/// creation index ([`ALLOC_SEQ`]) is `>= n`. Unset or unparsable is 0: every
/// allocation.
pub(crate) fn redzone_min_idx() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("METRALE_REDZONE_MIN_IDX")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0)
    })
}

/// 2026-09-25: `METRALE_REDZONE_TRACE_IDX=<n>`: log a backtrace when the
/// padded allocation with creation index `n` is made.
pub(crate) fn redzone_trace_idx() -> Option<usize> {
    static N: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("METRALE_REDZONE_TRACE_IDX")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
    })
}

/// 2026-09-25: Creation index source: incremented once per
/// `MetraleCudaBackend::alloc` call, failed calls included, across every
/// backend in the process.
pub(crate) static ALLOC_SEQ: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// 2026-09-25: `METRALE_REDZONE_FILL=<decimal byte>`: the byte written into
/// new guard bands and expected by `scan_redzones`. Unset or unparsable is
/// `0xEE`.
///
/// A configurable fill separates the two kinds of overrun: a write past the
/// end changes the band and `scan_redzones` reports it, while a read past the
/// end leaves the band intact but lets the fill reach the computation, so two
/// runs with different fills can produce different tokens.
pub(crate) fn redzone_fill() -> u8 {
    static F: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *F.get_or_init(|| {
        std::env::var("METRALE_REDZONE_FILL")
            .ok()
            .and_then(|v| v.parse::<u8>().ok())
            .unwrap_or(0xEE)
    })
}

impl MetraleCudaBackend {
    /// 2026-09-25: Register one allocation's guard band.
    pub(crate) fn record_redzone(
        &self,
        user_ptr: u64,
        user_bytes: usize,
        pad_bytes: usize,
        idx: usize,
    ) {
        self.redzones.lock().push(RedZone {
            user_ptr,
            user_bytes,
            pad_bytes,
            idx,
        });
    }

    pub(crate) fn forget_redzone(&self, user_ptr: u64) {
        self.redzones.lock().retain(|z| z.user_ptr != user_ptr);
    }

    /// 2026-09-25: Re-fill every guard band: `0xEE` for zones whose creation
    /// index is in `[lo, hi)`, `0x00` for all the others. Only guard-band
    /// bytes are written.
    ///
    /// For bisecting a read past the end: narrowing `[lo, hi)` to the range
    /// whose fill changes the output finds the allocation being read past.
    pub fn poison_redzones(&self, lo: usize, hi: usize) -> anyhow::Result<()> {
        let zones: Vec<RedZone> = self.redzones.lock().clone();
        for z in &zones {
            let v = if z.idx >= lo && z.idx < hi {
                0xEEu8
            } else {
                0x00u8
            };
            let st = unsafe { cuMemsetD8_v2(z.user_ptr + z.user_bytes as u64, v, z.pad_bytes) };
            if st != 0 {
                anyhow::bail!("poison_redzones: cuMemsetD8_v2 failed: status {st}");
            }
        }
        Ok(())
    }

    /// 2026-09-25: Read every guard band back and count the ones holding a
    /// byte other than `redzone_fill()`. Each is logged at ERROR with its
    /// creation index, size, the first and last changed pad offsets, how many
    /// pad bytes changed, and the first changed value, and is then re-filled,
    /// so a later scan reports only new writes.
    pub fn scan_redzones(&self) -> anyhow::Result<usize> {
        let fill = redzone_fill();
        let zones: Vec<RedZone> = self.redzones.lock().clone();
        let mut bad = 0usize;
        let mut host = Vec::new();
        for z in &zones {
            host.clear();
            host.resize(z.pad_bytes, 0u8);
            let st = unsafe {
                cuMemcpyDtoH_v2(
                    host.as_mut_ptr() as *mut std::ffi::c_void,
                    z.user_ptr + z.user_bytes as u64,
                    z.pad_bytes,
                )
            };
            if st != 0 {
                anyhow::bail!("redzone scan: cuMemcpyDtoH_v2 failed: status {st}");
            }
            let Some(first) = host.iter().position(|b| *b != fill) else {
                continue;
            };
            let changed = host.iter().filter(|b| **b != fill).count();
            let last = host.iter().rposition(|b| *b != fill).unwrap_or(first);
            tracing::error!(
                "🔴 REDZONE VIOLATION alloc#{} user_bytes={} pad={} :                  bytes [{}..={}] past the end were written ({} of {} pad bytes changed),                  first bad value {:#04x}",
                z.idx,
                z.user_bytes,
                z.pad_bytes,
                first,
                last,
                changed,
                z.pad_bytes,
                host[first],
            );
            bad += 1;
            let st = unsafe { cuMemsetD8_v2(z.user_ptr + z.user_bytes as u64, fill, z.pad_bytes) };
            if st != 0 {
                anyhow::bail!("redzone scan: refill cuMemsetD8_v2 failed: status {st}");
            }
        }
        tracing::info!(
            "redzone scan: {} zones checked, {} violated",
            zones.len(),
            bad
        );
        Ok(bad)
    }
}
