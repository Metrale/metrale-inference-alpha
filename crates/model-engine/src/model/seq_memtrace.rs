// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Per-sequence memory trace: one log line when a sequence is allocated and one
//! when it is freed, for measuring leaks per request.
//!
//! Each line carries readings taken together:
//!
//! * `live` — the count of live device allocations on the backend's own ledger, and
//!   `live_mb`, their bytes (`-1` when the backend keeps no byte ledger).
//! * `cumemgetinfo_mb` — `device_free_memory`, the driver's free memory
//!   (`cuMemGetInfo` on CUDA).
//! * `memavailable_mb` — `MemAvailable` from `/proc/meminfo`, the host side.
//!
//! Off unless `METRALE_SEQ_MEMTRACE` is set, to any value including `0`; when off, a
//! call costs one `OnceLock` read.
//!
//! Owner: model-engine.
//! Invariants: none beyond the types.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use metrale_gpu_runtime::gpu::GpuBackend;

static ENABLED: OnceLock<bool> = OnceLock::new();
static SEQ_NO: AtomicU64 = AtomicU64::new(0);

pub fn enabled() -> bool {
    *ENABLED.get_or_init(|| std::env::var("METRALE_SEQ_MEMTRACE").is_ok())
}

fn mem_available_bytes() -> Option<usize> {
    let contents = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in contents.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            let kb: usize = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kb * 1024);
        }
    }
    None
}

/// 2026-09-25: One trace line. `phase` is `alloc` (`alloc_sequence_dispatch`) or
/// `free` (end of `free_sequence_dispatch`). Logged at `info!`, so it shows
/// without raising the crate to debug.
pub fn trace(gpu: &dyn GpuBackend, phase: &str) {
    if !enabled() {
        return;
    }
    let n = match phase {
        // 2026-09-25: `alloc` takes the next index; any other phase reports the
        // latest index handed out, which is the freed sequence's own index only
        // when sequences do not overlap.
        "alloc" => SEQ_NO.fetch_add(1, Ordering::Relaxed),
        _ => SEQ_NO.load(Ordering::Relaxed).saturating_sub(1),
    };
    let mb = |b: usize| b as f64 / (1024.0 * 1024.0);
    let dev = gpu.device_free_memory().unwrap_or(0);
    let host = mem_available_bytes().unwrap_or(0);
    // 2026-09-25: Count and bytes: a count alone cannot see a leak that frees N
    // buffers and allocates N smaller ones. `live_bytes` is `None` on a backend
    // with no byte ledger, and `-1` then means "not reported", not zero.
    let live_mb = gpu
        .live_bytes()
        .map_or(-1.0, |b| b as f64 / (1024.0 * 1024.0));
    tracing::info!(
        "seqmem: seq={n} phase={phase} live={} live_mb={:.1} cumemgetinfo_mb={:.1} memavailable_mb={:.1}",
        gpu.live_alloc_count(),
        live_mb,
        mb(dev),
        mb(host),
    );
}
