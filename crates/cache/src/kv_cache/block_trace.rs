// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Per-block refcount event history, for finding the call site
//! behind a KV refcount bug.
//!
//! An unmatched decrement surfaces only later, when the rightful owner's
//! decrement finds 0 refs. With `METRALE_KV_TRACE=1`, every `alloc`,
//! `try_alloc`, `inc`, `dec` and `evict_return` on a block is recorded in a
//! ring for that block, with the caller's file and line (the `PagedKvCache`
//! methods are `#[track_caller]`). A decrement at 0 refs logs the ring.
//! Tracing costs one `Vec` per block plus a push per refcount operation, so it
//! is off unless the variable is `1`.
//!
//! Owner: cache.
//! Invariants: a ring holds at most `RING_LEN` events.

use std::panic::Location;
use std::sync::OnceLock;

/// 2026-09-25: Events kept per block; the oldest is dropped first.
const RING_LEN: usize = 24;

#[derive(Clone, Copy)]
pub(super) struct BlockEvent {
    op: &'static str,
    count_after: u32,
    file: &'static str,
    line: u32,
}

/// 2026-09-25: Per-block rings of refcount events; empty when tracing is off.
#[derive(Default)]
pub(super) struct BlockTrace {
    rings: Vec<Vec<BlockEvent>>,
}

pub(super) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("METRALE_KV_TRACE").as_deref() == Ok("1"))
}

impl BlockTrace {
    pub(super) fn new(num_blocks: usize) -> Self {
        if enabled() {
            tracing::info!(
                "METRALE_KV_TRACE=1: recording per-block refcount history \
                 ({RING_LEN} events x {num_blocks} blocks)"
            );
            Self {
                rings: vec![Vec::with_capacity(RING_LEN); num_blocks],
            }
        } else {
            Self::default()
        }
    }

    pub(super) fn is_on(&self) -> bool {
        !self.rings.is_empty()
    }

    pub(super) fn record(
        &mut self,
        idx: usize,
        op: &'static str,
        count_after: u32,
        loc: &'static Location<'static>,
    ) {
        let Some(ring) = self.rings.get_mut(idx) else {
            return;
        };
        if ring.len() == RING_LEN {
            ring.remove(0);
        }
        ring.push(BlockEvent {
            op,
            count_after,
            file: loc.file(),
            line: loc.line(),
        });
    }

    /// 2026-09-25: A block's events, oldest first, joined by ` | `.
    pub(super) fn dump(&self, idx: usize) -> String {
        let Some(ring) = self.rings.get(idx) else {
            return String::from("(no history)");
        };
        ring.iter()
            .map(|e| {
                let file = e.file.rsplit('/').next().unwrap_or(e.file);
                format!("{}->{} @{}:{}", e.op, e.count_after, file, e.line)
            })
            .collect::<Vec<_>>()
            .join(" | ")
    }
}
