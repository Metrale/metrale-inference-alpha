// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Small value types shared by the routers.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use std::time::Duration;

/// 2026-09-25: A launched step, to be settled with `DeviceIo::await_result`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ticket(pub u32);

/// 2026-09-25: How long a request router's `recv` may wait for something to arrive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaitPolicy {
    /// 2026-09-25: Return whatever is queued, at once.
    NoWait,
    /// 2026-09-25: Wait at most this long for a request, a rotation or the close.
    Bounded(Duration),
    /// 2026-09-25: Wait until a request or a rotation is queued, or the inbox closes.
    Block,
}

/// 2026-09-25: The KV pool and SSM snapshot counters a router reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceFacts {
    pub free_blocks: usize,
    pub total_blocks: usize,
    pub ssm_occupancy: Option<(u32, u32)>,
}

/// 2026-09-25: A device failure, split by when it happened: the forward itself (the
/// batch may be preemptible), the readback after it, or an effect.
#[derive(Debug)]
pub enum DeviceError {
    Forward(anyhow::Error),
    Readback(anyhow::Error),
    Effect(anyhow::Error),
}

impl DeviceError {
    pub fn into_inner(self) -> anyhow::Error {
        match self {
            Self::Forward(e) | Self::Readback(e) | Self::Effect(e) => e,
        }
    }

    /// 2026-09-25: A forward error whose message reports KV cache exhaustion (the pool
    /// ran dry mid-batch).
    pub fn is_kv_exhausted(&self) -> bool {
        matches!(self, Self::Forward(e) if format!("{e:#}").contains("KV cache exhausted"))
    }
}

impl std::fmt::Display for DeviceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Forward(e) | Self::Readback(e) | Self::Effect(e) => write!(f, "{e:#}"),
        }
    }
}
