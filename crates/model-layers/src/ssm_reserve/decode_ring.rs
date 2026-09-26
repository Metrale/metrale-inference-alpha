// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Depth of the per-sequence SSM decode-rollback ring: the depth
//! decision, the published depth, and the fit to free memory.
//!
//! Owner: model-layers (SSM reserve).
//! Invariants:
//! - The published depth is set at most once per process; later writes leave
//!   the first value in force.
//! - `parse_decode_ring_slots` accepts only `auto` and
//!   `0..=DECODE_ROLLBACK_RING_SLOTS`.
//!
//! The server's preflight reserve (`preflight/decode_ring.rs`, unless
//! `METRALE_SSM_RESERVE_RING_FULL` is present) and `TransformerModel::new`
//! (`impl_a1.rs`) both take the depth from [`decode_rollback_ring_slots`].
//! When preflight fits a smaller depth to free memory, it publishes that depth
//! through [`set_decode_ring_slots`] so the allocation follows it. The ring is
//! written by the scheduler's `rollback::snapshot_boundary_if_ssm` and read by
//! `rollback::rollback_to_boundary`; on a model with SSM layers and a depth of
//! 0, a rollback declines with `RollbackFallback::NoSsmSnapshot`.
//!
//! Environment, read in [`decode_rollback_ring_slots`]:
//! - `METRALE_SSM_DECODE_RING`: `1` forces the full depth and `0` forces 0,
//!   when no depth is published.
//! - `METRALE_DISABLE_WATCHDOGS`: `1` or `true` skips the ring when no depth
//!   is published and `METRALE_SSM_DECODE_RING` is not `1`.

/// 2026-09-25: Outcome of the ring-depth decision. `skip_reason` is `Some` only for
/// the implicit skips (speculative decode, watchdogs disabled), so the
/// allocating call site can log the saving.
pub struct DecodeRingDecision {
    pub slots: usize,
    pub skip_reason: Option<&'static str>,
}

/// 2026-09-25: Depths the preflight fit may choose, largest first. The first equals
/// [`metrale_kernels::DECODE_ROLLBACK_RING_SLOTS`] (8); each next depth halves
/// the ring's share of the reserve.
pub const DECODE_RING_FIT_LADDER: [usize; 5] = [8, 4, 2, 1, 0];

/// 2026-09-25: The published depth: an explicit `--ssm-decode-ring-slots N`
/// (`serve_flags.rs`, before preflight) or the depth preflight fitted. `auto`
/// publishes nothing, so `METRALE_SSM_DECODE_RING` applies unless preflight
/// publishes a fitted depth.
static DECODE_RING_SLOTS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();

/// 2026-09-25: Publishes the ring depth and returns the depth in force; the first
/// write wins. Preflight skips its fit when a depth is already published
/// ([`published_decode_ring_slots`]), so an explicit depth is not shrunk.
pub fn set_decode_ring_slots(slots: usize) -> usize {
    let _ = DECODE_RING_SLOTS.set(slots);
    *DECODE_RING_SLOTS.get().expect("just set")
}

/// 2026-09-25: The published depth, or `None`. Reads the cell without setting it.
pub fn published_decode_ring_slots() -> Option<usize> {
    DECODE_RING_SLOTS.get().copied()
}

/// 2026-09-25: Parses `--ssm-decode-ring-slots`: `auto` gives `None` (fit at
/// preflight), a number up to `DECODE_ROLLBACK_RING_SLOTS` gives `Some`, and
/// anything else an error. `validate_serve_args` and `publish_kernel_flags`
/// both parse through this.
pub fn parse_decode_ring_slots(s: &str) -> Result<Option<usize>, String> {
    if s == "auto" {
        return Ok(None);
    }
    let n: usize = s
        .parse()
        .map_err(|_| format!("unknown ssm-decode-ring-slots '{s}' (valid: auto, 0..=8)"))?;
    if n > metrale_kernels::DECODE_ROLLBACK_RING_SLOTS {
        return Err(format!(
            "ssm-decode-ring-slots {n} exceeds the {} the ring is sized for",
            metrale_kernels::DECODE_ROLLBACK_RING_SLOTS
        ));
    }
    Ok(Some(n))
}

/// 2026-09-25: The per-sequence ring depth, from the published depth and the
/// environment ([`decode_rollback_ring_slots_with`]). Preflight passes
/// `args.speculative || args.dflash` as `use_speculative`; the allocation must
/// receive the same flag, or the reserve and the ring differ.
pub fn decode_rollback_ring_slots(
    num_ssm_layers: usize,
    use_speculative: bool,
) -> DecodeRingDecision {
    let watchdogs_value = std::env::var("METRALE_DISABLE_WATCHDOGS").ok();
    let watchdogs_disabled = watchdogs_disabled_from_value(watchdogs_value.as_deref());
    let ring_override = std::env::var("METRALE_SSM_DECODE_RING").ok();
    decode_rollback_ring_slots_with(
        num_ssm_layers,
        use_speculative,
        published_decode_ring_slots(),
        ring_override.as_deref(),
        watchdogs_disabled,
    )
}

/// 2026-09-25: `METRALE_DISABLE_WATCHDOGS` truthiness: `1` or `true`, trimmed and
/// case-insensitive, as the server's `parse_disable_watchdogs` reads it.
pub fn watchdogs_disabled_from_value(value: Option<&str>) -> bool {
    value
        .map(|v| {
            let v = v.trim().to_ascii_lowercase();
            v == "1" || v == "true"
        })
        .unwrap_or(false)
}

/// 2026-09-25: Env-free core of [`decode_rollback_ring_slots`]. Precedence, highest
/// first:
///
/// 1. no SSM layers: 0;
/// 2. `published`: that depth, also under speculative decode or with watchdogs
///    disabled;
/// 3. `ring_override` `"1"` or `"0"`: `DECODE_ROLLBACK_RING_SLOTS` or 0;
/// 4. speculative decode or watchdogs disabled: 0, with a `skip_reason`;
/// 5. otherwise `DECODE_ROLLBACK_RING_SLOTS`.
pub fn decode_rollback_ring_slots_with(
    num_ssm_layers: usize,
    use_speculative: bool,
    published: Option<usize>,
    ring_override: Option<&str>,
    watchdogs_disabled: bool,
) -> DecodeRingDecision {
    if num_ssm_layers == 0 {
        return DecodeRingDecision {
            slots: 0,
            skip_reason: None,
        };
    }
    if let Some(slots) = published {
        return DecodeRingDecision {
            slots,
            skip_reason: None,
        };
    }
    match ring_override {
        Some("1") => DecodeRingDecision {
            slots: metrale_kernels::DECODE_ROLLBACK_RING_SLOTS,
            skip_reason: None,
        },
        Some("0") => DecodeRingDecision {
            slots: 0,
            skip_reason: None,
        },
        _ if use_speculative || watchdogs_disabled => DecodeRingDecision {
            slots: 0,
            skip_reason: Some(if use_speculative {
                "speculative decode active"
            } else {
                "watchdogs disabled"
            }),
        },
        _ => DecodeRingDecision {
            slots: metrale_kernels::DECODE_ROLLBACK_RING_SLOTS,
            skip_reason: None,
        },
    }
}

/// 2026-09-25: Largest [`DECODE_RING_FIT_LADDER`] depth, at most `start_slots`, with
/// `reserve_without_ring + depth * bytes_per_ring_slot <= free_mem` (saturating
/// arithmetic). `bytes_per_ring_slot` is one depth unit of the ring. Returns 0
/// also when no depth fits; the caller tells the two apart by comparing the
/// total against its limit (`preflight/decode_ring.rs`).
pub fn fit_decode_ring_slots(
    start_slots: usize,
    reserve_without_ring: usize,
    bytes_per_ring_slot: usize,
    free_mem: usize,
) -> usize {
    DECODE_RING_FIT_LADDER
        .iter()
        .copied()
        .filter(|&slots| slots <= start_slots)
        .find(|&slots| {
            reserve_without_ring.saturating_add(slots.saturating_mul(bytes_per_ring_slot))
                <= free_mem
        })
        .unwrap_or(0)
}

#[cfg(test)]
#[path = "decode_ring_tests.rs"]
mod decode_ring_tests;
