// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Sizing of the 2-rank all-reduce receive buffer and the check
//! that a payload fits it. At `world_size == 2` the all-reduce receives the
//! partner's data into this buffer, so a payload larger than it would write
//! past the allocation.
//!
//! Owner: metrale-comm.
//! Invariants:
//! - The size functions never wrap: an overflow is an error.

use anyhow::{Context, Result};

/// 2026-09-26: Bytes per element of the all-reduce: BF16. `NcclBackend`
/// divides byte counts by it to get NCCL element counts, and
/// [`required_model_recv_bytes`] sizes the buffer with it.
pub const ALL_REDUCE_DTYPE_BYTES: usize = 2;

/// 2026-09-26: `max_batch_tokens × hidden_size × dtype_bytes`, in bytes.
///
/// # Errors
/// When the product overflows `usize`.
pub fn required_recv_bytes(
    max_batch_tokens: usize,
    hidden_size: usize,
    dtype_bytes: usize,
) -> Result<usize> {
    max_batch_tokens
        .checked_mul(hidden_size)
        .and_then(|elems| elems.checked_mul(dtype_bytes))
        .with_context(|| {
            format!(
                "receive-buffer size overflows usize: \
                 max_batch_tokens={max_batch_tokens} × hidden_size={hidden_size} \
                 × dtype_bytes={dtype_bytes}"
            )
        })
}

/// 2026-09-26: Receive-buffer bytes for `tokens` rows of BF16 activations or
/// logits: `tokens × max(hidden, vocab) × 2`. Serve passes `max_batch_tokens`,
/// `hidden_size` and `vocab_size`.
///
/// # Errors
/// When the product overflows `usize`.
pub fn required_model_recv_bytes(tokens: usize, hidden: usize, vocab: usize) -> Result<usize> {
    required_recv_bytes(tokens, hidden.max(vocab), ALL_REDUCE_DTYPE_BYTES)
}

/// 2026-09-26: Refuse a payload of more than `capacity` bytes. A free
/// function, so the tests below run it without a communicator or a GPU.
pub(crate) fn ensure_payload_fits(
    bytes: usize,
    capacity: usize,
    rank: usize,
    world_size: usize,
) -> Result<()> {
    if bytes > capacity {
        anyhow::bail!(
            "2-rank all-reduce payload exceeds receive-buffer capacity: \
             requested {bytes} bytes ({} elements × {ALL_REDUCE_DTYPE_BYTES} B/elem), \
             capacity {capacity} bytes, rank {rank}, world_size {world_size}. \
             The receive buffer is sized from the configured maximum transfer \
             (max_batch_tokens × hidden_size × {ALL_REDUCE_DTYPE_BYTES} B); a larger \
             payload would write past the allocation. Refusing to send.",
            bytes / ALL_REDUCE_DTYPE_BYTES,
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const BF16: usize = 2;
    const FP32: usize = 4;

    #[test]
    fn short_prefill_still_covers_vocab_parallel_logits() {
        let capacity = required_model_recv_bytes(33, 1024, 163840).unwrap();
        assert!(ensure_payload_fits(163584 * 2, capacity, 0, 2).is_ok());
        assert!(ensure_payload_fits(33 * 163840 * 2, capacity, 0, 2).is_ok());
        assert!(required_model_recv_bytes(usize::MAX, 1024, 163840).is_err());
        assert_eq!(required_model_recv_bytes(2, 4096, 1024).unwrap(), 16384);
    }

    /// 2026-09-26: 64 MiB, the boundary the tests below compare against.
    const OLD_FIXED_BUFFER: usize = 64 * 1024 * 1024;

    /// 2026-09-26: The all-reduce element is two bytes (BF16).
    #[test]
    fn all_reduce_dtype_width_is_bf16() {
        assert_eq!(ALL_REDUCE_DTYPE_BYTES, BF16);
    }

    /// 2026-09-26: 8192 × 4096 × 2 is exactly 64 MiB, and a payload equal to
    /// the capacity is accepted.
    #[test]
    fn exact_boundary_bf16() {
        let need = required_recv_bytes(8192, 4096, BF16).unwrap();
        assert_eq!(need, 67_108_864);
        assert_eq!(
            need, OLD_FIXED_BUFFER,
            "DS4F default sat exactly on the cap"
        );
        assert!(ensure_payload_fits(need, need, 0, 2).is_ok());
    }

    /// 2026-09-26: One token more than 64 MiB fits a capacity sized for it and
    /// is refused against 64 MiB.
    #[test]
    fn over_boundary_bf16() {
        let need = required_recv_bytes(8193, 4096, BF16).unwrap();
        assert_eq!(need, 67_117_056);
        assert!(need > OLD_FIXED_BUFFER);
        assert!(ensure_payload_fits(need, need, 0, 2).is_ok());
        assert!(ensure_payload_fits(need, OLD_FIXED_BUFFER, 0, 2).is_err());
    }

    /// 2026-09-26: `hidden_size = 6144` at 8192 tokens needs 32 MiB more than
    /// 64 MiB, and is refused against 64 MiB.
    #[test]
    fn wide_model_bf16() {
        let need = required_recv_bytes(8192, 6144, BF16).unwrap();
        assert_eq!(need, 100_663_296);
        assert_eq!(
            need - OLD_FIXED_BUFFER,
            33_554_432,
            "the overrun this fixes"
        );
        assert!(ensure_payload_fits(need, need, 0, 2).is_ok());
        assert!(ensure_payload_fits(need, OLD_FIXED_BUFFER, 0, 2).is_err());
    }

    /// 2026-09-26: A 4-byte element doubles the size.
    #[test]
    fn fp32_is_representable() {
        let need = required_recv_bytes(8192, 4096, FP32).unwrap();
        assert_eq!(need, 134_217_728);
        assert_eq!(need, 2 * required_recv_bytes(8192, 4096, BF16).unwrap());
        assert_eq!(
            need,
            2 * OLD_FIXED_BUFFER,
            "FP32 would have been a 2× overrun"
        );
        assert!(ensure_payload_fits(need, need, 0, 2).is_ok());
    }

    /// 2026-09-26: One row (`hidden × 2`) fits a capacity sized for 8192 rows.
    #[test]
    fn decode_b1_bf16() {
        let cap = required_recv_bytes(8192, 4096, BF16).unwrap();
        let decode = 4096 * BF16;
        assert_eq!(decode, 8_192);
        assert!(ensure_payload_fits(decode, cap, 0, 2).is_ok());
    }

    /// 2026-09-26: Zero tokens size to zero, and a zero payload fits any
    /// capacity, including zero.
    #[test]
    fn zero_elements() {
        let cap = required_recv_bytes(8192, 4096, BF16).unwrap();
        assert_eq!(required_recv_bytes(0, 4096, BF16).unwrap(), 0);
        assert!(ensure_payload_fits(0, cap, 0, 2).is_ok());
        assert!(ensure_payload_fits(0, 0, 0, 2).is_ok());
    }

    /// 2026-09-26: An overflowing product is an error, not a wrapped value.
    #[test]
    fn overflow_is_rejected() {
        assert!(required_recv_bytes(usize::MAX, 4096, BF16).is_err());
        assert!(required_recv_bytes(usize::MAX, 1, 2).is_err());
        assert!(required_recv_bytes(usize::MAX / 2 + 1, 2, 1).is_err());
        // 2026-09-26: A wrapping multiply would give 0 here.
        assert!(required_recv_bytes(1 << 62, 4, 1).is_err());
    }

    /// 2026-09-26: One byte short is refused, and the message names the
    /// payload, the capacity and the world size.
    #[test]
    fn one_byte_short_is_rejected() {
        let need = required_recv_bytes(8192, 4096, BF16).unwrap();
        let err = ensure_payload_fits(need, need - 1, 1, 2)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains(&need.to_string()),
            "must report requested bytes: {err}"
        );
        assert!(
            err.contains(&(need - 1).to_string()),
            "must report capacity: {err}"
        );
        assert!(err.contains("world_size 2"), "must report the path: {err}");
    }
}
