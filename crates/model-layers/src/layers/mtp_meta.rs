// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The one definition of an MTP proposer's attention-metadata
//! slab, packed by `MtpHead::forward_one`, `forward_batch_position` and
//! `DeepseekV4MtpHead::propose`.
//!
//! Owner: model-layers (MTP head).
//! Invariants:
//! - `pack_mtp_attn_meta` returns a slab only when it fits the caller's
//!   `region_bytes`; otherwise it returns an error and builds nothing.

use anyhow::{Result, ensure};

/// 2026-09-25: Bytes of header before the block table. The three scalars sit
/// at offsets 0, 8 and 16 and the rest is zero.
pub(crate) const MTP_META_HEADER_BYTES: usize = 256;

/// 2026-09-25: `scratch` byte offset of the single-sequence MTP metadata
/// slab, used by `MtpHead::forward_one` and `DeepseekV4MtpHead::propose`.
/// The target's batched decode metadata starts at `scratch + 32768`. The
/// batched propose writes its own `propose_meta` allocation instead.
pub const MTP_META_OFFSET: usize = 49152;

fn mtp_meta_len(block_entries: usize) -> Result<usize> {
    block_entries
        .checked_mul(4)
        .and_then(|bt_bytes| MTP_META_HEADER_BYTES.checked_add(bt_bytes))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "MTP attention metadata exceeds meta stride: byte size overflows for a {block_entries}-entry block table"
            )
        })
}

/// 2026-09-25: Pack one MTP attention-metadata slab, or return an error when
/// it would not fit `region_bytes`.
///
/// Layout: `position` u32 at 0, `global_slot` i64 at 8, `seq_len` i32 at 16,
/// then the block table from `MTP_META_HEADER_BYTES`, four little-endian
/// bytes per entry. `region_bytes` is what the caller owns at the
/// destination: the per-sequence stride for the batched propose, or
/// `scratch_bytes - MTP_META_OFFSET` for the single-sequence callers.
pub fn pack_mtp_attn_meta(
    position: u32,
    global_slot: i64,
    seq_len: i32,
    block_table: &[u32],
    region_bytes: usize,
) -> Result<Vec<u8>> {
    let need = mtp_meta_len(block_table.len())?;
    // 2026-09-25: `scheduler/mtp_bootstrap_step.rs` matches the phrase
    // "exceeds meta stride" to log this error at debug instead of error.
    // Rewording it breaks that match; the test below pins the phrase.
    ensure!(
        need <= region_bytes,
        "MTP attention metadata exceeds meta stride: needs {need} B for a {}-entry \
         block table, have {region_bytes} B",
        block_table.len()
    );

    let mut buf = vec![0u8; need];
    buf[0..4].copy_from_slice(&position.to_le_bytes());
    buf[8..16].copy_from_slice(&global_slot.to_le_bytes());
    buf[16..20].copy_from_slice(&seq_len.to_le_bytes());
    for (i, &block) in block_table.iter().enumerate() {
        let at = MTP_META_HEADER_BYTES + i * 4;
        buf[at..at + 4].copy_from_slice(&block.to_le_bytes());
    }
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packs_the_documented_layout() {
        let buf = pack_mtp_attn_meta(7, 0x1234_5678_9abc, 42, &[3, 9], 4096).unwrap();
        assert_eq!(buf.len(), MTP_META_HEADER_BYTES + 8);
        assert_eq!(u32::from_le_bytes(buf[0..4].try_into().unwrap()), 7);
        assert_eq!(
            i64::from_le_bytes(buf[8..16].try_into().unwrap()),
            0x1234_5678_9abc
        );
        assert_eq!(i32::from_le_bytes(buf[16..20].try_into().unwrap()), 42);
        assert_eq!(u32::from_le_bytes(buf[256..260].try_into().unwrap()), 3);
        assert_eq!(u32::from_le_bytes(buf[260..264].try_into().unwrap()), 9);
        // 2026-09-25: Every byte the layout does not define is zero.
        assert!(buf[20..256].iter().all(|&b| b == 0));
    }

    /// 2026-09-25: A block table that would overrun the region is refused.
    #[test]
    fn refuses_a_block_table_that_would_overrun_the_region() {
        // 2026-09-25: A 2048-byte region with the 256-byte header holds 448
        // entries.
        let fits = vec![0u32; 448];
        assert_eq!(
            pack_mtp_attn_meta(0, 0, 1, &fits, 2048).unwrap().len(),
            2048
        );
        let over = vec![0u32; 449];
        let e = pack_mtp_attn_meta(0, 0, 1, &over, 2048).unwrap_err();
        // 2026-09-25: Pins the phrase `scheduler/mtp_bootstrap_step.rs`
        // matches.
        assert!(
            e.to_string().contains("exceeds meta stride"),
            "the demotion phrase mtp_bootstrap_step matches must survive: {e}"
        );
    }

    /// 2026-09-25: A region smaller than the header is refused.
    #[test]
    fn refuses_a_region_too_small_for_the_header() {
        assert!(pack_mtp_attn_meta(0, 0, 1, &[], MTP_META_HEADER_BYTES - 1).is_err());
        assert!(pack_mtp_attn_meta(0, 0, 1, &[], MTP_META_HEADER_BYTES).is_ok());
    }

    #[test]
    fn refuses_an_unrepresentable_block_table_length() {
        let err = mtp_meta_len(usize::MAX).unwrap_err();
        assert!(err.to_string().contains("exceeds meta stride"), "{err}");
    }
}
