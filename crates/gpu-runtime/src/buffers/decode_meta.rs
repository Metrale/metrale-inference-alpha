// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Fixed-stride batched-decode metadata layout, derived from the serve `max_batch_size`.
//!
//! `sizes.rs` sizes `scratch` from it, and metrale-model-engine's
//! `upload_batch_metadata_fixed` / `_at` take their offsets from it. Byte
//! offsets, with `R = rows`:
//!   positions  u32  [0,       4R)
//!   seq_slot   i32  [4R,      8R)   (per-request LoRA slot)
//!   slots      i64  [8R,     16R)
//!   seq_lens   i32  [16R,    20R)
//!   (unused         [20R,    24R))
//!   block_tbl  i32  [24R,    24R + R·max_blocks·4)
//!
//! Owner: gpu-runtime.
//! Invariants:
//! - `rows() >= DECODE_META_MIN_ROWS`.
//! - The regions do not overlap, and `slots_off()` is a multiple of 8.

/// 2026-09-25: The row floor: `rows = max(DECODE_META_MIN_ROWS, max_batch_size)`.
pub const DECODE_META_MIN_ROWS: usize = 32;

/// 2026-09-25: The largest serve `max_batch_size` accepted; `serve_load.rs`
/// refuses a larger one at startup. The layout itself has no upper bound.
pub const DECODE_META_MAX_ROWS: usize = 128;

/// 2026-09-25: The decode-metadata layout for `rows` rows (see the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodeMetaLayout {
    rows: usize,
}

impl DecodeMetaLayout {
    /// 2026-09-25: The layout for the serve `max_batch_size`. It applies only the
    /// floor; the ceiling `DECODE_META_MAX_ROWS` is checked by the serve path.
    pub fn for_max_batch_size(max_batch_size: usize) -> Self {
        Self {
            rows: max_batch_size.max(DECODE_META_MIN_ROWS),
        }
    }

    /// 2026-09-25: Row capacity, the widest `padded_n` the upload accepts.
    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn positions_off(&self) -> usize {
        0
    }

    pub fn seq_slot_off(&self) -> usize {
        4 * self.rows
    }

    pub fn slots_off(&self) -> usize {
        8 * self.rows
    }

    pub fn seq_lens_off(&self) -> usize {
        16 * self.rows
    }

    /// 2026-09-25: Offset of the flattened block table, row stride `max_blocks · 4` bytes.
    pub fn block_table_off(&self) -> usize {
        24 * self.rows
    }

    /// 2026-09-25: Bytes of the whole block for `max_blocks` blocks per row.
    pub fn meta_bytes(&self, max_blocks: usize) -> usize {
        self.block_table_off() + self.rows * max_blocks * 4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-09-25: At or below the floor the offsets are 0/128/256/512/768.
    #[test]
    fn legacy_layout_at_or_below_32() {
        for bs in [1usize, 31, 32] {
            let l = DecodeMetaLayout::for_max_batch_size(bs);
            assert_eq!(l.rows(), 32, "bs={bs}");
            assert_eq!(l.positions_off(), 0);
            assert_eq!(l.seq_slot_off(), 128);
            assert_eq!(l.slots_off(), 256);
            assert_eq!(l.seq_lens_off(), 512);
            assert_eq!(l.block_table_off(), 768);
            assert_eq!(l.meta_bytes(257), 768 + 32 * 257 * 4);
        }
    }

    /// 2026-09-25: Above the floor, rows follow `max_batch_size`, the regions
    /// are back to back except the 4R gap before the block table, and the i64
    /// slots are 8-byte aligned.
    #[test]
    fn widened_layout_arithmetic() {
        for bs in [33usize, 64, 128] {
            let l = DecodeMetaLayout::for_max_batch_size(bs);
            let r = l.rows();
            assert_eq!(r, bs, "bs={bs}: rows derive from bs above the floor");
            assert_eq!(l.seq_slot_off(), l.positions_off() + 4 * r);
            assert_eq!(l.slots_off(), l.seq_slot_off() + 4 * r);
            assert_eq!(l.slots_off() % 8, 0);
            assert_eq!(l.seq_lens_off(), l.slots_off() + 8 * r);
            assert_eq!(l.block_table_off(), l.seq_lens_off() + 8 * r);
            assert_eq!(l.meta_bytes(257), 24 * r + r * 257 * 4);
        }
    }

    #[test]
    fn policy_ceiling_and_floor_are_exact() {
        assert_eq!(DECODE_META_MIN_ROWS, 32);
        assert_eq!(DECODE_META_MAX_ROWS, 128);
    }
}
