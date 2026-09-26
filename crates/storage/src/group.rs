// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Group addressing for the high-speed-swap layer. A group is the K
//! or V stripe of one (layer, block, kv_head): `block_size × head_dim ×
//! elem_bytes` bytes, padded to a multiple of `fs_block_size`.
//!
//! Owner: metrale-storage high-speed swap.
//! Invariants:
//! - Within a layer, block `b` occupies `block_bytes()` contiguous bytes at
//!   `b · block_bytes()`: the K stripes of every kv_head, then the V stripes.
//! - For keys inside the layout, `group_id` maps each (layer, block, kind,
//!   kv_head) to a distinct id in `0 .. num_layers · 2 · num_blocks ·
//!   num_kv_heads`, in the same order as `file_offset`.

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GroupId(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvKind {
    K = 0,
    V = 1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GroupKey {
    pub layer: u32,
    pub block: u32,
    pub kv_head: u16,
    // 2026-09-25: `KvKind as u8`; `kind()` reads 0 as K and any other value as V.
    pub kv_kind: u8,
}

impl GroupKey {
    pub fn new(layer: u32, block: u32, kv_head: u16, kv_kind: KvKind) -> Self {
        Self {
            layer,
            block,
            kv_head,
            kv_kind: kv_kind as u8,
        }
    }
    pub fn kind(self) -> KvKind {
        match self.kv_kind {
            0 => KvKind::K,
            _ => KvKind::V,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GroupLayout {
    pub num_layers: u32,
    pub num_blocks: u32,
    pub num_kv_heads: u16,
    /// 2026-09-25: `block_size × head_dim × elem_bytes`, rounded up to a
    /// multiple of `fs_block_size`.
    pub group_stride: u64,
    /// 2026-09-25: The alignment unit `group_stride` is padded to.
    pub fs_block_size: u64,
}

impl GroupLayout {
    pub fn new(
        num_layers: u32,
        num_blocks: u32,
        num_kv_heads: u16,
        block_size: u32,
        head_dim: u32,
        elem_bytes: u32,
        fs_block_size: u64,
    ) -> Self {
        let raw = (block_size as u64) * (head_dim as u64) * (elem_bytes as u64);
        let group_stride = raw.div_ceil(fs_block_size) * fs_block_size;
        Self {
            num_layers,
            num_blocks,
            num_kv_heads,
            group_stride,
            fs_block_size,
        }
    }

    /// 2026-09-25: Bytes of one layer's file: K and V of every block and kv_head.
    pub fn bytes_per_layer(&self) -> u64 {
        2 * (self.num_blocks as u64) * (self.num_kv_heads as u64) * self.group_stride
    }

    /// 2026-09-25: Offset of `key` within its layer's file. Debug builds assert
    /// that `block` and `kv_head` are inside the layout.
    pub fn file_offset(&self, key: GroupKey) -> u64 {
        debug_assert!(key.block < self.num_blocks);
        debug_assert!(key.kv_head < self.num_kv_heads);
        let kv_stride = (self.num_kv_heads as u64) * self.group_stride;
        (key.block as u64) * (2 * kv_stride)
            + (key.kv_kind as u64) * kv_stride
            + (key.kv_head as u64) * self.group_stride
    }

    /// 2026-09-25: Dense `GroupId` for `key` across all layers.
    pub fn group_id(&self, key: GroupKey) -> GroupId {
        let per_layer = 2 * (self.num_blocks as u64) * (self.num_kv_heads as u64);
        let per_block = 2 * (self.num_kv_heads as u64);
        GroupId(
            (key.layer as u64) * per_layer
                + (key.block as u64) * per_block
                + (key.kv_kind as u64) * (self.num_kv_heads as u64)
                + (key.kv_head as u64),
        )
    }

    /// 2026-09-25: Bytes one group occupies on disk: `group_stride`.
    pub fn group_bytes(&self) -> u64 {
        self.group_stride
    }

    /// 2026-09-25: Bytes in one block: K and V of every kv_head, each at
    /// `group_stride` pitch. The unit of the `StorageBackend` block methods.
    pub fn block_bytes(&self) -> u64 {
        2 * (self.num_kv_heads as u64) * self.group_stride
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offset_and_id_are_consistent() {
        let l = GroupLayout::new(80, 4096, 8, 16, 128, 2, 4096);
        // 2026-09-25: 16 × 128 × 2 = 4096 bytes needs no padding.
        assert_eq!(l.group_stride, 4096);
        let k0 = GroupKey::new(0, 0, 0, KvKind::K);
        assert_eq!(l.file_offset(k0), 0);
        let k1 = GroupKey::new(0, 0, 0, KvKind::V);
        assert_eq!(l.file_offset(k1), (8u64) * 4096);
        let k2 = GroupKey::new(0, 1, 0, KvKind::K);
        assert_eq!(l.file_offset(k2), 2 * 8 * 4096);
        let k3 = GroupKey::new(0, 0, 7, KvKind::V);
        assert_eq!(l.file_offset(k3), 8 * 4096 + 7 * 4096);
    }

    #[test]
    fn rounds_up_to_fs_block() {
        // 2026-09-25: 16 × 96 × 2 = 3072 bytes pads to 4096.
        let l = GroupLayout::new(1, 1, 1, 16, 96, 2, 4096);
        assert_eq!(l.group_stride, 4096);
    }

    #[test]
    fn bytes_per_layer_correct() {
        let l = GroupLayout::new(1, 4, 2, 16, 128, 2, 4096);
        // 2026-09-25: 4 blocks × 2 kv_heads × 2 (K, V) × 4096 bytes.
        assert_eq!(l.bytes_per_layer(), 65536);
    }
}
