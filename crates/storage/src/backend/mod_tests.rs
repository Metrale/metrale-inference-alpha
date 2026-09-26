// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: GPU-free tests of the block-to-group fan-out:
//! `expand_blocks_to_groups` and the default `read_blocks` and
//! `write_block_from_host`, on host arithmetic only (no CUDA call, no I/O).
//!
//! Owner: metrale-storage high-speed swap.
//! Invariants: none beyond the types.

use super::*;
use crate::group::{GroupLayout, KvKind};

fn spec() -> GroupLayout {
    // 2026-09-25: 8 kv_heads; 16 × 128 × 2 = 4096-byte groups.
    GroupLayout::new(80, 4096, 8, 16, 128, 2, 4096)
}

/// 2026-09-25: One block expands to K(kh), V(kh) for each head, landing at
/// `base + kh·gs` and `base + (nkv + kh)·gs`.
#[test]
fn expand_matches_the_per_head_loop() {
    let s = spec();
    let nkv = s.num_kv_heads;
    let gs = s.group_stride;
    let base = 0xDEAD_0000u64;
    let (layer, block) = (7u32, 12u32);
    let br = BlockReadRequest {
        base_key: GroupKey::new(layer, block, 0, KvKind::K),
        dst_dev_ptr: base,
    };
    let got = expand_blocks_to_groups(&s, &[br]);
    assert_eq!(got.len(), 2 * nkv as usize);
    for kh in 0..nkv {
        let k = got[(2 * kh) as usize];
        let v = got[(2 * kh + 1) as usize];
        assert_eq!(k.group, GroupKey::new(layer, block, kh, KvKind::K));
        assert_eq!(k.dst_dev_ptr, base + (kh as u64) * gs);
        assert_eq!(v.group, GroupKey::new(layer, block, kh, KvKind::V));
        assert_eq!(v.dst_dev_ptr, base + (nkv as u64 + kh as u64) * gs);
    }
}

/// 2026-09-25: Two blocks expand to two consecutive runs of `2·nkv` requests,
/// each from its own base.
#[test]
fn expand_handles_multiple_blocks() {
    let s = spec();
    let nkv = s.num_kv_heads as usize;
    let reqs = [
        BlockReadRequest {
            base_key: GroupKey::new(1, 2, 0, KvKind::K),
            dst_dev_ptr: 0x1000,
        },
        BlockReadRequest {
            base_key: GroupKey::new(1, 9, 0, KvKind::K),
            dst_dev_ptr: 0x9000,
        },
    ];
    let got = expand_blocks_to_groups(&s, &reqs);
    assert_eq!(got.len(), 2 * (2 * nkv));
    assert_eq!(got[0].group, GroupKey::new(1, 2, 0, KvKind::K));
    assert_eq!(got[0].dst_dev_ptr, 0x1000);
    assert_eq!(got[2 * nkv].group, GroupKey::new(1, 9, 0, KvKind::K));
    assert_eq!(got[2 * nkv].dst_dev_ptr, 0x9000);
}

/// 2026-09-25: A backend that records each read as (layer, file offset, bytes,
/// dst) and each write as (layer, file offset, len), so the tests below compare
/// the default block methods' op streams with the per-head ones.
struct Recorder {
    spec: GroupLayout,
    reads: Vec<(u32, u64, usize, u64)>,
    writes: Vec<(u32, u64, usize)>,
}
impl StorageBackend for Recorder {
    fn read(&mut self, requests: &[ReadRequest], _stream: u64) -> Result<()> {
        let gb = self.spec.group_bytes() as usize;
        for r in requests {
            self.reads.push((
                r.group.layer,
                self.spec.file_offset(r.group),
                gb,
                r.dst_dev_ptr,
            ));
        }
        Ok(())
    }
    fn write_from_host(&mut self, key: GroupKey, src: &[u8]) -> Result<()> {
        self.writes
            .push((key.layer, self.spec.file_offset(key), src.len()));
        Ok(())
    }
    fn group_layout(&self) -> GroupLayout {
        self.spec
    }
}

#[test]
fn default_read_blocks_op_equivalent_to_per_head() {
    let s = spec();
    let base = 0x4000u64;
    let br = BlockReadRequest {
        base_key: GroupKey::new(4, 6, 0, KvKind::K),
        dst_dev_ptr: base,
    };
    let mut oracle = Recorder {
        spec: s,
        reads: vec![],
        writes: vec![],
    };
    let groups = expand_blocks_to_groups(&s, &[br]);
    oracle.read(&groups, 0).unwrap();
    let mut subject = Recorder {
        spec: s,
        reads: vec![],
        writes: vec![],
    };
    subject.read_blocks(&[br], 0).unwrap();
    assert_eq!(subject.reads, oracle.reads);
}

#[test]
fn default_write_block_op_equivalent_to_per_head() {
    let s = spec();
    let nkv = s.num_kv_heads as usize;
    let gs = s.group_stride as usize;
    let mut rec = Recorder {
        spec: s,
        reads: vec![],
        writes: vec![],
    };
    let buf = vec![0u8; 2 * nkv * gs];
    rec.write_block_from_host(GroupKey::new(2, 5, 0, KvKind::K), &buf)
        .unwrap();
    assert_eq!(rec.writes.len(), 2 * nkv);
    let mut expected: Vec<(u32, u64, usize)> = Vec::new();
    for kh in 0..s.num_kv_heads {
        expected.push((2, s.file_offset(GroupKey::new(2, 5, kh, KvKind::K)), gs));
        expected.push((2, s.file_offset(GroupKey::new(2, 5, kh, KvKind::V)), gs));
    }
    assert_eq!(rec.writes, expected);
    let bad = vec![0u8; 2 * nkv * gs - 1];
    assert!(
        rec.write_block_from_host(GroupKey::new(2, 5, 0, KvKind::K), &bad)
            .is_err()
    );
}
