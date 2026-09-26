// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Expert identity and on-disk record geometry for MoE expert
//! streaming. A record holds one (moe_layer, expert)'s gate, up and down NVFP4
//! projections behind a fixed header; each MoE layer's file holds
//! `num_experts` records at one stride, which the whole model shares.
//!
//! Owner: metrale-storage experts.
//! Invariants:
//! - Inside a record the header comes first, then gate, up and down (packed,
//!   then scale, for each), every sub-buffer at a multiple of `sub_align` and
//!   none overlapping.
//! - `record_stride` is `raw_bytes` rounded up to a multiple of `fs_block_size`.

/// 2026-09-25: Dense record id across the model:
/// `layer · num_experts + expert`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ExpertRecordId(pub u64);

/// 2026-09-25: One expert's record. `layer` is the dense MoE-layer index
/// (`0..num_moe_layers`), not the model layer;
/// `ExpertIndex::moe_layer_to_model_layer` maps one to the other.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ExpertKey {
    pub layer: u32,
    pub expert: u32,
}

impl ExpertKey {
    pub fn new(layer: u32, expert: u32) -> Self {
        Self { layer, expert }
    }
}

/// 2026-09-25: The three projections of an expert. Their order is the order
/// of the sub-buffers in a record and of the header's per-projection fields,
/// so changing it changes the on-disk format that
/// [`ExpertRecordHeader::VERSION`] identifies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Proj {
    Gate = 0,
    Up = 1,
    Down = 2,
}

impl Proj {
    pub const ALL: [Proj; 3] = [Proj::Gate, Proj::Up, Proj::Down];
}

/// 2026-09-25: Byte sizes of one NVFP4 projection's two buffers: `packed`,
/// two 4-bit values per byte (`n·k/2`), and `scale`, one byte per
/// `group_size` values (`n·k/group_size`). The per-projection scalars
/// `weight_scale_2` and `input_scale` are in the record header instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProjBytes {
    pub packed_bytes: u64,
    pub scale_bytes: u64,
}

impl ProjBytes {
    /// 2026-09-25: `n` output rows, `k` contraction dim, `group_size` values
    /// per scale.
    pub fn nvfp4(n: u64, k: u64, group_size: u64) -> Self {
        Self {
            packed_bytes: n * k / 2,
            scale_bytes: n * k / group_size,
        }
    }
}

/// 2026-09-25: Where each sub-buffer of an expert record sits, relative to the
/// record base; `pack_record` and `unpack_record` place and read bytes by it.
/// Every offset is a multiple of `sub_align`:
///
/// ```text
///   [ header (ExpertRecordHeader::BYTES) ]
///   [ gate.packed ][ gate.scale ]
///   [ up.packed   ][ up.scale   ]
///   [ down.packed ][ down.scale ]
///   [ pad to record_stride ]
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpertRecordSpec {
    pub inter: u64,
    pub hidden: u64,
    pub group_size: u64,
    pub sub_align: u64,
    /// 2026-09-25: `(packed_off, scale_off)` for gate, up, down.
    offsets: [(u64, u64); 3],
    bytes: [ProjBytes; 3],
    /// 2026-09-25: Header plus all sub-buffers with their padding, before
    /// `ExpertLayout` rounds it up to `record_stride`.
    raw_bytes: u64,
}

/// 2026-09-25: `off` rounded up to a multiple of `align`, a power of two.
#[inline]
fn align_up(off: u64, align: u64) -> u64 {
    (off + align - 1) & !(align - 1)
}

impl ExpertRecordSpec {
    /// 2026-09-25: The record layout for experts of intermediate size `inter`
    /// and hidden size `hidden`. Panics unless `sub_align` is a power of two.
    pub fn new(inter: u64, hidden: u64, group_size: u64, sub_align: u64) -> Self {
        assert!(
            sub_align.is_power_of_two(),
            "sub_align must be a power of two"
        );
        // 2026-09-25: gate and up are N=inter, K=hidden; down is N=hidden,
        // K=inter. Both byte counts are symmetric in (N, K), so the three sizes
        // are equal.
        let bytes = [
            ProjBytes::nvfp4(inter, hidden, group_size),
            ProjBytes::nvfp4(inter, hidden, group_size),
            ProjBytes::nvfp4(hidden, inter, group_size),
        ];
        let mut cursor = align_up(ExpertRecordHeader::BYTES, sub_align);
        let mut offsets = [(0u64, 0u64); 3];
        for i in 0..3 {
            let packed_off = cursor;
            cursor = align_up(packed_off + bytes[i].packed_bytes, sub_align);
            let scale_off = cursor;
            cursor = align_up(scale_off + bytes[i].scale_bytes, sub_align);
            offsets[i] = (packed_off, scale_off);
        }
        Self {
            inter,
            hidden,
            group_size,
            sub_align,
            offsets,
            bytes,
            raw_bytes: cursor,
        }
    }

    pub fn proj_bytes(&self, p: Proj) -> ProjBytes {
        self.bytes[p as usize]
    }

    /// 2026-09-25: Offset of `p`'s packed sub-buffer in the record.
    pub fn packed_off(&self, p: Proj) -> u64 {
        self.offsets[p as usize].0
    }

    /// 2026-09-25: Offset of `p`'s scale sub-buffer in the record.
    pub fn scale_off(&self, p: Proj) -> u64 {
        self.offsets[p as usize].1
    }

    /// 2026-09-25: See the `raw_bytes` field.
    pub fn raw_bytes(&self) -> u64 {
        self.raw_bytes
    }

    /// 2026-09-25: Sum of the six sub-buffers, without header and padding.
    pub fn payload_bytes(&self) -> u64 {
        self.bytes
            .iter()
            .map(|b| b.packed_bytes + b.scale_bytes)
            .sum()
    }
}

/// 2026-09-25: The fixed, versioned header at the front of every expert
/// record. It carries what the dims do not give, each projection's
/// `weight_scale_2` and `input_scale`, plus layer, expert and shape so a
/// mismatched file can be recognised.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExpertRecordHeader {
    pub layer: u32,
    pub expert: u32,
    pub inter: u32,
    pub hidden: u32,
    pub group_size: u32,
    /// 2026-09-25: `weight_scale_2` of gate, up, down.
    pub scale2: [f32; 3],
    /// 2026-09-25: `input_scale` of gate, up, down; `None` when the
    /// projection has none. Presence is a bit in the header's
    /// `input_scale_flags` word, so `PartialEq` needs no NaN sentinel.
    pub input_scale: [Option<f32>; 3],
}

impl ExpertRecordHeader {
    pub const MAGIC: u32 = 0x5850_5254;
    pub const VERSION: u32 = 1;
    /// 2026-09-25: Reserved header size; the fields use the first 56 bytes,
    /// so the header can grow without moving the sub-buffers.
    pub const BYTES: u64 = 256;

    /// 2026-09-25: Serialize into the fixed 256-byte header block,
    /// little-endian:
    ///   u32 magic, u32 version, u32 layer, u32 expert,
    ///   u32 inter, u32 hidden, u32 group_size, u32 input_scale_flags,
    ///   `f32 scale2[3]`, `f32 input_scale[3]` (0.0 where absent), zero pad to 256.
    /// `input_scale_flags` bit `i` set => projection `i` has an activation scale.
    pub fn to_bytes(&self) -> [u8; Self::BYTES as usize] {
        let mut out = [0u8; Self::BYTES as usize];
        let mut w = |off: usize, v: u32| out[off..off + 4].copy_from_slice(&v.to_le_bytes());
        w(0, Self::MAGIC);
        w(4, Self::VERSION);
        w(8, self.layer);
        w(12, self.expert);
        w(16, self.inter);
        w(20, self.hidden);
        w(24, self.group_size);
        let mut flags = 0u32;
        for (i, s) in self.input_scale.iter().enumerate() {
            if s.is_some() {
                flags |= 1 << i;
            }
        }
        w(28, flags);
        for (i, s) in self.scale2.iter().enumerate() {
            out[32 + i * 4..36 + i * 4].copy_from_slice(&s.to_le_bytes());
        }
        for (i, s) in self.input_scale.iter().enumerate() {
            let v = s.unwrap_or(0.0);
            out[44 + i * 4..48 + i * 4].copy_from_slice(&v.to_le_bytes());
        }
        out
    }

    /// 2026-09-25: Parse a header block. `None` when it is shorter than
    /// `BYTES` or the magic or version differs.
    pub fn from_bytes(buf: &[u8]) -> Option<Self> {
        if buf.len() < Self::BYTES as usize {
            return None;
        }
        let r = |off: usize| -> u32 {
            u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
        };
        let rf = |off: usize| -> f32 {
            f32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
        };
        if r(0) != Self::MAGIC || r(4) != Self::VERSION {
            return None;
        }
        let flags = r(28);
        let iscale = |i: usize, off: usize| -> Option<f32> {
            if flags & (1 << i) != 0 {
                Some(rf(off))
            } else {
                None
            }
        };
        Some(Self {
            layer: r(8),
            expert: r(12),
            inter: r(16),
            hidden: r(20),
            group_size: r(24),
            scale2: [rf(32), rf(36), rf(40)],
            input_scale: [iscale(0, 44), iscale(1, 48), iscale(2, 52)],
        })
    }
}

/// 2026-09-25: File geometry of an expert store: one file per MoE layer
/// (`ExpertIndex::file_name`), each holding `num_experts` records back to back
/// at `record_stride`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpertLayout {
    pub num_layers: u32,
    pub num_experts: u32,
    /// 2026-09-25: Record stride on disk, a multiple of `fs_block_size`.
    pub record_stride: u64,
    pub fs_block_size: u64,
}

impl ExpertLayout {
    /// 2026-09-25: `record_stride` is `spec.raw_bytes()` rounded up to a
    /// multiple of `fs_block_size`.
    pub fn from_spec(
        num_layers: u32,
        num_experts: u32,
        spec: &ExpertRecordSpec,
        fs_block_size: u64,
    ) -> Self {
        let record_stride = spec.raw_bytes().div_ceil(fs_block_size) * fs_block_size;
        Self {
            num_layers,
            num_experts,
            record_stride,
            fs_block_size,
        }
    }

    /// 2026-09-25: Bytes of one MoE layer's file.
    pub fn bytes_per_layer(&self) -> u64 {
        (self.num_experts as u64) * self.record_stride
    }

    /// 2026-09-25: Offset of `key`'s record in its layer file. Debug builds
    /// assert that `expert` is in range.
    pub fn file_offset(&self, key: ExpertKey) -> u64 {
        debug_assert!(key.expert < self.num_experts);
        (key.expert as u64) * self.record_stride
    }

    pub fn record_id(&self, key: ExpertKey) -> ExpertRecordId {
        ExpertRecordId((key.layer as u64) * (self.num_experts as u64) + (key.expert as u64))
    }

    /// 2026-09-25: Bytes of one record on disk: `record_stride`.
    pub fn record_bytes(&self) -> u64 {
        self.record_stride
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const A3B_INTER: u64 = 512;
    const A3B_HIDDEN: u64 = 2048;
    const GS: u64 = 16;

    #[test]
    fn a3b_per_expert_payload_matches_formula() {
        // 2026-09-25: Three projections at 1/2 + 1/16 bytes per weight.
        let spec = ExpertRecordSpec::new(A3B_INTER, A3B_HIDDEN, GS, 256);
        let expected = 3 * A3B_INTER * A3B_HIDDEN * 9 / 16;
        assert_eq!(spec.payload_bytes(), expected);
        assert_eq!(expected, 1_769_472);
    }

    #[test]
    fn projection_bytes_split_8_to_1() {
        // 2026-09-25: 1/2 byte per weight against 1/16: packed is 8× scale.
        let pb = ProjBytes::nvfp4(A3B_INTER, A3B_HIDDEN, GS);
        assert_eq!(pb.packed_bytes, A3B_INTER * A3B_HIDDEN / 2);
        assert_eq!(pb.scale_bytes, A3B_INTER * A3B_HIDDEN / 16);
        assert_eq!(pb.packed_bytes, 8 * pb.scale_bytes);
    }

    #[test]
    fn sub_buffers_are_aligned_and_non_overlapping() {
        let align = 256;
        let spec = ExpertRecordSpec::new(A3B_INTER, A3B_HIDDEN, GS, align);
        let mut prev_end = ExpertRecordHeader::BYTES;
        for p in Proj::ALL {
            let po = spec.packed_off(p);
            let so = spec.scale_off(p);
            let pb = spec.proj_bytes(p);
            assert_eq!(po % align, 0, "packed off aligned");
            assert_eq!(so % align, 0, "scale off aligned");
            assert!(po >= prev_end, "packed does not overlap previous");
            assert!(so >= po + pb.packed_bytes, "scale does not overlap packed");
            prev_end = so + pb.scale_bytes;
        }
        assert!(spec.raw_bytes() >= prev_end);
    }

    #[test]
    fn layout_offsets_are_record_strided() {
        let spec = ExpertRecordSpec::new(A3B_INTER, A3B_HIDDEN, GS, 256);
        let layout = ExpertLayout::from_spec(40, 256, &spec, 4096);
        assert_eq!(layout.record_stride % 4096, 0, "O_DIRECT alignment");
        assert!(layout.record_stride >= spec.raw_bytes());
        assert_eq!(layout.file_offset(ExpertKey::new(3, 0)), 0);
        assert_eq!(
            layout.file_offset(ExpertKey::new(3, 5)),
            5 * layout.record_stride
        );
        assert_eq!(layout.bytes_per_layer(), 256 * layout.record_stride);
    }

    #[test]
    fn record_id_is_dense_layer_major() {
        let spec = ExpertRecordSpec::new(A3B_INTER, A3B_HIDDEN, GS, 256);
        let layout = ExpertLayout::from_spec(40, 256, &spec, 4096);
        assert_eq!(layout.record_id(ExpertKey::new(0, 0)).0, 0);
        assert_eq!(layout.record_id(ExpertKey::new(0, 255)).0, 255);
        assert_eq!(layout.record_id(ExpertKey::new(1, 0)).0, 256);
    }

    #[test]
    fn header_round_trips() {
        let h = ExpertRecordHeader {
            layer: 7,
            expert: 42,
            inter: A3B_INTER as u32,
            hidden: A3B_HIDDEN as u32,
            group_size: GS as u32,
            scale2: [0.5, 0.25, 1.5],
            input_scale: [Some(2.0), None, Some(3.0)],
        };
        let bytes = h.to_bytes();
        assert_eq!(bytes.len(), ExpertRecordHeader::BYTES as usize);
        let back = ExpertRecordHeader::from_bytes(&bytes).expect("valid header");
        assert_eq!(back, h);
        assert_eq!(back.scale2, [0.5, 0.25, 1.5]);
        assert_eq!(back.input_scale, [Some(2.0), None, Some(3.0)]);
    }

    #[test]
    fn header_rejects_bad_magic_and_version() {
        let mut bytes = ExpertRecordHeader {
            layer: 0,
            expert: 0,
            inter: 1,
            hidden: 1,
            group_size: GS as u32,
            scale2: [1.0; 3],
            input_scale: [Some(1.0); 3],
        }
        .to_bytes();
        bytes[0] ^= 0xFF;
        assert!(ExpertRecordHeader::from_bytes(&bytes).is_none());
        assert!(ExpertRecordHeader::from_bytes(&bytes[..10]).is_none());
    }

    #[test]
    fn a3b_record_stride_is_4k_aligned_and_reasonable() {
        let spec = ExpertRecordSpec::new(A3B_INTER, A3B_HIDDEN, GS, 256);
        let layout = ExpertLayout::from_spec(40, 256, &spec, 4096);
        assert!(layout.record_stride - spec.raw_bytes() < 4096);
    }
}
