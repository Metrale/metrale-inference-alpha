// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The expert-store format. `pack_record` and `unpack_record` build
//! and parse one record in memory, with no I/O. `ExpertIndex` is the store's
//! `manifest.json`, from which a reader rebuilds the record spec and layout.
//! The layer files are written and read in `expert_pack_fs.rs`.
//!
//! Owner: metrale-storage experts.
//! Invariants:
//! - `ExpertIndex::load` and `ExpertFileReader::open` reject a manifest whose
//!   `version` is not `ExpertRecordHeader::VERSION`.

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::expert::ExpertKey;
use crate::expert::{ExpertLayout, ExpertRecordHeader, ExpertRecordSpec, Proj};

/// 2026-09-25: One projection's packed and scale bytes, as `pack_record`
/// writes them.
#[derive(Clone, Copy, Debug)]
pub struct ProjData<'a> {
    pub packed: &'a [u8],
    pub scale: &'a [u8],
}

/// 2026-09-25: One projection's sub-buffers inside a parsed record.
#[derive(Clone, Copy, Debug)]
pub struct ProjView<'a> {
    pub packed: &'a [u8],
    pub scale: &'a [u8],
}

/// 2026-09-25: Build one `stride`-byte record: the header at offset 0, each
/// projection's bytes at the spec's offsets, zeros elsewhere. Fails when
/// `stride` is below `spec.raw_bytes()` or a projection's lengths differ from
/// the spec.
pub fn pack_record(
    spec: &ExpertRecordSpec,
    stride: u64,
    header: &ExpertRecordHeader,
    projs: &[ProjData; 3],
) -> Result<Vec<u8>> {
    let stride = stride as usize;
    if (spec.raw_bytes() as usize) > stride {
        bail!(
            "record stride {} smaller than raw record bytes {}",
            stride,
            spec.raw_bytes()
        );
    }
    let mut buf = vec![0u8; stride];
    let hdr = header.to_bytes();
    buf[..hdr.len()].copy_from_slice(&hdr);

    for p in Proj::ALL {
        let pb = spec.proj_bytes(p);
        let d = &projs[p as usize];
        if d.packed.len() as u64 != pb.packed_bytes {
            bail!(
                "{:?} packed len {} != expected {}",
                p,
                d.packed.len(),
                pb.packed_bytes
            );
        }
        if d.scale.len() as u64 != pb.scale_bytes {
            bail!(
                "{:?} scale len {} != expected {}",
                p,
                d.scale.len(),
                pb.scale_bytes
            );
        }
        let po = spec.packed_off(p) as usize;
        let so = spec.scale_off(p) as usize;
        buf[po..po + d.packed.len()].copy_from_slice(d.packed);
        buf[so..so + d.scale.len()].copy_from_slice(d.scale);
    }
    Ok(buf)
}

/// 2026-09-25: Parse a record of at least `spec.raw_bytes()` bytes into its
/// header and projection views. Fails on a short buffer or a header whose
/// magic or version differs.
pub fn unpack_record<'a>(
    spec: &ExpertRecordSpec,
    buf: &'a [u8],
) -> Result<(ExpertRecordHeader, [ProjView<'a>; 3])> {
    if (buf.len() as u64) < spec.raw_bytes() {
        bail!(
            "record buffer {} smaller than raw record bytes {}",
            buf.len(),
            spec.raw_bytes()
        );
    }
    let header = ExpertRecordHeader::from_bytes(buf)
        .context("record header magic/version mismatch (wrong file or format version?)")?;
    let mut views = [ProjView {
        packed: &[],
        scale: &[],
    }; 3];
    for p in Proj::ALL {
        let pb = spec.proj_bytes(p);
        let po = spec.packed_off(p) as usize;
        let so = spec.scale_off(p) as usize;
        views[p as usize] = ProjView {
            packed: &buf[po..po + pb.packed_bytes as usize],
            scale: &buf[so..so + pb.scale_bytes as usize],
        };
    }
    Ok((header, views))
}

/// 2026-09-25: The store manifest, `manifest.json` beside the per-layer
/// files: the geometry a reader needs to rebuild the spec and layout and open
/// the files.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ExpertIndex {
    /// 2026-09-25: Must equal [`ExpertRecordHeader::VERSION`] for a reader to
    /// accept the store.
    pub version: u32,
    pub num_moe_layers: u32,
    pub num_experts: u32,
    pub inter: u64,
    pub hidden: u64,
    pub group_size: u64,
    pub sub_align: u64,
    pub fs_block_size: u64,
    pub record_stride: u64,
    pub record_raw_bytes: u64,
    /// 2026-09-25: Set to `FILE_TEMPLATE` by `new`; `file_name` does not read
    /// it.
    pub file_template: String,
    /// 2026-09-25: Dense MoE-layer index → model layer index; `new` sets
    /// `num_moe_layers` to its length.
    pub moe_layer_to_model_layer: Vec<u32>,
}

impl ExpertIndex {
    pub const FILE_TEMPLATE: &'static str = "experts_{:05}.xpr";
    pub const MANIFEST_NAME: &'static str = "manifest.json";

    pub fn new(
        inter: u64,
        hidden: u64,
        group_size: u64,
        sub_align: u64,
        fs_block_size: u64,
        moe_layer_to_model_layer: Vec<u32>,
        num_experts: u32,
    ) -> Self {
        let spec = ExpertRecordSpec::new(inter, hidden, group_size, sub_align);
        let layout = ExpertLayout::from_spec(
            moe_layer_to_model_layer.len() as u32,
            num_experts,
            &spec,
            fs_block_size,
        );
        Self {
            version: ExpertRecordHeader::VERSION,
            num_moe_layers: moe_layer_to_model_layer.len() as u32,
            num_experts,
            inter,
            hidden,
            group_size,
            sub_align,
            fs_block_size,
            record_stride: layout.record_stride,
            record_raw_bytes: spec.raw_bytes(),
            file_template: Self::FILE_TEMPLATE.to_string(),
            moe_layer_to_model_layer,
        }
    }

    /// 2026-09-25: Read and version-check `manifest.json` from `dir`, without
    /// opening the layer files.
    #[cfg(unix)]
    pub fn load(dir: &std::path::Path) -> Result<Self> {
        let p = dir.join(Self::MANIFEST_NAME);
        let json = std::fs::read_to_string(&p).with_context(|| format!("read {}", p.display()))?;
        let index: ExpertIndex =
            serde_json::from_str(&json).with_context(|| format!("parse {}", p.display()))?;
        if index.version != ExpertRecordHeader::VERSION {
            bail!(
                "manifest version {} != supported {}",
                index.version,
                ExpertRecordHeader::VERSION
            );
        }
        Ok(index)
    }

    pub fn spec(&self) -> ExpertRecordSpec {
        ExpertRecordSpec::new(self.inter, self.hidden, self.group_size, self.sub_align)
    }

    pub fn layout(&self) -> ExpertLayout {
        ExpertLayout::from_spec(
            self.num_moe_layers,
            self.num_experts,
            &self.spec(),
            self.fs_block_size,
        )
    }

    /// 2026-09-25: `experts_{moe_layer:05}.xpr`, whatever `file_template` holds.
    pub fn file_name(&self, moe_layer: u32) -> String {
        format!("experts_{moe_layer:05}.xpr")
    }

    /// 2026-09-25: Bytes of all layer files together.
    pub fn total_bytes(&self) -> u64 {
        (self.num_moe_layers as u64) * self.layout().bytes_per_layer()
    }
}

pub use fs_impl::{ExpertFileReader, ExpertFileWriter};

#[path = "expert_pack_fs.rs"]
mod fs_impl;

#[cfg(test)]
#[path = "expert_pack_tests.rs"]
mod tests;
