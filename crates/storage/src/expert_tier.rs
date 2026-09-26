// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `ExpertTier`: fetch one expert record into a slot and return
//! the device addresses of its six sub-buffers, with its scalars. The tiers are
//! `PosixTier` (read, then copy into a device buffer), `UmaArenaTier` (read
//! straight into a pinned arena slot) and `RdmaTier` (expert_tier_rdma.rs, from
//! a peer). It is not a `StorageBackend`, whose reads take KV group keys.
//!
//! Owner: metrale-storage experts.
//! Invariants:
//! - Every tier builds its `ExpertResidency` with `residency_from`, which
//!   rejects a record whose header names another (layer, expert).

use crate::tier::pio;
use anyhow::{Context, Result, bail};
use std::fs::{File, OpenOptions};
use std::path::Path;

use crate::cuda_min::{DeviceBuffer, copy_h_to_d_async, stream_sync};
use crate::expert::{ExpertKey, ExpertLayout, ExpertRecordHeader, ExpertRecordSpec, Proj};
use crate::expert_arena::ExpertArena;
use crate::expert_pack::{ExpertFileReader, ExpertIndex};

/// 2026-09-25: The six sub-buffer device addresses and the scalars of one
/// fetched expert, each array ordered gate, up, down.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExpertResidency {
    pub packed_addr: [u64; 3],
    pub scale_addr: [u64; 3],
    pub scale2: [f32; 3],
    /// 2026-09-25: `None` where the record has no input scale.
    pub input_scale: [Option<f32>; 3],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TierKind {
    Posix,
    Uma,
    Rdma,
}

/// 2026-09-25: A destination slot: which slab, and which slot in it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArenaSlot {
    pub slab: u32,
    pub slot: u32,
}

impl ArenaSlot {
    pub fn new(slab: u32, slot: u32) -> Self {
        Self { slab, slot }
    }
}

/// 2026-09-25: Fetch an expert record into a slot and return its residency.
pub trait ExpertTier: Send {
    fn fetch(&mut self, key: ExpertKey, slot: ArenaSlot, stream: u64) -> Result<ExpertResidency>;
    fn kind(&self) -> TierKind;
    /// 2026-09-25: Whether the tier can still serve fetches; the default is
    /// `true`.
    fn healthy(&self) -> bool {
        true
    }
}

/// 2026-09-25: Build an `ExpertResidency` from a record's header and the
/// record's device base address, using the spec's sub-buffer offsets.
pub(crate) fn residency_from(
    spec: &ExpertRecordSpec,
    record: &[u8],
    base_dev_va: u64,
    key: ExpertKey,
) -> Result<ExpertResidency> {
    let hdr = ExpertRecordHeader::from_bytes(record)
        .context("expert record header magic/version mismatch")?;
    // 2026-09-25: A record read from the wrong place fails here rather than
    // being served.
    if hdr.layer != key.layer || hdr.expert != key.expert {
        bail!(
            "expert record identity mismatch: header ({},{}) != requested {:?}",
            hdr.layer,
            hdr.expert,
            key
        );
    }
    let mut packed_addr = [0u64; 3];
    let mut scale_addr = [0u64; 3];
    for p in Proj::ALL {
        packed_addr[p as usize] = base_dev_va + spec.packed_off(p);
        scale_addr[p as usize] = base_dev_va + spec.scale_off(p);
    }
    Ok(ExpertResidency {
        packed_addr,
        scale_addr,
        scale2: hdr.scale2,
        input_scale: hdr.input_scale,
    })
}

/// 2026-09-25: Read the record into a host buffer (buffered, through
/// `ExpertFileReader`), copy it into the slot's part of one device buffer, and
/// sync the stream.
pub struct PosixTier {
    reader: ExpertFileReader,
    spec: ExpertRecordSpec,
    layout: ExpertLayout,
    /// 2026-09-25: One device buffer of `num_slabs * slots_per_slab` records.
    dev: DeviceBuffer,
    slots_per_slab: u32,
    num_slabs: u32,
}

impl PosixTier {
    pub fn open(dir: &Path, num_slabs: u32, slots_per_slab: u32) -> Result<Self> {
        let reader = ExpertFileReader::open(dir)?;
        let index: &ExpertIndex = reader.index();
        let spec = index.spec();
        let layout = index.layout();
        if num_slabs == 0 || slots_per_slab == 0 {
            bail!("PosixTier: zero geometry ({num_slabs},{slots_per_slab})");
        }
        let stride = layout.record_stride as usize;
        // 2026-09-25: Checked, so an overflowed size cannot allocate less than
        // the slots `slot_dev_va` addresses.
        let total = (num_slabs as usize)
            .checked_mul(slots_per_slab as usize)
            .and_then(|v| v.checked_mul(stride))
            .context("PosixTier: arena size overflow")?;
        let dev = DeviceBuffer::new(total)?;
        Ok(Self {
            reader,
            spec,
            layout,
            dev,
            slots_per_slab,
            num_slabs,
        })
    }

    fn slot_dev_va(&self, slot: ArenaSlot) -> Result<u64> {
        if slot.slab >= self.num_slabs || slot.slot >= self.slots_per_slab {
            bail!("PosixTier: slot {:?} out of range", slot);
        }
        let i = (slot.slab as u64) * (self.slots_per_slab as u64) + (slot.slot as u64);
        Ok(self.dev.ptr + i * self.layout.record_stride)
    }
}

impl ExpertTier for PosixTier {
    fn fetch(&mut self, key: ExpertKey, slot: ArenaSlot, stream: u64) -> Result<ExpertResidency> {
        let record = self.reader.read_record_raw(key)?;
        let dev_va = self.slot_dev_va(slot)?;
        copy_h_to_d_async(dev_va, record.as_ptr() as *const _, record.len(), stream)?;
        // 2026-09-25: The copy reads `record`, which is freed when this returns.
        stream_sync(stream)?;
        residency_from(&self.spec, &record, dev_va, key)
    }
    fn kind(&self) -> TierKind {
        TierKind::Posix
    }
}

/// 2026-09-25: Read the record straight into a pinned arena slot (O_DIRECT on
/// Linux); the returned addresses point into the arena, and nothing is copied.
pub struct UmaArenaTier {
    // 2026-09-25: One file per MoE layer, O_DIRECT on Linux.
    files: Vec<File>,
    spec: ExpertRecordSpec,
    layout: ExpertLayout,
    arena: ExpertArena,
}

impl UmaArenaTier {
    pub fn open(dir: &Path, num_slabs: u32, slots_per_slab: u32) -> Result<Self> {
        let reader = ExpertFileReader::open(dir)?;
        let index: &ExpertIndex = reader.index();
        let spec = index.spec();
        let layout = index.layout();
        // 2026-09-25: The reader's files are buffered; these are opened again
        // with `set_direct_flag`.
        let mut files = Vec::with_capacity(index.num_moe_layers as usize);
        for l in 0..index.num_moe_layers {
            let p = dir.join(index.file_name(l));
            let mut opts = OpenOptions::new();
            opts.read(true);
            set_direct_flag(&mut opts);
            let f = opts
                .open(&p)
                .with_context(|| format!("open {}", p.display()))?;
            files.push(f);
        }
        let arena = ExpertArena::new(num_slabs, slots_per_slab, layout.record_stride as usize)?;
        Ok(Self {
            files,
            spec,
            layout,
            arena,
        })
    }

    pub fn arena(&self) -> &ExpertArena {
        &self.arena
    }
}

impl ExpertTier for UmaArenaTier {
    fn fetch(&mut self, key: ExpertKey, slot: ArenaSlot, _stream: u64) -> Result<ExpertResidency> {
        let stride = self.layout.record_stride as usize;
        let host = self.arena.slot_host_ptr(slot.slab, slot.slot)?;
        let file = self
            .files
            .get(key.layer as usize)
            .with_context(|| format!("UmaArenaTier: no file for layer {}", key.layer))?;
        let off = self.layout.file_offset(key);
        // 2026-09-25: The stride is a multiple of 4096 (`ExpertArena::new`), as
        // O_DIRECT needs. SAFETY: `host` points at a slot of `stride` bytes
        // inside the pinned arena, and the slice covers exactly that slot.
        let dst = unsafe { std::slice::from_raw_parts_mut(host, stride) };
        pio::read_exact_at(file, dst, off)
            .with_context(|| format!("UmaArenaTier read {key:?} at {off}"))?;
        // 2026-09-25: SAFETY: the slot holds the `stride` bytes just read.
        let record = unsafe { std::slice::from_raw_parts(host, stride) };
        let dev_va = self.arena.slot_dev_va(slot.slab, slot.slot)?;
        residency_from(&self.spec, record, dev_va, key)
    }
    fn kind(&self) -> TierKind {
        TierKind::Uma
    }
}

/// 2026-09-25: Open the tier named by `backend`:
///   * `posix`, `uma`: read the store in `dir`;
///   * `rdma`: `RdmaTier` over TCP to `$METRALE_EXPERT_PEER`;
///   * `rdma-verbs`: `RdmaTier` with one-sided RDMA READ from
///     `$METRALE_EXPERT_PEER`.
pub fn open_tier(
    backend: &str,
    dir: &Path,
    num_slabs: u32,
    slots_per_slab: u32,
) -> Result<Box<dyn ExpertTier>> {
    // 2026-09-25: The RDMA tiers exist on unix only; elsewhere asking for one
    // is an error.
    let rdma = |use_verbs: bool| -> Result<Box<dyn ExpertTier>> {
        let flag = if use_verbs { "rdma-verbs" } else { "rdma" };
        #[cfg(unix)]
        {
            let addr = std::env::var("METRALE_EXPERT_PEER").map_err(|_| {
                anyhow::anyhow!("--expert-backend {flag} needs $METRALE_EXPERT_PEER=host:port")
            })?;
            Ok(Box::new(crate::expert_tier_rdma::RdmaTier::connect(
                &addr,
                num_slabs,
                slots_per_slab,
                use_verbs,
            )?))
        }
        #[cfg(not(unix))]
        {
            bail!("--expert-backend {flag} requires rdma-core, which is unix-only")
        }
    };
    match backend {
        "posix" => Ok(Box::new(PosixTier::open(dir, num_slabs, slots_per_slab)?)),
        "uma" => Ok(Box::new(UmaArenaTier::open(
            dir,
            num_slabs,
            slots_per_slab,
        )?)),
        "rdma" => rdma(false),
        "rdma-verbs" => rdma(true),
        other => bail!("unknown expert backend '{other}' (want posix|uma|rdma|rdma-verbs)"),
    }
}

/// 2026-09-25: Copy `len` bytes from a device address into a new Vec, then
/// sync the stream.
pub fn read_device(dev_va: u64, len: usize, stream: u64) -> Result<Vec<u8>> {
    use crate::cuda_min::copy_d_to_h_async;
    let mut out = vec![0u8; len];
    copy_d_to_h_async(out.as_mut_ptr() as *mut _, dev_va, len, stream)?;
    stream_sync(stream)?;
    Ok(out)
}

/// 2026-09-25: O_DIRECT on Linux; other targets read buffered (the `layout`
/// module header gives the reason for Windows).
#[cfg(target_os = "linux")]
fn set_direct_flag(opts: &mut OpenOptions) {
    use std::os::unix::fs::OpenOptionsExt;
    opts.custom_flags(libc::O_DIRECT);
}

#[cfg(not(target_os = "linux"))]
fn set_direct_flag(_opts: &mut OpenOptions) {}
