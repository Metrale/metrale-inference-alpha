// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: KV paging namespace and wire-key derivation, and the strict parsers for
//! the `METRALE_KV_PAGING*` variables.
//!
//! Owner: storage, KV paging tier.
//! Invariants:
//! - Every namespace returned is non-zero: `derive_kv_ns` falls back to `FNV_OFFSET`
//!   and an override of 0 is an error.
//! - The namespace is a pure function of its six arguments, encoded as the tagged
//!   records listed on [`derive_kv_ns`].
//! - The parsers never default: junk input is an error.
//!
//! The paging peer keys its residency by the u64 the client sends
//! (`snapshot_swap::run_paging_loop_shared`), so the namespace folded into that key
//! is what separates two models, and two clients of one model, on one peer. A KV
//! `GroupKey.block` is a client-local disk-block index
//! (`HighSpeedSwap::alloc_disk_block_id`), so two clients of one model use the same
//! block ids for unrelated data. The namespace therefore folds a per-client salt,
//! random per connect unless `METRALE_KV_PAGING_SALT` pins it: clients share peer
//! capacity, not cached blocks.
//!
//! The model fingerprint does not encode the KV cache dtype or block layout, and
//! `GroupLayout::group_id` depends on `num_blocks`, so the namespace folds the full
//! layout next to the fingerprint. The fingerprint arrives as a plain u64
//! (`ModelDims::model_fp`); this crate does not depend on the model config.

use std::num::NonZeroU64;

use anyhow::{Result, anyhow};

use crate::group::GroupLayout;

/// 2026-09-25: Encoding version, hashed as the tag-0x00 record. Changing it changes
/// every KV namespace.
pub const KV_NS_VERSION: u64 = 1;

/// 2026-09-25: Domain separator hashed into every KV namespace as the tag-0x02
/// record, so KV keys differ from SSM keys in the key material as well as by the
/// peer's per-`(kind, blob_bytes)` arena. ASCII `"KV"` + `"PAGE"` + 1.
pub const KV_DOMAIN: u64 = 0x4B56_5041_4745_0001;

pub(crate) use crate::tier::hash::{FNV_OFFSET, fnv1a_64};

pub(crate) use crate::tier::hash::mix64;

fn put_u64(buf: &mut Vec<u8>, tag: u8, v: u64) {
    buf.push(tag);
    buf.extend_from_slice(&v.to_le_bytes());
}

/// 2026-09-25: Derive the KV paging namespace: FNV-1a/64 over fixed-width
/// `[tag][8-byte LE]` records. A change to the fields or their order must bump
/// [`KV_NS_VERSION`]:
///
/// | tag  | field           | tag  | field          | tag  | field         |
/// |------|-----------------|------|----------------|------|---------------|
/// | 0x00 | KV_NS_VERSION   | 0x04 | block_size     | 0x08 | num_kv_heads  |
/// | 0x01 | model_fp        | 0x05 | head_dim       | 0x09 | fs_block_size |
/// | 0x02 | KV_DOMAIN       | 0x06 | num_layers     | 0x0a | group_stride  |
/// | 0x03 | elem_bytes      | 0x07 | num_blocks     | 0x0b | client_salt   |
///
/// `model_fp` is `ModelFingerprint::derive_kv` in metrale-model-engine: model type,
/// quantization identity, `METRALE_MODEL_ID` and geometry. A hash of 0 is replaced by
/// `FNV_OFFSET`.
pub fn derive_kv_ns(
    model_fp: u64,
    layout: &GroupLayout,
    elem_bytes: u32,
    block_size: u32,
    head_dim: u32,
    client_salt: u64,
) -> NonZeroU64 {
    let mut buf = Vec::with_capacity(12 * 9);
    for (tag, v) in [
        (0x00u8, KV_NS_VERSION),
        (0x01, model_fp),
        (0x02, KV_DOMAIN),
        (0x03, elem_bytes as u64),
        (0x04, block_size as u64),
        (0x05, head_dim as u64),
        (0x06, layout.num_layers as u64),
        (0x07, layout.num_blocks as u64),
        (0x08, layout.num_kv_heads as u64),
        (0x09, layout.fs_block_size),
        (0x0a, layout.group_stride),
        (0x0b, client_salt),
    ] {
        put_u64(&mut buf, tag, v);
    }
    let h = fnv1a_64(&buf);
    NonZeroU64::new(h).unwrap_or(NonZeroU64::new(FNV_OFFSET).expect("FNV offset is non-zero"))
}

/// 2026-09-25: Wire key for one KV block: `mix64(base_group_id, ns)`, the fold the SSM
/// tier's `PagingSnapshotStore::wire` also uses. For a fixed namespace it is a
/// bijection, so distinct group ids never share a key. The caller passes the id of
/// the block's first group, `group_id(GroupKey::new(layer, block, 0, K))`.
pub fn wire_key(ns: NonZeroU64, base_group_id: u64) -> u64 {
    mix64(base_group_id, ns.get())
}

/// 2026-09-25: Parse a u64 in decimal or `0x`/`0X` hex, after trimming; anything
/// else is an error naming `var`.
pub fn parse_u64_strict(var: &str, raw: &str) -> Result<u64> {
    let s = raw.trim();
    let parsed = match s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        Some(hex) => u64::from_str_radix(hex, 16),
        None => s.parse::<u64>(),
    };
    parsed.map_err(|e| anyhow!("{var}={raw:?} is not a valid u64 (decimal or 0x-hex): {e}"))
}

/// 2026-09-25: Resolve the `METRALE_KV_PAGING_NS` value: `None` returns `derived`; a
/// value is parsed strictly and 0 is an error.
pub fn resolve_kv_ns_from(override_raw: Option<&str>, derived: NonZeroU64) -> Result<NonZeroU64> {
    match override_raw {
        None => Ok(derived),
        Some(raw) => {
            let v = parse_u64_strict("METRALE_KV_PAGING_NS", raw)?;
            NonZeroU64::new(v).ok_or_else(|| {
                anyhow!(
                    "METRALE_KV_PAGING_NS=0 is invalid: ns=0 is unrepresentable (it would \
                     cross-serve KV state on a shared peer); unset it to use the derived \
                     namespace (logged at INFO on connect)"
                )
            })
        }
    }
}

/// 2026-09-25: Resolve the `METRALE_KV_PAGING_SALT` value: `None` returns `Ok(None)`,
/// and the caller draws a random salt; a value is parsed strictly.
pub fn resolve_salt_from(raw: Option<&str>) -> Result<Option<u64>> {
    raw.map(|r| parse_u64_strict("METRALE_KV_PAGING_SALT", r))
        .transpose()
}

/// 2026-09-25: Resolve the `METRALE_KV_PAGING` value after trimming: unset or `0`
/// is `false` (the raw one-sided `RdmaKvBackend`), `1` is `true` (the paging
/// backend), anything else is an error.
pub fn kv_paging_selected(raw: Option<&str>) -> Result<bool> {
    match raw.map(str::trim) {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => Err(anyhow!(
            "METRALE_KV_PAGING={other:?} is invalid: 1 = peer-owned paging KV, 0/unset = the \
             raw one-sided KV blade"
        )),
    }
}

/// 2026-09-25: Resolve `METRALE_KV_PAGING_ARENA_GB`, the peer arena size in GiB
/// (fractional accepted), floored to a multiple of `block_bytes`. Unset, non-finite,
/// not above 0, or smaller than one block is an error. The raw backend sizes its
/// arena to hold every group (`RdmaKvBackend::connect`); the paging arena has no
/// such size to derive, so it must be given.
pub fn resolve_arena_bytes_from(raw: Option<&str>, block_bytes: u64) -> Result<u64> {
    let raw = raw.ok_or_else(|| {
        anyhow!(
            "METRALE_KV_PAGING=1 requires METRALE_KV_PAGING_ARENA_GB (peer warm-arena size in \
             GiB, fractional ok) — explicit config or fail fast (PCND)"
        )
    })?;
    let gb: f64 = raw
        .trim()
        .parse()
        .map_err(|e| anyhow!("METRALE_KV_PAGING_ARENA_GB={raw:?} is not a number: {e}"))?;
    if !gb.is_finite() || gb <= 0.0 {
        return Err(anyhow!(
            "METRALE_KV_PAGING_ARENA_GB={raw:?} must be a finite value > 0"
        ));
    }
    let bb = block_bytes.max(1);
    let arena = ((gb * (1u64 << 30) as f64) as u64 / bb) * bb;
    if arena == 0 {
        return Err(anyhow!(
            "METRALE_KV_PAGING_ARENA_GB={raw} is smaller than one KV block ({bb} B) — the \
             warm arena must hold at least one block"
        ));
    }
    Ok(arena)
}

/// 2026-09-25: `true` when `kv_peer_set` and `flag_raw` selects the paging backend;
/// an invalid `flag_raw` is an error. `CascadeBackend` evicts through per-head
/// `write_from_host`, which `KvPagingBackend` refuses.
pub fn cascade_conflicts_with_paging(kv_peer_set: bool, flag_raw: Option<&str>) -> Result<bool> {
    Ok(kv_peer_set && kv_paging_selected(flag_raw)?)
}

#[cfg(test)]
#[path = "ns_tests.rs"]
mod tests;
