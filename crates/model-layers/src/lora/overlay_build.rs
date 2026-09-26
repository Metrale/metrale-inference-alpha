// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The device build of the token overlay, in two steps because
//! the adapter's tensors and the served embed/lm_head tables are not alive at
//! the same time:
//! - [`stage_overlay_raw`], at adapter load: copy the overlay tensors into
//!   device buffers of their own ([`OverlayRaw`]) while the adapter's
//!   [`WeightStore`] exists;
//! - [`build_overlay`], from model-engine's `set_lora_weights`: diff against
//!   the served tables, compact the overridden rows, and build the `slot_map`
//!   ([`EmbedOverlay`]).
//!
//! Owner: model-layers (lora).
//! Invariants: none beyond the types.

use anyhow::{Result, bail};
use metrale_config::PeftAdapterConfig;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_weights::weights::WeightStore;

use super::overlay::{
    OverlayTensors, ROWDIFF_THRESH, build_override_set, clamp_trainable_to_vocab, override_source,
};
use crate::layers::ops::token_overlay::{self, OverlayKernels};

const BF16_BYTES: usize = 2;
const F32_BYTES: usize = 4;

/// 2026-09-25: The adapter's overlay tensors copied to device buffers of their
/// own, with their row counts: `*_base` is `[*_r, h]` and `*_delta` is
/// `[*_t, h]`. `*_base` holds either the `token_adapter` base table or a
/// `modules_to_save` full table.
#[derive(Debug, Clone, Copy, Default)]
pub struct OverlayRaw {
    pub embed_base: Option<DevicePtr>,
    pub embed_delta: Option<DevicePtr>,
    pub embed_r: u32,
    pub embed_t: u32,
    pub lmhead_base: Option<DevicePtr>,
    pub lmhead_delta: Option<DevicePtr>,
    pub lmhead_r: u32,
    pub lmhead_t: u32,
}

/// 2026-09-25: One adapter's [`OverlayRaw`] and its trainable ids.
pub struct OverlayRawSlot {
    pub raw: OverlayRaw,
    /// 2026-09-25: PEFT `trainable_token_indices` as given; `build_overlay`
    /// clamps them to the served vocab.
    pub trainable: Vec<u32>,
}

/// 2026-09-25: One adapter slot's embed overlay: `rows` (`[n_override, h]`
/// BF16) in the order of `ids_dev` (ascending u32 vocab ids), and `slot_map`
/// (i32 `[vocab]`, the row index for an overridden id, else -1), which the
/// embed kernel indexes by token id. `lmhead` is `Some` only for an untied head
/// with its own overlay tensors.
#[derive(Debug)]
pub struct EmbedOverlay {
    pub rows: DevicePtr,
    pub ids_dev: DevicePtr,
    pub slot_map: DevicePtr,
    pub n_override: u32,
    /// 2026-09-25: The length of `slot_map`, the served vocab; the embed
    /// kernel skips token ids `>= vocab` (`token_overlay.cu`).
    pub vocab: u32,
    pub lmhead: Option<LmHeadOverlay>,
}

/// 2026-09-25: The overlay of an untied lm_head: `rows` (`[n_override, h]`
/// BF16) and their vocab ids. The lm_head kernel writes `dot(hidden, row)` to
/// each overridden id's logit.
#[derive(Debug)]
pub struct LmHeadOverlay {
    pub rows: DevicePtr,
    pub ids_dev: DevicePtr,
    pub n_override: u32,
}

/// 2026-09-25: f32 → BF16 bits, round to nearest even.
fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    let round = ((bits >> 16) & 1) + 0x7fff;
    (bits.wrapping_add(round) >> 16) as u16
}

/// 2026-09-25: Copy one store tensor to a new device buffer; returns it and
/// `shape[0]`. An error unless the shape is `[rows, h]`.
fn stage_tensor(
    store: &WeightStore,
    name: &str,
    h: usize,
    gpu: &dyn GpuBackend,
) -> Result<(DevicePtr, u32)> {
    let t = store.get(name)?;
    if t.shape.len() != 2 || t.shape[1] != h {
        bail!(
            "REJECT[overlay-shape]: '{name}' is {:?}, expected [rows, {h}] (hidden)",
            t.shape
        );
    }
    let bytes: usize = t.shape.iter().product::<usize>() * t.dtype.byte_size();
    let dst = gpu.alloc(bytes)?;
    gpu.copy_d2d(t.ptr, dst, bytes)?;
    Ok((dst, t.shape[0] as u32))
}

/// 2026-09-25: Copy one adapter's overlay tensors to device buffers of their
/// own. `Ok(None)` when it has none, or has a `lora_embedding` tensor (the
/// audit refuses those). A `modules_to_save` table goes into `*_base` when
/// there is no `token_adapter` base. An error when neither table has a base.
pub fn stage_overlay_raw(
    store: &WeightStore,
    overlay: &OverlayTensors,
    peft: &PeftAdapterConfig,
    h: usize,
    gpu: &dyn GpuBackend,
) -> Result<Option<OverlayRawSlot>> {
    if overlay.is_empty() || overlay.lora_embedding_seen {
        return Ok(None);
    }
    let mut raw = OverlayRaw::default();
    let embed_base_name = overlay.embed_base.as_ref().or(overlay.embed_full.as_ref());
    if let Some(name) = embed_base_name {
        let (p, r) = stage_tensor(store, name, h, gpu)?;
        raw.embed_base = Some(p);
        raw.embed_r = r;
    }
    if let Some(name) = &overlay.embed_delta {
        let (p, t) = stage_tensor(store, name, h, gpu)?;
        raw.embed_delta = Some(p);
        raw.embed_t = t;
    }
    let lmhead_base_name = overlay
        .lmhead_base
        .as_ref()
        .or(overlay.lmhead_full.as_ref());
    if let Some(name) = lmhead_base_name {
        let (p, r) = stage_tensor(store, name, h, gpu)?;
        raw.lmhead_base = Some(p);
        raw.lmhead_r = r;
    }
    if let Some(name) = &overlay.lmhead_delta {
        let (p, t) = stage_tensor(store, name, h, gpu)?;
        raw.lmhead_delta = Some(p);
        raw.lmhead_t = t;
    }
    if raw.embed_base.is_none() && raw.lmhead_base.is_none() {
        bail!(
            "REJECT[overlay-no-base]: overlay tensors present but no embed/lm_head base row table"
        );
    }
    Ok(Some(OverlayRawSlot {
        raw,
        trainable: peft.trainable_token_indices.clone(),
    }))
}

/// 2026-09-25: The overridden ids (ascending) and their `[n, h]` BF16 rows on
/// the device, for the embed and lm_head builds.
struct Compact {
    ids: Vec<u32>,
    rows_dev: DevicePtr,
    ids_dev: DevicePtr,
    n: u32,
}

/// 2026-09-25: Diff the first `min(r, vocab)` rows of `base` against `served`,
/// add the `kept` trainable ids, and build the BF16 replacement rows, each from
/// the delta or the base row per `override_source`. `None` when no row is
/// overridden. Delta rows are read as f32.
#[allow(clippy::too_many_arguments)]
fn compact_override(
    gpu: &dyn GpuBackend,
    kernels: &OverlayKernels,
    base: DevicePtr,
    delta: Option<DevicePtr>,
    r: u32,
    served: DevicePtr,
    vocab: usize,
    h: usize,
    kept: &[u32],
    stream: u64,
) -> Result<Option<Compact>> {
    let r_eff = (r as usize).min(vocab);
    let flags_dev = gpu.alloc(r_eff.max(1))?;
    if r_eff > 0 {
        token_overlay::embed_rowdiff(
            gpu,
            kernels.rowdiff,
            base,
            served,
            flags_dev,
            r_eff as u32,
            h as u32,
            ROWDIFF_THRESH,
            stream,
        )?;
        gpu.synchronize(stream)?;
    }
    let mut flags = vec![0u8; r_eff];
    if r_eff > 0 {
        gpu.copy_d2h(flags_dev, &mut flags)?;
    }
    let _ = gpu.free(flags_dev);
    let row_diff: Vec<bool> = flags.iter().map(|&b| b != 0).collect();
    let ids = build_override_set(&row_diff, kept);
    if ids.is_empty() {
        return Ok(None);
    }
    let n = ids.len();
    let mut compact = vec![0u8; n * h * BF16_BYTES];
    let mut frow = vec![0u8; h * F32_BYTES];
    for (ci, &id) in ids.iter().enumerate() {
        let dst = &mut compact[ci * h * BF16_BYTES..(ci + 1) * h * BF16_BYTES];
        match override_source(id, kept) {
            Some(k) => {
                let d = delta.ok_or_else(|| {
                    anyhow::anyhow!(
                        "REJECT[overlay-delta-missing]: trainable id {id} but no delta tensor"
                    )
                })?;
                gpu.copy_d2h(d.offset(k * h * F32_BYTES), &mut frow)?;
                for i in 0..h {
                    let x = f32::from_le_bytes([
                        frow[i * 4],
                        frow[i * 4 + 1],
                        frow[i * 4 + 2],
                        frow[i * 4 + 3],
                    ]);
                    dst[i * 2..i * 2 + 2].copy_from_slice(&f32_to_bf16(x).to_le_bytes());
                }
            }
            None => {
                gpu.copy_d2h(base.offset(id as usize * h * BF16_BYTES), dst)?;
            }
        }
    }
    let rows_dev = gpu.alloc(compact.len())?;
    gpu.copy_h2d(&compact, rows_dev)?;
    let ids_bytes: Vec<u8> = ids.iter().flat_map(|i| i.to_le_bytes()).collect();
    let ids_dev = gpu.alloc(ids_bytes.len())?;
    gpu.copy_h2d(&ids_bytes, ids_dev)?;
    Ok(Some(Compact {
        ids,
        rows_dev,
        ids_dev,
        n: n as u32,
    }))
}

/// 2026-09-25: Build one slot's [`EmbedOverlay`]: clamp the trainable ids,
/// diff and compact the embed rows, build the `slot_map`, and, for an untied
/// head (`!tied`) with its own base, the lm_head rows. The staged buffers are
/// freed when an overlay is returned. `Ok(None)` when the slot has no embed
/// base or overrides no row. An error when the overlay kernels are not loaded.
/// Model-engine passes `tied` when the lm_head shares the embed buffer or is
/// served quantized; `TokenOverlaySet::from_slots` then points the lm_head at
/// the embed rows.
#[allow(clippy::too_many_arguments)]
pub fn build_overlay(
    gpu: &dyn GpuBackend,
    kernels: &OverlayKernels,
    slot: &OverlayRawSlot,
    served_embed: DevicePtr,
    served_lmhead: DevicePtr,
    vocab: usize,
    h: usize,
    tied: bool,
    stream: u64,
) -> Result<Option<EmbedOverlay>> {
    if kernels.rowdiff.0 == 0 || kernels.embed_overlay.0 == 0 {
        bail!(
            "REJECT[overlay-kernels-missing]: adapter ships token-overlay tensors but the \
             token_overlay CUDA kernels are not loaded (rebuild with the kernel image)"
        );
    }
    let raw = &slot.raw;
    let Some(embed_base) = raw.embed_base else {
        return Ok(None);
    };
    let (kept, skipped) = clamp_trainable_to_vocab(&slot.trainable, raw.embed_r as usize, vocab)?;
    if skipped > 0 {
        tracing::warn!(
            "LoRA overlay: dropped {skipped} vocab-extension trainable id(s) beyond served vocab {vocab}"
        );
    }
    let Some(embed) = compact_override(
        gpu,
        kernels,
        embed_base,
        raw.embed_delta,
        raw.embed_r,
        served_embed,
        vocab,
        h,
        &kept,
        stream,
    )?
    else {
        return Ok(None);
    };
    let mut slot_map = vec![-1i32; vocab];
    for (ci, &id) in embed.ids.iter().enumerate() {
        slot_map[id as usize] = ci as i32;
    }
    let sm_bytes: Vec<u8> = slot_map.iter().flat_map(|i| i.to_le_bytes()).collect();
    let slot_map_dev = gpu.alloc(sm_bytes.len())?;
    gpu.copy_h2d(&sm_bytes, slot_map_dev)?;

    let lmhead = if let Some(base) = raw.lmhead_base.filter(|_| !tied) {
        compact_override(
            gpu,
            kernels,
            base,
            raw.lmhead_delta,
            raw.lmhead_r,
            served_lmhead,
            vocab,
            h,
            &kept,
            stream,
        )?
        .map(|c| LmHeadOverlay {
            rows: c.rows_dev,
            ids_dev: c.ids_dev,
            n_override: c.n,
        })
    } else {
        None
    };

    for p in [
        raw.embed_base,
        raw.embed_delta,
        raw.lmhead_base,
        raw.lmhead_delta,
    ]
    .into_iter()
    .flatten()
    {
        let _ = gpu.free(p);
    }

    Ok(Some(EmbedOverlay {
        rows: embed.rows_dev,
        ids_dev: embed.ids_dev,
        slot_map: slot_map_dev,
        n_override: embed.n,
        vocab: vocab as u32,
        lmhead,
    }))
}

#[cfg(test)]
#[path = "overlay_build_tests.rs"]
mod tests;
