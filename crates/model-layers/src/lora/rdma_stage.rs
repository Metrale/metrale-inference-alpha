// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The model-layers side of the RDMA adapter swap: fetch a
//! peer-staged adapter's manifest, map each tensor to its landing offset in a
//! pool slot, and rebuild the slot's pairs after model-engine's
//! `swap_lora_slot_from_peer` has landed them with `RdmaLoraLoader`.
//!
//! Owner: model-layers (lora).
//! Invariants: none beyond the types.

use std::collections::BTreeMap;

use anyhow::{Result, anyhow, bail};
use metrale_config::{ModelConfig, PeftAdapterConfig};
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_model_weights::weight_lora_rdma::{LoraAbKind, LoraLandTarget};
use metrale_storage::weight_peer::WeightManifest;

use super::{
    AdapterAb, LoraLayerWeights, LoraModule, LoraTarget, classify_key, module_slot_offsets,
    pool_slot_bytes, slot_base_offset,
};
use crate::layers::ops::lora_delta::LoraPair;
use crate::weight_map::DenseWeight;

/// 2026-09-25: The landing targets of one adapter's manifest in pool `slot`.
/// Each tensor goes through `classify_key` and lands at
/// `pool + slot_base_offset + a_off` or `+ b_off`. The rank is read from the
/// tensor shape (A is `[r, in]`, B is `[out, r]`). Refused: whatever
/// `classify_key` refuses, router and expert tensors, a wrong shape, a zero
/// rank or one above `max_rank`, duplicate or unpaired tensors, A/B rank
/// disagreement, and an empty manifest.
pub fn build_land_targets(
    manifest: &WeightManifest,
    cfg: &ModelConfig,
    pool: DevicePtr,
    slot: usize,
    max_rank: usize,
) -> Result<Vec<LoraLandTarget>> {
    let base = pool.0 + slot_base_offset(slot, cfg, max_rank) as u64;
    let mut targets = Vec::with_capacity(manifest.tensors.len());
    let mut pairs: BTreeMap<(usize, LoraModule), [Option<usize>; 2]> = BTreeMap::new();
    for rec in &manifest.tensors {
        let (layer, target, ab) = classify_key(&rec.name, cfg)?;
        // 2026-09-25: Only the slot pool is landed; router and expert pairs live
        // in the separate expert pool.
        let module = match target {
            LoraTarget::Attn(m) => m,
            LoraTarget::Router | LoraTarget::Expert { .. } => bail!(
                "lora rdma: '{}' is a router/expert delta (Feature-1); RDMA \
                 slot-swap stages the attention pool only",
                rec.name
            ),
        };
        let (a_off, b_off) = module_slot_offsets(cfg, max_rank, layer, module)
            .ok_or_else(|| anyhow!("lora rdma: layer {layer} not a full-attention slot layer"))?;
        let (out_dim, in_dim) = module.dims(cfg);
        // 2026-09-25: Check the whole shape before reading r from it: the
        // landing copies into a fixed-size region using these dimensions.
        let rank = match ab {
            AdapterAb::A if rec.shape.len() == 2 && rec.shape[1] == in_dim as u64 => {
                rec.shape[0] as usize
            }
            AdapterAb::B if rec.shape.len() == 2 && rec.shape[0] == out_dim as u64 => {
                rec.shape[1] as usize
            }
            AdapterAb::A => bail!(
                "REJECT[shape-mismatch]: '{}' is {:?}, expected [r, {}]",
                rec.name,
                rec.shape,
                in_dim
            ),
            AdapterAb::B => bail!(
                "REJECT[shape-mismatch]: '{}' is {:?}, expected [{}, r]",
                rec.name,
                rec.shape,
                out_dim
            ),
        };
        if rank == 0 {
            bail!("REJECT[shape-mismatch]: '{}' has zero rank", rec.name);
        }
        if rank > max_rank {
            bail!(
                "lora rdma: adapter rank {rank} for {} exceeds pool max_rank {max_rank}",
                rec.name
            );
        }
        let (kind, off) = match ab {
            AdapterAb::A => (LoraAbKind::A, a_off),
            AdapterAb::B => (LoraAbKind::B, b_off),
        };
        let pair = pairs.entry((layer, module)).or_default();
        let cell = &mut pair[ab as usize];
        if cell.is_some() {
            bail!("REJECT[duplicate-tensor]: two tensors map to layer {layer} {module:?} {ab:?}");
        }
        *cell = Some(rank);
        targets.push(LoraLandTarget {
            tensor_name: rec.name.clone(),
            kind,
            dst: base + off as u64,
            out_dim,
            in_dim,
            rank,
            max_rank,
        });
    }
    if targets.is_empty() {
        bail!("lora rdma: adapter manifest has no lora_A/lora_B tensors");
    }
    for ((layer, module), pair) in pairs {
        let [Some(a_rank), Some(b_rank)] = pair else {
            bail!(
                "REJECT[unpaired-tensor]: layer {layer} {module:?} has only one of lora_A/lora_B"
            );
        };
        if a_rank != b_rank {
            bail!(
                "REJECT[rank-mismatch]: layer {layer} {module:?} has A rank {a_rank}, B rank {b_rank}"
            );
        }
    }
    Ok(targets)
}

/// 2026-09-25: Rebuild a slot's per-layer [`LoraLayerWeights`] after an RDMA
/// landing. A `LoraPair` carries rank and scale, which the new adapter may
/// change, so the pairs are rebuilt. A (layer, module) gets a pair when both an
/// A and a B target land at its offsets. No GPU access. Refused: a rank that
/// differs between A, B and `peft.r`, and a target whose geometry differs from
/// the pool layout.
pub fn rebuild_slot_layers(
    targets: &[LoraLandTarget],
    cfg: &ModelConfig,
    peft: &PeftAdapterConfig,
    pool: DevicePtr,
    slot: usize,
    max_rank: usize,
) -> Result<Vec<Option<LoraLayerWeights>>> {
    let scale = peft.scaling();
    let base = pool.0 + slot_base_offset(slot, cfg, max_rank) as u64;
    let mut layers: Vec<Option<LoraLayerWeights>> =
        (0..cfg.num_hidden_layers).map(|_| None).collect();
    // 2026-09-25: Targets carry a destination address, not a layer, so walk the
    // pool layout (the `module_slot_offsets` walk) and match each (layer,
    // module)'s offsets against the targets' `dst`.
    for rec_layer in 0..cfg.num_hidden_layers {
        let mut lw = LoraLayerWeights::empty(rec_layer);
        let mut any = false;
        for module in LoraModule::ALL {
            if !module.applies_to_layer(cfg, rec_layer) {
                continue;
            }
            let (a_off, b_off) = module_slot_offsets(cfg, max_rank, rec_layer, module)
                .expect("applicable module has a slot offset");
            let a_dst = base + a_off as u64;
            let b_dst = base + b_off as u64;
            let a_t = targets
                .iter()
                .find(|t| t.kind == LoraAbKind::A && t.dst == a_dst);
            let b_t = targets
                .iter()
                .find(|t| t.kind == LoraAbKind::B && t.dst == b_dst);
            if let (Some(a), Some(b)) = (a_t, b_t) {
                let (out_dim, in_dim) = module.dims(cfg);
                if a.rank != b.rank || a.rank != peft.r {
                    bail!(
                        "REJECT[rank-mismatch]: layer {rec_layer} {module:?} has target ranks A={} B={}, config r={}",
                        a.rank,
                        b.rank,
                        peft.r
                    );
                }
                if a.max_rank != max_rank
                    || b.max_rank != max_rank
                    || a.out_dim != out_dim
                    || b.out_dim != out_dim
                    || a.in_dim != in_dim
                    || b.in_dim != in_dim
                {
                    bail!(
                        "REJECT[landing-geometry]: layer {rec_layer} {module:?} target geometry does not match the pool layout"
                    );
                }
                let pair = LoraPair {
                    a: DenseWeight {
                        weight: DevicePtr(a_dst),
                    },
                    b: DenseWeight {
                        weight: DevicePtr(b_dst),
                    },
                    rank: a.rank as u32,
                    k_in: in_dim as u32,
                    n_out: out_dim as u32,
                    scale,
                    max_rank: max_rank as u32,
                };
                match module {
                    LoraModule::QProj => lw.q_proj = Some(pair),
                    LoraModule::KProj => lw.k_proj = Some(pair),
                    LoraModule::VProj => lw.v_proj = Some(pair),
                    LoraModule::OProj => lw.o_proj = Some(pair),
                    LoraModule::GateProj => lw.gate_proj = Some(pair),
                    LoraModule::UpProj => lw.up_proj = Some(pair),
                    LoraModule::DownProj => lw.down_proj = Some(pair),
                    LoraModule::OutProj => lw.out_proj = Some(pair),
                }
                any = true;
            }
        }
        if any {
            layers[rec_layer] = Some(lw);
        }
    }
    Ok(layers)
}

/// 2026-09-25: `pool_slot_bytes`, for model-engine's RDMA swap, which zeroes
/// the slot before landing.
pub fn slot_bytes(cfg: &ModelConfig, max_rank: usize) -> usize {
    pool_slot_bytes(cfg, max_rank)
}

/// 2026-09-25: Fetch a peer-staged adapter's manifest over the `weight_peer`
/// control channel: connect, request, read the manifest, drop the connection.
#[cfg(feature = "cuda")]
pub fn fetch_adapter_manifest(peer_addr: &str, adapter_id: &str) -> Result<WeightManifest> {
    use std::net::TcpStream;

    use anyhow::Context;
    use metrale_storage::weight_peer::{read_weight_manifest, write_model_request};

    let mut stream =
        TcpStream::connect(peer_addr).with_context(|| format!("connect lora peer {peer_addr}"))?;
    stream.set_nodelay(true).ok();
    write_model_request(&mut stream, adapter_id).context("send adapter request")?;
    let manifest = read_weight_manifest(&mut stream).context("read adapter manifest")?;
    let _ = std::io::Write::write_all(&mut stream, &[]);
    Ok(manifest)
}

#[cfg(test)]
#[path = "rdma_stage_tests.rs"]
mod tests;
