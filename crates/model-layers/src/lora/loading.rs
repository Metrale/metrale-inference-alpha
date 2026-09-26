// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: LoRA adapter loading: the model-family check, the pack of
//! audited adapters into the rank-padded slot pool and the expert pool, the
//! pointer and scale tables, and the in-place disk swap of one slot.
//!
//! Owner: model-layers (lora).
//! Invariants:
//! - Every pool byte an adapter does not write is zero: both pools are zeroed
//!   after allocation, and `pack_store_into_slot` zeroes the slot before it
//!   packs.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, AtomicUsize};

use anyhow::{Result, bail};
use metrale_config::{ModelConfig, PeftAdapterConfig};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_weights::weights::WeightStore;

use super::*;
use crate::layers::ops::lora_delta::LoraPair;
use crate::weight_map::DenseWeight;

/// 2026-09-25: Refuses every model family except dense `qwen3_5`
/// (`is_qwen35_dense`), `holo3_1_moe` and `qwen3_6_moe`. It checks the
/// `model_type` after config dispatch, which can rewrite it.
fn check_family(cfg: &ModelConfig) -> Result<()> {
    if !(cfg.is_qwen35_dense()
        || cfg.model_type == "holo3_1_moe"
        || cfg.model_type == "qwen3_6_moe")
    {
        bail!(
            "REJECT[unvalidated-family]: LoRA v0 is validated on qwen3_5 dense \
             (holo-3.1-0.8b), holo3_1_moe (holo-3.1-35b-a3b), and qwen3_6_moe \
             (Qwen3.6-35B-A3B) only; model_type='{}', num_experts={}",
            cfg.model_type,
            cfg.num_experts
        );
    }
    Ok(())
}

/// 2026-09-25: Pack one audited adapter into pool `slot`, the byte region at
/// `slot * pool_slot_bytes`. The walk within a slot is the same for every
/// slot: layers ascending, then [`LoraModule::ALL`] filtered by
/// `applies_to_layer`, A then B. A is copied as `[r, in]` into the head of its
/// `[max_rank, in]` region; B's rows are re-strided from r to max_rank.
/// Returns the slot's pairs indexed by global layer and, per applicable
/// (layer, module), the (a_ptr, b_ptr) as raw u64, (0, 0) where the adapter
/// has no pair, for the pointer tables.
#[allow(clippy::type_complexity)]
fn pack_slot(
    slot: usize,
    name: &str,
    adapter_store: &WeightStore,
    peft: &PeftAdapterConfig,
    found: &BTreeMap<(usize, LoraModule), [Option<String>; 2]>,
    cfg: &ModelConfig,
    gpu: &dyn GpuBackend,
    pool: DevicePtr,
    max_lora_rank: usize,
) -> Result<(
    Vec<Option<LoraLayerWeights>>,
    BTreeMap<(usize, LoraModule), (u64, u64)>,
)> {
    let scale = peft.scaling();
    let slot_bytes = pool_slot_bytes(cfg, max_lora_rank);
    let mut layers: Vec<Option<LoraLayerWeights>> =
        (0..cfg.num_hidden_layers).map(|_| None).collect();
    let mut slot_ptrs: BTreeMap<(usize, LoraModule), (u64, u64)> = BTreeMap::new();
    let mut off = slot * slot_bytes;
    // 2026-09-25: The same order and predicate as `pool_slot_bytes` and
    // `module_slot_offsets`, so the offsets stay inside the slot and match the
    // RDMA landing offsets.
    for layer_idx in 0..cfg.num_hidden_layers {
        let mut lw = LoraLayerWeights::empty(layer_idx);
        let mut any = false;
        for module in LoraModule::ALL {
            if !module.applies_to_layer(cfg, layer_idx) {
                continue;
            }
            let (out_dim, in_dim) = module.dims(cfg);
            let a_off = off;
            let b_off = off + max_lora_rank * in_dim * BF16_BYTES;
            off = b_off + out_dim * max_lora_rank * BF16_BYTES;
            let a_ptr = DevicePtr(pool.0 + a_off as u64);
            let b_ptr = DevicePtr(pool.0 + b_off as u64);

            let mut this = (0u64, 0u64);
            if let Some([Some(a_key), Some(b_key)]) = found.get(&(layer_idx, module)) {
                let a_t = adapter_store.get(a_key)?;
                let mut a_host = vec![0u8; peft.r * in_dim * BF16_BYTES];
                gpu.copy_d2h(a_t.ptr, &mut a_host)?;
                gpu.copy_h2d(&a_host, a_ptr)?;
                let b_t = adapter_store.get(b_key)?;
                let mut b_src = vec![0u8; out_dim * peft.r * BF16_BYTES];
                gpu.copy_d2h(b_t.ptr, &mut b_src)?;
                let mut b_host = vec![0u8; out_dim * max_lora_rank * BF16_BYTES];
                for row in 0..out_dim {
                    let d = row * max_lora_rank * BF16_BYTES;
                    let s = row * peft.r * BF16_BYTES;
                    b_host[d..d + peft.r * BF16_BYTES]
                        .copy_from_slice(&b_src[s..s + peft.r * BF16_BYTES]);
                }
                gpu.copy_h2d(&b_host, b_ptr)?;

                let pair = LoraPair {
                    a: DenseWeight { weight: a_ptr },
                    b: DenseWeight { weight: b_ptr },
                    rank: peft.r as u32,
                    k_in: in_dim as u32,
                    n_out: out_dim as u32,
                    scale,
                    max_rank: max_lora_rank as u32,
                };
                tracing::info!(
                    "LoRA: slot {slot} '{name}' layer {layer_idx} {module:?} r={} \
                     scale={:.6} A=[{},{}] B=[{},{}] (padded to max_rank={})",
                    peft.r,
                    scale,
                    peft.r,
                    in_dim,
                    out_dim,
                    peft.r,
                    max_lora_rank
                );
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
                this = (a_ptr.0, b_ptr.0);
                any = true;
            }
            slot_ptrs.insert((layer_idx, module), this);
        }
        if any {
            layers[layer_idx] = Some(lw);
        }
    }
    debug_assert_eq!(off, (slot + 1) * slot_bytes);
    Ok((layers, slot_ptrs))
}

/// 2026-09-25: Load startup adapters: audit each one, check free memory, pack
/// adapter k into slot k, then build the per-module `[max_loras]` pointer
/// tables (0 where a slot has no pair) and the `[max_loras]` scale table.
/// Slots past the adapter count are empty placeholders.
///
/// Called through `ModelWeightLoader::load_lora_adapters` from model-engine
/// `factory::build_model` before `BufferArena::new` and before the free-memory
/// read that sizes the KV cache, so the pools come out of the KV budget.
pub fn load_lora_adapters_multi(
    adapters: &[LoraAdapterInput<'_>],
    cfg: &ModelConfig,
    gpu: &dyn GpuBackend,
    max_loras: usize,
    max_lora_rank: usize,
) -> Result<LoraWeights> {
    check_family(cfg)?;
    if adapters.is_empty() {
        bail!("REJECT[no-adapters]: load_lora_adapters_multi called with an empty set");
    }
    if adapters.len() > max_loras {
        bail!(
            "REJECT[too-many-adapters]: {} --lora-adapter given but --max-loras={} \
             (pool has {} slots); raise --max-loras or stage the extras on an \
             $METRALE_LORA_PEER for on-demand RDMA swap",
            adapters.len(),
            max_loras,
            max_loras
        );
    }

    // 2026-09-25: Audit every adapter before any allocation.
    let mut audited: Vec<AuditedAdapter> = Vec::with_capacity(adapters.len());
    for a in adapters {
        audited.push(audit_adapter(a.store, &a.peft, cfg, max_lora_rank)?);
    }

    // 2026-09-25: The expert/router pool is sized from each adapter's audited
    // keys at `max_lora_expert_rank`, summed over adapters.
    let expert_rank = max_lora_expert_rank();
    let expert_total: usize = adapters
        .iter()
        .zip(&audited)
        .map(|(_, au)| {
            let (ek, rl) = expert_pack::key_lists(&au.router, &au.experts);
            expert_router_bytes(cfg, &ek, &rl, expert_rank)
        })
        .sum();

    // 2026-09-25: Refuse unless free memory is at least twice both pools. Then
    // one allocation for all slots, zeroed, so padding rows and columns and
    // unused slots contribute nothing to the padded-rank contraction.
    let pool_bytes = pool_slot_bytes(cfg, max_lora_rank) * max_loras;
    let free = gpu.free_memory()?;
    if (pool_bytes + expert_total) * 2 > free {
        bail!(
            "OOM pre-flight (LoRA pool): {:.1} MiB attn pool ({} slots) + {:.1} MiB \
             expert/router pool would leave < 1× headroom of {:.1} MiB free; every \
             pool byte comes directly out of the KV-cache budget on GB10 unified memory",
            pool_bytes as f64 / (1024.0 * 1024.0),
            max_loras,
            expert_total as f64 / (1024.0 * 1024.0),
            free as f64 / (1024.0 * 1024.0),
        );
    }
    let pool = gpu.alloc(pool_bytes)?;
    gpu.memset(pool, 0, pool_bytes)?;
    // 2026-09-25: One zeroed expert/router pool for all adapters, allocated
    // only when some adapter has router or expert pairs.
    let expert_pool = if expert_total > 0 {
        let ep = gpu.alloc(expert_total)?;
        gpu.memset(ep, 0, expert_total)?;
        Some(ep)
    } else {
        None
    };
    let mut expert_off = 0usize;

    let mut slots: Vec<AdapterSlot> = Vec::with_capacity(adapters.len());
    let mut a_tabs: BTreeMap<(usize, LoraModule), Vec<u64>> = BTreeMap::new();
    let mut b_tabs: BTreeMap<(usize, LoraModule), Vec<u64>> = BTreeMap::new();
    // 2026-09-25: Upload each adapter's raw token-overlay tensors, `None` when
    // it ships none. The model builds the overlay tables from them later, in
    // `set_lora_weights`.
    let mut overlay_raw: Vec<Option<OverlayRawSlot>> = Vec::with_capacity(adapters.len());
    for (k, a) in adapters.iter().enumerate() {
        overlay_raw.push(stage_overlay_raw(
            a.store,
            &audited[k].overlay,
            &a.peft,
            cfg.hidden_size,
            gpu,
        )?);
        let (mut layers, slot_ptrs) = pack_slot(
            k,
            &a.name,
            a.store,
            &a.peft,
            &audited[k].attn,
            cfg,
            gpu,
            pool,
            max_lora_rank,
        )?;
        if let Some(ep) = expert_pool {
            let packed = expert_pack::pack_into(
                &mut layers,
                a.store,
                &a.peft,
                &audited[k].router,
                &audited[k].experts,
                cfg,
                gpu,
                ep,
                expert_rank,
                &mut expert_off,
            )?;
            if packed > 0 {
                tracing::info!(
                    "LoRA: slot {k} '{}' packed {packed} router/expert pair(s) \
                     (expert_rank={expert_rank})",
                    a.name
                );
            }
        }
        for ((layer, module), (a_ptr, b_ptr)) in slot_ptrs {
            a_tabs
                .entry((layer, module))
                .or_insert_with(|| vec![0u64; max_loras])[k] = a_ptr;
            b_tabs
                .entry((layer, module))
                .or_insert_with(|| vec![0u64; max_loras])[k] = b_ptr;
        }
        slots.push(AdapterSlot {
            name: a.name.clone(),
            adapter_config: a.peft.clone(),
            layers,
            generation: 0,
        });
    }

    // 2026-09-25: Slots `[pinned, max_loras)` are the promotion cache. They get
    // empty-named placeholders so a swap can write any slot index
    // (`swap_lora_slot_from_peer` refuses a slot that is not in `slots`). A
    // placeholder's pool region is zero, its pointer-table and scale entries
    // are 0, and its empty name maps to the base id. `pinned >= 1`: an empty
    // adapter set was refused above.
    let pinned = slots.len();
    let num_layers = cfg.num_hidden_layers;
    while slots.len() < max_loras {
        slots.push(AdapterSlot {
            name: String::new(),
            adapter_config: PeftAdapterConfig {
                r: 1,
                lora_alpha: 0.0,
                target_modules: Vec::new(),
                target_modules_pattern: None,
                use_rslora: false,
                layers_to_transform: None,
                trainable_token_indices: Vec::new(),
                modules_to_save: Vec::new(),
                lora_embedding: false,
            },
            layers: vec![None; num_layers],
            generation: 0,
        });
    }

    // 2026-09-25: The per-module `[max_loras]` u64 pointer tables. The model
    // builds each layer's bgmv route from them (model-engine
    // `impl_lora_rotate.rs`, `install_lora_layers`).
    let mk = |tab: &[u64]| -> Result<DevicePtr> {
        let bytes: Vec<u8> = tab.iter().flat_map(|p| p.to_le_bytes()).collect();
        let d = gpu.alloc(bytes.len())?;
        gpu.copy_h2d(&bytes, d)?;
        Ok(d)
    };
    let mut tables = BTreeMap::new();
    for (key, a_tab) in &a_tabs {
        let b_tab = &b_tabs[key];
        tables.insert(*key, (mk(a_tab)?, mk(b_tab)?));
    }

    // 2026-09-25: The `[max_loras]` f32 scale table, 0.0 for unused slots;
    // `lora_bgmv.cu` reads `scale_table[s]` for the row's slot.
    debug_assert_eq!(expert_off, expert_total, "expert pool filled exactly");
    let scale_vals = scale_table_values(adapters, max_loras);
    let scale_bytes: Vec<u8> = scale_vals.iter().flat_map(|s| s.to_le_bytes()).collect();
    let scale_table = gpu.alloc(scale_bytes.len())?;
    gpu.copy_h2d(&scale_bytes, scale_table)?;

    Ok(LoraWeights {
        name: slots[0].name.clone(),
        adapter_config: slots[0].adapter_config.clone(),
        max_rank: max_lora_rank,
        max_loras,
        pool,
        pool_bytes,
        expert_pool,
        expert_pool_bytes: expert_total,
        slots,
        active: 0,
        tables,
        scale_table,
        // 2026-09-25: One counter per pool index, `max_loras` of them.
        ref_counts: (0..max_loras).map(|_| AtomicUsize::new(0)).collect(),
        pinned,
        last_used: (0..max_loras).map(|_| AtomicU64::new(0)).collect(),
        lru_tick: AtomicU64::new(0),
        overlay_raw,
    })
}

/// 2026-09-25: Disk swap: audit `store` and pack it into existing pool `slot`
/// of `lw` in place, with the same `audit_adapter` and `pack_slot` as the
/// startup load. The slot region is zeroed first, since it still holds the
/// previous adapter. Then the slot's name, config and pairs are replaced, its
/// pointer and scale table entries refreshed, and its generation bumped.
/// Returns the new pairs so the caller can re-install them when the slot is
/// active.
///
/// Refused, before anything is written: `slot >= max_loras`, a slot with
/// in-flight sequences, and adapters with router, expert or token-overlay
/// tensors.
pub fn pack_store_into_slot(
    lw: &mut LoraWeights,
    slot: usize,
    name: &str,
    store: &WeightStore,
    peft: &PeftAdapterConfig,
    cfg: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<Vec<Option<LoraLayerWeights>>> {
    if slot >= lw.max_loras {
        bail!(
            "LoRA disk swap: slot {slot} >= max_loras {} (pool has {} slots)",
            lw.max_loras,
            lw.max_loras
        );
    }
    let busy = lw.slot_ref_count(slot);
    if busy > 0 {
        bail!(
            "LoRA disk swap REFUSED: slot {slot} has {busy} in-flight sequence(s) \
             (ref_count>0); cannot replace an adapter mid-decode"
        );
    }
    validate_peft_config(peft, lw.max_rank)?;
    let audited = audit_adapter(store, peft, cfg, lw.max_rank)?;
    if expert_pack::present(&audited.router, &audited.experts) {
        bail!(
            "LoRA disk swap REFUSED: adapter '{name}' carries router/expert deltas \
             (Feature-1); runtime slot-swap of the expert pool is a phase-2 followup"
        );
    }
    if !audited.overlay.is_empty() {
        bail!(
            "LoRA disk swap REFUSED: adapter '{name}' ships token-overlay tensors \
             (Feature-2); runtime slot-swap of the overlay tables is a phase-2 \
             followup (would silently drop the overlay otherwise)"
        );
    }
    let found = audited.attn;
    let slot_bytes = pool_slot_bytes(cfg, lw.max_rank);
    gpu.memset(
        DevicePtr(lw.pool.0 + (slot * slot_bytes) as u64),
        0,
        slot_bytes,
    )?;
    let (layers, _slot_ptrs) = pack_slot(
        slot,
        name,
        store,
        peft,
        &found,
        cfg,
        gpu,
        lw.pool,
        lw.max_rank,
    )?;
    lw.slots[slot].name = name.to_string();
    lw.slots[slot].adapter_config = peft.clone();
    lw.slots[slot].layers = layers.clone();
    lw.refresh_slot_tables(slot, &layers, peft.scaling(), gpu)?;
    // 2026-09-25: A new generation gives the slot a new adapter id, so a
    // same-name request cannot hit prefix-cache entries of the old contents.
    lw.slots[slot].generation = lw.slots[slot].generation.wrapping_add(1);
    Ok(layers)
}

/// 2026-09-25: `load_lora_adapters_multi` with one adapter, whose slot name is
/// empty.
pub fn load_lora_adapters_generic(
    adapter_store: &WeightStore,
    peft: &PeftAdapterConfig,
    cfg: &ModelConfig,
    gpu: &dyn GpuBackend,
    max_loras: usize,
    max_lora_rank: usize,
) -> Result<LoraWeights> {
    let inputs = [LoraAdapterInput {
        name: String::new(),
        store: adapter_store,
        peft: peft.clone(),
    }];
    load_lora_adapters_multi(&inputs, cfg, gpu, max_loras, max_lora_rank)
}
