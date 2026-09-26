// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: LoRA adapter types: the module and A/B enums, per-layer and
//! per-slot pairs, the loaded pool set [`LoraWeights`], and the view types the
//! victim search reads.
//!
//! Owner: model-layers (lora).
//! Invariants: none beyond the types.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, AtomicUsize};

use metrale_config::PeftAdapterConfig;
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_model_weights::weights::WeightStore;

use super::*;
use crate::layers::ops::lora_delta::LoraPair;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LoraModule {
    QProj,
    KProj,
    VProj,
    OProj,
    GateProj,
    UpProj,
    DownProj,
    /// 2026-09-25: The GDN block's output projection (`linear_attn.out_proj`),
    /// linear-attention value width to hidden. `OProj` instead reads the
    /// attention head width and applies only to full-attention layers.
    OutProj,
}

impl LoraModule {
    pub const ALL: [LoraModule; 8] = [
        Self::QProj,
        Self::KProj,
        Self::VProj,
        Self::OProj,
        Self::GateProj,
        Self::UpProj,
        Self::DownProj,
        Self::OutProj,
    ];

    pub fn is_dense_ffn(&self) -> bool {
        matches!(self, Self::GateProj | Self::UpProj | Self::DownProj)
    }

    pub fn is_gdn_out(&self) -> bool {
        matches!(self, Self::OutProj)
    }

    /// 2026-09-25: Whether `layer` can carry this module: q/k/v/o on
    /// full-attention layers; gate/up/down on every layer of a model with no
    /// routed experts (`num_experts == 0`); out_proj on every other layer.
    /// `pool_slot_bytes`, `module_slot_offsets`, `pack_slot` and the RDMA
    /// landing path all filter by it, so the reserved and the written offsets
    /// agree.
    pub fn applies_to_layer(&self, cfg: &metrale_config::ModelConfig, layer: usize) -> bool {
        use metrale_config::LayerType;
        let full_attn = cfg.layer_type(layer) == LayerType::FullAttention;
        if self.is_dense_ffn() {
            cfg.num_experts == 0
        } else if self.is_gdn_out() {
            !full_attn
        } else {
            full_attn
        }
    }

    /// 2026-09-25: The module's name in a PEFT `target_modules` list.
    pub fn peft_name(&self) -> &'static str {
        match self {
            Self::QProj => "q_proj",
            Self::KProj => "k_proj",
            Self::VProj => "v_proj",
            Self::OProj => "o_proj",
            Self::GateProj => "gate_proj",
            Self::UpProj => "up_proj",
            Self::DownProj => "down_proj",
            Self::OutProj => "out_proj",
        }
    }

    /// 2026-09-25: `(out_dim, in_dim)` of the base projection. The adapter's A
    /// is `[r, in_dim]` and its B is `[out_dim, r]`. On a gated-attention model
    /// (`attn_gated`) q_proj's output includes the gate, so it is twice
    /// `num_attention_heads * head_dim`.
    pub fn dims(&self, cfg: &metrale_config::ModelConfig) -> (usize, usize) {
        let h = cfg.hidden_size;
        match self {
            Self::QProj => (
                (if cfg.attn_gated { 2 } else { 1 }) * cfg.num_attention_heads * cfg.head_dim,
                h,
            ),
            Self::KProj | Self::VProj => (cfg.num_key_value_heads * cfg.head_dim, h),
            Self::OProj => (h, cfg.num_attention_heads * cfg.head_dim),
            Self::GateProj | Self::UpProj => (cfg.intermediate_size, h),
            Self::DownProj => (h, cfg.intermediate_size),
            Self::OutProj => (h, cfg.linear_num_value_heads * cfg.linear_value_head_dim),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AdapterAb {
    A = 0,
    B = 1,
}

/// 2026-09-25: One layer's adapted modules, as [`LoraPair`]s; `None` means the
/// module is not adapted. `Clone` because a rotation copies the new active
/// slot's layers out of the pool set before re-installing them.
#[derive(Clone)]
pub struct LoraLayerWeights {
    pub layer_idx: usize,
    pub q_proj: Option<LoraPair>,
    pub k_proj: Option<LoraPair>,
    pub v_proj: Option<LoraPair>,
    pub o_proj: Option<LoraPair>,
    pub gate_proj: Option<LoraPair>,
    pub up_proj: Option<LoraPair>,
    pub down_proj: Option<LoraPair>,
    pub out_proj: Option<LoraPair>,
    /// 2026-09-25: MoE router (`mlp.gate`) delta. `None` unless the adapter
    /// targets the router; loading such an adapter needs
    /// `METRALE_LORA_EXPERTS=1`.
    pub router: Option<LoraPair>,
    /// 2026-09-25: This layer's routed-expert pairs. `None` unless the adapter
    /// carries expert deltas (which also need `METRALE_LORA_EXPERTS=1`). They
    /// live in `LoraWeights::expert_pool`, outside the equal-size slot pool.
    pub experts: Option<ExpertLoraLayer>,
}

impl LoraLayerWeights {
    /// 2026-09-25: An entry for `layer_idx` with no module adapted.
    pub fn empty(layer_idx: usize) -> Self {
        Self {
            layer_idx,
            q_proj: None,
            k_proj: None,
            v_proj: None,
            o_proj: None,
            gate_proj: None,
            up_proj: None,
            down_proj: None,
            out_proj: None,
            router: None,
            experts: None,
        }
    }
}

/// 2026-09-25: One pool slot: the resident adapter's name and config, and its
/// pairs, which point into this slot's region of the shared pool. `layers` is
/// indexed by global layer and has `num_hidden_layers` entries. An empty
/// `name` marks a placeholder that holds no adapter.
#[derive(Clone)]
pub struct AdapterSlot {
    pub name: String,
    pub adapter_config: PeftAdapterConfig,
    pub layers: Vec<Option<LoraLayerWeights>>,
    /// 2026-09-25: Incremented (wrapping) each time a disk or peer swap
    /// replaces this slot's contents, and folded into [`adapter_id_hash`], so
    /// new weights under the same name get a new cache identity and miss the
    /// old prefix cache entries. It is 0 at load, where the fold is a no-op. A
    /// rotation does not change it.
    pub generation: u64,
}

/// 2026-09-25: One adapter to pack. `store` holds its tensors on the device as
/// BF16: `metrale_model_weights::weights::adapter::load_adapter_safetensors`
/// converts F16 and F32 on the host.
pub struct LoraAdapterInput<'a> {
    pub name: String,
    pub store: &'a WeightStore,
    pub peft: PeftAdapterConfig,
}

/// 2026-09-25: The loaded adapter set: one fixed-address pool of `max_loras`
/// equal-size slots padded to `max_rank`, one [`AdapterSlot`] per pool slot,
/// and per-(layer, module) `[max_loras]` device pointer tables.
pub struct LoraWeights {
    /// 2026-09-25: Name of the active adapter, for logs and status.
    pub name: String,
    /// 2026-09-25: Config of the adapter made active at load or by the last
    /// rotation, for logs. A swap into the active slot updates `name` but not
    /// this.
    pub adapter_config: PeftAdapterConfig,
    pub max_rank: usize,
    pub max_loras: usize,
    /// 2026-09-25: One allocation holding every slot's padded A and B.
    pub pool: DevicePtr,
    pub pool_bytes: usize,
    /// 2026-09-25: A separate allocation for the router and routed-expert
    /// padded A and B, sized from the audited keys of the startup adapters.
    /// `None` when no adapter targets experts or the router.
    pub expert_pool: Option<DevicePtr>,
    pub expert_pool_bytes: usize,
    /// 2026-09-25: Slot-indexed, `len() == max_loras`: the startup adapters,
    /// then placeholders. Slot `k`'s pairs live from pool byte offset
    /// `k * pool_slot_bytes`.
    pub slots: Vec<AdapterSlot>,
    /// 2026-09-25: Index into `slots` of the active adapter; 0 at load.
    pub active: usize,
    /// 2026-09-25: `(global layer, module)` to `(a_table, b_table)`. Each table
    /// is a device `[max_loras]` u64 array; 0 means that slot does not adapt
    /// the module.
    pub tables: BTreeMap<(usize, LoraModule), (DevicePtr, DevicePtr)>,
    /// 2026-09-25: Device `[max_loras]` f32 table: entry `k` is slot `k`'s
    /// `PeftAdapterConfig::scaling()` (alpha/r, or alpha/sqrt(r) with rsLoRA),
    /// the same scale its [`LoraPair`]s carry, and 0.0 for an unpacked slot.
    /// Its address is fixed at load; a swap rewrites one entry.
    pub scale_table: DevicePtr,
    /// 2026-09-25: In-flight sequence count per pool index,
    /// `len() == max_loras`. A sequence takes a count through
    /// [`Self::acquire_slot`] and returns it when it is freed. A swap into a
    /// slot whose count is above 0, or a rotation away from one, is refused.
    /// It is a parallel Vec because [`AdapterSlot`] derives `Clone` and
    /// [`AtomicUsize`] is not `Clone`.
    pub ref_counts: Vec<AtomicUsize>,
    /// 2026-09-25: Slots `[0, pinned)` hold the startup adapters and are never
    /// eviction candidates. Slots `[pinned, max_loras)` are the promotion
    /// cache, placeholders at load.
    pub pinned: usize,
    /// 2026-09-25: Last-used tick per pool index, `len() == max_loras`, set by
    /// [`Self::acquire_slot`] on the resolved index and by
    /// [`Self::touch_slot`].
    pub last_used: Vec<AtomicU64>,
    /// 2026-09-25: Source of the `last_used` ticks; incremented by each
    /// acquire and touch.
    pub lru_tick: AtomicU64,
    /// 2026-09-25: Raw token-overlay uploads, one per startup adapter (`None`
    /// when it ships no overlay tensors). The model's `set_lora_weights` takes
    /// them and builds the overlay tables, which need the served embed and
    /// lm_head tables.
    pub overlay_raw: Vec<Option<super::overlay_build::OverlayRawSlot>>,
}

/// 2026-09-25: A cache slot's state as `select_victim_slot` reads it, built by
/// [`LoraWeights::cache_slot_views`].
#[derive(Clone, Copy, Debug)]
pub struct SlotView {
    /// 2026-09-25: The slot holds an adapter (its name is not empty).
    pub filled: bool,
    /// 2026-09-25: In-flight sequences; 0 means evictable.
    pub ref_count: usize,
    /// 2026-09-25: Larger means more recently used.
    pub last_used: u64,
}

/// 2026-09-25: Why a promotion found no victim slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VictimError {
    /// 2026-09-25: Every cache slot has in-flight sequences. The caller may
    /// retry; a busy slot is never chosen.
    PoolFull,
}

impl LoraLayerWeights {
    /// 2026-09-25: This layer's pair for `module`; `None` when the module is
    /// not adapted. [`select_routed_pair`] reads pairs through it.
    pub fn module_pair(&self, module: LoraModule) -> Option<&LoraPair> {
        match module {
            LoraModule::QProj => self.q_proj.as_ref(),
            LoraModule::KProj => self.k_proj.as_ref(),
            LoraModule::VProj => self.v_proj.as_ref(),
            LoraModule::OProj => self.o_proj.as_ref(),
            LoraModule::GateProj => self.gate_proj.as_ref(),
            LoraModule::UpProj => self.up_proj.as_ref(),
            LoraModule::DownProj => self.down_proj.as_ref(),
            LoraModule::OutProj => self.out_proj.as_ref(),
        }
    }
}

impl LoraWeights {
    /// 2026-09-25: The active slot's pairs, indexed by global layer.
    pub fn active_layers(&self) -> &[Option<LoraLayerWeights>] {
        &self.slots[self.active].layers
    }

    /// 2026-09-25: Resolve a request's `adapter_slot` (`>= 0` that slot, `-1`
    /// the active one) and return it only when it is an in-range slot other
    /// than the active one; `None` otherwise. See [`routed_prefill_slot_of`].
    pub fn routed_prefill_slot(&self, adapter_slot: i32) -> Option<usize> {
        routed_prefill_slot_of(adapter_slot, self.active, self.slots.len())
    }

    /// 2026-09-25: The slot holding adapter `name`; placeholders never match.
    pub fn slot_of(&self, name: &str) -> Option<usize> {
        self.slots
            .iter()
            .position(|s| !s.name.is_empty() && s.name == name)
    }

    /// 2026-09-25: The names of all resident adapters, in slot order.
    pub fn adapter_names(&self) -> Vec<String> {
        self.slots
            .iter()
            .filter(|s| !s.name.is_empty())
            .map(|s| s.name.clone())
            .collect()
    }

    /// 2026-09-25: Cache identity for a request's `adapter_slot` (`>= 0` that
    /// slot, `-1` the active one): [`adapter_id_hash`] of the resolved slot's
    /// name and generation, or the base sentinel 0 when the slot is out of
    /// range or a placeholder.
    pub fn adapter_id_for_slot(&self, slot: i32) -> u64 {
        let resolved = if slot >= 0 {
            slot as usize
        } else {
            self.active
        };
        match self.slots.get(resolved) {
            Some(s) if !s.name.is_empty() => adapter_id_hash(&s.name, s.generation),
            Some(_) | None => 0,
        }
    }
}

#[path = "types_slots.rs"]
mod types_slots;

#[cfg(test)]
#[path = "types_tests.rs"]
mod tests;
