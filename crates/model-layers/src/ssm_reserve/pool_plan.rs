// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: The SSM state pool as one plan, which both the pre-load reserve (the server's
//! `preflight_reserve`) and the allocation (`SsmStatePool::new`) read, so the two cannot
//! differ (M5, LIFECYCLE-DESIGN.md section 3.3: "preflight == allocation").
//!
//! - [`pool_counts`]: how many units the pool holds: every slot plus the padding dummy, and
//!   with speculation the verify slots (plus their dummy) with their h and conv intermediates
//!   and one checkpoint each.
//! - [`PoolPlan::new`]: the bytes, from `metrale_circuit::state::StatePlan` over the recurrent
//!   states the model's circuit declares (`[[block.gdn.state]]`, `[[block.mamba.state]]`),
//!   evaluated under this process's (tensor-parallel local) config.
//!
//! Owner: model-layers (SSM reserve).
//! Invariants:
//! - A model whose `model_type` has a circuit is sized from the circuit's declarations only.
//!   A model without one is sized from `ModelConfig::ssm_h_state_bytes` /
//!   `ssm_conv_state_bytes`: the transitional source, removed at M3 when every family has a
//!   circuit. One source per family, never both.
//! - The dummy slots are counted: the allocator allocates them, so the reserve does too.

use std::collections::BTreeMap;

use anyhow::{Context, Result, bail, ensure};
use metrale_circuit::state::{
    Holding, StateDecl, StateDtype, StateFormat, StateInputs, StateKind, StatePlan, VerifyInputs,
    VerifySteps,
};
use metrale_config::ModelConfig;

use super::{SsmRollbackMode, mtp_state_slots, verify_slot_h_intermediates};

/// 2026-09-30: What decides the pool's unit counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolShape {
    /// 2026-09-30: Claimable slots (`--max-batch-size`).
    pub max_slots: usize,
    /// 2026-09-30: The verify pools exist (MTP, DFlash or self-speculation).
    pub spec: bool,
    /// 2026-09-30: Rows of the widest verify (`K`, the drafts plus one).
    pub num_intermediates: usize,
    /// 2026-09-30: Drafts per verify.
    pub num_drafts: usize,
    /// 2026-09-30: Every verify slot holds `K - 1` h intermediates (DFlash, or a K that is not
    /// the MTP `num_drafts + 1`); otherwise the MTP ladder tiers them.
    pub uniform_h: bool,
    /// 2026-09-30: `--ssm-rollback-mode`.
    pub rollback: SsmRollbackMode,
}

/// 2026-09-30: The verify pools' unit counts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifyCounts {
    /// 2026-09-30: Per verify slot, the dummy last: h intermediates.
    pub h_steps: Vec<usize>,
    /// 2026-09-30: Conv intermediates per verify slot (0 under replay).
    pub conv_steps: usize,
}

impl VerifyCounts {
    /// 2026-09-30: Verify slots, the dummy included.
    pub fn slots(&self) -> usize {
        self.h_steps.len()
    }
}

/// 2026-09-30: The pool's unit counts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PoolCounts {
    /// 2026-09-30: Live slots, the padding dummy (`max_slots`) included.
    pub slots: usize,
    /// 2026-09-30: The verify pools; `None` without speculation.
    pub verify: Option<VerifyCounts>,
}

/// 2026-09-30: [`pool_counts_with`] with the verify slots `mtp_state_slots` and the per-slot
/// h intermediates `verify_slot_h_intermediates` (the MTP ladder).
///
/// 2026-10-02: Under a published prompt-lookup copy tier (`copy_tier`) the copy slots hold
/// the copy depth and the verify rows widen to it ([`pool_counts_tiered`]).
pub fn pool_counts(shape: &PoolShape) -> PoolCounts {
    pool_counts_tiered(shape, super::copy_tier())
}

/// 2026-10-02: [`pool_counts`] with the copy tier passed in, so tests need not publish it.
/// The tier widens `num_intermediates` (every slot's conv intermediates and the dummy's h)
/// and raises the copy slots' h; a uniform-h pool takes the widened depth everywhere.
pub fn pool_counts_tiered(shape: &PoolShape, tier: Option<super::CopyTier>) -> PoolCounts {
    let widened = PoolShape {
        num_intermediates: super::tier_rows(shape.num_intermediates, tier),
        ..*shape
    };
    pool_counts_with(&widened, mtp_state_slots(shape.max_slots), |s| {
        super::tier_h(
            verify_slot_h_intermediates(s, shape.num_drafts, false),
            s,
            tier,
        )
    })
}

/// 2026-09-30: Pure: the counts for `mtp_slots` verify slots, slot `s` tiered to `slot_h(s)`
/// h intermediates (capped at `K - 1`). The verify dummy, at index `mtp_slots`, always gets
/// `K - 1`: batched-verify padding rows may write any live token index. Replay keeps no
/// intermediates, only the checkpoints.
pub fn pool_counts_with(
    shape: &PoolShape,
    mtp_slots: usize,
    slot_h: impl Fn(usize) -> usize,
) -> PoolCounts {
    let full = shape.num_intermediates.saturating_sub(1);
    let verify = shape.spec.then(|| {
        let replay = shape.rollback == SsmRollbackMode::Replay;
        VerifyCounts {
            h_steps: (0..=mtp_slots)
                .map(|s| match (replay, s == mtp_slots || shape.uniform_h) {
                    (true, _) => 0,
                    (false, true) => full,
                    (false, false) => slot_h(s).min(full),
                })
                .collect(),
            conv_steps: if replay { 0 } else { shape.num_intermediates },
        }
    });
    PoolCounts {
        slots: shape.max_slots + 1,
        verify,
    }
}

/// 2026-09-30: Where a pool's per-unit sizes come from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnitSource {
    /// 2026-09-30: The recurrent states the model's circuit declares.
    Circuit,
    /// 2026-09-30: `ModelConfig::ssm_h_state_bytes` / `ssm_conv_state_bytes`, for a
    /// `model_type` no circuit serves. Transitional: removed at M3.
    Transitional,
}

/// 2026-09-30: The dims the circuits' recurrent state shapes read, from this process's config
/// (after tensor-parallel sharding, `topology.rs`).
pub fn state_dims(c: &ModelConfig) -> BTreeMap<String, u64> {
    [
        ("lin_k_heads", c.linear_num_key_heads),
        ("lin_k_dim", c.linear_key_head_dim),
        ("lin_v_heads", c.linear_num_value_heads),
        ("lin_v_dim", c.linear_value_head_dim),
        ("mamba_heads", c.mamba_num_heads),
        ("mamba_head_dim", c.mamba_head_dim),
        ("ssm_state", c.ssm_state_size),
        ("ssm_groups", c.n_groups),
        ("conv_kernel", c.linear_conv_kernel_dim),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v as u64))
    .collect()
}

/// 2026-09-30: One recurrent layer's h and conv declarations, and where they came from.
pub fn recurrent_units(c: &ModelConfig) -> Result<(Vec<StateDecl>, UnitSource)> {
    if let Some(decls) = metrale_circuit::recurrent_states(&c.model_type, &state_dims(c))
        .with_context(|| format!("the circuit of `{}` declares no usable state", c.model_type))?
    {
        ensure!(
            !decls.is_empty() || c.num_ssm_layers() == 0,
            "`{}` has {} recurrent layers and its circuit declares no recurrent state",
            c.model_type,
            c.num_ssm_layers()
        );
        return Ok((decls, UnitSource::Circuit));
    }
    let decl = |local: &str, bytes: usize, format, verify| StateDecl {
        id: format!("transitional.{local}"),
        local: local.to_string(),
        block: "transitional".to_string(),
        layer: None,
        section: metrale_circuit::Section::Main,
        kind: StateKind::Recurrent,
        format,
        elements: (bytes / 4) as u64,
        verify: Some(verify),
        lifetime: metrale_circuit::state::Lifetime::Sequence,
    };
    Ok((
        vec![
            decl(
                "h",
                c.ssm_h_state_bytes(),
                StateFormat::Keyed("ssm_h_storage".into()),
                VerifySteps::H,
            ),
            decl(
                "conv",
                c.ssm_conv_state_bytes(),
                StateFormat::Fixed(StateDtype::F32),
                VerifySteps::Conv,
            ),
        ],
        UnitSource::Transitional,
    ))
}

/// 2026-09-30: The SSM pool of one model, sized.
#[derive(Debug, Clone)]
pub struct PoolPlan {
    /// 2026-09-30: Where the unit sizes came from.
    pub source: UnitSource,
    /// 2026-09-30: Recurrent layers.
    pub layers: usize,
    /// 2026-09-30: One h unit at FP32 width (the prefill staging blob).
    pub h_f32_unit: usize,
    /// 2026-09-30: One h unit as stored (half under `--ssm-h-dtype f16-pool`).
    pub h_stored_unit: usize,
    /// 2026-09-30: One conv unit.
    pub conv_unit: usize,
    /// 2026-09-30: The counts the plan was made for.
    pub counts: PoolCounts,
    layer: StatePlan,
    h_id: String,
    conv_id: String,
}

/// 2026-09-30: A recurrent state of the pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolState {
    H,
    Conv,
}

impl PoolPlan {
    /// 2026-09-30: The pool of `config` holding `counts`.
    pub fn new(config: &ModelConfig, counts: &PoolCounts, h_f16_pool: bool) -> Result<Self> {
        let (decls, source) = recurrent_units(config)?;
        let find = |v: VerifySteps| {
            decls
                .iter()
                .find(|d| d.verify == Some(v))
                .with_context(|| format!("no recurrent state keeps {v:?} intermediates"))
        };
        let layers = config.num_ssm_layers();
        if layers == 0 {
            return Ok(Self::empty(source, counts));
        }
        let (h, conv) = (find(VerifySteps::H)?, find(VerifySteps::Conv)?);
        let storage = if h_f16_pool {
            StateDtype::F16
        } else {
            StateDtype::F32
        };
        let inputs = StateInputs {
            formats: BTreeMap::from([("ssm_h_storage".to_string(), storage)]),
            slots: counts.slots as u64,
            verify: counts.verify.as_ref().map(|v| VerifyInputs {
                h_steps: v.h_steps.iter().map(|&n| n as u64).collect(),
                conv_steps: v.conv_steps as u64,
            }),
            kv: None,
            draft_kv: None,
        };
        let layer = StatePlan::new(&decls, &inputs)?;
        let unit = |d: &StateDecl, dtype: StateDtype| (d.elements * dtype.size()) as usize;
        let conv_dtype = match conv.format {
            StateFormat::Fixed(d) => d,
            StateFormat::Keyed(_) => bail!("the conv state's format is not fixed"),
        };
        Ok(Self {
            source,
            layers,
            h_f32_unit: unit(h, StateDtype::F32),
            h_stored_unit: unit(h, storage),
            conv_unit: unit(conv, conv_dtype),
            counts: counts.clone(),
            layer,
            h_id: h.id.clone(),
            conv_id: conv.id.clone(),
        })
    }

    fn empty(source: UnitSource, counts: &PoolCounts) -> Self {
        Self {
            source,
            layers: 0,
            h_f32_unit: 0,
            h_stored_unit: 0,
            conv_unit: 0,
            counts: counts.clone(),
            layer: StatePlan { terms: Vec::new() },
            h_id: String::new(),
            conv_id: String::new(),
        }
    }

    /// 2026-09-30: Bytes one layer holds of `state` for `holding`.
    pub fn layer_bytes(&self, state: PoolState, holding: Holding) -> usize {
        let id = match state {
            PoolState::H => &self.h_id,
            PoolState::Conv => &self.conv_id,
        };
        self.layer
            .bytes_where(|t| &t.state == id && t.holding == holding) as usize
    }

    /// 2026-09-30: Bytes of the whole pool: every layer.
    pub fn total(&self) -> usize {
        self.layers * self.layer.bytes() as usize
    }
}
