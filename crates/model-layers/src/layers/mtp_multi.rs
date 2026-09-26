// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Multi-module MTP proposer: draft slot i runs its own
//! `MtpHead` (own weights and KV cache), as
//! `modules[i].forward_one(previous draft token, previous module's hidden)`.
//!
//! Owner: model-layers (MTP head).
//! Invariants:
//! - `modules` is non-empty, and each state holds one `MtpProposerState`
//!   per module.
//! - A propose runs `min(num_drafts, modules.len())` modules, the first ones.

use std::any::Any;

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use crate::layer::ForwardContext;
use crate::layers::mtp_head::{MtpHead, MtpProposerState};
use crate::speculative::{DraftProposer, ProposerState};

/// 2026-09-25: Per-sequence state for `MultiModuleMtpHead`: one
/// `MtpProposerState` per module, since the modules do not share a KV cache.
pub struct MultiModuleMtpState {
    /// 2026-09-25: `per_module[i]` belongs to `modules[i]`; `alloc_state`
    /// builds one per module.
    pub per_module: Vec<MtpProposerState>,
    /// 2026-09-25: Drafts produced by the last `propose`; `after_verify`
    /// updates that many modules.
    pub last_num_drafted: usize,
}

impl ProposerState for MultiModuleMtpState {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// 2026-09-25: Independent MTP modules, one per draft slot.
pub struct MultiModuleMtpHead {
    /// 2026-09-25: Non-empty (`new` refuses an empty list). `build_mtp_proposer`
    /// builds one head per loaded MTP weight set.
    modules: Vec<MtpHead>,
}

impl std::fmt::Debug for MultiModuleMtpHead {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MultiModuleMtpHead")
            .field("modules", &self.modules.len())
            .finish()
    }
}

impl MultiModuleMtpHead {
    /// 2026-09-25: Assemble a multi-module proposer from per-module heads.
    /// Errors on an empty list.
    pub fn new(modules: Vec<MtpHead>) -> Result<Self> {
        anyhow::ensure!(
            !modules.is_empty(),
            "MultiModuleMtpHead requires at least one module (got 0); \
             caller should not construct this type for single-module MTP"
        );
        Ok(Self { modules })
    }

    /// 2026-09-25: Number of modules; `propose` caps `num_drafts` at it.
    pub fn num_modules(&self) -> usize {
        self.modules.len()
    }
}

impl DraftProposer for MultiModuleMtpHead {
    fn alloc_state(&self, _gpu: &dyn GpuBackend) -> Result<Box<dyn ProposerState>> {
        let per_module = (0..self.modules.len())
            .map(|_| MtpProposerState {
                block_table: Vec::new(),
                seq_len: 0,
                last_num_drafted: 0,
                last_pair_key: None,
                last_drafts: Vec::new(),
                pending_catchup: Vec::new(),
            })
            .collect();
        Ok(Box::new(MultiModuleMtpState {
            per_module,
            last_num_drafted: 0,
        }))
    }

    fn propose(
        &self,
        last_token: u32,
        target_hidden: DevicePtr,
        position: usize,
        num_drafts: usize,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
        draft_embed_target: Option<DevicePtr>,
        grammar_bitmask: Option<&[i32]>,
        _target_hidden_stack: Option<DevicePtr>,
    ) -> Result<Vec<u32>> {
        let mm_state = state
            .as_any_mut()
            .downcast_mut::<MultiModuleMtpState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid MultiModuleMtp state"))?;

        // 2026-09-25: More drafts than modules are clamped to the module
        // count without a warning.
        let k = num_drafts.min(self.modules.len());

        let mut drafts = Vec::with_capacity(k);
        let mut current_token = last_token;
        let mut current_hidden = target_hidden;

        for i in 0..k {
            // 2026-09-25: Only the last draft stages its embedding at
            // `draft_embed_target`; earlier drafts reach the next module as
            // the returned token.
            let embed_target = if i == k - 1 { draft_embed_target } else { None };

            // 2026-09-25: Every module gets the same single-position grammar
            // mask.
            let mask_for_draft = grammar_bitmask;

            let draft = self.modules[i].forward_one(
                current_token,
                current_hidden,
                position + i,
                &mut mm_state.per_module[i],
                ctx,
                stream,
                embed_target,
                mask_for_draft,
                i == 0,
            )?;

            tracing::debug!(
                "MultiMTP propose[{i}/{k}]: token={current_token} pos={} module_seq_len={} → draft={draft}",
                position + i,
                mm_state.per_module[i].seq_len,
            );

            drafts.push(draft);
            current_token = draft;
            // 2026-09-25: Module i + 1 reads module i's residual stream from
            // `hidden_states`, whatever `ModelLevers::mtp_chain_postnorm` says
            // (`MtpHead::chain_hidden` honours it).
            current_hidden = ctx.buffers.hidden_states();
        }

        mm_state.last_num_drafted = drafts.len();
        Ok(drafts)
    }

    fn read_deferred_draft_token(&self, gpu: &dyn GpuBackend) -> Result<u32> {
        // 2026-09-25: Reads the last module's deferred id. The last draft
        // came from `modules[k - 1]`, so this is that draft only when the
        // propose ran every module (k == modules.len()).
        self.modules
            .last()
            .expect("MultiModuleMtpHead::new enforces non-empty")
            .read_deferred_draft_token(gpu)
    }

    fn after_verify(
        &self,
        num_accepted: usize,
        state: &mut dyn ProposerState,
        stream: u64,
    ) -> Result<()> {
        let mm_state = state
            .as_any_mut()
            .downcast_mut::<MultiModuleMtpState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid MultiModuleMtp state"))?;

        // 2026-09-25: Each of the k modules wrote one row in the last
        // propose. Modules 0..num_accepted keep it; modules num_accepted..k
        // drop it.
        let k = mm_state.last_num_drafted;
        for (i, per) in mm_state.per_module.iter_mut().take(k).enumerate() {
            per.last_num_drafted = 1;
            let trim = if i < num_accepted { 0 } else { 1 };
            if trim > 0 {
                per.seq_len = per.seq_len.saturating_sub(trim);
            }
        }
        let _ = stream;
        tracing::debug!(
            "MultiMTP after_verify: accepted={num_accepted} of {k}; per-module trim done"
        );
        Ok(())
    }

    fn free_state(&self, _gpu: &dyn GpuBackend, state: &mut dyn ProposerState) -> Result<()> {
        let mm_state = state
            .as_any_mut()
            .downcast_mut::<MultiModuleMtpState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid MultiModuleMtp state"))?;
        // 2026-09-25: Return each module's blocks to that module's KV cache.
        for (i, per) in mm_state.per_module.iter_mut().enumerate() {
            let head = &self.modules[i];
            if !per.block_table.is_empty() {
                head.kv_cache_lock().free_blocks(&per.block_table);
                per.block_table.clear();
            }
            per.seq_len = 0;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_modules_rejected() {
        let err = MultiModuleMtpHead::new(vec![]).unwrap_err();
        assert!(
            err.to_string().contains("at least one module"),
            "unexpected error: {err}"
        );
    }
}
