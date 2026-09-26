// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `impl DraftProposer for DeepseekV4MtpHead`.
//!
//! Owner: model-arch, DeepSeek-V4 MTP.
//! Invariants: none beyond the types.

use super::*;

impl DraftProposer for DeepseekV4MtpHead {
    fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn ProposerState>> {
        Ok(Box::new(self.alloc_state_inner(gpu)?))
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
        _draft_embed_target: Option<DevicePtr>,
        grammar_bitmask: Option<&[i32]>,
        _target_hidden_stack: Option<DevicePtr>,
    ) -> Result<Vec<u32>> {
        let v4_state = state
            .as_any_mut()
            .downcast_mut::<DeepseekV4MtpProposerState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid V4 MTP proposer state"))?;

        let mut drafts = Vec::with_capacity(num_drafts);
        let mut current_token = last_token;
        let mut current_hidden = target_hidden;
        for i in 0..num_drafts {
            if grammar_bitmask.is_some() && i > 0 {
                tracing::warn!(
                    "V4 MTP grammar-masked drafting with num_drafts>1 (i={i}); \
                     mask held fixed across draft positions — acceptance may drop."
                );
            }
            let draft = self.forward_one(
                current_token,
                current_hidden,
                position + i,
                v4_state,
                ctx,
                stream,
                grammar_bitmask,
            )?;
            tracing::debug!(
                "V4 MTP propose[{i}]: token={current_token} pos={} mtp_seq_len={} → draft={draft}",
                position + i,
                v4_state.seq_len,
            );
            drafts.push(draft);
            current_token = draft;
            // 2026-09-25: Later drafts feed on the MTP head's own collapsed
            // hidden (`forward_one` leaves it in `hidden_states()`).
            current_hidden = ctx.buffers.hidden_states();
        }
        v4_state.last_num_drafted = drafts.len();
        Ok(drafts)
    }

    fn after_verify(
        &self,
        num_accepted: usize,
        state: &mut dyn ProposerState,
        _stream: u64,
    ) -> Result<()> {
        let v4_state = state
            .as_any_mut()
            .downcast_mut::<DeepseekV4MtpProposerState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid V4 MTP proposer state"))?;
        // 2026-09-25: Trim `drafted - accepted` rejected entries (drafted
        // counted as at least 1) from the MTP KV cache by rolling back
        // `seq_len`; the next propose overwrites the slots.
        let num_drafted = v4_state.last_num_drafted.max(1);
        let num_to_trim = num_drafted.saturating_sub(num_accepted);
        let old_sl = v4_state.seq_len;
        if num_to_trim > 0 {
            v4_state.seq_len = v4_state.seq_len.saturating_sub(num_to_trim);
        }
        tracing::debug!(
            "V4 MTP after_verify: accepted={num_accepted} drafted={num_drafted} \
             trim={num_to_trim} mtp_seq_len: {old_sl} → {}",
            v4_state.seq_len,
        );
        Ok(())
    }

    fn free_state(&self, _gpu: &dyn GpuBackend, state: &mut dyn ProposerState) -> Result<()> {
        let v4_state = state
            .as_any_mut()
            .downcast_mut::<DeepseekV4MtpProposerState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid V4 MTP proposer state"))?;
        if !v4_state.block_table.is_empty() {
            self.kv_cache.lock().free_blocks(&v4_state.block_table);
            v4_state.block_table.clear();
        }
        v4_state.seq_len = 0;
        Ok(())
    }
}
