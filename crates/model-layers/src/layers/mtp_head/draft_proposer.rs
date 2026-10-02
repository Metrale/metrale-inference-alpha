// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `DraftProposer` for [`MtpHead`]: single and batched propose, drafter-KV
//! hand-over (take, install, free), catch-up rows and the trim after verify.
//!
//! Owner: model-layers (MTP head).
//! Invariants:
//! - `install_drafter_kv` installs only into a state with no blocks and no rows.
//! - `after_verify` trims at most `max(last_num_drafted, 1)` rows.
use super::*;

impl DraftProposer for MtpHead {
    fn circuit_draft(
        &self,
        config: &metrale_config::ModelConfig,
        levers: &crate::layers::ops::ModelLevers,
    ) -> Option<crate::circuit_exec::DraftBinding> {
        Some(self.circuit_binding(config, levers))
    }

    fn set_circuit_draft(
        &self,
        runner: Option<std::sync::Arc<dyn crate::circuit_exec::DraftRunner>>,
    ) {
        self.install_circuit(runner);
    }

    fn alloc_state(&self, _gpu: &dyn GpuBackend) -> Result<Box<dyn ProposerState>> {
        Ok(Box::new(MtpProposerState {
            block_table: Vec::new(),
            seq_len: 0,
            last_num_drafted: 0,
            last_pair_key: None,
            last_drafts: Vec::new(),
            pending_catchup: Vec::new(),
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
        let mtp_state = state
            .as_any_mut()
            .downcast_mut::<MtpProposerState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid MTP proposer state"))?;

        // 2026-10-02: `--draft-confidence-stop`: the chain stops after the first
        // draft whose top-1 probability is below tau (`speculative::draft_stop`).
        let stop_lp = crate::speculative::draft_stop::draft_stop_logprob();
        let mut drafts = Vec::with_capacity(num_drafts);
        let mut current_token = last_token;
        let mut current_hidden = target_hidden;

        for i in 0..num_drafts {
            let embed_target = if i == num_drafts - 1 {
                draft_embed_target
            } else {
                None
            };
            // 2026-09-25: The grammar matcher is not advanced between draft positions:
            // every position gets the same mask, with a warning past the first.
            if grammar_bitmask.is_some() && i > 0 {
                tracing::warn!(
                    "MTP grammar-masked drafting called with num_drafts>1 (i={i}); \
                     mask held fixed across draft positions — acceptance may drop."
                );
            }
            let mask_for_draft = grammar_bitmask;
            let draft = self.forward_one(
                current_token,
                current_hidden,
                position + i,
                mtp_state,
                ctx,
                stream,
                embed_target,
                mask_for_draft,
                i == 0,
            )?;
            tracing::debug!(
                "MTP propose[{i}]: token={current_token} pos={} mtp_seq_len={} → draft={draft}",
                position + i,
                mtp_state.seq_len,
            );
            drafts.push(draft);
            current_token = draft;
            // 2026-09-25: Later drafts read the drafter's own hidden (`chain_hidden`).
            current_hidden = Self::chain_hidden(ctx);
            // 2026-10-02: The last draft's log-probability is NaN when it was not
            // measured (a masked or deferred draft), which never stops the chain.
            if let Some(ln_tau) = stop_lp {
                let lp = f32::from_bits(
                    self.last_conf_bits
                        .load(std::sync::atomic::Ordering::Relaxed),
                );
                if !lp.is_nan() && !crate::speculative::draft_stop::chain_continues(lp, ln_tau) {
                    break;
                }
            }
        }

        mtp_state.last_num_drafted = drafts.len();
        mtp_state.last_drafts.clone_from(&drafts);
        Ok(drafts)
    }

    fn propose_batch(
        &self,
        last_tokens: &[u32],
        target_hiddens: &[DevicePtr],
        positions: &[usize],
        num_drafts: usize,
        states: &mut [&mut dyn ProposerState],
        ctx: &ForwardContext,
        stream: u64,
        out_conf: Option<&mut Vec<Vec<f32>>>,
    ) -> Result<Option<Vec<Vec<u32>>>> {
        if !self.can_propose_batch(last_tokens.len(), ctx.buffers, ctx.config) {
            return Ok(None);
        }
        // 2026-09-30: Under `--forward circuit` the batch runs the circuit's n-row draft
        // program, which writes the D-Cut confidences; a width it has no program for, or a
        // caller without D-Cut (whose legacy argmax the plan does not state), drafts each
        // sequence alone on the single-row program (`forward.rs`).
        let circuit = self.circuit_draft.read().clone();
        if let Some(r) = &circuit
            && !(r.serves(last_tokens.len() as u64) && out_conf.is_some())
        {
            return Ok(None);
        }
        // 2026-09-25: The width policy sees only the config and the arena, so the
        // grouped FP8 decode's context terms (kill switch, FP32 routing, EP) are
        // checked here: a refusal falls back per sequence instead of failing
        // mid-chain, and is logged once per process.
        if let Some(moe) = self.moe_fp8.as_ref()
            && !moe.fp8_grouped_decode_ok(last_tokens.len(), ctx)
        {
            static LOGGED: std::sync::Once = std::sync::Once::new();
            LOGGED.call_once(|| {
                tracing::warn!(
                    "MTP propose_batch: grouped FP8 MoE decode refused for n={} (kill switch \
                     METRALE_NO_FP8_MOE_GROUPED_DECODE, FP32 routing or EP) — the MoE drafter \
                     falls back to per-sequence propose",
                    last_tokens.len()
                );
            });
            return Ok(None);
        }
        // 2026-09-25: A state that is not an `MtpProposerState` makes the whole batch
        // fall back per sequence.
        let mut mtp_states: Vec<&mut MtpProposerState> = Vec::with_capacity(states.len());
        for s in states.iter_mut() {
            match s.as_any_mut().downcast_mut::<MtpProposerState>() {
                Some(st) => mtp_states.push(st),
                None => return Ok(None),
            }
        }
        self.propose_batch_impl(
            circuit.as_deref(),
            last_tokens,
            target_hiddens,
            positions,
            num_drafts,
            &mut mtp_states,
            ctx,
            stream,
            out_conf,
        )
        .map(Some)
    }

    fn propose_batch_max(
        &self,
        buffers: &metrale_gpu_runtime::buffers::BufferArena,
        config: &metrale_config::ModelConfig,
    ) -> usize {
        MtpHead::propose_batch_max(self, buffers, config)
    }

    fn prefill_drafter(
        &self,
        prompt_tokens: &[u32],
        hiddens: DevicePtr,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<usize> {
        self.prefill_drafter_impl(prompt_tokens, hiddens, state, ctx, stream)
    }

    fn drafter_rows(&self, state: &mut dyn ProposerState) -> usize {
        state
            .as_any_mut()
            .downcast_mut::<MtpProposerState>()
            .map(|s| s.seq_len)
            .unwrap_or(0)
    }

    fn last_pair_key(&self, state: &mut dyn ProposerState) -> Option<usize> {
        state
            .as_any_mut()
            .downcast_mut::<MtpProposerState>()
            .and_then(|s| s.last_pair_key)
    }

    fn take_drafter_kv(
        &self,
        state: &mut dyn ProposerState,
    ) -> Option<(Vec<u32>, usize, Option<usize>)> {
        let st = state.as_any_mut().downcast_mut::<MtpProposerState>()?;
        if st.block_table.is_empty() || st.seq_len == 0 {
            return None;
        }
        let blocks = std::mem::take(&mut st.block_table);
        let rows = st.seq_len;
        let key = st.last_pair_key;
        // 2026-09-25: No blocks, no rows, no pair key: `free_state` then has nothing to
        // release.
        st.seq_len = 0;
        st.last_pair_key = None;
        st.last_num_drafted = 0;
        Some((blocks, rows, key))
    }

    fn install_drafter_kv(
        &self,
        state: &mut dyn ProposerState,
        blocks: Vec<u32>,
        rows: usize,
        last_pair_key: Option<usize>,
    ) -> bool {
        let Some(st) = state.as_any_mut().downcast_mut::<MtpProposerState>() else {
            return false;
        };
        // 2026-09-25: Only into an empty state; otherwise its blocks would leak.
        if !st.block_table.is_empty() || st.seq_len != 0 {
            return false;
        }
        st.block_table = blocks;
        st.seq_len = rows;
        st.last_pair_key = last_pair_key;
        true
    }

    fn free_drafter_kv(&self, blocks: &[u32]) {
        if !blocks.is_empty() {
            self.kv_cache.lock().free_blocks(blocks);
        }
    }

    fn catchup_drafter(
        &self,
        tokens: &[u32],
        hiddens: DevicePtr,
        row_base: usize,
        pos_base: usize,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<usize> {
        self.drafter_rows_impl(tokens, hiddens, row_base, pos_base, state, ctx, stream)
    }

    fn read_deferred_draft_token(&self, gpu: &dyn GpuBackend) -> Result<u32> {
        self.read_deferred_draft_token(gpu)
    }

    fn catchup_batch(
        &self,
        tokens: &[Vec<u32>],
        hiddens: &[Vec<DevicePtr>],
        first_pos: &[usize],
        states: &mut [&mut dyn ProposerState],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<usize> {
        self.catchup_batch_impl(tokens, hiddens, first_pos, states, ctx, stream)
    }

    fn after_verify(
        &self,
        num_accepted: usize,
        state: &mut dyn ProposerState,
        _stream: u64,
    ) -> Result<()> {
        let mtp_state = state
            .as_any_mut()
            .downcast_mut::<MtpProposerState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid MTP proposer state"))?;

        let num_drafted = mtp_state.last_num_drafted.max(1);
        // 2026-09-25: With exact KV, row 0 was built from the target hidden and stays;
        // rows 1.. were built from drafter hiddens and go, and the catch-up
        // rebuilds the accepted ones. Otherwise `mtp_rows_to_trim` decides.
        let num_to_trim = if self.kv_exact {
            num_drafted - 1
        } else {
            mtp_rows_to_trim(
                num_drafted,
                num_accepted,
                crate::speculative::mtp_refeed_accepted_enabled(),
            )
        };
        let old_sl = mtp_state.seq_len;
        if num_to_trim > 0 {
            mtp_state.seq_len = mtp_state.seq_len.saturating_sub(num_to_trim);
            // 2026-09-25: Trimmed rows have consecutive pair keys, so the newest
            // surviving key moves back by the same count.
            if let Some(k) = mtp_state.last_pair_key {
                mtp_state.last_pair_key = Some(k.saturating_sub(num_to_trim));
            }
        }
        tracing::debug!(
            "MTP after_verify: accepted={num_accepted} drafted={num_drafted} trim={num_to_trim} mtp_seq_len: {old_sl} → {}",
            mtp_state.seq_len,
        );
        Ok(())
    }

    fn free_state(&self, _gpu: &dyn GpuBackend, state: &mut dyn ProposerState) -> Result<()> {
        let mtp_state = state
            .as_any_mut()
            .downcast_mut::<MtpProposerState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid MTP proposer state"))?;
        if !mtp_state.block_table.is_empty() {
            self.kv_cache.lock().free_blocks(&mtp_state.block_table);
            mtp_state.block_table.clear();
        }
        mtp_state.seq_len = 0;
        Ok(())
    }
}
