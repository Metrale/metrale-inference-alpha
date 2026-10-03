// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `propose_batch_impl`, the batched propose loop: one
//! `forward_batch_position` per draft position. It is a child module of
//! `position` because that forward is private there.
//!
//! Owner: model-layers (MTP head).
//! Invariants:
//! - `last_num_drafted` is set to j + 1 once draft position j has completed,
//!   so a later error leaves it at the count of completed positions. It is
//!   not reset before position 0.

use super::*;

impl MtpHead {
    /// 2026-09-25: Name of the arm `gemm_rows` takes for the `fc` shape
    /// (N = h, K = 2h) at this width, for the "propose_batch active" log line.
    fn propose_proj_arm(
        &self,
        gpu: &dyn metrale_gpu_runtime::gpu::GpuBackend,
        n: usize,
        h: usize,
    ) -> &'static str {
        if ops::dense_gemv_tc::kernel_for(gpu, n as u32, h as u32, (2 * h) as u32).is_some() {
            return "TC-GEMV (kill: METRALE_NO_MTP_TC)";
        }
        match row_dispatch::drafter_row_kernel(
            n,
            h as u32,
            (2 * h) as u32,
            self.dense_gemv_batchm_k.0 != 0,
            row_dispatch::kv_gemv_pinned(),
            row_dispatch::small_m_tier_off(),
        ) {
            row_dispatch::RowKernel::Batchm => "GEMV-BATCHM",
            row_dispatch::RowKernel::Pipelined => "PIPELINED-GEMM",
            row_dispatch::RowKernel::GemvLoop => "GEMV-LOOP",
        }
    }

    /// 2026-09-25: Batched propose: `num_drafts` chained positions, each one
    /// n-row forward. Draft 0 reads the caller's target hiddens; draft j > 0
    /// reads row i of `Self::chain_hidden`, written by position j - 1.
    /// Returns the drafts per sequence and stores them in each state's
    /// `last_drafts`. Errors when the slice lengths differ. 2026-09-30: With `circuit`, each
    /// position runs that runner's n-row draft program, which writes the confidences, so the
    /// caller passes `out_conf` (`propose_batch` checks both).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn propose_batch_impl(
        &self,
        circuit: Option<&dyn crate::circuit_exec::DraftRunner>,
        last_tokens: &[u32],
        target_hiddens: &[DevicePtr],
        positions: &[usize],
        num_drafts: usize,
        states: &mut [&mut MtpProposerState],
        ctx: &ForwardContext,
        stream: u64,
        mut out_conf: Option<&mut Vec<Vec<f32>>>,
    ) -> Result<Vec<Vec<u32>>> {
        let n = last_tokens.len();
        ensure!(
            n == target_hiddens.len() && n == positions.len() && n == states.len(),
            "propose_batch: length mismatch"
        );
        // 2026-09-25: Log once per distinct n (a bitmask over `n & 31`), naming
        // the kernels this width selects. 2026-10-01: Legacy and the circuit's draft program
        // each log their own first propose at a width.
        static LOGGED_N: [std::sync::atomic::AtomicU32; 2] =
            [const { std::sync::atomic::AtomicU32::new(0) }; 2];
        let bit = 1u32 << (n & 31);
        let logged = &LOGGED_N[usize::from(circuit.is_some())];
        if (logged.fetch_or(bit, std::sync::atomic::Ordering::Relaxed) & bit) == 0 {
            // 2026-09-25: The same condition as the LM-head dispatch in
            // `forward_batch_position`.
            let v = if self.mtp_vocab_size > 0 {
                self.mtp_vocab_size.min(ctx.config.vocab_size as u32)
            } else {
                ctx.config.vocab_size as u32
            };
            let tc = self.mtp_tc_lm_head(ctx.gpu, n, v, ctx.config.hidden_size as u32);
            if circuit.is_some() {
                tracing::info!("MTP propose_batch active: n={n} on the circuit's draft program");
            } else if lm_head_rows_arm(n, tc, self.lm_head_twin_ready()) == LmHeadRowsArm::TileTwin
                && let Some((_, ldb)) = self.lm_head_nvfp4_t
            {
                tracing::info!(
                    "MTP propose_batch active: n={n} proj={} ffn={} pipelined_gemm={:#x} \
                     gemv_batchm={:#x} lm_head=TILE-TWIN (handle {:#x}, ldb={ldb}) \
                     — kill switch METRALE_NO_MTP_LMHEAD_TGEMM (presence)",
                    self.propose_proj_arm(ctx.gpu, n, ctx.config.hidden_size),
                    self.propose_ffn_arm_name(),
                    self.dense_gemm_pipelined_k.0,
                    self.dense_gemv_batchm_k.0,
                    self.w4a16_gemm_t_k.0,
                );
            } else {
                tracing::info!(
                    "MTP propose_batch active: n={n} proj={} ffn={} pipelined_gemm={:#x} \
                     gemv_batchm={:#x} lm_head_batchm={:#x}",
                    self.propose_proj_arm(ctx.gpu, n, ctx.config.hidden_size),
                    self.propose_ffn_arm_name(),
                    self.dense_gemm_pipelined_k.0,
                    self.dense_gemv_batchm_k.0,
                    self.lm_head_batch_kernel(n).0
                );
            }
        }
        // 2026-10-02: `--draft-confidence-stop`: with the log-probabilities
        // requested, the loop ends once every sequence's chain has stopped
        // (`draft_stop::chain_continues`). Each sequence keeps the drafts the
        // loop made; the scheduler truncates each chain to its own depth
        // (`draft_stop::chain_depth`, in `mtp_dcut::plan`).
        let stop_lp =
            crate::speculative::draft_stop::draft_stop_logprob().filter(|_| out_conf.is_some());
        let mut stopped = vec![false; n];

        let h = ctx.config.hidden_size;
        let mut cur_tokens = last_tokens.to_vec();
        let mut cur_positions = positions.to_vec();
        let mut ids = vec![0u32; n];
        let mut all: Vec<Vec<u32>> = vec![Vec::with_capacity(num_drafts); n];
        // 2026-09-25: D-Cut: per-sequence, per-position top-1 log-probability,
        // filled only when the caller passes `out_conf`.
        let mut lp = vec![0f32; n];
        if let Some(c) = out_conf.as_deref_mut() {
            c.clear();
            c.resize(n, Vec::with_capacity(num_drafts));
        }
        for j in 0..num_drafts {
            let hiddens_j: Vec<DevicePtr> = if j == 0 {
                target_hiddens.to_vec()
            } else {
                let chain = Self::chain_hidden(ctx);
                (0..n).map(|i| chain.offset(i * h * 2)).collect()
            };
            match circuit {
                Some(runner) => self.forward_batch_position_circuit(
                    runner,
                    &cur_tokens,
                    &hiddens_j,
                    &cur_positions,
                    states,
                    ctx,
                    stream,
                    &mut ids,
                    &mut lp,
                )?,
                None => self.forward_batch_position(
                    &cur_tokens,
                    &hiddens_j,
                    &cur_positions,
                    states,
                    ctx,
                    stream,
                    &mut ids,
                    if out_conf.is_some() {
                        Some(&mut lp[..])
                    } else {
                        None
                    },
                    j == 0,
                )?,
            }
            for i in 0..n {
                all[i].push(ids[i]);
                cur_positions[i] += 1;
            }
            if let Some(c) = out_conf.as_deref_mut() {
                for (i, row) in c.iter_mut().enumerate().take(n) {
                    row.push(lp[i]);
                }
            }
            cur_tokens.copy_from_slice(&ids);
            for state in states.iter_mut() {
                state.last_num_drafted = j + 1;
            }
            if let Some(ln_tau) = stop_lp {
                for (s, &l) in stopped.iter_mut().zip(&lp) {
                    *s = *s || !crate::speculative::draft_stop::chain_continues(l, ln_tau);
                }
                if stopped.iter().all(|&s| s) {
                    break;
                }
            }
        }
        for (state, drafts) in states.iter_mut().zip(all.iter()) {
            state.last_drafts.clone_from(drafts);
        }
        Ok(all)
    }
}
