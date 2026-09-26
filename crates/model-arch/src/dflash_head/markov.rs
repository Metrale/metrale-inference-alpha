// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

//! 2026-09-25: DSpark sequential Markov fixup over the DFlash block logits.
//!
//! Owner: model-arch (DFlash drafter).
//! Invariants:
//! - Row i's bias reads the token slot of row i-1; row 0 reads its own slot,
//!   which still holds `last_token` because row 0's argmax has not run yet.
//! - The loop reads and writes device memory only, so it runs inside the
//!   captured tail subgraph. `forward_block` copies the `[last_token, MASK, …]`
//!   token row into `draft_tokens_dev` before the captured region.
//!
//! Rows are sampled left to right instead of all at once:
//!
//! ```text
//!   logits[i] += markov_w2 @ markov_w1[prev_i]     (low-rank bigram bias)
//!   draft[i]   = argmax(logits[i])
//! ```
//!
//! where `prev_0 = last_token` and `prev_i = draft[i-1]` for i ≥ 1. Row 0 is not
//! biased when `METRALE_DSPARK_ANCHOR_BIAS=0`.
//!
//! Per biased row: one `[1, rank]` gather, one `[vocab, rank]` GEMV and one
//! `[vocab]` residual add, plus the argmax every row runs.

use anyhow::Result;

use super::BlockDiffusionDraftHead;
use metrale_model_layers::layer::ForwardContext;
use metrale_model_layers::layers::ops;

impl BlockDiffusionDraftHead {
    /// 2026-09-25: The drafter carries the Markov head, its scratch is allocated,
    /// and the Markov lever in `DFlashLevers` is on.
    pub(super) fn markov_active(&self) -> bool {
        self.markov_rank > 0
            && self.markov_w1.is_some()
            && self.markov_w2.is_some()
            && self.scratch.markov_embed.0 != 0
            && self.levers.dspark_markov
    }

    /// 2026-09-25: The confidence head also runs inside the sequential chain:
    /// weights present, scratch allocated, and `METRALE_DSPARK_CONF_TAU` > 0.
    pub(super) fn confidence_active(&self) -> bool {
        self.confidence_proj.is_some() && self.scratch.conf_out.0 != 0 && self.levers.conf_tau > 0.0
    }

    /// 2026-09-25: Sequential Markov-biased argmax over the `block_g()` block rows.
    /// `forward_block`'s tail calls it in place of the per-row argmax loop when
    /// [`Self::markov_active`] holds and the DFlash2 path is not active.
    ///
    /// `norm_noise`: the post-final-norm hidden rows the lm_head GEMM read; the
    /// confidence head reads row i of it.
    pub(super) fn markov_argmax_block(
        &self,
        ctx: &ForwardContext,
        norm_noise: metrale_gpu_runtime::gpu::DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let gpu = ctx.gpu;
        let bf16 = 2usize;
        let vocab = self.vocab_size as u32;
        let rank = self.markov_rank as u32;
        let w1 = self
            .markov_w1
            .as_ref()
            .expect("markov_active() checked markov_w1");
        let w2 = self
            .markov_w2
            .as_ref()
            .expect("markov_active() checked markov_w2");
        let anchor_bias = self.levers.dspark_anchor_bias;
        let conf_on = self.confidence_active();
        let bf16u = 2usize;

        for i in 0..self.block_g() {
            let logits_row = self.scratch.logits.offset(i * self.vocab_size * bf16);
            let token_slot = self.scratch.draft_tokens_dev.offset(i * 4);
            let biased = i > 0 || anchor_bias;
            if biased {
                // 2026-09-25: prev_0 is slot 0 itself (still `last_token`: this
                // read runs before row 0's argmax overwrites the slot); prev_i
                // for i ≥ 1 is row i-1's argmax output.
                let prev_slot = self
                    .scratch
                    .draft_tokens_dev
                    .offset(i.saturating_sub(1) * 4);
                ops::batched_embed(
                    gpu,
                    self.kernels.batched_embed,
                    prev_slot,
                    w1.weight,
                    self.scratch.markov_embed,
                    1,
                    rank,
                    stream,
                )?;
                ops::dense_gemv(
                    gpu,
                    self.kernels.dense_gemv,
                    self.scratch.markov_embed,
                    w2,
                    self.scratch.markov_bias,
                    vocab,
                    rank,
                    stream,
                )?;
                ops::residual_add(
                    gpu,
                    self.kernels.residual_add,
                    logits_row,
                    self.scratch.markov_bias,
                    vocab,
                    stream,
                )?;
                // 2026-09-25: Confidence head for row i:
                // conf[i] = W · [hidden_i ‖ markov_embed_i] + b, as GEMVs over
                // the `[1, hidden + rank]` weight, hidden columns first. The
                // embed half runs only when `confidence_with_markov`, and
                // stages through markov_bias[0], whose vocab-wide content the
                // residual_add above has already consumed.
                if conf_on {
                    let (w, b) = (
                        self.confidence_proj.as_ref().expect("confidence_active"),
                        self.confidence_bias.as_ref().expect("confidence_active"),
                    );
                    let conf_i = self.scratch.conf_out.offset(i * bf16u);
                    let hidden_row = norm_noise.offset(i * self.hidden_size * bf16u);
                    ops::dense_gemv(
                        gpu,
                        self.kernels.dense_gemv,
                        hidden_row,
                        w,
                        conf_i,
                        1,
                        self.hidden_size as u32,
                        stream,
                    )?;
                    if self.confidence_with_markov {
                        let w_embed = metrale_model_layers::weight_map::DenseWeight {
                            weight: w.weight.offset(self.hidden_size * bf16u),
                        };
                        ops::dense_gemv(
                            gpu,
                            self.kernels.dense_gemv,
                            self.scratch.markov_embed,
                            &w_embed,
                            self.scratch.markov_bias,
                            1,
                            rank,
                            stream,
                        )?;
                        ops::residual_add(
                            gpu,
                            self.kernels.residual_add,
                            conf_i,
                            self.scratch.markov_bias,
                            1,
                            stream,
                        )?;
                    }
                    ops::residual_add(gpu, self.kernels.residual_add, conf_i, b.weight, 1, stream)?;
                }
            }
            ops::argmax_bf16(
                gpu,
                self.kernels.argmax,
                logits_row,
                token_slot,
                vocab,
                stream,
            )?;
        }
        Ok(())
    }
}
