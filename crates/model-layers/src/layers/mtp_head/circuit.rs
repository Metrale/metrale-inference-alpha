// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-29: The MTP draft head for the circuit executor: its weights by slot, its attention
//! over its own KV cache, and the features the circuit does not model; and the runner that,
//! once installed, takes over the head's single-row forward (`forward_one`).
//!
//! Owner: model-layers (MTP head).
//! Invariants:
//! - Only the BF16 head with a dense FFN, a BF16 draft KV cache and the NVFP4 draft lm_head
//!   is bound; every other arm of `forward_one` is reported as unmodelled.
//! - A runner is installed only while the executor that compiled it is live
//!   (`TransformerModel::set_forward` removes it before freeing the executor).

use std::collections::BTreeMap;
use std::sync::Arc;

use metrale_cache::kv_cache::KvCacheDtype;
use metrale_circuit::LinearRole;

use super::{MtpHead, MtpQuantization, ProjectionWeight};
use crate::circuit_exec::{
    AttnFacts, BoundWeight, CircuitLayer, DraftBinding, DraftRunner, MixerFacts, RopeFacts,
    WeightSlot,
};

fn bf16(w: &ProjectionWeight, what: &str, unmodelled: &mut Vec<String>) -> Option<BoundWeight> {
    match w {
        ProjectionWeight::Bf16(d) => Some(BoundWeight::Dense(*d)),
        _ => {
            unmodelled.push(format!("a non-BF16 draft {what} projection"));
            None
        }
    }
}

impl MtpHead {
    /// 2026-09-29: The head as the circuit executor binds it.
    pub(super) fn circuit_binding(
        &self,
        config: &metrale_config::ModelConfig,
        levers: &crate::layers::ops::ModelLevers,
    ) -> DraftBinding {
        let mut unmodelled = Vec::new();
        let arms = [
            (self.quant != MtpQuantization::Bf16, "a non-BF16 draft head"),
            (!self.kv_bf16, "an FP8 draft KV cache"),
            (self.dense_ffn_generic.is_none(), "a MoE draft FFN"),
            (
                !self.gemv_sw || !levers.gemv_sw,
                "the draft lm_head off the single-warp GEMV",
            ),
            (
                levers.mtp_target_postnorm,
                "the target post-norm (METRALE_MTP_TARGET_POSTNORM)",
            ),
            (
                crate::speculative::draft_stop::draft_stop_logprob().is_some(),
                "the draft confidence stop (--draft-confidence-stop)",
            ),
            (
                levers.shadow_topk > 0,
                "shadow top-k logging (METRALE_MTP_SHADOW_TOPK)",
            ),
            (
                levers.mtp_debug_norms,
                "draft norm logging (METRALE_MTP_DEBUG_NORMS)",
            ),
            (
                levers.mtp_chain_postnorm,
                "the final-normed chain hidden (METRALE_MTP_CHAIN_POSTNORM)",
            ),
        ];
        for (present, what) in arms {
            if present {
                unmodelled.push(what.to_string());
            }
        }
        let dense = BoundWeight::Dense;
        let mut weights = BTreeMap::from([
            (WeightSlot::EmbedNorm, dense(self.pre_fc_norm_embedding)),
            (WeightSlot::HiddenNorm, dense(self.pre_fc_norm_hidden)),
            (WeightSlot::InputNorm, dense(self.input_layernorm)),
            (WeightSlot::PostNorm, dense(self.post_attn_layernorm)),
            (WeightSlot::FinalNorm, dense(self.norm)),
            (WeightSlot::QNorm, dense(self.q_norm)),
            (WeightSlot::KNorm, dense(self.k_norm)),
            (WeightSlot::LmHead, BoundWeight::Nvfp4(self.lm_head_nvfp4)),
        ]);
        let projections = [
            (WeightSlot::Linear(LinearRole::MtpFc), &self.fc, "fc"),
            (WeightSlot::Linear(LinearRole::Q), &self.q_proj, "q"),
            (WeightSlot::Linear(LinearRole::K), &self.k_proj, "k"),
            (WeightSlot::Linear(LinearRole::V), &self.v_proj, "v"),
            (WeightSlot::Linear(LinearRole::O), &self.o_proj, "o"),
        ];
        for (slot, w, what) in projections {
            if let Some(b) = bf16(w, what, &mut unmodelled) {
                weights.insert(slot, b);
            }
        }
        if let Some((gate, up, down)) = &self.dense_ffn_generic {
            for (slot, w, what) in [
                (WeightSlot::FfnGate, gate, "gate"),
                (WeightSlot::FfnUp, up, "up"),
                (WeightSlot::Linear(LinearRole::Down), down, "down"),
            ] {
                if let Some(b) = bf16(w, what, &mut unmodelled) {
                    weights.insert(slot, b);
                }
            }
        }
        let hd = config.head_dim as u32;
        let facts = AttnFacts {
            attn_layer_idx: self.attn_layer_idx,
            kv_dtype: if self.kv_bf16 {
                KvCacheDtype::Bf16
            } else {
                KvCacheDtype::Fp8
            },
            num_q_heads: config.num_attention_heads as u32,
            num_kv_heads: config.num_key_value_heads as u32,
            head_dim: hd,
            gated: true,
            rope: RopeFacts {
                mrope_interleaved: false,
                theta: config.rope_theta as f32,
                rotary_dim: config.rotary_dim() as u32,
            },
            sliding_window: 0,
            // 2026-09-29: `forward_one`'s `inv_sqrt_d`.
            softmax_scale: 1.0 / (hd as f32).sqrt(),
            // 2026-09-29: One row on `paged_decode_k` (`forward/attend.rs`). 2026-09-30: n rows
            // on it too (`forward_batch_position`): the head has no split-K arm.
            paged_decode_plain_rows: u128::MAX,
        };
        let cache = self.kv_cache.lock();
        DraftBinding {
            layer: CircuitLayer {
                mixer: MixerFacts::Attention(facts),
                weights,
                unmodelled,
            },
            k_pool: cache.k_pool_ptr(self.attn_layer_idx),
            v_pool: cache.v_pool_ptr(self.attn_layer_idx),
            block_size: cache.block_size() as u32,
            cache_stride: cache.cache_stride() as u64,
            // 2026-09-29: `forward_one`'s `v`.
            vocab: if self.mtp_vocab_size > 0 {
                self.mtp_vocab_size.min(config.vocab_size as u32)
            } else {
                config.vocab_size as u32
            },
            rows: self
                .propose_batch_scope_ok()
                .then(|| crate::circuit_exec::DraftRows {
                    meta: self.propose_meta,
                    lp_offset: super::forward_batch::LP_SCRATCH_OFF,
                    lm_head_gemv: (0..=super::batch_caps::PROPOSE_META_SEQS)
                        .map(|n| self.lm_head_batch_kernel(n))
                        .collect(),
                    lm_head_twin: self.lm_head_twin_ready(),
                }),
        }
    }

    /// 2026-09-29: Install or remove the runner `forward_one` hands its step to.
    pub(super) fn install_circuit(&self, runner: Option<Arc<dyn DraftRunner>>) {
        *self.circuit_draft.write() = runner;
    }
}

impl MtpHead {
    /// 2026-09-29: `forward_one` on the installed draft program: the host steps (the token's
    /// embedding into the program's embedding buffer, the target hidden into the stream
    /// buffer, the step's metadata), the program (the layer, the final norm, the lm_head and
    /// the argmax), then the token as `forward_one` returns it.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_one_circuit(
        &self,
        runner: &dyn DraftRunner,
        token: u32,
        target_hidden: metrale_gpu_runtime::gpu::DevicePtr,
        position: usize,
        state: &mut super::MtpProposerState,
        ctx: &crate::layer::ForwardContext,
        stream: u64,
        draft_embed_target: Option<metrale_gpu_runtime::gpu::DevicePtr>,
    ) -> anyhow::Result<u32> {
        let row_bytes = ctx.config.hidden_size * 2;
        let src = self.embed_tokens.weight.offset(token as usize * row_bytes);
        ctx.gpu
            .copy_d2d_async(src, ctx.buffers.ssm_qkvz(), row_bytes, stream)?;
        // 2026-09-29: The program reads the target hidden from the stream buffer; a chained
        // draft's target already is it.
        let hidden = ctx.buffers.hidden_states();
        if target_hidden != hidden {
            ctx.gpu
                .copy_d2d_async(target_hidden, hidden, row_bytes, stream)?;
        }
        let max_blocks = {
            let mut kv_cache = self.kv_cache.lock();
            self.upload_draft_meta(&mut kv_cache, state, position, ctx, stream)?
                .1
        };
        runner.run_draft(ctx.gpu, stream, 1, max_blocks)?;
        let token_id = self.draft_token(ctx, ctx.buffers.scratch(), draft_embed_target, stream)?;
        Self::finish_row(state, position);
        Ok(token_id)
    }
}
