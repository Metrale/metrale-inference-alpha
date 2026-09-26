// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Whole-request generate loops for self-speculative decoding and
//! for decoding with a draft proposer: prefill, then per step a decode, the
//! drafts, one verify and the rollback of rejected tokens.
//!
//! Owner: model-engine.
//! Invariants: none beyond the types.

#![allow(unused_imports, dead_code)]

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, bail};
use metrale_cache::kv_cache::PagedKvCache;
use metrale_config::{LayerType, ModelConfig};
use metrale_gpu_runtime::buffers::BufferArena;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle, KernelHandle};

use super::block_mgmt::{
    apply_evicted_blocks, ensure_blocks_through_decode, ensure_blocks_through_prefill,
    extract_layer_refs, reuse_prefix_match_disk_ids,
};
use super::ssm_pool::SsmStatePool;
use super::ssm_snapshot::SsmSnapshotPool;
use super::types::{PinnedMetaStaging, TransformerModel};
use crate::traits::{ChunkedPrefillPageMetadata, Model, SequenceState};
use crate::traits::{ModelForward, ModelLogits, ModelSsmState, ModelVerify};
use metrale_model_layers::layer::{
    AttnMetadataDev, ForwardContext, GdnPrefillBuffers, LayerState, SsmLayerState, TransformerLayer,
};
use metrale_model_layers::layers::ops;
use metrale_model_layers::speculative::DraftProposer;
use metrale_model_layers::weight_map::{DenseWeight, MtpWeights, QuantizedWeight};

impl TransformerModel {
    pub(super) fn generate_self_speculative_inner(
        &self,
        prompt_tokens: &[u32],
        params: &metrale_sampling::SamplingParams,
        num_drafts: usize,
        seq: &mut SequenceState,
        stream: u64,
    ) -> Result<crate::engine::GenerateResult> {
        let logits_ptr = self.prefill(prompt_tokens, seq, stream)?;
        let first_token = self.argmax_on_device(logits_ptr, stream)?;

        let mut output_tokens = Vec::with_capacity(params.max_tokens);
        output_tokens.push(first_token);

        if params.stop_token_ids.contains(&first_token) {
            return Ok(crate::engine::GenerateResult {
                output_tokens,
                finish_reason: "stop".to_string(),
            });
        }

        let mut total_accepted = 0usize;
        let mut total_proposed = 0usize;
        let mut total_steps = 0usize;

        while output_tokens.len() < params.max_tokens {
            let last_token = *output_tokens.last().unwrap();

            let logits = self.decode(last_token, seq, stream)?;
            let token_0 = self.argmax_on_device(logits, stream)?;

            let seq_len_before_draft = seq.seq_len;
            let tokens_before_draft = seq.tokens.len();

            let mut draft_tokens = Vec::with_capacity(num_drafts);
            let mut draft_token = token_0;
            for _ in 0..num_drafts {
                let logits = self.decode_draft(draft_token, seq, stream)?;
                draft_token = self.argmax_on_device(logits, stream)?;
                draft_tokens.push(draft_token);
            }

            // 2026-09-25: Rewind the sequence to its pre-draft length; the SSM
            // state needs no rewind because drafting skipped the SSM layers.
            seq.seq_len = seq_len_before_draft;
            seq.tokens.truncate(tokens_before_draft);

            let mut verify_tokens = vec![token_0];
            verify_tokens.extend_from_slice(&draft_tokens);

            self.checkpoint_ssm_states(seq)?;
            let seq_len_before_verify = seq.seq_len;

            let verified = self.decode_verify(&verify_tokens, seq, stream)?;

            // 2026-09-25: Accept drafts up to the first mismatch, which is
            // replaced by the verified token; a full accept adds the bonus token.
            output_tokens.push(token_0);
            let n_drafts = draft_tokens.len();
            let mut num_accepted = 0;

            for i in 0..n_drafts {
                if draft_tokens[i] == verified[i] {
                    output_tokens.push(draft_tokens[i]);
                    num_accepted += 1;
                } else {
                    output_tokens.push(verified[i]);
                    break;
                }
            }

            if num_accepted == n_drafts && n_drafts > 0 {
                output_tokens.push(verified[n_drafts]);
            }

            total_accepted += num_accepted;
            total_proposed += n_drafts;
            total_steps += 1;

            // 2026-09-25: The verify appended every verify token; keep token_0
            // and the accepted drafts, and drop the rest.
            let tokens_added = 1 + num_accepted;
            let expected_seq_len = seq_len_before_verify + tokens_added;

            if seq.seq_len > expected_seq_len {
                let extra = seq.seq_len - expected_seq_len;
                for _ in 0..extra {
                    seq.seq_len -= 1;
                    seq.tokens.pop();
                }
                // 2026-09-25: +1 because token_0 is always kept.
                self.rollback_ssm_states(seq, num_accepted + 1)?;
            }

            if let Some(last) = output_tokens.last()
                && params.stop_token_ids.contains(last)
            {
                break;
            }
        }

        output_tokens.truncate(params.max_tokens);

        if total_steps > 0 {
            tracing::info!(
                "Self-speculative decode: {} steps, {}/{} accepted ({:.0}%)",
                total_steps,
                total_accepted,
                total_proposed,
                if total_proposed > 0 {
                    total_accepted as f64 / total_proposed as f64 * 100.0
                } else {
                    0.0
                },
            );
        }

        let finish_reason = if output_tokens
            .last()
            .is_some_and(|t| params.stop_token_ids.contains(t))
        {
            "stop".to_string()
        } else {
            "length".to_string()
        };
        Ok(crate::engine::GenerateResult {
            output_tokens,
            finish_reason,
        })
    }

    pub(super) fn generate_speculative_inner(
        &self,
        prompt_tokens: &[u32],
        params: &metrale_sampling::SamplingParams,
        num_drafts: usize,
        proposer: &Arc<dyn DraftProposer>,
        seq: &mut SequenceState,
        stream: u64,
    ) -> Result<crate::engine::GenerateResult> {
        let mut prop_state = proposer.alloc_state(self.gpu.as_ref())?;

        let logits_ptr = self.prefill(prompt_tokens, seq, stream)?;
        let first_token = self.argmax_on_device(logits_ptr, stream)?;

        let mut output_tokens = Vec::with_capacity(params.max_tokens);
        output_tokens.push(first_token);

        if params.stop_token_ids.contains(&first_token) {
            return Ok(crate::engine::GenerateResult {
                output_tokens,
                finish_reason: "stop".to_string(),
            });
        }

        let mut total_accepted = 0usize;
        let mut total_proposed = 0usize;
        let mut total_steps = 0usize;

        while output_tokens.len() < params.max_tokens {
            let last_token = *output_tokens.last().unwrap();

            let logits = self.decode(last_token, seq, stream)?;
            let token_0 = self.argmax_on_device(logits, stream)?;

            let target_hidden = self.hidden_after_norm();
            let position = seq.seq_len;
            let ctx = ForwardContext {
                buffers: &self.buffers,
                hc_row_offset: 0,
                gpu: self.gpu.as_ref(),
                config: &self.config,
                dispatch: &self.dispatch,
                derived: &self.derived,
                levers: &self.levers,
                stats: &self.stats,
                attn_metadata: None,
                profile: false,
                comm: None,
                graph_capture: false,
                decode_step: false,
                gdn_exact_replay: false,
                gdn_write_on_accept: false,
                token_ids: None,
                host_token_ids: None,
                routed_lora_layers: None,
                midchunk_capture: None,
                moe_lora_route: self.decode_moe_route(),
            };
            let drafts = proposer.propose(
                token_0,
                target_hidden,
                position,
                num_drafts,
                prop_state.as_mut(),
                &ctx,
                stream,
                None,
                None,
                self.dflash_hidden_save,
            )?;
            let n_drafts = drafts.len();

            let mut verify_tokens = vec![token_0];
            verify_tokens.extend_from_slice(&drafts);

            self.checkpoint_ssm_states(seq)?;
            let seq_len_before = seq.seq_len;

            let verified = self.decode_verify(&verify_tokens, seq, stream)?;

            output_tokens.push(token_0);
            let mut num_accepted = 0;

            for i in 0..n_drafts {
                if drafts[i] == verified[i] {
                    output_tokens.push(drafts[i]);
                    num_accepted += 1;
                } else {
                    output_tokens.push(verified[i]);
                    break;
                }
            }

            if num_accepted == n_drafts && n_drafts > 0 {
                output_tokens.push(verified[n_drafts]);
            }

            total_accepted += num_accepted;
            total_proposed += n_drafts;
            total_steps += 1;

            let tokens_added = 1
                + num_accepted
                + if num_accepted == n_drafts && n_drafts > 0 {
                    1
                } else {
                    0
                };
            let expected_seq_len = seq_len_before + tokens_added;

            if seq.seq_len > expected_seq_len {
                let extra = seq.seq_len - expected_seq_len;
                for _ in 0..extra {
                    seq.seq_len -= 1;
                    seq.tokens.pop();
                }
                // 2026-09-25: The verify batch starts with token_0, which is
                // always kept. `rollback_ssm_states(seq, n)` restores
                // intermediate `n - 1`, the state after verify token `n - 1`, so
                // `num_accepted + 1` gives the state after token_0 and the
                // accepted drafts.
                self.rollback_ssm_states(seq, num_accepted + 1)?;
            }

            proposer.after_verify(num_accepted, prop_state.as_mut(), stream)?;

            if let Some(last) = output_tokens.last()
                && params.stop_token_ids.contains(last)
            {
                if total_steps > 0 {
                    tracing::info!(
                        "Speculative decode: {} steps, {}/{} accepted ({:.0}%)",
                        total_steps,
                        total_accepted,
                        total_proposed,
                        if total_proposed > 0 {
                            total_accepted as f64 / total_proposed as f64 * 100.0
                        } else {
                            0.0
                        },
                    );
                }
                return Ok(crate::engine::GenerateResult {
                    output_tokens,
                    finish_reason: "stop".to_string(),
                });
            }
        }

        output_tokens.truncate(params.max_tokens);

        if total_steps > 0 {
            tracing::info!(
                "Speculative decode: {} steps, {}/{} accepted ({:.0}%)",
                total_steps,
                total_accepted,
                total_proposed,
                if total_proposed > 0 {
                    total_accepted as f64 / total_proposed as f64 * 100.0
                } else {
                    0.0
                },
            );
        }

        Ok(crate::engine::GenerateResult {
            output_tokens,
            finish_reason: "length".to_string(),
        })
    }
}
