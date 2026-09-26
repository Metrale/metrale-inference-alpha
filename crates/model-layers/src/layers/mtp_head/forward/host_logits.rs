// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Host-side reads of the drafter logits in `MtpHead::forward_one`: the draft
//! confidence fold, the shadow top-k log and the grammar-masked argmax.
//!
//! Owner: model-layers (MTP head).
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::MtpHead;
use crate::layer::ForwardContext;
use crate::layers::ops;

impl MtpHead {
    pub(super) fn fold_draft_conf(&self, ctx: &ForwardContext, logits: DevicePtr, v: u32) {
        let vocab = v as usize;
        let mut bf16_buf = vec![0u8; vocab * 2];
        if ctx.gpu.copy_d2h(logits, &mut bf16_buf).is_ok() {
            let mut max = f32::NEG_INFINITY;
            for i in 0..vocab {
                let hi = u16::from_le_bytes([bf16_buf[2 * i], bf16_buf[2 * i + 1]]);
                let x = f32::from_bits((hi as u32) << 16);
                if x > max {
                    max = x;
                }
            }
            let mut denom = 0.0f64;
            for i in 0..vocab {
                let hi = u16::from_le_bytes([bf16_buf[2 * i], bf16_buf[2 * i + 1]]);
                let x = f32::from_bits((hi as u32) << 16);
                denom += ((x - max) as f64).exp();
            }
            let top1 = (1.0 / denom.max(1.0)) as f32;
            let cur = f32::from_bits(
                self.last_conf_bits
                    .load(std::sync::atomic::Ordering::Relaxed),
            );
            if top1 < cur {
                self.last_conf_bits
                    .store(top1.to_bits(), std::sync::atomic::Ordering::Relaxed);
            }
        }
    }

    pub(super) fn grammar_masked_argmax(
        &self,
        ctx: &ForwardContext,
        logits: DevicePtr,
        v: u32,
        bitmask: &[i32],
        out_ptr: DevicePtr,
        draft_embed_target: Option<DevicePtr>,
        h: u32,
        position: usize,
        stream: u64,
    ) -> Result<u32> {
        // 2026-09-25: Grammar-masked argmax on the host: copy the logits back, set
        // every token whose bitmask bit is clear to -inf, and take the
        // argmax.
        let vocab = v as usize;
        let mut bf16_buf = vec![0u8; vocab * 2];
        ctx.gpu.copy_d2h(logits, &mut bf16_buf)?;

        let mut f32_logits = vec![0.0f32; vocab];
        for i in 0..vocab {
            let lo = 0u16;
            let hi = u16::from_le_bytes([bf16_buf[2 * i], bf16_buf[2 * i + 1]]);
            f32_logits[i] = f32::from_bits(((hi as u32) << 16) | (lo as u32));
        }

        let mut any_allowed = false;
        for tok in 0..vocab {
            let word = tok / 32;
            let bit = tok % 32;
            let allowed = word < bitmask.len() && (bitmask[word] & (1i32 << bit)) != 0;
            if allowed {
                any_allowed = true;
            } else {
                f32_logits[tok] = f32::NEG_INFINITY;
            }
        }

        // 2026-09-25: With no allowed token, return 0 as a placeholder draft and
        // stage no embedding.
        Ok(if !any_allowed {
            tracing::warn!(
                target: "metrale_model_layers::layers::mtp_head::forward",
                "MTP grammar mask allowed zero tokens at pos {position}; \
                 returning 0 as pad-draft (will be rejected at verify)."
            );
            0u32
        } else {
            let mut best_tok = 0u32;
            let mut best_val = f32::NEG_INFINITY;
            for (i, &v) in f32_logits.iter().enumerate() {
                if v > best_val {
                    best_val = v;
                    best_tok = i as u32;
                }
            }

            // 2026-09-25: `embed_from_argmax` reads its token id from `out_ptr`, so
            // the host-chosen id is copied there first.
            if let Some(embed_target) = draft_embed_target {
                let tok_bytes = best_tok.to_le_bytes();
                ctx.gpu.copy_h2d(&tok_bytes, out_ptr)?;
                ops::embed_from_argmax(
                    ctx.gpu,
                    self.embed_from_argmax_k,
                    out_ptr,
                    self.embed_tokens.weight,
                    embed_target,
                    self.draft_token_id_dev,
                    h,
                    stream,
                )?;
            }
            best_tok
        })
    }
}

pub(super) fn log_shadow_topk(
    ctx: &ForwardContext,
    logits: DevicePtr,
    v: u32,
    shadow_k: usize,
    position: usize,
) {
    let vocab = v as usize;
    let mut bf16_buf = vec![0u8; vocab * 2];
    if ctx.gpu.copy_d2h(logits, &mut bf16_buf).is_ok() {
        let at = |i: usize| -> f32 {
            let hi = u16::from_le_bytes([bf16_buf[2 * i], bf16_buf[2 * i + 1]]);
            f32::from_bits((hi as u32) << 16)
        };
        let mut top: Vec<(f32, usize)> = Vec::with_capacity(shadow_k + 1);
        for i in 0..vocab {
            let x = at(i);
            if top.len() < shadow_k || x > top.last().map(|t| t.0).unwrap_or(f32::MIN) {
                let pos = top.partition_point(|t| t.0 >= x);
                top.insert(pos, (x, i));
                top.truncate(shadow_k);
            }
        }
        let max = top.first().map(|t| t.0).unwrap_or(0.0);
        let mut denom = 0.0f64;
        for i in 0..vocab {
            denom += ((at(i) - max) as f64).exp();
        }
        let ids: Vec<usize> = top.iter().map(|t| t.1).collect();
        let probs: Vec<f32> = top
            .iter()
            .map(|t| (((t.0 - max) as f64).exp() / denom.max(1e-30)) as f32)
            .collect();
        tracing::info!(
            target: "metrale_model_layers::layers::mtp_head::forward",
            "SHADOW_TOPK pos={position} ids={ids:?} probs={probs:?}"
        );
    }
}
