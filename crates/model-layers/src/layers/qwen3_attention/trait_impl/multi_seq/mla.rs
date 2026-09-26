// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: MLA (multi-head latent attention) for multi-sequence batched
//! decode: the absorbed-MLA chain of `decode::attention_forward_mla`, run once
//! per sequence inside one forward pass. The layer's standard projections are
//! null placeholders on MLA layers, so MLA must not take `ms_phase_qkv`.
//!
//! Owner: model-layers (qwen3 attention).
//! Invariants:
//! - Iteration `i` reads `normed` row `i` and writes `o_out` row `i`, and uses
//!   sequence `i`'s metadata: `positions[i]` (u32), `slot[i]` (i64),
//!   `seq_len[i]` (i32) and block-table row `i`. So each sequence reads and
//!   writes only its own latent-KV history.
//! - The scratch buffers (`ssm_ba`, `ssm_deinterleaved`, `expert_up_out`, ...)
//!   are shared by all iterations; every launch is on `c.stream`, so each
//!   iteration's reads of them are ordered after its own writes.
//!
//! The paged decode kernel (`paged_decode_attn_bf16`) can take several
//! sequences (grid `[num_q_heads, num_seqs, 1]`), but it runs once per
//! sequence here, because the absorbed Q is built in scratch that the next
//! iteration overwrites.

use anyhow::Result;
use metrale_cache::kv_cache::PagedKvCache;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::ctx::MultiSeqCtx;
use super::mla_gemv::MlaDims;
use crate::layer::AttnMetadataDev;
use crate::layers::qwen3_attention::Qwen3AttentionLayer;

mod decode_one;

impl Qwen3AttentionLayer {
    /// 2026-09-25: MLA decode for `c.n` sequences. Writes sequence `i`'s O
    /// projection to `moe_output[i*h .. (i+1)*h]` and returns the `moe_output`
    /// base pointer.
    ///
    /// The caller has already written the normalised input rows to `c.normed`.
    pub(super) fn ms_mla_decode(
        &self,
        c: &MultiSeqCtx<'_>,
        kv_cache: &mut PagedKvCache,
        meta: AttnMetadataDev,
    ) -> Result<DevicePtr> {
        let mla = self
            .mla
            .as_ref()
            .expect("ms_mla_decode called without MLA config");

        let h = c.h as u32;
        let nq = c.nq;
        let hd = c.hd;
        let eps = c.eps;
        let bf16 = c.bf16;
        let stream = c.stream;
        let bs = c.bs as usize;

        let q_lora = mla.q_lora_rank as u32;
        let kv_lora = mla.kv_lora_rank as u32;
        let mla_nope = mla.nope as u32;
        let mla_v_dim = mla.v_dim as u32;
        let mla_rope = mla.rope as u32;
        let mla_cache_dim = kv_lora + mla_rope;
        let q_dim = nq * hd;
        let inv_sqrt_d = self.effective_attn_scale(hd);

        // 2026-09-25: `ms_phase_o_proj` (the non-MLA path) returns `moe_output`;
        // this path writes the same buffer, so `ms_phase_ffn` reads one buffer.
        let o_out = c.fwd.buffers.moe_output();

        for i in 0..c.n {
            let normed_i = c.normed.offset(i * c.h * bf16);
            // 2026-09-25: Sequence `i`'s view of the batched metadata: positions
            // `[n]` u32, slot `[n]` i64, seq_len `[n]` i32, block_table
            // `[n * max_blocks_per_seq]` i32, as `ms_phase_rope` and
            // `ms_phase_cache_write` index them.
            let meta_i = AttnMetadataDev {
                positions: meta.positions.offset(i * 4),
                positions_h: meta.positions_h.offset(i * 4),
                positions_w: meta.positions_w.offset(i * 4),
                slot: meta.slot.offset(i * 8),
                seq_len: meta.seq_len.offset(i * 4),
                block_table: meta
                    .block_table
                    .offset(i * meta.max_blocks_per_seq as usize * 4),
                max_blocks_per_seq: meta.max_blocks_per_seq,
                num_seqs: 1,
                seq_slot: metrale_gpu_runtime::gpu::DevicePtr(0),
                moe_row_adapter: metrale_gpu_runtime::gpu::DevicePtr::NULL,
            };
            let o_out_i = o_out.offset(i * c.h * bf16);

            // 2026-09-25: DeepSeek-V4-Flash (`o_lora_rank > 0`) does not use the
            // absorbed chain: its loader (`deepseek_v4/load_layers.rs`,
            // `is_v4_flash`) sets `w_uk_t`, `w_uv` and `wkv_b` to null. Each row
            // runs the single-token `attention_forward_v4` instead.
            if mla.o_lora_rank > 0 {
                // 2026-09-25: The forward context with this row's metadata; the
                // other fields are copied.
                let ctx_i = crate::layer::ForwardContext {
                    attn_metadata: Some(meta_i),
                    midchunk_capture: None,
                    ..*c.fwd
                };
                // 2026-09-25: Q, K and V destinations in `qkv_output`: Q
                // `[nq * hd]`, then K and V `[nkv * hd]` each. The V4 layer is
                // built with `new_ungated`, so Q is `q_dim` wide.
                let qkv = c.fwd.buffers.qkv_output();
                let q_proj_bytes = q_dim as usize * bf16;
                let kv_bytes = (c.nkv * hd) as usize * bf16;
                let k_out = qkv.offset(q_proj_bytes);
                let v_out = k_out.offset(kv_bytes);
                let args = super::super::super::decode::attention_forward_mla::DecodeMlaArgs {
                    normed: normed_i,
                    q_out: qkv,
                    k_out,
                    v_out,
                    q_dim,
                    h,
                    nq,
                    hd,
                    eps,
                    bs,
                    stream,
                    // 2026-09-25: `None` skips the compressed-pool append, which
                    // only the single-sequence decode path drives (`DecodeMlaArgs::pos`).
                    pos: None,
                };
                let o_v4 = self.attention_forward_v4(kv_cache, &ctx_i, &args)?;
                // 2026-09-25: `attention_forward_v4` returns `qkv_output`, which
                // the next row reuses, so its output row is copied out now.
                c.fwd
                    .gpu
                    .copy_d2d_async(o_v4, o_out_i, c.h * bf16, stream)?;
                continue;
            }

            self.ms_mla_decode_one(
                c,
                kv_cache,
                &meta_i,
                normed_i,
                o_out_i,
                mla,
                MlaDims {
                    h,
                    nq,
                    hd,
                    q_dim,
                    q_lora,
                    kv_lora,
                    mla_nope,
                    mla_v_dim,
                    mla_rope,
                    mla_cache_dim,
                    eps,
                    bs,
                    inv_sqrt_d,
                    o_lora_rank: mla.o_lora_rank as u32,
                },
                stream,
            )?;
        }

        // 2026-09-25: `METRALE_MLA_HSD=1` diagnostic, layer 0 only: logs each
        // row's count of non-finite values and its max |x|.
        if std::env::var("METRALE_MLA_HSD").is_ok_and(|v| v == "1") && self.attn_layer_idx == 0 {
            c.fwd.gpu.synchronize(stream)?;
            for i in 0..c.n {
                let mut row = vec![0u8; c.h * bf16];
                let _ = c.fwd.gpu.copy_d2h(o_out.offset(i * c.h * bf16), &mut row);
                let vals: Vec<f32> = row
                    .chunks_exact(2)
                    .map(|x| f32::from_bits((u16::from_le_bytes([x[0], x[1]]) as u32) << 16))
                    .collect();
                let bad = vals.iter().filter(|v| !v.is_finite()).count();
                let absmax = vals.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                tracing::info!(
                    "MLA_HSD L0 s{i}: o_out non-finite={bad}/{} absmax={absmax:.4}",
                    vals.len(),
                );
            }
        }
        Ok(o_out)
    }
}
