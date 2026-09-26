// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Multi-sequence decode attention after the projections: RoPE,
//! the KV-cache write and the batched paged decode. The O projection is in `o_proj`.
//!
//! Owner: model-layers (qwen3 attention).
//! Invariants:
//! - `qkv_buf` is read, never written, by the paged decode: when Q must be
//!   rotated for TurboQuant it is first copied to a staging buffer.

use anyhow::Result;
use metrale_cache::kv_cache::PagedKvCache;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::ctx::MultiSeqCtx;
use crate::layer::AttnMetadataDev;
use crate::layers::ops;
use crate::layers::qwen3_attention::Qwen3AttentionLayer;

/// 2026-09-25: The batched KV-cache write (one launch for all rows) is on unless
/// `METRALE_NO_ATTN_BATCH_CACHE_WRITE=1`, read once per process.
fn batch_cache_write_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("METRALE_NO_ATTN_BATCH_CACHE_WRITE")
            .ok()
            .as_deref()
            != Some("1")
    })
}

impl Qwen3AttentionLayer {
    /// 2026-09-25: RoPE on each sequence's Q and K row, at that sequence's position.
    pub(super) fn ms_phase_rope(&self, c: &MultiSeqCtx<'_>, meta: AttnMetadataDev) -> Result<()> {
        let MultiSeqCtx {
            fwd,
            n,
            stream,
            nq,
            nkv,
            hd,
            bf16,
            q_proj_bytes,
            per_seq_qkv,
            qkv_buf,
            ..
        } = *c;
        // 2026-09-25: One launch for all n rows when `rope_forward_strided` is
        // loaded. The packed `rope_forward` assumes rows `num_heads * head_dim`
        // apart; these rows are `per_seq_qkv` apart, so the fallback loop calls it
        // once per row. The strided kernel has the same per-element math as the
        // packed one (`kernels/gb10/common/rope.cu`), so the results are
        // bit-identical. Off when `METRALE_NO_ROPE_STRIDED=1`, read once per process.
        fn rope_strided_enabled() -> bool {
            static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *ON.get_or_init(|| {
                std::env::var("METRALE_NO_ROPE_STRIDED").ok().as_deref() != Some("1")
            })
        }
        // 2026-09-25: A layer with a YaRN table (`yarn_inv_freq`, set by
        // `set_yarn_rope`) takes the loop: `rope_forward_strided` computes
        // plain-theta RoPE and has no inverse-frequency table argument.
        if n > 1
            && self.rope_strided_k.0 != 0
            && rope_strided_enabled()
            && self.yarn_inv_freq.is_null()
        {
            let stride_e = (per_seq_qkv / bf16) as u32;
            return ops::rope_strided(
                fwd.gpu,
                self.rope_strided_k,
                qkv_buf,
                qkv_buf.offset(q_proj_bytes),
                meta.positions,
                n as u32,
                nq,
                nkv,
                hd,
                self.rotary_dim_override
                    .unwrap_or(fwd.config.rotary_dim() as u32),
                self.rope_theta_override
                    .unwrap_or(fwd.config.rope_theta as f32),
                stride_e,
                stride_e,
                stream,
            );
        }
        for i in 0..n {
            let q_out_i = qkv_buf.offset(i * per_seq_qkv);
            let k_out_i = q_out_i.offset(q_proj_bytes);
            let pos_i = meta.positions.offset(i * 4); // 2026-09-25: u32 positions
            if self.yarn_inv_freq.is_null() {
                ops::rope(
                    fwd.gpu,
                    self.rope_k,
                    q_out_i,
                    k_out_i,
                    pos_i,
                    1,
                    nq,
                    nkv,
                    hd,
                    self.rotary_dim_override
                        .unwrap_or(fwd.config.rotary_dim() as u32),
                    self.rope_theta_override
                        .unwrap_or(fwd.config.rope_theta as f32),
                    stream,
                )?;
            } else {
                ops::rope_yarn_scaled(
                    fwd.gpu,
                    self.rope_yarn_scaled_k,
                    q_out_i,
                    k_out_i,
                    pos_i,
                    1,
                    nq,
                    nkv,
                    hd,
                    self.rotary_dim_override
                        .unwrap_or(fwd.config.rotary_dim() as u32),
                    self.yarn_inv_freq,
                    self.yarn_attention_factor,
                    stream,
                )?;
            }
        }
        Ok(())
    }

    /// 2026-09-25: Write each sequence's K and V row into the paged cache.
    pub(super) fn ms_phase_cache_write(
        &self,
        c: &MultiSeqCtx<'_>,
        kv_cache: &mut PagedKvCache,
        meta: AttnMetadataDev,
    ) -> Result<()> {
        let MultiSeqCtx {
            fwd,
            n,
            stream,
            nkv,
            hd,
            bs,
            bf16,
            q_proj_bytes,
            per_seq_qkv,
            qkv_buf,
            ..
        } = *c;
        let kv_stride = nkv * hd;
        // 2026-09-25: The `reshape_and_cache` kernels take explicit key and value
        // row strides and run one block per token (`reshape_and_cache.cu`), and
        // `meta.slot` holds one slot per sequence, so all n rows go in one launch.
        // Consecutive K rows are `per_seq_qkv` bytes apart; the stride is that gap
        // in elements. `METRALE_NO_ATTN_BATCH_CACHE_WRITE=1` selects the loop.
        let k_out_0 = qkv_buf.offset(q_proj_bytes);
        let v_out_0 = k_out_0.offset((nkv * hd) as usize * bf16);
        if n > 1 && batch_cache_write_enabled() && per_seq_qkv.is_multiple_of(bf16) {
            let row_stride = (per_seq_qkv / bf16) as u32;
            return self.write_kv_cache(
                fwd.gpu,
                k_out_0,
                v_out_0,
                kv_cache,
                meta.slot,
                n as u32,
                nkv,
                hd,
                bs,
                row_stride,
                row_stride,
                stream,
                fwd.graph_capture,
            );
        }
        for i in 0..n {
            let q_out_i = qkv_buf.offset(i * per_seq_qkv);
            let k_out_i = q_out_i.offset(q_proj_bytes);
            let v_out_i = k_out_i.offset((nkv * hd) as usize * bf16);
            let slot_i = meta.slot.offset(i * 8); // 2026-09-25: i64 slots
            self.write_kv_cache(
                fwd.gpu,
                k_out_i,
                v_out_i,
                kv_cache,
                slot_i,
                1,
                nkv,
                hd,
                bs,
                kv_stride,
                kv_stride,
                stream,
                fwd.graph_capture,
            )?;
        }
        Ok(())
    }

    /// 2026-09-25: Batched paged decode attention for all n rows. Returns the
    /// `attn_output` buffer, `[n, nq * hd]`.
    pub(super) fn ms_phase_paged_decode(
        &self,
        c: &MultiSeqCtx<'_>,
        kv_cache: &mut PagedKvCache,
        meta: AttnMetadataDev,
    ) -> Result<DevicePtr> {
        let MultiSeqCtx {
            fwd,
            n,
            stream,
            nq,
            nkv,
            hd,
            bs,
            bf16,
            q_dim,
            per_seq_qkv,
            qkv_buf,
            ..
        } = *c;
        // 2026-09-25: TurboQuant bookends, as in `decode/attention_forward.rs`. For
        // a WHT-rotated cache dtype the cache holds rotated K and V, so Q is
        // rotated before the paged decode and the output is rotated back after
        // it. The rotations run in place on Q, which is why Q is then staged.
        let (wht_k_dtype, wht_v_dtype) = self.kv_dtype.kv_pair();
        let k_is_turbo = wht_k_dtype.is_wht_rotated();
        let v_is_turbo = wht_v_dtype.is_wht_rotated();

        // 2026-09-25: The paged decode kernels read `Q + seq_idx * q_stride`
        // (`paged_decode_attn.cu`, both the plain and the split-K kernel), so
        // when nothing rewrites Q they read it in place from `qkv_buf` with
        // `q_stride = per_seq_qkv / 2`, and no copy is made. Same kernel, same
        // values: only the addressing differs. Under TurboQuant Q is copied to
        // `ssm_qkvz` first, because the bookends rotate it in place. The in-place
        // read is off when `METRALE_NO_ATTN_Q_INPLACE=1`, read once per process.
        fn q_inplace_enabled() -> bool {
            static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *ON.get_or_init(|| {
                std::env::var("METRALE_NO_ATTN_Q_INPLACE").ok().as_deref() != Some("1")
            })
        }
        let q_inplace =
            !k_is_turbo && !v_is_turbo && q_inplace_enabled() && per_seq_qkv.is_multiple_of(bf16);
        let q_contiguous = if q_inplace {
            qkv_buf
        } else {
            let staged = fwd.buffers.ssm_qkvz();
            for i in 0..n {
                let q_out_i = qkv_buf.offset(i * per_seq_qkv);
                fwd.gpu.copy_d2d_async(
                    q_out_i,
                    staged.offset(i * q_dim as usize * bf16),
                    q_dim as usize * bf16,
                    stream,
                )?;
            }
            staged
        };
        let q_stride = if q_inplace {
            (per_seq_qkv / bf16) as u32
        } else {
            nq * hd
        };
        let attn_out = fwd.buffers.attn_output();
        let inv_sqrt_d = self.effective_attn_scale(hd);
        let weight_pre_rotated = crate::layers::ops::ModelLevers::get().weight_pre_rotated;
        let wht_runtime_active = !weight_pre_rotated && (hd == 128 || hd == 256 || hd == 512);
        if k_is_turbo && self.innerq_apply_q_k.0 != 0 && hd == 128 {
            use metrale_gpu_runtime::kernel_args::KernelLaunch;
            KernelLaunch::new(fwd.gpu, self.innerq_apply_q_k)
                .grid([n as u32 * nq, 1, 1])
                .block([32, 1, 1])
                .arg_ptr(q_contiguous)
                .arg_u32(hd)
                .launch(stream)?;
        }
        if k_is_turbo && wht_runtime_active && self.wht_bf16_k.0 != 0 {
            use metrale_gpu_runtime::kernel_args::KernelLaunch;
            KernelLaunch::new(fwd.gpu, self.wht_bf16_k)
                .grid([n as u32 * nq, 1, 1]) // 2026-09-25: one warp per (seq, q_head)
                .block([32, 1, 1])
                .arg_ptr(q_contiguous)
                .arg_u32(hd)
                .launch(stream)?;
        }
        self.run_paged_decode(
            fwd.gpu,
            q_contiguous,
            kv_cache,
            attn_out,
            meta.block_table,
            meta.seq_len,
            meta.max_blocks_per_seq,
            n as u32,
            nq,
            nkv,
            hd,
            bs,
            inv_sqrt_d,
            q_stride,
            fwd.buffers.splitk_workspace(),
            fwd.levers.max_decode_seqs,
            stream,
        )?;
        if v_is_turbo && wht_runtime_active && self.wht_bf16_k_inv.0 != 0 {
            use metrale_gpu_runtime::kernel_args::KernelLaunch;
            KernelLaunch::new(fwd.gpu, self.wht_bf16_k_inv)
                .grid([n as u32 * nq, 1, 1])
                .block([32, 1, 1])
                .arg_ptr(attn_out)
                .arg_u32(hd)
                .launch(stream)?;
        }
        Ok(attn_out)
    }
}

mod o_proj;
