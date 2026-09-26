// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Conv1d + L2 norm + GDN over the rows of one sequence
//! (`decode_batched_conv_gdn`, called from `decode_batched_inner`). The WY kernel
//! selectors it shares with the cross-sequence batched verify are in
//! `trait_decode_batched_conv_gdn/wy_select.rs`.
//!
//! Owner: model-layers (Qwen3 SSM layer).
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::{Qwen3SsmLayer, SsmLayerState};
use crate::layer::ForwardContext;
use crate::layers::ops;

mod wy_select;

#[allow(clippy::too_many_arguments)]
pub(super) struct ConvGdnArgs {
    pub num_tokens: usize,
    pub deinterleaved: DevicePtr,
    pub gates_buf: DevicePtr,
    pub conv_out_buf: DevicePtr,
    pub gdn_out_buf: DevicePtr,
    /// 2026-09-25: Where this call's normed rows go: the out_proj input buffer (the
    /// un-offset `conv_out_buf`) advanced by `row0 * value_dim` BF16, because out_proj
    /// reads normed rows `value_dim` apart from row 0 while conv rows are `conv_dim`
    /// apart. Only the exact arms write through it; after the WY arms
    /// `decode_batched_inner` runs the norm itself.
    pub normed_out: DevicePtr,
    /// 2026-09-25: The h pool's per-slot byte pitch (`Qwen3SsmLayer::h_slot_stride_bytes`),
    /// half of `h_state_bytes` under `--ssm-h-dtype f16-pool`. The per-token h copies and
    /// the wyN intermediate stride use it.
    pub h_bytes: usize,
    pub conv_bytes: usize,
    pub qkvz_size: usize,
    pub conv_dim: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub d_conv: usize,
    pub qk_ch: u32,
    pub nk: usize,
    pub nv: usize,
    pub kd: usize,
    pub vd: usize,
    pub bf16: usize,
    pub fp32: usize,
    pub stream: u64,
}

impl Qwen3SsmLayer {
    /// 2026-09-25: Whether the fused K=2 verify kernels run: `gdn_verify_fused_conv_k2`
    /// (conv1d + L2 norm for both rows) and `gdn_verify_fused_norm_k2` (both gated norms).
    /// Only with `METRALE_GDN_FUSED_VERIFY=1` and both handles linked; only
    /// `kernels/gb10/common/gdn_verify_fused_k2.cu` defines them. The
    /// `gdn_verify_fused_microtest` example checks cos >= 0.99999 against the per-token
    /// path, not bitwise equality.
    pub(super) fn fused_verify_k2_enabled(&self) -> bool {
        self.gdn_verify_fused_conv_k2_k.0 != 0
            && self.gdn_verify_fused_norm_k2_k.0 != 0
            && matches!(
                std::env::var("METRALE_GDN_FUSED_VERIFY").ok().as_deref(),
                Some("1")
            )
    }

    /// 2026-09-25: Conv1d + L2 norm + GDN over the `num_tokens` rows of one sequence. Arms,
    /// in order: the exact arm under `--exact-verify`; WY kernels for K = 4, 3 and 2;
    /// `gated_delta_rule_wy17` for K = 17 (lever `gdn_wy17`, FP32 h-state only); the wyN
    /// kernel for K = 5..=16 when the h intermediates are pool-contiguous; otherwise a
    /// per-token conv + `gdn_decode` loop, which returns an error under an FP16 h-state.
    pub(super) fn decode_batched_conv_gdn(
        &self,
        ssm_state: &mut SsmLayerState,
        ctx: &ForwardContext,
        args: &ConvGdnArgs,
    ) -> Result<()> {
        let ConvGdnArgs {
            num_tokens,
            deinterleaved,
            gates_buf,
            conv_out_buf,
            gdn_out_buf,
            normed_out: _,
            h_bytes,
            conv_bytes,
            qkvz_size,
            conv_dim,
            key_dim,
            value_dim: _,
            d_conv,
            qk_ch,
            nk,
            nv,
            kd,
            vd,
            bf16,
            fp32,
            stream,
        } = *args;

        // 2026-09-25: `--exact-verify` (`verify_exact_enabled`, false under an FP16
        // h-state) runs the exact arm, which also writes the normed rows;
        // `decode_batched_inner` skips its norm on the same predicate. The WY arms below
        // feed their kernels BF16 conv rows where the single-token decode uses the FP32
        // conv when linked, so they are not bitwise equal to it (the negative-control leg
        // of the `verify_exact_microtest` example).
        if super::verify_exact_enabled() {
            return self.decode_batched_conv_gdn_exact(ssm_state, ctx, args);
        }

        if num_tokens == 4 {
            // 2026-09-25: K = 4: conv1d + L2 norm per row, then one `gated_delta_rule_wy4`
            // launch.
            for t in 0..4u32 {
                let qkv_t = deinterleaved.offset(t as usize * qkvz_size * bf16);
                let conv_out_t = conv_out_buf.offset(t as usize * conv_dim * bf16);
                ops::conv1d_update_l2norm(
                    ctx.gpu,
                    self.conv1d_l2norm_k,
                    ssm_state.conv_state,
                    qkv_t,
                    &self.ssm.conv1d,
                    conv_out_t,
                    conv_dim as u32,
                    d_conv as u32,
                    1,
                    qk_ch,
                    kd as u32,
                    1e-6,
                    stream,
                )?;
                // 2026-09-25: Conv intermediate K-1 is not written. `commit_accepted_prefix`
                // returns early at full accept and otherwise reads index
                // `num_accepted - 1 <= K - 2` (model-engine `async_chkpt.rs`).
                if t + 1 < 4 {
                    ctx.gpu.copy_d2d_async(
                        ssm_state.conv_state,
                        ssm_state.conv_state_intermediates[t as usize],
                        conv_bytes,
                        stream,
                    )?;
                }
            }

            let q_ptr = conv_out_buf;
            let k_ptr = conv_out_buf.offset(key_dim * bf16);
            let v_ptr = conv_out_buf.offset(key_dim * 2 * bf16);
            let gate_ptr = gates_buf;
            let beta_ptr = gates_buf.offset(nv * fp32);
            ops::gdn_decode_wy4(
                ctx.gpu,
                self.wy4_kernel(),
                ssm_state.h_state,
                q_ptr,
                k_ptr,
                v_ptr,
                gate_ptr,
                beta_ptr,
                gdn_out_buf,
                ssm_state.h_state_intermediates[0],
                ssm_state.h_state_intermediates[1],
                ssm_state.h_state_intermediates[2],
                1,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                conv_dim as u32,
                conv_dim as u32,
                (nv * 2) as u32,
                false,
                stream,
            )?;
        } else if num_tokens == 3 {
            // 2026-09-25: K = 3: conv1d + L2 norm per row, then one `gated_delta_rule_wy3`
            // launch (or its twin, `wy3_kernel`).
            for t in 0..3u32 {
                let qkv_t = deinterleaved.offset(t as usize * qkvz_size * bf16);
                let conv_out_t = conv_out_buf.offset(t as usize * conv_dim * bf16);
                ops::conv1d_update_l2norm(
                    ctx.gpu,
                    self.conv1d_l2norm_k,
                    ssm_state.conv_state,
                    qkv_t,
                    &self.ssm.conv1d,
                    conv_out_t,
                    conv_dim as u32,
                    d_conv as u32,
                    1,
                    qk_ch,
                    kd as u32,
                    1e-6,
                    stream,
                )?;
                // 2026-09-25: Conv intermediate K-1 is not written (see the K = 4 arm).
                if t + 1 < 3 {
                    ctx.gpu.copy_d2d_async(
                        ssm_state.conv_state,
                        ssm_state.conv_state_intermediates[t as usize],
                        conv_bytes,
                        stream,
                    )?;
                }
            }

            let q_ptr = conv_out_buf;
            let k_ptr = conv_out_buf.offset(key_dim * bf16);
            let v_ptr = conv_out_buf.offset(key_dim * 2 * bf16);
            let gate_ptr = gates_buf;
            let beta_ptr = gates_buf.offset(nv * fp32);
            ops::gdn_decode_wy3(
                ctx.gpu,
                self.wy3_kernel(kd, vd, 1),
                ssm_state.h_state,
                q_ptr,
                k_ptr,
                v_ptr,
                gate_ptr,
                beta_ptr,
                gdn_out_buf,
                ssm_state.h_state_intermediates[0],
                ssm_state.h_state_intermediates[1],
                1,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                conv_dim as u32,
                conv_dim as u32,
                (nv * 2) as u32,
                false,
                stream,
            )?;
        } else if num_tokens == 2 {
            // 2026-09-25: K = 2: conv1d + L2 norm for both rows, then one
            // `gated_delta_rule_wy2` launch (or its twin, `wy2_kernel`).
            if self.fused_verify_k2_enabled() {
                // 2026-09-25: One launch writes both conv rows and conv intermediate 0; the
                // window after row 1 stays in `conv_state`.
                ops::gdn_verify_fused_conv_k2(
                    ctx.gpu,
                    self.gdn_verify_fused_conv_k2_k,
                    ssm_state.conv_state,
                    deinterleaved,
                    &self.ssm.conv1d,
                    conv_out_buf,
                    ssm_state.conv_state_intermediates[0],
                    conv_dim as u32,
                    d_conv as u32,
                    qk_ch,
                    kd as u32,
                    qkvz_size as u32,
                    conv_dim as u32,
                    1e-6,
                    stream,
                )?;
            } else {
                let qkv_0 = deinterleaved;
                let conv_out_0 = conv_out_buf;
                ops::conv1d_update_l2norm(
                    ctx.gpu,
                    self.conv1d_l2norm_k,
                    ssm_state.conv_state,
                    qkv_0,
                    &self.ssm.conv1d,
                    conv_out_0,
                    conv_dim as u32,
                    d_conv as u32,
                    1,
                    qk_ch,
                    kd as u32,
                    1e-6,
                    stream,
                )?;
                ctx.gpu.copy_d2d_async(
                    ssm_state.conv_state,
                    ssm_state.conv_state_intermediates[0],
                    conv_bytes,
                    stream,
                )?;

                let qkv_1 = deinterleaved.offset(qkvz_size * bf16);
                let conv_out_1 = conv_out_buf.offset(conv_dim * bf16);
                ops::conv1d_update_l2norm(
                    ctx.gpu,
                    self.conv1d_l2norm_k,
                    ssm_state.conv_state,
                    qkv_1,
                    &self.ssm.conv1d,
                    conv_out_1,
                    conv_dim as u32,
                    d_conv as u32,
                    1,
                    qk_ch,
                    kd as u32,
                    1e-6,
                    stream,
                )?;
                // 2026-09-25: Conv intermediate K-1 is not written (see the K = 4 arm).
            }

            let q_ptr = conv_out_buf;
            let k_ptr = conv_out_buf.offset(key_dim * bf16);
            let v_ptr = conv_out_buf.offset(key_dim * 2 * bf16);
            let gate_ptr = gates_buf;
            let beta_ptr = gates_buf.offset(nv * fp32);
            ops::gdn_decode_wy2(
                ctx.gpu,
                self.wy2_kernel(kd, vd, 1),
                ssm_state.h_state,
                q_ptr,
                k_ptr,
                v_ptr,
                gate_ptr,
                beta_ptr,
                gdn_out_buf,
                ssm_state.h_state_intermediates[0],
                1,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                conv_dim as u32,
                conv_dim as u32,
                (nv * 2) as u32,
                false,
                stream,
            )?;
        } else if num_tokens == 17
            && self.gdn_wy17_k.0 != 0
            && ctx.levers.gdn_wy17
            && !super::ssm_h_fp16_enabled()
        {
            // 2026-09-25: K = 17: `gated_delta_rule_wy17` through the shared wyN body
            // (`decode_batched_conv_gdn_wyn`). There is no FP16 wy17, so under an FP16
            // h-state K = 17 reaches the per-token fallback, which returns an error.
            self.decode_batched_conv_gdn_wyn(ssm_state, ctx, args, self.gdn_wy17_k)?;
        } else if let Some(wyn_k) = self.wyn_kernel(num_tokens, ctx.levers.gdn_wyn).filter(|_| {
            // 2026-09-25: The wyN launch writes Hi_t at `h_state_intermediates[0] + t *
            // h_bytes`, so it runs only when the intermediates sit there.
            let h_base = ssm_state.h_state_intermediates[0];
            ssm_state
                .h_state_intermediates
                .iter()
                .take(num_tokens - 1)
                .enumerate()
                .all(|(t, p)| p.0 == h_base.0 + (t * h_bytes) as u64)
        }) {
            // 2026-09-25: K = 5..=16: the wyN kernel from `wyn_kernel` (lever `gdn_wyn`,
            // off with `METRALE_GDN_WYN=0`).
            self.decode_batched_conv_gdn_wyn(ssm_state, ctx, args, wyn_k)?;
        } else {
            // 2026-09-25: Per-token fallback for every other case. Its GDN kernel reads the
            // h-state as FP32, so under an FP16 h-state it returns an error instead.
            if super::ssm_h_fp16_enabled() {
                anyhow::bail!(
                    "METRALE_SSM_H_FP16: no FP16 fused arm for K={num_tokens} \
                     GDN verify (twin missing/killed or non-pool \
                     intermediates). The sequential fallback's FP32 kernels \
                     would read the FP16 pool as floats and emit fluent \
                     garbage, so this refuses instead. Run without \
                     --speculative at this width, or unset METRALE_SSM_H_FP16."
                );
            }
            // 2026-09-25: The conv rows are FP32 in `ssm_conv_out_f32` when
            // `causal_conv1d_update_l2norm_f32` is linked, as in `ssm_forward`, and BF16 in
            // `conv_out_buf` otherwise; the Q/K/V offsets use the matching element size.
            let use_f32_conv = self.conv1d_l2norm_f32_k.0 != 0;
            let conv_elem = if use_f32_conv { fp32 } else { bf16 };
            let conv_kernel = if use_f32_conv {
                self.conv1d_l2norm_f32_k
            } else {
                self.conv1d_l2norm_k
            };
            // 2026-09-25: `ssm_conv_out_f32` holds `m * ssm_qkvz_size` FP32 for the arena's
            // row capacity m (gpu-runtime `buffers/sizes.rs`); rows here are `conv_dim` FP32
            // apart.
            let f32_conv_base = ctx.buffers.ssm_conv_out_f32();

            for t in 0..(num_tokens as u32) {
                let qkv_t = deinterleaved.offset(t as usize * qkvz_size * bf16);
                let conv_out_t = if use_f32_conv {
                    f32_conv_base.offset(t as usize * conv_dim * fp32)
                } else {
                    conv_out_buf.offset(t as usize * conv_dim * bf16)
                };
                ops::conv1d_update_l2norm(
                    ctx.gpu,
                    conv_kernel,
                    ssm_state.conv_state,
                    qkv_t,
                    &self.ssm.conv1d,
                    conv_out_t,
                    conv_dim as u32,
                    d_conv as u32,
                    1,
                    qk_ch,
                    kd as u32,
                    1e-6,
                    stream,
                )?;

                let q_t = conv_out_t;
                let k_t = conv_out_t.offset(key_dim * conv_elem);
                let v_t = conv_out_t.offset(key_dim * 2 * conv_elem);
                let gate_beta_stride = nv * 2 * fp32;
                let gate_t = gates_buf.offset(t as usize * gate_beta_stride);
                let beta_t = gates_buf.offset(t as usize * gate_beta_stride + nv * fp32);
                let gdn_out_t = gdn_out_buf.offset(t as usize * args.value_dim * bf16);
                ops::gdn_decode(
                    ctx.gpu,
                    self.gdn_k,
                    ssm_state.h_state,
                    q_t,
                    k_t,
                    v_t,
                    gate_t,
                    beta_t,
                    gdn_out_t,
                    1,
                    nk as u32,
                    nv as u32,
                    kd as u32,
                    vd as u32,
                    stream,
                )?;

                // 2026-09-25: Intermediates K-1 are not written (see the K = 4 arm); the h
                // side is only required to hold K-1 of them (`decode_batched_inner`).
                if (t as usize) + 1 < num_tokens {
                    ctx.gpu.copy_d2d_async(
                        ssm_state.h_state,
                        ssm_state.h_state_intermediates[t as usize],
                        h_bytes,
                        stream,
                    )?;
                    ctx.gpu.copy_d2d_async(
                        ssm_state.conv_state,
                        ssm_state.conv_state_intermediates[t as usize],
                        conv_bytes,
                        stream,
                    )?;
                }
            }
        }

        Ok(())
    }
}
