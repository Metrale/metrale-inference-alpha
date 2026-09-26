// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The wyN GDN verify arm for one sequence (`decode_batched_conv_gdn_wyn`),
//! shared by K = 17 (`gated_delta_rule_wy17`) and K = 5..=16 (`gated_delta_rule_wy{K}`),
//! and the wyN kernel selectors.
//!
//! Owner: model-layers (Qwen3 SSM layer).
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::KernelHandle;

use super::trait_decode_batched_conv_gdn::ConvGdnArgs;
use super::{Qwen3SsmLayer, SsmLayerState};
use crate::layer::ForwardContext;
use crate::layers::ops;

impl Qwen3SsmLayer {
    /// 2026-09-25: The wyN kernel for `num_tokens` in 5..=16, or `None` when out of range,
    /// when `wyn_enabled` (lever `gdn_wyn`) is false, or when the handle did not resolve.
    /// The caller then runs the per-token fallback.
    pub(super) fn wyn_kernel(&self, num_tokens: usize, wyn_enabled: bool) -> Option<KernelHandle> {
        if !(5..=16).contains(&num_tokens) || !wyn_enabled {
            return None;
        }
        // 2026-09-25: With an FP16 h-state only the FP16 twin is correct. A zero twin
        // handle gives `None`, and the per-token fallback returns an error under an FP16
        // h-state.
        let k = if super::ssm_h_fp16_enabled() {
            self.gdn_wyn_f16_k[num_tokens - 5]
        } else {
            self.gdn_wyn_k[num_tokens - 5]
        };
        (k.0 != 0).then_some(k)
    }

    /// 2026-09-25: The pointer-table wyN twin (`gated_delta_rule_wy{K}_table`) for the
    /// cross-sequence batched verify at width `num_tokens` in 5..=16, or `None` under the
    /// same conditions as [`Self::wyn_kernel`], with the same FP16 selection. On `None`
    /// `decode_batched_conv_gdn_multi` declines and each sequence runs alone.
    // 2026-09-25: Called from `decode_batched_conv_gdn_multi` for widths 5..=16.
    // provenance-id: 526f6e616c6420522e205374657369616b
    #[allow(dead_code)]
    pub(super) fn wyn_table_kernel(
        &self,
        num_tokens: usize,
        wyn_enabled: bool,
    ) -> Option<KernelHandle> {
        if !(5..=16).contains(&num_tokens) || !wyn_enabled {
            return None;
        }
        let k = if super::ssm_h_fp16_enabled() {
            self.gdn_wyn_f16_table_k[num_tokens - 5]
        } else {
            self.gdn_wyn_table_k[num_tokens - 5]
        };
        (k.0 != 0).then_some(k)
    }

    /// 2026-09-25: The wyN verify arm for K = `args.num_tokens` rows of one sequence.
    ///
    /// Conv1d + L2 norm: one `gdn_verify_fused_conv_kn` launch that writes all K conv
    /// intermediates, when that kernel is linked, the first K conv intermediates are
    /// `conv_bytes` apart, and `METRALE_GDN_FUSED_CONV17` is not `0`; otherwise a per-row
    /// loop that writes intermediates 0..K-2. Then one `wy_kernel` launch writes the K
    /// output rows, the states after rows 0..K-2 to the h intermediates (`h_bytes`
    /// apart) and the final state to `h_state`. `wy_kernel` must be compiled for
    /// K = `args.num_tokens`.
    pub(super) fn decode_batched_conv_gdn_wyn(
        &self,
        ssm_state: &mut SsmLayerState,
        ctx: &ForwardContext,
        args: &ConvGdnArgs,
        wy_kernel: KernelHandle,
    ) -> Result<()> {
        let ConvGdnArgs {
            num_tokens,
            deinterleaved,
            gates_buf,
            conv_out_buf,
            gdn_out_buf,
            h_bytes,
            conv_bytes,
            qkvz_size,
            conv_dim,
            key_dim,
            d_conv,
            qk_ch,
            nk,
            nv,
            kd,
            vd,
            bf16,
            fp32,
            stream,
            ..
        } = *args;

        let conv_inter_base = ssm_state.conv_state_intermediates[0];
        let inter_contiguous = ssm_state
            .conv_state_intermediates
            .iter()
            .take(num_tokens)
            .enumerate()
            .all(|(t, p)| p.0 == conv_inter_base.0 + (t * conv_bytes) as u64);
        let fused_conv = self.gdn_verify_fused_conv_kn_k.0 != 0
            && inter_contiguous
            && !matches!(
                std::env::var("METRALE_GDN_FUSED_CONV17").ok().as_deref(),
                Some("0")
            );
        if fused_conv {
            ops::gdn_verify_fused_conv_kn(
                ctx.gpu,
                self.gdn_verify_fused_conv_kn_k,
                ssm_state.conv_state,
                deinterleaved,
                &self.ssm.conv1d,
                conv_out_buf,
                conv_inter_base,
                num_tokens as u32,
                conv_dim as u32,
                d_conv as u32,
                qk_ch,
                kd as u32,
                qkvz_size as u32,
                conv_dim as u32,
                (conv_bytes / 4) as u32,
                1e-6,
                stream,
            )?;
        } else {
            for t in 0..(num_tokens as u32) {
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
                // 2026-09-25: Conv intermediate K-1 is not written (see the K = 4 arm of
                // `decode_batched_conv_gdn`); the fused launch above writes all K.
                if (t as usize) + 1 < num_tokens {
                    ctx.gpu.copy_d2d_async(
                        ssm_state.conv_state,
                        ssm_state.conv_state_intermediates[t as usize],
                        conv_bytes,
                        stream,
                    )?;
                }
            }
        }

        let q_ptr = conv_out_buf;
        let k_ptr = conv_out_buf.offset(key_dim * bf16);
        let v_ptr = conv_out_buf.offset(key_dim * 2 * bf16);
        let gate_ptr = gates_buf;
        let beta_ptr = gates_buf.offset(nv * fp32);
        // 2026-09-25: The intermediate stride in the kernel's h element: `h_bytes / 4`
        // floats for the FP32 kernels, `h_bytes / 2` halves for the `_f16` twins.
        let inter_stride_floats = if super::ssm_h_fp16_enabled() {
            (h_bytes / 2) as u32
        } else {
            (h_bytes / 4) as u32
        };
        ops::gdn_decode_wyn(
            ctx.gpu,
            wy_kernel,
            ssm_state.h_state,
            q_ptr,
            k_ptr,
            v_ptr,
            gate_ptr,
            beta_ptr,
            gdn_out_buf,
            ssm_state.h_state_intermediates[0],
            inter_stride_floats,
            1,
            nk as u32,
            nv as u32,
            kd as u32,
            vd as u32,
            conv_dim as u32,
            conv_dim as u32,
            (nv * 2) as u32,
            stream,
        )
    }
}
