// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The exact verify arm for one sequence (`decode_batched_conv_gdn_exact`),
//! run under `--exact-verify`: per row, the conv, GDN and gated-norm kernels the
//! single-token decode (`ssm_forward`) selects, so the verify reproduces its bits.
//!
//! Owner: model-layers (Qwen3 SSM layer).
//! Invariants: none beyond the types.
//!
//! Kernel choice follows `ssm_forward`: the FP32 conv when linked, the FP32 GDN and
//! FP32-input norm when both are linked, and the fused GDN + norm under
//! `--gdn-fused-norm`. Two twins replace a kernel plus a copy: the `_snap` fused kernel
//! writes each h intermediate inline, and `gdn_verify_fused_conv_kn_f32` runs the conv
//! for all rows and writes the conv intermediates. Where a twin's handle is zero the arm
//! uses the parent kernel and a `copy_d2d_async` per row. The
//! `verify_exact_microtest` example (metrale-model-arch) checks the twins byte for byte
//! against the single-token chain. The arm writes the normed rows itself, and
//! `decode_batched_inner` skips its norm when `verify_exact_enabled()`.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::trait_decode_batched_conv_gdn::ConvGdnArgs;
use super::{Qwen3SsmLayer, SsmLayerState};
use crate::layer::ForwardContext;
use crate::layers::ops;

/// 2026-09-25: Byte offsets of verify row `t` in the exact arm's buffers, and whether
/// row `t` writes intermediates. Pure, and unit-tested below.
pub(super) struct ExactRow {
    /// 2026-09-25: Into `deinterleaved` (BF16 bytes): row `t`'s [Q|K|V|Z].
    pub qkv_in: usize,
    /// 2026-09-25: Into `ssm_conv_out_f32` (bytes): row `t`'s FP32 conv output. Rows are
    /// `qkvz_size` FP32 elements apart, not `conv_dim`, so each row keeps a `value_dim`
    /// tail for the unfused arm's FP32 GDN output, the layout `ssm_forward` uses for its
    /// one row.
    pub conv_out_f32: usize,
    /// 2026-09-25: Into `ssm_conv_out_f32` (bytes): row `t`'s FP32 GDN output, the row's
    /// `value_dim` tail after `conv_dim`.
    pub gdn_out_f32: usize,
    /// 2026-09-25: Into `deinterleaved` (BF16 bytes): row `t`'s Z gate.
    pub z: usize,
    /// 2026-09-25: Into `gates_buf` (bytes): row `t`'s FP32 gate; `beta` follows `nv`
    /// elements later.
    pub gate: usize,
    pub beta: usize,
    /// 2026-09-25: Into `ConvGdnArgs::normed_out` (BF16 bytes): row `t`'s normed output,
    /// the row out_proj reads.
    pub normed_out: usize,
    /// 2026-09-25: Whether row `t` writes h and conv intermediates: every row but the
    /// last (see the K = 4 arm of `decode_batched_conv_gdn`).
    pub snapshot: bool,
}

/// 2026-09-25: See [`ExactRow`].
pub(super) fn exact_row(
    t: usize,
    num_tokens: usize,
    qkvz_size: usize,
    conv_dim: usize,
    value_dim: usize,
    nv: usize,
) -> ExactRow {
    let (bf16, fp32) = (2usize, 4usize);
    let gate = t * nv * 2 * fp32;
    ExactRow {
        qkv_in: t * qkvz_size * bf16,
        conv_out_f32: t * qkvz_size * fp32,
        gdn_out_f32: t * qkvz_size * fp32 + conv_dim * fp32,
        z: t * qkvz_size * bf16 + conv_dim * bf16,
        gate,
        beta: gate + nv * fp32,
        normed_out: t * value_dim * bf16,
        snapshot: t + 1 < num_tokens,
    }
}

impl Qwen3SsmLayer {
    /// 2026-09-25: The exact conv + GDN + gated norm over the `num_tokens` verify rows of
    /// one sequence. Afterwards `h_state`, `conv_state`, intermediates 0..K-2 and the
    /// normed rows hold what `num_tokens` runs of `ssm_forward`'s chain produce, given
    /// that the twins match their parents (see the module header).
    pub(super) fn decode_batched_conv_gdn_exact(
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
            normed_out,
            h_bytes,
            conv_bytes,
            qkvz_size,
            conv_dim,
            key_dim,
            value_dim,
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
        let eps = ctx.config.rms_norm_eps as f32;

        // 2026-09-25: Kernel selection as in `ssm_forward`.
        let use_f32_conv = self.conv1d_l2norm_f32_k.0 != 0;
        let use_f32_gdn = self.gdn_f32_k.0 != 0 && self.gated_rms_norm_f32_k.0 != 0;
        let fused_gdn_norm = use_f32_gdn
            && self.gdn_f32_norm_k.0 != 0
            && crate::layers::qwen3_ssm::gdn_fused_norm_enabled();
        let snap = fused_gdn_norm && self.gdn_f32_norm_snap_k.0 != 0;
        let f32_conv_base = ctx.buffers.ssm_conv_out_f32();

        // 2026-09-25: The fused FP32 conv writes the conv intermediates `conv_bytes` apart
        // from the first, so it runs only when they sit there.
        let conv_inter_base = ssm_state.conv_state_intermediates[0];
        let fused_conv = use_f32_conv
            && self.gdn_verify_fused_conv_kn_f32_k.0 != 0
            && ssm_state
                .conv_state_intermediates
                .iter()
                .take(num_tokens)
                .enumerate()
                .all(|(t, p)| p.0 == conv_inter_base.0 + (t * conv_bytes) as u64);

        static LOGGED: std::sync::Once = std::sync::Once::new();
        LOGGED.call_once(|| {
            tracing::info!(
                "EXACT MTP verify ENGAGED (#435 route (a), opt-in --exact-verify): \
                 per-token sequential-decode kernel chain (f32_conv={use_f32_conv}, \
                 fused_gdn_norm={fused_gdn_norm}, snap_twin={snap}, \
                 fused_f32_conv={fused_conv}); omit the flag for the default WY arms"
            );
        });

        if fused_conv {
            ops::gdn_verify_fused_conv_kn_f32(
                ctx.gpu,
                self.gdn_verify_fused_conv_kn_f32_k,
                ssm_state.conv_state,
                deinterleaved,
                &self.ssm.conv1d,
                f32_conv_base,
                conv_inter_base,
                num_tokens as u32,
                conv_dim as u32,
                d_conv as u32,
                qk_ch,
                kd as u32,
                qkvz_size as u32,
                qkvz_size as u32,
                (conv_bytes / 4) as u32,
                1e-6,
                stream,
            )?;
        }

        for t in 0..num_tokens {
            let row = exact_row(t, num_tokens, qkvz_size, conv_dim, value_dim, nv);
            let qkv_t = deinterleaved.offset(row.qkv_in);

            let (conv_out_t, conv_elem) = if use_f32_conv {
                (f32_conv_base.offset(row.conv_out_f32), fp32)
            } else {
                (conv_out_buf.offset(t * conv_dim * bf16), bf16)
            };
            if !fused_conv {
                ops::conv1d_update_l2norm(
                    ctx.gpu,
                    if use_f32_conv {
                        self.conv1d_l2norm_f32_k
                    } else {
                        self.conv1d_l2norm_k
                    },
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
                if row.snapshot {
                    ctx.gpu.copy_d2d_async(
                        ssm_state.conv_state,
                        ssm_state.conv_state_intermediates[t],
                        conv_bytes,
                        stream,
                    )?;
                }
            }

            let q_t = conv_out_t;
            let k_t = conv_out_t.offset(key_dim * conv_elem);
            let v_t = conv_out_t.offset(key_dim * 2 * conv_elem);
            let gate_t = gates_buf.offset(row.gate);
            let beta_t = gates_buf.offset(row.beta);
            let z_t = deinterleaved.offset(row.z);
            let normed_t = normed_out.offset(row.normed_out);

            if fused_gdn_norm {
                if snap {
                    // 2026-09-25: The `_snap` kernel stores the h intermediate itself; a
                    // null pointer skips the store for the last row.
                    let h_inter = if row.snapshot {
                        ssm_state.h_state_intermediates[t]
                    } else {
                        DevicePtr::NULL
                    };
                    ops::gdn_decode_f32_norm_snap(
                        ctx.gpu,
                        self.gdn_f32_norm_snap_k,
                        ssm_state.h_state,
                        q_t,
                        k_t,
                        v_t,
                        gate_t,
                        beta_t,
                        z_t,
                        self.ssm.norm.weight,
                        normed_t,
                        h_inter,
                        1,
                        nk as u32,
                        nv as u32,
                        kd as u32,
                        vd as u32,
                        eps,
                        stream,
                    )?;
                } else {
                    ops::gdn_decode_f32_norm(
                        ctx.gpu,
                        self.gdn_f32_norm_k,
                        ssm_state.h_state,
                        q_t,
                        k_t,
                        v_t,
                        gate_t,
                        beta_t,
                        z_t,
                        self.ssm.norm.weight,
                        normed_t,
                        1,
                        nk as u32,
                        nv as u32,
                        kd as u32,
                        vd as u32,
                        eps,
                        stream,
                    )?;
                    if row.snapshot {
                        ctx.gpu.copy_d2d_async(
                            ssm_state.h_state,
                            ssm_state.h_state_intermediates[t],
                            h_bytes,
                            stream,
                        )?;
                    }
                }
            } else {
                // 2026-09-25: Unfused: the FP32 GDN and FP32-input norm when both are linked
                // (`use_f32_gdn`), the BF16 pair otherwise.
                let (gdn_kernel, norm_kernel, gdn_out_t) = if use_f32_gdn {
                    (
                        self.gdn_f32_k,
                        self.gated_rms_norm_f32_k,
                        f32_conv_base.offset(row.gdn_out_f32),
                    )
                } else {
                    (
                        self.gdn_k,
                        self.gated_rms_norm_k,
                        gdn_out_buf.offset(t * value_dim * bf16),
                    )
                };
                ops::gdn_decode(
                    ctx.gpu,
                    gdn_kernel,
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
                ops::gated_rms_norm(
                    ctx.gpu,
                    norm_kernel,
                    gdn_out_t,
                    z_t,
                    &self.ssm.norm,
                    normed_t,
                    nv as u32,
                    vd as u32,
                    vd as u32,
                    eps,
                    vd as u32,
                    stream,
                )?;
                if row.snapshot {
                    ctx.gpu.copy_d2d_async(
                        ssm_state.h_state,
                        ssm_state.h_state_intermediates[t],
                        h_bytes,
                        stream,
                    )?;
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::exact_row;

    // 2026-09-25: Qwen3.6-27B GDN shapes: nk = 16, nv = 48, kd = vd = 128, so
    // CONV_DIM = 2 * 16 * 128 + 48 * 128 and QKVZ = CONV_DIM + VALUE_DIM.
    const QKVZ: usize = 16384;
    const CONV_DIM: usize = 10240;
    const VALUE_DIM: usize = 6144;
    const NV: usize = 48;

    /// 2026-09-25: Row offsets stride by each buffer's row size on the 27B shapes: qkvz
    /// rows in `deinterleaved` and `ssm_conv_out_f32`, `value_dim` rows in the normed
    /// output, `2 * nv` FP32 in the gates.
    #[test]
    fn exact_row_strides_match_27b_shapes() {
        let r2 = exact_row(2, 4, QKVZ, CONV_DIM, VALUE_DIM, NV);
        assert_eq!(r2.qkv_in, 2 * QKVZ * 2, "deinterleaved rows are qkvz BF16");
        assert_eq!(r2.conv_out_f32, 2 * QKVZ * 4, "f32 conv rows are qkvz FP32");
        assert_eq!(
            r2.gdn_out_f32,
            2 * QKVZ * 4 + CONV_DIM * 4,
            "f32 gdn scratch is the conv row's Z-region tail"
        );
        assert_eq!(r2.z, 2 * QKVZ * 2 + CONV_DIM * 2, "Z sits after [Q|K|V]");
        assert_eq!(r2.gate, 2 * NV * 2 * 4, "gates rows are [gate|beta] FP32");
        assert_eq!(r2.beta, r2.gate + NV * 4);
        assert_eq!(
            r2.normed_out,
            2 * VALUE_DIM * 2,
            "normed rows are value_dim BF16"
        );
        // 2026-09-25: The FP32 GDN output fits the row's tail exactly.
        assert_eq!(QKVZ - CONV_DIM, VALUE_DIM);
    }

    /// 2026-09-25: Intermediates are written for rows 0..K-2 and not for row K-1.
    #[test]
    fn exact_row_snapshot_skips_only_last_token() {
        for k in 2..=8usize {
            for t in 0..k {
                let row = exact_row(t, k, QKVZ, CONV_DIM, VALUE_DIM, NV);
                assert_eq!(
                    row.snapshot,
                    t + 1 < k,
                    "snapshot policy wrong at t={t}, k={k}"
                );
            }
        }
    }
}
