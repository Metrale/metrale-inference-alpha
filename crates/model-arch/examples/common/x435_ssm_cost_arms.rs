// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The two timed arms of `x435_ssm_cost`. `fused_arm` runs the bf16
//! conv (`causal_conv1d_update_l2norm` per token when n = 1,
//! `gdn_verify_fused_conv_kn_batched` otherwise), one `gated_delta_rule_wy2` or
//! `_wy4` launch and one `gated_rms_norm` over n × k rows. `exact_arm` runs, per
//! token, the f32 conv and `gated_delta_rule_decode_f32_norm` (or their strided
//! forms), with the conv-state and h-state copies between tokens when `with_d2d`.
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.

use crate::*;

pub(crate) fn fused_arm(g: &dyn GpuBackend, kit: &Kit, b: &Bufs) -> Result<()> {
    let (n, k) = (b.n, b.k);
    if n == 1 {
        for t in 0..k {
            KernelLaunch::new(g, kit.conv_b)
                .grid([div_ceil(CONV_DIM as u32, 256), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(b.conv_state)
                .arg_ptr(b.input.offset(t * CONV_DIM * 2))
                .arg_ptr(b.wconv)
                .arg_ptr(DevicePtr::NULL)
                .arg_ptr(b.conv_out_b.offset(t * CONV_DIM * 2))
                .arg_u32(1)
                .arg_u32(CONV_DIM as u32)
                .arg_u32(D_CONV as u32)
                .arg_u32((2 * KEY_DIM) as u32)
                .arg_u32(KD as u32)
                .arg_f32(1e-6)
                .launch(0)?;
            if t + 1 < k {
                g.copy_d2d_async(
                    b.conv_state,
                    b.conv_inter.offset(t * CONV_ST * 4),
                    CONV_ST * 4,
                    0,
                )?;
            }
        }
    } else {
        KernelLaunch::new(g, kit.conv_kn_batched)
            .grid([div_ceil(CONV_DIM as u32, 256), n as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(b.conv_state)
            .arg_ptr(b.input)
            .arg_ptr(b.wconv)
            .arg_ptr(b.conv_out_b)
            .arg_ptr(b.conv_inter)
            .arg_u32(k as u32)
            .arg_u32(CONV_DIM as u32)
            .arg_u32(D_CONV as u32)
            .arg_u32((2 * KEY_DIM) as u32)
            .arg_u32(KD as u32)
            .arg_u32(CONV_DIM as u32)
            .arg_u32(CONV_DIM as u32)
            .arg_u32(CONV_ST as u32)
            .arg_f32(1e-6)
            .arg_u32(CONV_ST as u32)
            .arg_u32((k * CONV_DIM) as u32)
            .arg_u32((k * CONV_DIM) as u32)
            .arg_u32((k * CONV_ST) as u32)
            .launch(0)?;
    }
    let wy = if k == 2 { kit.wy2 } else { kit.wy4 };
    let mut l = KernelLaunch::new(g, wy)
        .grid([NV as u32, n as u32, 1])
        .block([128, 1, 1])
        .arg_ptr(b.h)
        .arg_ptr(b.conv_out_b)
        .arg_ptr(b.conv_out_b.offset(KEY_DIM * 2))
        .arg_ptr(b.conv_out_b.offset(2 * KEY_DIM * 2))
        .arg_ptr(b.gates)
        .arg_ptr(b.gates.offset(NV * 4))
        .arg_ptr(b.gdn_out);
    for i in 0..(k - 1) {
        l = l.arg_ptr(b.inter[i]);
    }
    l.arg_u32(n as u32)
        .arg_u32(NK as u32)
        .arg_u32(NV as u32)
        .arg_u32(KD as u32)
        .arg_u32(VD as u32)
        .arg_u32(CONV_DIM as u32)
        .arg_u32(CONV_DIM as u32)
        .arg_u32((2 * NV) as u32)
        .arg_u32(0)
        .launch(0)?;
    KernelLaunch::new(g, kit.grms)
        .grid([(n * k) as u32, 1, 1])
        .block([1024, 1, 1])
        .arg_ptr(b.gdn_out)
        .arg_ptr(b.z)
        .arg_ptr(b.normw)
        .arg_ptr(b.normed)
        .arg_u32(VALUE_DIM as u32)
        .arg_f32(1e-6)
        .arg_u32(VALUE_DIM as u32)
        .arg_u32(0)
        .launch(0)?;
    Ok(())
}

pub(crate) fn exact_arm(g: &dyn GpuBackend, kit: &Kit, b: &Bufs, with_d2d: bool) -> Result<()> {
    let (n, k) = (b.n, b.k);
    for t in 0..k {
        if n == 1 {
            KernelLaunch::new(g, kit.conv_f)
                .grid([div_ceil(CONV_DIM as u32, 256), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(b.conv_state)
                .arg_ptr(b.input.offset(t * CONV_DIM * 2))
                .arg_ptr(b.wconv)
                .arg_ptr(DevicePtr::NULL)
                .arg_ptr(b.conv_out_f.offset(t * CONV_DIM * 4))
                .arg_u32(1)
                .arg_u32(CONV_DIM as u32)
                .arg_u32(D_CONV as u32)
                .arg_u32((2 * KEY_DIM) as u32)
                .arg_u32(KD as u32)
                .arg_f32(1e-6)
                .launch(0)?;
            let ot = b.conv_out_f.offset(t * CONV_DIM * 4);
            KernelLaunch::new(g, kit.gdn_f32_norm)
                .grid([NV as u32, 1, 1])
                .block([128, 1, 1])
                .arg_ptr(b.h)
                .arg_ptr(ot)
                .arg_ptr(ot.offset(KEY_DIM * 4))
                .arg_ptr(ot.offset(2 * KEY_DIM * 4))
                .arg_ptr(b.gates.offset(t * 2 * NV * 4))
                .arg_ptr(b.gates.offset((t * 2 * NV + NV) * 4))
                .arg_ptr(b.z.offset(t * VALUE_DIM * 2))
                .arg_ptr(b.normw)
                .arg_ptr(b.normed.offset(t * VALUE_DIM * 2))
                .arg_u32(1)
                .arg_u32(NK as u32)
                .arg_u32(NV as u32)
                .arg_u32(KD as u32)
                .arg_u32(VD as u32)
                .arg_f32(1e-6)
                .launch(0)?;
        } else {
            // 2026-09-25: Strided kernels: one launch per token for all n sequences; token t of
            // sequence s is input row (s * k + t) * CONV_DIM.
            KernelLaunch::new(g, kit.conv_f_str)
                .grid([div_ceil(CONV_DIM as u32, 256), n as u32, 1])
                .block([256, 1, 1])
                .arg_ptr(b.conv_state)
                .arg_ptr(b.input.offset(t * CONV_DIM * 2))
                .arg_ptr(b.wconv)
                .arg_ptr(DevicePtr::NULL)
                .arg_ptr(b.conv_out_f.offset(t * CONV_DIM * 4))
                .arg_u32(n as u32)
                .arg_u32(CONV_DIM as u32)
                .arg_u32(D_CONV as u32)
                .arg_u32((2 * KEY_DIM) as u32)
                .arg_u32(KD as u32)
                .arg_f32(1e-6)
                .arg_u32((k * CONV_DIM) as u32)
                .arg_u32((k * CONV_DIM) as u32)
                .launch(0)?;
            let ot = b.conv_out_f.offset(t * CONV_DIM * 4);
            KernelLaunch::new(g, kit.gdn_f32_str_norm)
                .grid([NV as u32, n as u32, 1])
                .block([128, 1, 1])
                .arg_ptr(b.h)
                .arg_ptr(ot)
                .arg_ptr(ot.offset(KEY_DIM * 4))
                .arg_ptr(ot.offset(2 * KEY_DIM * 4))
                .arg_ptr(b.gates.offset(t * 2 * NV * 4))
                .arg_ptr(b.gates.offset((t * 2 * NV + NV) * 4))
                .arg_ptr(b.z.offset(t * VALUE_DIM * 2))
                .arg_ptr(b.normw)
                .arg_ptr(b.normed.offset(t * VALUE_DIM * 2))
                .arg_u32(n as u32)
                .arg_u32(NK as u32)
                .arg_u32(NV as u32)
                .arg_u32(KD as u32)
                .arg_u32(VD as u32)
                .arg_u32((k * CONV_DIM) as u32)
                .arg_u32((k * CONV_DIM) as u32)
                .arg_u32((k * 2 * NV) as u32)
                .arg_u32((k * VALUE_DIM) as u32)
                .arg_u32((k * VALUE_DIM) as u32)
                .arg_f32(1e-6)
                .launch(0)?;
        }
        if with_d2d && t + 1 < k {
            g.copy_d2d_async(
                b.conv_state,
                b.conv_inter.offset(t * n * CONV_ST * 4),
                n * CONV_ST * 4,
                0,
            )?;
            g.copy_d2d_async(b.h, b.inter[t.min(2)], n * H_NUMEL * 4, 0)?;
        }
    }
    Ok(())
}
