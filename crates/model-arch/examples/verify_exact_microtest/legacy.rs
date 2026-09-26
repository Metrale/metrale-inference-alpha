// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The negative-control leg of the exact-verify bitwise gate: the
//! default BF16 conv plus WY verify arms. `main.rs` requires its final h to
//! differ from the golden chain's.
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use crate::{
    CONV_DIM, H_ELEMS, Inputs, K, KD, KEY_DIM, Kernels, NK, NV, QKVZ, VALUE_DIM, VD, conv_launch,
    dn, up,
};

/// 2026-09-25: Negative control: BF16 conv plus `gated_delta_rule_wy4` over
/// the K tokens. Returns the final h, which must not match golden's.
pub(crate) fn run_legacy_wy4(g: &dyn GpuBackend, ks: &Kernels, inp: &Inputs) -> Result<Vec<u8>> {
    let state = up(g, &inp.conv0)?;
    let h = up(g, &inp.h0)?;
    let deint = up(g, &inp.deint)?;
    let gates = up(g, &inp.gates)?;
    let w = up(g, &inp.weight)?;
    let conv_rows = g.alloc(K * CONV_DIM * 2)?;
    let gdn_out = g.alloc(K * VALUE_DIM * 2)?;
    let his: Vec<DevicePtr> = (0..3)
        .map(|_| g.alloc(H_ELEMS * 4))
        .collect::<Result<_>>()?;
    for t in 0..K {
        conv_launch(
            g,
            ks.conv_bf16,
            state,
            deint.offset(t * QKVZ * 2),
            w,
            conv_rows.offset(t * CONV_DIM * 2),
        )?;
    }
    // 2026-09-25: gate/beta rows are [gate|beta] at a 2*NV stride; the wy
    // kernel takes the two base pointers and gb_stride = 2*NV.
    KernelLaunch::new(g, ks.wy4)
        .grid([NV as u32, 1, 1])
        .block([128, 1, 1])
        .arg_ptr(h)
        .arg_ptr(conv_rows)
        .arg_ptr(conv_rows.offset(KEY_DIM * 2))
        .arg_ptr(conv_rows.offset(KEY_DIM * 2 * 2))
        .arg_ptr(gates)
        .arg_ptr(gates.offset(NV * 4))
        .arg_ptr(gdn_out)
        .arg_ptr(his[0])
        .arg_ptr(his[1])
        .arg_ptr(his[2])
        .arg_u32(1)
        .arg_u32(NK as u32)
        .arg_u32(NV as u32)
        .arg_u32(KD as u32)
        .arg_u32(VD as u32)
        .arg_u32(CONV_DIM as u32)
        .arg_u32(CONV_DIM as u32)
        .arg_u32((NV * 2) as u32)
        .arg_u32(0)
        .launch(0)?;
    g.synchronize(0)?;
    dn(g, h, H_ELEMS * 4)
}
