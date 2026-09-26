// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Single-token decode forward of one Qwen3.5 GDN (linear-attention)
//! layer.
//!
//! Owner: model-layers.
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelArg};

use super::super::quant_weights::QuantWeights;
use super::{
    LinearAttentionLayer, LinearAttentionScratch, LinearAttentionState, Qwen35ForwardConfig,
    Qwen35Kernels,
};

/// 2026-09-25: Runs one GDN decoder layer for one token and returns `x_buf`,
/// into which the layer's output residual stream (`scratch.x_final`) is copied.
/// `intra_dump`, when given, receives named intermediate buffers after a
/// stream synchronize.
#[allow(clippy::too_many_arguments)]
pub fn forward_linear_attention<Q: QuantWeights>(
    gpu: &dyn GpuBackend,
    cfg: &Qwen35ForwardConfig,
    k: &Qwen35Kernels,
    layer: &LinearAttentionLayer<'_, Q>,
    state: &LinearAttentionState,
    scratch: &LinearAttentionScratch,
    x_in: DevicePtr,
    x_buf: DevicePtr,
    stream: u64,
    intra_dump: Option<&dyn Fn(&str, DevicePtr, u32) -> Result<()>>,
) -> Result<DevicePtr> {
    gpu.launch_typed(
        k.rms,
        [1, 1, 1],
        [128, 1, 1],
        0,
        stream,
        &[
            KernelArg::Bytes(&cfg.hidden.to_le_bytes()),
            KernelArg::Bytes(&cfg.rms_eps.to_le_bytes()),
            KernelArg::Buffer(x_in),
            KernelArg::Buffer(layer.input_ln),
            KernelArg::Buffer(scratch.x_norm),
        ],
    )?;
    // 2026-09-25: in_proj_a and in_proj_b read the same x_norm: one dual-output
    // GEMV computes both.
    layer.in_proj_a.gemv_gate_up_with(
        layer.in_proj_b,
        gpu,
        scratch.x_norm,
        scratch.dt_raw,
        scratch.b_raw,
        stream,
    )?;
    layer
        .in_proj_qkv
        .gemv(gpu, scratch.x_norm, scratch.qkv, stream)?;
    layer
        .in_proj_z
        .gemv(gpu, scratch.x_norm, scratch.z, stream)?;

    // 2026-09-25: One kernel runs the causal conv update and SiLU on every
    // channel, then a per-head L2 norm on the Q and K channels only. A block
    // is one head wide (block_x = k_head_dim_lin), so no block straddles the
    // Q/K and V ranges.
    let batch_one: u32 = 1;
    let block_x: u32 = cfg.k_head_dim_lin;
    let qkv_total_lin = cfg.qkv_total_lin();
    let blocks_per_batch = qkv_total_lin.div_ceil(block_x);
    let qk_channels: u32 = 2 * cfg.num_k_heads_lin * cfg.k_head_dim_lin;
    let l2_eps: f32 = 1e-6;
    gpu.launch_typed(
        k.conv1d,
        [blocks_per_batch * batch_one, 1, 1],
        [block_x, 1, 1],
        0,
        stream,
        &[
            KernelArg::Buffer(state.conv1d_state),
            KernelArg::Buffer(scratch.qkv),
            KernelArg::Buffer(layer.conv1d_weight),
            KernelArg::Buffer(scratch.qkv_smooth),
            KernelArg::Bytes(&batch_one.to_le_bytes()),
            KernelArg::Bytes(&qkv_total_lin.to_le_bytes()),
            KernelArg::Bytes(&cfg.conv_kernel_size.to_le_bytes()),
            KernelArg::Bytes(&qk_channels.to_le_bytes()),
            KernelArg::Bytes(&cfg.k_head_dim_lin.to_le_bytes()),
            KernelArg::Bytes(&l2_eps.to_le_bytes()),
        ],
    )?;
    // 2026-09-25: The GDN query scale 1/sqrt(k_head_dim_lin) is applied by
    // gated_delta_rule_decode to its output, not here.

    // 2026-09-25: gate = exp(softplus(dt + dt_bias) * -exp(A_log)), FP32.
    let num_state_heads = cfg.num_state_heads();
    gpu.launch_typed(
        k.gdn_gate,
        [num_state_heads.div_ceil(32), 1, 1],
        [32, 1, 1],
        0,
        stream,
        &[
            KernelArg::Bytes(&num_state_heads.to_le_bytes()),
            KernelArg::Buffer(scratch.dt_raw),
            KernelArg::Buffer(layer.dt_bias),
            KernelArg::Buffer(layer.a_log),
            KernelArg::Buffer(scratch.gate),
        ],
    )?;
    // 2026-09-25: beta = sigmoid(b_raw), FP32.
    gpu.launch_typed(
        k.sigmoid,
        [num_state_heads.div_ceil(32), 1, 1],
        [32, 1, 1],
        0,
        stream,
        &[
            KernelArg::Bytes(&num_state_heads.to_le_bytes()),
            KernelArg::Buffer(scratch.b_raw),
            KernelArg::Buffer(scratch.beta),
        ],
    )?;

    // 2026-09-25: qkv_smooth is BF16 `[Q | K | V]`; the views are byte offsets.
    let k_offset = (cfg.num_k_heads_lin * cfg.k_head_dim_lin) as usize * 2;
    let v_offset = (2 * cfg.num_k_heads_lin * cfg.k_head_dim_lin) as usize * 2;
    let q_view = scratch.qkv_smooth;
    let k_view = scratch.qkv_smooth.offset(k_offset);
    let v_view = scratch.qkv_smooth.offset(v_offset);

    let batch_size = 1u32;
    let total_groups = cfg.num_v_heads_lin * batch_size;
    gpu.launch_typed(
        k.gdn_dec,
        [total_groups, 1, 1],
        [128, 1, 1],
        0,
        stream,
        &[
            KernelArg::Buffer(state.gdn_state),
            KernelArg::Buffer(q_view),
            KernelArg::Buffer(k_view),
            KernelArg::Buffer(v_view),
            KernelArg::Buffer(scratch.gate),
            KernelArg::Buffer(scratch.beta),
            KernelArg::Buffer(scratch.y),
            KernelArg::Bytes(&batch_size.to_le_bytes()),
            KernelArg::Bytes(&cfg.num_k_heads_lin.to_le_bytes()),
            KernelArg::Bytes(&cfg.num_v_heads_lin.to_le_bytes()),
            KernelArg::Bytes(&cfg.k_head_dim_lin.to_le_bytes()),
            KernelArg::Bytes(&cfg.v_head_dim_lin.to_le_bytes()),
        ],
    )?;

    // 2026-09-25: Per-head RMSNorm: each of the num_v_heads_lin heads is one row
    // of v_head_dim_lin.
    gpu.launch_typed(
        k.rms,
        [cfg.num_v_heads_lin, 1, 1],
        [128, 1, 1],
        0,
        stream,
        &[
            KernelArg::Bytes(&cfg.v_head_dim_lin.to_le_bytes()),
            KernelArg::Bytes(&cfg.rms_eps.to_le_bytes()),
            KernelArg::Buffer(scratch.y),
            KernelArg::Buffer(layer.norm_weight),
            KernelArg::Buffer(scratch.y_norm),
        ],
    )?;

    // 2026-09-25: out = out_proj @ (silu(z) ⊙ y_norm).
    layer
        .out_proj
        .gemv_silu_gate(gpu, scratch.z, scratch.y_norm, scratch.out, stream)?;

    // 2026-09-25: One kernel writes x_resid = x_in + out and x_norm2 =
    // RMSNorm(x_resid) with post_ln.
    gpu.launch_typed(
        k.add_rms,
        [1, 1, 1],
        [128, 1, 1],
        0,
        stream,
        &[
            KernelArg::Bytes(&cfg.hidden.to_le_bytes()),
            KernelArg::Bytes(&cfg.rms_eps.to_le_bytes()),
            KernelArg::Buffer(x_in),
            KernelArg::Buffer(scratch.out),
            KernelArg::Buffer(layer.post_ln),
            KernelArg::Buffer(scratch.x_resid),
            KernelArg::Buffer(scratch.x_norm2),
        ],
    )?;
    // 2026-09-25: gate_proj and up_proj read the same x_norm2: one dual-output
    // GEMV.
    layer.gate_proj.gemv_gate_up_with(
        layer.up_proj,
        gpu,
        scratch.x_norm2,
        scratch.gate_act,
        scratch.up_act,
        stream,
    )?;
    // 2026-09-25: x_final = x_resid + down_proj @ (silu(gate_act) ⊙ up_act).
    layer.down_proj.gemv_silu_gate_resid(
        gpu,
        scratch.gate_act,
        scratch.up_act,
        scratch.x_resid,
        scratch.x_final,
        stream,
    )?;

    if let Some(dump) = intra_dump {
        gpu.synchronize(stream)?;
        let z_dim_lin = cfg.z_dim_lin();
        dump("gdn_x_norm", scratch.x_norm, cfg.hidden)?;
        dump("gdn_qkv_pre", scratch.qkv, qkv_total_lin)?;
        dump("gdn_qkv_smooth", scratch.qkv_smooth, qkv_total_lin)?;
        dump("gdn_y", scratch.y, z_dim_lin)?;
        dump("gdn_y_norm", scratch.y_norm, z_dim_lin)?;
        dump("gdn_out", scratch.out, cfg.hidden)?;
        dump("gdn_x_resid", scratch.x_resid, cfg.hidden)?;
        dump("gdn_x_final", scratch.x_final, cfg.hidden)?;
    }

    gpu.copy_d2d_async(scratch.x_final, x_buf, cfg.hidden as usize * 2, stream)?;
    Ok(x_buf)
}
