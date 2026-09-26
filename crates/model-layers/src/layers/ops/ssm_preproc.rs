// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Launchers for the GDN and gated-attention preprocessing kernels:
//! QKVZ and Q/gate deinterleave, the fused Q norm (and MRoPE), the batched sigmoid
//! gate, and the BA-projection gate transforms.
//!
//! Owner: model-layers ops.
//! Invariants: none beyond the types.

#![allow(unused_imports)]

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::layers::moe;
use crate::weight_map::{DenseWeight, Fp8DenseWeight, Fp8Weight, QuantizedWeight};

use super::*;

/// 2026-09-25: Deinterleave the QKVZ projection output from GQA-grouped to
/// sequential layout, per token.
///
/// Input row: [num_groups × (kd + kd + vpg*vd + vpg*vd)]
/// Output row: [Q_total | K_total | V_total | Z_total]
///
/// Kernel: `deinterleave_qkvz(interleaved, output, num_groups, head_k_dim,
///          vheads_per_group, head_v_dim)`
/// Grid: (num_tokens, ceil(total/256), 1)  Block: (256, 1, 1)
pub fn deinterleave_qkvz(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    interleaved: DevicePtr,
    output: DevicePtr,
    num_tokens: u32,
    num_groups: u32,
    head_k_dim: u32,
    vheads_per_group: u32,
    head_v_dim: u32,
    stream: u64,
) -> Result<()> {
    let group_dim = 2 * head_k_dim + 2 * vheads_per_group * head_v_dim;
    let total = num_groups * group_dim;
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, div_ceil(total, 256), 1])
        .block([256, 1, 1])
        .arg_ptr(interleaved)
        .arg_ptr(output)
        .arg_u32(num_groups)
        .arg_u32(head_k_dim)
        .arg_u32(vheads_per_group)
        .arg_u32(head_v_dim)
        .launch(stream)
}

/// 2026-09-25: Deinterleave Q/gate from per-head interleaved to contiguous
/// layout, in place, per token. Rows are `stride` elements apart.
///
/// Input layout:  [Q_h0(hd), G_h0(hd), Q_h1(hd), G_h1(hd), ...]
/// Output layout: [Q_h0(hd), Q_h1(hd), ..., G_h0(hd), G_h1(hd), ...]
///
/// Kernel: `deinterleave_qg(data, num_heads, head_dim, stride)`
/// Grid: (num_tokens, 1, 1)  Block: (256, 1, 1)
/// Dynamic shared memory: num_heads * head_dim * 2 BF16 values
pub fn deinterleave_qg(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    data: DevicePtr,
    num_tokens: u32,
    num_heads: u32,
    head_dim: u32,
    stride: u32,
    stream: u64,
) -> Result<()> {
    let shared_bytes = num_heads * head_dim * 2 * 2;
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([256, 1, 1])
        .shared_mem(shared_bytes)
        .arg_ptr(data)
        .arg_u32(num_heads)
        .arg_u32(head_dim)
        .arg_u32(stride)
        .launch(stream)
}

/// 2026-09-25: [`deinterleave_qg`] with Q written to `q_out` (contiguous
/// `[num_tokens, num_heads * head_dim]`) instead of in place. The gate is
/// written back into `data` at offset `num_heads * head_dim` of each row.
pub fn deinterleave_qg_split(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    data: DevicePtr,
    q_out: DevicePtr,
    num_tokens: u32,
    num_heads: u32,
    head_dim: u32,
    stride: u32,
    stream: u64,
) -> Result<()> {
    let shared_bytes = num_heads * head_dim * 2 * 2;
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([256, 1, 1])
        .shared_mem(shared_bytes)
        .arg_ptr(data)
        .arg_ptr(q_out)
        .arg_u32(num_heads)
        .arg_u32(head_dim)
        .arg_u32(stride)
        .launch(stream)
}

/// 2026-09-25: [`deinterleave_qg_split`] fused with the per-head Q RMS norm.
///
/// The gate goes to `data[q_total..]`; Q is normalized in shared memory and
/// written once, to `q_out`. `q_norm_weight` is `[head_dim]`, shared by every
/// head, and applied as `x * rms * (1 + w)`.
pub fn deinterleave_qg_split_qnorm(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    data: DevicePtr,
    q_out: DevicePtr,
    q_norm_weight: DevicePtr,
    num_tokens: u32,
    num_heads: u32,
    head_dim: u32,
    stride: u32,
    eps: f32,
    stream: u64,
) -> Result<()> {
    let shared_bytes = num_heads * head_dim * 2 * 2;
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([256, 1, 1])
        .shared_mem(shared_bytes)
        .arg_ptr(data)
        .arg_ptr(q_out)
        .arg_ptr(q_norm_weight)
        .arg_u32(num_heads)
        .arg_u32(head_dim)
        .arg_u32(stride)
        .arg_f32(eps)
        .launch(stream)
}

/// 2026-09-25: [`deinterleave_qg_split_qnorm`] plus rotate-half MRoPE on the
/// first `rotary_dim` dimensions of each Q head, using the per-token positions
/// `pos_t` / `pos_h` / `pos_w`.
///
/// The gate goes to `data[q_total..]`; Q is deinterleaved, normalized, rotated,
/// then written to `q_out`. The interleaved-MRoPE prefill (`cache_skip.rs`)
/// launches it under `METRALE_ATTN_PREFILL_FUSED_QROPE=1`.
#[allow(clippy::too_many_arguments)]
pub fn deinterleave_qg_split_qnorm_mrope(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    data: DevicePtr,
    q_out: DevicePtr,
    q_norm_weight: DevicePtr,
    pos_t: DevicePtr,
    pos_h: DevicePtr,
    pos_w: DevicePtr,
    num_tokens: u32,
    num_heads: u32,
    head_dim: u32,
    stride: u32,
    rotary_dim: u32,
    eps: f32,
    theta: f32,
    stream: u64,
) -> Result<()> {
    let raw_shared = num_heads * head_dim * 2 * 2;
    let norm_shared = num_heads * head_dim * 2;
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([256, 1, 1])
        .shared_mem(raw_shared + norm_shared)
        .arg_ptr(data)
        .arg_ptr(q_out)
        .arg_ptr(q_norm_weight)
        .arg_ptr(pos_t)
        .arg_ptr(pos_h)
        .arg_ptr(pos_w)
        .arg_u32(num_heads)
        .arg_u32(head_dim)
        .arg_u32(stride)
        .arg_u32(rotary_dim)
        .arg_f32(eps)
        .arg_f32(theta)
        .launch(stream)
}

/// 2026-09-25: [`sigmoid_gate_mul`] over `num_tokens` rows in one launch:
/// `output[t, d] = input[t, d] * sigmoid(gate[t * gate_stride + d])`.
/// `input` and `output` are contiguous `[num_tokens, dim]`.
pub fn sigmoid_gate_mul_batched(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    gate: DevicePtr,
    output: DevicePtr,
    dim: u32,
    gate_stride: u32,
    num_tokens: u32,
    stream: u64,
) -> Result<()> {
    let total = num_tokens * dim;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(total, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(gate)
        .arg_ptr(output)
        .arg_u32(dim)
        .arg_u32(gate_stride)
        .arg_u32(total)
        .launch(stream)
}

/// 2026-09-25: GDN gates from the interleaved BA projection and the learned
/// `A_log` / `dt_bias`: FP32 gate (decay) and beta (write gate) per value head
/// and token.
///
/// Input rows are `ba_stride` BF16 elements apart. Output rows of both
/// `gate_out` and `beta_out` are `2 * num_v_heads` floats apart.
///
/// Kernel: `compute_gdn_gates(ba_interleaved, A_log, dt_bias, gate_out,
///          beta_out, num_v_heads, num_groups, vheads_per_group, ba_stride)`
/// Grid: (num_tokens, 1, 1)  Block: (num_v_heads, 1, 1)
pub fn compute_gdn_gates(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    ba_interleaved: DevicePtr,
    a_log: DevicePtr,
    dt_bias: DevicePtr,
    gate_out: DevicePtr,
    beta_out: DevicePtr,
    num_tokens: u32,
    num_v_heads: u32,
    num_groups: u32,
    vheads_per_group: u32,
    ba_stride: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([num_v_heads, 1, 1])
        .arg_ptr(ba_interleaved)
        .arg_ptr(a_log)
        .arg_ptr(dt_bias)
        .arg_ptr(gate_out)
        .arg_ptr(beta_out)
        .arg_u32(num_v_heads)
        .arg_u32(num_groups)
        .arg_u32(vheads_per_group)
        .arg_u32(ba_stride)
        .launch(stream)
}

/// 2026-09-25: BA projection GEMV and the GDN gate/beta transforms in one
/// kernel, for one token, with no intermediate BA buffer.
///
/// Kernel: `dense_gemv_ba_gates(A, B, A_log, dt_bias, gate, beta, N, K, vpg)`
/// Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn dense_gemv_ba_gates(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    ba_weight: &DenseWeight,
    a_log: DevicePtr,
    dt_bias: DevicePtr,
    gate_out: DevicePtr,
    beta_out: DevicePtr,
    n: u32,
    k: u32,
    vheads_per_group: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(ba_weight.weight)
        .arg_ptr(a_log)
        .arg_ptr(dt_bias)
        .arg_ptr(gate_out)
        .arg_ptr(beta_out)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(vheads_per_group)
        .launch(stream)
}

/// 2026-09-25: BA GEMM and the GDN gate transforms for `m` tokens in one
/// kernel, with no intermediate BA buffer.
///
/// `input` is `[m, k_stride]` BF16 activations and `ba_weight` `[n, k]` BF16,
/// row-major. `gate_out` is one FP32 buffer, `gate_stride` floats per token:
///   gate_out[token * gate_stride + vh]      = gate (alpha→exp transform)
///   gate_out[token * gate_stride + nv + vh] = beta (sigmoid)
///
/// Kernel: `dense_gemm_ba_gates_prefill(A, B, A_log, dt_bias, gate_out, M, N, K,
///          K_stride, gate_stride, nv, vpg)`
/// Grid: (ceil(N/4), M_tokens, 1)  Block: (256, 1, 1)
///
/// `twin` is the one-CTA-per-token `dense_gemm_ba_gates_prefill_hopper` (module
/// `ssm_ba_gates_hopper`), or `KernelHandle(0)` where the target lacks it.
/// [`ba_gates_pick`] chooses between the two here, once, so the call sites do
/// not branch. Both take the same arguments; the
/// `native_ssm_ba_gates_hopper_microtest` example compares their outputs byte
/// for byte.
#[allow(clippy::too_many_arguments)]
pub fn dense_gemm_ba_gates_prefill(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    twin: KernelHandle,
    input: DevicePtr,
    ba_weight: &DenseWeight,
    a_log: DevicePtr,
    dt_bias: DevicePtr,
    gate_out: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    k_stride: u32,
    gate_stride: u32,
    nv: u32,
    vheads_per_group: u32,
    stream: u64,
) -> Result<()> {
    let requested = ssm_ba_gates_hopper_enabled();
    let pick = ba_gates_pick(
        requested,
        kernel,
        twin,
        m,
        n,
        k,
        k_stride,
        ba_gates_sm_count(gpu),
    );
    ba_gates_log(&pick, requested, m);
    if pick.twin {
        return dense_gemm_ba_gates_prefill_hopper(
            gpu,
            pick.kernel,
            input,
            ba_weight,
            a_log,
            dt_bias,
            gate_out,
            m,
            n,
            k,
            k_stride,
            gate_stride,
            nv,
            vheads_per_group,
            stream,
        );
    }
    KernelLaunch::new(gpu, pick.kernel)
        .grid([div_ceil(n, 4), m, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(ba_weight.weight)
        .arg_ptr(a_log)
        .arg_ptr(dt_bias)
        .arg_ptr(gate_out)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(k_stride)
        .arg_u32(gate_stride)
        .arg_u32(nv)
        .arg_u32(vheads_per_group)
        .launch(stream)
}
