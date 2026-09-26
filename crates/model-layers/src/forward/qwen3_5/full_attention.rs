// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Single-token decode forward of one Qwen3.5 full-attention layer,
//! with a bf16 or turbo KV cache.
//!
//! Owner: model-layers.
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelArg};

use super::super::quant_weights::QuantWeights;
use super::{
    FullAttentionLayer, FullAttentionScratch, LayerKvCache, Qwen35ForwardConfig, Qwen35Kernels,
};

/// 2026-09-25: Runs one full-attention decoder layer for one token and returns
/// `scratch.x_out`, which holds the layer's output residual stream.
#[allow(clippy::too_many_arguments)]
pub fn forward_full_attention<Q: QuantWeights>(
    gpu: &dyn GpuBackend,
    cfg: &Qwen35ForwardConfig,
    k: &Qwen35Kernels,
    layer: &FullAttentionLayer<'_, Q>,
    scratch: &FullAttentionScratch,
    kv: &LayerKvCache,
    inv_freq_ptr: DevicePtr,
    positions_ptr: DevicePtr,
    x_in: DevicePtr,
    cache_pos: u32,
    seq_len_attn: u32,
    stream: u64,
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
    layer
        .q_proj
        .gemv(gpu, scratch.x_norm, scratch.q_full, stream)?;
    // 2026-09-25: k_proj and v_proj read the same x_norm, so one dual-output
    // GEMV computes both.
    layer.k_proj.gemv_gate_up_with(
        layer.v_proj,
        gpu,
        scratch.x_norm,
        scratch.k,
        scratch.v,
        stream,
    )?;

    // 2026-09-25: q_proj writes `[num_heads, head_dim * 2]`, each head's row
    // `[Q_h | gate_h]`. Split Q and the gate into separate buffers.
    gpu.launch_typed(
        k.qkv_split,
        [cfg.head_dim, cfg.num_heads, 1],
        [1, 1, 1],
        0,
        stream,
        &[
            KernelArg::Bytes(&cfg.num_heads.to_le_bytes()),
            KernelArg::Bytes(&cfg.head_dim.to_le_bytes()),
            KernelArg::Buffer(scratch.q_full),
            KernelArg::Buffer(scratch.q_split),
            KernelArg::Buffer(scratch.gate_split),
        ],
    )?;
    let gate_view = scratch.gate_split;

    // 2026-09-25: Per-head Q and K RMSNorm: each head is one row of head_dim.
    gpu.launch_typed(
        k.rms,
        [cfg.num_heads, 1, 1],
        [128, 1, 1],
        0,
        stream,
        &[
            KernelArg::Bytes(&cfg.head_dim.to_le_bytes()),
            KernelArg::Bytes(&cfg.rms_eps.to_le_bytes()),
            KernelArg::Buffer(scratch.q_split),
            KernelArg::Buffer(layer.q_norm),
            KernelArg::Buffer(scratch.q_norm_out),
        ],
    )?;
    gpu.launch_typed(
        k.rms,
        [cfg.num_kv_heads, 1, 1],
        [128, 1, 1],
        0,
        stream,
        &[
            KernelArg::Bytes(&cfg.head_dim.to_le_bytes()),
            KernelArg::Bytes(&cfg.rms_eps.to_le_bytes()),
            KernelArg::Buffer(scratch.k),
            KernelArg::Buffer(layer.k_norm),
            KernelArg::Buffer(scratch.k_norm_out),
        ],
    )?;

    // 2026-09-25: RoPE rotates q_norm_out and k_norm_out in place.
    let half_dim = cfg.rotary_dim / 2;
    let n_tokens = 1u32;
    gpu.launch_typed(
        k.rope,
        [half_dim, cfg.num_heads, 1],
        [1, 1, 1],
        0,
        stream,
        &[
            KernelArg::Bytes(&n_tokens.to_le_bytes()),
            KernelArg::Bytes(&cfg.num_heads.to_le_bytes()),
            KernelArg::Bytes(&cfg.head_dim.to_le_bytes()),
            KernelArg::Bytes(&cfg.rotary_dim.to_le_bytes()),
            KernelArg::Buffer(positions_ptr),
            KernelArg::Buffer(inv_freq_ptr),
            KernelArg::Buffer(scratch.q_norm_out),
        ],
    )?;
    gpu.launch_typed(
        k.rope,
        [half_dim, cfg.num_kv_heads, 1],
        [1, 1, 1],
        0,
        stream,
        &[
            KernelArg::Bytes(&n_tokens.to_le_bytes()),
            KernelArg::Bytes(&cfg.num_kv_heads.to_le_bytes()),
            KernelArg::Bytes(&cfg.head_dim.to_le_bytes()),
            KernelArg::Bytes(&cfg.rotary_dim.to_le_bytes()),
            KernelArg::Buffer(positions_ptr),
            KernelArg::Buffer(inv_freq_ptr),
            KernelArg::Buffer(scratch.k_norm_out),
        ],
    )?;

    // 2026-09-25: The KV append takes the post-RoPE k_norm_out.
    let scale: f32 = 1.0 / (cfg.head_dim as f32).sqrt();
    if kv.dtype != super::MetalKvDtype::Bf16 {
        // 2026-09-25: Turbo formats. A quantized side is stored in the
        // WHT-rotated basis: K is rotated before the append and Q before
        // attention only when K is rotated; V is rotated before the append and
        // the attention output gets the inverse WHT only when V is rotated. In
        // the `Bf16KTurbo*V` formats K stays raw BF16, so Q stays raw too.
        let dt = kv.dtype;
        let hd_bytes = cfg.head_dim.to_le_bytes();
        if dt.k_is_rotated() {
            gpu.launch_typed(
                k.wht,
                [cfg.num_kv_heads, 1, 1],
                [32, 1, 1],
                0,
                stream,
                &[
                    KernelArg::Bytes(&hd_bytes),
                    KernelArg::Buffer(scratch.k_norm_out),
                ],
            )?;
        }
        if dt.v_is_rotated() {
            gpu.launch_typed(
                k.wht,
                [cfg.num_kv_heads, 1, 1],
                [32, 1, 1],
                0,
                stream,
                &[KernelArg::Bytes(&hd_bytes), KernelArg::Buffer(scratch.v)],
            )?;
        }
        let num_groups = cfg.kv_dim() / 16;
        let append_grid = [num_groups.div_ceil(64), 1, 1];
        // 2026-09-25: Sparse-V threshold: the turbo attention kernels skip V rows
        // whose unnormalized weight exp(score - max) is at or below it, so 0.0 skips
        // only zero-weight rows. METRALE_SPARSE_V_THRESHOLD sets it; unset or
        // unparsable gives 1e-3.
        let sparse_v: f32 = std::env::var("METRALE_SPARSE_V_THRESHOLD")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1e-3);
        use super::MetalKvDtype as D;
        match dt {
            D::Turbo8 | D::Turbo4 | D::Turbo3 | D::Turbo2 => {
                let (kvap_turbo, attn_turbo) = match dt {
                    D::Turbo8 => (k.kvap_turbo8, k.attn_turbo8),
                    D::Turbo4 => (k.kvap_turbo4, k.attn_turbo4),
                    D::Turbo3 => (k.kvap_turbo3, k.attn_turbo3),
                    _ => (k.kvap_turbo2, k.attn_turbo2),
                };
                let (k_scales, v_scales) = (
                    kv.k_scales.expect("sym turbo cache has k_scales"),
                    kv.v_scales.expect("sym turbo cache has v_scales"),
                );
                gpu.launch_typed(
                    kvap_turbo,
                    append_grid,
                    [64, 1, 1],
                    0,
                    stream,
                    &[
                        KernelArg::Bytes(&cfg.num_kv_heads.to_le_bytes()),
                        KernelArg::Bytes(&cfg.head_dim.to_le_bytes()),
                        KernelArg::Bytes(&cache_pos.to_le_bytes()),
                        KernelArg::Buffer(scratch.k_norm_out),
                        KernelArg::Buffer(scratch.v),
                        KernelArg::Buffer(kv.k),
                        KernelArg::Buffer(kv.v),
                        KernelArg::Buffer(k_scales),
                        KernelArg::Buffer(v_scales),
                    ],
                )?;
                gpu.launch_typed(
                    k.wht,
                    [cfg.num_heads, 1, 1],
                    [32, 1, 1],
                    0,
                    stream,
                    &[
                        KernelArg::Bytes(&hd_bytes),
                        KernelArg::Buffer(scratch.q_norm_out),
                    ],
                )?;
                gpu.launch_typed(
                    attn_turbo,
                    [cfg.num_heads, 1, 1],
                    [32, 1, 1],
                    0,
                    stream,
                    &[
                        KernelArg::Bytes(&seq_len_attn.to_le_bytes()),
                        KernelArg::Bytes(&cfg.num_heads.to_le_bytes()),
                        KernelArg::Bytes(&cfg.num_kv_heads.to_le_bytes()),
                        KernelArg::Bytes(&cfg.head_dim.to_le_bytes()),
                        KernelArg::Bytes(&scale.to_le_bytes()),
                        KernelArg::Bytes(&sparse_v.to_le_bytes()),
                        KernelArg::Buffer(scratch.q_norm_out),
                        KernelArg::Buffer(kv.k),
                        KernelArg::Buffer(kv.v),
                        KernelArg::Buffer(k_scales),
                        KernelArg::Buffer(v_scales),
                        KernelArg::Buffer(scratch.attn_out),
                    ],
                )?;
            }
            D::Bf16KTurbo4V | D::Bf16KTurbo3V | D::Bf16KTurbo2V => {
                let (kvap_asym, attn_asym) = match dt {
                    D::Bf16KTurbo4V => (k.kvap_bf16k_turbo4v, k.attn_bf16k_turbo4v),
                    D::Bf16KTurbo3V => (k.kvap_bf16k_turbo3v, k.attn_bf16k_turbo3v),
                    _ => (k.kvap_bf16k_turbo2v, k.attn_bf16k_turbo2v),
                };
                let v_scales = kv.v_scales.expect("asym cache has v_scales");
                gpu.launch_typed(
                    kvap_asym,
                    append_grid,
                    [64, 1, 1],
                    0,
                    stream,
                    &[
                        KernelArg::Bytes(&cfg.num_kv_heads.to_le_bytes()),
                        KernelArg::Bytes(&cfg.head_dim.to_le_bytes()),
                        KernelArg::Bytes(&cache_pos.to_le_bytes()),
                        KernelArg::Buffer(scratch.k_norm_out),
                        KernelArg::Buffer(scratch.v),
                        KernelArg::Buffer(kv.k),
                        KernelArg::Buffer(kv.v),
                        KernelArg::Buffer(v_scales),
                    ],
                )?;
                // 2026-09-25: K is not rotated, so Q is not rotated either.
                gpu.launch_typed(
                    attn_asym,
                    [cfg.num_heads, 1, 1],
                    [32, 1, 1],
                    0,
                    stream,
                    &[
                        KernelArg::Bytes(&seq_len_attn.to_le_bytes()),
                        KernelArg::Bytes(&cfg.num_heads.to_le_bytes()),
                        KernelArg::Bytes(&cfg.num_kv_heads.to_le_bytes()),
                        KernelArg::Bytes(&cfg.head_dim.to_le_bytes()),
                        KernelArg::Bytes(&scale.to_le_bytes()),
                        KernelArg::Bytes(&sparse_v.to_le_bytes()),
                        KernelArg::Buffer(scratch.q_norm_out),
                        KernelArg::Buffer(kv.k),
                        KernelArg::Buffer(kv.v),
                        KernelArg::Buffer(v_scales),
                        KernelArg::Buffer(scratch.attn_out),
                    ],
                )?;
            }
            D::Bf16 => unreachable!("outer branch excludes Bf16"),
        }
        if dt.v_is_rotated() {
            gpu.launch_typed(
                k.wht_inv,
                [cfg.num_heads, 1, 1],
                [32, 1, 1],
                0,
                stream,
                &[
                    KernelArg::Bytes(&hd_bytes),
                    KernelArg::Buffer(scratch.attn_out),
                ],
            )?;
        }
    } else {
        gpu.launch_typed(
            k.kvap,
            [cfg.head_dim, cfg.num_kv_heads, 1],
            [1, 1, 1],
            0,
            stream,
            &[
                KernelArg::Bytes(&cfg.num_kv_heads.to_le_bytes()),
                KernelArg::Bytes(&cfg.head_dim.to_le_bytes()),
                KernelArg::Bytes(&cache_pos.to_le_bytes()),
                KernelArg::Buffer(scratch.k_norm_out),
                KernelArg::Buffer(scratch.v),
                KernelArg::Buffer(kv.k),
                KernelArg::Buffer(kv.v),
            ],
        )?;

        gpu.launch_typed(
            k.attn,
            [cfg.num_heads, 1, 1],
            [32, 1, 1],
            0,
            stream,
            &[
                KernelArg::Bytes(&seq_len_attn.to_le_bytes()),
                KernelArg::Bytes(&cfg.num_heads.to_le_bytes()),
                KernelArg::Bytes(&cfg.num_kv_heads.to_le_bytes()),
                KernelArg::Bytes(&cfg.head_dim.to_le_bytes()),
                KernelArg::Bytes(&scale.to_le_bytes()),
                KernelArg::Buffer(scratch.q_norm_out),
                KernelArg::Buffer(kv.k),
                KernelArg::Buffer(kv.v),
                KernelArg::Buffer(scratch.attn_out),
            ],
        )?;
    }

    let q_only = cfg.q_only();
    gpu.launch_typed(
        k.sg,
        [q_only.div_ceil(64), 1, 1],
        [64, 1, 1],
        0,
        stream,
        &[
            KernelArg::Bytes(&q_only.to_le_bytes()),
            KernelArg::Buffer(gate_view),
            KernelArg::Buffer(scratch.attn_out),
            KernelArg::Buffer(scratch.gated_attn),
        ],
    )?;

    layer
        .o_proj
        .gemv(gpu, scratch.gated_attn, scratch.o, stream)?;

    // 2026-09-25: One kernel writes x_resid = x_in + o and x_norm2 =
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
            KernelArg::Buffer(scratch.o),
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
    // 2026-09-25: x_out = x_resid + down_proj @ (silu(gate_act) ⊙ up_act).
    layer.down_proj.gemv_silu_gate_resid(
        gpu,
        scratch.gate_act,
        scratch.up_act,
        scratch.x_resid,
        scratch.x_out,
        stream,
    )?;
    Ok(scratch.x_out)
}
