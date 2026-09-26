// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: MoE loaders for the MiniMax M2 (`block_sparse_moe`, `w1/w2/w3`) and Gemma-4 (`router`, `moe.experts`) layouts.
//!
//! Owner: model-layers (weight loading).
//! Invariants: none beyond the types.

#![allow(unused_imports)]

use anyhow::{Context, Result, bail, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_weights::weights::{WeightDtype, WeightStore};

use super::*;

/// 2026-09-25: Load a MiniMax M2 MoE block.
///
/// Differences from `load_moe_no_shared`:
///   * Prefix `{layer}.block_sparse_moe` instead of `{layer}.mlp`.
///   * Experts use `w1/w2/w3`: w1 = gate_proj, w2 = down_proj, w3 = up_proj.
///   * `e_score_correction_bias` goes into `MoeWeights.correction_bias`,
///     widened to F32 when it is BF16.
///
/// Like `load_moe_no_shared` it builds a zero-filled shared expert (packed
/// codes, scales and `weight_scale_2` all 0), which contributes 0.
#[allow(clippy::too_many_arguments)]
pub fn load_moe_minimax(
    store: &WeightStore,
    layer_prefix: &str,
    num_experts: usize,
    gpu: &dyn GpuBackend,
    config: &metrale_config::ModelConfig,
    variant: Nvfp4Variant,
    absmax_k: metrale_gpu_runtime::gpu::KernelHandle,
    quantize_k: metrale_gpu_runtime::gpu::KernelHandle,
    stream: u64,
) -> Result<MoeWeights> {
    let p = format!("{layer_prefix}.block_sparse_moe");

    // 2026-09-25: `dense_f32_safe` converts an F32 gate to BF16.
    let gate = dense_f32_safe(store, &format!("{p}.gate.weight"), gpu)?;
    // 2026-09-25: The router kernels read the bias as `const float*`, so a BF16
    // bias is widened to F32 on the host; any other dtype is passed through.
    let bias_key = format!("{p}.e_score_correction_bias");
    let bias_t = store.get(&bias_key)?;
    let correction_bias = if bias_t.dtype == metrale_model_weights::weights::WeightDtype::BF16 {
        let n = bias_t.num_elements();
        let mut bf16_buf = vec![0u8; n * 2];
        gpu.copy_d2h(bias_t.ptr, &mut bf16_buf)?;
        let mut f32_buf = vec![0u8; n * 4];
        for i in 0..n {
            // 2026-09-25: A BF16 value is the high half of an F32; the low half stays zero.
            f32_buf[i * 4 + 2] = bf16_buf[i * 2];
            f32_buf[i * 4 + 3] = bf16_buf[i * 2 + 1];
        }
        let ptr = gpu.alloc(f32_buf.len())?;
        gpu.copy_h2d(&f32_buf, ptr)?;
        DenseWeight { weight: ptr }
    } else {
        dense(store, &bias_key)?
    };

    // 2026-09-25: The shared-expert buffers round up and hold at least 1 byte,
    // so a hidden or intermediate size below the group size of 16 does not
    // request a 0-byte allocation. For sizes that divide evenly the byte counts
    // equal `load_moe_no_shared`'s.
    let h = config.hidden_size;
    let inter = config.moe_intermediate_size;
    let group_size = 16usize;
    let ceil_div = |n: usize, d: usize| -> usize { n.div_ceil(d) };
    let gu_packed_bytes = (inter * h).div_ceil(2).max(1);
    let gu_scale_bytes = (inter * ceil_div(h, group_size)).max(1);
    let d_packed_bytes = (h * inter).div_ceil(2).max(1);
    let d_scale_bytes = (h * ceil_div(inter, group_size)).max(1);

    let alloc_zero = |size: usize| -> Result<DevicePtr> {
        let sz = size.max(1);
        let ptr = gpu.alloc(sz)?;
        gpu.memset(ptr, 0, sz)?;
        Ok(ptr)
    };
    let mk_zero_quant = |packed_sz: usize, scale_sz: usize| -> Result<QuantizedWeight> {
        Ok(QuantizedWeight {
            weight: alloc_zero(packed_sz)?,
            weight_scale: alloc_zero(scale_sz)?,
            weight_scale_2: 0.0,
            input_scale: DevicePtr::NULL,
            weight_scale_2_vec: DevicePtr::NULL,
        })
    };
    let shared_expert = ExpertWeight {
        gate_proj: mk_zero_quant(gu_packed_bytes, gu_scale_bytes)?,
        up_proj: mk_zero_quant(gu_packed_bytes, gu_scale_bytes)?,
        down_proj: mk_zero_quant(d_packed_bytes, d_scale_bytes)?,
    };
    let shared_expert_gate = DenseWeight {
        weight: alloc_zero(h * 2)?,
    };

    // 2026-09-25: Per expert key:
    //   * pre-quantized NVFP4 → `quantized_auto`, which takes the store's pointers;
    //   * FP8 with `weight_scale_inv` → `dequant_fp8_blockscaled_to_bf16`, then
    //     `quantize_to_nvfp4`;
    //   * anything else → `dense_auto`, then `quantize_to_nvfp4`.
    // `(n, k)` is the on-disk shape: w1 and w3 are `[moe_intermediate, hidden]`,
    // w2 is `[hidden, moe_intermediate]`.
    let quant = |ep: &str, n: usize, k: usize| -> Result<QuantizedWeight> {
        // 2026-09-25: Pre-quantized means a `weight_packed` key under an NVFP4
        // variant, or a `weight_scale_2` key under `Standard`.
        let is_compressed = store.contains(&format!("{ep}.weight_packed"));
        let is_standard_nvfp4 = matches!(variant, Nvfp4Variant::Standard)
            && store.contains(&format!("{ep}.weight_scale_2"));
        if (matches!(
            variant,
            Nvfp4Variant::Standard | Nvfp4Variant::CompressedTensors
        ) && is_compressed)
            || is_standard_nvfp4
        {
            return quantized_auto(store, ep, gpu, variant);
        }
        // 2026-09-25: After quantizing, the BF16 dequant is freed, and so are an
        // FP8 source weight and its `weight_scale_inv`. The store keeps the freed
        // pointers, and nothing may read these keys afterwards.
        let wkey = format!("{ep}.weight");
        let scale_key = format!("{ep}.weight_scale_inv");
        let (src_ptr, src_is_fp8) = {
            let t = store.get(&wkey)?;
            (
                t.ptr,
                t.dtype == metrale_model_weights::weights::WeightDtype::FP8E4M3,
            )
        };
        let scale_ptr = if store.contains(&scale_key) {
            Some(store.get(&scale_key)?.ptr)
        } else {
            None
        };
        let (dense_w, owned_bf16, need_free_src) = if scale_ptr.is_some() {
            (dequant_fp8_blockscaled_to_bf16(store, ep, gpu)?, true, true)
        } else {
            (dense_auto(store, &wkey, gpu)?, src_is_fp8, src_is_fp8)
        };
        let nvfp4 = quantize_to_nvfp4(&dense_w, n, k, gpu, absmax_k, quantize_k, stream)?;
        if owned_bf16 {
            gpu.free(dense_w.weight)?;
        }
        if need_free_src {
            gpu.free(src_ptr)?;
            if let Some(sp) = scale_ptr {
                gpu.free(sp)?;
            }
        }
        Ok(nvfp4)
    };
    let inter_moe = config.moe_intermediate_size;
    let mut experts = Vec::with_capacity(num_experts);
    for e in 0..num_experts {
        if config.is_local_expert(e) {
            experts.push(ExpertWeight {
                gate_proj: quant(&format!("{p}.experts.{e}.w1"), inter_moe, h)?,
                down_proj: quant(&format!("{p}.experts.{e}.w2"), h, inter_moe)?,
                up_proj: quant(&format!("{p}.experts.{e}.w3"), inter_moe, h)?,
            });
        } else {
            experts.push(ExpertWeight::null());
        }
    }

    Ok(MoeWeights {
        gate,
        shared_expert,
        shared_expert_gate,
        experts,
        router_pre_norm: None,
        correction_bias: Some(correction_bias),
    })
}

/// 2026-09-25: Load a Gemma-4 MoE block.
///
/// - The router is `{lp}.router.proj.weight`, kept unfused as the gate.
/// - `router.scale` `[H]` becomes `router_pre_norm` (`scale * hidden_size^-0.5`).
/// - `router.per_expert_scale` `[E]` is folded into each expert's down_proj
///   `weight_scale_2`.
/// - Experts are `{lp}.moe.experts.{e}`; the shared expert is zero-filled.
/// - The gate and both router scales are read as BF16 without a dtype check.
pub fn load_moe_gemma4(
    store: &WeightStore,
    layer_prefix: &str,
    num_experts: usize,
    gpu: &dyn GpuBackend,
    config: &metrale_config::ModelConfig,
    variant: Nvfp4Variant,
    qctx: QuantizeCtx,
) -> Result<MoeWeights> {
    let h = config.hidden_size;
    let inter = config.moe_intermediate_size;

    let gate_key = format!("{layer_prefix}.router.proj.weight");
    let scale_key = format!("{layer_prefix}.router.scale");
    let per_exp_key = format!("{layer_prefix}.router.per_expert_scale");

    let gate_wt = store.get(&gate_key)?;
    let gate_bytes = num_experts * h * 2;
    let mut gate_buf = vec![0u8; gate_bytes];
    gpu.copy_d2h(gate_wt.ptr, &mut gate_buf)?;

    // 2026-09-25: `MoeLayer::router_input` runs `rms_norm` with this weight
    // before the gate GEMV, so the gate sees `rms_norm(x) * scale * hidden_size^-0.5`.
    let router_pre_norm = if store.contains(&scale_key) {
        let scale_wt = store.get(&scale_key)?;
        let mut scale_buf = vec![0u8; h * 2];
        gpu.copy_d2h(scale_wt.ptr, &mut scale_buf)?;
        let scalar_root = 1.0f32 / (h as f32).sqrt();
        for dim in 0..h {
            let bits = u16::from_le_bytes([scale_buf[dim * 2], scale_buf[dim * 2 + 1]]);
            let f = f32::from_bits((bits as u32) << 16) * scalar_root;
            let out = (f.to_bits() >> 16) as u16;
            scale_buf[dim * 2] = out as u8;
            scale_buf[dim * 2 + 1] = (out >> 8) as u8;
        }
        let pre_norm_ptr = gpu.alloc(h * 2)?;
        gpu.copy_h2d(&scale_buf, pre_norm_ptr)?;
        tracing::info!("Gemma-4 MoE: router pre-norm weight = scale * hidden_size^(-0.5)");
        Some(DenseWeight {
            weight: pre_norm_ptr,
        })
    } else {
        None
    };

    let gate_ptr = gpu.alloc(gate_bytes)?;
    gpu.copy_h2d(&gate_buf, gate_ptr)?;
    let gate = DenseWeight { weight: gate_ptr };

    let group_size = 16usize;
    let gu_packed = inter * h / 2;
    let gu_scale = inter * (h / group_size);
    let d_packed = h * inter / 2;
    let d_scale = h * (inter / group_size);

    let alloc_zero = |size: usize| -> Result<DevicePtr> {
        let ptr = gpu.alloc(size)?;
        gpu.memset(ptr, 0, size)?;
        Ok(ptr)
    };
    let mk_zero = |p: usize, s: usize| -> Result<QuantizedWeight> {
        Ok(QuantizedWeight {
            weight: alloc_zero(p)?,
            weight_scale: alloc_zero(s)?,
            weight_scale_2: 0.0,
            input_scale: DevicePtr::NULL,
            weight_scale_2_vec: DevicePtr::NULL,
        })
    };
    let shared_expert = ExpertWeight {
        gate_proj: mk_zero(gu_packed, gu_scale)?,
        up_proj: mk_zero(gu_packed, gu_scale)?,
        down_proj: mk_zero(d_packed, d_scale)?,
    };
    let shared_expert_gate = DenseWeight {
        weight: alloc_zero(h * 2)?,
    };

    // 2026-09-25: 1.0 for every expert when `router.per_expert_scale` is absent.
    let per_expert_scales: Vec<f32> = if store.contains(&per_exp_key) {
        let per_exp_wt = store.get(&per_exp_key)?;
        let mut buf = vec![0u8; num_experts * 2];
        gpu.copy_d2h(per_exp_wt.ptr, &mut buf)?;
        (0..num_experts)
            .map(|e| {
                let bits = u16::from_le_bytes([buf[e * 2], buf[e * 2 + 1]]);
                f32::from_bits((bits as u32) << 16)
            })
            .collect()
    } else {
        vec![1.0f32; num_experts]
    };

    let mut experts = Vec::with_capacity(num_experts);
    for e in 0..num_experts {
        let ep = format!("{layer_prefix}.moe.experts.{e}");
        if config.is_local_expert(e) {
            let mut down = quantized_any(
                store,
                &format!("{ep}.down_proj"),
                h,
                inter,
                gpu,
                variant,
                qctx,
            )?;
            // 2026-09-25: `weight_scale_2` multiplies every dequantized value of
            // down_proj, so this scales expert e's output by `per_expert_scales[e]`.
            down.weight_scale_2 *= per_expert_scales[e];
            experts.push(ExpertWeight {
                gate_proj: quantized_any(
                    store,
                    &format!("{ep}.gate_proj"),
                    inter,
                    h,
                    gpu,
                    variant,
                    qctx,
                )?,
                up_proj: quantized_any(
                    store,
                    &format!("{ep}.up_proj"),
                    inter,
                    h,
                    gpu,
                    variant,
                    qctx,
                )?,
                down_proj: down,
            });
        } else {
            experts.push(ExpertWeight::null());
        }
    }

    Ok(MoeWeights {
        gate,
        shared_expert,
        shared_expert_gate,
        experts,
        router_pre_norm,
        correction_bias: None,
    })
}
