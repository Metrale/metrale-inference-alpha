// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Qwen3.5-family loaders: the separate-projection linear-attention weights and three MoE loaders (NVFP4, native FP8 experts, no shared expert).
//!
//! Owner: model-layers (weight loading).
//! Invariants: none beyond the types.

#![allow(unused_imports)]

use anyhow::{Context, Result, bail, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_weights::weights::{WeightDtype, WeightStore};

use super::*;

#[path = "ssm_qwen35/dequant_fp8.rs"]
mod dequant_fp8;
use dequant_fp8::dequant_fp8_block_slice_bf16;

/// 2026-09-25: Qwen3.5 linear-attention weights with separate projections. The
/// projections, `conv1d` and `norm` are BF16; `a_log` and `dt_bias` are FP32.
pub struct SsmWeightsQwen35 {
    /// 2026-09-25: Q, K and V projections in one weight (no Z).
    pub in_proj_qkv: DenseWeight,
    pub in_proj_z: DenseWeight,
    pub in_proj_a: DenseWeight,
    pub in_proj_b: DenseWeight,
    pub conv1d: DenseWeight,
    pub a_log: DenseWeight,
    pub dt_bias: DenseWeight,
    pub norm: DenseWeight,
    pub out_proj: DenseWeight,
}

/// 2026-09-25: Load the Qwen3.5 `{layer_prefix}.linear_attn` weights.
pub fn load_ssm_qwen35(
    store: &WeightStore,
    layer_prefix: &str,
    gpu: &dyn GpuBackend,
    // 2026-09-25: Unused: each projection is loaded by its own on-disk dtype.
    _variant: Nvfp4Variant,
) -> Result<SsmWeightsQwen35> {
    let p = format!("{layer_prefix}.linear_attn");

    // 2026-09-25: A projection with `weight_packed` (NVFP4, 2 values per byte) is
    // dequantized to BF16 with dims from the packed shape; any other goes
    // through `dense_auto`.
    let load_proj = |prefix: &str| -> Result<DenseWeight> {
        if store.contains(&format!("{prefix}.weight_packed")) {
            let shape = store.get(&format!("{prefix}.weight_packed"))?.shape.clone();
            dequant_nvfp4_to_bf16(store, prefix, shape[0], shape[1] * 2, gpu)
        } else {
            dense_auto(store, &format!("{prefix}.weight"), gpu)
        }
    };

    Ok(SsmWeightsQwen35 {
        in_proj_qkv: load_proj(&format!("{p}.in_proj_qkv"))?,
        in_proj_z: load_proj(&format!("{p}.in_proj_z"))?,
        in_proj_a: load_proj(&format!("{p}.in_proj_a"))?,
        in_proj_b: load_proj(&format!("{p}.in_proj_b"))?,
        conv1d: dense_auto(store, &format!("{p}.conv1d.weight"), gpu)?,
        a_log: dense_keep_f32(store, &format!("{p}.A_log"), gpu)?,
        dt_bias: dense_keep_f32(store, &format!("{p}.dt_bias"), gpu)?,
        norm: dense_f32_safe(store, &format!("{p}.norm.weight"), gpu)?,
        out_proj: load_proj(&format!("{p}.out_proj"))?,
    })
}

/// 2026-09-25: Load a Qwen3.5 MoE block (`{layer_prefix}.mlp`) as NVFP4.
///
/// Routed experts that this rank does not hold, and all routed experts when
/// `skip_routed_experts` is set, are `ExpertWeight::null()`. The shared expert
/// is always loaded.
pub fn load_moe_qwen35(
    store: &WeightStore,
    layer_prefix: &str,
    num_experts: usize,
    gpu: &dyn GpuBackend,
    config: &metrale_config::ModelConfig,
    variant: Nvfp4Variant,
    absmax_k: metrale_gpu_runtime::gpu::KernelHandle,
    quantize_k: metrale_gpu_runtime::gpu::KernelHandle,
    stream: u64,
    skip_routed_experts: bool,
) -> Result<MoeWeights> {
    let p = format!("{layer_prefix}.mlp");

    let gate = dense_auto(store, &format!("{p}.gate.weight"), gpu)?;
    let shared_expert_gate = dense_auto(store, &format!("{p}.shared_expert_gate.weight"), gpu)?;

    let inter = config.moe_intermediate_size;
    let h = config.hidden_size;

    let qctx = QuantizeCtx {
        absmax_k,
        quantize_k,
        stream,
    };

    let fused_gate_up_key = format!("{p}.experts.gate_up_proj");
    let fused_down_key = format!("{p}.experts.down_proj");
    // 2026-09-25: Fused layout: one `experts.gate_up_proj` `[E, 2*inter, h]` and
    // one `experts.down_proj` `[E, h, inter]` per layer, sliced per expert and
    // quantized to NVFP4. A BF16 slice is quantized directly; an FP8E4M3 slice
    // is first dequantized to BF16 with its `{key}_scale_inv` block scales
    // `[E, sn, sk]`. The dtype decides, not `variant`.
    let is_fused = store.contains(&fused_gate_up_key) && store.contains(&fused_down_key);
    let fused_is_fp8 = is_fused
        && store
            .get(&fused_gate_up_key)
            .map(|w| w.dtype == WeightDtype::FP8E4M3)
            .unwrap_or(false);

    let load_expert_fused = |expert_idx: usize| -> Result<ExpertWeight> {
        let fused_gu = store.get(&fused_gate_up_key)?;
        let fused_d = store.get(&fused_down_key)?;
        if fused_is_fp8 {
            let gu_s = store.get(&format!("{fused_gate_up_key}_scale_inv"))?;
            let d_s = store.get(&format!("{fused_down_key}_scale_inv"))?;
            let (gu_sn, gu_sk) = (gu_s.shape[1], gu_s.shape[2]);
            let (d_sn, d_sk) = (d_s.shape[1], d_s.shape[2]);
            let gu_s_f32 = gu_s.dtype == WeightDtype::FP32;
            let d_s_f32 = d_s.dtype == WeightDtype::FP32;
            let gu_w_stride = 2 * inter * h;
            let d_w_stride = h * inter;
            let gu_s_elem = if gu_s_f32 { 4 } else { 2 };
            let d_s_elem = if d_s_f32 { 4 } else { 2 };
            let gu_s_stride = gu_sn * gu_sk * gu_s_elem;
            let d_s_stride = d_sn * d_sk * d_s_elem;
            // 2026-09-25: Dequantize all of `gate_up[e]`, then take gate as rows
            // `0..inter` and up as rows `inter..2*inter`.
            let gu_bf16 = dequant_fp8_block_slice_bf16(
                gpu,
                fused_gu.ptr.offset(expert_idx * gu_w_stride),
                gu_s.ptr.offset(expert_idx * gu_s_stride),
                2 * inter,
                h,
                gu_sn,
                gu_sk,
                gu_s_f32,
            )?;
            let down_bf16 = dequant_fp8_block_slice_bf16(
                gpu,
                fused_d.ptr.offset(expert_idx * d_w_stride),
                d_s.ptr.offset(expert_idx * d_s_stride),
                h,
                inter,
                d_sn,
                d_sk,
                d_s_f32,
            )?;
            let gate_dw = DenseWeight { weight: gu_bf16 };
            let up_dw = DenseWeight {
                weight: gu_bf16.offset(inter * h * 2),
            };
            let down_dw = DenseWeight { weight: down_bf16 };
            let out = ExpertWeight {
                gate_proj: quantize_to_nvfp4(
                    &gate_dw, inter, h, gpu, absmax_k, quantize_k, stream,
                )?,
                up_proj: quantize_to_nvfp4(&up_dw, inter, h, gpu, absmax_k, quantize_k, stream)?,
                down_proj: quantize_to_nvfp4(
                    &down_dw, h, inter, gpu, absmax_k, quantize_k, stream,
                )?,
            };
            gpu.free(gu_bf16)?;
            gpu.free(down_bf16)?;
            Ok(out)
        } else {
            let bf16 = 2usize;
            let gu_per_expert_bytes = 2 * inter * h * bf16;
            let d_per_expert_bytes = h * inter * bf16;
            let gate_off = expert_idx * gu_per_expert_bytes;
            let up_off = gate_off + inter * h * bf16;
            let down_off = expert_idx * d_per_expert_bytes;
            let gate_dw = DenseWeight {
                weight: fused_gu.ptr.offset(gate_off),
            };
            let up_dw = DenseWeight {
                weight: fused_gu.ptr.offset(up_off),
            };
            let down_dw = DenseWeight {
                weight: fused_d.ptr.offset(down_off),
            };
            Ok(ExpertWeight {
                gate_proj: quantize_to_nvfp4(
                    &gate_dw, inter, h, gpu, absmax_k, quantize_k, stream,
                )?,
                up_proj: quantize_to_nvfp4(&up_dw, inter, h, gpu, absmax_k, quantize_k, stream)?,
                down_proj: quantize_to_nvfp4(
                    &down_dw, h, inter, gpu, absmax_k, quantize_k, stream,
                )?,
            })
        }
    };

    // 2026-09-25: `quantized_any` picks the format per key, so a BF16 expert in
    // an FP8 or NVFP4 checkpoint still loads.
    let load_expert = |prefix: &str| -> Result<ExpertWeight> {
        Ok(ExpertWeight {
            gate_proj: quantized_any(
                store,
                &format!("{prefix}.gate_proj"),
                inter,
                h,
                gpu,
                variant,
                qctx,
            )?,
            up_proj: quantized_any(
                store,
                &format!("{prefix}.up_proj"),
                inter,
                h,
                gpu,
                variant,
                qctx,
            )?,
            down_proj: quantized_any(
                store,
                &format!("{prefix}.down_proj"),
                h,
                inter,
                gpu,
                variant,
                qctx,
            )?,
        })
    };

    let shared_expert = load_expert(&format!("{p}.shared_expert"))?;

    let mut experts = Vec::with_capacity(num_experts);
    for e in 0..num_experts {
        if skip_routed_experts || !config.is_local_expert(e) {
            experts.push(ExpertWeight::null());
        } else if is_fused {
            experts.push(load_expert_fused(e)?);
        } else {
            experts.push(load_expert(&format!("{p}.experts.{e}"))?);
        }
    }

    // 2026-09-25: Every loaded expert has its own NVFP4 copy here, so the fused source
    // tensors are freed; a failed free is ignored. The store keeps the freed
    // pointers, and nothing may read these keys afterwards.
    if is_fused {
        if let Ok(w) = store.get(&fused_gate_up_key) {
            let _ = gpu.free(w.ptr);
        }
        if let Ok(w) = store.get(&fused_down_key) {
            let _ = gpu.free(w.ptr);
        }
    }

    Ok(MoeWeights {
        gate,
        shared_expert,
        shared_expert_gate,
        experts,
        router_pre_norm: None,
        correction_bias: None,
    })
}

/// 2026-09-25: Load the routed experts of `{layer_prefix}.mlp` as native FP8
/// block-scaled tables, one entry per expert id. Experts this rank does not
/// hold are NULL entries tagged `Fp8BlockScaled`.
pub fn load_moe_qwen35_fp8_experts(
    store: &WeightStore,
    layer_prefix: &str,
    num_experts: usize,
    gpu: &dyn GpuBackend,
    config: &metrale_config::ModelConfig,
) -> Result<Vec<Fp8ExpertWeight>> {
    let p = format!("{layer_prefix}.mlp");
    let mut fp8_experts = Vec::with_capacity(num_experts);

    for e in 0..num_experts {
        if config.is_local_expert(e) {
            let ep = format!("{p}.experts.{e}");
            fp8_experts.push(Fp8ExpertWeight {
                gate_proj: load_fp8_block_scaled_as_fp8weight(
                    store,
                    &format!("{ep}.gate_proj"),
                    gpu,
                )?,
                up_proj: load_fp8_block_scaled_as_fp8weight(store, &format!("{ep}.up_proj"), gpu)?,
                down_proj: load_fp8_block_scaled_as_fp8weight(
                    store,
                    &format!("{ep}.down_proj"),
                    gpu,
                )?,
            });
        } else {
            let null_block = Fp8Weight {
                weight: DevicePtr::NULL,
                row_scale: DevicePtr::NULL,
                n: 0,
                k: 0,
                scale_format: WeightQuantFormat::Fp8BlockScaled,
            };
            fp8_experts.push(Fp8ExpertWeight {
                gate_proj: null_block,
                up_proj: null_block,
                down_proj: null_block,
            });
        }
    }

    // 2026-09-25: The shared expert's FP8 tables are loaded (so a missing or
    // malformed one is an error) and then dropped: `_shared_fp8` is unused.
    let shared_prefix = format!("{p}.shared_expert");
    let _shared_fp8 = Fp8ExpertWeight {
        gate_proj: load_fp8_block_scaled_as_fp8weight(
            store,
            &format!("{shared_prefix}.gate_proj"),
            gpu,
        )?,
        up_proj: load_fp8_block_scaled_as_fp8weight(
            store,
            &format!("{shared_prefix}.up_proj"),
            gpu,
        )?,
        down_proj: load_fp8_block_scaled_as_fp8weight(
            store,
            &format!("{shared_prefix}.down_proj"),
            gpu,
        )?,
    };

    Ok(fp8_experts)
}

/// 2026-09-25: Load a MoE block that has no shared expert. A zero-filled shared
/// expert of real expert size stands in: its packed codes, scales and
/// `weight_scale_2` are all 0, so it contributes 0. Routed experts go through
/// `quantized_auto`, which panics for `Fp8Dequanted` and `Bf16Raw`.
pub fn load_moe_no_shared(
    store: &WeightStore,
    layer_prefix: &str,
    num_experts: usize,
    gpu: &dyn GpuBackend,
    config: &metrale_config::ModelConfig,
    variant: Nvfp4Variant,
) -> Result<MoeWeights> {
    let p = format!("{layer_prefix}.mlp");

    let gate = dense(store, &format!("{p}.gate.weight"))?;

    let h = config.hidden_size;
    let inter = config.moe_intermediate_size;
    let group_size = 16usize;

    let gu_packed_bytes = inter * h / 2;
    let gu_scale_bytes = inter * (h / group_size);
    let d_packed_bytes = h * inter / 2;
    let d_scale_bytes = h * (inter / group_size);

    let alloc_zero = |size: usize| -> Result<DevicePtr> {
        let ptr = gpu.alloc(size)?;
        gpu.memset(ptr, 0, size)?;
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
    // 2026-09-25: A zero gate gives sigmoid(0) = 0.5, times a zero shared output.
    let shared_expert_gate = DenseWeight {
        weight: alloc_zero(h * 2)?,
    };

    let mut experts = Vec::with_capacity(num_experts);
    for e in 0..num_experts {
        if config.is_local_expert(e) {
            experts.push(ExpertWeight {
                gate_proj: quantized_auto(
                    store,
                    &format!("{p}.experts.{e}.gate_proj"),
                    gpu,
                    variant,
                )?,
                up_proj: quantized_auto(store, &format!("{p}.experts.{e}.up_proj"), gpu, variant)?,
                down_proj: quantized_auto(
                    store,
                    &format!("{p}.experts.{e}.down_proj"),
                    gpu,
                    variant,
                )?,
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
        correction_bias: None,
    })
}
