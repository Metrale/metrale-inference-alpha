// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Nemotron-H weight structs and loaders for the Mamba-2 mixer, attention and MoE layers.
//!
//! Owner: model-layers (weight loading).
//! Invariants: none beyond the types.

#![allow(unused_imports)]

use anyhow::{Context, Result, bail, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_weights::weights::{WeightDtype, WeightStore};

use super::*;

/// 2026-09-25: Nemotron-H Mamba-2 mixer weights.
///
/// in_proj produces `[z(d_inner), x(d_inner), B(n_groups*state), C(n_groups*state), dt(num_heads)]`.
pub struct NemotronSsmWeights {
    /// 2026-09-25: NVFP4, or `QuantizedWeight::null()` when the layer is not NVFP4.
    pub in_proj: QuantizedWeight,
    /// 2026-09-25: NVFP4, or `QuantizedWeight::null()` when the layer is not NVFP4.
    pub out_proj: QuantizedWeight,
    pub conv1d_weight: DenseWeight,
    /// 2026-09-25: F32, widened from the BF16 checkpoint tensor at load.
    pub conv1d_bias: DenseWeight,
    /// 2026-09-25: F32, widened from the BF16 checkpoint tensor at load.
    pub a_log: DenseWeight,
    /// 2026-09-25: The D skip connection; F32, widened from BF16 at load.
    pub d_param: DenseWeight,
    /// 2026-09-25: F32, widened from the BF16 checkpoint tensor at load.
    pub dt_bias: DenseWeight,
    pub ssm_norm: DenseWeight,
}

/// 2026-09-25: Nemotron-H expert: `up_proj` and `down_proj`, no `gate_proj`.
#[derive(Debug, Clone, Copy)]
pub struct NemotronExpertWeight {
    pub up_proj: QuantizedWeight,
    pub down_proj: QuantizedWeight,
}

impl NemotronExpertWeight {
    pub fn null() -> Self {
        Self {
            up_proj: QuantizedWeight::null(),
            down_proj: QuantizedWeight::null(),
        }
    }
}

/// 2026-09-25: Nemotron-H MoE layer weights.
pub struct NemotronMoeWeights {
    /// 2026-09-25: Router gate, BF16 (converted when the checkpoint stores FP32).
    pub gate: DenseWeight,
    /// 2026-09-25: The store's `gate.e_score_correction_bias` pointer, unconverted;
    /// `moe_topk_sigmoid` reads it as F32.
    pub e_score_correction_bias: DenseWeight,
    /// 2026-09-25: Routed experts as NVFP4; experts this rank does not hold are null.
    pub experts: Vec<NemotronExpertWeight>,
    /// 2026-09-25: Shared-expert up_proj as NVFP4; NULL when `shared_up_fp8` is `Some`.
    pub shared_up: QuantizedWeight,
    /// 2026-09-25: Shared-expert up_proj as the checkpoint's own FP8 bytes, in an
    /// owned copy. `Some` when `METRALE_NEMOTRON_NATIVE_FP8_SSM` is unset, `1`,
    /// `both` or `decode`, the tensor has `weight_scale` but no `weight_scale_2`,
    /// and it loads.
    pub shared_up_fp8: Option<Fp8Weight>,
    /// 2026-09-25: Shared-expert down_proj as NVFP4; NULL when `shared_down_fp8` is `Some`.
    pub shared_down: QuantizedWeight,
    /// 2026-09-25: Shared-expert down_proj as native FP8, under the same condition
    /// as `shared_up_fp8`. When its kernels are available, decode applies relu²
    /// in place and `w8a16_gemv` to it, and launches the fused NVFP4 kernel
    /// without its shared slot.
    pub shared_down_fp8: Option<Fp8Weight>,
    /// 2026-09-25: LatentMoE fc1, BF16 (FP8 is dequantized at load). `Some` when
    /// `config.moe_latent_size > 0`.
    pub fc1_latent_proj: Option<DenseWeight>,
    /// 2026-09-25: LatentMoE fc2, BF16 (FP8 is dequantized at load). `Some` when
    /// `config.moe_latent_size > 0`.
    pub fc2_latent_proj: Option<DenseWeight>,
}

/// 2026-09-25: Format of a Mamba-2 layer's `in_proj`, detected at load.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NemotronSsmQuant {
    /// 2026-09-25: `weight_scale` and `weight_scale_2` present.
    Nvfp4,
    /// 2026-09-25: `weight_scale` without `weight_scale_2`.
    Fp8,
    /// 2026-09-25: No `weight_scale`.
    Bf16,
}

/// 2026-09-25: Load a Nemotron-H Mamba-2 mixer (`{layer_prefix}.mixer`).
///
/// The format is read from `in_proj` alone. Unless it is NVFP4, `in_proj` and
/// `out_proj` come back as `QuantizedWeight::null()`, and the returned
/// `NemotronSsmQuant` tells the caller how to load them.
pub fn load_nemotron_ssm(
    store: &WeightStore,
    _layer: usize,
    gpu: &dyn GpuBackend,
    layer_prefix: &str,
) -> Result<(NemotronSsmWeights, NemotronSsmQuant)> {
    let p = format!("{layer_prefix}.mixer");
    let has_scale = store.contains(&format!("{p}.in_proj.weight_scale"));
    let has_scale2 = store.contains(&format!("{p}.in_proj.weight_scale_2"));
    let quant = if has_scale && has_scale2 {
        NemotronSsmQuant::Nvfp4
    } else if has_scale {
        NemotronSsmQuant::Fp8
    } else {
        NemotronSsmQuant::Bf16
    };
    let in_proj = if quant == NemotronSsmQuant::Nvfp4 {
        quantized(store, &format!("{p}.in_proj"), gpu)?
    } else {
        QuantizedWeight::null()
    };
    let out_proj = if quant == NemotronSsmQuant::Nvfp4 {
        quantized(store, &format!("{p}.out_proj"), gpu)?
    } else {
        QuantizedWeight::null()
    };
    Ok((
        NemotronSsmWeights {
            in_proj,
            out_proj,
            conv1d_weight: dense(store, &format!("{p}.conv1d.weight"))?,
            conv1d_bias: dense_bf16_as_f32(store, &format!("{p}.conv1d.bias"), gpu)?,
            a_log: dense_bf16_as_f32(store, &format!("{p}.A_log"), gpu)?,
            d_param: dense_bf16_as_f32(store, &format!("{p}.D"), gpu)?,
            dt_bias: dense_bf16_as_f32(store, &format!("{p}.dt_bias"), gpu)?,
            ssm_norm: dense(store, &format!("{p}.norm.weight"))?,
        },
        quant,
    ))
}

/// 2026-09-25: Load a Nemotron-H attention mixer.
///
/// Returns `(attn, q, k, v, o_dense, is_nvfp4)`. When `q_proj` has a
/// `weight_scale_2`, q/k/v are `Some` NVFP4 weights, `attn.o_proj` is NVFP4
/// and the dense fields are NULL. Otherwise q/k/v/o are BF16 in `attn` and
/// `o_dense` (an FP8 projection with `weight_scale` is dequantized), q/k/v are
/// `None`, and `attn.o_proj` is null. The q/k norms are always NULL.
pub fn load_nemotron_attention(
    store: &WeightStore,
    layer: usize,
    gpu: &dyn GpuBackend,
    layer_prefix: &str,
) -> Result<(
    AttentionWeights,
    Option<QuantizedWeight>,
    Option<QuantizedWeight>,
    Option<QuantizedWeight>,
    DenseWeight,
    bool,
)> {
    let p = format!("{layer_prefix}.mixer");
    let is_nvfp4 = store.contains(&format!("{p}.q_proj.weight_scale_2"));
    let is_fp8 = !is_nvfp4 && store.contains(&format!("{p}.q_proj.weight_scale"));
    let dummy = DenseWeight {
        weight: DevicePtr::NULL,
    };

    let (q_dense, k_dense, v_dense, o_dense, o_proj, q_nvfp4, k_nvfp4, v_nvfp4) = if is_nvfp4 {
        let q = quantized(store, &format!("{p}.q_proj"), gpu)?;
        let k = quantized(store, &format!("{p}.k_proj"), gpu)?;
        let v = quantized(store, &format!("{p}.v_proj"), gpu)?;
        let o = quantized(store, &format!("{p}.o_proj"), gpu)?;
        (dummy, dummy, dummy, dummy, o, Some(q), Some(k), Some(v))
    } else {
        let load_proj = |name: &str| -> Result<DenseWeight> {
            let prefix = format!("{p}.{name}");
            if store.contains(&format!("{prefix}.weight_scale")) {
                dequant_fp8_to_bf16(store, &prefix, gpu)
            } else {
                dense(store, &format!("{prefix}.weight"))
            }
        };
        let q = load_proj("q_proj")?;
        let k = load_proj("k_proj")?;
        let v = load_proj("v_proj")?;
        let o = load_proj("o_proj")?;
        if is_fp8 && layer < 2 {
            tracing::info!("L{layer} Attention: FP8 → BF16 (runtime quantization to NVFP4)");
        }
        (q, k, v, o, QuantizedWeight::null(), None, None, None)
    };

    let (k_scale, v_scale) = load_kv_scales(store, &p, gpu);
    let attn = AttentionWeights {
        q_proj: q_dense,
        k_proj: k_dense,
        v_proj: v_dense,
        o_proj,
        q_norm: dummy,
        k_norm: dummy,
        q_norm_full: None,
        k_norm_full: None,
        k_scale,
        v_scale,
    };
    Ok((attn, q_nvfp4, k_nvfp4, v_nvfp4, o_dense, is_nvfp4))
}

/// 2026-09-25: Load a Nemotron-H MoE mixer.
///
/// Expert projections that are not NVFP4 on disk (FP8 or BF16) are quantized
/// to NVFP4 at load, except shared-expert projections kept as native FP8.
/// That needs `absmax_k` and `quantize_k` (it panics without them). Expert FP8
/// dequants go into `scratch` when it is given. The LatentMoE projections stay BF16.
pub fn load_nemotron_moe(
    store: &WeightStore,
    layer: usize,
    num_experts: usize,
    gpu: &dyn GpuBackend,
    config: &metrale_config::ModelConfig,
    absmax_k: Option<metrale_gpu_runtime::gpu::KernelHandle>,
    quantize_k: Option<metrale_gpu_runtime::gpu::KernelHandle>,
    stream: u64,
    scratch: Option<DevicePtr>,
    layer_prefix: &str,
) -> Result<NemotronMoeWeights> {
    let p = format!("{layer_prefix}.mixer");
    let gate_name = format!("{p}.gate.weight");
    let gate_w = store.get(&gate_name)?;
    let gate = if gate_w.dtype == WeightDtype::FP32 {
        dense_f32_as_bf16(store, &gate_name, gpu)?
    } else {
        DenseWeight { weight: gate_w.ptr }
    };
    let e_score_correction_bias = dense(store, &format!("{p}.gate.e_score_correction_bias"))?;

    // 2026-09-25: `weight_scale_2` marks NVFP4; `weight_scale` alone marks FP8.
    let shared_up_prefix = format!("{p}.shared_experts.up_proj");
    let shared_up_has_s2 = store.contains(&format!("{shared_up_prefix}.weight_scale_2"));
    let shared_up_has_s = store.contains(&format!("{shared_up_prefix}.weight_scale"));
    // 2026-09-25: The same `METRALE_NEMOTRON_NATIVE_FP8_SSM` modes as the Mamba-2
    // loader (`weight_loader/nemotron/ssm_layer.rs`): unset, `1`, `both` or
    // `decode` keep an FP8 shared expert as FP8.
    let native_fp8_mode =
        std::env::var("METRALE_NEMOTRON_NATIVE_FP8_SSM").unwrap_or_else(|_| "1".to_string());
    let want_native_fp8 = matches!(native_fp8_mode.as_str(), "1" | "both" | "decode");
    let shared_up_fp8 = if want_native_fp8 && !shared_up_has_s2 && shared_up_has_s {
        match load_fp8_block_scaled_as_fp8weight(store, &shared_up_prefix, gpu) {
            Ok(mut w) => {
                let bytes = (w.n as usize) * (w.k as usize);
                let owned = gpu.alloc(bytes)?;
                gpu.copy_d2d(w.weight, owned, bytes)?;
                w.weight = owned;
                Some(w)
            }
            Err(e) => {
                tracing::warn!("shared_up native FP8 unavailable ({e}) — using NVFP4 requant");
                None
            }
        }
    } else {
        None
    };
    let shared_up = if shared_up_fp8.is_some() {
        QuantizedWeight::null()
    } else if shared_up_has_s2 {
        quantized(store, &shared_up_prefix, gpu)?
    } else {
        let bf16 = if shared_up_has_s {
            if let Some(s) = scratch {
                dequant_fp8_to_bf16_into(store, &shared_up_prefix, gpu, s)?
            } else {
                dequant_fp8_to_bf16(store, &shared_up_prefix, gpu)?
            }
        } else {
            dense(store, &format!("{shared_up_prefix}.weight"))?
        };
        quantize_to_nvfp4(
            &bf16,
            config.shared_expert_intermediate_size,
            config.hidden_size,
            gpu,
            absmax_k.unwrap(),
            quantize_k.unwrap(),
            stream,
        )?
    };

    let shared_down_prefix = format!("{p}.shared_experts.down_proj");
    let shared_down_has_s2 = store.contains(&format!("{shared_down_prefix}.weight_scale_2"));
    let shared_down_has_s = store.contains(&format!("{shared_down_prefix}.weight_scale"));
    let shared_down_fp8 = if want_native_fp8 && !shared_down_has_s2 && shared_down_has_s {
        match load_fp8_block_scaled_as_fp8weight(store, &shared_down_prefix, gpu) {
            Ok(mut w) => {
                let bytes = (w.n as usize) * (w.k as usize);
                let owned = gpu.alloc(bytes)?;
                gpu.copy_d2d(w.weight, owned, bytes)?;
                w.weight = owned;
                Some(w)
            }
            Err(e) => {
                tracing::warn!("shared_down native FP8 unavailable ({e}) — using NVFP4 requant");
                None
            }
        }
    } else {
        None
    };
    let shared_down = if shared_down_fp8.is_some() {
        QuantizedWeight::null()
    } else if shared_down_has_s2 {
        quantized(store, &shared_down_prefix, gpu)?
    } else {
        let bf16 = if shared_down_has_s {
            if let Some(s) = scratch {
                dequant_fp8_to_bf16_into(store, &shared_down_prefix, gpu, s)?
            } else {
                dequant_fp8_to_bf16(store, &shared_down_prefix, gpu)?
            }
        } else {
            dense(store, &format!("{shared_down_prefix}.weight"))?
        };
        quantize_to_nvfp4(
            &bf16,
            config.hidden_size,
            config.shared_expert_intermediate_size,
            gpu,
            absmax_k.unwrap(),
            quantize_k.unwrap(),
            stream,
        )?
    };

    // 2026-09-25: fc1 and fc2 stay BF16 for the model's lifetime, so their
    // dequants get their own allocations, never `scratch`.
    let (fc1_latent_proj, fc2_latent_proj) = if config.moe_latent_size > 0 {
        let fc1_prefix = format!("{p}.fc1_latent_proj");
        let fc1 = if store.contains(&format!("{fc1_prefix}.weight_scale")) {
            dequant_fp8_to_bf16(store, &fc1_prefix, gpu)?
        } else {
            dense(store, &format!("{fc1_prefix}.weight"))?
        };
        let fc2_prefix = format!("{p}.fc2_latent_proj");
        let fc2 = if store.contains(&format!("{fc2_prefix}.weight_scale")) {
            dequant_fp8_to_bf16(store, &fc2_prefix, gpu)?
        } else {
            dense(store, &format!("{fc2_prefix}.weight"))?
        };
        (Some(fc1), Some(fc2))
    } else {
        (None, None)
    };

    // 2026-09-25: The first local expert's `up_proj` decides the format of every
    // routed expert. The intermediate size is per layer
    // (`moe_intermediate_size_for`).
    let moe_input = config.moe_input_size();
    let moe_inter = config.moe_intermediate_size_for(layer);
    let first_local = (0..num_experts).find(|e| config.is_local_expert(*e));
    let experts_are_nvfp4 = first_local
        .is_none_or(|e| store.contains(&format!("{p}.experts.{e}.up_proj.weight_scale_2")));
    let experts_are_fp8 = !experts_are_nvfp4
        && first_local
            .is_some_and(|e| store.contains(&format!("{p}.experts.{e}.up_proj.weight_scale")));
    if !experts_are_nvfp4 && layer < 2 {
        tracing::info!(
            "L{layer} MoE experts: {} → NVFP4 (runtime quantization, {} experts)",
            if experts_are_fp8 { "FP8" } else { "BF16" },
            num_experts,
        );
    }

    let mut experts = Vec::with_capacity(num_experts);
    for e in 0..num_experts {
        if config.is_local_expert(e) {
            let up_prefix = format!("{p}.experts.{e}.up_proj");
            let down_prefix = format!("{p}.experts.{e}.down_proj");
            let (up_proj, down_proj) = if experts_are_nvfp4 {
                (
                    quantized(store, &up_prefix, gpu)?,
                    quantized(store, &down_prefix, gpu)?,
                )
            } else {
                let up_bf16 = if experts_are_fp8 {
                    if let Some(s) = scratch {
                        dequant_fp8_to_bf16_into(store, &up_prefix, gpu, s)?
                    } else {
                        dequant_fp8_to_bf16(store, &up_prefix, gpu)?
                    }
                } else {
                    dense(store, &format!("{up_prefix}.weight"))?
                };
                let up = quantize_to_nvfp4(
                    &up_bf16,
                    moe_inter,
                    moe_input,
                    gpu,
                    absmax_k.unwrap(),
                    quantize_k.unwrap(),
                    stream,
                )?;
                let down_bf16 = if experts_are_fp8 {
                    if let Some(s) = scratch {
                        dequant_fp8_to_bf16_into(store, &down_prefix, gpu, s)?
                    } else {
                        dequant_fp8_to_bf16(store, &down_prefix, gpu)?
                    }
                } else {
                    dense(store, &format!("{down_prefix}.weight"))?
                };
                let down = quantize_to_nvfp4(
                    &down_bf16,
                    moe_input,
                    moe_inter,
                    gpu,
                    absmax_k.unwrap(),
                    quantize_k.unwrap(),
                    stream,
                )?;
                (up, down)
            };
            experts.push(NemotronExpertWeight { up_proj, down_proj });
        } else {
            experts.push(NemotronExpertWeight::null());
        }
    }

    Ok(NemotronMoeWeights {
        gate,
        e_score_correction_bias,
        experts,
        shared_up,
        shared_up_fp8,
        shared_down_fp8,
        shared_down,
        fc1_latent_proj,
        fc2_latent_proj,
    })
}
