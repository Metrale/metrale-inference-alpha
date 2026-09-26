// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The linear-attention arm of `load_layers`: picks layer `i`'s linear-attention
//! builder and builds the layer.
//!
//! Owner: model-arch weight loader (Qwen3.5).
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_model_layers::layer::TransformerLayer;
use metrale_model_layers::weight_map::Nvfp4Variant;

use super::load_cx::{LayerIn, LoadCx};
use super::proj_is_native_fp8;

/// 2026-09-26: Layer `i`'s linear-attention layer, from the builder the rules below select.
pub(super) fn build_linear_attention(
    cx: &LoadCx,
    lp: &str,
    i: usize,
    force_nvfp4_all: bool,
    fp4_proj_decode: bool,
    parts: LayerIn,
) -> Result<Box<dyn TransformerLayer>> {
    let LoadCx {
        store,
        config,
        gpu,
        variant,
        h,
        absmax_k,
        quantize_k,
        stream,
        modelopt_mixed_precision,
        native_modelopt_ssm,
        ..
    } = *cx;
    let LayerIn {
        input_norm,
        post_attn_norm,
        ffn,
    } = parts;
    // 2026-09-25: LinearAttention builders, first match wins:
    // - `METRALE_HOLO_NATIVE_FP8_SSM=1` on the Holo ModelOpt checkpoint: native FP8;
    // - the Holo ModelOpt checkpoint, unless `METRALE_HOLO_FP4_PROJ_DECODE=1`: BF16 dense;
    // - `Fp8Dequanted` with a block-scaled FP8 SSM, unless `METRALE_FORCE_NVFP4_ALL=1` or
    //   `METRALE_HOLO_FP4_PROJ_DECODE=1`: native FP8, whose decode and prefill both read
    //   the block-scaled weights (`qwen3_ssm/trait_prefill_proj.rs` `qkvz_fp8w` arms);
    // - everything else: NVFP4.
    // 2026-09-25: Native FP8 needs `in_proj_qkv` to be block-scaled FP8
    // (`proj_is_native_fp8`). An FP8 checkpoint whose SSM is not goes to the NVFP4
    // builder as `Bf16Raw`, so the builder does not add the `Fp8Dequanted` FP8 casts.
    let ssm_native_fp8 = proj_is_native_fp8(store, &format!("{lp}.linear_attn.in_proj_qkv"));
    let ssm_variant = if matches!(variant, Nvfp4Variant::Fp8Dequanted) && !ssm_native_fp8 {
        Nvfp4Variant::Bf16Raw
    } else {
        variant
    };
    let layer = match variant {
        _ if native_modelopt_ssm => super::linear_attn_arms::build_linear_attention_fp8(
            i,
            store,
            lp,
            gpu,
            variant,
            config,
            h,
            stream,
            input_norm,
            post_attn_norm,
            ffn,
        )?,
        // 2026-09-25: `METRALE_HOLO_FP4_PROJ_DECODE=1` sends the Holo ModelOpt SSM to
        // the NVFP4 builder instead.
        _ if modelopt_mixed_precision && !fp4_proj_decode => {
            super::linear_attn_arms::build_linear_attention_dense_bf16(
                i,
                store,
                lp,
                gpu,
                variant,
                config,
                h,
                input_norm,
                post_attn_norm,
                ffn,
            )?
        }
        Nvfp4Variant::Fp8Dequanted if !(force_nvfp4_all || fp4_proj_decode) && ssm_native_fp8 => {
            super::linear_attn_arms::build_linear_attention_fp8(
                i,
                store,
                lp,
                gpu,
                variant,
                config,
                h,
                stream,
                input_norm,
                post_attn_norm,
                ffn,
            )?
        }
        _ => super::linear_attn_arms::build_linear_attention_nvfp4(
            store,
            lp,
            gpu,
            ssm_variant,
            config,
            h,
            absmax_k,
            quantize_k,
            stream,
            input_norm,
            post_attn_norm,
            ffn,
        )?,
    };
    Ok(layer)
}
