// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: [`load_full_attention`], the `LayerType::FullAttention` arm of
//! `Qwen35DenseWeightLoader::load_layers`: it picks the attention weight form (keep-packed
//! Q2_0, or one of the `attn_arms` builders by NVFP4 variant), builds the layer, adds the
//! prefill copies and the native FP8 overlay, and pushes the layer.
//!
//! Owner: model-arch weight loader (Qwen3.5 dense).
//! Invariants:
//! - `attn_idx` advances by exactly one for every layer pushed here.

use anyhow::Result;
use metrale_model_layers::layer::TransformerLayer;
use metrale_model_layers::layers::Qwen3AttentionLayer;
use metrale_model_layers::layers::qwen3_attention::Fp8TwinSet;
use metrale_model_layers::weight_map::{
    DenseWeight, Fp8Weight, Nvfp4Variant, load_fp8_block_scaled_as_fp8weight,
};

use super::attn_arms;
use super::fp8_residency::{self, DerivedResidency};
use super::load_cx::{Flow, LayerIn, LoadCx};
use super::{dense_fp8_enabled, proj_is_native_fp8, proj_q2_group};
use crate::tp_shard::{TpShardKind, load_qkvo_tp};

/// 2026-09-26: Builds full-attention layer `l.i`, pushes it to `layers` and advances
/// `attn_idx`. Returns [`Flow::Continue`] for the keep-packed Q2_0 arm, which logs its own
/// progress line, and [`Flow::Proceed`] otherwise.
pub(super) fn load_full_attention(
    cx: &LoadCx<'_>,
    residency: &mut DerivedResidency,
    layers: &mut Vec<Box<dyn TransformerLayer>>,
    attn_idx: &mut usize,
    l: LayerIn<'_>,
) -> Result<Flow> {
    let LoadCx {
        store,
        config,
        gpu,
        layer_kv_dtypes,
        variant,
        route_env,
        ..
    } = *cx;
    let LayerIn {
        i,
        lp,
        input_norm,
        post_attn_norm,
        ffn,
    } = l;
    let p = format!("{lp}.self_attn");
    let tp_rank = config.tp_rank;
    let tp_size = config.tp_world_size.max(1);
    // 2026-09-25: Set by the Bf16Raw arm and installed after the layer is built.
    let mut o_dense_bf16: Option<DenseWeight> = None;

    // 2026-09-25: Keep-packed Q2_0 q/k/v/o (TP=1, unless `METRALE_NO_Q2_ATTN`) are
    // installed with `set_packed_q2_weights`; no NVFP4 copies are built.
    let attn_q2 = tp_size == 1
        && std::env::var_os("METRALE_NO_Q2_ATTN").is_none()
        && proj_q2_group(store, &format!("{p}.q_proj")).is_some()
        && proj_q2_group(store, &format!("{p}.k_proj")).is_some()
        && proj_q2_group(store, &format!("{p}.v_proj")).is_some()
        && proj_q2_group(store, &format!("{p}.o_proj")).is_some();
    if attn_q2 {
        let attn_layer = attn_arms::build_q2_attention(
            cx,
            &p,
            *attn_idx,
            LayerIn {
                i,
                lp,
                input_norm,
                post_attn_norm,
                ffn,
            },
        )?;
        layers.push(Box::new(attn_layer));
        *attn_idx += 1;
        if (i + 1) % 10 == 0 {
            tracing::info!(target: "metrale_model_arch::weight_loader::qwen35_dense", "Loaded layers 0..{}", i + 1);
        }
        return Ok(Flow::Continue);
    }
    // 2026-09-25: Whether the native FP8 overlay below replaces q/k/v/o; decided
    // before the NVFP4 build so `RouteEnv::attn_nvfp4` can skip building copies
    // nothing reads.
    let attn_fp8 = dense_fp8_enabled()
        && config.tp_world_size.max(1) == 1
        && matches!(variant, Nvfp4Variant::Fp8Dequanted)
        && proj_is_native_fp8(store, &format!("{p}.q_proj"));
    let attn_nvfp4 = route_env.attn_nvfp4(attn_fp8);
    let (attn, q_nvfp4, k_nvfp4, v_nvfp4) = match variant {
        Nvfp4Variant::CompressedTensors => {
            attn_arms::compressed_tensors_arm(cx, &p, tp_rank, tp_size)?
        }
        Nvfp4Variant::Standard | Nvfp4Variant::Fp8Dequanted if !attn_nvfp4 => {
            attn_arms::nvfp4_skipped_arm(cx, residency, &p, attn_fp8)?
        }
        Nvfp4Variant::Standard | Nvfp4Variant::Fp8Dequanted => {
            attn_arms::bf16_then_nvfp4_arm(cx, &p, tp_rank, tp_size)?
        }
        Nvfp4Variant::Bf16Raw => {
            attn_arms::bf16_raw_arm(cx, &p, tp_rank, tp_size, &mut o_dense_bf16)?
        }
    };

    let mut attn_layer = Qwen3AttentionLayer::new(
        input_norm,
        attn,
        post_attn_norm,
        ffn,
        *attn_idx,
        q_nvfp4,
        k_nvfp4,
        v_nvfp4,
        gpu,
        layer_kv_dtypes[*attn_idx],
        config.fp8_kv_calibration_tokens,
        config,
    )?;
    // 2026-09-25: Transposed NVFP4 copies of q/k/v/o for the prefill GEMMs.
    if let (Some(qw), Some(kw), Some(vw)) = (q_nvfp4, k_nvfp4, v_nvfp4) {
        let (nh, hd) = (config.num_attention_heads, config.head_dim);
        let (nkv, hh) = (config.num_key_value_heads, config.hidden_size);
        let q_n = nh * hd * if config.attn_gated { 2 } else { 1 };
        let qt = qw.transpose_for_gemm(gpu, q_n, hh)?;
        let kt = kw.transpose_for_gemm(gpu, nkv * hd, hh)?;
        let vt = vw.transpose_for_gemm(gpu, nkv * hd, hh)?;
        let op = &attn_layer.attn.o_proj;
        let ot = op.transpose_for_gemm(gpu, hh, nh * hd)?;
        attn_layer.set_prefill_weights(Some(qt), Some(kt), Some(vt), Some(ot));
        // 2026-09-25: A fused `[q|k|v]` transposed copy runs the three prefill
        // projections in one launch. The GEMM applies one `weight_scale_2` per
        // launch, so it is built only when the three are bit-equal.
        let scales_equal = qw.weight_scale_2.to_bits() == kw.weight_scale_2.to_bits()
            && kw.weight_scale_2.to_bits() == vw.weight_scale_2.to_bits();
        if scales_equal {
            let fused =
                metrale_model_layers::weight_map::QuantizedWeight::transpose_concat_for_gemm(
                    gpu,
                    &[(&qw, q_n), (&kw, nkv * hd), (&vw, nkv * hd)],
                    hh,
                )?;
            attn_layer.set_fused_qkv_prefill_weight(Some(fused));
        } else if *attn_idx == 0 {
            tracing::warn!(target: "metrale_model_arch::weight_loader::qwen35_dense", "attention q/k/v have differing weight_scale_2 — fused QKV GEMM disabled (3 separate launches per layer)"
            );
        }
    }
    // 2026-09-25: Set only on the Bf16Raw arm, whose q_nvfp4 is None, so the block
    // above did not run for it.
    if let Some(o_dense) = o_dense_bf16 {
        attn_layer.set_o_dense_bf16(o_dense);
    }
    // 2026-09-25: Native FP8 q/k/v/o (`attn_fp8`): `set_fp8_weights` replaces the
    // decode weights, and `RouteEnv::attn_nvfp4` decided above whether NVFP4 copies
    // were built.
    if attn_fp8 {
        install_fp8_overlay(cx, residency, &mut attn_layer, &p, i)?;
    }
    layers.push(Box::new(attn_layer));
    *attn_idx += 1;
    Ok(Flow::Proceed)
}

/// 2026-09-26: The native FP8 q/k/v/o overlay of an `attn_fp8` layer and the FP8 prefill
/// twins `RouteEnv::attn_fp8_twins` selects. A failed twin transpose is logged, not
/// returned.
fn install_fp8_overlay(
    cx: &LoadCx<'_>,
    residency: &mut DerivedResidency,
    attn_layer: &mut Qwen3AttentionLayer,
    p: &str,
    i: usize,
) -> Result<()> {
    let LoadCx {
        store,
        config,
        gpu,
        stream,
        route_env,
        ..
    } = *cx;
    let load_fp8_proj =
        |name: &str, _n: usize, _k: usize, _kind: TpShardKind| -> Result<Fp8Weight> {
            // 2026-09-25: Only the widened `row_scale` is a new allocation, so the
            // store adopts it; the E4M3 bytes stay the store's.
            let w = load_fp8_block_scaled_as_fp8weight(store, &format!("{p}.{name}"), gpu)?;
            let bytes = (w.n as usize).div_ceil(128) * (w.k as usize).div_ceil(128) * 4;
            store
                .derived()
                .adopt("attn fp8 block scale (widened)", w.row_scale, bytes);
            Ok(w)
        };
    let [q_fp8, k_fp8, v_fp8, o_fp8] = load_qkvo_tp(config, load_fp8_proj)?;
    attn_layer.set_fp8_weights(Some(q_fp8), Some(k_fp8), Some(v_fp8), Some(o_fp8));
    // 2026-09-25: Build only the FP8 prefill twins `RouteEnv::attn_fp8_twins`
    // selects (`fp8_residency.rs` says why K and V always are), handing each to
    // the store so teardown releases it.
    let want = route_env.attn_fp8_twins(true, attn_layer.has_w8a8_prefill_kernels());
    let (nh, hd) = (config.num_attention_heads, config.head_dim);
    let (nkv, hh) = (config.num_key_value_heads, config.hidden_size);
    let q_n = nh * hd * if config.attn_gated { 2 } else { 1 };
    let twin_bytes = |set| fp8_residency::attn_fp8_twin_bytes(set, q_n, nkv * hd, nh * hd, hh);
    residency.twins.attn_fp8 |= want.any();
    residency.skip(twin_bytes(Fp8TwinSet::ALL) - twin_bytes(want));
    residency.keep(twin_bytes(want));
    if let Err(e) =
        attn_layer.transpose_fp8_for_prefill_selected(gpu, stream, want, Some(store.derived()))
    {
        tracing::warn!(target: "metrale_model_arch::weight_loader::qwen35_dense", "Layer {i}: dense FP8 transpose failed: {e}");
    }
    Ok(())
}
