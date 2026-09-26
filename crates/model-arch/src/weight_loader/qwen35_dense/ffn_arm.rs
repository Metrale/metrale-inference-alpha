// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: [`build_dense_ffn`], the dense FFN of one `Qwen35DenseWeightLoader` layer:
//! NVFP4, native block-scaled FP8 (with gate and up optionally fused), a BF16 snapshot, or
//! keep-packed Q2_0, as selected per layer.
//!
//! Owner: model-arch weight loader (Qwen3.5 dense).
//! Invariants:
//! - `load_layers` calls it once per layer, after the layer's two norms and before its
//!   attention or GDN arm.

use anyhow::Result;
use metrale_model_layers::layers::{DenseFfnLayer, FfnComponent};
use metrale_model_layers::weight_map::fp8_lut::load_dense_ffn;
use metrale_model_layers::weight_map::quant_helpers::dense_auto;
use metrale_model_layers::weight_map::{
    DenseWeight, Fp8Weight, Nvfp4Variant, load_fp8_block_scaled_as_fp8weight,
};

use super::fp8_residency::{self, DerivedResidency};
use super::load_cx::LoadCx;
use super::{
    concat_fp8_block_scaled, ffn_fp8_arm_selected, ffn_gateup_fused_selected, ffn_inter,
    packed_q2_from_store, proj_q2_group,
};

/// 2026-09-26: Builds layer `lp`'s dense FFN and records its derived bytes in `residency`.
pub(super) fn build_dense_ffn(
    cx: &LoadCx<'_>,
    residency: &mut DerivedResidency,
    lp: &str,
) -> Result<FfnComponent> {
    let LoadCx {
        store,
        config,
        gpu,
        variant,
        absmax_k,
        quantize_k,
        stream,
        h,
        route_env,
        ..
    } = *cx;
    // 2026-09-25: Dense FFN. `ffn_fp8` (`ffn_fp8_arm_selected`) loads gate/up/down as
    // block-scaled FP8. For Bf16Raw, BF16 copies of gate/up/down are taken first:
    // `load_dense_ffn` quantizes through `quantized_any`, whose Bf16Raw arm frees the
    // store's BF16 buffer, and `dense_auto` returns that same pointer. `ffn_q2`:
    // keep-packed Q2_0 gate/up/down (TP=1) are installed as they are
    // (`set_q2_weights`), with no NVFP4 copy.
    let ffn_fp8 = ffn_fp8_arm_selected(store, config, variant, lp);
    let ffn_q2 = config.tp_world_size.max(1) == 1
        && proj_q2_group(store, &format!("{lp}.mlp.gate_proj")).is_some()
        && proj_q2_group(store, &format!("{lp}.mlp.up_proj")).is_some()
        && proj_q2_group(store, &format!("{lp}.mlp.down_proj")).is_some();
    // 2026-09-25: Keep-packed Q2 projections are not BF16, and `dense_auto` has no arm
    // for them, so they take no snapshot.
    let ffn_bf16_snapshot = if !ffn_q2 && matches!(variant, Nvfp4Variant::Bf16Raw) {
        let inter = if config.intermediate_size > 0 {
            config.intermediate_size
        } else {
            config.moe_intermediate_size
        };
        let clone_bf16 = |name: &str, rows: usize, cols: usize| -> Result<DenseWeight> {
            let src = dense_auto(store, &format!("{lp}.mlp.{name}.weight"), gpu)?;
            let bytes = rows * cols * 2;
            let dst = gpu.alloc(bytes)?;
            // 2026-09-25: The copy must be a new allocation: `load_dense_ffn` frees the
            // store's buffer below.
            debug_assert_ne!(
                dst, src.weight,
                "ffn_bf16_snapshot must be a fresh allocation, not an alias of the store ptr"
            );
            gpu.copy_d2d(src.weight, dst, bytes)?;
            Ok(DenseWeight { weight: dst })
        };
        Some((
            clone_bf16("gate_proj", inter, h)?,
            clone_bf16("up_proj", inter, h)?,
            clone_bf16("down_proj", h, inter)?,
        ))
    } else {
        None
    };

    // 2026-09-25: `RouteEnv::plan` decides whether the NVFP4 FFN weights are built
    // (`fp8_residency.rs`).
    let plan = route_env.plan(ffn_fp8, false, false);
    let ffn_nvfp4 = !ffn_q2 && plan.ffn_nvfp4;
    let ffn_weights = if ffn_nvfp4 {
        load_dense_ffn(
            store, lp, gpu, variant, absmax_k, quantize_k, stream, config,
        )?
    } else {
        // 2026-09-25: No NVFP4 FFN weights: the layer runs from the Q2 weights (`ffn_q2`)
        // or the FP8 weights installed below (`plan.ffn_nvfp4` is false only when
        // `ffn_fp8`, `fp8_residency.rs`).
        if ffn_fp8 {
            residency.skip(fp8_residency::dense_ffn_nvfp4_bytes(h, ffn_inter(config)));
        }
        use metrale_model_layers::weight_map::QuantizedWeight;
        metrale_model_layers::layers::dense_ffn::DenseFfnWeights {
            gate_proj: QuantizedWeight::null(),
            up_proj: QuantizedWeight::null(),
            down_proj: QuantizedWeight::null(),
            gate_proj_t: None,
            up_proj_t: None,
            down_proj_t: None,
        }
    };
    residency.twins.ffn_nvfp4 |= ffn_nvfp4;
    let mut dffn = DenseFfnLayer::new(ffn_weights, gpu)?;
    if ffn_q2 {
        dffn.set_q2_weights(
            packed_q2_from_store(store, &format!("{lp}.mlp.gate_proj"))?,
            packed_q2_from_store(store, &format!("{lp}.mlp.up_proj"))?,
            packed_q2_from_store(store, &format!("{lp}.mlp.down_proj"))?,
            gpu,
        );
    }
    if ffn_fp8 {
        // 2026-09-25: The FP8 bytes stay the store's; only the widened FP32 scale grid is a
        // new allocation (`load_fp8_block_scaled_as_fp8weight`), so the store adopts it.
        let load_ffn_fp8 = |name: &str| -> Result<Fp8Weight> {
            let w = load_fp8_block_scaled_as_fp8weight(store, &format!("{lp}.mlp.{name}"), gpu)?;
            let bytes = (w.n as usize).div_ceil(128) * (w.k as usize).div_ceil(128) * 4;
            store
                .derived()
                .adopt("ffn fp8 block scale (widened)", w.row_scale, bytes);
            Ok(w)
        };
        let mut gate = load_ffn_fp8("gate_proj")?;
        let mut up = load_ffn_fp8("up_proj")?;
        let down = load_ffn_fp8("down_proj")?;
        // 2026-09-25: Fused gate+up (`ffn_gateup_fused_selected`): one `[2*inter, hidden]`
        // E4M3 buffer and one FP32 scale grid, with `gate` and `up` re-pointed at views
        // inside them, so the fused and unfused GEMMs read the same bytes.
        // `prune_after_load` then frees the two source store tensors.
        let inter = ffn_inter(config);
        let fused = if ffn_gateup_fused_selected(store, config, variant, lp) {
            let fused = concat_fp8_block_scaled(&gate, &up, h, gpu)?;
            let (w_bytes, s_bytes) = fp8_residency::ffn_gateup_fused_parts(h, inter);
            // 2026-09-25: The concat copied both widened grids; disown the per-projection
            // ones before freeing them, or teardown frees them again.
            let d = store.derived();
            let grid = inter.div_ceil(128) * h.div_ceil(128) * 4;
            for ptr in [gate.row_scale, up.row_scale] {
                d.disown(ptr);
                gpu.free(ptr)?;
            }
            residency.free(2 * grid);
            // 2026-09-25: `gate` is the head of the fused buffer and `up` starts `inter *
            // h` bytes in; the scale grids meet at the same boundary because the selector
            // requires `inter % 128 == 0`.
            gate.weight = fused.weight;
            gate.row_scale = fused.row_scale;
            up.weight = fused.weight.offset(inter * h);
            up.row_scale = fused.row_scale.offset(grid);
            d.adopt("ffn gate+up fp8 concat", fused.weight, w_bytes);
            d.adopt("ffn gate+up fp8 block scale", fused.row_scale, s_bytes);
            residency.keep(w_bytes + s_bytes);
            residency.twins.ffn_gateup_fused = true;
            Some(fused)
        } else {
            None
        };
        dffn.set_fp8_weights(gate, up, down);
        if let Some(fused) = fused {
            dffn.set_fp8_gate_up_fused(fused);
        }
    }
    // 2026-09-25: With `METRALE_FFN_MMQ` set, `finalize_q4k_load` builds the Q4_K prefill
    // copies and frees the transposed `_t` copies now, before the KV cache is sized.
    // `finalize_nvfp4_mmq_load` does the same for the W4A4 FP4-MMQ gate/up arm, which is
    // on unless `METRALE_NO_FFN_NVFP4_MMQ` is set. It is skipped when a LoRA adapter will
    // be installed (`adapter_max_rank > 0`, set in model-engine `factory/build.rs`): that
    // arm is off while an adapter is installed (`dense_ffn.rs`), and prefill would then
    // need the `_t` copies it frees.
    dffn.finalize_q4k_load(gpu, h as u32, config.intermediate_size as u32, stream)?;
    if config.adapter_max_rank == 0 {
        dffn.finalize_nvfp4_mmq_load(gpu, h as u32, config.intermediate_size as u32, stream)?;
    }
    if let Some((g, u, d)) = ffn_bf16_snapshot {
        dffn.set_bf16_weights(g, u, d);
    }
    Ok(FfnComponent::Dense(dffn))
}
