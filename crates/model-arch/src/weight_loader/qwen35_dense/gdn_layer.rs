// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: [`load_linear_attention`], the `LayerType::LinearAttention` arm of
//! `Qwen35DenseWeightLoader::load_layers`, with its keep-packed Q2_0 and native FP8 GDN
//! builders. The remaining forms (native NVFP4, BF16 dequant, NVFP4 requant) are in
//! `gdn_dequant`.
//!
//! Owner: model-arch weight loader (Qwen3.5 dense).
//! Invariants:
//! - The native FP8 GDN arm runs exactly when `gdn_fp8_arm_selected` holds, the predicate
//!   `prune_after_load` frees by.

use anyhow::Result;
use metrale_model_layers::layer::TransformerLayer;
use metrale_model_layers::layers::Qwen3SsmLayer;
use metrale_model_layers::weight_map::fp8_lut::{dequant_nvfp4_to_bf16, interleave_ba};
use metrale_model_layers::weight_map::quant_helpers::dense_auto;
use metrale_model_layers::weight_map::{
    DenseWeight, PackedQ2Weight, SsmWeights, dense, dense_f32_safe, dense_keep_f32,
    load_fp8_block_scaled_as_fp8weight, quantize_to_nvfp4,
};
use metrale_model_weights::weights::WeightDtype;

use super::fp8_residency::DerivedResidency;
use super::gdn_dequant::{self, GdnIn};
use super::load_cx::{Flow, LayerIn, LoadCx};
use super::{concat_fp8_block_scaled, gdn_fp8_arm_selected, packed_q2_from_store, proj_q2_group};
use crate::tp_shard::TpGdnDims;

/// 2026-09-26: Builds linear-attention layer `l.i` and pushes it to `layers`. Returns
/// [`Flow::Continue`] for every arm but the final NVFP4 requant, which returns
/// [`Flow::Proceed`] (from `gdn_dequant`).
pub(super) fn load_linear_attention(
    cx: &LoadCx<'_>,
    residency: &mut DerivedResidency,
    layers: &mut Vec<Box<dyn TransformerLayer>>,
    l: LayerIn<'_>,
) -> Result<Flow> {
    let LoadCx {
        store, config, gpu, ..
    } = *cx;
    let LayerIn {
        i,
        lp,
        input_norm,
        post_attn_norm,
        ffn,
    } = l;
    let nv = config.linear_num_value_heads;
    let nk = config.linear_num_key_heads;
    // 2026-09-25: `config` holds per-rank linear head counts (server
    // `serve_phases/topology.rs`); `TpGdnDims` rebuilds the full on-disk sizes for
    // the loads and concatenation, and `value_dim` stays per rank.
    let tp_size = config.tp_world_size.max(1);
    let dims = TpGdnDims::from_config(config);
    let qkv_rows = dims.full_conv_dim();
    let z_rows = dims.full_value_dim();
    let value_dim = nv * config.linear_value_head_dim;
    let la = format!("{lp}.linear_attn");

    // 2026-09-25: Keep-packed Q2_0 GDN (TP=1, unless `METRALE_NO_Q2_GDN`):
    // `in_proj_qkv` and `in_proj_z` are concatenated into one packed `[Q|K|V|Z]`;
    // out_proj is quantized to NVFP4. `gdn_fp8_arm_selected` tests the same
    // condition.
    let gdn_q2 = config.tp_world_size.max(1) == 1
        && std::env::var_os("METRALE_NO_Q2_GDN").is_none()
        && proj_q2_group(store, &format!("{la}.in_proj_qkv")).is_some()
        && proj_q2_group(store, &format!("{la}.in_proj_z")).is_some();
    if gdn_q2 {
        let layer = build_gdn_q2(
            cx,
            LayerIn {
                i,
                lp,
                input_norm,
                post_attn_norm,
                ffn,
            },
            &la,
            nv,
            nk,
            value_dim,
        )?;
        layers.push(Box::new(layer));
        return Ok(Flow::Continue);
    }

    // 2026-09-25: Each GDN projection is loaded by its own on-disk form: a
    // `weight_packed` or a UInt8 `.weight` is NVFP4 and is dequantized to BF16;
    // anything else goes through `dense_auto`.
    let load_ssm_proj = |name: &str, rows: usize, cols: usize| -> Result<DenseWeight> {
        if store.contains(&format!("{name}.weight_packed")) {
            dequant_nvfp4_to_bf16(store, name, rows, cols, gpu)
        } else if matches!(
            store.get(&format!("{name}.weight")).map(|w| w.dtype),
            Ok(WeightDtype::UInt8)
        ) {
            dequant_nvfp4_to_bf16(store, name, rows, cols, gpu)
        } else {
            dense_auto(store, &format!("{name}.weight"), gpu)
        }
    };
    // 2026-09-25: Native FP8 GDN (`gdn_fp8_arm_selected`; off with
    // `METRALE_NO_GDN_FP8`): in_proj_qkv and in_proj_z are concatenated on the
    // device into one block-scaled `[Q|K|V|Z]` FP8 weight and, with out_proj,
    // installed by `set_fp8_decode_weights`. Decode and prefill both read them
    // (`qkvz_fp8w` arms of `qwen3_ssm/trait_prefill_proj.rs`).
    if gdn_fp8_arm_selected(store, &la, config.tp_world_size) {
        let layer = build_gdn_fp8(
            cx,
            residency,
            LayerIn {
                i,
                lp,
                input_norm,
                post_attn_norm,
                ffn,
            },
            &la,
            nv,
            nk,
        )?;
        layers.push(Box::new(layer));
        return Ok(Flow::Continue);
    }

    let g = GdnIn {
        la: &la,
        nv,
        nk,
        tp_size,
        dims,
        qkv_rows,
        z_rows,
        value_dim,
    };
    gdn_dequant::load_gdn_dequant(
        cx,
        layers,
        LayerIn {
            i,
            lp,
            input_norm,
            post_attn_norm,
            ffn,
        },
        &g,
        &load_ssm_proj,
    )
}

/// 2026-09-26: The keep-packed Q2_0 GDN layer (`gdn_q2`): one packed `[Q|K|V|Z]` and an
/// NVFP4 out_proj.
fn build_gdn_q2(
    cx: &LoadCx<'_>,
    l: LayerIn<'_>,
    la: &str,
    nv: usize,
    nk: usize,
    value_dim: usize,
) -> Result<Qwen3SsmLayer> {
    let LoadCx {
        store,
        config,
        gpu,
        absmax_k,
        quantize_k,
        stream,
        h,
        ..
    } = *cx;
    let LayerIn {
        lp,
        input_norm,
        post_attn_norm,
        ffn,
        ..
    } = l;
    let qkv_q2 = packed_q2_from_store(store, &format!("{la}.in_proj_qkv"))?;
    let z_q2 = packed_q2_from_store(store, &format!("{la}.in_proj_z"))?;
    anyhow::ensure!(
        qkv_q2.group == z_q2.group && qkv_q2.k == z_q2.k,
        "GDN packed qkv/z group|k mismatch ({},{} vs {},{})",
        qkv_q2.group,
        qkv_q2.k,
        z_q2.group,
        z_q2.k
    );
    // 2026-09-25: Byte-concatenate whole packed rows, `[Q|K|V]` then `[Z]`; a
    // row is `(k / group) * block_bytes`, so no block is split.
    let group = qkv_q2.group as usize;
    let block_bytes = 2 + group / 4;
    let row_bytes = (qkv_q2.k as usize / group) * block_bytes;
    let qkv_bytes = qkv_q2.n as usize * row_bytes;
    let z_bytes = z_q2.n as usize * row_bytes;
    let qkvz_buf = gpu.alloc(qkv_bytes + z_bytes)?;
    gpu.copy_d2d(qkv_q2.weight, qkvz_buf, qkv_bytes)?;
    gpu.copy_d2d(z_q2.weight, qkvz_buf.offset(qkv_bytes), z_bytes)?;
    let qkvz_q2 = PackedQ2Weight {
        weight: qkvz_buf,
        n: qkv_q2.n + z_q2.n,
        k: qkv_q2.k,
        group: qkv_q2.group,
    };
    let in_proj_a = dense_auto(store, &format!("{la}.in_proj_a.weight"), gpu)?;
    let in_proj_b = dense_auto(store, &format!("{la}.in_proj_b.weight"), gpu)?;
    let conv1d = dense(store, &format!("{la}.conv1d.weight"))?;
    let a_log = dense_keep_f32(store, &format!("{la}.A_log"), gpu)?;
    let dt_bias = dense_keep_f32(store, &format!("{la}.dt_bias"), gpu)?;
    let norm = dense_f32_safe(store, &format!("{la}.norm.weight"), gpu)?;
    let ba_dense = interleave_ba(&in_proj_a, &in_proj_b, nv, nk, h, gpu)?;
    let out_proj_dense = dense_auto(store, &format!("{la}.out_proj.weight"), gpu)?;
    let out_proj_nvfp4 = quantize_to_nvfp4(
        &out_proj_dense,
        h,
        value_dim,
        gpu,
        absmax_k,
        quantize_k,
        stream,
    )?;
    let out_proj_nvfp4_t = out_proj_nvfp4.transpose_for_gemm(gpu, h, value_dim)?;
    gpu.free(out_proj_dense.weight)?;
    let ssm = SsmWeights {
        in_proj_qkvz: DenseWeight {
            weight: metrale_gpu_runtime::gpu::DevicePtr::NULL,
        },
        in_proj_ba: ba_dense,
        conv1d,
        a_log,
        dt_bias,
        norm,
        out_proj: out_proj_nvfp4,
    };
    let mut layer = Qwen3SsmLayer::new_sequential(
        input_norm,
        ssm,
        post_attn_norm,
        ffn,
        None,
        None,
        Some(out_proj_nvfp4_t),
        config,
        gpu,
    )?;
    layer.set_packed_q2_qkvz(qkvz_q2, gpu);
    layer.predequant_for_prefill(gpu, config, stream)?;
    tracing::info!(target: "metrale_model_arch::weight_loader::qwen35_dense", "SSM[{lp}] native keep-packed Q2_0 GDN: qkvz 2-bit \
         (concat qkv+z row-permuted), out_proj NVFP4"
    );
    Ok(layer)
}

/// 2026-09-26: The native block-scaled FP8 GDN layer (`gdn_fp8_arm_selected`): one fused
/// `[Q|K|V|Z]` FP8 weight and the FP8 out_proj, installed by `set_fp8_decode_weights`.
fn build_gdn_fp8(
    cx: &LoadCx<'_>,
    residency: &mut DerivedResidency,
    l: LayerIn<'_>,
    la: &str,
    nv: usize,
    nk: usize,
) -> Result<Qwen3SsmLayer> {
    let LoadCx {
        store,
        config,
        gpu,
        h,
        ..
    } = *cx;
    let LayerIn {
        lp,
        input_norm,
        post_attn_norm,
        ffn,
        ..
    } = l;
    let in_proj_a = dense(store, &format!("{la}.in_proj_a.weight"))?;
    let in_proj_b = dense(store, &format!("{la}.in_proj_b.weight"))?;
    let conv1d = dense(store, &format!("{la}.conv1d.weight"))?;
    let a_log = dense_keep_f32(store, &format!("{la}.A_log"), gpu)?;
    let dt_bias = dense_keep_f32(store, &format!("{la}.dt_bias"), gpu)?;
    let norm = dense_f32_safe(store, &format!("{la}.norm.weight"), gpu)?;
    let ba_dense = interleave_ba(&in_proj_a, &in_proj_b, nv, nk, h, gpu)?;
    let qkv_f = load_fp8_block_scaled_as_fp8weight(store, &format!("{la}.in_proj_qkv"), gpu)?;
    let z_f = load_fp8_block_scaled_as_fp8weight(store, &format!("{la}.in_proj_z"), gpu)?;
    let out_f = load_fp8_block_scaled_as_fp8weight(store, &format!("{la}.out_proj"), gpu)?;
    let qkvz_f = concat_fp8_block_scaled(&qkv_f, &z_f, h, gpu)?;
    // 2026-09-25: The concat copied both scale grids, so the per-projection
    // ones are freed; the weight bytes stay the store's.
    let scale_bytes = |n: usize, k: usize| n.div_ceil(128) * k.div_ceil(128) * 4;
    gpu.free(qkv_f.row_scale)?;
    gpu.free(z_f.row_scale)?;
    residency.free(scale_bytes(qkv_f.n as usize, h) + scale_bytes(z_f.n as usize, h));
    // 2026-09-25: The store adopts every buffer this arm allocated (the fused
    // weight, both scale grids and the BA interleave), so teardown releases
    // them.
    let qkvz_bytes = (qkvz_f.n as usize) * h;
    let qkvz_scale_bytes = scale_bytes(qkv_f.n as usize, h) + scale_bytes(z_f.n as usize, h);
    let ba_bytes = nv * 2 * h * 2;
    let out_scale_bytes = scale_bytes(out_f.n as usize, out_f.k as usize);
    let d = store.derived();
    d.adopt("ssm qkvz fp8 concat", qkvz_f.weight, qkvz_bytes);
    d.adopt(
        "ssm qkvz fp8 block scale",
        qkvz_f.row_scale,
        qkvz_scale_bytes,
    );
    d.adopt(
        "ssm out_proj fp8 block scale",
        out_f.row_scale,
        out_scale_bytes,
    );
    d.adopt("ssm in_proj_ba interleaved", ba_dense.weight, ba_bytes);
    residency.keep(qkvz_bytes + qkvz_scale_bytes + out_scale_bytes + ba_bytes);
    residency.twins.ssm_fp8_concat = true;
    let ssm = SsmWeights {
        in_proj_qkvz: DenseWeight {
            weight: metrale_gpu_runtime::gpu::DevicePtr::NULL,
        },
        in_proj_ba: ba_dense,
        conv1d,
        a_log,
        dt_bias,
        norm,
        out_proj: metrale_model_layers::weight_map::QuantizedWeight::null(),
    };
    let mut layer = Qwen3SsmLayer::new_sequential(
        input_norm,
        ssm,
        post_attn_norm,
        ffn,
        None,
        None,
        None,
        config,
        gpu,
    )?;
    layer.set_fp8_decode_weights(Some(qkvz_f), Some(out_f));
    tracing::info!(target: "metrale_model_arch::weight_loader::qwen35_dense", "SSM[{lp}] native FP8 GDN: qkvz+out_proj block-scaled FP8 \
         (no NVFP4 requant; prefill+decode via w8a16)"
    );
    Ok(layer)
}
