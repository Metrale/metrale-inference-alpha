// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: [`load_gdn_dequant`], the GDN forms `gdn_layer::load_linear_attention`
//! reaches after its keep-packed Q2_0 and native FP8 tests: pre-quantized NVFP4 from disk,
//! the BF16 `[Q|K|V|Z]` kept as it is (`Bf16Raw`, `METRALE_GDN_BF16_WEIGHTS=1`), or that
//! BF16 requantized to NVFP4 with optional FP8 prefill casts and per-row FP8 prefill
//! weights.
//!
//! Owner: model-arch weight loader (Qwen3.5 dense).
//! Invariants:
//! - Under TP the dequant path slices every buffer to this rank's heads before any
//!   consumer reads it; `norm` is not sliced.

use anyhow::Result;
use metrale_model_layers::layer::TransformerLayer;
use metrale_model_layers::layers::Qwen3SsmLayer;
use metrale_model_layers::weight_map::fp8_lut::{gpu_concat_rows, interleave_ba};
use metrale_model_layers::weight_map::{
    DenseWeight, Nvfp4Variant, SsmWeights, dense, dense_f32_safe, dense_keep_f32,
    quantize_to_nvfp4, quantized_auto,
};
use metrale_model_weights::weights::WeightDtype;

use super::load_cx::{Flow, LayerIn, LoadCx};
use super::rowwise_fp8;
use crate::tp_shard::{
    TpGdnDims, shard_gdn_ba_rows, shard_gdn_conv_rows, shard_gdn_out_proj_row_parallel,
    shard_gdn_qkvz_rows, shard_gdn_value_vector,
};

/// 2026-09-26: The values the linear-attention arm computes before its first weight form
/// test: the `{lp}.linear_attn` prefix, the per-rank head counts, the TP size and
/// dimensions, the full on-disk `[Q|K|V]` and `[Z]` row counts, and the per-rank
/// `value_dim`.
pub(super) struct GdnIn<'a> {
    pub(super) la: &'a str,
    pub(super) nv: usize,
    pub(super) nk: usize,
    pub(super) tp_size: usize,
    pub(super) dims: TpGdnDims,
    pub(super) qkv_rows: usize,
    pub(super) z_rows: usize,
    pub(super) value_dim: usize,
}

/// 2026-09-26: Builds GDN layer `l.i` in one of the forms named in the module header and
/// pushes it to `layers`. Returns [`Flow::Proceed`] for the NVFP4 requant form and
/// [`Flow::Continue`] for the others.
pub(super) fn load_gdn_dequant(
    cx: &LoadCx<'_>,
    layers: &mut Vec<Box<dyn TransformerLayer>>,
    l: LayerIn<'_>,
    g: &GdnIn<'_>,
    load_ssm_proj: &impl Fn(&str, usize, usize) -> Result<DenseWeight>,
) -> Result<Flow> {
    let LoadCx {
        store,
        config,
        gpu,
        variant,
        absmax_k,
        quantize_k,
        stream,
        h,
        bf16_to_fp8_k,
        ..
    } = *cx;
    let LayerIn {
        i,
        lp,
        input_norm,
        post_attn_norm,
        ffn,
    } = l;
    let GdnIn {
        la,
        nv,
        nk,
        tp_size,
        dims,
        qkv_rows,
        z_rows,
        value_dim,
    } = *g;
    // 2026-09-25: A_log and dt_bias load as FP32 (`dense_keep_f32`): the GDN
    // kernels read them as `const float*` (`ssm_preprocess.cu`,
    // `mamba2_ssm_decode.cu`). in_proj_a/b go through `load_ssm_proj`, so a UInt8
    // NVFP4 A/B is dequantized.
    let in_proj_a = load_ssm_proj(&format!("{la}.in_proj_a"), nv, h)?;
    let in_proj_b = load_ssm_proj(&format!("{la}.in_proj_b"), nv, h)?;
    let conv1d = dense(store, &format!("{la}.conv1d.weight"))?;
    let a_log = dense_keep_f32(store, &format!("{la}.A_log"), gpu)?;
    let dt_bias = dense_keep_f32(store, &format!("{la}.dt_bias"), gpu)?;
    // 2026-09-25: `dense_f32_safe`: an FP32 norm is truncated to BF16, a BF16 one
    // passed through.
    let norm = dense_f32_safe(store, &format!("{la}.norm.weight"), gpu)?;
    let ba_dense = interleave_ba(&in_proj_a, &in_proj_b, nv, nk, h, gpu)?;
    let qkvz_size = config.ssm_qkvz_size();

    // 2026-09-25: When `in_proj_qkv`, `in_proj_z` and `out_proj` all have a UInt8
    // `.weight` (pre-quantized NVFP4), they load as NVFP4 directly and QKV/Z are
    // concatenated on the device. A layer with only some of them UInt8 takes the
    // path below, where `load_ssm_proj` dequantizes each UInt8 tensor.
    let native_nvfp4 = matches!(
        store
            .get(&format!("{la}.in_proj_qkv.weight"))
            .map(|w| w.dtype),
        Ok(WeightDtype::UInt8)
    ) && matches!(
        store
            .get(&format!("{la}.in_proj_z.weight"))
            .map(|w| w.dtype),
        Ok(WeightDtype::UInt8)
    ) && matches!(
        store.get(&format!("{la}.out_proj.weight")).map(|w| w.dtype),
        Ok(WeightDtype::UInt8)
    );
    if native_nvfp4 {
        let qkv_qw = quantized_auto(
            store,
            &format!("{la}.in_proj_qkv"),
            gpu,
            Nvfp4Variant::Standard,
        )?;
        let z_qw = quantized_auto(
            store,
            &format!("{la}.in_proj_z"),
            gpu,
            Nvfp4Variant::Standard,
        )?;
        let qkvz_nvfp4 = qkv_qw.concat_rows(&z_qw, qkv_rows, z_rows, h, gpu)?;
        let qkvz_nvfp4_t = qkvz_nvfp4.transpose_for_gemm(gpu, qkvz_size, h)?;

        let out_proj_nvfp4 = quantized_auto(
            store,
            &format!("{la}.out_proj"),
            gpu,
            Nvfp4Variant::Standard,
        )?;
        let out_proj_nvfp4_t = out_proj_nvfp4.transpose_for_gemm(gpu, h, value_dim)?;

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
            Some(qkvz_nvfp4),
            Some(qkvz_nvfp4_t),
            Some(out_proj_nvfp4_t),
            config,
            gpu,
        )?;
        layer.predequant_for_prefill(gpu, config, stream)?;
        tracing::info!(target: "metrale_model_arch::weight_loader::qwen35_dense", "SSM[{lp}] native NVFP4 GDN: qkvz+out_proj loaded pre-quantized \
             (U8-packed on disk; no BF16 dequant/requant roundtrip)"
        );
        layers.push(Box::new(layer));
        return Ok(Flow::Continue);
    }

    // 2026-09-25: `METRALE_FP8_ROWWISE=1` with per-row FP8 `in_proj_qkv`,
    // `in_proj_z` and `out_proj` (TP=1): those are also loaded for the per-row
    // prefill arm; the NVFP4 build below still runs for decode.
    let rowwise_gdn = rowwise_fp8::rowwise_fp8_enabled()
        && rowwise_fp8::proj_is_fp8_per_row(store, &format!("{la}.in_proj_qkv"))
        && rowwise_fp8::proj_is_fp8_per_row(store, &format!("{la}.in_proj_z"))
        && rowwise_fp8::proj_is_fp8_per_row(store, &format!("{la}.out_proj"));
    let (qkvz_rowwise, out_proj_rowwise) = if rowwise_gdn && tp_size == 1 {
        let qkv_r = rowwise_fp8::load_fp8_per_row(store, &format!("{la}.in_proj_qkv"), gpu)?;
        let z_r = rowwise_fp8::load_fp8_per_row(store, &format!("{la}.in_proj_z"), gpu)?;
        let out_r = rowwise_fp8::load_fp8_per_row(store, &format!("{la}.out_proj"), gpu)?;
        let qkvz_r = rowwise_fp8::concat_fp8_per_row(&qkv_r, &z_r, h, gpu)?;
        // 2026-09-25: The concat copied both scale vectors, so the per-projection
        // ones are freed; the weight bytes stay the store's.
        gpu.free(qkv_r.row_scale)?;
        gpu.free(z_r.row_scale)?;
        (Some(qkvz_r), Some(out_r))
    } else {
        (None, None)
    };

    let qkv_dense = load_ssm_proj(&format!("{la}.in_proj_qkv"), qkv_rows, h)?;
    let z_dense = load_ssm_proj(&format!("{la}.in_proj_z"), z_rows, h)?;
    let out_proj_dense = load_ssm_proj(&format!("{la}.out_proj"), h, dims.full_value_dim())?;

    let qkvz_dense = gpu_concat_rows(&qkv_dense, qkv_rows, &z_dense, z_rows, h, gpu)?;
    // 2026-09-25: `gpu_concat_rows` copied both, so the loaded QKV and Z are freed.
    // They are fresh buffers when `load_ssm_proj` dequantized them; a BF16
    // projection is the `WeightStore`'s own pointer (`dense_auto`).
    gpu.free(qkv_dense.weight)?;
    gpu.free(z_dense.weight)?;

    let ba_dense = interleave_ba(&in_proj_a, &in_proj_b, dims.full_nv, dims.full_nk, h, gpu)?;

    // 2026-09-25: Under TP, cut the full buffers to this rank's heads, before both
    // consumers below (as in `qwen35/load_layers/linear_attn_arms.rs`). `norm` is
    // one `[vd]` gain shared by every value head and is not sliced. The fresh
    // `[Q|K|V|Z]` and BA buffers are freed once sliced.
    let (qkvz_dense, ba_dense, conv1d, a_log, dt_bias, out_proj_dense) = if tp_size > 1 {
        let d_conv = config.linear_conv_kernel_dim;
        let (qkvz_ptr, _, _) = shard_gdn_qkvz_rows(qkvz_dense.weight, &dims, gpu)?;
        gpu.free(qkvz_dense.weight)?;
        let (ba_ptr, _, _) = shard_gdn_ba_rows(ba_dense.weight, &dims, gpu)?;
        gpu.free(ba_dense.weight)?;
        let (conv_ptr, _, _) = shard_gdn_conv_rows(conv1d.weight, &dims, d_conv, gpu)?;
        let (a_log_ptr, _) = shard_gdn_value_vector(a_log.weight, &dims, 1, 4, gpu)?;
        let (dt_bias_ptr, _) = shard_gdn_value_vector(dt_bias.weight, &dims, 1, 4, gpu)?;
        let (out_ptr, _, _) = shard_gdn_out_proj_row_parallel(out_proj_dense.weight, &dims, gpu)?;
        (
            DenseWeight { weight: qkvz_ptr },
            DenseWeight { weight: ba_ptr },
            DenseWeight { weight: conv_ptr },
            DenseWeight { weight: a_log_ptr },
            DenseWeight {
                weight: dt_bias_ptr,
            },
            DenseWeight { weight: out_ptr },
        )
    } else {
        (qkvz_dense, ba_dense, conv1d, a_log, dt_bias, out_proj_dense)
    };

    // 2026-09-25: Bf16Raw: keep the BF16 `[Q|K|V|Z]` (`in_proj_qkvz`) and out_proj
    // (`out_proj_dense`) and build no NVFP4 or FP8 copy; `ssm.out_proj` stays NULL.
    if matches!(variant, Nvfp4Variant::Bf16Raw) {
        let ssm = SsmWeights {
            in_proj_qkvz: qkvz_dense,
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
        layer.out_proj_dense = Some(out_proj_dense);
        layers.push(Box::new(layer));
        return Ok(Flow::Continue);
    }

    let qkvz_size = config.ssm_qkvz_size();

    // 2026-09-25: `METRALE_GDN_BF16_WEIGHTS=1` keeps the BF16 `[Q|K|V|Z]` and
    // out_proj instead of quantizing them to NVFP4.
    let gdn_bf16 = matches!(
        std::env::var("METRALE_GDN_BF16_WEIGHTS").ok().as_deref(),
        Some("1")
    );
    if gdn_bf16 {
        let ssm = SsmWeights {
            in_proj_qkvz: DenseWeight {
                weight: qkvz_dense.weight,
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
        layer.out_proj_dense = Some(out_proj_dense);
        tracing::info!(target: "metrale_model_arch::weight_loader::qwen35_dense", "SSM[{lp}] METRALE_GDN_BF16_WEIGHTS: qkvz + out_proj kept BF16 \
             (≥FP8; NVFP4 requant skipped)"
        );
        layers.push(Box::new(layer));
        return Ok(Flow::Continue);
    }

    let qkvz_nvfp4 =
        quantize_to_nvfp4(&qkvz_dense, qkvz_size, h, gpu, absmax_k, quantize_k, stream)?;

    let qkvz_nvfp4_t = qkvz_nvfp4.transpose_for_gemm(gpu, qkvz_size, h)?;

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

    // 2026-09-25: FP8 casts of the BF16 `[Q|K|V|Z]` and out_proj with no scale
    // (`bf16_to_fp8`): `fp8_gemm_n128` takes no scale argument and reads the bytes
    // as the weight values (`qwen3_ssm/init_fp8.rs`).
    let (qkvz_fp8_prefill, out_proj_fp8_prefill) = if let Some(b2f_k) = bf16_to_fp8_k {
        let qkvz_total = (qkvz_size * h) as u32;
        let qkvz_fp8 = gpu.alloc(qkvz_size * h)?;
        metrale_model_layers::layers::ops::bf16_to_fp8(
            gpu,
            b2f_k,
            qkvz_dense.weight,
            qkvz_fp8,
            qkvz_total,
            stream,
        )?;
        let out_total = (h * value_dim) as u32;
        let out_fp8 = gpu.alloc(h * value_dim)?;
        metrale_model_layers::layers::ops::bf16_to_fp8(
            gpu,
            b2f_k,
            out_proj_dense.weight,
            out_fp8,
            out_total,
            stream,
        )?;
        gpu.synchronize(stream)?;
        (Some(qkvz_fp8), Some(out_fp8))
    } else {
        (None, None)
    };

    // 2026-09-25: Nothing in this layer reads the BF16 `[Q|K|V|Z]` or out_proj
    // (`in_proj_qkvz` is NULL and `out_proj_dense` unset), so they are freed.
    gpu.free(qkvz_dense.weight)?;
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
        Some(qkvz_nvfp4),
        Some(qkvz_nvfp4_t),
        Some(out_proj_nvfp4_t),
        config,
        gpu,
    )?;
    layer.predequant_for_prefill(gpu, config, stream)?;
    // 2026-09-25: After `predequant_for_prefill`, which sets `out_proj_fp8` from
    // the NVFP4 weight, so the FP8 casts replace it. `qkvz_fp8`/`out_proj_fp8` feed
    // the `fp8_gemm_n128` arms of prefill and of the batched decode/verify
    // projections (`qwen3_ssm/init_fp8.rs`, `trait_decode_batched.rs`).
    if qkvz_fp8_prefill.is_some() || out_proj_fp8_prefill.is_some() {
        layer.set_fp8_prefill_only_weights(qkvz_fp8_prefill, out_proj_fp8_prefill);
    }
    // 2026-09-25: `METRALE_FP8_ROWWISE`: the checkpoint's per-row FP8, tried first
    // by the prefill projections (`qwen3_ssm/trait_prefill_proj.rs`,
    // `trait_prefill_helper.rs`); no decode path reads it.
    if qkvz_rowwise.is_some() {
        layer.set_fp8_rowwise_prefill_weights(qkvz_rowwise, out_proj_rowwise);
        if i == 0 {
            tracing::info!(target: "metrale_model_arch::weight_loader::qwen35_dense", "SSM[{lp}] METRALE_FP8_ROWWISE: qkvz + out_proj prefill via \
                 native per-row FP8 (no BF16 dequant, no NVFP4 requant); \
                 decode keeps NVFP4"
            );
        }
    }
    layers.push(Box::new(layer));
    Ok(Flow::Proceed)
}
