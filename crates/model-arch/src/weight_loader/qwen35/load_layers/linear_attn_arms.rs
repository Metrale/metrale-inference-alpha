// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The Qwen3.5 linear-attention (GDN) layer builders: native block-scaled FP8
//! (`build_linear_attention_fp8`), BF16 dense (`build_linear_attention_dense_bf16`) and
//! NVFP4 (`build_linear_attention_nvfp4`, in `nvfp4.rs`).
//!
//! Owner: model-arch weight loader (Qwen3.5).
//! Invariants: none beyond the types.

use anyhow::{Result, ensure};
use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_model_weights::weights::WeightStore;

use crate::tp_shard::{
    TpGdnDims, shard_gdn_ba_rows, shard_gdn_conv_rows, shard_gdn_out_proj_row_parallel,
    shard_gdn_qkvz_rows, shard_gdn_value_vector,
};
use metrale_model_layers::layer::TransformerLayer;
use metrale_model_layers::layers::{FfnComponent, Qwen3SsmLayer};
use metrale_model_layers::weight_map::fp8_lut::{gpu_concat_rows, interleave_ba};
use metrale_model_layers::weight_map::quant_helpers::dense_auto;
use metrale_model_layers::weight_map::{
    DenseWeight, Fp8Weight, Nvfp4Variant, QuantizedWeight, SsmWeights, WeightQuantFormat,
    dense_f32_safe, dense_keep_f32, load_fp8_block_scaled_as_fp8weight, load_ssm_qwen35,
    quantize_to_nvfp4,
};

mod nvfp4;
pub(crate) use nvfp4::build_linear_attention_nvfp4;

/// 2026-09-25: Loads `{p}.in_proj_qkv`, `{p}.in_proj_z` and `{p}.out_proj` with
/// `load_fp8_block_scaled_as_fp8weight` and returns `(qkvz, out_proj)`. `qkvz` is QKV and Z
/// concatenated along N into one new `[Nq+Nz, h]` FP8 buffer, with their FP32 block-scale
/// rows concatenated to match. Errors unless Nq, Nz and h are multiples of 128; panics
/// unless each weight is tagged `Fp8BlockScaled`. Used by the native-FP8 build and by the
/// `METRALE_HOLO_FP8_SSM_DECODE` overlay on the BF16 dense build.
fn load_ssm_fp8_decode_weights(
    layer_idx: usize,
    store: &WeightStore,
    p: &str,
    gpu: &dyn GpuBackend,
    h: usize,
) -> Result<(Fp8Weight, Fp8Weight)> {
    let qkv_fp8 = load_fp8_block_scaled_as_fp8weight(store, &format!("{p}.in_proj_qkv"), gpu)?;
    let z_fp8 = load_fp8_block_scaled_as_fp8weight(store, &format!("{p}.in_proj_z"), gpu)?;
    let out_fp8 = load_fp8_block_scaled_as_fp8weight(store, &format!("{p}.out_proj"), gpu)?;

    qkv_fp8.scale_format.expect(
        WeightQuantFormat::Fp8BlockScaled,
        "load_ssm_fp8_decode_weights::qkv_fp8 from disk",
    );
    z_fp8.scale_format.expect(
        WeightQuantFormat::Fp8BlockScaled,
        "load_ssm_fp8_decode_weights::z_fp8 from disk",
    );
    out_fp8.scale_format.expect(
        WeightQuantFormat::Fp8BlockScaled,
        "load_ssm_fp8_decode_weights::out_fp8 from disk",
    );

    let qkv_rows = qkv_fp8.n as usize;
    let z_rows = z_fp8.n as usize;
    let qkvz_n = qkv_rows + z_rows;

    let qkvz_weight_ptr = gpu.alloc(qkvz_n * h)?;
    gpu.copy_d2d(qkv_fp8.weight, qkvz_weight_ptr, qkv_rows * h)?;
    gpu.copy_d2d(
        z_fp8.weight,
        qkvz_weight_ptr.offset(qkv_rows * h),
        z_rows * h,
    )?;

    const BS: usize = 128;
    ensure!(
        qkv_rows.is_multiple_of(BS),
        "SSM L{layer_idx}: qkv_rows={qkv_rows} not divisible by BS={BS} (FP8 block size)",
    );
    ensure!(
        z_rows.is_multiple_of(BS),
        "SSM L{layer_idx}: z_rows={z_rows} not divisible by BS={BS} (FP8 block size)",
    );
    ensure!(
        h.is_multiple_of(BS),
        "SSM L{layer_idx}: hidden_size={h} not divisible by BS={BS}",
    );
    let scale_cols = h / BS;
    let scale_row_bytes = scale_cols * 4;
    let qkv_scale_rows = qkv_rows / BS;
    let z_scale_rows = z_rows / BS;
    let qkvz_scale_bytes = (qkv_scale_rows + z_scale_rows) * scale_row_bytes;
    let qkvz_scale_ptr = gpu.alloc(qkvz_scale_bytes)?;
    gpu.copy_d2d(
        qkv_fp8.row_scale,
        qkvz_scale_ptr,
        qkv_scale_rows * scale_row_bytes,
    )?;
    gpu.copy_d2d(
        z_fp8.row_scale,
        qkvz_scale_ptr.offset(qkv_scale_rows * scale_row_bytes),
        z_scale_rows * scale_row_bytes,
    )?;

    let qkvz_fp8 = Fp8Weight {
        weight: qkvz_weight_ptr,
        row_scale: qkvz_scale_ptr,
        n: qkvz_n as u32,
        k: h as u32,
        scale_format: WeightQuantFormat::Fp8BlockScaled,
    };
    Ok((qkvz_fp8, out_fp8))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_linear_attention_fp8(
    layer_idx: usize,
    store: &WeightStore,
    lp: &str,
    gpu: &dyn GpuBackend,
    _variant: Nvfp4Variant,
    config: &ModelConfig,
    h: usize,
    _stream: u64,
    input_norm: DenseWeight,
    post_attn_norm: DenseWeight,
    ffn: FfnComponent,
) -> Result<Box<dyn TransformerLayer>> {
    // 2026-09-25: TP=1 only: the `shard_gdn_*` slicers cut dense rows and have no form for
    // the block-scale grid.
    ensure!(
        config.tp_world_size.max(1) == 1,
        "Native block-scaled FP8 SSM (linear_attn) supports TP=1 only (got tp={}); \
         GDN HeadParallel FP8 scale slicing is deferred. Use the NVFP4 decode path \
         (METRALE_HOLO_FP4_PROJ_DECODE=1) or run --tp-size 1 for FP8.",
        config.tp_world_size,
    );

    let p = format!("{lp}.linear_attn");
    tracing::info!("Layer {layer_idx}: loading SSM FP8 native (block-scaled decode + prefill)");

    let (qkvz_fp8, out_fp8) = load_ssm_fp8_decode_weights(layer_idx, store, &p, gpu, h)?;
    tracing::info!(
        "Layer {layer_idx}: SSM QKVZ FP8 [{},{h}] block-scaled, out_proj FP8 [{},{}] block-scaled",
        qkvz_fp8.n,
        out_fp8.n,
        out_fp8.k
    );

    let nv = config.linear_num_value_heads;
    let nk = config.linear_num_key_heads;
    let in_proj_a = dense_auto(store, &format!("{p}.in_proj_a.weight"), gpu)?;
    let in_proj_b = dense_auto(store, &format!("{p}.in_proj_b.weight"), gpu)?;
    let ba_dense = interleave_ba(
        &DenseWeight {
            weight: in_proj_a.weight,
        },
        &DenseWeight {
            weight: in_proj_b.weight,
        },
        nv,
        nk,
        h,
        gpu,
    )?;

    // 2026-09-25: The dense `in_proj_qkvz` and quantized `out_proj` slots stay NULL: the
    // projections are the FP8 weights installed by `set_fp8_decode_weights`.
    let ssm = SsmWeights {
        in_proj_qkvz: DenseWeight {
            weight: metrale_gpu_runtime::gpu::DevicePtr::NULL,
        },
        in_proj_ba: ba_dense,
        conv1d: dense_auto(store, &format!("{p}.conv1d.weight"), gpu)?,
        a_log: dense_keep_f32(store, &format!("{p}.A_log"), gpu)?,
        dt_bias: dense_keep_f32(store, &format!("{p}.dt_bias"), gpu)?,
        norm: dense_f32_safe(store, &format!("{p}.norm.weight"), gpu)?,
        out_proj: QuantizedWeight::null(),
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
    layer.set_fp8_decode_weights(Some(qkvz_fp8), Some(out_fp8));
    tracing::info!("Layer {layer_idx}: SSM native FP8 — w8a16 decode + prefill");
    Ok(Box::new(layer))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_linear_attention_dense_bf16(
    layer_idx: usize,
    store: &WeightStore,
    lp: &str,
    gpu: &dyn GpuBackend,
    variant: Nvfp4Variant,
    config: &ModelConfig,
    h: usize,
    input_norm: DenseWeight,
    post_attn_norm: DenseWeight,
    ffn: FfnComponent,
) -> Result<Box<dyn TransformerLayer>> {
    // 2026-09-25: `config` holds per-rank linear head counts; `TpGdnDims` rebuilds the full
    // on-disk sizes for the slicers. At TP=1 the `shard_gdn_*` slicers return their source
    // pointer.
    let tp_size = config.tp_world_size.max(1);
    let dims = TpGdnDims::from_config(config);
    tracing::info!(
        "Layer {layer_idx}: loading SSM FP8 projections as BF16 dense \
         (tp={tp_size}, local_nk={}, local_nv={})",
        dims.local_nk,
        dims.local_nv,
    );

    let ssm35 = load_ssm_qwen35(store, lp, gpu, variant)?;

    // 2026-09-25: Concatenate the full `[Q|K|V]` and `[Z]`, then slice each of the four
    // segments to this rank's heads (`shard_gdn_qkvz_rows`).
    let qkvz_full = gpu_concat_rows(
        &ssm35.in_proj_qkv,
        dims.full_conv_dim(),
        &ssm35.in_proj_z,
        dims.full_value_dim(),
        h,
        gpu,
    )?;
    // 2026-09-25: `gpu_concat_rows` copied both projections into a new buffer, so the
    // loaded ones are freed. They are fresh buffers when `load_ssm_qwen35` dequantized them
    // (FP8 or NVFP4 on disk); a BF16 projection is the `WeightStore`'s own pointer
    // (`dense_auto`).
    let _ = gpu.free(ssm35.in_proj_qkv.weight);
    let _ = gpu.free(ssm35.in_proj_z.weight);
    let (qkvz_ptr, _, _) = shard_gdn_qkvz_rows(qkvz_full.weight, &dims, gpu)?;
    if tp_size > 1 {
        let _ = gpu.free(qkvz_full.weight);
    }
    let qkvz_dense = DenseWeight { weight: qkvz_ptr };

    // 2026-09-25: Interleave the full BA rows, then slice; each rank's rows start on a
    // key-head group boundary (`shard_gdn_ba_rows`).
    let ba_full = interleave_ba(
        &DenseWeight {
            weight: ssm35.in_proj_a.weight,
        },
        &DenseWeight {
            weight: ssm35.in_proj_b.weight,
        },
        dims.full_nv,
        dims.full_nk,
        h,
        gpu,
    )?;
    let (ba_ptr, _, _) = shard_gdn_ba_rows(ba_full.weight, &dims, gpu)?;
    if tp_size > 1 {
        let _ = gpu.free(ba_full.weight);
    }
    let ba_dense = DenseWeight { weight: ba_ptr };

    // 2026-09-25: conv1d is sliced by QKV channel, a_log and dt_bias by value head (4-byte
    // elements), and out_proj row-parallel on the value dim.
    let d_conv = config.linear_conv_kernel_dim;
    let (conv_ptr, _, _) = shard_gdn_conv_rows(ssm35.conv1d.weight, &dims, d_conv, gpu)?;
    let (a_log_ptr, _) = shard_gdn_value_vector(ssm35.a_log.weight, &dims, 1, 4, gpu)?;
    let (dt_bias_ptr, _) = shard_gdn_value_vector(ssm35.dt_bias.weight, &dims, 1, 4, gpu)?;
    // 2026-09-25: `norm` is one `[vd]` gain shared by every value head (the GDN kernels
    // index it by position within the head, e.g. `gated_delta_rule.cu` `norm_weight[vlocal]`),
    // so every rank keeps all of it.
    let norm_ptr = ssm35.norm.weight;
    let (out_proj_ptr, _, _) = shard_gdn_out_proj_row_parallel(ssm35.out_proj.weight, &dims, gpu)?;

    let ssm = SsmWeights {
        in_proj_qkvz: qkvz_dense,
        in_proj_ba: ba_dense,
        conv1d: DenseWeight { weight: conv_ptr },
        a_log: DenseWeight { weight: a_log_ptr },
        dt_bias: DenseWeight {
            weight: dt_bias_ptr,
        },
        norm: DenseWeight { weight: norm_ptr },
        out_proj: QuantizedWeight::null(),
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
    layer.out_proj_dense = Some(DenseWeight {
        weight: out_proj_ptr,
    });
    // 2026-09-25: `METRALE_HOLO_FP8_SSM_DECODE=1` also installs the checkpoint's block-scaled
    // FP8 QKVZ and out_proj with `set_fp8_decode_weights`. Decode reads them, and so do the
    // QKVZ prefill arms that test `qkvz_fp8w`, which come before the BF16 weights
    // (`qwen3_ssm/trait_prefill_proj.rs`). They are loaded unsharded, hence TP=1 only.
    if tp_size == 1 && std::env::var("METRALE_HOLO_FP8_SSM_DECODE").ok().as_deref() == Some("1") {
        let p = format!("{lp}.linear_attn");
        let (qkvz_fp8, out_fp8) = load_ssm_fp8_decode_weights(layer_idx, store, &p, gpu, h)?;
        layer.set_fp8_decode_weights(Some(qkvz_fp8), Some(out_fp8));
        tracing::info!("Layer {layer_idx}: SSM FP8 decode overlay installed (BF16 prefill kept)");
    }
    Ok(Box::new(layer))
}
