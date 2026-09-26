// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `build_linear_attention_nvfp4`: a Qwen3.5 linear-attention (GDN) layer whose
//! QKVZ and out_proj are quantized to NVFP4 at load.
//!
//! Owner: model-arch weight loader (Qwen3.5).
//! Invariants: none beyond the types.

use super::*;

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_linear_attention_nvfp4(
    store: &WeightStore,
    lp: &str,
    gpu: &dyn GpuBackend,
    variant: Nvfp4Variant,
    config: &ModelConfig,
    h: usize,
    absmax_k: metrale_gpu_runtime::gpu::KernelHandle,
    quantize_k: metrale_gpu_runtime::gpu::KernelHandle,
    stream: u64,
    input_norm: DenseWeight,
    post_attn_norm: DenseWeight,
    ffn: FfnComponent,
) -> Result<Box<dyn TransformerLayer>> {
    // 2026-09-25: Each projection is sliced to this rank's heads while BF16, then quantized
    // to NVFP4. At TP=1 the `shard_gdn_*` slicers return their source pointer.
    let tp_size = config.tp_world_size.max(1);
    let dims = TpGdnDims::from_config(config);

    let ssm35 = load_ssm_qwen35(store, lp, gpu, variant)?;

    // 2026-09-25: Concatenate the full `[Q|K|V]` and `[Z]`, then slice each segment to
    // this rank's heads (`shard_gdn_qkvz_rows`).
    let qkvz_full = gpu_concat_rows(
        &ssm35.in_proj_qkv,
        dims.full_conv_dim(),
        &ssm35.in_proj_z,
        dims.full_value_dim(),
        h,
        gpu,
    )?;
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

    let d_conv = config.linear_conv_kernel_dim;
    let (conv_ptr, _, _) = shard_gdn_conv_rows(ssm35.conv1d.weight, &dims, d_conv, gpu)?;
    let (a_log_ptr, _) = shard_gdn_value_vector(ssm35.a_log.weight, &dims, 1, 4, gpu)?;
    let (dt_bias_ptr, _) = shard_gdn_value_vector(ssm35.dt_bias.weight, &dims, 1, 4, gpu)?;
    // 2026-09-25: `norm` is one `[vd]` gain shared by every value head (the GDN kernels
    // index it by position within the head, e.g. `gated_delta_rule.cu` `norm_weight[vlocal]`),
    // so every rank keeps all of it.
    let norm_ptr = ssm35.norm.weight;
    let conv1d_local = DenseWeight { weight: conv_ptr };
    let a_log_local = DenseWeight { weight: a_log_ptr };
    let dt_bias_local = DenseWeight {
        weight: dt_bias_ptr,
    };
    let norm_local = DenseWeight { weight: norm_ptr };

    // 2026-09-25: out_proj is row-parallel: slice its input (value) dim to this rank,
    // then quantize the local `[h, local_value_dim]` weight.
    let (out_proj_ptr, _, _) = shard_gdn_out_proj_row_parallel(ssm35.out_proj.weight, &dims, gpu)?;
    let out_proj_local = DenseWeight {
        weight: out_proj_ptr,
    };

    // 2026-09-25: `config` head counts are per rank (topology divides them by the TP size).
    let nv = config.linear_num_value_heads;
    let qkvz_size = config.ssm_qkvz_size();
    let qkvz_nvfp4 =
        quantize_to_nvfp4(&qkvz_dense, qkvz_size, h, gpu, absmax_k, quantize_k, stream)?;

    let qkvz_nvfp4_t = qkvz_nvfp4.transpose_for_gemm(gpu, qkvz_size, h)?;

    let value_dim = nv * config.linear_value_head_dim;
    let out_proj_nvfp4 = quantize_to_nvfp4(
        &out_proj_local,
        h,
        value_dim,
        gpu,
        absmax_k,
        quantize_k,
        stream,
    )?;

    let out_proj_nvfp4_t = out_proj_nvfp4.transpose_for_gemm(gpu, h, value_dim)?;

    // 2026-09-25: For the `Fp8Dequanted` variant, also cast the BF16 QKVZ and out_proj
    // to FP8 with no scale (`bf16_to_fp8`); they are installed below with
    // `set_fp8_prefill_only_weights`.
    let (qkvz_fp8_prefill, out_proj_fp8_prefill) = if matches!(variant, Nvfp4Variant::Fp8Dequanted)
    {
        tracing::info!(
            "SSM[{lp}] in_proj_qkv + out_proj via native FP8 prefill GEMM \
                 (BF16 act × FP8 weight via fp8_gemm_n128)"
        );
        let b2f_k = gpu.kernel("w4a16", "bf16_to_fp8")?;
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
            out_proj_local.weight,
            out_fp8,
            out_total,
            stream,
        )?;
        gpu.synchronize(stream)?;
        (Some(qkvz_fp8), Some(out_fp8))
    } else {
        (None, None)
    };

    let ssm = SsmWeights {
        in_proj_qkvz: qkvz_dense,
        in_proj_ba: ba_dense,
        conv1d: conv1d_local,
        a_log: a_log_local,
        dt_bias: dt_bias_local,
        norm: norm_local,
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
    // 2026-09-25: Under `metrale_hip` neither the FP8 pre-dequant nor the FP8 casts are
    // installed (kernels/strix-hip has no `fp8_gemm_n128`), so prefill uses the NVFP4
    // weights.
    if !cfg!(metrale_hip) {
        layer.predequant_for_prefill(gpu, config, stream)?;
        // 2026-09-25: After `predequant_for_prefill`, which sets `out_proj_fp8` from the
        // NVFP4 weight, so the FP8 casts replace it. `qkvz_fp8`/`out_proj_fp8` feed the
        // `fp8_gemm_n128` arms of prefill and of the batched decode/verify projections
        // (`qwen3_ssm/init_fp8.rs`, `trait_decode_batched.rs`).
        if qkvz_fp8_prefill.is_some() || out_proj_fp8_prefill.is_some() {
            layer.set_fp8_prefill_only_weights(qkvz_fp8_prefill, out_proj_fp8_prefill);
        }
    }
    // 2026-09-25: `METRALE_GDN_BF16_WEIGHTS=1` also installs this rank's BF16 `out_proj`
    // (from `load_ssm_qwen35`) as `out_proj_dense`, whose prefill arm comes before the FP8
    // and NVFP4 arms (`qwen3_ssm/trait_prefill_helper.rs`).
    if matches!(
        std::env::var("METRALE_GDN_BF16_WEIGHTS").ok().as_deref(),
        Some("1")
    ) {
        layer.out_proj_dense = Some(out_proj_local);
        tracing::info!(
            "SSM[{lp}] METRALE_GDN_BF16_WEIGHTS: out_proj routed through BF16 dense_gemm (overrides FP8/NVFP4)"
        );
    }
    Ok(Box::new(layer))
}
