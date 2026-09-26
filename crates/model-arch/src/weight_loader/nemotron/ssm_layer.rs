// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Nemotron-H Mamba-2 SSM layer construction: loads the mixed-quant
//! `in_proj`/`out_proj` and picks the decode and prefill weight forms.
//!
//! Owner: model-arch weight loader (Nemotron-H).
//! Invariants:
//! - A prefill copy (FP8 pre-dequant or transposed NVFP4) is derived only from
//!   non-NULL NVFP4 `ssm.in_proj`/`ssm.out_proj`: loaded from an NVFP4 checkpoint
//!   or produced by the requant arm, which runs under the same gate.

use anyhow::{Result, ensure};
use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_model_weights::weights::WeightStore;

use super::NemotronHWeightLoader;
use crate::nemotron_mamba2::NemotronMamba2Layer;
use metrale_model_layers::weight_map::fp8_dequant::dequant_fp8_to_bf16_into;
use metrale_model_layers::weight_map::{
    DenseWeight, NemotronSsmQuant, dense, load_fp8_block_scaled_as_fp8weight, load_nemotron_ssm,
    quantize_to_nvfp4,
};

impl NemotronHWeightLoader {
    /// 2026-09-25: Builds one Mamba-2 SSM layer, for the `LayerType::LinearAttention` arm of
    /// `load_layers` (`nemotron.rs`).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn build_ssm_layer(
        gpu: &dyn GpuBackend,
        store: &WeightStore,
        config: &ModelConfig,
        i: usize,
        h: usize,
        lp: &str,
        norm: DenseWeight,
        quantize_k: KernelHandle,
        absmax_k: KernelHandle,
        scratch: DevicePtr,
        stream: u64,
    ) -> Result<NemotronMamba2Layer> {
        // 2026-09-25: `load_nemotron_ssm` reads the format (NVFP4, FP8 or BF16) from
        // `in_proj` and returns NULL projections unless it is NVFP4.
        let (mut ssm, quant_kind) = load_nemotron_ssm(store, i, gpu, lp)?;
        let p = format!("{lp}.mixer");
        // 2026-09-25: `METRALE_NEMOTRON_NATIVE_FP8_SSM` selects how an FP8 layer runs.
        // Unset, "1" or "both": FP8 for decode (`w8a16_gemv`) and prefill
        // (`w8a16_gemm_pipelined` or `w8a16_gemm`). "decode": FP8 for decode, NVFP4
        // requant for prefill. Any other value: NVFP4 requant for both.
        let mode =
            std::env::var("METRALE_NEMOTRON_NATIVE_FP8_SSM").unwrap_or_else(|_| "1".to_string());
        let native_fp8_enabled = matches!(mode.as_str(), "1" | "both" | "decode");
        let native_prefill_wanted = matches!(mode.as_str(), "1" | "both");
        // 2026-09-25: Native FP8 also needs an FP8 `out_proj`, `w8a16_gemv` and one of
        // the two `w8a16_gemm` kernels; without them the load warns once and takes the
        // requant arm.
        let out_is_fp8 = store.contains(&format!("{p}.out_proj.weight_scale"));
        let gemv_k = metrale_model_layers::layers::try_kernel(gpu, "w8a16_gemv", "w8a16_gemv");
        let gemm_pipe_k = metrale_model_layers::layers::try_kernel(
            gpu,
            "w8a16_gemm_pipelined",
            "w8a16_gemm_pipelined",
        );
        let gemm_k = metrale_model_layers::layers::try_kernel(gpu, "w8a16_gemm", "w8a16_gemm");
        let native_fp8 = native_fp8_enabled
            && quant_kind == NemotronSsmQuant::Fp8
            && out_is_fp8
            && gemv_k.0 != 0
            && (gemm_pipe_k.0 != 0 || gemm_k.0 != 0);
        // 2026-09-25: A BF16 checkpoint layer (no `weight_scale` on `in_proj` or
        // `out_proj`) stays BF16 through `dense_gemv_bf16` / `dense_gemm_bf16_pipelined`
        // unless `METRALE_NEMOTRON_NATIVE_BF16_SSM=0` or either kernel is missing; then
        // it is quantized to NVFP4.
        let out_is_bf16 = !store.contains(&format!("{p}.out_proj.weight_scale"));
        let dgemm_k =
            metrale_model_layers::layers::try_kernel(gpu, "gemm", "dense_gemm_bf16_pipelined");
        let dgemv_k = metrale_model_layers::layers::try_kernel(gpu, "gemv", "dense_gemv_bf16");
        let native_bf16 = std::env::var("METRALE_NEMOTRON_NATIVE_BF16_SSM").as_deref() != Ok("0")
            && quant_kind == NemotronSsmQuant::Bf16
            && out_is_bf16
            && dgemm_k.0 != 0
            && dgemv_k.0 != 0;
        if native_fp8_enabled && quant_kind == NemotronSsmQuant::Fp8 && !native_fp8 {
            static NATIVE_FALLBACK_WARN: std::sync::Once = std::sync::Once::new();
            NATIVE_FALLBACK_WARN.call_once(|| {
                tracing::warn!(
                    "L{i} SSM: native FP8 unavailable (out_proj_fp8={out_is_fp8} \
                     w8a16_gemv={} w8a16_gemm_pipelined={} w8a16_gemm={}) — falling back \
                     to the FP8→BF16→NVFP4 double-quant path",
                    gemv_k.0 != 0,
                    gemm_pipe_k.0 != 0,
                    gemm_k.0 != 0,
                );
            });
        }
        let native_prefill = native_fp8 && native_prefill_wanted;
        tracing::info!(
            "L{i} SSM quant={quant_kind:?} native_fp8={native_fp8} native_bf16={native_bf16} \
             native_prefill={native_prefill} \
             in_proj_size={} d_inner={} h={h}",
            config.mamba2_in_proj_size(),
            config.mamba2_d_inner(),
        );
        let native = if native_fp8 {
            let mut in_fp8 =
                load_fp8_block_scaled_as_fp8weight(store, &format!("{p}.in_proj"), gpu)?;
            let mut out_fp8 =
                load_fp8_block_scaled_as_fp8weight(store, &format!("{p}.out_proj"), gpu)?;
            // 2026-09-25: Copy the FP8 bytes into buffers the layer owns rather than keep
            // the `WeightStore` pointer that `load_fp8_block_scaled_as_fp8weight` returns
            // (loaders_fp8.rs `let weight_ptr = w.ptr`).
            for w in [&mut in_fp8, &mut out_fp8] {
                let bytes = (w.n as usize) * (w.k as usize);
                let owned = gpu.alloc(bytes)?;
                gpu.copy_d2d(w.weight, owned, bytes)?;
                w.weight = owned;
            }
            if i == 0 {
                let mut head = [0u8; 16];
                gpu.copy_d2h(in_fp8.weight, &mut head)?;
                tracing::debug!(
                    "L0 in_proj FP8 head: {}",
                    head.iter()
                        .map(|b| format!("{b:02x}"))
                        .collect::<Vec<_>>()
                        .join(" ")
                );
            }
            // 2026-09-25: `w8a16_gemv` reads K in 16-value chunks and never reads a K % 16
            // tail (kernels/gb10/common/w8a16_gemv.cu), so K must be a multiple of 16.
            ensure!(
                in_fp8.n as usize == config.mamba2_in_proj_size() && in_fp8.k as usize == h,
                "L{i} SSM in_proj FP8 shape [{},{}] != [{},{h}]",
                in_fp8.n,
                in_fp8.k,
                config.mamba2_in_proj_size(),
            );
            ensure!(
                out_fp8.n as usize == h && out_fp8.k as usize == config.mamba2_d_inner(),
                "L{i} SSM out_proj FP8 shape [{},{}] != [{h},{}]",
                out_fp8.n,
                out_fp8.k,
                config.mamba2_d_inner(),
            );
            ensure!(
                in_fp8.k % 16 == 0 && out_fp8.k % 16 == 0,
                "L{i} SSM: w8a16 requires K%16==0, got in_proj K={} out_proj K={}",
                in_fp8.k,
                out_fp8.k,
            );
            Some((in_fp8, out_fp8))
        } else {
            None
        };
        // 2026-09-25: Requant to NVFP4 (FP8 -> BF16 -> NVFP4, or BF16 -> NVFP4) for a
        // non-NVFP4 layer whose prefill is not native: the prefill copies below and the
        // NVFP4 decode path read `ssm.in_proj`/`ssm.out_proj`, which `load_nemotron_ssm`
        // leaves NULL unless the checkpoint is NVFP4.
        if !native_prefill && !native_bf16 && quant_kind != NemotronSsmQuant::Nvfp4 {
            let in_proj_dense = if quant_kind == NemotronSsmQuant::Fp8 {
                dequant_fp8_to_bf16_into(store, &format!("{p}.in_proj"), gpu, scratch)?
            } else {
                dense(store, &format!("{p}.in_proj.weight"))?
            };
            ssm.in_proj = quantize_to_nvfp4(
                &in_proj_dense,
                config.mamba2_in_proj_size(),
                h,
                gpu,
                absmax_k,
                quantize_k,
                stream,
            )?;
            let out_proj_dense = if out_is_fp8 {
                dequant_fp8_to_bf16_into(store, &format!("{p}.out_proj"), gpu, scratch)?
            } else {
                dense(store, &format!("{p}.out_proj.weight"))?
            };
            ssm.out_proj = quantize_to_nvfp4(
                &out_proj_dense,
                h,
                config.mamba2_d_inner(),
                gpu,
                absmax_k,
                quantize_k,
                stream,
            )?;
        }
        // 2026-09-25: Prefill copies of the NVFP4 projections, built only when prefill
        // is neither native FP8 nor native BF16 (both leave the projections NULL):
        // - by default, pre-dequantized FP8 E4M3 `[N, K]` (`predequant_nvfp4_to_fp8`),
        //   read by `fp8_gemm_t` / `fp8_fp8_gemm_t`;
        // - with `METRALE_NO_SSM_FP8_PREFILL` set, transposed NVFP4 for `w4a16_gemm_t`
        //   / `w4a16_gemm_t_m128`;
        // - with `METRALE_NO_SSM_PREFILL_T` also set, no copy.
        let fp8_prefill =
            !native_prefill && !native_bf16 && std::env::var("METRALE_NO_SSM_FP8_PREFILL").is_err();
        let prefill_t = !native_prefill
            && !native_bf16
            && !fp8_prefill
            && std::env::var("METRALE_NO_SSM_PREFILL_T").is_err();
        let proj_t = if prefill_t {
            let in_t = ssm
                .in_proj
                .transpose_for_gemm(gpu, config.mamba2_in_proj_size(), h)?;
            let out_t = ssm
                .out_proj
                .transpose_for_gemm(gpu, h, config.mamba2_d_inner())?;
            Some((in_t, out_t))
        } else {
            None
        };
        let proj_fp8 = if fp8_prefill {
            let pdq_k = gpu.kernel("w4a16", "predequant_nvfp4_to_fp8")?;
            let in_fp8 = ssm.in_proj.predequant_to_fp8(
                gpu,
                pdq_k,
                config.mamba2_in_proj_size(),
                h,
                stream,
            )?;
            let out_fp8 =
                ssm.out_proj
                    .predequant_to_fp8(gpu, pdq_k, h, config.mamba2_d_inner(), stream)?;
            Some((in_fp8, out_fp8))
        } else {
            None
        };
        let bf16w = if native_bf16 {
            Some((
                dense(store, &format!("{p}.in_proj.weight"))?,
                dense(store, &format!("{p}.out_proj.weight"))?,
            ))
        } else {
            None
        };
        let mut layer = NemotronMamba2Layer::new(norm, ssm, config, gpu, i)?;
        if let Some((in_w, out_w)) = bf16w {
            layer.set_bf16_weights(in_w, out_w);
            ensure!(
                layer.bf16_native_ready(),
                "L{i} SSM: native BF16 selected but the dense kernels are missing"
            );
        }
        if let Some((in_fp8, out_fp8)) = native {
            layer.set_fp8_weights(Some(in_fp8), Some(out_fp8), native_prefill)?;
        }
        if let Some((in_t, out_t)) = proj_t {
            layer.set_prefill_weights(Some(in_t), Some(out_t));
        }
        if let Some((in_fp8, out_fp8)) = proj_fp8 {
            layer.set_fp8_prefill_weights(in_fp8, out_fp8);
        }
        Ok(layer)
    }
}
