// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Per-layer loading helpers for DeepSeek-V4: expert projections by
//! on-disk format, the routed/shared quant-format tags, mHC sites and the router bias.
//!
//! Owner: model-arch weight loader.
//! Invariants:
//! - `detect_routed_scale_kind` answers `Mxfp4E8m0` only when the probed
//!   expert's `w1` is one that `load_expert_proj` lands on its MXFP4 arm.
//! - `load_correction_bias` never returns `Ok(None)`.

use anyhow::{Context, Result};
use metrale_cache::kv_cache::KvCacheDtype;
use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_model_weights::weights::WeightStore;

use metrale_model_layers::layer::TransformerLayer;
use metrale_model_layers::layers::FfnComponent;
use metrale_model_layers::layers::MoeLayer;
use metrale_model_layers::layers::qwen3_attention::{
    CompressorWeights, HcHeadWeights, HcSiteWeights, HcWeights, MlaWeights, Qwen3AttentionLayer,
};
use metrale_model_layers::weight_map::quant_helpers::{dense_auto, quantized_v2};
use metrale_model_layers::weight_map::{
    AttentionWeights, DenseWeight, ExpertWeight, MoeWeights, QuantizedWeight, dense, quantized,
};

use super::{attn_sink, compute, csa_ape};

mod layer;
pub use layer::assemble_layer;

/// 2026-09-25: Load one MoE expert projection, choosing the loader by the tensors present:
/// - `.weight_packed` (compressed-tensors NVFP4): `quantized_v2`;
/// - else `.weight_scale_2` (ModelOpt NVFP4): `quantized`;
/// - else an FP8E4M3 `.weight`: `quantized_from_fp8`, which dequantizes the
///   block-scaled FP8 to BF16 and re-quantizes it to NVFP4 at load;
/// - else a UInt8 `.weight` with a `.scale`: `quantized_mxfp4_e8m0` (MXFP4, one
///   E8M0 scale per 32 weights).
///
/// Any other `.weight` dtype is an error.
fn load_expert_proj(
    store: &WeightStore,
    prefix: &str,
    gpu: &dyn GpuBackend,
    qctx: metrale_model_layers::weight_map::QuantizeCtx,
) -> Result<QuantizedWeight> {
    use metrale_model_weights::weights::WeightDtype;
    if store.contains(&format!("{prefix}.weight_packed")) {
        return quantized_v2(store, prefix, gpu);
    }
    if store.contains(&format!("{prefix}.weight_scale_2")) {
        return quantized(store, prefix, gpu);
    }
    let (n, shape_k, dtype) = {
        let w = store
            .get(&format!("{prefix}.weight"))
            .with_context(|| format!("{prefix}: no .weight tensor"))?;
        (w.shape[0], w.shape[1], w.dtype)
    };
    match dtype {
        WeightDtype::FP8E4M3 => metrale_model_layers::weight_map::quantized_from_fp8(
            store,
            prefix,
            n,
            shape_k,
            gpu,
            qctx.absmax_k,
            qctx.quantize_k,
            qctx.stream,
        ),
        // 2026-09-25: MXFP4 lands without a transcode: the store bytes are used
        // as they are, and `detect_routed_scale_kind` tags the layer so the MoE
        // launches its E8M0 kernels.
        WeightDtype::UInt8 => {
            let qw = metrale_model_layers::weight_map::quantized_mxfp4_e8m0(store, prefix)?;
            maybe_dump_expert0(prefix, &qw, gpu)?;
            Ok(qw)
        }
        other => anyhow::bail!(
            "{prefix}: unsupported expert weight dtype {other:?} (expected FP8E4M3 or UInt8)"
        ),
    }
}

/// 2026-09-25: Debug dump, active only with `METRALE_DUMP_EXPERT0=1` and a prefix
/// ending in `layers.0.ffn.experts.0.w1`: copies the device weight and scale
/// bytes the MXFP4 arm landed to `/tmp/metrale_expert0_w1_{weight,scale}.bin`.
/// The copy sizes are fixed (4194304 B and 262144 B), not taken from the tensor.
fn maybe_dump_expert0(prefix: &str, qw: &QuantizedWeight, gpu: &dyn GpuBackend) -> Result<()> {
    if std::env::var("METRALE_DUMP_EXPERT0").as_deref() != Ok("1") {
        return Ok(());
    }
    if !prefix.ends_with("layers.0.ffn.experts.0.w1") {
        return Ok(());
    }
    let mut w = vec![0u8; 4194304];
    gpu.copy_d2h(qw.weight, &mut w)?;
    let mut s = vec![0u8; 262144];
    gpu.copy_d2h(qw.weight_scale, &mut s)?;
    std::fs::write("/tmp/metrale_expert0_w1_weight.bin", &w)?;
    std::fs::write("/tmp/metrale_expert0_w1_scale.bin", &s)?;
    tracing::info!(
        "METRALE_DUMP_EXPERT0: dumped {prefix} weight={} B scale={} B to /tmp/metrale_expert0_w1_*.bin",
        w.len(),
        s.len()
    );
    Ok(())
}

/// 2026-09-25: The format the routed experts land in, for `MoeLayer::experts_scale_kind`.
/// `Mxfp4E8m0` when the probed `w1` has no `.weight_packed`, no `.weight_scale_2`,
/// a `.scale` and a UInt8 `.weight`, the inputs `load_expert_proj` sends to its
/// MXFP4 arm; `Nvfp4` otherwise. Only the first expert this rank loads is probed
/// (the first expert with `force_all_experts`); `Nvfp4` when the rank loads none.
fn detect_routed_scale_kind(
    store: &WeightStore,
    layer_prefix: &str,
    config: &ModelConfig,
    force_all_experts: bool,
) -> metrale_model_layers::weight_map::WeightQuantFormat {
    use metrale_model_layers::weight_map::WeightQuantFormat;
    use metrale_model_weights::weights::WeightDtype;
    for e in 0..config.num_experts {
        if force_all_experts || config.is_local_expert(e) {
            let wp = format!("{layer_prefix}.ffn.experts.{e}.w1");
            let native = !store.contains(&format!("{wp}.weight_packed"))
                && !store.contains(&format!("{wp}.weight_scale_2"))
                && store.contains(&format!("{wp}.scale"))
                && store
                    .get(&format!("{wp}.weight"))
                    .map(|w| w.dtype == WeightDtype::UInt8)
                    .unwrap_or(false);
            return if native {
                WeightQuantFormat::Mxfp4E8m0
            } else {
                WeightQuantFormat::Nvfp4
            };
        }
    }
    WeightQuantFormat::Nvfp4
}

/// 2026-09-25: The test of `detect_routed_scale_kind`, on `{layer}.ffn.shared_experts.w1`.
fn detect_shared_scale_kind(
    store: &WeightStore,
    layer_prefix: &str,
) -> metrale_model_layers::weight_map::WeightQuantFormat {
    use metrale_model_layers::weight_map::WeightQuantFormat;
    use metrale_model_weights::weights::WeightDtype;
    let wp = format!("{layer_prefix}.ffn.shared_experts.w1");
    let native = !store.contains(&format!("{wp}.weight_packed"))
        && !store.contains(&format!("{wp}.weight_scale_2"))
        && store.contains(&format!("{wp}.scale"))
        && store
            .get(&format!("{wp}.weight"))
            .map(|w| w.dtype == WeightDtype::UInt8)
            .unwrap_or(false);
    if native {
        WeightQuantFormat::Mxfp4E8m0
    } else {
        WeightQuantFormat::Nvfp4
    }
}

/// 2026-09-25: Load one HC site (`attn` or `ffn`) as FP32 device buffers, named
/// either `{layer}.hc_{site}_{fn,base,scale}` or `{layer}.{site}_hc.{fn,base,scale}`.
/// A missing tensor or a wrong element count is an error, never a skip.
fn load_hc_site(
    store: &WeightStore,
    layer_prefix: &str,
    site: &str,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<HcSiteWeights> {
    let hc = config.hc_mult;
    let mix_hc = (2 + hc) * hc;
    let hc_dim = hc * config.hidden_size;
    let hc_fn = load_hc_f32(
        store,
        &[
            format!("{layer_prefix}.hc_{site}_fn"),
            format!("{layer_prefix}.{site}_hc.fn"),
        ],
        mix_hc * hc_dim,
        gpu,
    )
    .with_context(|| format!("DeepSeek-V4 HC {site} fn ({layer_prefix})"))?;
    let hc_base = load_hc_f32(
        store,
        &[
            format!("{layer_prefix}.hc_{site}_base"),
            format!("{layer_prefix}.{site}_hc.base"),
        ],
        mix_hc,
        gpu,
    )
    .with_context(|| format!("DeepSeek-V4 HC {site} base ({layer_prefix})"))?;
    let hc_scale = load_hc_f32(
        store,
        &[
            format!("{layer_prefix}.hc_{site}_scale"),
            format!("{layer_prefix}.{site}_hc.scale"),
        ],
        3,
        gpu,
    )
    .with_context(|| format!("DeepSeek-V4 HC {site} scale ({layer_prefix})"))?;
    Ok(HcSiteWeights {
        hc_fn,
        hc_base,
        hc_scale,
        lowrank: None,
    })
}

/// 2026-09-25: Resolve the first existing tensor among `candidates` as an F32
/// device pointer: F32 as stored, BF16 widened into a new buffer. Errors if none
/// exists, if its element count is not `expect_n`, or on any other dtype.
pub fn load_hc_f32(
    store: &WeightStore,
    candidates: &[String],
    expect_n: usize,
    gpu: &dyn GpuBackend,
) -> Result<DevicePtr> {
    use metrale_model_weights::weights::WeightDtype;
    let Some(t) = candidates.iter().find_map(|k| store.get(k).ok()) else {
        anyhow::bail!("HC tensor not found; tried {candidates:?}");
    };
    let n = t.num_elements();
    anyhow::ensure!(
        n == expect_n,
        "HC tensor length {n} != expected {expect_n} (tried {candidates:?})"
    );
    match t.dtype {
        WeightDtype::BF16 => {
            let mut bf16_buf = vec![0u8; n * 2];
            gpu.copy_d2h(t.ptr, &mut bf16_buf)?;
            let mut f32_buf = vec![0u8; n * 4];
            for i in 0..n {
                f32_buf[i * 4 + 2] = bf16_buf[i * 2];
                f32_buf[i * 4 + 3] = bf16_buf[i * 2 + 1];
            }
            let ptr = gpu.alloc(f32_buf.len())?;
            gpu.copy_h2d(&f32_buf, ptr)?;
            Ok(ptr)
        }
        WeightDtype::FP32 => Ok(t.ptr),
        other => anyhow::bail!(
            "load_hc_f32: unsupported dtype {:?} for HC weight (tried {candidates:?}). \
             HC kernels expect F32; BF16 is auto-widened. FP8/E8M0 weights need dequant support.",
            other
        ),
    }
}

/// 2026-09-25: Load the router's expert-selection bias, `[num_experts]`, from the
/// first candidate name present: BF16 is widened into a new F32 buffer, and any
/// other dtype is returned as stored. Never returns `Ok(None)`: with no candidate
/// present it returns a zeroed F32 buffer.
fn load_correction_bias(
    store: &WeightStore,
    layer_prefix: &str,
    num_experts: usize,
    gpu: &dyn GpuBackend,
) -> Result<Option<DenseWeight>> {
    use metrale_model_weights::weights::WeightDtype;
    let candidates = [
        format!("{layer_prefix}.ffn.gate.e_score_correction_bias"),
        format!("{layer_prefix}.ffn.gate.correction_bias"),
        format!("{layer_prefix}.mlp.gate.e_score_correction_bias"),
        format!("{layer_prefix}.ffn.gate.bias"),
        format!("{layer_prefix}.mlp.gate.bias"),
    ];
    let Some(bias_t) = candidates.iter().find_map(|k| store.get(k).ok()) else {
        // 2026-09-25: A zeroed bias, not `None`: the MoE takes the biased top-k
        // path, whose kernel follows `scoring_func`, only when a bias is present;
        // without one it routes with plain softmax top-k.
        let bytes = num_experts * 4;
        let ptr = gpu.alloc(bytes)?;
        gpu.memset(ptr, 0, bytes)?;
        return Ok(Some(DenseWeight { weight: ptr }));
    };
    let n = bias_t.num_elements();
    anyhow::ensure!(
        n == num_experts,
        "DeepSeek-V4 correction_bias length {n} != num_experts {num_experts}"
    );
    // 2026-09-25: The biased top-k kernels read the bias as `const float*`; BF16
    // is widened exactly (low 16 bits zero). Any other dtype is passed as stored.
    if bias_t.dtype == WeightDtype::BF16 {
        let mut bf16_buf = vec![0u8; n * 2];
        gpu.copy_d2h(bias_t.ptr, &mut bf16_buf)?;
        let mut f32_buf = vec![0u8; n * 4];
        for i in 0..n {
            f32_buf[i * 4 + 2] = bf16_buf[i * 2];
            f32_buf[i * 4 + 3] = bf16_buf[i * 2 + 1];
        }
        let ptr = gpu.alloc(f32_buf.len())?;
        gpu.copy_h2d(&f32_buf, ptr)?;
        Ok(Some(DenseWeight { weight: ptr }))
    } else {
        Ok(Some(DenseWeight { weight: bias_t.ptr }))
    }
}
