// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `Qwen35DenseWeightLoader`, the loader for dense Qwen3.5-family checkpoints
//! (`is_qwen35_dense`): full-attention and linear-attention (GDN) layers with a dense FFN,
//! each built from the checkpoint's form (native FP8, NVFP4, BF16 or keep-packed Q2_0),
//! and `prune_after_load`.
//!
//! Owner: model-arch weight loader (Qwen3.5 dense).
//! Invariants:
//! - `prune_after_load` frees a layer's store tensors only when the predicate
//!   `load_layers` used for that layer (`gdn_fp8_arm_selected`,
//!   `ffn_gateup_fused_selected`) holds.

use anyhow::Result;
use metrale_cache::kv_cache::KvCacheDtype;
use metrale_config::{LayerType, ModelConfig};
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_model_weights::weights::{WeightDtype, WeightStore};

use super::{ModelWeightLoader, WeightFormat};
use metrale_model_layers::layer::TransformerLayer;
use metrale_model_layers::weight_map::loaders_moe::load_mtp;
use metrale_model_layers::weight_map::{
    DenseWeight, Fp8Weight, MtpWeights, Nvfp4Variant, PackedQ2Weight, dense, detect_nvfp4_variant,
};

/// 2026-09-25: The Q2_0 group size when `{prefix}.weight` is a keep-packed ternary tensor
/// (`WeightDtype::PackedQ2_0`, which the GGUF loader produces under
/// `METRALE_GGUF_NATIVE_Q2=1`), else `None`.
fn proj_q2_group(store: &WeightStore, prefix: &str) -> Option<u16> {
    store
        .get(&format!("{prefix}.weight"))
        .ok()
        .and_then(|w| w.q2_group())
}

/// 2026-09-25: Wraps the store's keep-packed `block_q2_0` buffer for `{prefix}.weight` as a
/// [`PackedQ2Weight`] (`[n, k]` and group) without copying; the store keeps owning it.
/// Errors unless the tensor is keep-packed Q2_0 and 2-D.
fn packed_q2_from_store(store: &WeightStore, prefix: &str) -> Result<PackedQ2Weight> {
    let w = store.get(&format!("{prefix}.weight"))?;
    let group = w
        .q2_group()
        .ok_or_else(|| anyhow::anyhow!("{prefix}.weight is not keep-packed Q2_0"))?;
    anyhow::ensure!(
        w.shape.len() == 2,
        "packed Q2_0 {prefix}.weight must be 2D, got {:?}",
        w.shape
    );
    Ok(PackedQ2Weight {
        weight: w.ptr,
        n: w.shape[0] as u32,
        k: w.shape[1] as u32,
        group,
    })
}

/// 2026-09-25: True when `{prefix}.weight` is FP8E4M3 with a block scale: a
/// `.weight_scale_inv`, or a 2D `.weight_scale`. The same test as
/// `qwen35::load_layers::proj_is_native_fp8`.
fn proj_is_native_fp8(store: &WeightStore, prefix: &str) -> bool {
    let is_fp8_weight = store
        .get(&format!("{prefix}.weight"))
        .map(|w| w.dtype == WeightDtype::FP8E4M3)
        .unwrap_or(false);
    let has_block_scale = store.contains(&format!("{prefix}.weight_scale_inv"))
        || store
            .get(&format!("{prefix}.weight_scale"))
            .map(|s| s.shape.len() == 2)
            .unwrap_or(false);
    is_fp8_weight && has_block_scale
}

/// 2026-09-25: True when `{prefix}.weight` is a 2-D FP8E4M3 tensor whose
/// `weight_scale_inv` or `weight_scale` is a single value or exactly the
/// `[ceil(N/128), ceil(K/128)]` grid `w8a16` indexes as `block_scale[n/128, k/128]`
/// (`load_fp8_block_scaled_as_fp8weight` repeats a single value over that grid).
/// A per-row `[N, 1]` scale returns false: read as a grid, row `n` would take the scale
/// in cell `n/128`. Such a projection is dequantized instead
/// (`dequant_fp8_blockscaled_to_bf16` infers `[1, K]` blocks from the scale's shape).
fn proj_is_fp8_any_scale(store: &WeightStore, prefix: &str) -> bool {
    let Ok(w) = store.get(&format!("{prefix}.weight")) else {
        return false;
    };
    if w.dtype != WeightDtype::FP8E4M3 || w.shape.len() != 2 {
        return false;
    }
    let (n, k) = (w.shape[0], w.shape[1]);

    for key in [
        format!("{prefix}.weight_scale_inv"),
        format!("{prefix}.weight_scale"),
    ] {
        let Ok(s) = store.get(&key) else { continue };
        if s.num_elements() == 1 {
            return true;
        }
        if s.shape.len() == 2 && s.shape[0] == n.div_ceil(128) && s.shape[1] == k.div_ceil(128) {
            return true;
        }
    }
    false
}

/// 2026-09-25: Concatenates two block-scaled FP8 weights along rows,
/// `[n_a, k] ++ [n_b, k] -> [n_a+n_b, k]`, into new weight and FP32 scale-grid buffers.
/// The caller must ensure `n_a % 128 == 0`, so the two scale grids meet at a block
/// boundary; nothing checks it.
fn concat_fp8_block_scaled(
    a: &Fp8Weight,
    b: &Fp8Weight,
    k: usize,
    gpu: &dyn GpuBackend,
) -> Result<Fp8Weight> {
    let kb = k.div_ceil(128);
    let a_w = a.n as usize * k;
    let b_w = b.n as usize * k;
    let weight = gpu.alloc(a_w + b_w)?;
    gpu.copy_d2d(a.weight, weight, a_w)?;
    gpu.copy_d2d(b.weight, weight.offset(a_w), b_w)?;
    let a_s = (a.n as usize).div_ceil(128) * kb * 4;
    let b_s = (b.n as usize).div_ceil(128) * kb * 4;
    let row_scale = gpu.alloc(a_s + b_s)?;
    gpu.copy_d2d(a.row_scale, row_scale, a_s)?;
    gpu.copy_d2d(b.row_scale, row_scale.offset(a_s), b_s)?;
    Ok(Fp8Weight {
        weight,
        row_scale,
        n: a.n + b.n,
        k: k as u32,
        scale_format: metrale_model_layers::weight_map::WeightQuantFormat::Fp8BlockScaled,
    })
}

/// 2026-09-25: `METRALE_DENSE_FP8=1` enables the native FP8 dense-FFN and attention arms
/// (`ffn_fp8_arm_selected`, `attn_fp8`); off by default. Measured 2026-06-29 on a 9B FP8
/// checkpoint: correct text at ~30 tok/s against ~40 tok/s on the NVFP4 path.
fn dense_fp8_enabled() -> bool {
    std::env::var("METRALE_DENSE_FP8").as_deref() == Ok("1")
}

/// 2026-09-25: Whether the native block-scaled FP8 GDN arm runs for this SSM layer.
/// `load_layers` and `prune_after_load` both call it; the second frees the store tensors
/// the first copied into the fused `[QKV|Z]` weight. The keep-packed Q2 arm is tested
/// first in `load_layers`, so its condition is part of this one.
fn gdn_fp8_arm_selected(store: &WeightStore, la: &str, tp_size: usize) -> bool {
    let q2 = tp_size.max(1) == 1
        && std::env::var_os("METRALE_NO_Q2_GDN").is_none()
        && proj_q2_group(store, &format!("{la}.in_proj_qkv")).is_some()
        && proj_q2_group(store, &format!("{la}.in_proj_z")).is_some();
    !q2 && std::env::var_os("METRALE_NO_GDN_FP8").is_none()
        && proj_is_fp8_any_scale(store, &format!("{la}.in_proj_qkv"))
        && proj_is_fp8_any_scale(store, &format!("{la}.in_proj_z"))
        && proj_is_fp8_any_scale(store, &format!("{la}.out_proj"))
}

/// 2026-09-25: The dense-FFN width: `intermediate_size`, or `moe_intermediate_size` when
/// that is 0; the same order as `load_dense_ffn` (`weight_map/fp8_lut.rs`).
fn ffn_inter(config: &ModelConfig) -> usize {
    if config.intermediate_size > 0 {
        config.intermediate_size
    } else {
        config.moe_intermediate_size
    }
}

/// 2026-09-25: Whether the native FP8 dense-FFN arm runs for this layer:
/// `METRALE_DENSE_FP8=1`, TP=1, the `Fp8Dequanted` variant and a block-scaled FP8
/// `mlp.gate_proj`. `load_layers` calls it, and `prune_after_load` through
/// `ffn_gateup_fused_selected`.
fn ffn_fp8_arm_selected(
    store: &WeightStore,
    config: &ModelConfig,
    variant: Nvfp4Variant,
    lp: &str,
) -> bool {
    dense_fp8_enabled()
        && config.tp_world_size.max(1) == 1
        && matches!(variant, Nvfp4Variant::Fp8Dequanted)
        && proj_is_native_fp8(store, &format!("{lp}.mlp.gate_proj"))
}

/// 2026-09-25: Whether this layer's gate and up are fused into one `[2*inter, hidden]`
/// block-scaled FP8 weight: the FP8 FFN arm is selected, the target arms
/// `ffn_gateup_fused` (`[defaults]`, or `METRALE_FFN_GATEUP_FUSED`), the model has no
/// experts, both widths are multiples of 128 (so the concatenated scale grids meet on a
/// block boundary), and both tensors are `[inter, hidden]`. `prune_after_load` frees
/// the source tensors of exactly the layers this returns true for.
fn ffn_gateup_fused_selected(
    store: &WeightStore,
    config: &ModelConfig,
    variant: Nvfp4Variant,
    lp: &str,
) -> bool {
    let inter = ffn_inter(config);
    let hidden = config.hidden_size;
    // 2026-09-25: Checked here rather than at the call site because `prune_after_load`
    // frees on this answer: tensors whose shape is not `[inter, hidden]` from `config`
    // are not fused, and so are not freed.
    let on_disk = |name: &str| {
        store
            .get(&format!("{lp}.mlp.{name}.weight"))
            .is_ok_and(|w| w.shape == [inter, hidden])
    };
    ffn_fp8_arm_selected(store, config, variant, lp)
        && metrale_model_layers::layers::dense_ffn::gateup_fused::ffn_gateup_fused()
        && config.num_experts == 0
        && inter > 0
        && inter.is_multiple_of(128)
        && hidden.is_multiple_of(128)
        && on_disk("gate_proj")
        && on_disk("up_proj")
}

// 2026-09-25: `pub` because `weight_loader/mod.rs` re-exports them for the server's
// preflight (`preflight/headroom.rs`), which reads `predicted_residency`; that module
// prices its terms with `fp8_residency`'s shape helpers.
pub mod fp8_residency;
mod loaders_b;
pub mod predicted_residency;
mod rowwise_fp8;

mod attn_arms;
mod attn_layer;
mod ffn_arm;
mod gdn_dequant;
mod gdn_layer;
mod load_cx;
mod prune;

use fp8_residency::{DerivedResidency, RouteEnv};
use load_cx::{Flow, LayerIn, LoadCx};

pub struct Qwen35DenseWeightLoader;

impl ModelWeightLoader for Qwen35DenseWeightLoader {
    fn supports_tp(&self) -> bool {
        // 2026-09-25: The NVFP4 and BF16 attention arms shard Q/K/V/O, and the GDN dequant
        // path shards by head (`shard_gdn_*`). The native FP8 attention and FFN arms, the
        // UInt8 attention weights and the keep-packed Q2 arms require TP=1; the native FP8
        // and pre-quantized NVFP4 GDN arms do not shard.
        true
    }

    fn load_layers(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        layer_kv_dtypes: &[KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        let layer_types = if config.layer_types.is_empty() {
            (0..config.num_hidden_layers)
                .map(|i| config.layer_type(i))
                .collect::<Vec<_>>()
        } else {
            config.layer_types.clone()
        };

        let mut layers: Vec<Box<dyn TransformerLayer>> =
            Vec::with_capacity(config.num_hidden_layers);
        let mut attn_idx = 0usize;

        let absmax_k = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
        let quantize_k = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;
        let stream = gpu.default_stream();
        let h = config.hidden_size;

        let variant = detect_nvfp4_variant(store, config);
        let weight_format = WeightFormat::detect(store, config);
        tracing::info!(
            "Weight format: {:?}, NVFP4 variant: {:?}",
            weight_format,
            variant
        );

        // 2026-09-25: Unless `METRALE_NO_GDN_FP8_PREFILL` is set, the GDN dequant path below
        // also builds FP8 casts of its BF16 `[Q|K|V|Z]` and out_proj (`bf16_to_fp8`).
        let fp8_ssm_prefill = std::env::var_os("METRALE_NO_GDN_FP8_PREFILL").is_none();
        let bf16_to_fp8_k = if fp8_ssm_prefill {
            tracing::info!(
                "SSM in_proj_qkv + out_proj via native FP8 prefill GEMM \
                 (BF16 act × FP8 weight via fp8_gemm_n128). PREFILL ONLY: \
                 decode + batched verify read the NVFP4 copy — weight-streaming \
                 GEMV at M<=8, tile GEMM above it (trait_decode_batched.rs). \
                 The FP8 copy reaches decode only at M<=8 on a build whose \
                 batched NVFP4 GEMVs are absent"
            );
            Some(gpu.kernel("w4a16", "bf16_to_fp8")?)
        } else {
            None
        };

        // 2026-09-25: `METRALE_MEM_PROFILE` (any value) logs free GPU memory at the start
        // and at every 8th layer.
        let mem_profile = std::env::var("METRALE_MEM_PROFILE").is_ok();
        let log_free = |tag: &str| {
            if mem_profile && let Ok(free) = gpu.free_memory() {
                tracing::info!("MEM_PROFILE[{tag}]: {:.2} GB GPU-free", free as f64 / 1e9);
            }
        };
        log_free("dense-load-start");

        // 2026-09-25: The routing levers are read once per load (`RouteEnv::from_env`,
        // `fp8_residency.rs`).
        let route_env = RouteEnv::from_env();
        let mut residency = DerivedResidency::default();
        let cx = LoadCx {
            store,
            config,
            gpu,
            layer_kv_dtypes,
            variant,
            absmax_k,
            quantize_k,
            stream,
            h,
            bf16_to_fp8_k,
            route_env: &route_env,
        };

        for (i, lt) in layer_types.iter().enumerate() {
            if i % 8 == 0 {
                log_free(&format!("layer-{i}"));
            }
            let lp = config.layer_prefix(i);
            let input_norm = dense(store, &format!("{lp}.input_layernorm.weight"))?;
            let post_attn_norm = dense(store, &format!("{lp}.post_attention_layernorm.weight"))?;
            let ffn = ffn_arm::build_dense_ffn(&cx, &mut residency, &lp)?;

            match lt {
                LayerType::FullAttention => {
                    let l = LayerIn {
                        i,
                        lp: &lp,
                        input_norm,
                        post_attn_norm,
                        ffn,
                    };
                    if let Flow::Continue = attn_layer::load_full_attention(
                        &cx,
                        &mut residency,
                        &mut layers,
                        &mut attn_idx,
                        l,
                    )? {
                        continue;
                    }
                }
                LayerType::LinearAttention => {
                    let l = LayerIn {
                        i,
                        lp: &lp,
                        input_norm,
                        post_attn_norm,
                        ffn,
                    };
                    if let Flow::Continue =
                        gdn_layer::load_linear_attention(&cx, &mut residency, &mut layers, l)?
                    {
                        continue;
                    }
                }
                LayerType::SlidingAttention => {
                    unreachable!("unexpected SlidingAttention in this loader")
                }
                LayerType::Moe => unreachable!("Qwen3.5 dense has no standalone MoE layers"),
                // 2026-09-25: GLM-5.3 `deepseek_sparse_attention` needs a DSA indexer and per-query
                // top-k, which this loader does not build; it is refused, not served as dense
                // attention.
                LayerType::SparseAttention => anyhow::bail!(
                    "layer {i}: SparseAttention needs a DSA indexer and per-query top-k; Qwen3.5 dense has neither"
                ),
            }

            if (i + 1) % 10 == 0 {
                tracing::info!("Loaded layers 0..{}", i + 1);
                metrale_telemetry::progress::layer(i + 1, config.num_hidden_layers);
            }
        }

        tracing::info!(
            "Qwen3.5 dense weight loader: {} layers ({} attention, {} SSM, dense FFN)",
            layers.len(),
            attn_idx,
            layers.len() - attn_idx,
        );
        // 2026-09-25: Logged at info on every load: `weights` is the checkpoint as the store
        // holds it, `derived` what this loader built and the store owns, `not built` what
        // the residency plan skipped.
        tracing::info!("{}", residency.summary(store.resident_bytes()));
        for (label, bytes, count) in store.derived().by_label() {
            tracing::debug!(
                "  derived-weight owner: {:>9.1} MB x{:<5} {label}",
                bytes as f64 / (1024.0 * 1024.0),
                count,
            );
        }

        Ok(layers)
    }

    /// 2026-09-25: Frees the store tensors that `load_layers` copied into fused buffers and
    /// no longer reads: `in_proj_qkv`, `in_proj_z` (with their scales), `in_proj_a` and
    /// `in_proj_b` where `gdn_fp8_arm_selected` holds, and `mlp.gate_proj`/`mlp.up_proj`
    /// (with their scales) where `ffn_gateup_fused_selected` holds. `out_proj`, `conv1d`,
    /// `A_log`, `dt_bias`, `norm` and `mlp.down_proj` are kept: layers may still alias them.
    fn prune_after_load(
        &self,
        store: &mut WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<()> {
        let doomed = prune::doomed_after_load(store, config);
        if doomed.is_empty() {
            return Ok(());
        }
        let (count, bytes) = store.free_matching(gpu, |name| doomed.contains(name))?;
        tracing::info!(
            "native FP8: released {count} store tensors ({:.2} GB) consumed by the fused              [QKV|Z] SSM concat, the BA interleave and the dense-FFN gate+up fusion;              out_proj/conv1d/A_log/dt_bias/norm/down_proj kept (still aliased)",
            bytes as f64 / 1e9,
        );
        Ok(())
    }

    fn load_embedding(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        loaders_b::load_embedding(store, config)
    }

    fn load_final_norm(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        loaders_b::load_final_norm(store, config)
    }

    fn load_lm_head(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        loaders_b::load_lm_head(store, config, gpu)
    }

    fn load_mtp_weights(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<Option<MtpWeights>> {
        if !store.contains("mtp.fc.weight") {
            return Ok(None);
        }
        let variant = detect_nvfp4_variant(store, config);
        tracing::info!(
            "Loading dense MTP weights (variant={:?}, hidden={}, inter={})",
            variant,
            config.hidden_size,
            config.intermediate_size,
        );
        // 2026-09-25: `load_mtp` returns a dense head (`dense_ffn` set) when the MTP layer has
        // `mlp.gate_proj.weight` and no router.
        let mtp = load_mtp(store, config.num_experts, gpu, variant)?;
        if mtp.dense_ffn.is_some() {
            tracing::info!("Dense MTP head ready (FP8 e4m3 projections + dense gate/up/down MLP)");
        } else {
            tracing::info!(
                "MoE MTP head ready ({} experts) — dense loader sees MoE bundle",
                mtp.experts.len(),
            );
        }
        Ok(Some(mtp))
    }

    fn load_vision_encoder(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<Option<metrale_model_layers::layers::VisionTower>> {
        // 2026-09-25: The MoE loader's `load_vision_encoder` reads only `store`, `config` and
        // `gpu`, so the dense loader reuses it.
        super::qwen35::Qwen35WeightLoader.load_vision_encoder(store, config, gpu)
    }
}
