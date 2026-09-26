// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `assemble_layer`: build one DeepSeek-V4 layer (MLA attention,
//! MoE FFN, optional compressor, mHC) from its loaded tensors.
//!
//! Owner: model-arch weight loader.
//! Invariants:
//! - A returned layer always has mHC weights: `hc_mult == 0` is an error.
//! - The router gate is never quantized (`gate_nvfp4` is `None`).

use super::*;

#[allow(clippy::too_many_arguments)]
pub fn assemble_layer(
    layer_idx: usize,
    layer_prefix: &str,
    // 2026-09-25: Load every expert regardless of EP sharding. The MTP module
    // sets it: its MoE runs with `comm: None`, so no all-reduce would add the
    // experts other ranks hold.
    force_all_experts: bool,
    input_norm: DenseWeight,
    post_attn_norm: DenseWeight,
    wq_a: DenseWeight,
    wq_a_nvfp4: Option<QuantizedWeight>,
    wq_b: DenseWeight,
    wq_b_nvfp4: Option<QuantizedWeight>,
    q_a_norm: DenseWeight,
    wkv_a: DenseWeight,
    wkv_a_nvfp4: Option<QuantizedWeight>,
    wkv_b: DenseWeight,
    kv_a_norm: DenseWeight,
    o_dense: DenseWeight,
    o_nvfp4: Option<QuantizedWeight>,
    w_uk_t: DenseWeight,
    w_uv: DenseWeight,
    wq_b_rope: DenseWeight,
    w_qk_absorbed: DenseWeight,
    w_uk_block_diag: DenseWeight,
    w_uv_block_diag: DenseWeight,
    yarn_inv_freq: DevicePtr,
    wo_a: DenseWeight,
    hc_head: Option<HcHeadWeights>,
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    layer_kv_dtypes: &[KvCacheDtype],
) -> Result<Box<dyn TransformerLayer>> {
    // 2026-09-25: `layer_prefix` is the tensor-name prefix: `layers.{idx}` for a
    // main layer, `mtp.0` for the MTP module. `layer_idx` indexes
    // `compress_ratios` and `layer_kv_dtypes` and is compared with
    // `num_hash_layers`; an index past the end of either list gives no
    // compressor and BF16 KV.
    let lp = layer_prefix.to_string();
    let h = config.hidden_size;
    let kv_dtype = layer_kv_dtypes
        .get(layer_idx)
        .copied()
        .unwrap_or(KvCacheDtype::Bf16);

    // 2026-09-25: Kernels for the FP8 → NVFP4 conversion in `load_expert_proj`.
    let qctx = metrale_model_layers::weight_map::QuantizeCtx {
        absmax_k: gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?,
        quantize_k: gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?,
        stream: gpu.default_stream(),
    };

    let p = &lp;
    let gate = dense(store, &format!("{p}.ffn.gate.weight"))?;
    // 2026-09-25: The router gate is used as stored, not quantized to NVFP4: a
    // 4-bit copy is too coarse for the low-margin top-k expert selection.
    let gate_nvfp4 = None;

    let mut experts = Vec::with_capacity(config.num_experts);
    for e in 0..config.num_experts {
        if force_all_experts || config.is_local_expert(e) {
            let ep = format!("{p}.ffn.experts.{e}");
            let gate_proj = load_expert_proj(store, &format!("{ep}.w1"), gpu, qctx)
                .with_context(|| format!("DeepSeek-V4 expert {e}: w1"))?;
            let up_proj = load_expert_proj(store, &format!("{ep}.w3"), gpu, qctx)
                .with_context(|| format!("DeepSeek-V4 expert {e}: w3"))?;
            let down_proj = load_expert_proj(store, &format!("{ep}.w2"), gpu, qctx)
                .with_context(|| format!("DeepSeek-V4 expert {e}: w2"))?;
            experts.push(ExpertWeight {
                gate_proj,
                up_proj,
                down_proj,
            });
        } else {
            experts.push(ExpertWeight::null());
        }
    }
    // 2026-09-25: Every rank loads the whole shared expert; it is not EP-sharded.
    // With EP the decode MoE adds it once, after the routed all-reduce.
    let sep = format!("{p}.ffn.shared_experts");
    let shared_expert = ExpertWeight {
        gate_proj: load_expert_proj(store, &format!("{sep}.w1"), gpu, qctx)
            .with_context(|| "DeepSeek-V4 shared expert: w1")?,
        up_proj: load_expert_proj(store, &format!("{sep}.w3"), gpu, qctx)
            .with_context(|| "DeepSeek-V4 shared expert: w3")?,
        down_proj: load_expert_proj(store, &format!("{sep}.w2"), gpu, qctx)
            .with_context(|| "DeepSeek-V4 shared expert: w2")?,
    };

    // 2026-09-25: The shared-expert gate is optional. Found, the MoE adds
    // `sigmoid(gate · x) * shared`; absent, it stays NULL and the MoE adds the
    // shared expert ungated.
    let shared_expert_gate = match store
        .get(&format!("{p}.ffn.shared_expert_gate.weight"))
        .or_else(|_| store.get(&format!("{p}.mlp.shared_expert_gate.weight")))
    {
        Ok(t) => DenseWeight { weight: t.ptr },
        Err(_) => DenseWeight {
            weight: DevicePtr::NULL,
        },
    };

    let correction_bias = load_correction_bias(store, &lp, config.num_experts, gpu)?;

    let moe_weights = MoeWeights {
        gate,
        shared_expert,
        shared_expert_gate,
        experts,
        router_pre_norm: None,
        correction_bias,
    };
    // 2026-09-25: Layers below `num_hash_layers` pick experts from the static
    // `tid2eid` table (`[vocab_size, top_k]` i64, row = token id) instead of
    // top-k over the gate scores; the gate still supplies the sqrtsoftplus
    // weights. `Some(table)` is what marks a hash-routed layer.
    let tid2eid_dev = if layer_idx < config.num_hash_layers {
        let t = store
            .get(&format!("{lp}.ffn.gate.tid2eid"))
            .with_context(|| {
                format!(
                    "DeepSeek-V4 hash layer {layer_idx}: missing ffn.gate.tid2eid \
                 (num_hash_layers={})",
                    config.num_hash_layers
                )
            })?;
        let expected = config.vocab_size * config.num_experts_per_tok;
        anyhow::ensure!(
            t.num_elements() == expected,
            "DeepSeek-V4 tid2eid layer {layer_idx}: {} elements != vocab_size*top_k ({})",
            t.num_elements(),
            expected
        );
        Some(t.ptr)
    } else {
        None
    };
    let mut moe = MoeLayer::new_with_hash(
        moe_weights,
        config.num_experts,
        gate_nvfp4,
        tid2eid_dev,
        gpu,
        config,
    )?;
    // 2026-09-25: The routed format selects the NVFP4 or the E8M0 MoE kernels.
    moe.experts_scale_kind = detect_routed_scale_kind(store, p, config, force_all_experts);
    // 2026-09-25: Tagged separately: with E8M0 routed experts the fused decode
    // kernel requires an NVFP4 shared expert and panics on any other tag.
    moe.shared_experts_scale_kind = detect_shared_scale_kind(store, p);

    let wkv_a_rope = if let Ok(rope_w) = store.get(&format!("{lp}.attn.wkv_rope.weight")) {
        DenseWeight { weight: rope_w.ptr }
    } else {
        let rope_bytes = config.qk_rope_head_dim * h * 2;
        let rope_ptr = gpu.alloc(rope_bytes)?;
        gpu.memset(rope_ptr, 0, rope_bytes)?;
        DenseWeight { weight: rope_ptr }
    };

    // 2026-09-25: `compress_ratios[layer_idx]` 0 or missing: no compressor. Below
    // 128: CSA (`proj_dim = 2 * head_dim`, plus a previous-window buffer for the
    // overlap). 128 and above: HCA (`proj_dim = head_dim`).
    let compressor = {
        let ratio = config.compress_ratios.get(layer_idx).copied().unwrap_or(0);
        if ratio > 0 {
            let cp = format!("{lp}.attn.compressor");
            let is_csa = ratio < 128;
            let hd = config.head_dim;
            let proj_dim = if is_csa { 2 * hd } else { hd };
            let wkv = dense(store, &format!("{cp}.wkv.weight"))?;
            let wgate = dense(store, &format!("{cp}.wgate.weight"))?;
            // 2026-09-25: Loaded as stored, with no offset: the compressor norm runs
            // `rms_norm_w_k`, which is `rms_norm_vanilla` for V4.
            let norm = dense_auto(store, &format!("{cp}.norm.weight"), gpu)?;
            let ape = super::csa_ape::load_ape_f32(store, &format!("{cp}.ape"), gpu)?;
            // 2026-09-25: One byte per element, `max_position_embeddings / ratio`
            // blocks rounded up: sized from the model's maximum positions, not the
            // runtime maximum sequence length. A block is `qk_nope_head_dim +
            // qk_rope_head_dim` wide, the width the prefill builds comp_k at.
            let hd_mla = config.qk_nope_head_dim + config.qk_rope_head_dim;
            let pool_blocks = config.max_position_embeddings.div_ceil(ratio);
            let pool = gpu.alloc(pool_blocks * hd_mla)?;
            gpu.memset(pool, 0, pool_blocks * hd_mla)?;
            let h = config.hidden_size;
            let ring = gpu.alloc(ratio * h * 2)?;
            let (prev_win, stage) = if is_csa {
                (gpu.alloc(ratio * h * 2)?, gpu.alloc(2 * ratio * h * 2)?)
            } else {
                (
                    metrale_gpu_runtime::gpu::DevicePtr::NULL,
                    metrale_gpu_runtime::gpu::DevicePtr::NULL,
                )
            };
            Some(CompressorWeights {
                wkv,
                wgate,
                norm,
                ape,
                ratio,
                proj_dim,
                is_csa,
                pool,
                pool_blocks,
                ring,
                prev_win,
                stage,
            })
        } else {
            None
        }
    };

    let attn_sink =
        super::attn_sink::load_attn_sink_f32(store, &format!("{lp}.attn.attn_sink"), gpu)?;

    // 2026-09-25: FP8 copies of the MLA projections for the decode GEMVs, which
    // read them with `w8a16_gemv` at one byte per weight instead of the BF16
    // copy's two. `None` when the tensor is absent or not a 2-D FP8E4M3 weight.
    let load_fp8_mla = |suffix: &str| {
        metrale_model_layers::weight_map::load_fp8_block_scaled_as_fp8weight(
            store,
            &format!("{lp}.attn.{suffix}"),
            gpu,
        )
        .ok()
    };
    let wq_a_fp8 = load_fp8_mla("wq_a");
    let wq_b_fp8 = load_fp8_mla("wq_b");
    let wo_b_fp8 = load_fp8_mla("wo_b");
    let wo_a_fp8 = load_fp8_mla("wo_a");
    let wkv_a_fp8 = load_fp8_mla("wkv");

    let mla = MlaWeights {
        wq_a,
        wq_a_nvfp4,
        wq_a_fp8,
        wq_b,
        wq_b_nvfp4,
        wq_b_fp8,
        q_a_norm,
        wkv_a,
        wkv_a_nvfp4,
        wkv_a_fp8,
        wkv_b,
        kv_a_norm,
        wkv_a_rope,
        wkv_a_merged: DenseWeight {
            weight: wkv_a.weight,
        },
        wo: o_dense,
        wo_nvfp4: o_nvfp4,
        wo_a,
        wo_a_nvfp4: None,
        wo_a_fp8,
        wo_b: o_dense,
        wo_b_nvfp4: None,
        wo_b_fp8,
        w_uk_t,
        w_uv,
        wq_b_rope,
        w_qk_absorbed,
        w_uk_block_diag,
        w_uv_block_diag,
        yarn_inv_freq,
        main_inv_freq: super::compute::main_inv_freq(config, gpu)?,
        q_lora_rank: config.q_lora_rank,
        kv_lora_rank: config.kv_lora_rank,
        o_lora_rank: config.o_lora_rank,
        nope: config.qk_nope_head_dim,
        rope: config.qk_rope_head_dim,
        v_dim: config.v_head_dim,
        compressor,
        attn_sink,
    };

    let attn = AttentionWeights {
        q_proj: DenseWeight {
            weight: DevicePtr::NULL,
        },
        k_proj: DenseWeight {
            weight: DevicePtr::NULL,
        },
        v_proj: DenseWeight {
            weight: DevicePtr::NULL,
        },
        o_proj: QuantizedWeight::null(),
        q_norm: DenseWeight {
            weight: DevicePtr::NULL,
        },
        k_norm: DenseWeight {
            weight: DevicePtr::NULL,
        },
        q_norm_full: None,
        k_norm_full: None,
        k_scale: 1.0,
        v_scale: 1.0,
    };
    let mut layer = Qwen3AttentionLayer::new_ungated(
        input_norm,
        attn,
        post_attn_norm,
        FfnComponent::Moe(moe),
        layer_idx,
        None,
        None,
        None,
        gpu,
        kv_dtype,
        0,
        config,
    )?;
    layer.set_mla_weights(mla);

    if config.hc_mult > 0 {
        let attn = load_hc_site(store, &lp, "attn", config, gpu)?;
        let ffn = load_hc_site(store, &lp, "ffn", config, gpu)?;
        layer.set_hc_weights(HcWeights {
            attn,
            ffn,
            head: hc_head.clone(),
            hc_mult: config.hc_mult,
            sinkhorn_iters: config.hc_sinkhorn_iters,
            hc_eps: config.hc_eps,
            // 2026-09-25: Every V4 layer is an attention layer, so `layer_idx` is
            // the model layer index.
            is_first_model_layer: layer_idx == 0,
            is_last_model_layer: layer_idx + 1 == config.num_hidden_layers,
        });
    }

    anyhow::ensure!(
        layer.hc.is_some(),
        "DeepSeek-V4 requires the mHC highway (hc_mult > 0, got {}): without it the fused \
         residual+norm kernels would apply the offset-from-1 convention to exactly-loaded \
         norm weights",
        config.hc_mult
    );

    Ok(Box::new(layer))
}
