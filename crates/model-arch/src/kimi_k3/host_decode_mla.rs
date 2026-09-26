// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests of `K3BoundLayer::decode_host` with CUDA MLA on a mock GPU: the second token uploads as many bytes as the first, a snapshot carries the device KV, and `serve_max_seq_len = 0` is refused.
//!
//! Owner: model-arch, Kimi K3.
//! Invariants: none beyond the types.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use half::bf16;
use metrale_config::ModelConfig;
use metrale_gpu_runtime::buffers::BufferArena;
use metrale_gpu_runtime::gpu::mock::MockGpuBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_weights::kimi_k3_host::{K3CpuModel, MixerKind};
use metrale_model_weights::weights::WeightDtype;
use parking_lot::Mutex;

use super::bound::{K3BoundLayer, K3HostShared};
use super::state::K3CpuFallbackState;
use metrale_model_layers::layer::{ForwardContext, MoeLoraRoute};
use metrale_model_layers::layers::ops::{DerivedWeights, GemmDispatch, ModelLevers, ModelStats};
use metrale_model_layers::weight_map::DenseWeight;

fn tiny_config(hidden: usize, eps: f32, theta: f32, inter: usize) -> ModelConfig {
    let mut c = ModelConfig::qwen3_next_80b_nvfp4();
    c.hidden_size = hidden;
    c.intermediate_size = inter;
    c.vocab_size = 32;
    c.num_experts = 1;
    c.num_hidden_layers = 8;
    c.rms_norm_eps = eps as f64;
    c.rope_theta = theta as f64;
    c.linear_num_key_heads = 1;
    c.linear_key_head_dim = 2;
    c.linear_num_value_heads = 1;
    c.linear_value_head_dim = 2;
    c.num_attention_heads = 1;
    c.head_dim = 4;
    c.serve_max_seq_len = 16;
    c
}

fn mla_decode_keep_state(
    tokens: usize,
) -> anyhow::Result<(
    usize,
    usize,
    metrale_model_weights::kimi_k3_host::LayerCache,
)> {
    let gpu = MockGpuBackend::new();
    let model = K3CpuModel::synthetic_tiny();
    let h = model.graph.hidden;
    let config = tiny_config(h, model.eps, model.rope_theta, model.dense_intermediate);
    let dummy = gpu.alloc((h * 2).max(1)).unwrap();
    let shared = Arc::new(K3HostShared {
        config: config.clone(),
        graph: model.graph.clone(),
        kda: model.kda,
        mla: model.mla,
        moe: model.moe,
        output_res_proj: DenseWeight { weight: dummy },
        output_res_norm: DenseWeight { weight: dummy },
        output_res_proj_meta: (WeightDtype::BF16, h),
        output_res_norm_meta: (WeightDtype::BF16, h),
        output_host: OnceLock::new(),
        kda_kernels: OnceLock::new(),
        mla_kernels: OnceLock::new(),
        moe_kernels: OnceLock::new(),
        attnres: Mutex::new(HashMap::new()),
    });
    let hidden = gpu.alloc(h * 2).unwrap();
    let raw: Vec<u8> = (0..h)
        .flat_map(|i| bf16::from_f32(0.1 * (i as f32 + 1.0)).to_le_bytes())
        .collect();
    gpu.copy_h2d(&raw, hidden).unwrap();
    let buffers = BufferArena::new(&config, 2, 16, 16, 2, &gpu).unwrap();
    let dispatch = GemmDispatch::defaults();
    let derived = DerivedWeights::new();
    let levers = ModelLevers::defaults();
    let stats = ModelStats::new();
    let ctx = ForwardContext {
        buffers: &buffers,
        hc_row_offset: 0,
        gpu: &gpu,
        config: &config,
        dispatch: &dispatch,
        derived: &derived,
        levers: &levers,
        stats: &stats,
        attn_metadata: None,
        profile: false,
        comm: None,
        graph_capture: false,
        decode_step: true,
        gdn_exact_replay: false,
        gdn_write_on_accept: false,
        token_ids: None,
        host_token_ids: None,
        routed_lora_layers: None,
        midchunk_capture: None,
        moe_lora_route: MoeLoraRoute::Fold,
    };
    let seed = K3BoundLayer {
        index: 0,
        spec: model.layers[0].spec,
        weights: Vec::new(),
        weight_meta: Vec::new(),
        mxfp4_experts: Vec::new(),
        host: {
            let host = OnceLock::new();
            let _ = host.set(model.layers[0].clone());
            host
        },
        shared: shared.clone(),
    };
    let mut seed_state =
        K3CpuFallbackState::new(metrale_model_weights::kimi_k3_host::LayerCache::Kda(
            metrale_model_weights::kimi_k3_host::KdaState::new(&model.kda),
        ));
    seed.decode_host(
        hidden,
        DevicePtr::NULL,
        &mut seed_state,
        0,
        &ctx,
        3,
        false,
        false,
    )?;
    let mla_idx = 3;
    assert_eq!(model.layers[mla_idx].spec.mixer, MixerKind::Mla);
    let layer = K3BoundLayer {
        index: mla_idx,
        spec: model.layers[mla_idx].spec,
        weights: Vec::new(),
        weight_meta: Vec::new(),
        mxfp4_experts: Vec::new(),
        host: {
            let host = OnceLock::new();
            let _ = host.set(model.layers[mla_idx].clone());
            host
        },
        shared: shared.clone(),
    };
    let mut state = K3CpuFallbackState::new(metrale_model_weights::kimi_k3_host::LayerCache::Mla(
        metrale_model_weights::kimi_k3_host::MlaKv::default(),
    ));
    let mut last_h2d = gpu.h2d_bytes();
    let mut token_h2d = Vec::new();
    for pos in 0..tokens {
        layer.decode_host(
            hidden,
            DevicePtr::NULL,
            &mut state,
            pos,
            &ctx,
            3,
            false,
            true,
        )?;
        let now = gpu.h2d_bytes();
        token_h2d.push(now - last_h2d);
        last_h2d = now;
    }
    let bytes = state.snapshot(&gpu, 3)?;
    let cache = metrale_model_weights::kimi_k3_host::LayerCache::from_bytes(&bytes)?;
    state.release(&gpu)?;
    Ok((token_h2d[0], token_h2d[tokens - 1], cache))
}

#[test]
fn serving_resident_mla_h2d_does_not_grow_and_snapshot_keeps_history() {
    let (t0, t1, cache) = mla_decode_keep_state(2).unwrap();
    assert_eq!(
        t1, t0,
        "serving CUDA MLA must not re-upload KV history (t0={t0} t1={t1})"
    );
    let metrale_model_weights::kimi_k3_host::LayerCache::Mla(kv) = cache else {
        panic!("expected MLA snapshot");
    };
    assert_eq!(
        kv.seq_len, 2,
        "snapshot must download authoritative device KV"
    );
}

#[test]
fn serving_cuda_mla_requires_serve_max_seq_len() {
    let gpu = MockGpuBackend::new();
    let model = K3CpuModel::synthetic_tiny();
    let h = model.graph.hidden;
    let mut config = tiny_config(h, model.eps, model.rope_theta, model.dense_intermediate);
    config.serve_max_seq_len = 0;
    let dummy = gpu.alloc((h * 2).max(1)).unwrap();
    let shared = Arc::new(K3HostShared {
        config: config.clone(),
        graph: model.graph.clone(),
        kda: model.kda,
        mla: model.mla,
        moe: model.moe,
        output_res_proj: DenseWeight { weight: dummy },
        output_res_norm: DenseWeight { weight: dummy },
        output_res_proj_meta: (WeightDtype::BF16, h),
        output_res_norm_meta: (WeightDtype::BF16, h),
        output_host: OnceLock::new(),
        kda_kernels: OnceLock::new(),
        mla_kernels: OnceLock::new(),
        moe_kernels: OnceLock::new(),
        attnres: Mutex::new(HashMap::new()),
    });
    let hidden = gpu.alloc(h * 2).unwrap();
    let buffers = BufferArena::new(&config, 2, 16, 16, 2, &gpu).unwrap();
    let dispatch = GemmDispatch::defaults();
    let derived = DerivedWeights::new();
    let levers = ModelLevers::defaults();
    let stats = ModelStats::new();
    let ctx = ForwardContext {
        buffers: &buffers,
        hc_row_offset: 0,
        gpu: &gpu,
        config: &config,
        dispatch: &dispatch,
        derived: &derived,
        levers: &levers,
        stats: &stats,
        attn_metadata: None,
        profile: false,
        comm: None,
        graph_capture: false,
        decode_step: true,
        gdn_exact_replay: false,
        gdn_write_on_accept: false,
        token_ids: None,
        host_token_ids: None,
        routed_lora_layers: None,
        midchunk_capture: None,
        moe_lora_route: MoeLoraRoute::Fold,
    };
    let layer = K3BoundLayer {
        index: 3,
        spec: model.layers[3].spec,
        weights: Vec::new(),
        weight_meta: Vec::new(),
        mxfp4_experts: Vec::new(),
        host: {
            let host = OnceLock::new();
            let _ = host.set(model.layers[3].clone());
            host
        },
        shared,
    };
    let mut state = K3CpuFallbackState::new(metrale_model_weights::kimi_k3_host::LayerCache::Mla(
        metrale_model_weights::kimi_k3_host::MlaKv::default(),
    ));
    let err = layer
        .decode_host(hidden, DevicePtr::NULL, &mut state, 0, &ctx, 3, false, true)
        .unwrap_err()
        .to_string();
    assert!(err.contains("serve_max_seq_len"), "{err}");
}
