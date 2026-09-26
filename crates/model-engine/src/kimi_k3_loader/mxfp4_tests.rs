// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests that a store from the rank-aware K3 loader binds packed experts and dense
//! shards for this rank without a second device allocation.
//!
//! Owner: model-engine Kimi K3 loader.
//! Invariants: none beyond the types.
use super::*;
use metrale_config::parse_config;
use metrale_core::scope::ModelResource;
use metrale_gpu_runtime::gpu::mock::MockGpuBackend;
use metrale_model_weights::weights::{K3SafetensorsLoader, WeightLoader};

#[test]
fn packed_loader_marker_binds_local_weights_without_second_allocation() {
    for world in [2, 4, 8] {
        let mut config = parse_config(include_str!(
            "../../../../docs/k3/fixtures/Kimi-K3-0.40B-config.json"
        ))
        .unwrap();
        config.weight_prefix.clear();
        config.tp_rank = world - 1;
        config.tp_world_size = world;
        config.num_attention_heads /= world;
        config.num_key_value_heads /= world;
        config.linear_num_key_heads /= world;
        config.linear_num_value_heads /= world;
        config.hidden_size = 8;
        config.vocab_size = 8;
        config.intermediate_size = 8;
        config.shared_expert_intermediate_size = 8;
        config.linear_num_key_heads = 1;
        config.linear_num_value_heads = 1;
        config.linear_key_head_dim = 8;
        config.linear_value_head_dim = 8;
        config.num_hidden_layers = 2;
        config.layer_types.truncate(2);
        config.num_experts = 1;
        config.moe_intermediate_size = 256;
        config.moe_latent_size = 64;
        let dir = std::env::temp_dir().join(format!(
            "metrale-k3-model-packed-{}-{world}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        for (projection, n, k) in [("w1", 256, 64), ("w2", 64, 256), ("w3", 256, 64)] {
            for (suffix, divisor, value) in
                [("weight_packed", 2, 0x22u8), ("weight_scale", 32, 127)]
            {
                let name =
                    format!("model.layers.1.block_sparse_moe.experts.0.{projection}.{suffix}");
                let start = data.len();
                data.resize(start + n * k / divisor, value);
                header.insert(name, serde_json::json!({"dtype":"U8", "shape":[n,k/divisor], "data_offsets":[start,data.len()]}));
            }
        }
        for name in metrale_model_weights::kimi_k3_host::weights::required_names(&config) {
            if name.contains(".block_sparse_moe.experts.") {
                continue;
            }
            let (axis, n, k) = tensor_plan(&name, MixerKind::Kda, &config);
            let shape = if axis == metrale_model_weights::kimi_k3_host::tp::TpAxis::Replicated {
                metrale_model_weights::kimi_k3_host::weights_geometry::expected_replicated_shape(
                    &name, &config,
                )
                .unwrap()
            } else {
                vec![n, k]
            };
            let start = data.len();
            data.resize(start + shape.iter().product::<usize>() * 2, 0);
            header.insert(name, serde_json::json!({"dtype":"BF16", "shape":shape, "data_offsets":[start,data.len()]}));
        }
        let mut json = serde_json::to_vec(&header).unwrap();
        while !json.len().is_multiple_of(8) {
            json.push(b' ');
        }
        let mut file = (json.len() as u64).to_le_bytes().to_vec();
        file.extend(json);
        file.extend(data);
        std::fs::write(dir.join("model.safetensors"), file).unwrap();
        let gpu = MockGpuBackend::new();
        // 2026-09-25: A full packed tensor is 256 * 64 / 2 = 8192 bytes, so this cap fails any
        // upload of a whole tensor.
        gpu.set_max_allocation_bytes(8192 / world);
        let mut store = K3SafetensorsLoader::new(config.clone())
            .unwrap()
            .load(&dir, &gpu, 0)
            .unwrap();
        let allocations = gpu.alloc_count();
        for projection in ["w1", "w2", "w3"] {
            let prefix = format!("model.layers.1.block_sparse_moe.experts.0.{projection}");
            validate_packed_partition(&store, &prefix, &config).unwrap();
            let bound = quantized_k3_mxfp4_e8m0(&store, &prefix).unwrap();
            assert_eq!(
                gpu.read_alloc(bound.weight).unwrap(),
                vec![0x22; 8192 / world]
            );
            assert_eq!(
                gpu.read_alloc(bound.weight_scale).unwrap(),
                vec![127; 512 / world]
            );
            let mut wrong_rank = config.clone();
            wrong_rank.tp_rank = 0;
            assert!(validate_packed_partition(&store, &prefix, &wrong_rank).is_err());
        }
        let dense_name = "model.layers.0.self_attn.q_proj.weight";
        let original = store.get(dense_name).unwrap();
        let (dense, meta) = super::super::tp::load_sharded(
            &store,
            dense_name,
            MixerKind::Kda,
            metrale_model_weights::kimi_k3_host::MlpKind::Dense,
            &config,
            &gpu,
        )
        .unwrap();
        assert_eq!(dense.weight, original.ptr);
        assert_eq!(meta.numel, original.num_elements());
        assert_eq!(gpu.alloc_count(), allocations);
        store.release(&gpu).unwrap();
        assert_eq!(gpu.alloc_count(), 0);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
