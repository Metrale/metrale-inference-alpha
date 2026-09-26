// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;

use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_gpu_runtime::gpu::mock::MockGpuBackend;
use metrale_model_weights::weights::{WeightDtype, WeightTensor};

use super::*;

fn tensor(ptr: u64) -> WeightTensor {
    WeightTensor {
        ptr: DevicePtr(ptr),
        shape: vec![4],
        dtype: WeightDtype::BF16,
    }
}

fn config() -> ModelConfig {
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.weight_prefix = "backbone".to_string();
    config
}

#[test]
fn nemotron_layout_routes_embedding_norm_and_tied_head() {
    let store = WeightStore::from_map(HashMap::from([
        ("backbone.embeddings.weight".to_string(), tensor(11)),
        ("backbone.norm_f.weight".to_string(), tensor(12)),
    ]));
    let gpu = MockGpuBackend::new();
    let loader = NemotronHWeightLoader;
    let config = config();

    assert_eq!(
        loader.load_embedding(&store, &config, &gpu).unwrap().weight,
        DevicePtr(11)
    );
    assert_eq!(
        loader
            .load_final_norm(&store, &config, &gpu)
            .unwrap()
            .weight,
        DevicePtr(12)
    );
    assert_eq!(
        loader.load_lm_head(&store, &config, &gpu).unwrap().weight,
        DevicePtr(11)
    );
    assert!(loader.supports_tp());
    assert!(
        loader
            .load_mtp_weights(&store, &config, &gpu)
            .unwrap()
            .is_none()
    );
}

#[test]
fn explicit_lm_head_takes_precedence_over_tied_embedding() {
    let store = WeightStore::from_map(HashMap::from([
        ("backbone.embeddings.weight".to_string(), tensor(11)),
        ("lm_head.weight".to_string(), tensor(13)),
    ]));

    let actual = NemotronHWeightLoader
        .load_lm_head(&store, &config(), &MockGpuBackend::new())
        .unwrap();
    assert_eq!(actual.weight, DevicePtr(13));
}
