// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The Kimi K3 `ModelWeightLoader`: binds BF16 or FP32 checkpoints, and packed
//! MXFP4 experts only when `K3_ALLOW_MXFP4=1`.
//!
//! Owner: model-engine Kimi K3 loader.
//! Invariants:
//! - Every `load_*` method except `load_mtp_weights` (which returns `Ok(None)`) calls
//!   `refuse_mxfp4` before it binds anything.

use anyhow::{Result, bail};
use metrale_cache::kv_cache::KvCacheDtype;
use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_model_weights::weights::WeightStore;

use metrale_model_arch::weight_loader::ModelWeightLoader;
use metrale_model_layers::layer::TransformerLayer;
use metrale_model_layers::weight_map::DenseWeight;

mod bf16;
mod classes;
mod dry_run;
mod mxfp4;
mod tp;

pub use dry_run::{KimiK3DryRun, dry_run_index_json, dry_run_weight_map};

pub struct KimiK3WeightLoader;

impl KimiK3WeightLoader {
    pub fn dry_run(index_json: &str, language_model_only: bool) -> Result<KimiK3DryRun> {
        dry_run::dry_run_index_json(index_json, language_model_only)
    }
}

/// 2026-09-25: Fail when a tensor name in `store` contains `weight_packed` and `K3_ALLOW_MXFP4`
/// is not `1`.
pub fn refuse_mxfp4(store: &WeightStore) -> Result<()> {
    if store.names().any(|n| n.contains("weight_packed"))
        && !metrale_model_weights::kimi_k3_host::mxfp4::allow_mxfp4()
    {
        bail!("S5 MXFP4 not this slice");
    }
    Ok(())
}

impl ModelWeightLoader for KimiK3WeightLoader {
    fn supports_tp(&self) -> bool {
        // 2026-09-25: The split of each tensor is `tensor_plan` (see `tp.rs`). `lm_head` is bound
        // whole on every rank (`bf16::load_lm_head`).
        true
    }

    fn binds_vision_encoder(&self) -> bool {
        false
    }

    fn load_layers(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        _layer_kv_dtypes: &[KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        refuse_mxfp4(store)?;
        bf16::load_layers(store, config, gpu)
    }

    fn load_embedding(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        refuse_mxfp4(store)?;
        bf16::load_embedding(store, config, gpu)
    }

    fn load_final_norm(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        refuse_mxfp4(store)?;
        bf16::load_final_norm(store, config, gpu)
    }

    fn load_lm_head(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        refuse_mxfp4(store)?;
        bf16::load_lm_head(store, config, gpu)
    }

    fn load_mtp_weights(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<Option<metrale_model_layers::weight_map::MtpWeights>> {
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::factory::loader_for_config;
    use metrale_config::parse_config;
    use metrale_gpu_runtime::gpu::GpuBackend;
    use metrale_model_weights::weights::{WeightDtype, WeightTensor};
    use std::collections::HashMap;
    use std::process::Command;

    #[test]
    fn kimi_k3_supports_tp() {
        assert!(KimiK3WeightLoader.supports_tp());
    }

    #[test]
    fn kimi_k3_does_not_probe_nvfp4_tgemm() {
        assert!(
            !metrale_model_layers::layers::tgemm_probe_ok("kimi_k3"),
            "BF16/FP32 twin must not look up w4a16_gemm_t_p3"
        );
        assert!(metrale_model_layers::layers::tgemm_probe_ok("qwen3_5_moe"));
    }

    #[test]
    fn loader_for_config_kimi_k3() {
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        for ty in ["kimi_k3", "kimi_linear", "Kimi-K3"] {
            config.model_type = ty.to_string();
            let loader = loader_for_config(&config).expect(ty);
            let store = WeightStore::empty();
            let gpu = metrale_gpu_runtime::gpu::mock::MockGpuBackend::new();
            let err = match loader.load_layers(&store, &config, &gpu, &[]) {
                Ok(_) => panic!("{ty}: empty store must not bind"),
                Err(e) => e.to_string(),
            };
            assert!(
                err.contains("not found") || err.contains("S5 MXFP4"),
                "{ty}: {err}"
            );
            assert!(!err.contains("K3-WIP"), "{ty}: stale WIP bail: {err}");
        }
    }

    #[test]
    fn load_bails_on_mxfp4_packed() {
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        config.model_type = "kimi_k3".into();
        let gpu = metrale_gpu_runtime::gpu::mock::MockGpuBackend::new();
        let ptr = gpu.alloc(4).unwrap();
        let store = WeightStore::from_map(HashMap::from([(
            "language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight_packed".to_string(),
            WeightTensor {
                ptr,
                shape: vec![2, 2],
                dtype: WeightDtype::UInt8,
            },
        )]));
        let loader = KimiK3WeightLoader;
        let err = match loader.load_layers(&store, &config, &gpu, &[]) {
            Ok(_) => panic!("packed MXFP4 must not bind layers"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("S5 MXFP4 not this slice"), "{err}");
        let err = match loader.load_embedding(&store, &config, &gpu) {
            Ok(_) => panic!("packed MXFP4 must not bind embed"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("S5 MXFP4 not this slice"), "{err}");
    }

    #[test]
    fn packed_tp_refuses_before_weight_binding() {
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        config.tp_world_size = 8;
        let gpu = metrale_gpu_runtime::gpu::mock::MockGpuBackend::new();
        let store = WeightStore::from_map(HashMap::from([(
            "language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight_packed".into(),
            WeightTensor {
                ptr: gpu.alloc(16).unwrap(),
                shape: vec![1, 16],
                dtype: WeightDtype::UInt8,
            },
        )]));
        let err = match bf16::load_layers(&store, &config, &gpu) {
            Ok(_) => panic!("packed TP must refuse an unmarked store"),
            Err(err) => err.to_string(),
        };
        assert!(err.contains("does not slice packed MXFP4"), "{err}");
    }

    #[test]
    fn load_allow_mxfp4_lands_packed_experts() {
        let this = format!(
            "{}::load_allow_mxfp4_lands_packed_experts",
            module_path!()
                .split_once("::")
                .map_or(module_path!(), |(_, p)| p)
        );
        const MARKER: &str = "K3_MXFP4_GPU_ALLOW_CHILD";
        if std::env::var_os(MARKER).is_some() {
            const TWIN: &str = include_str!("../../../docs/k3/fixtures/Kimi-K3-0.40B-config.json");
            let mut config = parse_config(TWIN).expect("0.40B twin");
            config.tp_world_size = 1;
            let gpu = metrale_gpu_runtime::gpu::mock::MockGpuBackend::new();
            let graph = metrale_model_weights::kimi_k3_host::K3Graph::from_config(&config);
            let mut map = HashMap::new();
            let mut put = |name: String, shape: Vec<usize>, dtype: WeightDtype| {
                let n = shape.iter().product::<usize>().max(1);
                let ptr = gpu.alloc(n.max(4)).unwrap();
                map.insert(name, WeightTensor { ptr, shape, dtype });
            };
            put(
                bf16::text_key(&config, "model.embed_tokens.weight"),
                vec![2],
                WeightDtype::BF16,
            );
            put(
                bf16::text_key(&config, "model.norm.weight"),
                vec![2],
                WeightDtype::BF16,
            );
            put(
                bf16::text_key(&config, "model.output_attn_res_proj.weight"),
                vec![2],
                WeightDtype::BF16,
            );
            put(
                bf16::text_key(&config, "model.output_attn_res_norm.weight"),
                vec![2],
                WeightDtype::BF16,
            );
            put(
                bf16::text_key(&config, "lm_head.weight"),
                vec![2],
                WeightDtype::BF16,
            );
            let packed_w1 = bf16::text_key(
                &config,
                "model.layers.1.block_sparse_moe.experts.0.w1.weight",
            );
            for spec in &graph.layers {
                for k in bf16::layer_keys(
                    &config,
                    spec.index,
                    spec.mixer,
                    spec.mlp,
                    config.num_experts,
                ) {
                    if k == packed_w1 {
                        continue;
                    }
                    put(k, vec![2], WeightDtype::BF16);
                }
            }
            let prefix = packed_w1.trim_end_matches(".weight");
            put(
                format!("{prefix}.weight_packed"),
                vec![config.moe_intermediate_size, config.moe_latent_size / 2],
                WeightDtype::UInt8,
            );
            put(
                format!("{prefix}.weight_scale"),
                vec![config.moe_intermediate_size, config.moe_latent_size / 32],
                WeightDtype::UInt8,
            );
            let store = WeightStore::from_map(map);
            refuse_mxfp4(&store).expect("K3_ALLOW_MXFP4=1 must skip S5 refuse");
            let loader = KimiK3WeightLoader;
            let layers = loader
                .load_layers(&store, &config, &gpu, &[])
                .expect("packed expert must land on DSV4 E8M0");
            assert_eq!(layers.len(), 8);
            return;
        }
        let output = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", this.as_str()])
            .env(MARKER, "1")
            .env("K3_ALLOW_MXFP4", "1")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "ALLOW child failed:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            String::from_utf8_lossy(&output.stdout).contains("1 passed"),
            "the child must run exactly this test: {}",
            String::from_utf8_lossy(&output.stdout)
        );
    }

    #[test]
    fn load_bf16_twin_binds_layers() {
        const TWIN: &str = include_str!("../../../docs/k3/fixtures/Kimi-K3-0.40B-config.json");
        let config = parse_config(TWIN).expect("0.40B twin");
        let gpu = metrale_gpu_runtime::gpu::mock::MockGpuBackend::new();
        let graph = metrale_model_weights::kimi_k3_host::K3Graph::from_config(&config);
        let mut map = HashMap::new();
        let mut put = |name: String| {
            let ptr = gpu.alloc(4).unwrap();
            map.insert(
                name,
                WeightTensor {
                    ptr,
                    shape: vec![2],
                    dtype: WeightDtype::BF16,
                },
            );
        };
        put(bf16::text_key(&config, "model.embed_tokens.weight"));
        put(bf16::text_key(&config, "model.norm.weight"));
        put(bf16::text_key(&config, "model.output_attn_res_proj.weight"));
        put(bf16::text_key(&config, "model.output_attn_res_norm.weight"));
        put(bf16::text_key(&config, "lm_head.weight"));
        for spec in &graph.layers {
            for k in bf16::layer_keys(
                &config,
                spec.index,
                spec.mixer,
                spec.mlp,
                config.num_experts,
            ) {
                put(k);
            }
        }
        let store = WeightStore::from_map(map);
        let loader = KimiK3WeightLoader;
        let layers = loader.load_layers(&store, &config, &gpu, &[]).unwrap();
        assert_eq!(layers.len(), 8);
        for layer in &layers {
            assert!(layer.decode_graph_unsupported());
            assert!(layer.decode_multi_seq_unsupported());
            assert!(layer.decode_verify_multi_unsupported());
            assert!(layer.decode_rollback_unsupported());
            assert!(layer.has_aux_state());
        }
        loader.load_embedding(&store, &config, &gpu).unwrap();
        loader.load_final_norm(&store, &config, &gpu).unwrap();
        loader.load_lm_head(&store, &config, &gpu).unwrap();
    }
}
