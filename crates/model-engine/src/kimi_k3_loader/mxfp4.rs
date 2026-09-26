// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Binds Kimi K3 packed experts (`weight_packed` + `weight_scale`) as native MXFP4
//! E8M0 weights and checks their rank-local shapes.
//!
//! The GEMM is `moe_w4a16_grouped_gemm_ptrtable_e8m0`, built for K3 from the DeepSeek V4 source
//! (`[sources] use` in `kernels/gb10/kimi-k3/mxfp4/KERNEL.toml`). The loader refuses packed
//! experts unless `K3_ALLOW_MXFP4=1` (`refuse_mxfp4`).
//!
//! Owner: model-engine Kimi K3 loader.
//! Invariants: none beyond the types.

use anyhow::{Result, ensure};
use metrale_config::ModelConfig;
use metrale_core::mxfp4_e8m0::GROUP_SIZE;
use metrale_model_weights::kimi_k3_host::MixerKind;
use metrale_model_weights::kimi_k3_host::tp::{plan_tensor_bytes, tensor_plan};
use metrale_model_weights::weights::WeightDtype;
use metrale_model_weights::weights::WeightStore;

use metrale_model_layers::weight_map::{QuantizedWeight, quantized_mxfp4_e8m0_pair};

/// 2026-09-25: Bind `{prefix}.weight_packed` and `{prefix}.weight_scale` without a transcode,
/// through `quantized_mxfp4_e8m0_pair`, the lander DeepSeek V4 also uses.
pub(super) fn quantized_k3_mxfp4_e8m0(
    store: &WeightStore,
    prefix: &str,
) -> Result<QuantizedWeight> {
    quantized_mxfp4_e8m0_pair(
        store,
        &format!("{prefix}.weight_packed"),
        &format!("{prefix}.weight_scale"),
    )
}

/// 2026-09-25: Check that the packed weight is U8, the scale is U8 or E8M0 bytes, and both shapes
/// equal this rank's local shapes from the TP plan. A multi-rank store must be pre-partitioned.
pub(super) fn validate_packed_partition(
    store: &WeightStore,
    prefix: &str,
    config: &ModelConfig,
) -> Result<()> {
    let marked = super::tp::is_prepartitioned(store, config)?;
    ensure!(
        config.tp_world_size <= 1 || marked,
        "K3 TP does not slice packed MXFP4 after upload; use the rank-aware checkpoint loader"
    );
    let weight_key = format!("{prefix}.weight_packed");
    let scale_key = format!("{prefix}.weight_scale");
    let (_, n, k) = tensor_plan(&weight_key, MixerKind::Kda, config);
    let wp = plan_tensor_bytes(&weight_key, &[n, k / 2], 1, MixerKind::Kda, config)?;
    let sp = plan_tensor_bytes(&scale_key, &[n, k / GROUP_SIZE], 1, MixerKind::Kda, config)?;
    let weight = store.get(&weight_key)?;
    let scale = store.get(&scale_key)?;
    ensure!(
        weight.dtype == WeightDtype::UInt8,
        "{weight_key}: expected packed U8, got {:?}",
        weight.dtype
    );
    ensure!(
        matches!(scale.dtype, WeightDtype::UInt8 | WeightDtype::FP8E8M0),
        "{scale_key}: expected E8M0 byte storage, got {:?}",
        scale.dtype
    );
    ensure!(
        weight.shape == wp.local_shape && scale.shape == sp.local_shape,
        "{prefix}: packed/scales shape {:?}/{:?} does not match TP-local {:?}/{:?}",
        weight.shape,
        scale.shape,
        wp.local_shape,
        sp.local_shape
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use metrale_gpu_runtime::gpu::GpuBackend;
    use metrale_gpu_runtime::gpu::mock::MockGpuBackend;
    use metrale_model_weights::weights::{WeightDtype, WeightTensor};
    use std::collections::HashMap;

    #[test]
    fn k3_packed_names_land_on_dsv4_e8m0_pair() {
        let gpu = MockGpuBackend::new();
        let packed = gpu.alloc(16).unwrap();
        let scale = gpu.alloc(1).unwrap();
        let store = WeightStore::from_map(HashMap::from([
            (
                "experts.0.w1.weight_packed".to_string(),
                WeightTensor {
                    ptr: packed,
                    shape: vec![1, 16],
                    dtype: WeightDtype::UInt8,
                },
            ),
            (
                "experts.0.w1.weight_scale".to_string(),
                WeightTensor {
                    ptr: scale,
                    shape: vec![1],
                    dtype: WeightDtype::UInt8,
                },
            ),
        ]));
        let qw = quantized_k3_mxfp4_e8m0(&store, "experts.0.w1").unwrap();
        assert_eq!(qw.weight, packed);
        assert_eq!(qw.weight_scale, scale);
        assert_eq!(qw.weight_scale_2, 1.0);
    }
}

#[cfg(test)]
#[path = "mxfp4_tests.rs"]
mod partition_tests;
