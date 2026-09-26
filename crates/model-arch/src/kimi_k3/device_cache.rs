// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `DeviceHybridCache`: device copies of every layer's Kimi K3 mixer state, built from a host [`HybridCache`].
//!
//! Owner: model-arch, Kimi K3.
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_weights::kimi_k3_host::{HybridCache, KdaConfig, LayerCache};

use super::kda_cuda::KdaDeviceState;

/// 2026-09-25: One layer's device state: KDA state for a KDA layer, or MLA
/// `k`/`v` buffers sized for `cap` tokens for an MLA layer.
pub struct DeviceLayerCache {
    pub kda: Option<KdaDeviceState>,
    pub mla_k: Option<DevicePtr>,
    pub mla_v: Option<DevicePtr>,
    pub seq_len: usize,
    pub cap: usize,
}

pub struct DeviceHybridCache {
    pub layers: Vec<DeviceLayerCache>,
}

impl DeviceHybridCache {
    /// 2026-09-25: Allocates each layer's device state and uploads the host
    /// state into it. An MLA layer's `cap` is `mla_cap`, raised to the host
    /// history length.
    pub fn from_host(
        gpu: &dyn GpuBackend,
        host: &HybridCache,
        kda_cfg: &KdaConfig,
        mla_cap: usize,
        mla_k_row: usize,
        mla_v_row: usize,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(host.layers.len());
        for slot in &host.layers {
            layers.push(match slot {
                LayerCache::Kda(s) => DeviceLayerCache {
                    kda: Some(KdaDeviceState::alloc_and_upload(gpu, s)?),
                    mla_k: None,
                    mla_v: None,
                    seq_len: 0,
                    cap: 0,
                },
                LayerCache::Mla(kv) => {
                    let cap = mla_cap.max(kv.seq_len.max(1));
                    let k = gpu.alloc((cap * mla_k_row * 4).max(1))?;
                    let v = gpu.alloc((cap * mla_v_row * 4).max(1))?;
                    if !kv.k.is_empty() {
                        let kb: Vec<u8> = kv.k.iter().flat_map(|x| x.to_le_bytes()).collect();
                        gpu.copy_h2d(&kb, k)?;
                    }
                    if !kv.v.is_empty() {
                        let vb: Vec<u8> = kv.v.iter().flat_map(|x| x.to_le_bytes()).collect();
                        gpu.copy_h2d(&vb, v)?;
                    }
                    DeviceLayerCache {
                        kda: None,
                        mla_k: Some(k),
                        mla_v: Some(v),
                        seq_len: kv.seq_len,
                        cap,
                    }
                }
            });
        }
        let _ = kda_cfg;
        Ok(Self { layers })
    }

    pub fn kda(&self, layer: usize) -> Option<&KdaDeviceState> {
        self.layers.get(layer).and_then(|l| l.kda.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use metrale_config::parse_config;
    use metrale_gpu_runtime::gpu::mock::MockGpuBackend;
    use metrale_model_weights::kimi_k3_host::{K3Graph, MixerKind};

    #[test]
    fn twin_device_cache_kda_slots_are_resident() {
        const TWIN: &str = include_str!("../../../../docs/k3/fixtures/Kimi-K3-0.40B-config.json");
        let c = parse_config(TWIN).unwrap();
        let g = K3Graph::from_config(&c);
        let kda = KdaConfig::twin_0_40b();
        let host = HybridCache::from_graph(&g, &kda);
        let gpu = MockGpuBackend::new();
        let before = gpu.d2h_blocking_count();
        let dev = DeviceHybridCache::from_host(&gpu, &host, &kda, 16, 8 * 96, 8 * 64).unwrap();
        assert_eq!(gpu.d2h_blocking_count(), before, "seed is H2D, not D2H");
        for i in [0, 1, 2, 4, 5, 6] {
            assert!(dev.kda(i).is_some(), "KDA layer {i}");
            assert!(matches!(g.layers[i].mixer, MixerKind::Kda));
        }
        for i in [3, 7] {
            assert!(dev.layers[i].mla_k.is_some());
            assert!(dev.kda(i).is_none());
        }
    }
}
