// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `K3CpuFallbackState`, one sequence's Kimi K3 mixer state: the host [`LayerCache`] and the device copies the CUDA mixers use.
//!
//! Snapshots and restores use the [`LayerCache`] byte format.
//!
//! Owner: model-arch, Kimi K3.
//! Invariants:
//! - `restore` rejects a blob that does not parse, or whose mixer kind or
//!   geometry differs from the current state, before it changes anything.
//! - `release` drops both device copies even when freeing one fails.

use std::any::Any;

use anyhow::{Result, bail, ensure};
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_model_weights::kimi_k3_host::{LayerCache, MlaConfig};

use super::kda_cuda::KdaDeviceState;
use super::mla_cuda::MlaDeviceKv;

use metrale_model_layers::layer::LayerState;

/// 2026-09-25: Mixer state owned by one sequence; `K3BoundLayer::release_state`
/// frees its device copies.
pub struct K3CpuFallbackState {
    pub cache: LayerCache,
    /// 2026-09-25: Once present, this is the current KDA state and `cache` holds
    /// only what it was seeded from.
    pub device_kda: Option<KdaDeviceState>,
    /// 2026-09-25: Once present, this is the current MLA KV and `cache` holds only
    /// what it was seeded from.
    pub device_mla: Option<MlaDeviceKv>,
}

impl K3CpuFallbackState {
    pub fn new(cache: LayerCache) -> Self {
        Self {
            cache,
            device_kda: None,
            device_mla: None,
        }
    }

    pub fn ensure_device_kda(&mut self, gpu: &dyn GpuBackend) -> Result<()> {
        if self.device_kda.is_none()
            && let LayerCache::Kda(host) = &self.cache
        {
            self.device_kda = Some(KdaDeviceState::alloc_and_upload(gpu, host)?);
        }
        Ok(())
    }

    pub fn ensure_device_mla(
        &mut self,
        gpu: &dyn GpuBackend,
        cfg: &MlaConfig,
        cap: usize,
    ) -> Result<()> {
        if let Some(device) = &self.device_mla {
            return device.validate_cfg(cfg);
        }
        let LayerCache::Mla(host) = &self.cache else {
            bail!("K3 CUDA MLA: mixer state is not MLA KV");
        };
        let k_row = cfg.heads * cfg.qk_head_dim();
        let v_row = cfg.heads * cfg.v_head_dim;
        self.device_mla = Some(MlaDeviceKv::alloc_and_upload(gpu, host, cap, k_row, v_row)?);
        Ok(())
    }

    pub fn snapshot(&self, gpu: &dyn GpuBackend, stream: u64) -> Result<Vec<u8>> {
        let mut cache = self.cache.clone();
        if let (Some(device), LayerCache::Kda(host)) = (&self.device_kda, &mut cache) {
            gpu.synchronize(stream)?;
            device.download(gpu, host)?;
        }
        if let (Some(device), LayerCache::Mla(host)) = (&self.device_mla, &mut cache) {
            gpu.synchronize(stream)?;
            device.download(gpu, host)?;
        }
        Ok(cache.to_bytes())
    }

    pub fn restore(&mut self, gpu: &dyn GpuBackend, bytes: &[u8], stream: u64) -> Result<()> {
        let cache = LayerCache::from_bytes(bytes)?;
        match (&self.cache, &cache) {
            (LayerCache::Kda(current), LayerCache::Kda(restored)) => ensure!(
                current.conv.len() == restored.conv.len()
                    && current.recurrent.len() == restored.recurrent.len(),
                "K3 restore: KDA state geometry mismatch"
            ),
            (LayerCache::Mla(current), LayerCache::Mla(restored)) => {
                validate_mla_restore(current, restored, self.device_mla.as_ref())?;
            }
            _ => bail!("K3 restore: mixer variant mismatch"),
        }
        gpu.synchronize(stream)?;
        self.release(gpu)?;
        self.cache = cache;
        Ok(())
    }

    pub fn release(&mut self, gpu: &dyn GpuBackend) -> Result<()> {
        let kda = if let Some(device) = self.device_kda.take() {
            device.free(gpu)
        } else {
            Ok(())
        };
        let mla = if let Some(device) = self.device_mla.take() {
            device.free(gpu)
        } else {
            Ok(())
        };
        kda.and(mla)
    }
}

fn validate_mla_restore(
    current: &metrale_model_weights::kimi_k3_host::MlaKv,
    restored: &metrale_model_weights::kimi_k3_host::MlaKv,
    device: Option<&MlaDeviceKv>,
) -> Result<()> {
    if restored.seq_len == 0 {
        ensure!(
            restored.k.is_empty() && restored.v.is_empty(),
            "K3 restore: empty MLA seq with payload"
        );
        return Ok(());
    }
    ensure!(
        restored.k.len()
            == restored
                .seq_len
                .saturating_mul(restored.k.len() / restored.seq_len)
            && restored.v.len()
                == restored
                    .seq_len
                    .saturating_mul(restored.v.len() / restored.seq_len)
            && restored.k.len() / restored.seq_len > 0
            && restored.v.len() / restored.seq_len > 0,
        "K3 restore: MLA KV rank"
    );
    let k_row = restored.k.len() / restored.seq_len;
    let v_row = restored.v.len() / restored.seq_len;
    if current.seq_len > 0 {
        ensure!(
            current.k.len() / current.seq_len == k_row
                && current.v.len() / current.seq_len == v_row,
            "K3 restore: MLA KV geometry mismatch"
        );
    } else if let Some(device) = device {
        ensure!(
            device.k_row() == k_row && device.v_row() == v_row,
            "K3 restore: MLA KV geometry mismatch"
        );
    }
    Ok(())
}

impl LayerState for K3CpuFallbackState {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use metrale_gpu_runtime::gpu::GpuBackend;
    use metrale_gpu_runtime::gpu::mock::MockGpuBackend;
    use metrale_model_weights::kimi_k3_host::{KdaConfig, KdaState};

    #[test]
    fn resident_snapshot_restore_and_release_preserve_authoritative_state() {
        let gpu = MockGpuBackend::new();
        let mut state =
            K3CpuFallbackState::new(LayerCache::Kda(KdaState::new(&KdaConfig::twin_0_40b())));
        state.ensure_device_kda(&gpu).unwrap();
        state.ensure_device_kda(&gpu).unwrap();
        assert_eq!(gpu.alloc_count(), 2, "reuse resident state between tokens");
        assert!(state.restore(&gpu, b"invalid", 0).is_err());
        assert_eq!(
            gpu.alloc_count(),
            2,
            "invalid snapshot must leave state intact"
        );
        let device = state.device_kda.as_ref().unwrap();
        gpu.copy_h2d(&7.25f32.to_le_bytes(), device.recurrent)
            .unwrap();
        let bytes = state.snapshot(&gpu, 0).unwrap();
        let LayerCache::Kda(saved) = LayerCache::from_bytes(&bytes).unwrap() else {
            panic!("expected KDA");
        };
        assert_eq!(saved.recurrent[0], 7.25);
        state.restore(&gpu, &bytes, 0).unwrap();
        assert!(state.device_kda.is_none());
        assert_eq!(gpu.alloc_count(), 0);
        state.ensure_device_kda(&gpu).unwrap();
        assert_eq!(state.snapshot(&gpu, 0).unwrap(), bytes);
        state.release(&gpu).unwrap();
        state.release(&gpu).unwrap();
        assert_eq!(gpu.alloc_count(), 0);
    }

    #[test]
    fn failed_resident_allocation_releases_partial_state() {
        let gpu = MockGpuBackend::new();
        let cfg = KdaConfig::twin_0_40b();
        gpu.set_max_allocation_bytes(cfg.conv_elems() * 4);
        let mut state = K3CpuFallbackState::new(LayerCache::Kda(KdaState::new(&cfg)));
        assert!(state.ensure_device_kda(&gpu).is_err());
        assert!(state.device_kda.is_none());
        assert_eq!(gpu.alloc_count(), 0);
    }

    #[test]
    fn resident_mla_snapshot_restore_and_empty_restore() {
        use metrale_model_weights::kimi_k3_host::{MlaConfig, MlaKv};

        let gpu = MockGpuBackend::new();
        let cfg = MlaConfig::twin_0_40b();
        let k_row = cfg.heads * cfg.qk_head_dim();
        let v_row = cfg.heads * cfg.v_head_dim;
        let mut seeded = MlaKv::default();
        seeded.append(&vec![0.5f32; k_row], &vec![0.25f32; v_row]);
        let mut state = K3CpuFallbackState::new(LayerCache::Mla(seeded));
        state.ensure_device_mla(&gpu, &cfg, 8).unwrap();
        state.ensure_device_mla(&gpu, &cfg, 8).unwrap();
        assert_eq!(gpu.alloc_count(), 2, "reuse resident MLA between tokens");
        let bytes = state.snapshot(&gpu, 0).unwrap();
        let LayerCache::Mla(saved) = LayerCache::from_bytes(&bytes).unwrap() else {
            panic!("expected MLA");
        };
        assert_eq!(saved.seq_len, 1);
        assert_eq!(saved.k[0], 0.5);
        assert!(state.restore(&gpu, b"invalid", 0).is_err());
        assert_eq!(
            gpu.alloc_count(),
            2,
            "invalid snapshot must leave state intact"
        );
        state.restore(&gpu, &bytes, 0).unwrap();
        assert!(state.device_mla.is_none());
        let empty = LayerCache::Mla(MlaKv::default()).to_bytes();
        state.restore(&gpu, &empty, 0).unwrap();
        let LayerCache::Mla(cleared) = &state.cache else {
            panic!("expected MLA");
        };
        assert_eq!(cleared.seq_len, 0);
        state.ensure_device_mla(&gpu, &cfg, 8).unwrap();
        assert_eq!(state.device_mla.as_ref().unwrap().seq_len, 0);
        state.release(&gpu).unwrap();
        assert_eq!(gpu.alloc_count(), 0);
    }
}
