// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: DeepSeek-V4 compressor position bias (`attn.compressor.ape`), loaded as FP32.
//!
//! `csa_compress` reads `ape` as `const float*` `[ratio, proj_dim]` and adds
//! row `r` to the gate of window slot `r` before the per-dim softmax.
//!
//! Owner: model-arch weight loader.
//! Invariants: `load_ape_f32` returns a pointer to FP32 data or an error; a
//! missing tensor is an error.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_weights::weights::{WeightDtype, WeightStore};

/// 2026-09-25: Load the compressor `ape` as a device FP32 buffer:
/// - F32: the store pointer itself;
/// - BF16: widened into a newly allocated FP32 buffer;
/// - missing, or any other dtype: an error naming the key.
pub(super) fn load_ape_f32(
    store: &WeightStore,
    key: &str,
    gpu: &dyn GpuBackend,
) -> Result<DevicePtr> {
    let t = store.get(key)?;
    match t.dtype {
        WeightDtype::FP32 => Ok(t.ptr),
        WeightDtype::BF16 => {
            let mut bf16_buf = vec![0u8; t.num_elements() * 2];
            gpu.copy_d2h(t.ptr, &mut bf16_buf)?;
            let f32_buf = super::attn_sink::bf16_bytes_to_f32_bytes(&bf16_buf);
            let ptr = gpu.alloc(f32_buf.len())?;
            gpu.copy_h2d(&f32_buf, ptr)?;
            Ok(ptr)
        }
        other => anyhow::bail!(
            "DeepSeek-V4 compressor.ape '{key}': unexpected dtype {:?} \
             (csa_compress indexes ape as F32; only F32 pass-through or BF16 widening supported)",
            other
        ),
    }
}

#[cfg(test)]
mod csa_ape_dtype_tests {
    use std::collections::HashMap;

    use metrale_gpu_runtime::gpu::mock::MockGpuBackend;
    use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
    use metrale_model_weights::weights::{WeightDtype, WeightStore, WeightTensor};

    use super::load_ape_f32;

    const KEY: &str = "layers.2.attn.compressor.ape";

    fn store_with(ptr: DevicePtr, dtype: WeightDtype, shape: Vec<usize>) -> WeightStore {
        WeightStore::from_map(HashMap::from([(
            KEY.to_string(),
            WeightTensor { ptr, shape, dtype },
        )]))
    }

    #[test]
    fn fp32_checkpoint_pointer_passes_through_unchanged() {
        let gpu = MockGpuBackend::new();
        let ptr = gpu.alloc(32).unwrap();
        let store = store_with(ptr, WeightDtype::FP32, vec![2, 4]);

        assert_eq!(load_ape_f32(&store, KEY, &gpu).unwrap(), ptr);
    }

    #[test]
    fn bf16_checkpoint_is_widened_into_a_new_exact_fp32_buffer() {
        let gpu = MockGpuBackend::new();
        let bf16 = [0x3d97u16, 0xbfc0, 0x3f00, 0x4140];
        let source: Vec<u8> = bf16.iter().flat_map(|bits| bits.to_le_bytes()).collect();
        let source_ptr = gpu.alloc(source.len()).unwrap();
        gpu.copy_h2d(&source, source_ptr).unwrap();
        let store = store_with(source_ptr, WeightDtype::BF16, vec![2, 2]);

        let widened_ptr = load_ape_f32(&store, KEY, &gpu).unwrap();
        assert_ne!(widened_ptr, source_ptr);
        let mut widened = vec![0u8; bf16.len() * 4];
        gpu.copy_d2h(widened_ptr, &mut widened).unwrap();
        let actual: Vec<u32> = widened
            .chunks_exact(4)
            .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
            .collect();
        assert_eq!(
            actual,
            bf16.iter()
                .map(|bits| (*bits as u32) << 16)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn missing_required_ape_names_the_tensor() {
        let gpu = MockGpuBackend::new();
        let error = load_ape_f32(&WeightStore::empty(), KEY, &gpu)
            .unwrap_err()
            .to_string();
        assert!(error.contains(KEY), "{error}");
    }

    #[test]
    fn unsupported_checkpoint_dtype_names_the_tensor_and_contract() {
        let gpu = MockGpuBackend::new();
        let store = store_with(DevicePtr::NULL, WeightDtype::FP8E4M3, vec![2, 2]);
        let error = load_ape_f32(&store, KEY, &gpu).unwrap_err().to_string();

        assert!(error.contains(KEY), "{error}");
        assert!(error.contains("FP8E4M3"), "{error}");
        assert!(error.contains("indexes ape as F32"), "{error}");
    }
}
