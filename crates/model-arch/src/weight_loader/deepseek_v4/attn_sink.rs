// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: DeepSeek-V4 per-head attention sink (`attn.attn_sink`), loaded as FP32.
//!
//! The kernels that take it (`attn_prefill_512`, `prefill_attn_compressed`,
//! `mla_paged_decode_fp8`) read it as `const float*` and skip it when it is NULL.
//!
//! Owner: model-arch weight loader.
//! Invariants: `load_attn_sink_f32` returns NULL or a pointer to FP32 data; a
//! dtype other than F32 or BF16 is an error.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_weights::weights::{WeightDtype, WeightStore};

/// 2026-09-25: Load the per-head attention sink as a device FP32 buffer:
/// - F32: the store pointer itself;
/// - BF16: widened into a newly allocated FP32 buffer;
/// - missing: `DevicePtr::NULL`;
/// - any other dtype: an error naming the key and dtype.
pub(super) fn load_attn_sink_f32(
    store: &WeightStore,
    key: &str,
    gpu: &dyn GpuBackend,
) -> Result<DevicePtr> {
    let t = match store.get(key) {
        Err(_) => return Ok(DevicePtr::NULL),
        Ok(t) => t,
    };
    match t.dtype {
        WeightDtype::FP32 => Ok(t.ptr),
        WeightDtype::BF16 => {
            let mut bf16_buf = vec![0u8; t.num_elements() * 2];
            gpu.copy_d2h(t.ptr, &mut bf16_buf)?;
            let f32_buf = bf16_bytes_to_f32_bytes(&bf16_buf);
            let ptr = gpu.alloc(f32_buf.len())?;
            gpu.copy_h2d(&f32_buf, ptr)?;
            Ok(ptr)
        }
        other => anyhow::bail!(
            "DeepSeek-V4 attn_sink '{key}': unexpected dtype {:?} \
             (sink kernels require F32; only F32 pass-through or BF16 widening supported)",
            other
        ),
    }
}

/// 2026-09-25: Widen little-endian BF16 bytes to F32 bytes exactly: each BF16
/// value becomes the high 16 bits of its F32 word, and the low 16 bits are zero.
pub(crate) fn bf16_bytes_to_f32_bytes(bf16: &[u8]) -> Vec<u8> {
    let n = bf16.len() / 2;
    let mut out = vec![0u8; n * 4];
    for i in 0..n {
        out[i * 4 + 2] = bf16[i * 2];
        out[i * 4 + 3] = bf16[i * 2 + 1];
    }
    out
}

#[cfg(test)]
mod attn_sink_dtype_tests {
    use std::collections::HashMap;

    use metrale_gpu_runtime::gpu::mock::MockGpuBackend;
    use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
    use metrale_model_weights::weights::{WeightDtype, WeightStore, WeightTensor};

    use super::load_attn_sink_f32;

    const KEY: &str = "layers.0.attn.attn_sink";

    fn store_with(ptr: DevicePtr, dtype: WeightDtype, elements: usize) -> WeightStore {
        WeightStore::from_map(HashMap::from([(
            KEY.to_string(),
            WeightTensor {
                ptr,
                shape: vec![elements],
                dtype,
            },
        )]))
    }

    #[test]
    fn fp32_checkpoint_pointer_passes_through_unchanged() {
        let gpu = MockGpuBackend::new();
        let ptr = gpu.alloc(16).unwrap();
        let store = store_with(ptr, WeightDtype::FP32, 4);

        assert_eq!(load_attn_sink_f32(&store, KEY, &gpu).unwrap(), ptr);
    }

    #[test]
    fn bf16_checkpoint_is_widened_into_a_new_exact_fp32_buffer() {
        let gpu = MockGpuBackend::new();
        let bf16 = [0x3f80u16, 0x3fc0, 0xbf80, 0xbdbc];
        let source: Vec<u8> = bf16.iter().flat_map(|bits| bits.to_le_bytes()).collect();
        let source_ptr = gpu.alloc(source.len()).unwrap();
        gpu.copy_h2d(&source, source_ptr).unwrap();
        let store = store_with(source_ptr, WeightDtype::BF16, bf16.len());

        let widened_ptr = load_attn_sink_f32(&store, KEY, &gpu).unwrap();
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
    fn missing_sink_returns_null_without_allocating() {
        let gpu = MockGpuBackend::new();
        assert_eq!(
            load_attn_sink_f32(&WeightStore::empty(), KEY, &gpu).unwrap(),
            DevicePtr::NULL
        );
    }

    #[test]
    fn unsupported_checkpoint_dtype_names_the_tensor_and_contract() {
        let gpu = MockGpuBackend::new();
        let store = store_with(DevicePtr::NULL, WeightDtype::FP8E4M3, 4);
        let error = load_attn_sink_f32(&store, KEY, &gpu)
            .unwrap_err()
            .to_string();

        assert!(error.contains(KEY), "{error}");
        assert!(error.contains("FP8E4M3"), "{error}");
        assert!(error.contains("require F32"), "{error}");
    }
}
