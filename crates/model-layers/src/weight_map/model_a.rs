// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Basic store accessors: raw pointers, scalars, KV scales, ModelOpt NVFP4, native MXFP4, and dtype-converting dense loads.
//!
//! Owner: model-layers (weight loading).
//! Invariants: none beyond the types.

#![allow(unused_imports)]

use anyhow::{Context, Result, bail, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_weights::weights::{WeightDtype, WeightStore};

use super::*;

/// 2026-09-25: Whole-model weights built by `ModelWeights::from_store`.
pub struct ModelWeights {
    pub embed_tokens: DenseWeight,
    pub final_norm: DenseWeight,
    /// 2026-09-25: `lm_head.weight`, or `embed_tokens` when the checkpoint has none.
    pub lm_head: DenseWeight,
    pub layers: Vec<LayerWeights>,
}

/// 2026-09-25: The device pointer of tensor `name`.
pub(crate) fn ptr(store: &WeightStore, name: &str) -> Result<DevicePtr> {
    Ok(store.get(name)?.ptr)
}

/// 2026-09-25: Read a one-element FP32 tensor to the host; any other dtype or size is an error.
pub(crate) fn scalar_f32(store: &WeightStore, name: &str, gpu: &dyn GpuBackend) -> Result<f32> {
    let w = store.get(name)?;
    ensure!(
        w.dtype == WeightDtype::FP32,
        "Expected FP32 for {name}, got {:?}",
        w.dtype
    );
    ensure!(
        w.num_elements() == 1,
        "Expected scalar for {name}, got {} elements",
        w.num_elements()
    );
    let mut buf = [0u8; 4];
    gpu.copy_d2h(w.ptr, &mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

/// 2026-09-25: Read the FP8 KV-cache scales `{attn_prefix}.k_proj.k_scale` and
/// `{attn_prefix}.v_proj.v_scale`. Each is 1.0 when absent (debug log) or
/// unreadable (warning).
pub fn load_kv_scales(store: &WeightStore, attn_prefix: &str, gpu: &dyn GpuBackend) -> (f32, f32) {
    let k_key = format!("{attn_prefix}.k_proj.k_scale");
    let v_key = format!("{attn_prefix}.v_proj.v_scale");

    let k_scale = if store.contains(&k_key) {
        match scalar_f32(store, &k_key, gpu) {
            Ok(v) => {
                tracing::debug!("Loaded k_scale={v:.6} from {k_key}");
                v
            }
            Err(e) => {
                tracing::warn!("Failed to load {k_key}: {e:#}, using 1.0");
                1.0
            }
        }
    } else {
        tracing::debug!("No {k_key} in checkpoint, using k_scale=1.0");
        1.0
    };

    let v_scale = if store.contains(&v_key) {
        match scalar_f32(store, &v_key, gpu) {
            Ok(v) => {
                tracing::debug!("Loaded v_scale={v:.6} from {v_key}");
                v
            }
            Err(e) => {
                tracing::warn!("Failed to load {v_key}: {e:#}, using 1.0");
                1.0
            }
        }
    } else {
        tracing::debug!("No {v_key} in checkpoint, using v_scale=1.0");
        1.0
    };

    (k_scale, v_scale)
}

/// 2026-09-25: Build a `QuantizedWeight` from ModelOpt NVFP4 keys: `weight`,
/// `weight_scale`, the FP32 scalar `weight_scale_2` (read to the host), and an
/// optional `input_scale`.
pub fn quantized(
    store: &WeightStore,
    prefix: &str,
    gpu: &dyn GpuBackend,
) -> Result<QuantizedWeight> {
    let input_scale_key = format!("{prefix}.input_scale");
    Ok(QuantizedWeight {
        weight: ptr(store, &format!("{prefix}.weight"))?,
        weight_scale: ptr(store, &format!("{prefix}.weight_scale"))?,
        weight_scale_2: scalar_f32(store, &format!("{prefix}.weight_scale_2"), gpu)?,
        input_scale: if store.contains(&input_scale_key) {
            ptr(store, &input_scale_key)?
        } else {
            DevicePtr::NULL
        },
        weight_scale_2_vec: DevicePtr::NULL,
    })
}

/// 2026-09-25: [`quantized_mxfp4_e8m0_pair`] on `{prefix}.weight` and `{prefix}.scale`.
pub fn quantized_mxfp4_e8m0(store: &WeightStore, prefix: &str) -> Result<QuantizedWeight> {
    quantized_mxfp4_e8m0_pair(
        store,
        &format!("{prefix}.weight"),
        &format!("{prefix}.scale"),
    )
}

/// 2026-09-25: Hand out a native MXFP4 weight's store pointers unchanged: E2M1
/// nibbles `[n, k/2]` in `weight_key` and one E8M0 scale per 32 weights in
/// `scale_key`. Any other group size inferred from the element counts is an
/// error. `weight_scale_2` is 1.0 because the format has no per-tensor scale.
pub fn quantized_mxfp4_e8m0_pair(
    store: &WeightStore,
    weight_key: &str,
    scale_key: &str,
) -> Result<QuantizedWeight> {
    let w = store.get(weight_key)?;
    ensure!(
        w.shape.len() == 2,
        "{weight_key}: expected packed rank-2 matrix"
    );
    let n = w.shape[0];
    let k_packed = w.shape[1];
    let total_nibbles = n * k_packed * 2;
    let scale_t = store.get(scale_key)?;
    let num_groups = scale_t.num_elements();
    ensure!(
        num_groups > 0 && total_nibbles.is_multiple_of(num_groups),
        "{weight_key}: MXFP4 weight nibbles {total_nibbles} not divisible by E8M0 scale groups {num_groups}"
    );
    let block = total_nibbles / num_groups;
    ensure!(
        block == 32,
        "{weight_key}: native MXFP4 expects GROUP_SIZE=32, inferred {block} (scale groups {num_groups}) \
         — refusing to land a non-MX checkpoint on the transcode-free path"
    );
    Ok(QuantizedWeight {
        weight: ptr(store, weight_key)?,
        weight_scale: ptr(store, scale_key)?,
        weight_scale_2: 1.0,
        input_scale: DevicePtr::NULL,
        weight_scale_2_vec: DevicePtr::NULL,
    })
}

pub fn dense(store: &WeightStore, name: &str) -> Result<DenseWeight> {
    let w = store.get(name)?;
    Ok(DenseWeight { weight: w.ptr })
}

/// 2026-09-25: Load `{prefix}.weight` as BF16: a BF16 tensor is the store
/// pointer, FP8E4M3 goes through `dequant_fp8_blockscaled_to_bf16`, and any
/// other dtype is an error.
pub fn dense_auto_fp8_or_bf16(
    store: &WeightStore,
    prefix: &str,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    let w = store.get(&format!("{prefix}.weight"))?;
    match w.dtype {
        WeightDtype::BF16 => Ok(DenseWeight { weight: w.ptr }),
        WeightDtype::FP8E4M3 => dequant_fp8_blockscaled_to_bf16(store, prefix, gpu),
        other => anyhow::bail!(
            "dense_auto_fp8_or_bf16: unsupported dtype {:?} for {prefix}.weight",
            other
        ),
    }
}

/// 2026-09-25: Load `name` as BF16: an FP32 tensor is truncated into a new
/// buffer, any other dtype is returned as the store pointer unchanged.
pub fn dense_f32_safe(
    store: &WeightStore,
    name: &str,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    let w = store.get(name)?;
    if w.dtype == WeightDtype::FP32 {
        // 2026-09-25: `f32_to_bf16_trunc` keeps the high 16 bits of each f32
        // (round toward zero). It runs on the default stream and is not
        // synchronized; a reader on another stream must synchronize it first.
        let n = w.num_elements();
        let ptr = gpu.alloc(n * 2)?;
        let trunc = gpu.kernel("quantize_nvfp4", "f32_to_bf16_trunc")?;
        let blocks = (n.div_ceil(256) as u32).max(1);
        metrale_gpu_runtime::kernel_args::KernelLaunch::new(gpu, trunc)
            .grid([blocks, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(w.ptr)
            .arg_ptr(ptr)
            .arg_u32(n as u32)
            .launch(gpu.default_stream())?;
        Ok(DenseWeight { weight: ptr })
    } else {
        Ok(DenseWeight { weight: w.ptr })
    }
}

/// 2026-09-25: Load `name` as FP32: an FP32 tensor is the store pointer, BF16 is
/// widened on the host into a new buffer, and any other dtype is an error.
/// `load_ssm` uses it for `A_log` and `dt_bias`.
pub fn dense_keep_f32(
    store: &WeightStore,
    name: &str,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    let w = store.get(name)?;
    match w.dtype {
        WeightDtype::FP32 => {
            // 2026-09-25: Already FP32: the store's pointer, unconverted.
            Ok(DenseWeight { weight: w.ptr })
        }
        WeightDtype::BF16 => {
            tracing::info!(
                "dense_keep_f32: promoting {name} from BF16 to FP32 ({:?})",
                w.shape
            );
            let n = w.num_elements();
            let mut bf16_buf = vec![0u8; n * 2];
            gpu.copy_d2h(w.ptr, &mut bf16_buf)?;
            let f32_buf: Vec<u8> = bf16_buf
                .chunks_exact(2)
                .flat_map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    let f32_bits = (bits as u32) << 16;
                    f32_bits.to_le_bytes()
                })
                .collect();
            let ptr = gpu.alloc(f32_buf.len())?;
            gpu.copy_h2d(&f32_buf, ptr)?;
            Ok(DenseWeight { weight: ptr })
        }
        other => {
            bail!("dense_keep_f32: unsupported dtype {:?} for {name}", other);
        }
    }
}

/// 2026-09-25: Load a BF16 tensor into a new F32 device buffer, widened on the
/// host; any other dtype is an error. The Nemotron-H Mamba loader uses it for
/// `conv1d.bias`, `A_log`, `D` and `dt_bias`.
pub(crate) fn dense_bf16_as_f32(
    store: &WeightStore,
    name: &str,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    let w = store.get(name)?;
    ensure!(
        w.dtype == WeightDtype::BF16,
        "Expected BF16 for {name}, got {:?}",
        w.dtype
    );
    let n = w.num_elements();
    let mut bf16_buf = vec![0u8; n * 2];
    gpu.copy_d2h(w.ptr, &mut bf16_buf)?;
    let f32_buf: Vec<u8> = bf16_buf
        .chunks_exact(2)
        .flat_map(|c| {
            let bits = u16::from_le_bytes([c[0], c[1]]);
            let f32_bits = (bits as u32) << 16;
            f32_bits.to_le_bytes()
        })
        .collect();
    let ptr = gpu.alloc(f32_buf.len())?;
    gpu.copy_h2d(&f32_buf, ptr)?;
    Ok(DenseWeight { weight: ptr })
}

/// 2026-09-25: Load an FP32 tensor into a new BF16 device buffer, converted on
/// the host with `f32_to_bf16`; any other dtype is an error.
pub fn dense_f32_as_bf16(
    store: &WeightStore,
    name: &str,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    let w = store.get(name)?;
    ensure!(
        w.dtype == WeightDtype::FP32,
        "Expected FP32 for {name}, got {:?}",
        w.dtype
    );
    let n = w.num_elements();
    let mut f32_buf = vec![0u8; n * 4];
    gpu.copy_d2h(w.ptr, &mut f32_buf)?;
    let bf16_buf: Vec<u8> = f32_buf
        .chunks_exact(4)
        .flat_map(|c| {
            let val = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            f32_to_bf16(val).to_le_bytes()
        })
        .collect();
    let ptr = gpu.alloc(bf16_buf.len())?;
    gpu.copy_h2d(&bf16_buf, ptr)?;
    Ok(DenseWeight { weight: ptr })
}
