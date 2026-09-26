// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: FP8-to-BF16 dequantization, dtype-dispatching dense loads, and the compressed-tensors NVFP4 loader.
//!
//! Owner: model-layers (weight loading).
//! Invariants: none beyond the types.

#![allow(unused_imports)]

use anyhow::{Context, Result, bail, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_weights::weights::{WeightDtype, WeightStore};

use super::*;

/// 2026-09-25: Convert FP8 E4M3 bytes to little-endian BF16 bytes on the host,
/// multiplying each value by `scale`.
pub(super) fn dequant_fp8_bytes_to_bf16(fp8_buf: &[u8], scale: f32) -> Vec<u8> {
    fp8_buf
        .iter()
        .flat_map(|&byte| {
            let val = fp8_e4m3_to_f32(byte) * scale;
            f32_to_bf16(val).to_le_bytes()
        })
        .collect()
}

/// 2026-09-25: Dequantize a block-scaled FP8 E4M3 weight `{prefix}.weight`
/// `[N, K]` into a new BF16 device buffer:
/// `bf16[i,j] = fp8[i,j] * scale[i/block_n, j/block_k]`, with the block sizes
/// inferred from the scale's shape.
///
/// A `weight_scale_inv`, or else a 2-D `weight_scale`, must be BF16 or FP32 and
/// is applied on the device by `dequant_fp8_blockscaled_bf16`. Otherwise a 1-D
/// `weight_scale` or a `.scale` (FP32, BF16 or E8M0) is applied on the host.
/// Having none of them is an error.
pub fn dequant_fp8_blockscaled_to_bf16(
    store: &WeightStore,
    prefix: &str,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

    let w = store.get(&format!("{prefix}.weight"))?;
    ensure!(
        w.dtype == WeightDtype::FP8E4M3,
        "Expected FP8E4M3 for {prefix}.weight, got {:?}",
        w.dtype,
    );
    ensure!(
        w.shape.len() == 2,
        "Expected 2D weight for {prefix}, got {:?}",
        w.shape
    );
    let n = w.shape[0];
    let k = w.shape[1];
    let total = n * k;
    let byte_size = w.byte_size();
    ensure!(
        total == byte_size,
        "FP8 size mismatch: total={total} byte_size={byte_size}"
    );

    // 2026-09-25: The kernel reads a 2-D BF16 or FP32 scale only, so E8M0
    // `.scale` and 1-D `weight_scale` fall through to the host path. Keep this a
    // fall-through: a weight with only `.scale` has no `weight_scale*` key.
    let gpu_scale = store
        .get(&format!("{prefix}.weight_scale_inv"))
        .ok()
        .or_else(|| {
            store
                .get(&format!("{prefix}.weight_scale"))
                .ok()
                .filter(|s| s.shape.len() == 2)
        });
    if let Some(s) = gpu_scale {
        ensure!(
            s.dtype == WeightDtype::BF16 || s.dtype == WeightDtype::FP32,
            "Expected BF16 or FP32 for {prefix} GPU block scale, got {:?}",
            s.dtype,
        );
        let sn = s.shape[0];
        let sk = s.shape[1];
        let block_n = (n / sn) as u32;
        let block_k = (k / sk) as u32;
        let scale_is_f32 = s.dtype == WeightDtype::FP32;

        let out = gpu.alloc(total * 2)?;

        let stream = gpu.default_stream();
        let kernel = gpu.kernel(
            "dequant_fp8_blockscaled_bf16",
            "dequant_fp8_blockscaled_bf16",
        )?;
        KernelLaunch::new(gpu, kernel)
            .grid([div_ceil(k as u32, 64), div_ceil(n as u32, 4), 1])
            .block([64, 4, 1])
            .arg_ptr(w.ptr)
            .arg_ptr(s.ptr)
            .arg_ptr(out)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .arg_u32(block_n)
            .arg_u32(block_k)
            .arg_u32(sk as u32)
            .arg_u32(scale_is_f32 as u32)
            .launch(stream)?;
        // 2026-09-25: Not synchronized: `out` is written on the default stream,
        // and a reader on another stream must synchronize it first.

        tracing::debug!(
            "GPU-dequanted FP8 blockscaled {prefix}: [{n}, {k}] block=[{block_n}, {block_k}] → BF16",
        );
        return Ok(DenseWeight { weight: out });
    }

    // 2026-09-25: Host path: download the FP8 weight and the scale, dequantize,
    // upload the BF16 result.
    enum ScaleDtype {
        Fp32,
        Bf16,
        E8M0,
    }
    let mut fp8_buf = vec![0u8; total];
    gpu.copy_d2h(w.ptr, &mut fp8_buf).with_context(|| {
        format!(
            "D2H failed for {prefix}.weight: ptr={}, size={total}",
            w.ptr.0
        )
    })?;

    let (scale_buf, _sn, sk, block_n, block_k, scale_dtype) = if let Ok(s) =
        store.get(&format!("{prefix}.weight_scale"))
    {
        ensure!(
            s.dtype == WeightDtype::BF16 || s.dtype == WeightDtype::FP32,
            "Expected BF16 or FP32 2-D block scale for {prefix}.weight_scale, got {:?}",
            s.dtype,
        );
        let rank = s.shape.len();
        let (sn, sk) = if rank == 2 {
            (s.shape[0], s.shape[1])
        } else if rank == 1 {
            (s.shape[0], 1)
        } else {
            bail!(
                "Expected 1-D or 2-D scale for {prefix}.weight_scale, got shape {:?}",
                s.shape
            );
        };
        let block_n = if sn > 1 { n / sn } else { n };
        let block_k = if sk > 1 { k / sk } else { k };
        let scale_is_f32 = s.dtype == WeightDtype::FP32;
        let scale_bytes_per = if scale_is_f32 { 4 } else { 2 };
        let mut buf = vec![0u8; sn * sk * scale_bytes_per];
        gpu.copy_d2h(s.ptr, &mut buf).with_context(|| {
            format!(
                "D2H failed for {prefix}.weight_scale: ptr={}, size={}",
                s.ptr.0,
                sn * sk * scale_bytes_per
            )
        })?;
        let sd = if scale_is_f32 {
            ScaleDtype::Fp32
        } else {
            ScaleDtype::Bf16
        };
        (buf, sn, sk, block_n, block_k, sd)
    } else if let Ok(s) = store.get(&format!("{prefix}.scale")) {
        let rank = s.shape.len();
        let (sn, sk) = if rank == 2 {
            (s.shape[0], s.shape[1])
        } else if rank == 1 {
            (s.shape[0], 1)
        } else {
            bail!(
                "Expected 1-D or 2-D scale for {prefix}.scale, got shape {:?}",
                s.shape
            );
        };
        let block_n = if sn > 1 { n / sn } else { n };
        let block_k = if sk > 1 { k / sk } else { k };
        let sd = match s.dtype {
            WeightDtype::FP32 => ScaleDtype::Fp32,
            WeightDtype::BF16 => ScaleDtype::Bf16,
            WeightDtype::FP8E8M0 => ScaleDtype::E8M0,
            other => bail!(
                "Expected FP32, BF16, or FP8E8M0 for {prefix}.scale, got {:?}",
                other,
            ),
        };
        let scale_bytes_per = s.dtype.byte_size();
        let mut buf = vec![0u8; sn * sk * scale_bytes_per];
        gpu.copy_d2h(s.ptr, &mut buf).with_context(|| {
            format!(
                "D2H failed for {prefix}.scale: ptr={}, size={}",
                s.ptr.0,
                sn * sk * scale_bytes_per
            )
        })?;
        (buf, sn, sk, block_n, block_k, sd)
    } else {
        bail!(
            "FP8 tensor {prefix}: no .weight_scale_inv, .weight_scale, or .scale found for dequant"
        );
    };

    let mut bf16_out = vec![0u8; total * 2];
    for row in 0..n {
        let scale_row = row / block_n;
        for col in 0..k {
            let scale_col = col / block_k;
            let scale_idx = scale_row * sk + scale_col;
            let scale_f32 = match scale_dtype {
                ScaleDtype::E8M0 => fp8_e8m0_to_f32(scale_buf[scale_idx]),
                ScaleDtype::Fp32 => {
                    let b = [
                        scale_buf[scale_idx * 4],
                        scale_buf[scale_idx * 4 + 1],
                        scale_buf[scale_idx * 4 + 2],
                        scale_buf[scale_idx * 4 + 3],
                    ];
                    f32::from_le_bytes(b)
                }
                ScaleDtype::Bf16 => {
                    let b = [scale_buf[scale_idx * 2], scale_buf[scale_idx * 2 + 1]];
                    bf16_bytes_to_f32(b)
                }
            };

            let fp8_byte = fp8_buf[row * k + col];
            let val = fp8_e4m3_to_f32(fp8_byte) * scale_f32;
            let bf16_val = f32_to_bf16(val);
            let out_idx = (row * k + col) * 2;
            let [lo, hi] = bf16_val.to_le_bytes();
            bf16_out[out_idx] = lo;
            bf16_out[out_idx + 1] = hi;
        }
    }

    let out = gpu.alloc(bf16_out.len())?;
    gpu.copy_h2d(&bf16_out, out)?;
    tracing::debug!(
        "CPU-dequanted FP8 blockscaled {prefix}: [{n}, {k}] block=[{block_n}, {block_k}] → BF16",
    );
    Ok(DenseWeight { weight: out })
}

/// 2026-09-25: Widen little-endian BF16 bytes to f32.
pub(super) fn bf16_bytes_to_f32(bytes: [u8; 2]) -> f32 {
    let bits = u16::from_le_bytes(bytes);
    f32::from_bits((bits as u32) << 16)
}

/// 2026-09-25: Load `name` as a BF16 `DenseWeight`, converting by dtype:
/// - BF16: the store pointer itself;
/// - FP32: `dense_f32_safe`;
/// - FP8E4M3: `dequant_fp8_blockscaled_to_bf16` when the weight has a
///   `weight_scale_inv`, a multi-element `weight_scale` or a `.scale`, else the
///   per-tensor `dequant_fp8_to_bf16`;
/// - UInt8 (packed NVFP4, `[n, k/2]`): `dequant_nvfp4_to_bf16`.
///
/// Any other dtype is an error. FP8 and NVFP4 names must end in `.weight`.
pub fn dense_auto(store: &WeightStore, name: &str, gpu: &dyn GpuBackend) -> Result<DenseWeight> {
    let w = store.get(name)?;
    match w.dtype {
        WeightDtype::BF16 => Ok(DenseWeight { weight: w.ptr }),
        WeightDtype::FP32 => dense_f32_safe(store, name, gpu),
        WeightDtype::FP8E4M3 => {
            let prefix = name
                .strip_suffix(".weight")
                .ok_or_else(|| anyhow::anyhow!("FP8 tensor {name} doesn't end with .weight"))?;
            // 2026-09-25: A one-element `weight_scale` is a per-tensor scale; every
            // other scale form is per block.
            let has_blockscale = store.contains(&format!("{prefix}.weight_scale_inv"));
            let has_per_row_scale = store
                .get(&format!("{prefix}.weight_scale"))
                .map(|s| s.num_elements() > 1)
                .unwrap_or(false);
            let has_v4_scale = store.contains(&format!("{prefix}.scale"));
            if has_blockscale || has_per_row_scale || has_v4_scale {
                dequant_fp8_blockscaled_to_bf16(store, prefix, gpu)
            } else {
                dequant_fp8_to_bf16(store, prefix, gpu)
            }
        }
        WeightDtype::UInt8 => {
            let prefix = name
                .strip_suffix(".weight")
                .ok_or_else(|| anyhow::anyhow!("NVFP4 tensor {name} doesn't end with .weight"))?;
            if w.shape.len() != 2 {
                anyhow::bail!(
                    "dense_auto: packed NVFP4 {name} must be 2-D, got {:?}",
                    w.shape
                );
            }
            crate::weight_map::dequant_nvfp4_to_bf16(store, prefix, w.shape[0], w.shape[1] * 2, gpu)
        }
        other => anyhow::bail!("dense_auto: unsupported dtype {:?} for {name}", other),
    }
}

/// 2026-09-25: Build a `QuantizedWeight` from compressed-tensors NVFP4 keys:
/// `weight_packed`, `weight_scale`, `weight_global_scale` and an optional
/// `input_global_scale`. compressed-tensors stores the global scale as the
/// reciprocal of `weight_scale_2`, so `weight_scale_2 = 1 / weight_global_scale`.
pub fn quantized_v2(
    store: &WeightStore,
    prefix: &str,
    gpu: &dyn GpuBackend,
) -> Result<QuantizedWeight> {
    let raw_global_scale = scalar_f32(store, &format!("{prefix}.weight_global_scale"), gpu)?;
    // 2026-09-25: A zero or non-finite global scale is a load error, because its
    // reciprocal would be stored as `weight_scale_2`.
    if !raw_global_scale.is_finite() || raw_global_scale.abs() < f32::MIN_POSITIVE {
        anyhow::bail!(
            "{prefix}.weight_global_scale is non-finite or zero ({raw_global_scale}); \
             checkpoint likely corrupted"
        );
    }
    Ok(QuantizedWeight {
        weight: ptr(store, &format!("{prefix}.weight_packed"))?,
        weight_scale: ptr(store, &format!("{prefix}.weight_scale"))?,
        weight_scale_2: 1.0 / raw_global_scale,
        // 2026-09-25: Weight-only checkpoints have no `input_global_scale`; then NULL.
        input_scale: ptr(store, &format!("{prefix}.input_global_scale")).unwrap_or(DevicePtr::NULL),
        weight_scale_2_vec: DevicePtr::NULL,
    })
}
