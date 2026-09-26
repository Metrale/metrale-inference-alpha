// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Per-row FP8 GDN projections (`METRALE_FP8_ROWWISE=1`): detect,
//! load and concatenate FP8 E4M3 weights whose `weight_scale` holds one
//! multiplier per output row (`[N]` or `[N,1]`).
//!
//! Owner: model-arch weight loader (Qwen3.5 dense).
//! Invariants: none beyond the types.
//!
//! The block-scaled FP8 GDN arm (`proj_is_fp8_any_scale`) accepts only a
//! per-tensor scalar or a 128x128 block grid, so without this lever these
//! projections are dequantized to BF16 and re-quantized to NVFP4. With it,
//! the loader also installs the per-row weights through
//! `set_fp8_rowwise_prefill_weights`. Only the row-wise GDN prefill arms read
//! them, and they multiply a BF16 copy dequantized once per layer
//! (`qwen3_ssm/rowwise_bf16.rs`). The NVFP4 copy is still built
//! (`load_ssm_proj`), because the decode FP8 fields accept only block-scaled
//! weights (`set_fp8_decode_weights`). Design notes:
//! `docs/fp8-rowwise-mixed-precision.md`.

use anyhow::Result;
use metrale_gpu_runtime::gpu::GpuBackend;

use metrale_model_layers::weight_map::{Fp8Weight, WeightQuantFormat};
use metrale_model_weights::weights::{WeightDtype, WeightStore};

/// 2026-09-25: Whether `METRALE_FP8_ROWWISE=1`. Off unless set.
pub(super) fn rowwise_fp8_enabled() -> bool {
    std::env::var("METRALE_FP8_ROWWISE").as_deref() == Ok("1")
}

/// 2026-09-25: True when `{prefix}.weight` is a 2-D FP8 E4M3 tensor and
/// `{prefix}.weight_scale` holds one multiplier per output row (`[N]` or
/// `[N,1]`). For `N > 1`, no scale tensor passes both this and
/// `proj_is_fp8_any_scale`, which accepts a per-tensor scalar or a 128x128
/// block grid.
pub(super) fn proj_is_fp8_per_row(store: &WeightStore, prefix: &str) -> bool {
    let Ok(w) = store.get(&format!("{prefix}.weight")) else {
        return false;
    };
    if w.dtype != WeightDtype::FP8E4M3 || w.shape.len() != 2 {
        return false;
    }
    let Ok(s) = store.get(&format!("{prefix}.weight_scale")) else {
        return false;
    };
    scale_is_per_row(w.shape[0], &s.shape, s.num_elements())
}

/// 2026-09-25: Whether `scale` is one multiplier per output row of an `[n, k]`
/// weight: exactly `n` elements, shaped `[n]` or `[n, 1]`. A per-tensor scalar
/// and a block grid fail the element count.
pub(super) fn scale_is_per_row(n: usize, scale_shape: &[usize], scale_elems: usize) -> bool {
    scale_elems == n && matches!(scale_shape.len(), 1 | 2) && scale_shape[0] == n
}

/// 2026-09-25: Load one per-row FP8 projection as an `Fp8Weight` tagged
/// `Fp8PerRow`. The weight bytes stay store-owned. An F32 scale is used in
/// place; a BF16 scale is widened to F32 on the host into a new device
/// buffer; any other dtype is an error.
pub(super) fn load_fp8_per_row(
    store: &WeightStore,
    prefix: &str,
    gpu: &dyn GpuBackend,
) -> Result<Fp8Weight> {
    let w = store.get(&format!("{prefix}.weight"))?;
    let (n, k) = (w.shape[0], w.shape[1]);
    let s = store.get(&format!("{prefix}.weight_scale"))?;
    anyhow::ensure!(
        s.num_elements() == n,
        "{prefix}.weight_scale must hold exactly one scale per row ([N] or [N,1]); \
         got shape {:?} for a [{n}, {k}] weight",
        s.shape,
    );
    let row_scale = match s.dtype {
        WeightDtype::FP32 => s.ptr,
        WeightDtype::BF16 => {
            let mut bf16 = vec![0u8; n * 2];
            gpu.copy_d2h(s.ptr, &mut bf16)?;
            let mut f32s = vec![0u8; n * 4];
            for i in 0..n {
                let v = f32::from_bits(
                    (u16::from_le_bytes([bf16[i * 2], bf16[i * 2 + 1]]) as u32) << 16,
                );
                f32s[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
            let p = gpu.alloc(n * 4)?;
            gpu.copy_h2d(&f32s, p)?;
            p
        }
        other => {
            anyhow::bail!("{prefix}.weight_scale: unsupported dtype {other:?} (want F32/BF16)")
        }
    };
    Ok(Fp8Weight {
        weight: w.ptr,
        row_scale,
        n: n as u32,
        k: k as u32,
        scale_format: WeightQuantFormat::Fp8PerRow,
    })
}

/// 2026-09-25: Concatenate two per-row FP8 weights along rows into new
/// buffers, `[a.n + b.n, k]`. The scale vectors are copied end to end. Errors
/// unless both are tagged `Fp8PerRow`.
pub(super) fn concat_fp8_per_row(
    a: &Fp8Weight,
    b: &Fp8Weight,
    k: usize,
    gpu: &dyn GpuBackend,
) -> Result<Fp8Weight> {
    anyhow::ensure!(
        a.scale_format == WeightQuantFormat::Fp8PerRow
            && b.scale_format == WeightQuantFormat::Fp8PerRow,
        "concat_fp8_per_row needs two Fp8PerRow weights, got {:?} and {:?}",
        a.scale_format,
        b.scale_format,
    );
    let (a_w, b_w) = (a.n as usize * k, b.n as usize * k);
    let weight = gpu.alloc(a_w + b_w)?;
    gpu.copy_d2d(a.weight, weight, a_w)?;
    gpu.copy_d2d(b.weight, weight.offset(a_w), b_w)?;
    let (a_s, b_s) = (a.n as usize * 4, b.n as usize * 4);
    let row_scale = gpu.alloc(a_s + b_s)?;
    gpu.copy_d2d(a.row_scale, row_scale, a_s)?;
    gpu.copy_d2d(b.row_scale, row_scale.offset(a_s), b_s)?;
    Ok(Fp8Weight {
        weight,
        row_scale,
        n: a.n + b.n,
        k: k as u32,
        scale_format: WeightQuantFormat::Fp8PerRow,
    })
}

#[cfg(test)]
#[path = "rowwise_fp8_tests.rs"]
mod rowwise_fp8_tests;
