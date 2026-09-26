// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Load-time weight conversions: NVFP4 and NVFP4-with-E8M0 to BF16,
//! the E8M0 lookup, dense-FFN loading, row concatenation and the BA
//! interleave for the GDN gate kernel.
//!
//! Owner: model-layers (weight loading).
//! Invariants: none beyond the types.

#![allow(unused_imports)]

use anyhow::{Context, Result, bail, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_weights::weights::{WeightDtype, WeightStore};

use super::*;

/// 2026-09-25: Dequantize an NVFP4 weight to a new BF16 GPU buffer, on the GPU.
///
/// Detects the layout from the store keys:
/// - **compressed-tensors**: `weight_packed`, `weight_scale`, `weight_global_scale` (reciprocal)
/// - **Standard (modelopt)**: `weight`, `weight_scale`, `weight_scale_2` (direct multiplier)
pub fn dequant_nvfp4_to_bf16(
    store: &WeightStore,
    prefix: &str,
    n: usize,
    k: usize,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    let total = n * k;

    let (packed_ptr, scale_ptr, global_scale, is_reciprocal) =
        if store.contains(&format!("{prefix}.weight_packed")) {
            let pp = ptr(store, &format!("{prefix}.weight_packed"))?;
            let sp = ptr(store, &format!("{prefix}.weight_scale"))?;
            let gs = scalar_f32(store, &format!("{prefix}.weight_global_scale"), gpu)?;
            (pp, sp, gs, true)
        } else {
            let pp = ptr(store, &format!("{prefix}.weight"))?;
            let sp = ptr(store, &format!("{prefix}.weight_scale"))?;
            let gs = scalar_f32(store, &format!("{prefix}.weight_scale_2"), gpu)?;
            (pp, sp, gs, false)
        };

    // 2026-09-25: Fold the global-scale convention into one multiplier for the
    // kernel: compressed-tensors stores a reciprocal global (val = E2M1 *
    // fp8_scale / global), ModelOpt a direct multiplier (val = E2M1 * fp8_scale
    // * global). A zero reciprocal global gives 0.
    let combined_global = if is_reciprocal {
        if global_scale != 0.0 {
            1.0 / global_scale
        } else {
            0.0
        }
    } else {
        global_scale
    };

    // 2026-09-25: One sync after the launch, so the BF16 is ready when this
    // returns.
    let out = gpu.alloc(total * 2)?;
    let kernel = gpu.kernel("dequant_nvfp4_bf16", "dequant_nvfp4_to_bf16")?;
    let stream = gpu.default_stream();
    metrale_gpu_runtime::kernel_args::KernelLaunch::new(gpu, kernel)
        .grid([n as u32, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(packed_ptr)
        .arg_ptr(scale_ptr)
        .arg_ptr(out)
        .arg_f32(combined_global)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .launch(stream)?;
    gpu.synchronize(stream)?;
    Ok(DenseWeight { weight: out })
}

/// 2026-09-25: Dequantize a 4-bit E2M1 weight with E8M0 (power-of-2) per-block
/// scales and no global scale to BF16 on the host, then upload it. `.weight`
/// holds two E2M1 values per byte and `.scale` one E8M0 byte per block; the
/// block size is `total / scale elements`. It has no caller in this crate.
#[allow(dead_code)]
pub(crate) fn dequant_nvfp4_e8m0_to_bf16(
    store: &WeightStore,
    prefix: &str,
    n: usize,
    k: usize,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    let total = n * k;
    let packed_bytes = total / 2;
    let packed_ptr = ptr(store, &format!("{prefix}.weight"))?;
    let scale_t = store.get(&format!("{prefix}.scale"))?;
    let num_groups = scale_t.num_elements();
    ensure!(
        num_groups > 0 && total.is_multiple_of(num_groups),
        "{prefix}: weight elems {total} not divisible by E8M0 scale groups {num_groups}"
    );
    let block = total / num_groups;

    let mut packed = vec![0u8; packed_bytes];
    let mut scales = vec![0u8; num_groups];
    gpu.copy_d2h(packed_ptr, &mut packed)?;
    gpu.copy_d2h(scale_t.ptr, &mut scales)?;

    let e2m1_table: [f32; 16] = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];
    // 2026-09-25: Scale group `g` covers flat weight indices
    // `g*block .. (g+1)*block`; an even flat index is the low nibble, as in the
    // `dequant_nvfp4_bf16` kernel.
    let mut bf16_out = vec![0u16; total];
    for group in 0..num_groups {
        let block_scale = fp8_e8m0_to_f32(scales[group]);
        for elem in 0..block {
            let flat_idx = group * block + elem;
            let byte_idx = flat_idx / 2;
            let nibble = if flat_idx.is_multiple_of(2) {
                packed[byte_idx] & 0x0F
            } else {
                (packed[byte_idx] >> 4) & 0x0F
            };
            bf16_out[flat_idx] = f32_to_bf16(e2m1_table[nibble as usize] * block_scale);
        }
    }

    let buf = gpu.alloc(total * 2)?;
    // 2026-09-25: SAFETY: `bf16_out` is `vec![0u16; total]`, so `bf16_out.len() == total` and
    // every element is initialised (zeroed at construction, then overwritten by the
    // `group`/`elem` dequant loop above). `total * 2 == bf16_out.len() *
    // size_of::<u16>()`, so the span is exactly the Vec's buffer. Shared borrow
    // only; `buf` was allocated at `total * 2` bytes so the H2D destination matches.
    let bf16_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(bf16_out.as_ptr() as *const u8, total * 2) };
    gpu.copy_h2d(bf16_bytes, buf)?;
    Ok(DenseWeight { weight: buf })
}

/// 2026-09-25: FP8 E4M3 decode and the f32 → BF16 cast, from
/// `metrale_core::numeric`, for the `weight_map` modules.
pub(super) use metrale_core::numeric::{FP8_E4M3_LUT, f32_to_bf16, fp8_e4m3_to_f32};

/// 2026-09-25: FP8 E8M0 → f32 lookup table (256 entries): `2^(exp - 127)` for
/// exp 1..=254. exp 0 and exp 255 (NaN) map to 0.0.
const FP8_E8M0_LUT: [f32; 256] = {
    let mut table = [0.0f32; 256];
    let mut i: u32 = 0;
    while i < 256 {
        let exp = i as u8;
        table[i as usize] = if exp == 0 {
            0.0f32
        } else if exp == 255 {
            0.0f32
        } else {
            f32::from_bits((exp as u32) << 23)
        };
        i += 1;
    }
    table
};

/// 2026-09-25: Convert an FP8 E8M0 byte to f32 through `FP8_E8M0_LUT`.
#[inline(always)]
pub(super) fn fp8_e8m0_to_f32(bits: u8) -> f32 {
    FP8_E8M0_LUT[bits as usize]
}

/// 2026-09-25: Load a dense (non-MoE) FFN's gate/up/down projections as NVFP4,
/// with transposed copies of each.
pub fn load_dense_ffn(
    store: &WeightStore,
    prefix: &str,
    gpu: &dyn GpuBackend,
    variant: Nvfp4Variant,
    absmax_k: metrale_gpu_runtime::gpu::KernelHandle,
    quantize_k: metrale_gpu_runtime::gpu::KernelHandle,
    stream: u64,
    config: &metrale_config::ModelConfig,
) -> Result<crate::layers::dense_ffn::DenseFfnWeights> {
    use crate::layers::dense_ffn::DenseFfnWeights;
    match variant {
        Nvfp4Variant::Fp8Dequanted => {
            // 2026-09-25: The FFN width is `intermediate_size`, or
            // `moe_intermediate_size` when `intermediate_size` is 0.
            let inter = if config.intermediate_size > 0 {
                config.intermediate_size
            } else {
                config.moe_intermediate_size
            };
            let h = config.hidden_size;
            let gate = quantized_from_fp8(
                store,
                &format!("{prefix}.mlp.gate_proj"),
                inter,
                h,
                gpu,
                absmax_k,
                quantize_k,
                stream,
            )?;
            let up = quantized_from_fp8(
                store,
                &format!("{prefix}.mlp.up_proj"),
                inter,
                h,
                gpu,
                absmax_k,
                quantize_k,
                stream,
            )?;
            let down = quantized_from_fp8(
                store,
                &format!("{prefix}.mlp.down_proj"),
                h,
                inter,
                gpu,
                absmax_k,
                quantize_k,
                stream,
            )?;
            // 2026-09-25: Transposed copies for the `w4a16_gemm_t*` prefill kernels.
            Ok(DenseFfnWeights {
                gate_proj: gate,
                up_proj: up,
                down_proj: down,
                gate_proj_t: Some(gate.transpose_for_gemm(gpu, inter, h)?),
                up_proj_t: Some(up.transpose_for_gemm(gpu, inter, h)?),
                down_proj_t: Some(down.transpose_for_gemm(gpu, h, inter)?),
            })
        }
        Nvfp4Variant::Bf16Raw => {
            // 2026-09-25: An unquantized BF16 FFN is quantized to NVFP4 at load
            // through `quantized_any`; `quantized_auto` refuses `Bf16Raw`.
            let inter = if config.intermediate_size > 0 {
                config.intermediate_size
            } else {
                config.moe_intermediate_size
            };
            let h = config.hidden_size;
            let qctx = QuantizeCtx {
                absmax_k,
                quantize_k,
                stream,
            };
            let gate = quantized_any(
                store,
                &format!("{prefix}.mlp.gate_proj"),
                inter,
                h,
                gpu,
                variant,
                qctx,
            )?;
            let up = quantized_any(
                store,
                &format!("{prefix}.mlp.up_proj"),
                inter,
                h,
                gpu,
                variant,
                qctx,
            )?;
            let down = quantized_any(
                store,
                &format!("{prefix}.mlp.down_proj"),
                h,
                inter,
                gpu,
                variant,
                qctx,
            )?;
            Ok(DenseFfnWeights {
                gate_proj: gate,
                up_proj: up,
                down_proj: down,
                gate_proj_t: Some(gate.transpose_for_gemm(gpu, inter, h)?),
                up_proj_t: Some(up.transpose_for_gemm(gpu, inter, h)?),
                down_proj_t: Some(down.transpose_for_gemm(gpu, h, inter)?),
            })
        }
        _ => {
            // 2026-09-25: `quantized_any` detects each key's layout, so FP8 keys
            // inside an NVFP4 checkpoint are dequantized and requantized rather
            // than failing on a missing `weight_global_scale`.
            let inter_ = if config.intermediate_size > 0 {
                config.intermediate_size
            } else {
                config.moe_intermediate_size
            };
            let h_ = config.hidden_size;
            let qctx = QuantizeCtx {
                absmax_k,
                quantize_k,
                stream,
            };
            let gate = quantized_any(
                store,
                &format!("{prefix}.mlp.gate_proj"),
                inter_,
                h_,
                gpu,
                variant,
                qctx,
            )?;
            let up = quantized_any(
                store,
                &format!("{prefix}.mlp.up_proj"),
                inter_,
                h_,
                gpu,
                variant,
                qctx,
            )?;
            let down = quantized_any(
                store,
                &format!("{prefix}.mlp.down_proj"),
                h_,
                inter_,
                gpu,
                variant,
                qctx,
            )?;
            let inter = if config.intermediate_size > 0 {
                config.intermediate_size
            } else {
                config.moe_intermediate_size
            };
            let h = config.hidden_size;
            Ok(DenseFfnWeights {
                gate_proj: gate,
                up_proj: up,
                down_proj: down,
                gate_proj_t: Some(gate.transpose_for_gemm(gpu, inter, h)?),
                up_proj_t: Some(up.transpose_for_gemm(gpu, inter, h)?),
                down_proj_t: Some(down.transpose_for_gemm(gpu, h, inter)?),
            })
        }
    }
}

/// 2026-09-25: `load_mtp` under the Qwen3.5 name.
#[allow(dead_code)]
pub(crate) fn load_mtp_qwen35(
    store: &WeightStore,
    num_experts: usize,
    gpu: &dyn GpuBackend,
    variant: Nvfp4Variant,
) -> Result<MtpWeights> {
    load_mtp(store, num_experts, gpu, variant)
}

/// 2026-09-25: GPU-concatenate two BF16 matrices row-wise: [A; B] →
/// [A_rows + B_rows, K], in a new buffer. Both inputs must be contiguous with
/// the same K.
pub fn gpu_concat_rows(
    a: &DenseWeight,
    a_rows: usize,
    b: &DenseWeight,
    b_rows: usize,
    k: usize,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    let a_bytes = a_rows * k * 2;
    let b_bytes = b_rows * k * 2;
    let total = a_bytes + b_bytes;
    let buf = gpu.alloc(total)?;
    tracing::debug!(
        target: "metrale_model_layers::weight_map::concat",
        a_rows, b_rows, k, a_bytes, b_bytes, total,
        a = a.weight.0, b = b.weight.0, buf = buf.0,
        "gpu concat rows [a;b]"
    );
    gpu.copy_d2d(a.weight, buf, a_bytes)?;
    gpu.copy_d2d(b.weight, buf.offset(a_bytes), b_bytes)?;
    Ok(DenseWeight { weight: buf })
}

/// 2026-09-25: Interleave A (alpha) and B (beta) rows on the host into the BA
/// layout `dense_gemv_ba_gates` reads, and upload it.
///
/// Per group of `vpg = nv / nk` value heads: the group's `vpg` beta rows,
/// then its `vpg` alpha rows. A and B are `[nv, K]` BF16; the result is
/// `[2*nv, K]` BF16 on the GPU.
pub fn interleave_ba(
    a_weight: &DenseWeight,
    b_weight: &DenseWeight,
    nv: usize,
    nk: usize,
    k: usize,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    let vpg = nv / nk;
    let row_bytes = k * 2;
    let ba_size = nv * 2;

    let mut a_cpu = vec![0u8; nv * row_bytes];
    let mut b_cpu = vec![0u8; nv * row_bytes];
    gpu.copy_d2h(a_weight.weight, &mut a_cpu)?;
    gpu.copy_d2h(b_weight.weight, &mut b_cpu)?;

    let mut ba_cpu = vec![0u8; ba_size * row_bytes];
    for g in 0..nk {
        for v in 0..vpg {
            let vh = g * vpg + v;
            let dst_row = g * (2 * vpg) + v;
            ba_cpu[dst_row * row_bytes..(dst_row + 1) * row_bytes]
                .copy_from_slice(&b_cpu[vh * row_bytes..(vh + 1) * row_bytes]);
            let dst_row = g * (2 * vpg) + vpg + v;
            ba_cpu[dst_row * row_bytes..(dst_row + 1) * row_bytes]
                .copy_from_slice(&a_cpu[vh * row_bytes..(vh + 1) * row_bytes]);
        }
    }

    let buf = gpu.alloc(ba_size * row_bytes)?;
    gpu.copy_h2d(&ba_cpu, buf)?;
    Ok(DenseWeight { weight: buf })
}
