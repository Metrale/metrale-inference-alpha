// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: FP8 block-scaled weight loading, BF16-to-NVFP4 quantization, and the attention, SSM and MoE layer loaders.
//!
//! Owner: model-layers (weight loading).
//! Invariants:
//! - Every `Fp8Weight` returned here holds its block scale as an FP32 device buffer.

#![allow(unused_imports)]

use anyhow::{Context, Result, bail, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_weights::weights::{WeightDtype, WeightStore};

use super::*;

/// 2026-09-25: Load an FP8 E4M3 block-scaled checkpoint weight as a native [`Fp8Weight`].
///
/// `{prefix}.weight` must be a 2D FP8E4M3 tensor `[N, K]`. The block scale is
/// read from the first of these that exists: `{prefix}.weight_scale_inv`, a 2D
/// `{prefix}.weight_scale`, or a 2D F8_E8M0 `{prefix}.scale`. The scale must be
/// BF16, FP32 or F8_E8M0, and it is widened once into an FP32 device buffer.
/// When none exists, a scalar `{prefix}.weight_scale` is repeated over a
/// `[ceil(N/128), ceil(K/128)]` FP32 grid. `w8a16_gemv` reads this buffer as
/// `const float*` and applies `block_scale[n/128, k/128]` to each weight.
pub fn load_fp8_block_scaled_as_fp8weight(
    store: &WeightStore,
    prefix: &str,
    gpu: &dyn GpuBackend,
) -> Result<Fp8Weight> {
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
    let weight_ptr = w.ptr;

    let scale_inv_key = format!("{prefix}.weight_scale_inv");
    let plain_scale_key = format!("{prefix}.weight_scale");
    let e8m0_scale_key = format!("{prefix}.scale");
    let block_scale_key = if store.contains(&scale_inv_key) {
        Some(scale_inv_key.clone())
    } else if store
        .get(&plain_scale_key)
        .map(|s| s.shape.len() == 2)
        .unwrap_or(false)
    {
        Some(plain_scale_key.clone())
    } else if store
        .get(&e8m0_scale_key)
        .map(|s| s.shape.len() == 2 && s.dtype == WeightDtype::FP8E8M0)
        .unwrap_or(false)
    {
        Some(e8m0_scale_key.clone())
    } else {
        None
    };
    let row_scale = if let Some(scale_key) = block_scale_key {
        let s = store.get(&scale_key)?;
        ensure!(
            s.shape.len() == 2,
            "Expected 2D shape for {scale_key}, got {:?}",
            s.shape,
        );
        ensure!(
            matches!(
                s.dtype,
                WeightDtype::BF16 | WeightDtype::FP32 | WeightDtype::FP8E8M0
            ),
            "Expected BF16, FP32, or F8_E8M0 for {scale_key}, got {:?}",
            s.dtype,
        );

        tracing::debug!(
            "FP8 block scales: {prefix} [{n},{k}] scale=[{},{}] dtype={:?} -> FP32",
            s.shape[0],
            s.shape[1],
            s.dtype,
        );

        // 2026-09-25: Widen on the device so the kernels read one FP32 layout
        // whatever dtype the checkpoint stored the scale in.
        let scale_total = s.shape[0] * s.shape[1];
        let row_scale = gpu.alloc(scale_total * 4)?;
        let kernel = gpu.kernel("widen_block_scale_f32", "widen_block_scale_f32")?;
        let stream = gpu.default_stream();
        let input_dtype = match s.dtype {
            WeightDtype::BF16 => 0,
            WeightDtype::FP32 => 1,
            WeightDtype::FP8E8M0 => 2,
            _ => unreachable!("validated block-scale dtype"),
        };
        crate::layers::ops::widen_block_scale_f32(
            gpu,
            kernel,
            s.ptr,
            row_scale,
            scale_total as u32,
            input_dtype,
            stream,
        )?;
        gpu.synchronize(stream)?;
        row_scale
    } else {
        let scalar_key = plain_scale_key;
        let scale = scalar_f32(store, &scalar_key, gpu)
            .with_context(|| format!("Missing {scale_inv_key} or scalar {scalar_key}"))?;
        let n_blocks = n.div_ceil(128);
        let k_blocks = k.div_ceil(128);
        let scale_total = n_blocks * k_blocks;
        tracing::debug!(
            "FP8 scalar scale: {prefix} [{n},{k}] scale={scale:.8} -> [{n_blocks},{k_blocks}] FP32"
        );
        let mut scale_buf = Vec::with_capacity(scale_total * 4);
        for _ in 0..scale_total {
            scale_buf.extend_from_slice(&scale.to_le_bytes());
        }
        let ptr = gpu.alloc(scale_buf.len())?;
        gpu.copy_h2d(&scale_buf, ptr)?;
        ptr
    };

    Ok(Fp8Weight {
        weight: weight_ptr,
        row_scale,
        n: n as u32,
        k: k as u32,
        scale_format: WeightQuantFormat::Fp8BlockScaled,
    })
}

/// 2026-09-25: Quantize a BF16 dense `[n, k]` weight to NVFP4 on the GPU.
///
/// Two kernels: `absmax_kernel` finds the global absolute max, then
/// `quantize_kernel` writes packed E2M1 (`n*k/2` bytes) and one scale byte per
/// 16 weights. The global scale is `global_max / (6 * 448)`, or 1.0 for an
/// all-zero weight. It synchronizes `stream` after each kernel, so it belongs
/// at load time, not in a forward pass.
pub fn quantize_to_nvfp4(
    bf16_weight: &DenseWeight,
    n: usize,
    k: usize,
    gpu: &dyn GpuBackend,
    absmax_kernel: metrale_gpu_runtime::gpu::KernelHandle,
    quantize_kernel: metrale_gpu_runtime::gpu::KernelHandle,
    stream: u64,
) -> Result<QuantizedWeight> {
    use metrale_gpu_runtime::kernel_args::KernelLaunch;
    use std::sync::atomic::{AtomicU64, Ordering};

    static T_ALLOC_MAX: AtomicU64 = AtomicU64::new(0);
    static T_LAUNCH1: AtomicU64 = AtomicU64::new(0);
    static T_SYNC1: AtomicU64 = AtomicU64::new(0);
    static T_D2H: AtomicU64 = AtomicU64::new(0);
    static T_ALLOC_OUT: AtomicU64 = AtomicU64::new(0);
    static T_LAUNCH2: AtomicU64 = AtomicU64::new(0);
    static T_SYNC2: AtomicU64 = AtomicU64::new(0);
    static N_CALLS: AtomicU64 = AtomicU64::new(0);

    let total = n * k;

    let t = std::time::Instant::now();
    let max_buf = gpu.alloc(4)?;
    gpu.memset(max_buf, 0, 4)?;
    T_ALLOC_MAX.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);

    let t = std::time::Instant::now();
    let grid1 = (total / 256).clamp(1, 1024) as u32;
    KernelLaunch::new(gpu, absmax_kernel)
        .grid([grid1, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(bf16_weight.weight)
        .arg_ptr(max_buf)
        .arg_u32(total as u32)
        .launch(stream)?;
    T_LAUNCH1.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);

    let t = std::time::Instant::now();
    gpu.synchronize(stream)?;
    T_SYNC1.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    let t = std::time::Instant::now();
    let mut max_bytes = [0u8; 4];
    gpu.copy_d2h(max_buf, &mut max_bytes)?;
    T_D2H.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    let global_max = f32::from_le_bytes(max_bytes);

    // 2026-09-25: 6.0 is the E2M1 maximum and 448.0 the E4M3 maximum.
    let scale2 = if global_max > 0.0 {
        global_max / (6.0 * 448.0)
    } else {
        1.0
    };

    // 2026-09-25: The first 5 calls on this backend log their absmax
    // (`OpCache::first_n` counts per backend).
    if gpu.op_cache().first_n("diag:quantize_nvfp4_absmax", 5) {
        tracing::info!(
            "quantize_to_nvfp4: n={n} k={k} total={total} global_max={global_max:.6} scale2={scale2:.8} grid1={grid1}",
        );
    }

    let t = std::time::Instant::now();
    let packed_buf = gpu.alloc(n * k / 2)?;
    let scale_buf = gpu.alloc(n * k / 16)?;
    T_ALLOC_OUT.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);

    let t = std::time::Instant::now();
    KernelLaunch::new(gpu, quantize_kernel)
        .grid([n as u32, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(bf16_weight.weight)
        .arg_ptr(packed_buf)
        .arg_ptr(scale_buf)
        .arg_f32(scale2)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .launch(stream)?;
    T_LAUNCH2.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);

    let t = std::time::Instant::now();
    gpu.synchronize(stream)?;
    T_SYNC2.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);

    let c = N_CALLS.fetch_add(1, Ordering::Relaxed) + 1;
    if c.is_multiple_of(512) {
        let ms = |a: &AtomicU64| a.load(Ordering::Relaxed) as f64 / 1.0e6;
        tracing::info!(
            "quantize_to_nvfp4 PROFILE after {c} calls (ms total): alloc_max={:.1} launch1={:.1} \
             sync1={:.1} d2h={:.1} alloc_out={:.1} launch2={:.1} sync2={:.1} | sum={:.1} \
             per_call={:.3}ms",
            ms(&T_ALLOC_MAX),
            ms(&T_LAUNCH1),
            ms(&T_SYNC1),
            ms(&T_D2H),
            ms(&T_ALLOC_OUT),
            ms(&T_LAUNCH2),
            ms(&T_SYNC2),
            ms(&T_ALLOC_MAX)
                + ms(&T_LAUNCH1)
                + ms(&T_SYNC1)
                + ms(&T_D2H)
                + ms(&T_ALLOC_OUT)
                + ms(&T_LAUNCH2)
                + ms(&T_SYNC2),
            (ms(&T_ALLOC_MAX)
                + ms(&T_LAUNCH1)
                + ms(&T_SYNC1)
                + ms(&T_D2H)
                + ms(&T_ALLOC_OUT)
                + ms(&T_LAUNCH2)
                + ms(&T_SYNC2))
                / c as f64,
        );
    }

    Ok(QuantizedWeight {
        weight: packed_buf,
        weight_scale: scale_buf,
        weight_scale_2: scale2,
        input_scale: DevicePtr::NULL,
        weight_scale_2_vec: DevicePtr::NULL,
    })
}

/// 2026-09-25: Load the `{layer_prefix}.self_attn` weights of a full-attention layer.
pub fn load_attention(
    store: &WeightStore,
    layer_prefix: &str,
    gpu: &dyn GpuBackend,
    variant: Nvfp4Variant,
    qctx: QuantizeCtx,
    config: &metrale_config::ModelConfig,
) -> Result<AttentionWeights> {
    let p = format!("{layer_prefix}.self_attn");
    let (k_scale, v_scale) = load_kv_scales(store, &p, gpu);
    let h = config.hidden_size;
    let qkv_out = config.num_attention_heads * config.head_dim;
    // 2026-09-25: q/k/v are `DenseWeight`, so an NVFP4 `.weight_packed`
    // projection is dequantized to BF16 here; otherwise `.weight` goes through
    // `dense_auto`. The dims come from the packed tensor `[out, in/2]`, not
    // from config, because a gated q_proj carries the gate rows as well.
    let load_qkv = |name: &str| -> Result<DenseWeight> {
        match store.get(&format!("{p}.{name}.weight_packed")) {
            Ok(w) => crate::weight_map::dequant_nvfp4_to_bf16(
                store,
                &format!("{p}.{name}"),
                w.shape[0],
                w.shape[1] * 2,
                gpu,
            ),
            Err(_) => dense_auto(store, &format!("{p}.{name}.weight"), gpu),
        }
    };
    Ok(AttentionWeights {
        q_proj: load_qkv("q_proj")?,
        k_proj: load_qkv("k_proj")?,
        v_proj: load_qkv("v_proj")?,
        o_proj: quantized_any(
            store,
            &format!("{p}.o_proj"),
            h,
            qkv_out,
            gpu,
            variant,
            qctx,
        )?,
        q_norm: dense(store, &format!("{p}.q_norm.weight"))?,
        k_norm: dense(store, &format!("{p}.k_norm.weight"))?,
        q_norm_full: None,
        k_norm_full: None,
        k_scale,
        v_scale,
    })
}

/// 2026-09-25: Load the `{layer_prefix}.linear_attn` weights of a linear-attention layer.
pub fn load_ssm(
    store: &WeightStore,
    layer_prefix: &str,
    gpu: &dyn GpuBackend,
    variant: Nvfp4Variant,
    qctx: QuantizeCtx,
    config: &metrale_config::ModelConfig,
) -> Result<SsmWeights> {
    let p = format!("{layer_prefix}.linear_attn");
    let h = config.hidden_size;
    let d_inner = config.linear_value_head_dim * config.linear_num_value_heads;
    Ok(SsmWeights {
        in_proj_qkvz: dense_auto(store, &format!("{p}.in_proj_qkvz.weight"), gpu)?,
        in_proj_ba: dense_auto(store, &format!("{p}.in_proj_ba.weight"), gpu)?,
        conv1d: dense(store, &format!("{p}.conv1d.weight"))?,
        a_log: dense_keep_f32(store, &format!("{p}.A_log"), gpu)?,
        dt_bias: dense_keep_f32(store, &format!("{p}.dt_bias"), gpu)?,
        norm: dense(store, &format!("{p}.norm.weight"))?,
        out_proj: quantized_any(
            store,
            &format!("{p}.out_proj"),
            h,
            d_inner,
            gpu,
            variant,
            qctx,
        )?,
    })
}

/// 2026-09-25: Load the `{layer_prefix}.mlp` MoE weights of a layer.
///
/// Only experts for which `config.is_local_expert(e)` holds are loaded; the
/// others are `ExpertWeight::null()`.
pub fn load_moe(
    store: &WeightStore,
    layer_prefix: &str,
    num_experts: usize,
    gpu: &dyn GpuBackend,
    config: &metrale_config::ModelConfig,
    variant: Nvfp4Variant,
    qctx: QuantizeCtx,
) -> Result<MoeWeights> {
    load_moe_inner(
        store,
        layer_prefix,
        num_experts,
        gpu,
        config,
        variant,
        qctx,
        false,
    )
}

/// 2026-09-25: [`load_moe`] with every routed expert left as `ExpertWeight::null()`.
/// The native-FP8 qwen3 loader uses it and loads the routed experts separately.
pub fn load_moe_skip_experts(
    store: &WeightStore,
    layer_prefix: &str,
    num_experts: usize,
    gpu: &dyn GpuBackend,
    config: &metrale_config::ModelConfig,
    variant: Nvfp4Variant,
    qctx: QuantizeCtx,
) -> Result<MoeWeights> {
    load_moe_inner(
        store,
        layer_prefix,
        num_experts,
        gpu,
        config,
        variant,
        qctx,
        true,
    )
}
