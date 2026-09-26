// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `run_shape` and its `ShapeResult` for `w4a16_bf16_v2_microtest`.
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.

use crate::*;

pub(crate) struct ShapeResult {
    /// 2026-09-25: Base `w4a16_gemm` against v2 (`w4a16_gemm_t_m128_bf16_v2`);
    /// `main` gates its cosine.
    pub(crate) base_vs_v2: Stats,
    /// 2026-09-25: Crush v1 (`w4a16::w4a16_gemm_t_m128`) against crush v2
    /// (`w4a16_v2::w4a16_gemm_t_m128_v2`); `main` requires at least 99.9999% of
    /// the outputs bit-identical. `None` when the target has no v2 kernel and
    /// the legs are skipped.
    pub(crate) crush_v1_vs_v2: Option<Stats>,
    /// 2026-09-25: v1 (`w4a16_gemm_t_m128_bf16`) against v2; `main` requires at
    /// least 99.9999% of the outputs bit-identical.
    pub(crate) v1_vs_v2: Stats,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_shape(
    gpu: &dyn GpuBackend,
    stream: u64,
    base_h: metrale_gpu_runtime::gpu::KernelHandle,
    bf16_h: metrale_gpu_runtime::gpu::KernelHandle,
    v2_h: metrale_gpu_runtime::gpu::KernelHandle,
    crush1_h: metrale_gpu_runtime::gpu::KernelHandle,
    crush2_h: metrale_gpu_runtime::gpu::KernelHandle,
    seed: u64,
    m: usize,
    n: usize,
    k: usize,
) -> Result<ShapeResult> {
    let mut rng = Rng(seed ^ ((m as u64) << 32) ^ ((n as u64) << 16) ^ (k as u64));

    // 2026-09-25: A [M, K] BF16: uniform draws in [-1, 1), rounded to BF16.
    let a_bf16: Vec<u16> = (0..m * k)
        .map(|_| f32_to_bf16_bits(rng.uniform(-1.0, 1.0)))
        .collect();
    let a_ptr = upload(gpu, &u16s_to_le(&a_bf16))?;

    let w = gen_weight(&mut rng, n, k);

    let packed_nt = upload(gpu, &w.packed_nt)?;
    let scale_nt = upload(gpu, &w.scale_nt)?;
    // 2026-09-25: Negative control: with METRALE_MICROTEST_NEGCTL set to any value, the
    // transposed-layout kernels get the non-transposed packed weights and scales, which a
    // discriminating comparison must reject.
    let neg_ctl = std::env::var_os("METRALE_MICROTEST_NEGCTL").is_some();
    let packed_t = if neg_ctl {
        upload(gpu, &w.packed_nt)?
    } else {
        upload(gpu, &w.packed_t)?
    };
    let scale_t = if neg_ctl {
        upload(gpu, &w.scale_nt)?
    } else {
        upload(gpu, &w.scale_t)?
    };

    let c_base = gpu.alloc(m * n * 2)?;
    let c_bf16 = gpu.alloc(m * n * 2)?;
    let c_v2 = gpu.alloc(m * n * 2)?;

    KernelLaunch::new(gpu, base_h)
        .grid([n.div_ceil(64) as u32, m.div_ceil(64) as u32, 1])
        .block([128, 1, 1])
        .arg_ptr(a_ptr)
        .arg_ptr(packed_nt)
        .arg_ptr(scale_nt)
        .arg_f32(w.scale2)
        .arg_ptr(c_base)
        .arg_u32(m as u32)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .launch(stream)?;

    KernelLaunch::new(gpu, bf16_h)
        .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1])
        .block([128, 1, 1])
        .arg_ptr(a_ptr)
        .arg_ptr(packed_t)
        .arg_ptr(scale_t)
        .arg_f32(w.scale2)
        .arg_ptr(c_bf16)
        .arg_u32(m as u32)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .launch(stream)?;

    // 2026-09-25: v2 takes v1's launch and weight layout plus a ninth argument, `ldb`, the
    // transposed-B row stride (N here).
    KernelLaunch::new(gpu, v2_h)
        .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1])
        .block([128, 1, 1])
        .arg_ptr(a_ptr)
        .arg_ptr(packed_t)
        .arg_ptr(scale_t)
        .arg_f32(w.scale2)
        .arg_ptr(c_v2)
        .arg_u32(m as u32)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .arg_u32(n as u32)
        .launch(stream)?;

    // 2026-09-25: The crush pair runs on the same transposed weights and is gated on bit
    // identity with each other, not on closeness to base.
    let (c_crush1, c_crush2) = if crush1_h.0 != 0 && crush2_h.0 != 0 {
        let c1 = gpu.alloc(m * n * 2)?;
        let c2 = gpu.alloc(m * n * 2)?;
        KernelLaunch::new(gpu, crush1_h)
            .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1])
            .block([128, 1, 1])
            .arg_ptr(a_ptr)
            .arg_ptr(packed_t)
            .arg_ptr(scale_t)
            .arg_f32(w.scale2)
            .arg_ptr(c1)
            .arg_u32(m as u32)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .launch(stream)?;
        KernelLaunch::new(gpu, crush2_h)
            .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(a_ptr)
            .arg_ptr(packed_t)
            .arg_ptr(scale_t)
            .arg_f32(w.scale2)
            .arg_ptr(c2)
            .arg_u32(m as u32)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .launch(stream)?;
        (Some(c1), Some(c2))
    } else {
        (None, None)
    };

    gpu.synchronize(stream)?;

    let mut raw_base = vec![0u8; m * n * 2];
    let mut raw_bf16 = vec![0u8; m * n * 2];
    let mut raw_v2 = vec![0u8; m * n * 2];
    gpu.copy_d2h(c_base, &mut raw_base)?;
    gpu.copy_d2h(c_bf16, &mut raw_bf16)?;
    gpu.copy_d2h(c_v2, &mut raw_v2)?;
    let to_u16 = |raw: &[u8]| -> Vec<u16> {
        raw.chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect()
    };
    let out_base = to_u16(&raw_base);
    let out_bf16 = to_u16(&raw_bf16);
    let out_v2 = to_u16(&raw_v2);

    // 2026-09-25: An all-zero base or v2 output fails the shape.
    let base_nz = out_base.iter().filter(|&&x| x != 0).count();
    let v2_nz = out_v2.iter().filter(|&&x| x != 0).count();
    if base_nz == 0 || v2_nz == 0 {
        bail!("dead output: base_nonzero={base_nz} v2_nonzero={v2_nz} (M={m} N={n} K={k})");
    }

    let base_vs_v2 = compare(&out_base, &out_v2);
    let v1_vs_v2 = compare(&out_bf16, &out_v2);
    let crush_v1_vs_v2 = if let (Some(c1), Some(c2)) = (c_crush1, c_crush2) {
        let mut raw1 = vec![0u8; m * n * 2];
        let mut raw2 = vec![0u8; m * n * 2];
        gpu.copy_d2h(c1, &mut raw1)?;
        gpu.copy_d2h(c2, &mut raw2)?;
        let o1 = to_u16(&raw1);
        let o2 = to_u16(&raw2);
        let nz1 = o1.iter().filter(|&&x| x != 0).count();
        if nz1 == 0 {
            bail!("dead crush-v1 output (M={m} N={n} K={k})");
        }
        let _ = gpu.free(c1);
        let _ = gpu.free(c2);
        Some(compare(&o1, &o2))
    } else {
        None
    };

    for p in [
        a_ptr, packed_nt, scale_nt, packed_t, scale_t, c_base, c_bf16, c_v2,
    ] {
        let _ = gpu.free(p);
    }

    Ok(ShapeResult {
        base_vs_v2,
        v1_vs_v2,
        crush_v1_vs_v2,
    })
}
