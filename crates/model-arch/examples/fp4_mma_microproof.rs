// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: A hand-written block-scaled FP4 MMA GEMM against the CUTLASS NVFP4 GEMM.
//!
//! Owner: model-arch examples.
//! Invariants:
//! - Returns an error unless the cosine of the two BF16 outputs is >= 0.999 at M=64,
//!   N=1024, K=2048.
//!
//! The kernel module `fp4_mma_microtest` (kernels/gb10/holo-3.1-0.8b/nvfp4) runs one
//!   mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3
//! per 16x8 output tile, looped over K, after `fp4_microtest_pack` quantises A and B. The
//! oracle is `metrale_gpu_runtime::cutlass::nvfp4_gemm_bf16_act_weight_t`.
//!
//! Build:
//!   cargo build --release -p metrale-model-arch --example fp4_mma_microproof \
//!     --no-default-features --features "cuda gpu-examples"
//! Run: target/release/examples/fp4_mma_microproof

use anyhow::{Result, bail};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / ((1u64 << 24) as f32)
    }
    fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.unit()
    }
}

fn f32_to_bf16_bits(f: f32) -> u16 {
    let bits = f.to_bits();
    if (bits & 0x7FFF_FFFF) > 0x7F80_0000 {
        return ((bits >> 16) | 0x0040) as u16;
    }
    let rounding_bias = 0x7FFF + ((bits >> 16) & 1);
    (bits.wrapping_add(rounding_bias) >> 16) as u16
}
fn bf16_bits_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

fn u16s_to_le(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn upload_bytes(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}
fn read_bf16(gpu: &dyn GpuBackend, ptr: DevicePtr, m: usize, n: usize) -> Result<Vec<u16>> {
    let mut raw = vec![0u8; m * n * 2];
    gpu.copy_d2h(ptr, &mut raw)?;
    Ok(raw
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
}

fn cosine_u16(a: &[u16], b: &[u16]) -> f64 {
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for i in 0..a.len() {
        let x = bf16_bits_to_f32(a[i]) as f64;
        let y = bf16_bits_to_f32(b[i]) as f64;
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na > 0.0 && nb > 0.0 {
        dot / (na.sqrt() * nb.sqrt())
    } else {
        0.0
    }
}

fn main() -> Result<()> {
    let (m, n, k) = (64usize, 1024usize, 2048usize);
    let seed = 0x_5151_A7A7u64;
    println!("=== Phase-1 hand-rolled Sm120 FP4 MMA microproof ===");
    println!("M={m} N={n} K={k}; oracle = nvfp4_gemm_bf16_act_weight_t (same Sm120 FP4 math)");

    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.create_stream()?;

    let mut rng = Rng(seed);
    let a_bf16: Vec<u16> = (0..m * k)
        .map(|_| f32_to_bf16_bits(rng.uniform(-1.0, 1.0)))
        .collect();
    let b_bf16: Vec<u16> = (0..n * k)
        .map(|_| f32_to_bf16_bits(rng.uniform(-0.5, 0.5)))
        .collect();

    let a_ptr = upload_bytes(gpu, &u16s_to_le(&a_bf16))?;
    let b_ptr = upload_bytes(gpu, &u16s_to_le(&b_bf16))?;

    // 2026-09-25: Oracle. B is packed once: nibbles [N, K/2], E4M3 scales [K/16, N].
    let packed_len = (k / 2) * n;
    let scale_len = (k / 16) * n;
    let packed_ptr = gpu.alloc(packed_len)?;
    let scale_ptr = gpu.alloc(scale_len)?;
    metrale_gpu_runtime::cutlass::pack_bf16_weight_to_nvfp4_t(
        b_ptr.0,
        packed_ptr.0,
        scale_ptr.0,
        n as u32,
        k as u32,
        stream,
    )?;
    gpu.synchronize(stream)?;

    let out_oracle = gpu.alloc(m * n * 2)?;
    metrale_gpu_runtime::cutlass::nvfp4_gemm_bf16_act_weight_t(
        a_ptr.0,
        packed_ptr.0,
        scale_ptr.0,
        1.0,
        out_oracle.0,
        m as u32,
        n as u32,
        k as u32,
        stream,
    )?;
    gpu.synchronize(stream)?;
    let c_oracle = read_bf16(gpu, out_oracle, m, n)?;

    // 2026-09-25: The hand-written path packs A and B row-major: nibbles [rows, K/2],
    // scales [rows, K/16].
    let pack_handle = gpu.kernel("fp4_mma_microtest", "fp4_microtest_pack")?;
    let mma_handle = gpu.kernel("fp4_mma_microtest", "fp4_microtest_mma")?;

    let a_packed = gpu.alloc((k / 2) * m)?;
    let a_scales = gpu.alloc((k / 16) * m)?;
    let b_packed = gpu.alloc((k / 2) * n)?;
    let b_scales = gpu.alloc((k / 16) * n)?;

    let groups = (k / 16) as u32;
    let pack_a = || -> Result<()> {
        KernelLaunch::new(gpu, pack_handle)
            .grid([m as u32, div_ceil(groups, 128), 1])
            .block([128, 1, 1])
            .arg_ptr(a_ptr)
            .arg_ptr(a_packed)
            .arg_ptr(a_scales)
            .arg_u32(m as u32)
            .arg_u32(k as u32)
            .launch(stream)?;
        Ok(())
    };
    let pack_b = || -> Result<()> {
        KernelLaunch::new(gpu, pack_handle)
            .grid([n as u32, div_ceil(groups, 128), 1])
            .block([128, 1, 1])
            .arg_ptr(b_ptr)
            .arg_ptr(b_packed)
            .arg_ptr(b_scales)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .launch(stream)?;
        Ok(())
    };
    pack_a()?;
    pack_b()?;
    gpu.synchronize(stream)?;

    let out_mma = gpu.alloc(m * n * 2)?;
    let mma_launch = || -> Result<()> {
        KernelLaunch::new(gpu, mma_handle)
            .grid([div_ceil(n as u32, 8), div_ceil(m as u32, 16), 1])
            .block([32, 1, 1])
            .arg_ptr(a_packed)
            .arg_ptr(a_scales)
            .arg_ptr(b_packed)
            .arg_ptr(b_scales)
            .arg_ptr(out_mma)
            .arg_u32(m as u32)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .launch(stream)?;
        Ok(())
    };
    mma_launch()?;
    gpu.synchronize(stream)?;
    let c_mma = read_bf16(gpu, out_mma, m, n)?;

    let cos = cosine_u16(&c_mma, &c_oracle);
    println!();
    println!("hand-rolled MMA  vs  CUTLASS collective  cosine = {cos:.6}");
    println!(
        "sample [0][0..4]: mma={:?} oracle={:?}",
        &c_mma[0..4]
            .iter()
            .map(|&b| bf16_bits_to_f32(b))
            .collect::<Vec<_>>(),
        &c_oracle[0..4]
            .iter()
            .map(|&b| bf16_bits_to_f32(b))
            .collect::<Vec<_>>()
    );
    println!(
        "sample [1][0..4]: mma={:?} oracle={:?}",
        &c_mma[n..n + 4]
            .iter()
            .map(|&b| bf16_bits_to_f32(b))
            .collect::<Vec<_>>(),
        &c_oracle[n..n + 4]
            .iter()
            .map(|&b| bf16_bits_to_f32(b))
            .collect::<Vec<_>>()
    );

    for p in [
        a_ptr, b_ptr, packed_ptr, scale_ptr, out_oracle, a_packed, a_scales, b_packed, b_scales,
        out_mma,
    ] {
        gpu.free(p).ok();
    }

    if cos >= 0.999 {
        println!("\nPASS: cos {cos:.6} >= 0.999");
        Ok(())
    } else {
        bail!("FAIL: cos {cos:.6} < 0.999");
    }
}
