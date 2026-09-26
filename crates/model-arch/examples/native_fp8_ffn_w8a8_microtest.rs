// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Dense-FFN W8A8 block-scaled prefill GEMMs at Qwen3.8-27B shapes:
//! the in-tree `fp8_gemm_t_blockscaled` kernel and, with
//! `METRALE_CUBLAS_GEMM=1`, cuBLASLt, against the W8A16
//! `w8a16_gemm_pipelined` reference.
//!
//! Owner: model-arch examples (GPU microtests).
//! Invariants:
//! - The run exits 0 only if, for both shapes at every M in `BATCHES`, the
//!   W8A8 kernel meets `COSINE_GATE` and the rel_rms gate against W8A16, and,
//!   when cuBLASLt runs, it has no element outside the one-BF16-ULP bound
//!   against the kernel, meets `CUBLAS_COSINE_GATE` and `CUBLAS_REL_RMS_GATE`
//!   against it, and meets `COSINE_GATE` against W8A16.
//!
//! Reported per shape and M:
//!   * W8A8 kernel vs W8A16: max_abs, cosine, relative RMS. W8A8 quantizes the
//!     activation to E4M3 per token and 128-wide K group, which W8A16 does
//!     not, so this comparison is a tolerance. `METRALE_W8A8_REL_RMS_GATE`
//!     overrides the rel_rms gate (`REL_RMS_GATE` otherwise); the value used
//!     is printed.
//!   * cuBLASLt vs the W8A8 kernel, on the same quantized activation: the
//!     per-element bound and the reported `sign_flips`, `unequal_bf16` and
//!     ordinal `max_ulp` are defined in `common/native_fp8_bf16_compare.rs`.
//!   * TFLOP/s for each path, from CUDA events over `ITERS` launches after
//!     `WARMUP`.
//!
//! Scale layout: `METRALE_CUBLAS_SCALE_LAYOUT=rowmajor` feeds cuBLASLt the
//! quantizer's `[M, K/128]` scales untransposed; any other value (or none)
//! transposes them to `[K/128, M_pad]` first, the layout cuBLASLt reads. The
//! `rowmajor` run is a control that is expected to fail.
//!
//! Run:
//!   cargo run --release -p metrale-model-arch --features cuda,gpu-examples \
//!     --example native_fp8_ffn_w8a8_microtest
//!   METRALE_CUBLAS_GEMM=1 cargo run --release -p metrale-model-arch \
//!     --features cuda,gpu-examples --example native_fp8_ffn_w8a8_microtest
//!   METRALE_CUBLAS_GEMM=1 METRALE_CUBLAS_SCALE_LAYOUT=rowmajor cargo run --release \
//!     -p metrale-model-arch --features cuda,gpu-examples \
//!     --example native_fp8_ffn_w8a8_microtest

use anyhow::{Result, bail};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_layers::layers::ops;
use metrale_model_layers::weight_map::{Fp8Weight, WeightQuantFormat};

#[path = "common/native_fp8_bf16_compare.rs"]
pub(crate) mod native_fp8_bf16_compare;
use native_fp8_bf16_compare::{
    CUBLAS_COSINE_GATE, CUBLAS_REL_RMS_GATE, CUBLAS_SMALL_MAGNITUDE, compare,
};

// 2026-09-25: CUDA driver event API, for GPU-side timing. The signatures are
// the ones `examples/w8a16_microtest.rs` declares.
unsafe extern "C" {
    fn cuEventCreate(event: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(event: u64, stream: u64) -> i32;
    fn cuEventSynchronize(event: u64) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: u64, end: u64) -> i32;
    fn cuEventDestroy_v2(event: u64) -> i32;
}

/// 2026-09-25: Qwen3.8-27B dense FFN, `hidden_dim` 5120 and
/// `intermediate_size` 17408 (`kernels/hopper/qwen3.8-27b/MODEL.toml`).
const H: usize = 5120;
const INTER: usize = 17408;
const BATCHES: [usize; 2] = [64, 1193];
const BLOCK: usize = 128;
const ITERS: u32 = 10;
const WARMUP: u32 = 3;
const COSINE_GATE: f64 = 0.999;
const REL_RMS_GATE: f64 = 0.02;

struct Rng(u64);
impl Rng {
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.0 >> 32) as u32
    }
    /// 2026-09-25: Uniform in [-1, 1], in steps of 1/1024.
    fn unit(&mut self) -> f32 {
        (self.next_u32() % 2049) as f32 / 1024.0 - 1.0
    }
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len().max(1))?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}

fn download_bf16(gpu: &dyn GpuBackend, ptr: DevicePtr, elems: usize) -> Result<Vec<u16>> {
    let mut raw = vec![0u8; elems * 2];
    gpu.copy_d2h(ptr, &mut raw)?;
    Ok(raw
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
}

/// 2026-09-25: GPU time per iteration for `launch`, in seconds (CUDA events,
/// no host sync between iterations).
fn time_gpu(
    gpu: &dyn GpuBackend,
    stream: u64,
    mut launch: impl FnMut() -> Result<()>,
) -> Result<f64> {
    for _ in 0..WARMUP {
        launch()?;
    }
    gpu.synchronize(stream)?;
    let (mut ev_start, mut ev_end) = (0u64, 0u64);
    for (ev, what) in [(&mut ev_start, "start"), (&mut ev_end, "end")] {
        let rc = unsafe { cuEventCreate(ev, 0) };
        if rc != 0 {
            bail!("cuEventCreate({what}) failed: status {rc}");
        }
    }
    if unsafe { cuEventRecord(ev_start, stream) } != 0 {
        bail!("cuEventRecord(start) failed");
    }
    for _ in 0..ITERS {
        launch()?;
    }
    if unsafe { cuEventRecord(ev_end, stream) } != 0 {
        bail!("cuEventRecord(end) failed");
    }
    if unsafe { cuEventSynchronize(ev_end) } != 0 {
        bail!("cuEventSynchronize failed");
    }
    let mut ms: f32 = 0.0;
    if unsafe { cuEventElapsedTime(&mut ms, ev_start, ev_end) } != 0 {
        bail!("cuEventElapsedTime failed");
    }
    unsafe {
        cuEventDestroy_v2(ev_start);
        cuEventDestroy_v2(ev_end);
    }
    Ok((ms as f64 / 1e3) / ITERS as f64)
}

fn tflops(m: usize, n: usize, k: usize, secs: f64) -> f64 {
    2.0 * m as f64 * n as f64 * k as f64 / secs / 1e12
}

struct Shape {
    label: &'static str,
    n: usize,
    k: usize,
}

fn main() -> Result<()> {
    let gpu = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let stream = 0u64;
    let w8a16_k = gpu.kernel("w8a16_gemm_pipelined", "w8a16_gemm_pipelined")?;
    // 2026-09-25: `Fp8ActQuant::resolve` loads the Hopper twin too when the
    // image has it, and each launch picks the twin or the shared kernel as a
    // layer would (`ops::fp8_act_quant_floor`).
    // `native_fp8_act_quant_hopper_microtest` gates the two as byte-identical.
    let quant_k = ops::Fp8ActQuant::resolve(&gpu);
    let w8a8_k = gpu.kernel("fp8_gemm_t_blockscaled", "fp8_gemm_t_blockscaled")?;
    let scale_kmajor_k = gpu.kernel("fp8_scale_transpose", "fp8_act_scale_to_kmajor")?;
    let want_cublas = std::env::var("METRALE_CUBLAS_GEMM").as_deref() == Ok("1");
    let kmajor = ops::cublas_scale_layout_kmajor();
    let rel_rms_gate = std::env::var("METRALE_W8A8_REL_RMS_GATE")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(REL_RMS_GATE);

    println!(
        "dense-FFN W8A8 microtest — H={H} INTER={INTER}  cuBLASLt={}  \
         scale_layout={}  gates: cosine>={COSINE_GATE} rel_rms<={rel_rms_gate}  \
         cuBLASLt-vs-kernel: over_1ulp==0 (small-value escape \
         |v|<{CUBLAS_SMALL_MAGNITUDE}) cosine>={CUBLAS_COSINE_GATE} \
         rel_rms<={CUBLAS_REL_RMS_GATE}",
        if want_cublas {
            "on (METRALE_CUBLAS_GEMM=1)"
        } else {
            "off"
        },
        if kmajor {
            "kmajor [K/128,M_pad] (documented)"
        } else {
            "rowmajor [M,K/128] (pre-fix control, expected to FAIL)"
        }
    );

    let mut rng = Rng(0x9_17_09_28_2026);
    let max_m_pad = BATCHES
        .iter()
        .map(|m| m.div_ceil(16) * 16)
        .max()
        .expect("BATCHES is non-empty");
    let max_k = H.max(INTER);
    let max_n = H.max(INTER);

    // 2026-09-25: Activations [max_m_pad, max_k] BF16, one buffer sliced per
    // shape. Buffers are sized for ceil16(M) rows because the cuBLASLt arm
    // reads and writes that many.
    let act_host: Vec<u8> = (0..max_m_pad * max_k)
        .flat_map(|_| bf16::from_f32(rng.unit()).to_bits().to_le_bytes())
        .collect();
    let act = upload(&gpu, &act_host)?;
    let a_fp8 = gpu.alloc(max_m_pad * max_k)?;
    let a_scale = gpu.alloc(max_m_pad * (max_k / BLOCK) * 4)?;
    // 2026-09-25: `[K/128, ceil16(M)]` transposed scales for the cuBLASLt arm,
    // the layout adapter's destination, same element count as `a_scale`.
    let a_scale_kmajor = gpu.alloc(max_m_pad * (max_k / BLOCK) * 4)?;
    let out_ref = gpu.alloc(max_m_pad * max_n * 2)?;
    let out_w8a8 = gpu.alloc(max_m_pad * max_n * 2)?;
    let out_cublas = gpu.alloc(max_m_pad * max_n * 2)?;

    let shapes = [
        Shape {
            label: "gate/up",
            n: INTER,
            k: H,
        },
        Shape {
            label: "down",
            n: H,
            k: INTER,
        },
    ];
    let mut failures: Vec<String> = Vec::new();

    for shape in shapes {
        let (n, k) = (shape.n, shape.k);
        // 2026-09-25: E4M3 bytes with a magnitude code of 0x00..=0x7E, so never
        // the NaN encodings 0x7F/0xFF.
        let w_host: Vec<u8> = (0..n * k)
            .map(|_| {
                let x = rng.next_u32();
                ((x % 127) as u8) | (((x >> 7) & 1) as u8) << 7
            })
            .collect();
        // 2026-09-25: Block scales [N/128, K/128] FP32, indexed
        // `scale[n/128][k/128]`.
        let s_host: Vec<u8> = (0..(n / BLOCK) * (k / BLOCK))
            .flat_map(|_| ((rng.next_u32() % 16 + 1) as f32 / 1024.0).to_le_bytes())
            .collect();
        let weight = upload(&gpu, &w_host)?;
        let scale = upload(&gpu, &s_host)?;
        let fp8w = Fp8Weight {
            weight,
            row_scale: scale,
            n: n as u32,
            k: k as u32,
            scale_format: WeightQuantFormat::Fp8BlockScaled,
        };

        for m in BATCHES {
            let (mu, nu, ku) = (m as u32, n as u32, k as u32);
            // 2026-09-25: W8A16 reference.
            let w8a16 = || {
                ops::w8a16_gemm_pipelined(
                    &gpu, w8a16_k, act, weight, scale, out_ref, mu, nu, ku, stream,
                )
            };
            w8a16()?;
            gpu.synchronize(stream)?;
            let ref_bits = download_bf16(&gpu, out_ref, m * n)?;
            let t_w8a16 = time_gpu(&gpu, stream, w8a16)?;

            // 2026-09-25: W8A8: quantize once, then the in-tree GEMM.
            ops::per_token_group_quant_fp8(&gpu, quant_k, act, a_fp8, a_scale, mu, ku, stream)?;
            gpu.synchronize(stream)?;
            let w8a8 = || {
                ops::fp8_gemm_t_blockscaled(
                    &gpu, w8a8_k, a_fp8, a_scale, weight, scale, out_w8a8, mu, nu, ku, stream,
                )
            };
            w8a8()?;
            gpu.synchronize(stream)?;
            let w8a8_bits = download_bf16(&gpu, out_w8a8, m * n)?;
            let t_w8a8 = time_gpu(&gpu, stream, w8a8)?;

            let c = compare(&w8a8_bits, &ref_bits);
            println!(
                "[{}] M={m} N={n} K={k}\n  W8A16 ref : {:>8.3} ms  {:>7.2} TFLOP/s\n  \
                 W8A8 kern : {:>8.3} ms  {:>7.2} TFLOP/s  ({:.2}x)  \
                 max_abs={:.6} cosine={:.6} rel_rms={:.4}",
                shape.label,
                t_w8a16 * 1e3,
                tflops(m, n, k, t_w8a16),
                t_w8a8 * 1e3,
                tflops(m, n, k, t_w8a8),
                t_w8a16 / t_w8a8,
                c.max_abs,
                c.cosine,
                c.rel_rms,
            );
            if !(c.cosine >= COSINE_GATE) || !c.cosine.is_finite() {
                failures.push(format!(
                    "{} M={m}: cosine {:.6} < {COSINE_GATE}",
                    shape.label, c.cosine
                ));
            }
            if !(c.rel_rms <= rel_rms_gate) || !c.rel_rms.is_finite() {
                failures.push(format!(
                    "{} M={m}: rel_rms {:.4} > {rel_rms_gate}",
                    shape.label, c.rel_rms
                ));
            }

            // 2026-09-25: cuBLASLt on the same quantized activation.
            if want_cublas {
                let cublas = || {
                    ops::cublas_fp8_proj_prequant(
                        &gpu,
                        scale_kmajor_k,
                        a_fp8,
                        a_scale,
                        a_scale_kmajor,
                        &fp8w,
                        out_cublas,
                        mu,
                        nu,
                        ku,
                        stream,
                    )
                };
                cublas()?;
                gpu.synchronize(stream)?;
                let cub_bits = download_bf16(&gpu, out_cublas, m * n)?;
                let t_cub = time_gpu(&gpu, stream, cublas)?;
                let d = compare(&cub_bits, &w8a8_bits);
                let r = compare(&cub_bits, &ref_bits);
                println!(
                    "  cuBLASLt  : {:>8.3} ms  {:>7.2} TFLOP/s  ({:.2}x vs W8A16, {:.2}x vs kernel)\n    \
                     vs kernel: over_1ulp={}/{} sign_flips={} unequal_bf16={} max_ulp={} \
                     max_abs={:.6} cosine={:.9} rel_rms={:.2e}\n    \
                     vs W8A16 : max_abs={:.6} cosine={:.6} rel_rms={:.4}",
                    t_cub * 1e3,
                    tflops(m, n, k, t_cub),
                    t_w8a16 / t_cub,
                    t_w8a8 / t_cub,
                    d.over_bound,
                    m * n,
                    d.sign_flips,
                    d.unequal,
                    d.max_ulp,
                    d.max_abs,
                    d.cosine,
                    d.rel_rms,
                    r.max_abs,
                    r.cosine,
                    r.rel_rms,
                );
                // 2026-09-25: Same quantized inputs and the same FP32 scales;
                // the bounds are explained beside the gate constants in
                // `common/native_fp8_bf16_compare.rs`.
                if d.over_bound > 0 {
                    failures.push(format!(
                        "{} M={m}: cuBLASLt vs kernel {} of {} elements outside one BF16 ULP \
                         (small-value escape |v|<{CUBLAS_SMALL_MAGNITUDE}) — check the VEC128 \
                         act-scale / BLK128x128 weight-scale layouts",
                        shape.label,
                        d.over_bound,
                        m * n
                    ));
                }
                if !(d.cosine >= CUBLAS_COSINE_GATE) || !d.cosine.is_finite() {
                    failures.push(format!(
                        "{} M={m}: cuBLASLt vs kernel cosine {:.9} < {CUBLAS_COSINE_GATE}",
                        shape.label, d.cosine
                    ));
                }
                if !(d.rel_rms <= CUBLAS_REL_RMS_GATE) || !d.rel_rms.is_finite() {
                    failures.push(format!(
                        "{} M={m}: cuBLASLt vs kernel rel_rms {:.2e} > {CUBLAS_REL_RMS_GATE:.0e}",
                        shape.label, d.rel_rms
                    ));
                }
                if !(r.cosine >= COSINE_GATE) {
                    failures.push(format!(
                        "{} M={m}: cuBLASLt vs W8A16 cosine {:.6} < {COSINE_GATE}",
                        shape.label, r.cosine
                    ));
                }
            }
        }
        gpu.free(weight).ok();
        gpu.free(scale).ok();
    }

    for p in [
        act,
        a_fp8,
        a_scale,
        a_scale_kmajor,
        out_ref,
        out_w8a8,
        out_cublas,
    ] {
        gpu.free(p).ok();
    }
    if failures.is_empty() {
        println!("RESULT: PASS (all shapes within cosine/rel_rms/one-BF16-ULP gates)");
        Ok(())
    } else {
        for f in &failures {
            eprintln!("FAIL: {f}");
        }
        bail!("{} gate(s) failed", failures.len())
    }
}
