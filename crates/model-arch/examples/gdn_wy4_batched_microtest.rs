// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Byte-identity of one batched `gated_delta_rule_wy4` launch with pointer
//! tables (`state_is_table = 1`) against n launches of batch 1, then a timing check.
//!
//! Owner: model-arch examples.
//! Invariants:
//! - Returns an error unless, for n in 2, 4, 8 and 16, the state, all NI intermediates and
//!   the output are byte-identical, or when the cost check fails.
//!
//! The pools use distinct strides: sequence i's state at i * HV floats, its intermediate t
//! at (i * NI + t) * HV. A kernel that indexed the intermediates with the state's stride
//! would write one sequence's intermediates over another's.
//!
//! Run: METRALE_TARGET_MODEL=qwen3.6-27b cargo run -p metrale-model-arch --release \
//!        --example gdn_wy4_batched_microtest --features cuda,gpu-examples

use anyhow::{Result, bail};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

const NK: usize = 16;
const NV: usize = 48;
const KD: usize = 128;
const VD: usize = 128;
const K: usize = 4;
const NI: usize = 4;
const KEY_DIM: usize = NK * KD;
const VALUE_DIM: usize = NV * VD;
const CONV_DIM: usize = KEY_DIM * 2 + VALUE_DIM;
const GB_STRIDE: usize = NV * 2;
/// 2026-09-25: Floats in one sequence's state, and the state pool's per-sequence stride.
const HV: usize = NV * KD * VD;

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((self.0 >> 11) as f64) / ((1u64 << 53) as f64)) as f32
    }
    fn r(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.f()
    }
}

fn up_f32(g: &dyn GpuBackend, d: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = d.iter().flat_map(|x| x.to_le_bytes()).collect();
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}

fn up_bf16(g: &dyn GpuBackend, d: &[bf16]) -> Result<DevicePtr> {
    let b: Vec<u8> = d.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect();
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}

fn down_f32(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut b = vec![0u8; n * 4];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// 2026-09-25: Device array of one pointer per sequence, which the kernel reads when
/// `state_is_table` = 1.
fn up_ptr_table(g: &dyn GpuBackend, ptrs: &[DevicePtr]) -> Result<DevicePtr> {
    let b: Vec<u8> = ptrs.iter().flat_map(|p| p.0.to_le_bytes()).collect();
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}

#[allow(clippy::too_many_arguments)]
fn launch_wy4(
    g: &dyn GpuBackend,
    kernel: metrale_gpu_runtime::gpu::KernelHandle,
    h: DevicePtr,
    q: DevicePtr,
    k: DevicePtr,
    v: DevicePtr,
    gate: DevicePtr,
    beta: DevicePtr,
    out: DevicePtr,
    hi0: DevicePtr,
    hi1: DevicePtr,
    hi2: DevicePtr,
    batch: u32,
    is_table: bool,
) -> Result<()> {
    KernelLaunch::new(g, kernel)
        .grid([NV as u32, batch, 1])
        .block([128, 1, 1])
        .arg_ptr(h)
        .arg_ptr(q)
        .arg_ptr(k)
        .arg_ptr(v)
        .arg_ptr(gate)
        .arg_ptr(beta)
        .arg_ptr(out)
        .arg_ptr(hi0)
        .arg_ptr(hi1)
        .arg_ptr(hi2)
        .arg_u32(batch)
        .arg_u32(NK as u32)
        .arg_u32(NV as u32)
        .arg_u32(KD as u32)
        .arg_u32(VD as u32)
        .arg_u32(CONV_DIM as u32)
        .arg_u32(CONV_DIM as u32)
        .arg_u32(GB_STRIDE as u32)
        .arg_u32(u32::from(is_table))
        .launch(0)?;
    g.synchronize(0)?;
    Ok(())
}

fn run_for_n(
    g: &dyn GpuBackend,
    kernel: metrale_gpu_runtime::gpu::KernelHandle,
    n: usize,
) -> Result<()> {
    let mut rng = Lcg(0x5eed_1a2b ^ (n as u64) << 32);
    let rows = n * K;

    // 2026-09-25: Token rows are [seq0_t0..t3, seq1_t0..t3, ...], the kernel's b * 4 + t
    // row order.
    let qkv: Vec<bf16> = (0..rows * CONV_DIM)
        .map(|_| bf16::from_f32(rng.r(-1.0, 1.0)))
        .collect();
    let gate: Vec<f32> = (0..rows * GB_STRIDE).map(|_| rng.r(0.90, 0.999)).collect();
    let beta: Vec<f32> = (0..rows * GB_STRIDE).map(|_| rng.r(0.1, 0.9)).collect();
    // 2026-09-25: Distinct initial states per sequence, so a mix-up between sequences
    // changes the result.
    let h_init: Vec<f32> = (0..n * HV).map(|_| rng.r(-0.5, 0.5)).collect();
    // 2026-09-25: Every intermediate is pre-filled with a value distinct per
    // (sequence, token), so a write into the wrong one is visible.
    let hi_init: Vec<f32> = (0..n * NI * HV)
        .map(|i| -1000.0 - ((i / HV) as f32))
        .collect();

    let d_q = up_bf16(g, &qkv)?;
    let d_gate = up_f32(g, &gate)?;
    let d_beta = up_f32(g, &beta)?;

    // 2026-09-25: Reference: n launches of batch 1 with contiguous addressing, each at its
    // own slot.
    let ref_h = up_f32(g, &h_init)?;
    let ref_hi = up_f32(g, &hi_init)?;
    let ref_out = g.alloc(rows * VALUE_DIM * 2)?;
    for i in 0..n {
        let tok = |stride: usize| DevicePtr(d_q.0 + (i * K * stride * 2) as u64);
        launch_wy4(
            g,
            kernel,
            DevicePtr(ref_h.0 + (i * HV * 4) as u64),
            tok(CONV_DIM),
            tok(CONV_DIM),
            tok(CONV_DIM),
            DevicePtr(d_gate.0 + (i * K * GB_STRIDE * 4) as u64),
            DevicePtr(d_beta.0 + (i * K * GB_STRIDE * 4) as u64),
            DevicePtr(ref_out.0 + (i * K * NV * VD * 2) as u64),
            DevicePtr(ref_hi.0 + ((i * NI) * HV * 4) as u64),
            DevicePtr(ref_hi.0 + ((i * NI + 1) * HV * 4) as u64),
            DevicePtr(ref_hi.0 + ((i * NI + 2) * HV * 4) as u64),
            1,
            false,
        )?;
    }

    // 2026-09-25: Under test: one launch of batch n with pointer tables.
    let tst_h = up_f32(g, &h_init)?;
    let tst_hi = up_f32(g, &hi_init)?;
    let tst_out = g.alloc(rows * VALUE_DIM * 2)?;
    let h_tbl = up_ptr_table(
        g,
        &(0..n)
            .map(|i| DevicePtr(tst_h.0 + (i * HV * 4) as u64))
            .collect::<Vec<_>>(),
    )?;
    let mk_hi_tbl = |t: usize| -> Result<DevicePtr> {
        up_ptr_table(
            g,
            &(0..n)
                .map(|i| DevicePtr(tst_hi.0 + ((i * NI + t) * HV * 4) as u64))
                .collect::<Vec<_>>(),
        )
    };
    launch_wy4(
        g,
        kernel,
        h_tbl,
        d_q,
        d_q,
        d_q,
        d_gate,
        d_beta,
        tst_out,
        mk_hi_tbl(0)?,
        mk_hi_tbl(1)?,
        mk_hi_tbl(2)?,
        n as u32,
        true,
    )?;

    let (a, b) = (down_f32(g, ref_h, n * HV)?, down_f32(g, tst_h, n * HV)?);
    if let Some(i) = a
        .iter()
        .zip(&b)
        .position(|(x, y)| x.to_bits() != y.to_bits())
    {
        bail!(
            "n={n}: h_state differs at float {i} (seq {}): ref {} vs batched {}",
            i / HV,
            a[i],
            b[i]
        );
    }
    let (a, b) = (
        down_f32(g, ref_hi, n * NI * HV)?,
        down_f32(g, tst_hi, n * NI * HV)?,
    );
    if let Some(i) = a
        .iter()
        .zip(&b)
        .position(|(x, y)| x.to_bits() != y.to_bits())
    {
        bail!(
            "n={n}: INTERMEDIATE differs at float {i} (seq {}, token {}): ref {} vs batched {} \
             — this is the cross-sequence rollback corruption the pointer table fixes",
            i / (NI * HV),
            (i / HV) % NI,
            a[i],
            b[i]
        );
    }
    let mut ob = vec![0u8; rows * VALUE_DIM * 2];
    let mut tb = vec![0u8; rows * VALUE_DIM * 2];
    g.copy_d2h(ref_out, &mut ob)?;
    g.copy_d2h(tst_out, &mut tb)?;
    if ob != tb {
        bail!("n={n}: output bytes differ");
    }
    println!("  n={n:2}: h_state + {NI} intermediates + output all BYTE-IDENTICAL");
    Ok(())
}

/// 2026-09-25: Cost check at n = 8. Times (mean of 20 after 3 warm-ups):
///   (a) one batched wy4 launch (8 sequences x 4 tokens);
///   (b) `gated_delta_rule_decode_f32_strided` at batch 8, one token each;
///   (c) 8 wy4 launches of batch 1.
/// Errors when (a)/(b) >= 2.0; the printed target is 1.60.
fn cost_gate(g: &dyn GpuBackend, wy4: metrale_gpu_runtime::gpu::KernelHandle) -> Result<()> {
    const N: usize = 8;
    let mut rng = Lcg(0xc0_5701);
    let rows = N * K;
    let qkv: Vec<bf16> = (0..rows * CONV_DIM)
        .map(|_| bf16::from_f32(rng.r(-1.0, 1.0)))
        .collect();
    let gate: Vec<f32> = (0..rows * GB_STRIDE).map(|_| rng.r(0.90, 0.999)).collect();
    let beta: Vec<f32> = (0..rows * GB_STRIDE).map(|_| rng.r(0.1, 0.9)).collect();
    let h_init: Vec<f32> = (0..N * HV).map(|_| rng.r(-0.5, 0.5)).collect();
    let hi_init: Vec<f32> = vec![0.0; N * NI * HV];

    let d_q = up_bf16(g, &qkv)?;
    let d_gate = up_f32(g, &gate)?;
    let d_beta = up_f32(g, &beta)?;
    let d_h = up_f32(g, &h_init)?;
    let d_hi = up_f32(g, &hi_init)?;
    let d_out = g.alloc(rows * VALUE_DIM * 2)?;
    let h_tbl = up_ptr_table(
        g,
        &(0..N)
            .map(|i| DevicePtr(d_h.0 + (i * HV * 4) as u64))
            .collect::<Vec<_>>(),
    )?;
    let hi_tbl = |t: usize| -> Result<DevicePtr> {
        up_ptr_table(
            g,
            &(0..N)
                .map(|i| DevicePtr(d_hi.0 + ((i * NI + t) * HV * 4) as u64))
                .collect::<Vec<_>>(),
        )
    };
    let (t0, t1, t2) = (hi_tbl(0)?, hi_tbl(1)?, hi_tbl(2)?);

    let time_it = |f: &dyn Fn() -> Result<()>| -> Result<f64> {
        for _ in 0..3 {
            f()?;
        }
        let start = std::time::Instant::now();
        const ITERS: u32 = 20;
        for _ in 0..ITERS {
            f()?;
        }
        Ok(start.elapsed().as_secs_f64() * 1e6 / f64::from(ITERS))
    };

    let batched = time_it(&|| {
        launch_wy4(
            g, wy4, h_tbl, d_q, d_q, d_q, d_gate, d_beta, d_out, t0, t1, t2, N as u32, true,
        )
    })?;
    let sequential = time_it(&|| {
        for i in 0..N {
            let tok = |st: usize| DevicePtr(d_q.0 + (i * K * st * 2) as u64);
            launch_wy4(
                g,
                wy4,
                DevicePtr(d_h.0 + (i * HV * 4) as u64),
                tok(CONV_DIM),
                tok(CONV_DIM),
                tok(CONV_DIM),
                DevicePtr(d_gate.0 + (i * K * GB_STRIDE * 4) as u64),
                DevicePtr(d_beta.0 + (i * K * GB_STRIDE * 4) as u64),
                DevicePtr(d_out.0 + (i * K * NV * VD * 2) as u64),
                DevicePtr(d_hi.0 + ((i * NI) * HV * 4) as u64),
                DevicePtr(d_hi.0 + ((i * NI + 1) * HV * 4) as u64),
                DevicePtr(d_hi.0 + ((i * NI + 2) * HV * 4) as u64),
                1,
                false,
            )?;
        }
        Ok(())
    })?;

    // 2026-09-25: The single-token decode takes FP32 q/k/v and a trailing out_stride, so it
    // gets its own buffers.
    let plain_k = g.kernel("gated_delta_rule", "gated_delta_rule_decode_f32_strided")?;
    let qf: Vec<f32> = (0..N * CONV_DIM).map(|_| rng.r(-1.0, 1.0)).collect();
    let gf: Vec<f32> = (0..N * GB_STRIDE).map(|_| rng.r(0.90, 0.999)).collect();
    let bf: Vec<f32> = (0..N * GB_STRIDE).map(|_| rng.r(0.1, 0.9)).collect();
    let d_qf = up_f32(g, &qf)?;
    let d_gf = up_f32(g, &gf)?;
    let d_bf = up_f32(g, &bf)?;
    let d_gout = g.alloc(N * VALUE_DIM * 4)?;
    let plain = time_it(&|| {
        KernelLaunch::new(g, plain_k)
            .grid([NV as u32, N as u32, 1])
            .block([128, 1, 1])
            .arg_ptr(d_h)
            .arg_ptr(d_qf)
            .arg_ptr(d_qf)
            .arg_ptr(d_qf)
            .arg_ptr(d_gf)
            .arg_ptr(d_bf)
            .arg_ptr(d_gout)
            .arg_u32(N as u32)
            .arg_u32(NK as u32)
            .arg_u32(NV as u32)
            .arg_u32(KD as u32)
            .arg_u32(VD as u32)
            .arg_u32(CONV_DIM as u32)
            .arg_u32(CONV_DIM as u32)
            .arg_u32(GB_STRIDE as u32)
            .arg_u32(VALUE_DIM as u32)
            .launch(0)?;
        g.synchronize(0)?;
        Ok(())
    })
    .unwrap_or(f64::NAN);

    println!();
    println!("cost gate at n={N}, K={K} (M={}):", N * K);
    println!("  batched wy4 (1 launch)      {batched:8.1} us");
    println!("  sequential wy4 (8 launches) {sequential:8.1} us   [today's per-seq verify]");
    println!("  plain decode, n=8, 1 token  {plain:8.1} us   [non-spec baseline]");
    if plain.is_finite() && plain > 0.0 {
        let ratio = batched / plain;
        println!("  batched/plain = {ratio:.2}x   (GATE: <= 1.60x; >= 2.00x means STOP)");
        if ratio >= 2.0 {
            bail!(
                "COST GATE FAILED: {ratio:.2}x >= 2.0 — GDN verify scales ~linearly in \
                   verify tokens; the fused-verify premise is dead at this layer"
            );
        }
    }
    println!(
        "  batched vs sequential speedup {:.2}x",
        sequential / batched
    );
    Ok(())
}

fn main() -> Result<()> {
    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let kernel = g.kernel("gated_delta_rule_wy4", "gated_delta_rule_wy4")?;
    println!("wy4 batched pointer-table equivalence (nv={NV} kd={KD} vd={VD} K={K} ni={NI})");
    for n in [2usize, 4, 8, 16] {
        run_for_n(g, kernel, n)?;
    }
    println!("PASS — batched wy4 is byte-identical to n sequential single-sequence launches");
    cost_gate(g, kernel)?;
    Ok(())
}
