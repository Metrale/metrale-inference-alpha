// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Bitwise oracle for the Hopper SSM BA-gates twin
//! `dense_gemm_ba_gates_prefill_hopper` against the gb10 parent
//! `dense_gemm_ba_gates_prefill`.
//!
//! Owner: model-arch examples (GPU microtests).
//! Invariants:
//! - The run exits 0 only if, at every token count M in the `main` list, the
//!   twin's gate and beta bytes equal the parent's, no guard byte changed, and
//!   the one-ulp KNOWN_BAD control was seen.
//!
//! `kernels/hopper/common/ssm_ba_gates_hopper.cu` runs one CTA per token
//! instead of the parent's `ceil(N/4)` and widens A from BF16 once per uint4
//! instead of once per output. It keeps the parent's lane-strided `kv` sweep,
//! the 5-step shuffle, the `warp_even + warp_odd` cross-warp sum and the
//! transforms, so the gate is byte equality. `max_abs` is printed only to make
//! a failure diagnosable.
//!
//! The twin runs at every M, including those below `ops::ba_gates_min_tokens`
//! (264 tokens at 132 SMs), which the dispatcher would send to the parent:
//! bit identity must hold at every M, while the floor is a speed rule.
//!
//! Guard bytes bracket both output buffers: the twin computes its output
//! indices from a tiled group sweep, and an index error there can write
//! outside the output, where a payload comparison would not see it.
//!
//! Timing is reported per arm as µs, as compulsory GB/s (one read of the
//! activation block and of the BA weight, one write of the gate block), and as
//! the activation-row reads per token: `N` (96) for the parent,
//! `ceil(ceil(N/4)/8) * 4` (12) for the twin.
//!
//!   cargo run -p metrale-model-arch --release --features cuda,gpu-examples \
//!       --example native_ssm_ba_gates_hopper_microtest

use anyhow::Result;
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_model_layers::layers::ops;
use metrale_model_layers::weight_map::DenseWeight;

/// 2026-09-25: Harness geometry: 48 value heads, so `N` = 2 x 48 = 96 BA
/// outputs (`ssm_ba_size`), over K = 5120, with 2 value heads per group.
const NV: usize = 48;
const N: usize = 2 * NV;
const K: usize = 5120;
const VPG: usize = 2;
const GATE_STRIDE: usize = 2 * NV;

const GUARD: usize = 64;
const SENTINEL: u8 = 0x5a;
const WARMUP: u32 = 3;
const REPS: u32 = 10;

/// 2026-09-25: Deterministic LCG with a fixed seed, so both kernels get the
/// same inputs and a failure reproduces.
struct Lcg(u32);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((self.0 >> 8) & 0xFF_FFFF) as f32 / 8_388_608.0 - 1.0
    }
}

fn up_bf16(g: &dyn GpuBackend, d: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = d
        .iter()
        .flat_map(|x| bf16::from_f32(*x).to_bits().to_le_bytes())
        .collect();
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}

fn up_f32(g: &dyn GpuBackend, d: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = d.iter().flat_map(|x| x.to_le_bytes()).collect();
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}

/// 2026-09-25: Allocate `bytes` of payload with `GUARD` sentinel bytes on each
/// side and return (base, payload).
fn guarded(g: &dyn GpuBackend, bytes: usize) -> Result<(DevicePtr, DevicePtr)> {
    let base = g.alloc(bytes + 2 * GUARD)?;
    g.copy_h2d(&vec![SENTINEL; bytes + 2 * GUARD], base)?;
    Ok((base, base.offset(GUARD)))
}

fn guards_intact(g: &dyn GpuBackend, base: DevicePtr, bytes: usize) -> Result<bool> {
    let mut raw = vec![0u8; bytes + 2 * GUARD];
    g.copy_d2h(base, &mut raw)?;
    Ok(raw[..GUARD].iter().all(|b| *b == SENTINEL)
        && raw[GUARD + bytes..].iter().all(|b| *b == SENTINEL))
}

fn dn(g: &dyn GpuBackend, p: DevicePtr, bytes: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; bytes];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}

/// 2026-09-25: (differing 4-byte lanes, largest absolute difference).
fn diff(a: &[u8], b: &[u8]) -> (usize, f32) {
    let mut n = 0usize;
    let mut worst = 0.0f32;
    for (x, y) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
        if x != y {
            n += 1;
            let (xv, yv) = (
                f32::from_le_bytes(x.try_into().unwrap()),
                f32::from_le_bytes(y.try_into().unwrap()),
            );
            worst = worst.max((xv - yv).abs());
        }
    }
    (n, worst)
}

fn time_us(g: &dyn GpuBackend, mut run: impl FnMut() -> Result<()>) -> Result<f64> {
    for _ in 0..WARMUP {
        run()?;
    }
    g.synchronize(0)?;
    let t0 = std::time::Instant::now();
    for _ in 0..REPS {
        run()?;
    }
    g.synchronize(0)?;
    Ok(t0.elapsed().as_secs_f64() * 1e6 / f64::from(REPS))
}

/// 2026-09-25: Device-side inputs shared by both arms of one leg.
struct Leg {
    a: DevicePtr,
    b: DenseWeight,
    a_log: DevicePtr,
    dt_bias: DevicePtr,
}

fn inputs(g: &dyn GpuBackend, m: usize) -> Result<Leg> {
    let mut r = Lcg(0x5EED_1234);
    // 2026-09-25: Small magnitudes keep the gate transform out of saturation,
    // where a reduction-order difference would be hidden.
    let a: Vec<f32> = (0..m * K).map(|_| r.next_f32() * 0.5).collect();
    let b: Vec<f32> = (0..N * K).map(|_| r.next_f32() * 0.02).collect();
    let a_log: Vec<f32> = (0..NV).map(|_| r.next_f32() * 2.0).collect();
    let dt_bias: Vec<f32> = (0..NV).map(|_| r.next_f32()).collect();
    Ok(Leg {
        a: up_bf16(g, &a)?,
        b: DenseWeight {
            weight: up_bf16(g, &b)?,
        },
        a_log: up_f32(g, &a_log)?,
        dt_bias: up_f32(g, &dt_bias)?,
    })
}

/// 2026-09-25: One token count: both arms, byte comparison, guards, timings.
fn leg(g: &dyn GpuBackend, parent: KernelHandle, twin: KernelHandle, m: usize) -> Result<bool> {
    let inp = inputs(g, m)?;
    let out_bytes = m * GATE_STRIDE * 4;
    let (ref_base, ref_p) = guarded(g, out_bytes)?;
    let (new_base, new_p) = guarded(g, out_bytes)?;

    let m32 = m as u32;
    // 2026-09-25: `KernelHandle(0)` for the twin makes `ba_gates_pick` choose
    // the gb10 parent at every M, whatever `[defaults] ssm_ba_gates_hopper`
    // says for this build.
    let run_parent = |dst: DevicePtr| -> Result<()> {
        ops::dense_gemm_ba_gates_prefill(
            g,
            parent,
            KernelHandle(0),
            inp.a,
            &inp.b,
            inp.a_log,
            inp.dt_bias,
            dst,
            m32,
            N as u32,
            K as u32,
            K as u32,
            GATE_STRIDE as u32,
            NV as u32,
            VPG as u32,
            0,
        )
    };
    // 2026-09-25: The twin's launcher directly, bypassing the token-count
    // guard: the guard is a speed rule and this file checks correctness.
    let run_twin = |dst: DevicePtr| -> Result<()> {
        ops::dense_gemm_ba_gates_prefill_hopper(
            g,
            twin,
            inp.a,
            &inp.b,
            inp.a_log,
            inp.dt_bias,
            dst,
            m32,
            N as u32,
            K as u32,
            K as u32,
            GATE_STRIDE as u32,
            NV as u32,
            VPG as u32,
            0,
        )
    };

    run_parent(ref_p)?;
    run_twin(new_p)?;
    g.synchronize(0)?;

    let ref_bytes = dn(g, ref_p, out_bytes)?;
    let new_bytes = dn(g, new_p, out_bytes)?;
    let (nd, worst) = diff(&ref_bytes, &new_bytes);
    let guards = guards_intact(g, ref_base, out_bytes)? && guards_intact(g, new_base, out_bytes)?;

    // 2026-09-25: KNOWN_BAD: flip the lowest bit (one ulp) of the first
    // reference value, the smallest defect the byte gate claims to catch, and
    // require exactly one more differing lane.
    let mut bad = ref_bytes.clone();
    let poisoned = f32::from_le_bytes(bad[..4].try_into().unwrap());
    let nudged = f32::from_bits(poisoned.to_bits() ^ 1);
    bad[..4].copy_from_slice(&nudged.to_le_bytes());
    let (bad_n, _) = diff(&bad, &new_bytes);
    let control_ok = bad_n == nd + 1;

    let t_parent = time_us(g, || run_parent(ref_p))?;
    let t_twin = time_us(g, || run_twin(new_p))?;
    // 2026-09-25: Compulsory traffic: the activation block read once, the BA
    // weight read once, the gate block written once.
    let compulsory = (m * K * 2 + N * K * 2 + out_bytes) as f64;
    let gbps = |us: f64| compulsory / (us * 1e-6) / 1e9;
    // 2026-09-25: Activation-row reads per token: the parent does one per BA
    // output; the twin one per (64-lane group, output-group tile).
    let n_groups = N.div_ceil(ops::BA_GATES_OUTS as usize);
    let parent_amp = N;
    let twin_amp = n_groups.div_ceil(ops::BA_GATES_GROUPS as usize) * ops::BA_GATES_OUTS as usize;

    eprintln!(
        "  M={m:<5} out_diff={nd} max_abs={worst:.3e} guards={} control={} | \
         parent {t_parent:9.2} us {:7.1} GB/s (A reads {parent_amp}) | \
         twin {t_twin:9.2} us {:7.1} GB/s (A reads {twin_amp}) | {:.2}x",
        if guards { "ok" } else { "CLOBBERED" },
        if control_ok { "refused" } else { "BLIND" },
        gbps(t_parent),
        gbps(t_twin),
        t_parent / t_twin,
    );

    for p in [
        inp.a,
        inp.b.weight,
        inp.a_log,
        inp.dt_bias,
        ref_base,
        new_base,
    ] {
        g.free(p).ok();
    }
    anyhow::ensure!(guards, "M={m}: a kernel wrote outside its output buffer");
    anyhow::ensure!(
        control_ok,
        "M={m}: the KNOWN_BAD control did not trip — a one-ulp poison in the \
         reference produced {bad_n} differing lanes against a clean {nd}, so \
         this leg proves nothing"
    );
    anyhow::ensure!(
        nd == 0,
        "M={m}: {nd} gate/beta lanes differ from the gb10 parent (max_abs \
         {worst:.3e}); the twin must be BIT-identical"
    );
    Ok(true)
}

fn main() -> Result<()> {
    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;

    let parent = g.kernel("ssm_preprocess", "dense_gemm_ba_gates_prefill")?;
    let twin = g.kernel("ssm_ba_gates_hopper", "dense_gemm_ba_gates_prefill_hopper")?;

    let sms = ops::ba_gates_sm_count(g);
    eprintln!(
        "native_ssm_ba_gates_hopper_microtest: N={N} K={K} nv={NV} vpg={VPG} \
         sm_count={sms} twin_floor={} tokens",
        ops::ba_gates_min_tokens(sms)
    );
    eprintln!(
        "  block={} lanes/out={} outs/tile-step={} groups/tile={}",
        ops::BA_GATES_BLOCK,
        ops::BA_GATES_LANES,
        ops::BA_GATES_OUTS,
        ops::BA_GATES_GROUPS,
    );

    // 2026-09-25: 17 and 25 are below the twin's token floor at 132 SMs;
    // 1168 and 4576 are above it.
    for m in [17usize, 25, 1168, 4576] {
        leg(g, parent, twin, m)?;
    }
    eprintln!("  ALL LEGS BIT-IDENTICAL");
    Ok(())
}
