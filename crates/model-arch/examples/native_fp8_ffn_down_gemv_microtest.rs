// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: GPU oracle for the M=1 native-FP8 decode down projection: the
//! fused `w8a16_gemv_silu_input` against the split-SiLU arm
//! (`Fp8DownArm::SplitSilu`: `moe_silu_mul` staged in place over `gate_out`,
//! then `w8a16_gemv`).
//!
//! Owner: model-arch examples (GPU microtests).
//! Invariants:
//! - The run exits 0 only if staging the SwiGLU in place gives the same
//!   activation bytes and the same down-projection bytes as staging it into a
//!   separate buffer, and no route wrote outside its guarded output.
//!
//! Why in place is checked: `DenseFfnLayer::forward` calls
//! `ops::silu_mul(gate_out, up_out, gate_out)`, so `gate` and `output` alias
//! although `moe_silu_mul` declares both `__restrict__`. Each thread reads its
//! own element before writing it, so the bytes are expected to match; any
//! difference fails the run.
//!
//! The split-vs-fused difference is reported, not gated, as unequal count,
//! max abs and max BF16 ULP: `moe_silu_mul` rounds `g*(1/(1+e^-g))*u` to BF16,
//! where the fused kernel keeps `(g/(1+e^-g))*u` in FP32 into the dot product.
//! Both arms are also timed (`synchronize` plus a host `Instant` over `REPS`
//! after `WARMUP` runs; `GpuBackend` has no event elapsed-time query) and
//! reported as µs and GB/s of FP8 weight bytes.
//!
//! Run:
//!     cargo run --release --example native_fp8_ffn_down_gemv_microtest \
//!       --features cuda,gpu-examples
use anyhow::{Result, ensure};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_model_layers::layers::ops;
use std::time::Instant;

/// 2026-09-25: Qwen3.8-27B `hidden_dim` 5120 and `intermediate_size` 17408
/// (`kernels/hopper/qwen3.8-27b/MODEL.toml`).
const H: u32 = 5120;
const INTER: u32 = 17408;
const GUARD: usize = 64;
const REPS: u32 = 20;
const WARMUP: u32 = 3;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.0 >> 32) as u32
    }
    /// 2026-09-25: A BF16 value in [-1, 1], in steps of 1/1024.
    fn act(&mut self) -> [u8; 2] {
        bf16::from_f32(((self.next() % 2049) as f32 - 1024.0) / 1024.0)
            .to_bits()
            .to_le_bytes()
    }
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}

/// 2026-09-25: Synchronised wall clock over `REPS`, after `WARMUP` untimed
/// runs. Returns microseconds per rep.
fn time_us(gpu: &dyn GpuBackend, mut run: impl FnMut() -> Result<()>) -> Result<f64> {
    for _ in 0..WARMUP {
        run()?;
    }
    gpu.synchronize(0)?;
    let t0 = Instant::now();
    for _ in 0..REPS {
        run()?;
    }
    gpu.synchronize(0)?;
    Ok(t0.elapsed().as_secs_f64() * 1e6 / f64::from(REPS))
}

/// 2026-09-25: GB/s of FP8 weight bytes for one pass over an `[n, k]` weight.
fn gbs(n: u32, k: u32, us: f64) -> f64 {
    (u64::from(n) * u64::from(k)) as f64 / (us * 1e-6) / 1e9
}

/// 2026-09-25: Unequal elements, max absolute difference and max BF16 ULP
/// distance between two BF16 buffers.
fn compare(expected: &[u8], actual: &[u8]) -> (usize, f32, u32) {
    let mut unequal = 0;
    let mut max_abs = 0.0_f32;
    let mut max_ulp = 0_u32;
    for (e, a) in expected.chunks_exact(2).zip(actual.chunks_exact(2)) {
        let eb = u16::from_le_bytes([e[0], e[1]]);
        let ab = u16::from_le_bytes([a[0], a[1]]);
        if eb == ab {
            continue;
        }
        unequal += 1;
        let (ef, af) = (bf16::from_bits(eb).to_f32(), bf16::from_bits(ab).to_f32());
        max_abs = max_abs.max((ef - af).abs());
        // 2026-09-25: Monotone-ordinal ULP: map the sign-magnitude bits onto a
        // signed line.
        let ord = |b: u16| -> i32 {
            if b & 0x8000 != 0 {
                -((b & 0x7FFF) as i32)
            } else {
                b as i32
            }
        };
        max_ulp = max_ulp.max((ord(eb) - ord(ab)).unsigned_abs());
    }
    (unequal, max_abs, max_ulp)
}

struct Kernels {
    gemv: KernelHandle,
    silu_input: KernelHandle,
    silu_mul: KernelHandle,
}

/// 2026-09-25: One projection's device-side inputs, allocated once and reused
/// by every route.
struct Case {
    name: &'static str,
    n: u32,
    k: u32,
    weight: DevicePtr,
    scale: DevicePtr,
    /// 2026-09-25: `[K]` BF16 gate vector, and the host bytes it was uploaded
    /// from, which restore `inplace` before every in-place run.
    gate: DevicePtr,
    gate_host: Vec<u8>,
    /// 2026-09-25: `[K]` BF16 up vector.
    up: DevicePtr,
    /// 2026-09-25: `[K]` BF16 staging buffer for `silu(gate)*up`, written
    /// out of place.
    act: DevicePtr,
    /// 2026-09-25: `[K]` BF16 scratch in the role of `gate_out`: the SwiGLU is
    /// staged over it in place, as `DenseFfnLayer::forward` does.
    inplace: DevicePtr,
}

impl Case {
    fn build(
        gpu: &dyn GpuBackend,
        rng: &mut Rng,
        name: &'static str,
        n: u32,
        k: u32,
    ) -> Result<Self> {
        let (nn, kk) = (n as usize, k as usize);
        // 2026-09-25: E4M3 byte draws skip 0x7F/0xFF (NaN).
        let weights: Vec<u8> = (0..nn * kk)
            .map(|_| {
                let x = rng.next();
                ((x % 127) as u8) | (((x >> 7) & 1) as u8 * 128)
            })
            .collect();
        let scales: Vec<u8> = (0..(nn / 128) * (kk / 128))
            .flat_map(|_| (((rng.next() % 16 + 1) as f32) / 1024.0).to_le_bytes())
            .collect();
        let gate: Vec<u8> = (0..kk).flat_map(|_| rng.act()).collect();
        let up: Vec<u8> = (0..kk).flat_map(|_| rng.act()).collect();
        Ok(Self {
            name,
            n,
            k,
            weight: upload(gpu, &weights)?,
            scale: upload(gpu, &scales)?,
            gate: upload(gpu, &gate)?,
            gate_host: gate,
            up: upload(gpu, &up)?,
            act: upload(gpu, &vec![0_u8; kk * 2])?,
            inplace: upload(gpu, &vec![0_u8; kk * 2])?,
        })
    }

    /// 2026-09-25: Restore the `inplace` scratch, which the in-place route
    /// overwrites.
    fn reload_inplace(&self, gpu: &dyn GpuBackend) -> Result<()> {
        gpu.copy_h2d(&self.gate_host, self.inplace)
    }
}

/// 2026-09-25: A guarded `[1, N]` BF16 output: sentinel bytes either side,
/// re-stamped by `reset` so a write past either end is visible.
struct Out {
    base: DevicePtr,
    sentinel: Vec<u8>,
    bytes: usize,
}

impl Out {
    fn new(gpu: &dyn GpuBackend, n: u32) -> Result<Self> {
        let bytes = n as usize * 2;
        let sentinel = vec![0x5a_u8; bytes + 2 * GUARD];
        Ok(Self {
            base: upload(gpu, &sentinel)?,
            sentinel,
            bytes,
        })
    }
    fn ptr(&self) -> DevicePtr {
        self.base.offset(GUARD)
    }
    fn reset(&self, gpu: &dyn GpuBackend) -> Result<()> {
        gpu.copy_h2d(&self.sentinel, self.base)
    }
    /// 2026-09-25: Read back the payload, refusing any change to a guard byte.
    fn read(&self, gpu: &dyn GpuBackend) -> Result<Vec<u8>> {
        let mut host = vec![0_u8; self.sentinel.len()];
        gpu.copy_d2h(self.base, &mut host)?;
        for (i, b) in host.iter().enumerate() {
            let in_payload = (GUARD..GUARD + self.bytes).contains(&i);
            ensure!(
                in_payload || *b == 0x5a,
                "route wrote outside its extent at byte {i}"
            );
        }
        Ok(host[GUARD..GUARD + self.bytes].to_vec())
    }
}

/// 2026-09-25: `silu_mul` into a separate buffer, then the plain scalar GEMV.
fn staged_route(gpu: &dyn GpuBackend, kern: &Kernels, c: &Case, out: DevicePtr) -> Result<()> {
    ops::silu_mul(gpu, kern.silu_mul, c.gate, c.up, c.act, c.k, 0)?;
    ops::w8a16_gemv(gpu, kern.gemv, c.act, c.weight, c.scale, out, c.n, c.k, 0)
}

/// 2026-09-25: The `SplitSilu` arm's staging: `silu_mul` in place over the
/// gate buffer, then the plain scalar GEMV over that same buffer.
fn inplace_route(gpu: &dyn GpuBackend, kern: &Kernels, c: &Case, out: DevicePtr) -> Result<()> {
    c.reload_inplace(gpu)?;
    ops::silu_mul(gpu, kern.silu_mul, c.inplace, c.up, c.inplace, c.k, 0)?;
    ops::w8a16_gemv(
        gpu, kern.gemv, c.inplace, c.weight, c.scale, out, c.n, c.k, 0,
    )
}

/// 2026-09-25: The in-place SwiGLU staging must be byte-identical to the
/// out-of-place one. Returns the failure count (0 or 1).
fn bit_identity_gate(gpu: &dyn GpuBackend, kern: &Kernels, c: &Case) -> Result<usize> {
    let (staged, inplace) = (Out::new(gpu, c.n)?, Out::new(gpu, c.n)?);
    staged.reset(gpu)?;
    inplace.reset(gpu)?;
    staged_route(gpu, kern, c, staged.ptr())?;
    inplace_route(gpu, kern, c, inplace.ptr())?;
    gpu.synchronize(0)?;
    // 2026-09-25: The staged activation itself, then the projection that
    // consumed it.
    let mut act_host = vec![0_u8; c.k as usize * 2];
    let mut inplace_host = vec![0_u8; c.k as usize * 2];
    gpu.copy_d2h(c.act, &mut act_host)?;
    gpu.copy_d2h(c.inplace, &mut inplace_host)?;
    let (act_unequal, ..) = compare(&act_host, &inplace_host);
    let (unequal, max_abs, _) = compare(&staged.read(gpu)?, &inplace.read(gpu)?);
    let failed = act_unequal != 0 || unequal != 0;
    let verdict = if failed { "FAIL" } else { "PASS" };
    println!(
        "  [{verdict}] {:<8} in-place vs out-of-place SwiGLU staging: \
         activation unequal={act_unequal}, down unequal={unequal} max_abs={max_abs:.9}",
        c.name
    );
    Ok(usize::from(failed))
}

fn main() -> Result<()> {
    let gpu = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let kern = Kernels {
        gemv: gpu.kernel("w8a16_gemv", "w8a16_gemv")?,
        silu_input: gpu.kernel("w8a16_gemv_fused", "w8a16_gemv_silu_input")?,
        silu_mul: gpu.kernel("moe_silu_mul", "moe_silu_mul")?,
    };
    let mut rng = Rng(0x0928_2026_5a5a_0001);
    let mut failures = 0_usize;

    let down = Case::build(&gpu, &mut rng, "down", H, INTER)?;

    // 2026-09-25: 1. Bit identity of in-place and out-of-place staging.
    println!("== split-SiLU structural oracle (in-place staging, hard unequal=0) ==");
    failures += bit_identity_gate(&gpu, &kern, &down)?;

    // 2026-09-25: 2. Down projection: the two arms, numerics then time.
    println!(
        "\n== down N={H} K={INTER} (grid {}, 89.1 MB FP8/layer) ==",
        H.div_ceil(4)
    );

    let fused = Out::new(&gpu, H)?;
    let staged = Out::new(&gpu, H)?;
    for o in [&fused, &staged] {
        o.reset(&gpu)?;
    }
    ops::w8a16_gemv_silu_input(
        &gpu,
        kern.silu_input,
        down.gate,
        down.up,
        down.weight,
        down.scale,
        fused.ptr(),
        H,
        INTER,
        0,
    )?;
    staged_route(&gpu, &kern, &down, staged.ptr())?;
    gpu.synchronize(0)?;
    let (fused_b, staged_b) = (fused.read(&gpu)?, staged.read(&gpu)?);

    // 2026-09-25: Reported, not asserted to zero (see the module header).
    let (u1, a1, ulp1) = compare(&fused_b, &staged_b);
    println!("   split-SiLU vs fused silu_input: unequal={u1} max_abs={a1:.6} max_ulp={ulp1}");

    let t_fused = time_us(&gpu, || {
        ops::w8a16_gemv_silu_input(
            &gpu,
            kern.silu_input,
            down.gate,
            down.up,
            down.weight,
            down.scale,
            fused.ptr(),
            H,
            INTER,
            0,
        )
    })?;
    let t_staged = time_us(&gpu, || staged_route(&gpu, &kern, &down, staged.ptr()))?;
    println!(
        "   OLD fused silu_input         {t_fused:8.1} us  {:7.0} GB/s  (nsys: 103.9 us / 858 GB/s)",
        gbs(H, INTER, t_fused)
    );
    println!(
        "   NEW silu_mul + w8a16_gemv    {t_staged:8.1} us  {:7.0} GB/s  {:.2}x",
        gbs(H, INTER, t_staged),
        t_fused / t_staged
    );
    println!(
        "   target >= 1,700 GB/s (~52 us): staged {}",
        if gbs(H, INTER, t_staged) >= 1700.0 {
            "MET"
        } else {
            "miss"
        }
    );

    ensure!(
        failures == 0,
        "{failures} bit-identity gate(s) failed: the in-place SwiGLU staging the \
         decode arm runs is not the out-of-place one"
    );
    println!("\nALL PASS: the in-place SwiGLU staging is byte-identical to the staged route");
    Ok(())
}
