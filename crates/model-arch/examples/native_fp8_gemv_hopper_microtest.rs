// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: GPU oracle for the M=1 W8A16 decode GEMVs `w8a16_gemv` and
//! `w8a16_gemv_dual` of whichever target the build is for (the
//! `kernels/hopper/common` sources on a hopper build, the gb10 ones
//! otherwise), against a host model of the gb10 reduction order.
//!
//! Owner: model-arch examples (GPU microtests).
//! Invariants:
//! - The run exits 0 only if every output of every shape in `SHAPES`, and both
//!   projections of the dual, is byte-equal to [`host_gemv`], and no launch
//!   wrote outside its guarded output.
//!
//! The reference is a host model because a hopper build holds no gb10 kernel
//! to compare against: `kernels/hopper/common/w8a16_gemv.cu` replaces the gb10
//! source. [`host_gemv`] reproduces `kernels/gb10/common/w8a16_gemv.cu`'s
//! order (64 lanes per output walking the same chunks, a separate multiply and
//! add per element under `--fmad=false`, the 5-step
//! `__shfl_down_sync` tree, the two-warp add, one `__float2bfloat16`), so
//! byte equality means the target's kernel kept that arithmetic and its order.
//! The dual is checked the same way, one projection at a time.
//!
//! Each shape is also timed (`synchronize` plus a host `Instant` over `REPS`
//! after `WARMUP` runs; `GpuBackend` has no event elapsed-time query) and
//! reported as µs, GB/s of FP8 weight bytes, and ms per decode step at the
//! shape's launch count. There is no "before" column: a build holds only its
//! own target's kernel.
//!
//! Run (the kernel is selected by the build target, not by an env var):
//!     cargo run --release --example native_fp8_gemv_hopper_microtest \
//!       --features cuda,gpu-examples
use anyhow::{Result, ensure};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_model_layers::layers::ops;
use std::time::Instant;

/// 2026-09-25: (name, N, K, launches per decode step). The launch counts
/// follow the layer counts of `kernels/hopper/qwen3.8-27b/MODEL.toml`: 64
/// layers, 48 linear-attention, 16 full-attention.
const SHAPES: &[(&str, u32, u32, u32)] = &[
    ("ffn down", 5120, 17408, 64),
    ("ssm in_proj_qkvz", 16384, 5120, 48),
    ("ssm out_proj", 5120, 6144, 48),
    ("attn q", 12288, 5120, 16),
    ("attn k/v", 1024, 5120, 32),
    ("attn o", 5120, 6144, 16),
];
/// 2026-09-25: Gate and up `(N, K)`, one `w8a16_gemv_dual` launch per layer.
const DUAL: (u32, u32) = (17408, 5120);
const GUARD: usize = 64;
const REPS: u32 = 20;
const WARMUP: u32 = 3;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.0 >> 32) as u32
    }
    /// 2026-09-25: BF16 bits in [-1, 1], in steps of 1/1024. Bits rather
    /// than bytes, so the host oracle and the uploaded buffer hold the same
    /// values and not two decodings of them.
    fn act(&mut self) -> u16 {
        half::bf16::from_f32(((self.next() % 2049) as f32 - 1024.0) / 1024.0).to_bits()
    }
    /// 2026-09-25: An E4M3 byte from 0x00..=0x7E and 0x80..=0xFE. 0x7F/0xFF,
    /// the format's NaN codes, are excluded: they are the only codes where the
    /// gb10 LUT (+-0) and the Hopper `cvt` (NaN) decodes differ.
    fn e4m3(&mut self) -> u8 {
        let x = self.next();
        ((x % 127) as u8) | (((x >> 7) & 1) as u8) << 7
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

/// 2026-09-25: K values one lane consumes per chunk, one `uint4` of FP8 bytes:
/// `K_PER_CHUNK` in `kernels/hopper/common/w8a16_gemv_hopper.cuh`, the gb10
/// kernel's `k16 * 16` stride.
const K_PER_CHUNK: u32 = 16;
/// 2026-09-25: Lanes cooperating on one output, `BLOCK_SIZE / N_PER_BLOCK` =
/// 256 / 4.
const LANES_PER_OUT: u32 = 64;
/// 2026-09-25: One FP8 block-scale tile, in both N and K.
const FP8_BLOCK: u32 = 128;

/// 2026-09-25: E4M3 byte -> the FP32 value `E4M3_LUT` holds for it.
///
/// Computed from the format rather than pasted as 256 literals, so the oracle
/// cannot drift from the table by a transcription slip: sign, 4 exponent bits
/// (bias 7), 3 mantissa bits, subnormal when the exponent field is 0. Every
/// finite E4M3 is exactly representable in FP32. 0x7F/0xFF, which the LUT maps
/// to +-0 and `cvt` to NaN, are excluded from the drawn alphabet
/// ([`Rng::e4m3`]) and so never reach here.
fn e4m3_to_f32(byte: u8) -> f32 {
    let sign = if byte & 0x80 != 0 { -1.0_f32 } else { 1.0 };
    let exp = i32::from((byte >> 3) & 0x0F);
    let mant = f32::from(byte & 0x07);
    if exp == 0 {
        // 2026-09-25: Subnormal: mant * 2^-9 (no implicit leading 1, bias 7,
        // 3 mantissa bits).
        sign * mant * 2.0_f32.powi(-9)
    } else {
        sign * (1.0 + mant / 8.0) * 2.0_f32.powi(exp - 7)
    }
}

/// 2026-09-25: The oracle: `kernels/gb10/common/w8a16_gemv.cu`, evaluated on
/// the host with its reduction order preserved element for element.
///
/// Every step is the kernel's, in the kernel's order, because a reference that
/// merely computes the same dot product would pass a kernel that
/// reassociated the sum:
///
///  * lane `l` of 64 walks chunks `l, l+64, l+128, …`;
///  * inside a chunk, elements 0..15 in index order, each `w = LUT[b] * scale`
///    then `acc += a * w` as a separate multiply and add (`--fmad=false` in
///    `kernels/gb10/common/KERNEL.toml`; Rust does not contract either);
///  * a 5-step `__shfl_down_sync` tree per 32-lane warp. An out-of-range
///    source lane returns the caller's own value, which is reproduced here
///    rather than skipped; it does not reach lane 0;
///  * the two warps' lane-0 values added in smem order, then one
///    `__float2bfloat16` (round-to-nearest-even, as `bf16::from_f32` is).
///
/// Returns `[1, N]` BF16 little-endian bytes, the kernel's own output layout,
/// so the comparison is on bytes and never on a re-decoded float.
fn host_gemv(n: u32, k: u32, weight: &[u8], scale: &[f32], act: &[u16]) -> Vec<u8> {
    let chunks = k / K_PER_CHUNK;
    let k_blocks = k.div_ceil(FP8_BLOCK);
    let mut out = Vec::with_capacity(n as usize * 2);
    for row in 0..n {
        let n_block = row / FP8_BLOCK;
        let base = row as usize * k as usize;
        let mut part = [0.0_f32; 64];
        for (l, acc) in part.iter_mut().enumerate() {
            let mut chunk = l as u32;
            while chunk < chunks {
                let first = chunk * K_PER_CHUNK;
                let s = scale[(n_block * k_blocks + first / FP8_BLOCK) as usize];
                for j in 0..K_PER_CHUNK {
                    let idx = (first + j) as usize;
                    let w = e4m3_to_f32(weight[base + idx]) * s;
                    *acc += half::bf16::from_bits(act[idx]).to_f32() * w;
                }
                chunk += LANES_PER_OUT;
            }
        }
        let mut warp_sum = [0.0_f32; 2];
        for (w, sum) in warp_sum.iter_mut().enumerate() {
            let mut v = [0.0_f32; 32];
            v.copy_from_slice(&part[w * 32..w * 32 + 32]);
            for offset in [16_usize, 8, 4, 2, 1] {
                let snap = v;
                for (i, cell) in v.iter_mut().enumerate() {
                    // 2026-09-25: `__shfl_down_sync` past the warp returns the
                    // caller's own value; that is `snap[i] + snap[i]`, not a
                    // skipped add.
                    *cell = snap[i] + snap[if i + offset < 32 { i + offset } else { i }];
                }
            }
            *sum = v[0];
        }
        out.extend_from_slice(
            &half::bf16::from_f32(warp_sum[0] + warp_sum[1])
                .to_bits()
                .to_le_bytes(),
        );
    }
    out
}

struct Kernels {
    gemv: KernelHandle,
    dual: KernelHandle,
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

/// 2026-09-25: One projection's inputs, on the device for the kernel and kept
/// on the host for [`host_gemv`], so the oracle reads the same bytes the kernel
/// does.
struct Case {
    n: u32,
    k: u32,
    weight: DevicePtr,
    scale: DevicePtr,
    input: DevicePtr,
    weights: Vec<u8>,
    scales: Vec<f32>,
    act: Vec<u16>,
}

impl Case {
    fn build(gpu: &dyn GpuBackend, rng: &mut Rng, n: u32, k: u32) -> Result<Self> {
        let (nn, kk) = (n as usize, k as usize);
        let weights: Vec<u8> = (0..nn * kk).map(|_| rng.e4m3()).collect();
        let scales: Vec<f32> = (0..nn.div_ceil(128) * kk.div_ceil(128))
            .map(|_| ((rng.next() % 16 + 1) as f32) / 1024.0)
            .collect();
        let act: Vec<u16> = (0..kk).map(|_| rng.act()).collect();
        let scale_bytes: Vec<u8> = scales.iter().flat_map(|s| s.to_le_bytes()).collect();
        let input: Vec<u8> = act.iter().flat_map(|a| a.to_le_bytes()).collect();
        Ok(Self {
            n,
            k,
            weight: upload(gpu, &weights)?,
            scale: upload(gpu, &scale_bytes)?,
            input: upload(gpu, &input)?,
            weights,
            scales,
            act,
        })
    }
    /// 2026-09-25: The oracle's answer for this case, over `act` (the dual's
    /// two projections share one activation, so it is passed rather than read).
    fn expected(&self, act: &[u16]) -> Vec<u8> {
        host_gemv(self.n, self.k, &self.weights, &self.scales, act)
    }
}

/// 2026-09-25: One shape: bit identity against the host oracle, then the
/// timing. Returns the failure count (0 or 1).
fn shape(
    gpu: &dyn GpuBackend,
    k: &Kernels,
    rng: &mut Rng,
    row: (&str, u32, u32, u32),
) -> Result<usize> {
    let (name, n, kk, launches) = row;
    let c = Case::build(gpu, rng, n, kk)?;
    let t = Out::new(gpu, n)?;
    t.reset(gpu)?;
    ops::w8a16_gemv(gpu, k.gemv, c.input, c.weight, c.scale, t.ptr(), n, kk, 0)?;
    gpu.synchronize(0)?;
    let unequal = c
        .expected(&c.act)
        .chunks_exact(2)
        .zip(t.read(gpu)?.chunks_exact(2))
        .filter(|(a, b)| a != b)
        .count();

    let new_us = time_us(gpu, || {
        ops::w8a16_gemv(gpu, k.gemv, c.input, c.weight, c.scale, t.ptr(), n, kk, 0)
    })?;
    println!(
        "  [{}] {name:<16} N={n:<6} K={kk:<6} grid={:<5} unequal={unequal}  \
         {new_us:8.1} us {:7.0} GB/s  ({launches}x/step: {:.2} ms)",
        if unequal == 0 { "PASS" } else { "FAIL" },
        n.div_ceil(4),
        gbs(n, kk, new_us),
        new_us * f64::from(launches) / 1e3,
    );
    Ok(usize::from(unequal != 0))
}

/// 2026-09-25: The dual: each projection must equal the oracle over its own
/// weights and the one activation the two share.
fn dual(gpu: &dyn GpuBackend, k: &Kernels, rng: &mut Rng) -> Result<usize> {
    let (n, kk) = DUAL;
    let (gate, up) = (Case::build(gpu, rng, n, kk)?, Case::build(gpu, rng, n, kk)?);
    let outs: Vec<Out> = (0..2).map(|_| Out::new(gpu, n).unwrap()).collect();
    for o in &outs {
        o.reset(gpu)?;
    }
    // 2026-09-25: The dual shares one activation; the second case uses the
    // first's.
    let (a, act) = (gate.input, &gate.act);
    let launch = || {
        ops::w8a16_gemv_dual(
            gpu,
            k.dual,
            a,
            gate.weight,
            gate.scale,
            outs[0].ptr(),
            up.weight,
            up.scale,
            outs[1].ptr(),
            n,
            kk,
            0,
        )
    };
    launch()?;
    gpu.synchronize(0)?;
    let mut unequal = 0;
    for (case, out) in [(&gate, &outs[0]), (&up, &outs[1])] {
        unequal += case
            .expected(act)
            .chunks_exact(2)
            .zip(out.read(gpu)?.chunks_exact(2))
            .filter(|(x, y)| x != y)
            .count();
    }

    let new_us = time_us(gpu, launch)?;
    println!(
        "  [{}] {:<16} N=2x{n:<4} K={kk:<6} grid={:<5} unequal={unequal}  \
         {new_us:8.1} us {:7.0} GB/s  (64x/step: {:.2} ms)",
        if unequal == 0 { "PASS" } else { "FAIL" },
        "gate+up dual",
        n.div_ceil(4),
        gbs(2 * n, kk, new_us),
        new_us * 64.0 / 1e3,
    );
    Ok(usize::from(unequal != 0))
}

fn main() -> Result<()> {
    let gpu = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let kern = Kernels {
        gemv: gpu.kernel("w8a16_gemv", "w8a16_gemv")?,
        dual: gpu.kernel("w8a16_gemv_fused", "w8a16_gemv_dual")?,
    };
    let mut rng = Rng(0x0928_2026_5a5a_0010);
    let mut failures = 0_usize;

    println!(
        "== W8A16 M=1 decode GEMV: the target's kernel vs the gb10 chain ==\n\
         == reference = host_gemv, an exact model of that chain's reduction \
         order; hard unequal=0 =="
    );
    for row in SHAPES {
        failures += shape(&gpu, &kern, &mut rng, *row)?;
    }
    failures += dual(&gpu, &kern, &mut rng)?;

    println!(
        "\n{}",
        if failures == 0 {
            "ALL SHAPES BIT-IDENTICAL"
        } else {
            "FAILURES — the override changed the arithmetic, not just the instructions"
        }
    );
    ensure!(
        failures == 0,
        "{failures} shape(s) diverged from the gb10 chain"
    );
    Ok(())
}
