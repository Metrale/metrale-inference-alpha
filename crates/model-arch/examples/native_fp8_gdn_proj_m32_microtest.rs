// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: GPU oracle for `w8a16_gemm_pipelined_m32`, the 32-row M-tile
//! twin of `w8a16_gemm_pipelined`, at the block-scaled FP8 N x K shapes in
//! `SHAPES`. The GDN batched-verify arms take the twin at 1..=32 rows
//! (`ops::w8a16_gemm_pipelined_by_m`), and the multi-seq attention Q/K/V tier
//! takes its strided launch above 16 rows.
//!
//! Owner: model-arch examples (GPU microtests).
//! Invariants:
//! - The run exits 0 only if, for every shape at every M in `MS`, the twin's
//!   `[M, N]` output is byte-identical to `w8a16_gemm_pipelined`, the strided
//!   launch (A pitch K + `A_PAD`, C pitch N + `C_PAD`) reproduces it row for
//!   row, no launch wrote outside rows 0..M or into a guard or pitch gap, and
//!   the twin is within the tensor-core budget of
//!   `layers::dense_ffn::m16_tc::oracle` (and `REL_RMS_GATE`) against the
//!   per-row scalar `w8a16_gemv`. The known-bad controls in `controls` must
//!   each be refused.
//!
//! Each shape is also timed at M = 16 and 32, both tiles, over `REPS`
//! launches after one warmup, as µs per launch and weight GB/s.
//!
//! Run:
//!   cargo run --release -p metrale-model-arch --features cuda,gpu-examples \
//!     --example native_fp8_gdn_proj_m32_microtest
use anyhow::{Result, ensure};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_layers::layers::dense_ffn::m16_tc::oracle::{
    M16_TC_MAX_ULP, compare_m16_tc_block,
};
use metrale_model_layers::layers::ops;
use std::time::Instant;

const MAX_M: usize = 32;
const GUARD: usize = 64;
const SENTINEL: u8 = 0x5a;
const A_PAD: usize = 8; // 2026-09-25: elements; keeps strided A rows 16 B-aligned.
const C_PAD: usize = 64;
const REPS: usize = 20;
/// 2026-09-25: Block-level rel_rms gate, the value `common/m16_tc_compare.rs`
/// uses.
const REL_RMS_GATE: f64 = 1e-3;
const MS: [usize; 7] = [1, 5, 7, 16, 17, 24, 32];
/// 2026-09-25: (name, N, K).
const SHAPES: [(&str, usize, usize); 4] = [
    ("35B in_proj_qkvz", 12288, 2048),
    ("35B out_proj", 2048, 4096),
    ("27B in_proj_qkvz", 12288, 5120),
    ("27B out_proj", 5120, 4096),
];

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.0 >> 32) as u32
    }
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}

fn values(bytes: &[u8]) -> Vec<f64> {
    bytes
        .chunks_exact(2)
        .map(|x| f64::from(bf16::from_bits(u16::from_le_bytes([x[0], x[1]])).to_f32()))
        .collect()
}

/// 2026-09-25: A guarded output `[rows, pitch]` BF16 filled with `SENTINEL`;
/// `image` is that fill, used to reset the buffer and to compare against.
struct Out {
    base: DevicePtr,
    image: Vec<u8>,
}

impl Out {
    fn new(gpu: &dyn GpuBackend, rows: usize, pitch: usize) -> Result<Self> {
        let image = vec![SENTINEL; rows * pitch * 2 + 2 * GUARD];
        Ok(Self {
            base: upload(gpu, &image)?,
            image,
        })
    }
    fn ptr(&self) -> DevicePtr {
        self.base.offset(GUARD)
    }
    fn reset(&self, gpu: &dyn GpuBackend) -> Result<()> {
        gpu.copy_h2d(&self.image, self.base)
    }
    fn read(&self, gpu: &dyn GpuBackend) -> Result<Vec<u8>> {
        let mut v = vec![0_u8; self.image.len()];
        gpu.copy_d2h(self.base, &mut v)?;
        Ok(v)
    }
}

/// 2026-09-25: Bytes outside `live` must be sentinel; values inside must be
/// finite. `live` extents are byte offsets past the leading guard.
fn check_extent(
    observed: &[u8],
    sentinel: &[u8],
    live: &[(usize, usize)],
    who: &str,
) -> Result<()> {
    ensure!(observed.len() == sentinel.len(), "{who}: extent mismatch");
    let mut mask = vec![false; sentinel.len()];
    for &(start, len) in live {
        mask[GUARD + start..GUARD + start + len].fill(true);
    }
    for i in 0..sentinel.len() {
        if !mask[i] {
            ensure!(
                observed[i] == sentinel[i],
                "{who} wrote outside its extent at byte {i}"
            );
        }
    }
    for &(start, len) in live {
        ensure!(
            values(&observed[GUARD + start..GUARD + start + len])
                .iter()
                .all(|x| x.is_finite()),
            "{who}: nonfinite output"
        );
    }
    Ok(())
}

/// 2026-09-25: Byte-identical live extents. The extents may sit at different
/// pitches on the two sides; `live_a[i]` corresponds to `live_b[i]`.
fn check_bits(
    a: &[u8],
    live_a: &[(usize, usize)],
    b: &[u8],
    live_b: &[(usize, usize)],
    what: &str,
) -> Result<()> {
    ensure!(live_a.len() == live_b.len(), "{what}: row-count mismatch");
    for (&(sa, la), &(sb, lb)) in live_a.iter().zip(live_b) {
        ensure!(la == lb, "{what}: extent length mismatch");
        let (x, y) = (
            &a[GUARD + sa..GUARD + sa + la],
            &b[GUARD + sb..GUARD + sb + lb],
        );
        if x != y {
            let diff = x
                .chunks_exact(2)
                .zip(y.chunks_exact(2))
                .filter(|(p, q)| p != q)
                .count();
            anyhow::bail!("{what}: {diff} BF16 element(s) differ (row at byte {sa})");
        }
    }
    Ok(())
}

/// 2026-09-25: The tensor-core budget vs the scalar reference over `[m, n]`.
fn check_budget(
    actual: &[u8],
    reference: &[u8],
    m: usize,
    n: usize,
    k: usize,
) -> Result<(i32, f64)> {
    let a = &actual[GUARD..GUARD + m * n * 2];
    let r = &reference[GUARD..GUARD + m * n * 2];
    let d = compare_m16_tc_block(a, r, n, k);
    if let Some(o) = d.over_budget.first() {
        anyhow::bail!(
            "vs scalar: {} element(s) over the {M16_TC_MAX_ULP}-ULP / floor budget, first at \
             ({}, {}) scalar {:e} tile {:e} {} ULP",
            d.over_budget.len(),
            o.row,
            o.col,
            o.reference,
            o.actual,
            o.ulp
        );
    }
    ensure!(
        d.rel_rms <= REL_RMS_GATE,
        "vs scalar: rel_rms {:.3e} > {REL_RMS_GATE:.0e}",
        d.rel_rms
    );
    Ok((d.max_ulp, d.rel_rms))
}

fn rows(m: usize, n: usize, pitch: usize) -> Vec<(usize, usize)> {
    (0..m).map(|r| (r * pitch * 2, n * 2)).collect()
}

fn time_launch(gpu: &dyn GpuBackend, mut f: impl FnMut() -> Result<()>) -> Result<f64> {
    f()?;
    gpu.synchronize(0)?;
    let t = Instant::now();
    for _ in 0..REPS {
        f()?;
    }
    gpu.synchronize(0)?;
    Ok(t.elapsed().as_secs_f64() * 1e6 / REPS as f64)
}

fn main() -> Result<()> {
    let gpu = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let scalar = gpu.kernel("w8a16_gemv", "w8a16_gemv")?;
    let full = gpu.kernel("w8a16_gemm_pipelined", "w8a16_gemm_pipelined")?;
    let m32 = gpu.kernel("w8a16_gemm_pipelined_m32", "w8a16_gemm_pipelined_m32")?;
    let mut rng = Lcg(0x0323_8a16_2026_u64);
    let mut failures = 0usize;
    let mut controls_done = false;

    for (name, n, k) in SHAPES {
        let weight: Vec<u8> = (0..n * k)
            .map(|_| {
                let x = rng.next();
                ((x % 127) as u8) | (((x >> 7) & 1) as u8 * 128)
            })
            .collect();
        let scale: Vec<u8> = (0..(n / 128) * (k / 128))
            .flat_map(|_| (((rng.next() % 16 + 1) as f32) / 1024.0).to_le_bytes())
            .collect();
        let acts: Vec<u16> = (0..MAX_M * k)
            .map(|_| bf16::from_f32(((rng.next() % 2049) as f32 - 1024.0) / 1024.0).to_bits())
            .collect();
        let acts_bytes: Vec<u8> = acts.iter().flat_map(|v| v.to_le_bytes()).collect();
        // 2026-09-25: Strided A: the same rows at pitch K + A_PAD, with the pad
        // elements set to +inf so a kernel that reads past K shows up in the
        // bits.
        let acts_strided: Vec<u8> = (0..MAX_M)
            .flat_map(|r| {
                let mut row: Vec<u16> = acts[r * k..(r + 1) * k].to_vec();
                row.extend(std::iter::repeat_n(0x7f80_u16, A_PAD));
                row
            })
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let w = upload(&gpu, &weight)?;
        let s = upload(&gpu, &scale)?;
        let a = upload(&gpu, &acts_bytes)?;
        let a_s = upload(&gpu, &acts_strided)?;
        let out_full = Out::new(&gpu, MAX_M, n)?;
        let out_m32 = Out::new(&gpu, MAX_M, n)?;
        let out_scalar = Out::new(&gpu, MAX_M, n)?;
        let out_strided = Out::new(&gpu, MAX_M, n + C_PAD)?;
        let weight_bytes = (n * k) as f64;

        for m in MS {
            for o in [&out_full, &out_m32, &out_scalar, &out_strided] {
                o.reset(&gpu)?;
            }
            let (mu, nu, ku) = (m as u32, n as u32, k as u32);
            ops::w8a16_gemm_pipelined(&gpu, full, a, w, s, out_full.ptr(), mu, nu, ku, 0)?;
            ops::w8a16_gemm_pipelined_m32(&gpu, m32, a, w, s, out_m32.ptr(), mu, nu, ku, 0)?;
            for r in 0..m {
                ops::w8a16_gemv(
                    &gpu,
                    scalar,
                    a.offset(r * k * 2),
                    w,
                    s,
                    out_scalar.ptr().offset(r * n * 2),
                    nu,
                    ku,
                    0,
                )?;
            }
            ops::w8a16_gemm_pipelined_m32_strided(
                &gpu,
                m32,
                a_s,
                w,
                s,
                out_strided.ptr(),
                mu,
                nu,
                ku,
                (k + A_PAD) as u32,
                (n + C_PAD) as u32,
                0,
            )?;
            gpu.synchronize(0)?;
            let (full_o, m32_o, sc_o, st_o) = (
                out_full.read(&gpu)?,
                out_m32.read(&gpu)?,
                out_scalar.read(&gpu)?,
                out_strided.read(&gpu)?,
            );
            let live = rows(m, n, n);
            let live_s = rows(m, n, n + C_PAD);

            if !controls_done {
                controls(&m32_o, &out_m32.image, &live, &sc_o, m, n, k)?;
                controls_done = true;
            }

            let verdict = (|| -> Result<(i32, f64)> {
                check_extent(&full_o, &out_full.image, &live, "128-tile")?;
                check_extent(&m32_o, &out_m32.image, &live, "32-tile")?;
                check_extent(&st_o, &out_strided.image, &live_s, "32-tile strided")?;
                check_bits(&m32_o, &live, &full_o, &live, "32-tile vs 128-tile")?;
                check_bits(
                    &st_o,
                    &live_s,
                    &m32_o,
                    &live,
                    "strided vs contiguous 32-tile",
                )?;
                check_budget(&m32_o, &sc_o, m, n, k)
            })();
            match verdict {
                Ok((max_ulp, rel_rms)) => println!(
                    "PASS {name} M={m} N={n} K={k}: bits==128-tile, strided==contiguous, \
                     vs scalar max_ulp={max_ulp} rel_rms={rel_rms:.3e}"
                ),
                Err(e) => {
                    println!("FAIL {name} M={m} N={n} K={k}: {e}");
                    failures += 1;
                }
            }
        }

        // 2026-09-25: Timing: both tiles at M=16 and M=32.
        for m in [16usize, 32] {
            let (mu, nu, ku) = (m as u32, n as u32, k as u32);
            let t_full = time_launch(&gpu, || {
                ops::w8a16_gemm_pipelined(&gpu, full, a, w, s, out_full.ptr(), mu, nu, ku, 0)
            })?;
            let t_m32 = time_launch(&gpu, || {
                ops::w8a16_gemm_pipelined_m32(&gpu, m32, a, w, s, out_m32.ptr(), mu, nu, ku, 0)
            })?;
            println!(
                "TIME {name} M={m}: 128-tile {t_full:.1} us ({:.0} GB/s)  32-tile {t_m32:.1} us \
                 ({:.0} GB/s)  speedup {:.2}x",
                weight_bytes / t_full / 1e3,
                weight_bytes / t_m32 / 1e3,
                t_full / t_m32
            );
        }
    }

    ensure!(failures == 0, "{failures} case(s) failed");
    println!(
        "ALL PASS: w8a16_gemm_pipelined_m32 byte-identical to w8a16_gemm_pipelined at M in \
         {MS:?} on {} shapes, strided == contiguous, within the tensor-core budget vs scalar",
        SHAPES.len()
    );
    Ok(())
}

/// 2026-09-25: Known-bad controls, run once against the real checks: a
/// one-bit output flip must fail the byte comparison; a write one byte past
/// row M (when M < `MAX_M`) and a guard write must fail the extent check; and
/// flipping the top exponent bit (BF16 bit 14) of the twin's output at the
/// column of the scalar row 0's largest |value| must fail the scalar budget.
fn controls(
    m32_o: &[u8],
    image: &[u8],
    live: &[(usize, usize)],
    sc_o: &[u8],
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    let mut flipped = m32_o.to_vec();
    flipped[GUARD] ^= 1;
    let e = check_bits(&flipped, live, m32_o, live, "control").expect_err("bit flip admitted");
    println!("KNOWN_BAD output-bit: refused: {e}");

    let mut past_m = m32_o.to_vec();
    if m < MAX_M {
        past_m[GUARD + m * n * 2] ^= 1;
        let e = check_extent(&past_m, image, live, "control").expect_err("row past M admitted");
        println!("KNOWN_BAD row-past-M: refused: {e}");
    }
    let mut guard = m32_o.to_vec();
    guard[0] ^= 1;
    let e = check_extent(&guard, image, live, "control").expect_err("guard write admitted");
    println!("KNOWN_BAD guard: refused: {e}");

    // 2026-09-25: Largest |value| of row 0, so the flipped element is not a
    // near-zero value the absolute floor could admit.
    let row0 = values(&sc_o[GUARD..GUARD + n * 2]);
    let idx = (0..n)
        .max_by(|&i, &j| row0[i].abs().total_cmp(&row0[j].abs()))
        .unwrap_or(0);
    let mut exp = m32_o.to_vec();
    exp[GUARD + idx * 2 + 1] ^= 0x40;
    let e = check_budget(&exp, sc_o, m, n, k).expect_err("exponent flip admitted");
    println!("KNOWN_BAD exponent-bit: refused: {e}");
    Ok(())
}
