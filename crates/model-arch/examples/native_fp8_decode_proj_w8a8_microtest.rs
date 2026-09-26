// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The 5..=16-row decode projections at Qwen3.8-27B shapes: the
//! `w8a16_gemv_batch16[_strided]` GEMV against the W8A8 block-scaled cuBLASLt
//! route (`ops::decode_w8a8_quant_act` + `ops::decode_w8a8_gemm`).
//!
//! Owner: model-arch examples (GPU microtests).
//! Invariants:
//! - The run exits 0 only if, for every projection at every M in `ROWS`, the
//!   W8A8 output has cosine >= `COSINE_GATE` and relative RMS <= the gate
//!   against the GEMV over rows 0..M, every value in rows 0..`MAX_M` of the
//!   projection's own columns is finite, and every other byte of its arena
//!   still holds `SENTINEL`.
//!
//! W8A8 quantizes the activation to E4M3 per token and 128-wide K group; the
//! GEMV reads BF16 activations. The two differ by that quantization, so the
//! gate is a tolerance, not byte equality. `METRALE_W8A8_REL_RMS_GATE`
//! overrides the relative-RMS gate (`REL_RMS_GATE` otherwise).
//!
//! cuBLASLt is given `m_pad = ceil16(M)` rows, which is `MAX_M` (16) for every
//! M in `ROWS`, and writes rows M..16 too. Those rows must stay finite and
//! inside the projection's own columns.
//!
//! Each projection is also timed over `REPS` launches per route and reported
//! as µs and as weight bytes per second.
//!
//! Attention shapes from `kernels/hopper/qwen3.8-27b/MODEL.toml`: hidden 5120,
//! head_dim 256, 24 q-heads / 4 kv-heads, output gate on, so `q_proj` = 12288,
//! kv 1024, a [Q|K|V] slot of 14336 elements, and `o_proj` N=5120 over K=6144.
//! The GDN arms use `QKVZ_N` = 16384 over K=5120 for `in_proj_qkvz` and
//! N=5120 over `VALUE_DIM` = 6144 for `out_proj`.
//!
//! Run: `cargo run --release -p metrale-model-arch --features
//! cuda,gpu-examples --example native_fp8_decode_proj_w8a8_microtest`. The
//! example calls the two ops directly, so `ops::decode_w8a8_selected` and the
//! levers it reads are not consulted.

use anyhow::{Result, ensure};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_model_layers::layers::ops;
use std::time::Instant;

const H: usize = 5120;
const QKVZ_N: usize = 16384;
const VALUE_DIM: usize = 6144;
const Q_PROJ_DIM: usize = 12288;
const KV_DIM: usize = 1024;
const PER_SEQ_QKV: usize = Q_PROJ_DIM + 2 * KV_DIM;
const O_K: usize = 6144;
const MAX_M: usize = 16;
const ROWS: [usize; 3] = [5, 8, 16];
const REPS: usize = 20;
const GUARD: usize = 64;
const SENTINEL: u8 = 0x5a;
const COSINE_GATE: f64 = 0.999;
const REL_RMS_GATE: f64 = 3e-2;

/// 2026-09-25: One projection under test. `ldc` is the BF16 elements between
/// output rows (equal to `n` when contiguous) and `offset` this projection's
/// BF16 element offset inside a row; `act` is `[MAX_M, k]` BF16.
struct Proj {
    name: &'static str,
    n: usize,
    k: usize,
    ldc: usize,
    offset: usize,
    weight: DevicePtr,
    scale: DevicePtr,
    act: DevicePtr,
}

impl Proj {
    fn strided(&self) -> bool {
        self.ldc != self.n
    }
    /// 2026-09-25: Live extents (byte offset, byte length) for `rows` output
    /// rows.
    fn live(&self, rows: usize) -> Vec<(usize, usize)> {
        (0..rows)
            .map(|r| ((r * self.ldc + self.offset) * 2, self.n * 2))
            .collect()
    }
    fn weight_bytes(&self) -> f64 {
        (self.n * self.k) as f64
    }
}

struct Kernels {
    batch16: KernelHandle,
    batch16_strided: KernelHandle,
    quant: ops::Fp8ActQuant,
    kmajor: KernelHandle,
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}

fn values(bytes: &[u8]) -> Vec<f64> {
    bytes
        .chunks_exact(2)
        .map(|x| bf16::from_bits(u16::from_le_bytes([x[0], x[1]])).to_f32() as f64)
        .collect()
}

/// 2026-09-25: Cosine and relative RMS over the live extents.
fn metrics(observed: &[u8], baseline: &[u8], live: &[(usize, usize)]) -> (f64, f64) {
    let (mut dot, mut na, mut nb, mut se, mut ss) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for &(start, len) in live {
        let a = values(&observed[GUARD + start..GUARD + start + len]);
        let b = values(&baseline[GUARD + start..GUARD + start + len]);
        for (x, y) in a.iter().zip(b.iter()) {
            dot += x * y;
            na += x * x;
            nb += y * y;
            se += (x - y) * (x - y);
            ss += y * y;
        }
    }
    let cosine = if na > 0.0 && nb > 0.0 {
        dot / (na.sqrt() * nb.sqrt())
    } else {
        0.0
    };
    let rel_rms = if ss > 0.0 { (se / ss).sqrt() } else { 0.0 };
    (cosine, rel_rms)
}

/// 2026-09-25: The oracle. `live` is what the route is compared on (rows
/// 0..m); `written` is everything it may write (rows 0..m_pad, since cuBLASLt
/// writes the pad rows). Outside `written` the sentinel must survive; inside
/// it every value must be finite, pad rows included.
fn check(
    observed: &[u8],
    baseline: &[u8],
    sentinel: &[u8],
    live: &[(usize, usize)],
    written: &[(usize, usize)],
    gate: f64,
) -> Result<()> {
    ensure!(
        observed.len() == sentinel.len() && baseline.len() == sentinel.len(),
        "output extent mismatch"
    );
    let mut mask = vec![false; sentinel.len()];
    for &(start, len) in written {
        mask[GUARD + start..GUARD + start + len].fill(true);
    }
    for i in 0..sentinel.len() {
        if !mask[i] {
            ensure!(
                observed[i] == sentinel[i],
                "route wrote outside its extent at byte {i}"
            );
        }
    }
    for &(start, len) in written {
        ensure!(
            values(&observed[GUARD + start..GUARD + start + len])
                .iter()
                .all(|x| x.is_finite()),
            "nonfinite projection output (phantom rows included)"
        );
    }
    let (cosine, rel_rms) = metrics(observed, baseline, live);
    ensure!(cosine >= COSINE_GATE, "cosine {cosine:.6} < {COSINE_GATE}");
    ensure!(rel_rms <= gate, "rel_rms {rel_rms:.4e} > {gate:.1e}");
    Ok(())
}

/// 2026-09-25: The GEMV reference: one `w8a16_gemv_batch16[_strided]` launch
/// on the BF16 activations.
fn run_batch16(
    gpu: &dyn GpuBackend,
    k: &Kernels,
    p: &Proj,
    out: DevicePtr,
    m: usize,
) -> Result<()> {
    if p.strided() {
        return ops::w8a16_gemv_batch16_strided(
            gpu,
            k.batch16_strided,
            p.act,
            p.weight,
            p.scale,
            out.offset(p.offset * 2),
            m as u32,
            p.n as u32,
            p.k as u32,
            p.k as u32,
            p.ldc as u32,
            0,
        );
    }
    ops::w8a16_gemv_batch16(
        gpu, k.batch16, p.act, p.weight, p.scale, out, m as u32, p.n as u32, p.k as u32, 0,
    )
}

/// 2026-09-25: The route under test: `ops::decode_w8a8_quant_act`, then
/// `ops::decode_w8a8_gemm`, the pair the attention layer calls
/// (`qwen3_attention/trait_impl/multi_seq/w8a8_decode.rs`).
fn run_w8a8(
    gpu: &dyn GpuBackend,
    scratch: &ops::DecodeW8a8Scratch,
    p: &Proj,
    out: DevicePtr,
    m: usize,
) -> Result<()> {
    ops::decode_w8a8_quant_act(gpu, scratch, p.act, m as u32, p.k as u32, 0)?;
    let plan = if p.strided() {
        ops::DecodeW8a8Plan::strided(m, p.n as u32, p.k as u32, p.ldc as u32, usize::MAX)
    } else {
        ops::DecodeW8a8Plan::contiguous(m, p.n as u32, p.k as u32, usize::MAX)
    };
    let w = metrale_model_layers::weight_map::Fp8Weight {
        weight: p.weight,
        row_scale: p.scale,
        n: p.n as u32,
        k: p.k as u32,
        scale_format: metrale_model_layers::weight_map::WeightQuantFormat::Fp8BlockScaled,
    };
    ops::decode_w8a8_gemm(scratch, &w, out.offset(p.offset * 2), &plan, 0)
}

fn main() -> Result<()> {
    let gpu = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let k = Kernels {
        batch16: gpu.kernel("w8a16_gemv_batch4", "w8a16_gemv_batch16")?,
        batch16_strided: gpu.kernel("w8a16_gemv_batch4", "w8a16_gemv_batch16_strided")?,
        quant: ops::Fp8ActQuant::resolve(&gpu),
        kmajor: gpu.kernel("fp8_scale_transpose", "fp8_act_scale_to_kmajor")?,
    };
    let gate: f64 = std::env::var("METRALE_W8A8_REL_RMS_GATE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(REL_RMS_GATE);
    println!(
        "gates: cosine >= {COSINE_GATE}, rel_rms <= {gate:.1e} \
         (E4M3 activation-quant floor is ~2-2.6e-2; see the module header)"
    );

    // 2026-09-25: A struct, not a nest of closures: three generators sharing
    // one `random` closure would each hold a mutable borrow of it, and this file
    // should not depend on the order in which they happen to be last used.
    struct Rng(u64);
    impl Rng {
        fn bits(&mut self) -> u32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
            (self.0 >> 32) as u32
        }
        /// 2026-09-25: FP8 E4M3 bytes with a random sign and a magnitude code
        /// of 0x00..=0x7E, so never the NaN code 0x7F.
        fn fp8(&mut self, n: usize, depth: usize) -> Vec<u8> {
            (0..n * depth)
                .map(|_| {
                    let x = self.bits();
                    ((x % 127) as u8) | (((x >> 7) & 1) as u8 * 128)
                })
                .collect()
        }
        /// 2026-09-25: One FP32 scale per 128x128 weight block.
        fn scales(&mut self, n: usize, depth: usize) -> Vec<u8> {
            (0..(n / 128) * (depth / 128))
                .flat_map(|_| (((self.bits() % 16 + 1) as f32) / 1024.0).to_le_bytes())
                .collect()
        }
        fn acts(&mut self, elems: usize) -> Vec<u8> {
            (0..elems)
                .flat_map(|_| {
                    bf16::from_f32(((self.bits() % 2049) as f32 - 1024.0) / 1024.0)
                        .to_bits()
                        .to_le_bytes()
                })
                .collect()
        }
    }
    let mut rng = Rng(0x0927_2026_5A5A_0001);

    // 2026-09-25: Two activations: `[MAX_M, H]` feeds qkvz and q/k/v,
    // `[MAX_M, 6144]` feeds the GDN out_proj and the attention o_proj.
    let act_h = upload(&gpu, &rng.acts(MAX_M * H))?;
    let act_v = upload(&gpu, &rng.acts(MAX_M * O_K))?;

    let mk = |rng: &mut Rng, name, n: usize, kk: usize, ldc: usize, offset, act| -> Result<Proj> {
        Ok(Proj {
            name,
            n,
            k: kk,
            ldc,
            offset,
            weight: upload(&gpu, &rng.fp8(n, kk))?,
            scale: upload(&gpu, &rng.scales(n, kk))?,
            act,
        })
    };
    // 2026-09-25: Projections with the same row pitch share one arena, and
    // q/k/v sit at their own offsets in the `PER_SEQ_QKV` row, so a write into
    // a neighbour's slot breaks the sentinel.
    let ssm_qkvz = mk(&mut rng, "ssm in_proj_qkvz", QKVZ_N, H, QKVZ_N, 0, act_h)?;
    let ssm_out = mk(&mut rng, "ssm out_proj", H, VALUE_DIM, H, 0, act_v)?;
    let q = mk(
        &mut rng,
        "attn q_proj",
        Q_PROJ_DIM,
        H,
        PER_SEQ_QKV,
        0,
        act_h,
    )?;
    let kp = mk(
        &mut rng,
        "attn k_proj",
        KV_DIM,
        H,
        PER_SEQ_QKV,
        Q_PROJ_DIM,
        act_h,
    )?;
    let v = mk(
        &mut rng,
        "attn v_proj",
        KV_DIM,
        H,
        PER_SEQ_QKV,
        Q_PROJ_DIM + KV_DIM,
        act_h,
    )?;
    let o = mk(&mut rng, "attn o_proj", H, O_K, H, 0, act_v)?;
    let projections = [&ssm_qkvz, &ssm_out, &q, &kp, &v, &o];

    // 2026-09-25: One output arena per row pitch, sized for all `MAX_M` rows.
    let mut arenas: Vec<(usize, Vec<u8>, DevicePtr)> = Vec::new();
    for ldc in [QKVZ_N, H, PER_SEQ_QKV] {
        let sentinel = vec![SENTINEL; MAX_M * ldc * 2 + 2 * GUARD];
        let base = upload(&gpu, &sentinel)?;
        arenas.push((ldc, sentinel, base));
    }
    let arena = |ldc: usize| arenas.iter().find(|a| a.0 == ldc).expect("arena");

    // 2026-09-25: The activation-quant scratch, sized for `MAX_M` rows at the
    // larger K.
    let kg = H.max(O_K) / 128;
    let scratch = ops::DecodeW8a8Scratch {
        act_fp8: gpu.alloc(MAX_M * H.max(O_K))?,
        act_fp8_bytes: MAX_M * H.max(O_K),
        act_scale: gpu.alloc(MAX_M * kg * 4)?,
        act_scale_bytes: MAX_M * kg * 4,
        act_scale_kmajor: gpu.alloc(MAX_M * kg * 4)?,
        act_scale_kmajor_bytes: MAX_M * kg * 4,
        quant_k: k.quant,
        scale_kmajor_k: k.kmajor,
    };

    let mut failures = 0usize;
    let mut controls_done = false;
    for m in ROWS {
        println!("\n=== M = {m} (cuBLASLt pad = {MAX_M}) ===");
        for p in projections {
            let (ldc, sentinel, base) = arena(p.ldc);
            let (ldc, base) = (*ldc, *base);
            let out = base.offset(GUARD);
            let live = p.live(m);
            // 2026-09-25: The W8A8 route writes rows 0..MAX_M (`m_pad`).
            let padded = p.live(MAX_M);

            let capture = |w8a8: bool| -> Result<Vec<u8>> {
                gpu.copy_h2d(sentinel, base)?;
                if w8a8 {
                    run_w8a8(&gpu, &scratch, p, out, m)?;
                } else {
                    run_batch16(&gpu, &k, p, out, m)?;
                }
                gpu.synchronize(0)?;
                let mut host = vec![0_u8; sentinel.len()];
                gpu.copy_d2h(base, &mut host)?;
                Ok(host)
            };
            let reference = capture(false)?;
            let observed = capture(true)?;

            if !controls_done && p.strided() {
                // 2026-09-25: A green run has to be able to go red: four
                // corruptions of the observed buffer, each refused by `check`.
                for control in ["gap", "guard", "nonfinite", "value"] {
                    let mut bad = observed.clone();
                    match control {
                        // 2026-09-25: The arena's last element, outside the
                        // first strided projection's columns.
                        "gap" => bad[GUARD + MAX_M * ldc * 2 - 2] ^= 1,
                        "guard" => bad[0] ^= 1,
                        "nonfinite" => {
                            bad[GUARD..GUARD + 2].copy_from_slice(&0x7fc0_u16.to_le_bytes())
                        }
                        _ => {
                            for x in bad[GUARD..GUARD + p.n * 2].chunks_exact_mut(2) {
                                x.copy_from_slice(&bf16::from_f32(1.0e3).to_bits().to_le_bytes());
                            }
                        }
                    }
                    let err = check(&bad, &reference, sentinel, &live, &padded, gate)
                        .expect_err("known-bad output was admitted by the real oracle");
                    println!("KNOWN_BAD {control}: refused: {err}");
                }
                controls_done = true;
            }

            let (cosine, rel_rms) = metrics(&observed, &reference, &live);
            let phantom = if m < MAX_M {
                let rows: Vec<_> = (m..MAX_M)
                    .map(|r| ((r * p.ldc + p.offset) * 2, p.n * 2))
                    .collect();
                let vals: Vec<f64> = rows
                    .iter()
                    .flat_map(|&(s, l)| values(&observed[GUARD + s..GUARD + s + l]))
                    .collect();
                format!(
                    " phantom_rows={}..{} finite={} max_abs={:.3}",
                    m,
                    MAX_M,
                    vals.iter().all(|x| x.is_finite()),
                    vals.iter().fold(0.0_f64, |a, x| a.max(x.abs()))
                )
            } else {
                String::new()
            };
            println!(
                "  {:<18} N={:<5} K={:<5} {:<10} cosine={cosine:.6} rel_rms={rel_rms:.4e}{phantom}",
                p.name,
                p.n,
                p.k,
                if p.strided() { "strided" } else { "contiguous" },
            );
            if let Err(e) = check(&observed, &reference, sentinel, &live, &padded, gate) {
                println!("  FAIL {} M={m}: {e}", p.name);
                failures += 1;
            }
        }

        println!("  --- timing, {REPS} reps per projection ---");
        for p in projections {
            let out = arena(p.ldc).2.offset(GUARD);
            for (label, w8a8) in [("batch16", false), ("W8A8-cuBLASLt", true)] {
                if w8a8 {
                    run_w8a8(&gpu, &scratch, p, out, m)?;
                } else {
                    run_batch16(&gpu, &k, p, out, m)?;
                }
                gpu.synchronize(0)?;
                let t0 = Instant::now();
                for _ in 0..REPS {
                    if w8a8 {
                        run_w8a8(&gpu, &scratch, p, out, m)?;
                    } else {
                        run_batch16(&gpu, &k, p, out, m)?;
                    }
                }
                gpu.synchronize(0)?;
                let us = t0.elapsed().as_secs_f64() * 1e6 / REPS as f64;
                // 2026-09-25: "GB/s-equiv" is the weight bytes over the wall
                // time, for both routes; it is not a measured memory
                // bandwidth.
                println!(
                    "  {:<18} {:<14} {us:>9.1} us  {:>7.0} GB/s-equiv",
                    p.name,
                    label,
                    p.weight_bytes() / (us / 1e6) / 1e9
                );
            }
        }
    }

    ensure!(failures == 0, "{failures} case(s) failed");
    println!(
        "\nALL PASS: Qwen3.8-27B decode projections at M in {ROWS:?} — W8A8 cuBLASLt within \
         cosine {COSINE_GATE} / rel_rms {gate:.1e} of the batch16 GEMV, strided gaps intact, \
         phantom rows finite and confined to their own slots.\n\
         Serve spelling: METRALE_CUBLAS_GEMM=ffn,ssm,attn (METRALE_NO_W8A8_DECODE_PROJ reverts)."
    );
    Ok(())
}
