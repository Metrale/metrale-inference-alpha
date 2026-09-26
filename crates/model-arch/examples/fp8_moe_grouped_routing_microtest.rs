// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The per-token MoE router (`dense_gemv_bf16` + `moe_topk_softmax`, as in
//! `MoeLayer::forward`) against the batched router (`dense_gemm_bf16` +
//! `moe_topk_softmax_batched`, as in the BF16-gate arm of `forward_fp8_grouped_decode`),
//! at Qwen3.6-35B-A3B shapes (hidden 2048, 256 experts, top-8).
//!
//! Owner: model-arch examples.
//! Invariants:
//! - Returns an error unless, for every M and both `normalize` settings, every row passes
//!   `check`:
//!   * the two logit rows agree to within `LOGIT_TOL` everywhere;
//!   * a differing expert set is admitted only when the per-token row's k-th/(k+1)-th
//!     logit margin is at most twice that row's logit discrepancy;
//!   * where the sets agree, every expert's weight agrees to `W_TOL`.
//! - Admitted set differences are counted and printed.
//!
//! On the first case the oracle must refuse two known-bad inputs: the row's lowest-logit
//! expert substituted into slot 0, and a weight perturbed by 10 x `W_TOL`.
//!
//! Run (GB10):
//!   cargo run --release -p metrale-model-arch --features cuda,gpu-examples \
//!     --example fp8_moe_grouped_routing_microtest

use anyhow::{Result, ensure};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_layers::layers::ops;
use metrale_model_layers::weight_map::DenseWeight;

const H: usize = 2048;
const E: usize = 256;
const TOP_K: usize = 8;
const MAX_M: usize = 32;
/// 2026-09-25: Two BF16 ulps for |logit| in [1, 2), where the ulp is 2^-7.
const LOGIT_TOL: f32 = 2.0 / 128.0;
const W_TOL: f32 = 2e-3;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len().max(16))?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}

fn bf16_bytes(rng: &mut Rng, n: usize, scale: f32) -> Vec<u8> {
    (0..n)
        .flat_map(|_| {
            bf16::from_f32(((rng.next() % 2049) as f32 - 1024.0) / 1024.0 * scale)
                .to_bits()
                .to_le_bytes()
        })
        .collect()
}

fn read_bf16(gpu: &dyn GpuBackend, ptr: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut buf = vec![0u8; n * 2];
    gpu.copy_d2h(ptr, &mut buf)?;
    Ok(buf
        .chunks_exact(2)
        .map(|b| bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
        .collect())
}

fn read_u32(gpu: &dyn GpuBackend, ptr: DevicePtr, n: usize) -> Result<Vec<u32>> {
    let mut buf = vec![0u8; n * 4];
    gpu.copy_d2h(ptr, &mut buf)?;
    Ok(buf
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

fn read_f32(gpu: &dyn GpuBackend, ptr: DevicePtr, n: usize) -> Result<Vec<f32>> {
    Ok(read_u32(gpu, ptr, n)?
        .into_iter()
        .map(f32::from_bits)
        .collect())
}

/// 2026-09-25: One router's output for m rows, read back to the host.
struct Routing {
    logits: Vec<f32>,
    idx: Vec<u32>,
    w: Vec<f32>,
}

/// 2026-09-25: The oracle. Returns the number of rows whose expert sets differ within the
/// admitted margin, or an error for the first inadmissible difference.
fn check(per_token: &Routing, batched: &Routing, m: usize) -> Result<usize> {
    ensure!(per_token.logits.len() >= m * E && batched.logits.len() >= m * E);
    ensure!(per_token.idx.len() >= m * TOP_K && batched.idx.len() >= m * TOP_K);
    let mut flips = 0usize;
    for t in 0..m {
        let lt = &per_token.logits[t * E..(t + 1) * E];
        let lb = &batched.logits[t * E..(t + 1) * E];
        let mut disc = 0f32;
        for (a, b) in lt.iter().zip(lb) {
            ensure!(a.is_finite() && b.is_finite(), "row {t}: nonfinite logit");
            disc = disc.max((a - b).abs());
        }
        ensure!(
            disc <= LOGIT_TOL,
            "row {t}: batched router logits differ from per-token by {disc} (> {LOGIT_TOL})"
        );
        let mut st: Vec<u32> = per_token.idx[t * TOP_K..(t + 1) * TOP_K].to_vec();
        let mut sb: Vec<u32> = batched.idx[t * TOP_K..(t + 1) * TOP_K].to_vec();
        st.sort_unstable();
        sb.sort_unstable();
        ensure!(
            st.windows(2).all(|p| p[0] != p[1]) && sb.windows(2).all(|p| p[0] != p[1]),
            "row {t}: duplicate expert in a top-k set"
        );
        ensure!(
            st.iter().chain(sb.iter()).all(|&e| (e as usize) < E),
            "row {t}: expert id out of range"
        );
        if st != sb {
            let mut sorted = lt.to_vec();
            sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
            let margin = sorted[TOP_K - 1] - sorted[TOP_K];
            ensure!(
                margin <= 2.0 * disc,
                "row {t}: expert sets differ ({st:?} vs {sb:?}) but the k-th margin {margin} \
                 exceeds twice the logit discrepancy {disc} — not a rounding flip"
            );
            flips += 1;
            continue;
        }
        // 2026-09-25: Same set: weights are matched per expert, since slot order may differ.
        for k in 0..TOP_K {
            let e = per_token.idx[t * TOP_K + k];
            let wt = per_token.w[t * TOP_K + k];
            let kb = (0..TOP_K)
                .find(|&j| batched.idx[t * TOP_K + j] == e)
                .expect("set equality checked");
            let wb = batched.w[t * TOP_K + kb];
            ensure!(
                wt.is_finite() && wb.is_finite(),
                "row {t}: nonfinite weight"
            );
            ensure!(
                (wt - wb).abs() <= W_TOL,
                "row {t} expert {e}: weight {wb} (batched) vs {wt} (per-token)"
            );
        }
    }
    Ok(flips)
}

fn main() -> Result<()> {
    let gpu = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let gemv_k = gpu.kernel("gemv", "dense_gemv_bf16")?;
    let gemm_k = gpu.kernel("gemm", "dense_gemm_bf16")?;
    let topk_k = gpu.kernel("moe_topk", "moe_topk_softmax")?;
    let topk_batched_k = gpu.kernel("moe_topk", "moe_topk_softmax_batched")?;
    let mut rng = Rng(0x726f_7574_6539_2026);

    let gate = DenseWeight {
        weight: upload(&gpu, &bf16_bytes(&mut rng, E * H, 0.05))?,
    };
    let input = upload(&gpu, &bf16_bytes(&mut rng, MAX_M * H, 1.0))?;
    let logits_t = gpu.alloc(MAX_M * E * 2)?;
    let logits_b = gpu.alloc(MAX_M * E * 2)?;
    let idx_t = gpu.alloc(MAX_M * TOP_K * 4)?;
    let w_t = gpu.alloc(MAX_M * TOP_K * 4)?;
    let idx_b = gpu.alloc(MAX_M * TOP_K * 4)?;
    let w_b = gpu.alloc(MAX_M * TOP_K * 4)?;

    let mut failures = 0usize;
    let mut first = true;
    for normalize in [true, false] {
        for m in [2usize, 4, 8, 16, 32] {
            for t in 0..m {
                ops::dense_gemv(
                    &gpu,
                    gemv_k,
                    input.offset(t * H * 2),
                    &gate,
                    logits_t.offset(t * E * 2),
                    E as u32,
                    H as u32,
                    0,
                )?;
                ops::moe_topk_softmax(
                    &gpu,
                    topk_k,
                    logits_t.offset(t * E * 2),
                    idx_t.offset(t * TOP_K * 4),
                    w_t.offset(t * TOP_K * 4),
                    E as u32,
                    TOP_K as u32,
                    normalize,
                    0,
                )?;
            }
            ops::dense_gemm(
                &gpu, gemm_k, input, &gate, logits_b, m as u32, E as u32, H as u32, 0,
            )?;
            ops::moe_topk_softmax_batched(
                &gpu,
                topk_batched_k,
                logits_b,
                idx_b,
                w_b,
                E as u32,
                TOP_K as u32,
                normalize,
                m as u32,
                0,
            )?;
            gpu.synchronize(0)?;
            let per_token = Routing {
                logits: read_bf16(&gpu, logits_t, m * E)?,
                idx: read_u32(&gpu, idx_t, m * TOP_K)?,
                w: read_f32(&gpu, w_t, m * TOP_K)?,
            };
            let batched = Routing {
                logits: read_bf16(&gpu, logits_b, m * E)?,
                idx: read_u32(&gpu, idx_b, m * TOP_K)?,
                w: read_f32(&gpu, w_b, m * TOP_K)?,
            };

            if first {
                let mut bad = Routing {
                    logits: batched.logits.clone(),
                    idx: batched.idx.clone(),
                    w: batched.w.clone(),
                };
                let lowest = (0..E)
                    .min_by(|&a, &b| {
                        per_token.logits[a]
                            .partial_cmp(&per_token.logits[b])
                            .unwrap()
                    })
                    .unwrap() as u32;
                bad.idx[0] = lowest;
                let err = check(&per_token, &bad, m)
                    .expect_err("a far-off expert was admitted as a rounding flip");
                println!("KNOWN_BAD wrong-expert: refused: {err}");
                let mut bad = Routing {
                    logits: batched.logits.clone(),
                    idx: batched.idx.clone(),
                    w: batched.w.clone(),
                };
                bad.w[0] += 10.0 * W_TOL;
                match check(&per_token, &bad, m) {
                    Err(err) => println!("KNOWN_BAD weight: refused: {err}"),
                    Ok(_) => {
                        // 2026-09-25: Admitted only when row 0's sets differ, in which
                        // case `check` does not compare its weights.
                        let s0: std::collections::BTreeSet<u32> =
                            per_token.idx[..TOP_K].iter().copied().collect();
                        let b0: std::collections::BTreeSet<u32> =
                            batched.idx[..TOP_K].iter().copied().collect();
                        ensure!(s0 != b0, "perturbed weight admitted on an agreeing row");
                        println!(
                            "KNOWN_BAD weight: row 0 flipped on this draw (weights uncompared)"
                        );
                    }
                }
                first = false;
            }

            match check(&per_token, &batched, m) {
                Ok(flips) => println!(
                    "normalize={normalize:5} M={m:2}: logits agree within {LOGIT_TOL:.4}; \
                     expert sets {} rows identical, {flips} rounding flip(s) — OK",
                    m - flips
                ),
                Err(e) => {
                    println!("FAIL normalize={normalize} M={m}: {e}");
                    failures += 1;
                }
            }
        }
    }
    ensure!(failures == 0, "{failures} case(s) failed");
    println!("ALL PASS: batched router == per-token router up to admissible rounding flips");
    Ok(())
}
