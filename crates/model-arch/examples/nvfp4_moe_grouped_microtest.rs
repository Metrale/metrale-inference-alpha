// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-27: The grouped NVFP4 MoE decode (`moe_fp8_grouped_sort`,
//! `moe_expert_{gate_up,down}_act_nvfp4_grouped`, the grouped FP8 expert kernels for the FP8
//! projections, `moe_weighted_sum_blend_fp8_grouped`) at Qwen3.6-35B-A3B shapes, in the three
//! configurations `forward_nvfp4_grouped_decode` launches:
//! - `all-nvfp4`: routed and shared experts NVFP4 (a checkpoint without FP8 experts);
//! - `nvfp4`: routed experts NVFP4, the shared expert FP8 (`--expert-quantization nvfp4`);
//! - `nvfp4-gate-up`: routed gate/up NVFP4, routed down and the shared expert FP8
//!   (`--expert-quantization nvfp4-gate-up`).
//!
//! Owner: model-arch examples.
//! Invariants:
//! - Exits 1 unless, for every configuration, (a) for every M each row's blended output bytes
//!   equal that row's bytes at M = MAX_M (the same routing and input), (b) at M = 4 the routed
//!   SiLU products, the routed down outputs and the shared expert's output agree with an f64
//!   host reference within a relative 2 % of the row's largest value, and (c) the invariance
//!   oracle is live: comparing each row against its neighbour's bytes finds a difference.
//!
//! Row t routes to the same experts at every M (the first M rows of one MAX_M-row draw), so
//! only the number of rows sharing the launch changes. Weights are random NVFP4 (packed E2M1,
//! E4M3 block scales of 16, per-tensor scale 2) and FP8 (E4M3 with 128x128 block scales). Mean
//! times of the expert kernels over 20 launches are printed per M.
//!
//!   cargo run --release -p metrale-model-arch --features cuda,gpu-examples \
//!     --example nvfp4_moe_grouped_microtest -- [experts] [zipf_alpha] [iters]
//!
//! Arguments: routed experts (default 32, so rows share experts; 256 is the model's) and a
//! Zipf exponent for the routing (default 0.9).

use anyhow::{Context, Result};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_model_layers::layers::ops;

#[path = "common/nvfp4_moe_fixture.rs"]
mod fixture;
#[path = "common/nvfp4_moe_legs.rs"]
mod legs;
use fixture::*;
use legs::*;

const H: usize = 2048;
const INTER: usize = 512;
const TOP_K: usize = 8;
// 2026-10-02: The buffers' width: the tensor-core leg's envelope. Each leg checks its own
// envelope (`Leg::max_m`), at the widths of WIDTHS it admits.
const MAX_M: usize = 256;
const WIDTHS: [usize; 15] = [1, 2, 3, 4, 5, 8, 13, 16, 32, 63, 64, 96, 128, 200, 255];

/// 2026-09-27: The f64 host reference at M = 4: routed SiLU products and down outputs per
/// sorted position, and each token's shared-expert output.
fn host_reference(
    w: &Weights,
    leg: Leg,
    input_f: &[f64],
    r: &Run,
    m: usize,
    e: usize,
) -> Result<()> {
    let (_, act_h, down_h, sh_h, sorted_h, offsets_h, _) = r;
    let (down_f, sh_f) = (bf16_to_f64(down_h), bf16_to_f64(sh_h));
    for ex in 0..e {
        for pos in offsets_h[ex] as usize..offsets_h[ex + 1] as usize {
            let x = &input_f[sorted_h[pos] as usize * H..][..H];
            let a = silu_product(&w.gate[ex], &w.up[ex], x, INTER);
            let down: &dyn HostDot = if leg.fp8_down() {
                &w.down8[ex]
            } else {
                &w.down[ex]
            };
            let d: Vec<f64> = (0..H).map(|c| down.dot(c, &a)).collect();
            close(
                &act_h[pos * INTER..][..INTER],
                &a,
                &format!("act e{ex} pos{pos}"),
            )?;
            close(&down_f[pos * H..][..H], &d, &format!("down e{ex} pos{pos}"))?;
        }
    }
    let sh: [&dyn HostDot; 3] = if leg.fp8_shared() {
        [&w.sh8[0], &w.sh8[1], &w.sh8[2]]
    } else {
        [&w.sh[0], &w.sh[1], &w.sh[2]]
    };
    for t in 0..m {
        let a = silu_product(sh[0], sh[1], &input_f[t * H..][..H], INTER);
        let d: Vec<f64> = (0..H).map(|c| sh[2].dot(c, &a)).collect();
        close(&sh_f[t * H..][..H], &d, &format!("shared t{t}"))?;
    }
    Ok(())
}

fn main() -> Result<()> {
    let num_experts: usize = arg(1, 32);
    let alpha: f64 = arg(2, 0.9);
    // 2026-10-02: Launches per timed width (third argument; default 20). A large count makes each
    // width's window long enough for an NVML energy integral, printed as `window` lines.
    let iters: usize = arg(3, 20);
    let set = metrale_kernels::ptx_for_exact_target("qwen3.6-35b-a3b", "nvfp4")
        .context("no compiled qwen3.6-35b-a3b/nvfp4 kernel set")?;
    let backend = MetraleCudaBackend::new(0, &set.modules)?;
    let g: &dyn GpuBackend = &backend;
    const FP8: &str = "moe_shared_expert_fused_fp8_grouped";
    let k = Kernels {
        gate_up: g.kernel("moe_nvfp4_grouped", "moe_expert_gate_up_act_nvfp4_grouped")?,
        down: g.kernel("moe_nvfp4_grouped", "moe_expert_down_act_nvfp4_grouped")?,
        gate_up_tc: g.kernel(
            "moe_nvfp4_grouped_tc",
            "moe_expert_gate_up_act_nvfp4_grouped_tc",
        )?,
        down_tc: g.kernel(
            "moe_nvfp4_grouped_tc",
            "moe_expert_down_act_nvfp4_grouped_tc",
        )?,
        fp8_gate_up: g.kernel(FP8, "moe_expert_gate_up_act_fp8_grouped")?,
        fp8_down: g.kernel(FP8, "moe_expert_down_act_fp8_grouped")?,
        sort: g.kernel("moe_fp8_grouped_sort", "moe_fp8_grouped_sort")?,
        blend: g.kernel(
            "moe_fp8_grouped_blend",
            "moe_weighted_sum_blend_fp8_grouped",
        )?,
    };

    let mut rng = Rng(7);
    let mats = |rng: &mut Rng, n: usize, kk: usize| -> Result<Vec<Mat>> {
        (0..num_experts).map(|_| Mat::new(g, rng, n, kk)).collect()
    };
    let gate = mats(&mut rng, INTER, H)?;
    let up = mats(&mut rng, INTER, H)?;
    let down = mats(&mut rng, H, INTER)?;
    let down8: Vec<Fp8Mat> = (0..num_experts)
        .map(|_| Fp8Mat::new(g, &mut rng, H, INTER))
        .collect::<Result<_>>()?;
    let sh = [
        Mat::new(g, &mut rng, INTER, H)?,
        Mat::new(g, &mut rng, INTER, H)?,
        Mat::new(g, &mut rng, H, INTER)?,
    ];
    let sh8 = [
        Fp8Mat::new(g, &mut rng, INTER, H)?,
        Fp8Mat::new(g, &mut rng, INTER, H)?,
        Fp8Mat::new(g, &mut rng, H, INTER)?,
    ];
    let w = Weights {
        gate_t: nvfp4_table(g, &gate.iter().collect::<Vec<_>>())?,
        up_t: nvfp4_table(g, &up.iter().collect::<Vec<_>>())?,
        down_t: nvfp4_table(g, &down.iter().collect::<Vec<_>>())?,
        down8_t: fp8_table(g, &down8.iter().collect::<Vec<_>>())?,
        gate,
        up,
        down,
        down8,
        sh,
        sh8,
    };

    // 2026-09-27: MAX_M rows of input, routing and slot weights; width M uses the first M.
    let input_bytes: Vec<u8> = (0..MAX_M * H)
        .flat_map(|_| {
            bf16::from_f64(rng.unit() * 2.0 - 1.0)
                .to_bits()
                .to_le_bytes()
        })
        .collect();
    let input_f = bf16_to_f64(&input_bytes);
    let pop: Vec<f64> = (0..num_experts)
        .map(|e| ((e + 1) as f64).powf(-alpha))
        .collect();
    let total: f64 = pop.iter().sum();
    let mut routing: Vec<u32> = Vec::with_capacity(MAX_M * TOP_K);
    for _ in 0..MAX_M {
        let mut row: Vec<u32> = Vec::new();
        while row.len() < TOP_K {
            let (u, mut acc, mut e) = (rng.unit() * total, 0.0, 0usize);
            while e + 1 < num_experts && acc + pop[e] < u {
                acc += pop[e];
                e += 1;
            }
            if !row.contains(&(e as u32)) {
                row.push(e as u32);
            }
        }
        routing.extend(row);
    }
    let te_max = MAX_M * TOP_K;
    let bytes = |v: Vec<u8>| upload(g, &v);
    let b = Bufs {
        input: upload(g, &input_bytes)?,
        idx: bytes(routing.iter().flat_map(|v| v.to_le_bytes()).collect())?,
        slot_w: bytes(
            (0..te_max)
                .flat_map(|_| (rng.unit() as f32 / TOP_K as f32).to_le_bytes())
                .collect(),
        )?,
        sh_gate_vec: bytes(
            (0..H)
                .flat_map(|_| {
                    bf16::from_f64((rng.unit() - 0.5) * 0.05)
                        .to_bits()
                        .to_le_bytes()
                })
                .collect(),
        )?,
        sort: ops::Fp8GroupedSortOut {
            sorted_token_ids: g.alloc(te_max * 4)?,
            sorted_expert_ids: g.alloc(te_max * 4)?,
            expert_offsets: g.alloc((num_experts + 1) * 4)?,
            token_to_perm: g.alloc(te_max * 4)?,
            active_experts: g.alloc(te_max.min(num_experts) * 4 + 16)?,
            active_count: g.alloc(16)?,
        },
        act: g.alloc(te_max * INTER * 4)?,
        sh_act: g.alloc(MAX_M * INTER * 4)?,
        down_out: g.alloc(te_max * H * 2)?,
        sh_out: g.alloc(MAX_M * H * 2)?,
        output: g.alloc(MAX_M * H * 2)?,
    };

    let mut failures = 0usize;
    let row = H * 2;
    // 2026-10-02: Expert weight bytes one call streams for the first m rows: every distinct
    // routed expert's gate, up and down plus the shared expert, NVFP4 (packed + scales).
    let expert_bytes = (3 * INTER * H) as f64 * (0.5 + 1.0 / 16.0);
    let streamed = |m: usize| {
        let mut seen = vec![false; num_experts];
        for &e in &routing[..m * TOP_K] {
            seen[e as usize] = true;
        }
        (seen.iter().filter(|s| **s).count() + 1) as f64 * expert_bytes
    };
    let mut blended: Vec<(Leg, Vec<u8>)> = Vec::new();
    for leg in [Leg::AllNvfp4, Leg::Nvfp4, Leg::Nvfp4GateUp, Leg::AllNvfp4Tc] {
        let max_m = leg.max_m();
        let full = run(g, &k, &w, &b, leg, max_m, num_experts, 0)?.0;
        let mut bad_widths = Vec::new();
        let mut times = Vec::new();
        for &m in WIDTHS.iter().filter(|&&m| m <= max_m) {
            let t0 = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0.0, |d| d.as_secs_f64());
            let r = run(g, &k, &w, &b, leg, m, num_experts, iters)?;
            if iters > 20 {
                // 2026-10-02: The window an NVML energy log is cut on (`ITERS` microbench mode).
                println!("window {} M{m} {t0:.4}", leg.name());
            }
            if (0..m).any(|t| r.0[t * row..(t + 1) * row] != full[t * row..(t + 1) * row]) {
                bad_widths.push(m);
            }
            times.push(format!(
                "M{m}:{:.0}us/{:.0}GB/s",
                r.6,
                streamed(m) / r.6 / 1e3
            ));
        }
        if matches!(leg, Leg::AllNvfp4 | Leg::AllNvfp4Tc) {
            blended.push((leg, full[..ops_rows::SCALAR_MAX * row].to_vec()));
        }
        let live = (0..max_m - 1)
            .filter(|t| full[t * row..(t + 1) * row] != full[(t + 1) * row..(t + 2) * row])
            .count();
        let r4 = run(g, &k, &w, &b, leg, 4, num_experts, 0)?;
        let reference = host_reference(&w, leg, &input_f, &r4, 4, num_experts);
        let ok = bad_widths.is_empty() && live > 0 && reference.is_ok();
        println!(
            "{:<14} rows equal to M={max_m} at every M: {}  oracle live: {live}/{}  host reference \
             at M=4: {}  {}",
            leg.name(),
            if bad_widths.is_empty() {
                "yes".to_string()
            } else {
                format!("NO at {bad_widths:?}")
            },
            max_m - 1,
            match &reference {
                Ok(()) => "PASS".to_string(),
                Err(e) => format!("FAIL {e:#}"),
            },
            times.join(" ")
        );
        failures += usize::from(!ok);
    }
    // 2026-10-02: Informative: how far the tensor-core leg's blended rows sit from the CUDA-core
    // leg's (different summation order and SiLU carrier), relative to each row's largest value.
    if let [(_, a), (_, c)] = &blended[..] {
        let (fa, fc) = (bf16_to_f64(a), bf16_to_f64(c));
        let worst = fa
            .chunks_exact(H)
            .zip(fc.chunks_exact(H))
            .map(|(x, y)| {
                let scale = x.iter().fold(1e-30f64, |m, v| m.max(v.abs()));
                x.iter()
                    .zip(y)
                    .fold(0f64, |m, (p, q)| m.max((p - q).abs() / scale))
            })
            .fold(0f64, f64::max);
        println!(
            "all-nvfp4 vs all-nvfp4-tc blended, rows 0..{}: worst |diff| / row max = {worst:.2e}",
            ops_rows::SCALAR_MAX
        );
    }
    println!(
        "nvfp4_moe_grouped_microtest experts={num_experts} alpha={alpha}: {}",
        if failures == 0 {
            "ALL PASS"
        } else {
            "FAILURES"
        }
    );
    if failures > 0 {
        std::process::exit(1);
    }
    Ok(())
}
