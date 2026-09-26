// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Cases `test_a`, `test_b` and `test_c` of `kda_chunk_microtest`.
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.

#![allow(unused_imports)]

use crate::*;
use anyhow::{Result, bail};
use kda_chunk_mutant::*;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_arch::glm5next_kda_ref::{KdaDims, kda_chunked, kda_recurrent_prenorm};
use serde_json::Value;
use std::time::Instant;

/// 2026-09-25: The embedded fixture (`FIXTURE`, T = 6): every output token and the final
/// state against the golden at chunk 2 and at chunk 4, whose second chunk is padded by 2.
pub(crate) fn test_a(gpu: &Gpu) -> Result<bool> {
    let v: Value = serde_json::from_str(FIXTURE)?;
    let f = &v["fixture"];
    let (h, d, t) = (
        f["heads"].as_u64().unwrap() as usize,
        f["head_dim"].as_u64().unwrap() as usize,
        f["tokens"].as_u64().unwrap() as usize,
    );
    let ins = Inputs {
        q: arr(&v, "outputs", "q_l2"),
        k: arr(&v, "outputs", "k_l2"),
        v: arr(&v, "inputs", "v_in"),
        gate: arr(&v, "outputs", "gate"),
        beta: arr(&v, "outputs", "beta"),
    };
    let mut ok = true;
    for (c, oname, sname) in [
        (2usize, "core_chunked", "state_chunked"),
        (4, "core_chunked_c4_padded", "state_chunked_c4_padded"),
    ] {
        let mut st = vec![0.0f32; h * d * d];
        let o = gpu.chunk(&ins, t, h, d, c, &mut st, 0.0)?;
        let eo = compare(&o, &arr(&v, "outputs", oname));
        let es = compare(&st, &arr(&v, "outputs", sname));
        report(&format!("A chunk={c} T={t} out vs HF"), &eo);
        report(&format!("A chunk={c} T={t} final state vs HF"), &es);
        ok &= within(&eo) && within(&es);
    }
    Ok(ok)
}

/// 2026-09-25: GPU chunked against GPU recurrent on random inputs at seven (T, chunk)
/// shapes, each within the CPU reference's own chunked-against-recurrent difference. Reads
/// no golden.
pub(crate) fn test_b(gpu: &Gpu) -> Result<bool> {
    let (h, d) = (PROD_H, PROD_D);
    let mut rng = Lcg(0xB10C_5EED);
    let mut ok = true;
    println!(
        "  {:<11}{:>6}{:>12}{:>12}{:>12}{:>12}{:>11}",
        "shape", "chunk", "out max_abs", "out max_rel", "st max_abs", "st max_rel", "GPU/floor"
    );
    for &(t, c) in &[
        (5usize, 8usize),
        (16, 16),
        (17, 16),
        (64, 16),
        (70, 16),
        (129, 32),
        (256, 32),
    ] {
        let n = t * h * d;
        let ins = Inputs {
            q: l2_rows(&rng.vec(n), d),
            k: l2_rows(&rng.vec(n), d),
            v: rng.vec(n),
            gate: rng
                .vec(n)
                .iter()
                .map(|x| -5.0 * (1.0 / (1.0 + (-(x * 3.0)).exp())))
                .collect(),
            beta: rng
                .vec(t * h)
                .iter()
                .map(|x| 1.0 / (1.0 + (-x).exp()))
                .collect(),
        };
        let s0: Vec<f32> = rng.vec(h * d * d).iter().map(|x| x * 0.05).collect();
        let (mut sc, mut sr) = (s0.clone(), s0.clone());
        let oc = gpu.chunk(&ins, t, h, d, c, &mut sc, 0.0)?;
        let or = gpu.recurrent(&ins, t, h, d, &mut sr)?;

        // 2026-09-25: The same two formulations on the CPU reference give the floor.
        let dims = KdaDims {
            hidden: 0,
            heads: h,
            head_dim: d,
            tokens: t,
        };
        let mut fc = s0.clone();
        let cpu_chunk = kda_chunked(
            &ins.q, &ins.k, &ins.v, &ins.gate, &ins.beta, dims, c, &mut fc,
        );
        let mut fr = s0.clone();
        let cpu_rec =
            kda_recurrent_prenorm(&ins.q, &ins.k, &ins.v, &ins.gate, &ins.beta, dims, &mut fr);
        let floor_o = compare(&cpu_chunk, &cpu_rec);
        let floor_s = compare(&fc, &fr);

        let eo = compare(&oc, &or);
        let es = compare(&sc, &sr);
        let ratio = if floor_o.max_abs > 0.0 {
            eo.max_abs / floor_o.max_abs
        } else {
            1.0
        };
        let good = within_floor(&eo, &floor_o) && within_floor(&es, &floor_s);
        println!(
            "  T={t:<9}{c:>6}{:>12.3e}{:>12.3e}{:>12.3e}{:>12.3e}{:>11.3}  {}",
            eo.max_abs,
            eo.max_rel,
            es.max_abs,
            es.max_rel,
            ratio,
            if good { "ok" } else { "FAIL" }
        );
        ok &= good;
    }
    Ok(ok)
}

pub(crate) fn l2_rows(x: &[f32], d: usize) -> Vec<f32> {
    x.chunks_exact(d)
        .flat_map(|r| {
            let inv = 1.0 / (r.iter().map(|a| a * a).sum::<f32>() + 1e-6).sqrt();
            r.iter().map(move |a| a * inv)
        })
        .collect()
}

/// 2026-09-25: The production-geometry golden (`PROD`) at chunks 2, 4, 8, 16 and 32, after
/// checking that the LCG reproduces the generator's probe bit for bit.
pub(crate) fn test_c(gpu: &Gpu) -> Result<bool> {
    let v: Value = serde_json::from_str(&PROD)?;
    let f = &v["fixture"];
    let (h, d, t) = (
        f["heads"].as_u64().unwrap() as usize,
        f["head_dim"].as_u64().unwrap() as usize,
        f["tokens"].as_u64().unwrap() as usize,
    );
    let stride = f["sample_stride"].as_u64().unwrap() as usize;
    let lb = f["lower_bound"].as_f64().unwrap() as f32;
    assert_eq!((h, d), (PROD_H, PROD_D));

    let probe: Vec<f32> = v["lcg_probe"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect();
    let mut pr = Lcg(0x5EED_C400);
    if pr
        .vec(probe.len())
        .iter()
        .zip(&probe)
        .any(|(a, b)| a.to_bits() != b.to_bits())
    {
        println!("    ! LCG mismatch with the generator");
        return Ok(false);
    }
    let hf_self = v["hf_self_checks"]["chunk_vs_recurrent_max_abs"]
        .as_f64()
        .unwrap();
    println!("  oracle self-consistency: HF chunk vs HF recurrent = {hf_self:.3e}");

    let mut rng = Lcg(0x5EED_C400);
    let s0: Vec<f32> = rng.vec(h * d * d).iter().map(|x| x * 0.05).collect();
    let n = t * h * d;
    let ins = Inputs {
        q: l2_rows(&rng.vec(n), d),
        k: l2_rows(&rng.vec(n), d),
        v: rng.vec(n),
        gate: rng
            .vec(n)
            .iter()
            .map(|x| lb * (1.0 / (1.0 + (-(x * 3.0)).exp())))
            .collect(),
        beta: rng
            .vec(t * h)
            .iter()
            .map(|x| 1.0 / (1.0 + (-x).exp()))
            .collect(),
    };
    let want_o = arr(&v, "outputs", "out");
    let want_s = arr(&v, "outputs", "state_sample");
    let want_ck = v["state_checksum"].as_f64().unwrap();

    // 2026-09-25: Floor: the CPU reference at chunk 2 against the golden.
    let mut cpu_state = s0.clone();
    let cpu_o = kda_chunked(
        &ins.q,
        &ins.k,
        &ins.v,
        &ins.gate,
        &ins.beta,
        KdaDims {
            hidden: 0,
            heads: h,
            head_dim: d,
            tokens: t,
        },
        2,
        &mut cpu_state,
    );
    let floor = compare(&cpu_o, &want_o);
    report("C floor: CPU-ref vs HF (no GPU)", &floor);

    let mut ok = true;
    for &c in &[2usize, 4, 8, 16, 32] {
        let mut st = s0.clone();
        let o = gpu.chunk(&ins, t, h, d, c, &mut st, 0.0)?;
        let eo = compare(&o, &want_o);
        let samp: Vec<f32> = st.iter().step_by(stride).copied().collect();
        let es = compare(&samp, &want_s);
        let ck = checksum(&st);
        let ck_rel = (ck - want_ck).abs() / want_ck.abs().max(1.0);
        let r = if floor.max_abs > 0.0 {
            eo.max_abs / floor.max_abs
        } else {
            1.0
        };
        report(&format!("C chunk={c} out vs HF"), &eo);
        println!(
            "     state sample max_abs={:.3e}  full-state checksum rel={ck_rel:.3e}  GPU/floor={r:.3}  smem prep/scan={}/{} B",
            es.max_abs,
            smem_prepare(c, d),
            smem_scan(c, d)
        );
        ok &= within(&eo) && within(&es) && ck_rel < 1e-6 && r <= MAX_FLOOR_RATIO;

        // 2026-09-25: GPU chunked against GPU recurrent, gated at chunk 2 only, against the
        // CPU chunked-against-recurrent floor.
        let mut sr = s0.clone();
        let or = gpu.recurrent(&ins, t, h, d, &mut sr)?;
        let ec = compare(&o, &or);
        if c == 2 {
            let dims = KdaDims {
                hidden: 0,
                heads: h,
                head_dim: d,
                tokens: t,
            };
            let mut fr = s0.clone();
            let cpu_rec =
                kda_recurrent_prenorm(&ins.q, &ins.k, &ins.v, &ins.gate, &ins.beta, dims, &mut fr);
            let floor_cr = compare(&cpu_o, &cpu_rec);
            report("C floor: CPU chunk vs CPU recurrent", &floor_cr);
            report("C chunk vs recurrent-kernel path", &ec);
            ok &= within_floor(&ec, &floor_cr);
        }
    }
    Ok(ok)
}
