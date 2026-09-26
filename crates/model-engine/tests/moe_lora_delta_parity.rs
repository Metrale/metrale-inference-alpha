// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: MoE LoRA delta folds on the GPU against a host reference; the
//! numpy counterpart is `scripts/moe_lora_oracle.py`.
//!
//! Owner: model-engine tests.
//! Invariants: none beyond the types.
//!
//! Small sizes: hidden 32, expert width 16, 4 experts, adapter rank 4 padded to
//! 8, top-2, 6 tokens. A, B and the base output are random BF16, and
//! `SCALE = ALPHA / R` (0.75). The host reference `host_fold_row` rounds where
//! the kernels do: BF16 xa, BF16 delta, then `base + scale * delta` in FP32,
//! rounded to BF16. Results must be within one BF16 ULP of it (`cmp_tol`)
//! unless stated otherwise:
//!
//!   CHECK 1: router fold, `lora::apply_router_lora`.
//!   CHECK 2: grouped prefill down fold, `ops::moe_lora_grouped_down` with
//!            x_gather = 0 over rows sorted by expert; expert 1 has no adapter
//!            (a NULL table cell) and is left unchanged.
//!   CHECK 3: the same fold in two row windows is bit-identical to CHECK 2.
//!   CHECK 4: with token 0 mapped to the base model, its rows are bit-identical
//!            to the base and the other rows to CHECK 2.
//!   CHECK 5: gate/up fold with x_gather = 1 (token-major x gathered through
//!            `sorted_token_ids`) and an expert table shorter than the layer.
//!   CHECK 6: decode fold `ops::moe_lora_gather_bgmv` over `indices_dev`, with
//!            token 5 mapped to the base model.
//!   CHECK 7: `lora::apply_expert_lora_sorted` gives CHECK 2's reference result.
//!
//! `#[ignore]`d because it needs a GPU. On a GB10 host:
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL='*' METRALE_TARGET_QUANT='*' \
//!     cargo test -p metrale-model-engine --test moe_lora_delta_parity -- --ignored --nocapture

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_layers::layers::ops;
use metrale_model_layers::layers::ops::lora_delta::{LoraKernels, LoraPair};
use metrale_model_layers::layers::ops::moe_lora_grouped::{MoeExpertRoute, pack_expert_tables};
use metrale_model_layers::lora::{
    ExpertLoraLayer, ExpertProj, apply_expert_lora_sorted, apply_router_lora,
};
use metrale_model_layers::weight_map::DenseWeight;

#[path = "arm2_common/support.rs"]
mod support;
use support::{Rng, bf16_bits_to_f32, cmp_tol, f32_to_bf16_bits, rd_u16, setup, up_i32, up_u16};

const H: usize = 32;
const INTER: usize = 16;
const E: usize = 4;
const R: usize = 4;
const MAX_RANK: usize = 8;
const TOP_K: usize = 2;
const T: usize = 6;
const TE: usize = T * TOP_K;
const ALPHA: f32 = 3.0;
const SCALE: f32 = ALPHA / R as f32;
const SEED: u64 = 0x_10AA_D317_0335_0001;

/// 2026-09-25: One adapter pair in the padded layout `lora/loading.rs` builds:
/// A is `[MAX_RANK, k_in]` with the R real rows first and zero rows after; B is
/// `[n_out, MAX_RANK]` with the R real columns first and zero columns after.
struct HostPair {
    a: Vec<u16>,
    b: Vec<u16>,
    k_in: usize,
    n_out: usize,
}

fn gen_pair(rng: &mut Rng, k_in: usize, n_out: usize) -> HostPair {
    let mut a = vec![0u16; MAX_RANK * k_in];
    for j in 0..R {
        for k in 0..k_in {
            a[j * k_in + k] = f32_to_bf16_bits(rng.unit() * 2.0 - 1.0);
        }
    }
    let mut b = vec![0u16; n_out * MAX_RANK];
    for n in 0..n_out {
        for j in 0..R {
            b[n * MAX_RANK + j] = f32_to_bf16_bits(rng.unit() * 2.0 - 1.0);
        }
    }
    HostPair { a, b, k_in, n_out }
}

fn gen_bf16(rng: &mut Rng, n: usize) -> Vec<u16> {
    (0..n)
        .map(|_| f32_to_bf16_bits(rng.unit() * 2.0 - 1.0))
        .collect()
}

/// 2026-09-25: The host reference: `base += SCALE * (B @ (A @ x))` for one row,
/// with BF16 xa, BF16 delta, and the scale applied in FP32 after the delta is
/// rounded, as `moe_lora_grouped_down.cu` and `bf16_scaled_add` do.
fn host_fold_row(p: &HostPair, x: &[u16], base: &mut [u16]) {
    let mut xa = [0u16; MAX_RANK];
    for (j, slot) in xa.iter_mut().enumerate() {
        let mut acc = 0f32;
        for k in 0..p.k_in {
            acc += bf16_bits_to_f32(x[k]) * bf16_bits_to_f32(p.a[j * p.k_in + k]);
        }
        *slot = f32_to_bf16_bits(acc);
    }
    for (n, out) in base.iter_mut().enumerate() {
        let mut acc = 0f32;
        for j in 0..MAX_RANK {
            acc += bf16_bits_to_f32(xa[j]) * bf16_bits_to_f32(p.b[n * MAX_RANK + j]);
        }
        let delta = bf16_bits_to_f32(f32_to_bf16_bits(acc));
        *out = f32_to_bf16_bits(bf16_bits_to_f32(*out) + SCALE * delta);
    }
}

fn assert_tol(label: &str, got: &[u16], want: &[u16]) {
    let (pass, exact, max_ulp, worst) = cmp_tol(got, want);
    println!(
        "  {label}: exact {exact}/{} max_ulp {max_ulp} (worst idx {worst})",
        got.len()
    );
    assert!(
        pass,
        "{label}: kernel diverged from the delta oracle by {max_ulp} bf16 ULP"
    );
}

fn to_pair(hp: &HostPair, gpu: &dyn GpuBackend) -> Result<(LoraPair, DevicePtr, DevicePtr)> {
    let a = up_u16(gpu, &hp.a)?;
    let b = up_u16(gpu, &hp.b)?;
    Ok((
        LoraPair {
            a: DenseWeight { weight: a },
            b: DenseWeight { weight: b },
            rank: R as u32,
            k_in: hp.k_in as u32,
            n_out: hp.n_out as u32,
            scale: SCALE,
            max_rank: MAX_RANK as u32,
        },
        a,
        b,
    ))
}

fn route_of(pairs: &[(u16, LoraPair)], gpu: &dyn GpuBackend) -> Result<MoeExpertRoute> {
    let entries: Vec<(u16, u64, u64, f32)> = pairs
        .iter()
        .map(|(e, p)| (*e, p.a.weight.0, p.b.weight.0, p.scale))
        .collect();
    let t = pack_expert_tables(&entries).expect("non-empty");
    let up64 = |v: &[u64]| -> Result<DevicePtr> {
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        let d = gpu.alloc(bytes.len())?;
        gpu.copy_h2d(&bytes, d)?;
        Ok(d)
    };
    let sbytes: Vec<u8> = t.scale.iter().flat_map(|s| s.to_le_bytes()).collect();
    let sdev = gpu.alloc(sbytes.len())?;
    gpu.copy_h2d(&sbytes, sdev)?;
    let sample = &pairs[0].1;
    Ok(MoeExpertRoute {
        a_table: up64(&t.a)?,
        b_table: up64(&t.b)?,
        scale_table: sdev,
        n_experts: t.n_experts,
        k_in: sample.k_in,
        n_out: sample.n_out,
        max_rank: sample.max_rank,
    })
}

// 2026-09-25: One `#[test]`, run on the thread that created the backend, whose
// context is current only there.
#[test]
#[ignore = "requires a GB10 GPU + compiled kernel set (CI links libcuda stubs only)"]
fn moe_lora_delta_parity() -> Result<()> {
    let (backend, st) = setup()?;
    let gpu: &dyn GpuBackend = &backend;
    let kernels = LoraKernels::new(gpu)?;
    let mut rng = Rng(SEED);

    // 2026-09-25: Token t routes to experts t % E and (t + 1) % E; rows are
    // sorted by expert.
    let tok_experts: Vec<[usize; 2]> = (0..T).map(|t| [t % E, (t + 1) % E]).collect();
    let mut sorted_token_ids: Vec<i32> = Vec::new();
    let mut expert_offsets: Vec<i32> = vec![0];
    for e in 0..E {
        for (t, xs) in tok_experts.iter().enumerate() {
            if xs.contains(&e) {
                sorted_token_ids.push(t as i32);
            }
        }
        expert_offsets.push(sorted_token_ids.len() as i32);
    }
    assert_eq!(sorted_token_ids.len(), TE);
    let offs_dev = up_i32(gpu, &expert_offsets)?;
    let stid_dev = up_i32(gpu, &sorted_token_ids)?;
    let expert_of_row = |r: usize| {
        (0..E)
            .find(|&e| r < expert_offsets[e + 1] as usize)
            .unwrap()
    };

    // 2026-09-25: Scratch: xa [rows, MAX_RANK] and delta [rows, max(H, E)].
    let xa_dev = gpu.alloc(TE * MAX_RANK * 2)?;
    let delta_dev = gpu.alloc(TE * H.max(E) * 2)?;

    // 2026-09-25: CHECK 1: router fold.
    let router = gen_pair(&mut rng, H, E);
    let (router_pair, ..) = to_pair(&router, gpu)?;
    let x_tok = gen_bf16(&mut rng, T * H);
    let x_tok_dev = up_u16(gpu, &x_tok)?;
    let logits = gen_bf16(&mut rng, T * E);
    let logits_dev = up_u16(gpu, &logits)?;
    apply_router_lora(
        gpu,
        &kernels,
        &router_pair,
        x_tok_dev,
        logits_dev,
        T as u32,
        TE as u32,
        xa_dev,
        delta_dev,
        st,
    )?;
    gpu.synchronize(st)?;
    let mut want = logits.clone();
    for t in 0..T {
        host_fold_row(
            &router,
            &x_tok[t * H..(t + 1) * H],
            &mut want[t * E..(t + 1) * E],
        );
    }
    assert_tol("CHECK 1 router", &rd_u16(gpu, logits_dev, T * E)?, &want);

    // 2026-09-25: CHECK 2: grouped prefill down fold; experts 0, 2 and 3 have
    // adapters, expert 1 does not.
    let down: Vec<(u16, HostPair)> = [0u16, 2, 3]
        .iter()
        .map(|&e| (e, gen_pair(&mut rng, INTER, H)))
        .collect();
    let down_pairs: Vec<(u16, LoraPair)> = down
        .iter()
        .map(|(e, hp)| Ok((*e, to_pair(hp, gpu)?.0)))
        .collect::<Result<_>>()?;
    let down_route = route_of(&down_pairs, gpu)?;
    assert_eq!(down_route.n_experts, E as u32);
    let x_sorted = gen_bf16(&mut rng, TE * INTER);
    let x_sorted_dev = up_u16(gpu, &x_sorted)?;
    let base_down = gen_bf16(&mut rng, TE * H);
    let out_dev = up_u16(gpu, &base_down)?;
    ops::moe_lora_grouped_down(
        gpu,
        &kernels,
        &down_route,
        x_sorted_dev,
        out_dev,
        offs_dev,
        stid_dev,
        DevicePtr::NULL,
        xa_dev,
        0,
        TE as u32,
        0,
        st,
    )?;
    gpu.synchronize(st)?;
    let mut want_down = base_down.clone();
    for r in 0..TE {
        if let Some((_, hp)) = down.iter().find(|(e, _)| *e as usize == expert_of_row(r)) {
            host_fold_row(
                hp,
                &x_sorted[r * INTER..(r + 1) * INTER],
                &mut want_down[r * H..(r + 1) * H],
            );
        }
    }
    let got_down = rd_u16(gpu, out_dev, TE * H)?;
    assert_tol("CHECK 2 grouped down", &got_down, &want_down);

    // 2026-09-25: CHECK 3: rows [0, 7) and [7, TE) in two calls.
    let out2_dev = up_u16(gpu, &base_down)?;
    for (lo, hi) in [(0u32, 7u32), (7, TE as u32)] {
        ops::moe_lora_grouped_down(
            gpu,
            &kernels,
            &down_route,
            x_sorted_dev,
            out2_dev,
            offs_dev,
            stid_dev,
            DevicePtr::NULL,
            xa_dev,
            lo,
            hi,
            0,
            st,
        )?;
    }
    gpu.synchronize(st)?;
    let got_chunked = rd_u16(gpu, out2_dev, TE * H)?;
    assert_eq!(
        got_chunked, got_down,
        "CHECK 3: chunked fold must be bit-identical"
    );
    println!("  CHECK 3 chunked windows: bit-identical");

    // 2026-09-25: CHECK 4: row map with token 0 at -1 (base model).
    let mut row_map = vec![0i32; T];
    row_map[0] = -1;
    let map_dev = up_i32(gpu, &row_map)?;
    let out3_dev = up_u16(gpu, &base_down)?;
    ops::moe_lora_grouped_down(
        gpu,
        &kernels,
        &down_route,
        x_sorted_dev,
        out3_dev,
        offs_dev,
        stid_dev,
        map_dev,
        xa_dev,
        0,
        TE as u32,
        0,
        st,
    )?;
    gpu.synchronize(st)?;
    let got_skip = rd_u16(gpu, out3_dev, TE * H)?;
    for r in 0..TE {
        let want_row = if sorted_token_ids[r] == 0 {
            &base_down
        } else {
            &got_down
        };
        assert_eq!(
            &got_skip[r * H..(r + 1) * H],
            &want_row[r * H..(r + 1) * H],
            "CHECK 4: row {r} (token {})",
            sorted_token_ids[r]
        );
    }
    println!("  CHECK 4 row-adapter skip: base rows untouched, adapted rows identical");

    // 2026-09-25: CHECK 5: gate/up fold; only experts 0 and 1 have adapters,
    // so the table has 2 entries and experts 2 and 3 are left unchanged.
    let gate: Vec<(u16, HostPair)> = [0u16, 1]
        .iter()
        .map(|&e| (e, gen_pair(&mut rng, H, INTER)))
        .collect();
    let gate_pairs: Vec<(u16, LoraPair)> = gate
        .iter()
        .map(|(e, hp)| Ok((*e, to_pair(hp, gpu)?.0)))
        .collect::<Result<_>>()?;
    let gate_route = route_of(&gate_pairs, gpu)?;
    assert_eq!(gate_route.n_experts, 2);
    let base_gate = gen_bf16(&mut rng, TE * INTER);
    let gout_dev = up_u16(gpu, &base_gate)?;
    ops::moe_lora_grouped_down(
        gpu,
        &kernels,
        &gate_route,
        x_tok_dev,
        gout_dev,
        offs_dev,
        stid_dev,
        DevicePtr::NULL,
        xa_dev,
        0,
        TE as u32,
        1,
        st,
    )?;
    gpu.synchronize(st)?;
    let mut want_gate = base_gate.clone();
    for r in 0..TE {
        if let Some((_, hp)) = gate.iter().find(|(e, _)| *e as usize == expert_of_row(r)) {
            let t = sorted_token_ids[r] as usize;
            host_fold_row(
                hp,
                &x_tok[t * H..(t + 1) * H],
                &mut want_gate[r * INTER..(r + 1) * INTER],
            );
        }
    }
    assert_tol(
        "CHECK 5 gate/up",
        &rd_u16(gpu, gout_dev, TE * INTER)?,
        &want_gate,
    );

    // 2026-09-25: CHECK 6: decode gather fold, with token 5 at -1 (base model).
    let indices: Vec<u32> = tok_experts
        .iter()
        .flat_map(|xs| xs.iter().map(|&e| e as u32))
        .collect();
    let idx_dev = support::up_u32(gpu, &indices)?;
    let mut dec_map = vec![0i32; T];
    dec_map[5] = -1;
    let dec_map_dev = up_i32(gpu, &dec_map)?;
    let x_dec = gen_bf16(&mut rng, TE * INTER);
    let x_dec_dev = up_u16(gpu, &x_dec)?;
    let base_dec = gen_bf16(&mut rng, TE * H);
    let d_out_dev = up_u16(gpu, &base_dec)?;
    ops::moe_lora_gather_bgmv(
        gpu,
        &kernels,
        &down_route,
        x_dec_dev,
        d_out_dev,
        idx_dev,
        dec_map_dev,
        xa_dev,
        TE as u32,
        TOP_K as u32,
        0,
        st,
    )?;
    gpu.synchronize(st)?;
    let mut want_dec = base_dec.clone();
    for row in 0..TE {
        let e = indices[row] as usize;
        if dec_map[row / TOP_K] < 0 {
            continue;
        }
        if let Some((_, hp)) = down.iter().find(|(de, _)| *de as usize == e) {
            host_fold_row(
                hp,
                &x_dec[row * INTER..(row + 1) * INTER],
                &mut want_dec[row * H..(row + 1) * H],
            );
        }
    }
    assert_tol(
        "CHECK 6 decode gather",
        &rd_u16(gpu, d_out_dev, TE * H)?,
        &want_dec,
    );

    // 2026-09-25: CHECK 7: `apply_expert_lora_sorted` over the CHECK 2 inputs.
    let mut layer = ExpertLoraLayer::default();
    for (e, p) in &down_pairs {
        layer.pairs.insert((*e, ExpertProj::Down), *p);
    }
    let out4_dev = up_u16(gpu, &base_down)?;
    let offs_host: Vec<u32> = expert_offsets.iter().map(|&v| v as u32).collect();
    apply_expert_lora_sorted(
        gpu,
        &kernels,
        &layer,
        ExpertProj::Down,
        &offs_host,
        x_sorted_dev,
        out4_dev,
        TE as u32,
        xa_dev,
        delta_dev,
        st,
    )?;
    gpu.synchronize(st)?;
    assert_tol(
        "CHECK 7 host-loop entry",
        &rd_u16(gpu, out4_dev, TE * H)?,
        &want_down,
    );

    println!("moe_lora_delta_parity: ALL CHECKS PASSED (scale={SCALE})");
    Ok(())
}
