// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The MLA check of `dsa_indexer_microtest`: rebuilds the indexer
//! selection with the CPU reference, runs `dsa_topk_to_mask` and
//! `dsa_mla_masked_attn` over four regimes, and compares the attention and final
//! outputs with the bf16 golden.
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.

#![allow(unused_imports)]

use crate::*;
use anyhow::{Context, Result, bail};
use dsa_indexer_layer::*;
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_arch::glm5next_dsa_ref as dref;
use metrale_model_arch::glm5next_dsa_ref::{DsaDims, INVALID};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

/// 2026-09-25: The indexer's selection from the CPU reference, with each
/// intermediate rounded to bf16. `q_resid` is `rms_norm(q_a_proj(hidden))`, where
/// `indexer_layer` uses a random residual.
#[allow(clippy::too_many_arguments)]
pub(crate) fn reference_selection(
    hidden: &[f32],
    valid: &[u8],
    pkt: &Packet,
    dims: DsaDims,
    s: usize,
) -> Result<Vec<i32>> {
    let (hid, ih, ihd, kp, ql) = (
        dims.hidden,
        dims.index_heads,
        dims.index_head_dim,
        dims.index_kpool,
        dims.q_lora_rank,
    );
    let qa = round_bf16(&gemm(
        hidden,
        s,
        hid,
        &round_bf16(&pkt.f32s("self_attn.q_a_proj.weight")?),
        ql,
    ));
    let q_resid = round_bf16(&dref::rms_norm(
        &qa,
        &pkt.f32s("self_attn.q_a_layernorm.weight")?,
        ql,
        1e-5,
    ));
    let q = round_bf16(&gemm(
        &q_resid,
        s,
        ql,
        &round_bf16(&pkt.f32s("self_attn.indexer.wq_b.weight")?),
        ih * ihd,
    ));
    let k_raw = round_bf16(&gemm(
        hidden,
        s,
        hid,
        &round_bf16(&pkt.f32s("self_attn.indexer.wk.weight")?),
        ihd,
    ));
    let k_normed = round_bf16(&dref::layer_norm(
        &k_raw,
        &pkt.f32s("self_attn.indexer.k_norm.weight")?,
        &pkt.f32s("self_attn.indexer.k_norm.bias")?,
        ihd,
        1e-6,
    ));
    let gate = round_bf16(&gemm(
        hidden,
        s,
        hid,
        &round_bf16(&pkt.f32s("self_attn.indexer.index_kpool_compress_gate")?),
        ihd,
    ));
    let wp = round_bf16(&gemm(
        hidden,
        s,
        hid,
        &round_bf16(&pkt.f32s("self_attn.indexer.weights_proj.weight")?),
        ih,
    ));
    let hscale = (ih as f32).powf(-0.5);
    let weights: Vec<f32> = wp.iter().map(|x| x * hscale).collect();
    let ape = pkt.f32s("self_attn.indexer.index_kpool_compress_ape")?;

    let pools = dref::pool_states(&k_normed, &gate, valid, &ape, dims, s);
    let vc: Vec<u8> = (0..s)
        .flat_map(|rr| (0..pools.n_pools).map(move |p| (rr, p)))
        .map(|(rr, p)| {
            let e = pools.indices[p * kp + kp - 1];
            let ec = e.clamp(0, s as i32 - 1) as usize;
            (pools.valid[p] != 0 && dref::visible(valid, rr, ec)) as u8
        })
        .collect();
    let raw = dref::index_scores(&q, &weights, &pools, dims, s);
    let masked: Vec<f32> = raw
        .iter()
        .enumerate()
        .map(|(i, v)| if vc[i] != 0 { *v } else { f32::MIN })
        .collect();
    let select_k = dims.select_k(pools.n_pools);
    let sel = dref::topk_pools(&masked, &vc, pools.n_pools, s, select_k);
    let qpos: Vec<usize> = (0..s).collect();
    Ok(dref::expand_selection(
        &sel, &pools, &vc, valid, &qpos, valid, dims, s, select_k,
    ))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn mla_layer(
    gpu: &dyn GpuBackend,
    k: &Kernels,
    gold: &Value,
    dims: DsaDims,
    layer: usize,
    pkt: &Packet,
) -> Result<bool> {
    let (hid, h, nope, vd, kvl, ql) = (
        dims.hidden,
        dims.heads,
        dims.qk_nope_head_dim,
        dims.v_head_dim,
        dims.kv_lora_rank,
        dims.q_lora_rank,
    );
    let w_qa = round_bf16(&pkt.f32s("self_attn.q_a_proj.weight")?);
    let n_qa = pkt.f32s("self_attn.q_a_layernorm.weight")?;
    let w_qb = round_bf16(&pkt.f32s("self_attn.q_b_proj.weight")?);
    let w_kva = round_bf16(&pkt.f32s("self_attn.kv_a_proj_with_mqa.weight")?);
    let n_kva = pkt.f32s("self_attn.kv_a_layernorm.weight")?;
    let w_kvb = round_bf16(&pkt.f32s("self_attn.kv_b_proj.weight")?);
    let w_o = round_bf16(&pkt.f32s("self_attn.o_proj.weight")?);
    let _ = ql;

    let mut all_ok = true;
    println!(
        "\n  layer {layer}   {:<12} {:>6} {:>9} {:>11} {:>11} {:>11} {:>8}",
        "regime", "S", "visible", "attn maxabs", "floorB", "final maxabs", "verdict"
    );
    for (rname, s, pad) in [
        ("short7", 7usize, 0usize),
        ("medium64", 64, 0),
        ("ragged13", 13, 5),
        ("sparse2176", 2176, 0),
    ] {
        let gb = &gold["by_layer"][layer.to_string()][format!("bf16__{rname}")];
        let gf = &gold["by_layer"][layer.to_string()][format!("f32__{rname}")];
        if gb.is_null() {
            bail!("MLA golden missing bf16__{rname} for layer {layer}");
        }
        let mut rng = Lcg(0x0D5A_C0DE);
        let hidden = round_bf16(&rng.vec(s * hid).iter().map(|x| x * 0.5).collect::<Vec<_>>());
        // 2026-09-25: `ragged13` has five leading pad tokens; the other regimes have none.
        let valid: Vec<u8> = (0..s).map(|i| (i >= pad) as u8).collect();

        let qa = gemm(&hidden, s, hid, &w_qa, ql);
        let q_resid = round_bf16(&dref::rms_norm(&round_bf16(&qa), &n_qa, ql, 1e-5));
        let q = round_bf16(&gemm(&q_resid, s, ql, &w_qb, h * nope));
        let kva = gemm(&hidden, s, hid, &w_kva, kvl);
        let kv_c = round_bf16(&dref::rms_norm(&round_bf16(&kva), &n_kva, kvl, 1e-5));
        let (kk, vv) = dref::expand_kv(&kv_c, &w_kvb, dims, s);
        let (kk, vv) = (round_bf16(&kk), round_bf16(&vv));

        // 2026-09-25: The selection comes from the CPU reference; `vis_ok` below compares
        // its per-row visible counts with the golden's `visible_per_row`.
        let width = dims.out_width();
        let topk = reference_selection(&hidden, &valid, pkt, dims, s)?;

        let d_tk = up_i32(gpu, &topk)?;
        let d_mask = gpu.alloc(s * s)?;
        KernelLaunch::new(gpu, k.mask)
            .grid([s as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(d_tk)
            .arg_ptr(d_mask)
            .arg_u32(s as u32)
            .arg_u32(width as u32)
            .arg_u32(s as u32)
            .launch(0)?;
        let d_q = up_bf16(gpu, &q)?;
        let d_k = up_bf16(gpu, &kk)?;
        let d_v = up_bf16(gpu, &vv)?;
        let d_o = gpu.alloc(s * h * vd * 4)?;
        // 2026-09-25: The kernel stages the S-float score row in dynamic shared memory.
        let smem = s * 4;
        if smem > SMEM_CEILING {
            bail!("MLA needs {smem} B shared for S={s}; ceiling {SMEM_CEILING} (S <= 12288)");
        }
        KernelLaunch::new(gpu, k.mla)
            .grid([s as u32, h as u32, 1])
            .block([256, 1, 1])
            .shared_mem(smem as u32)
            .arg_ptr(d_q)
            .arg_ptr(d_k)
            .arg_ptr(d_v)
            .arg_ptr(d_mask)
            .arg_ptr(d_o)
            .arg_u32(s as u32)
            .arg_u32(s as u32)
            .arg_u32(h as u32)
            .arg_u32(nope as u32)
            .arg_u32(vd as u32)
            .arg_f32((nope as f32).powf(-0.5))
            .arg_u32(1)
            .launch(0)?;
        gpu.synchronize(0)?;

        let attn = down_f32(gpu, d_o, s * h * vd)?;
        let mask = down_u8(gpu, d_mask, s * s)?;
        let visible: usize = (0..s)
            .map(|rr| {
                mask[rr * s..(rr + 1) * s]
                    .iter()
                    .filter(|x| **x != 0)
                    .count()
            })
            .max()
            .unwrap_or(0);

        let e_a = entry(gb, "attn_out")?;
        e_a.expect_n(s * h * vd, "attn_out")?;
        let e_af = entry(gf, "attn_out")?;
        let ga: Vec<f32> = e_a.data.iter().map(|x| *x as f32).collect();
        let gaf: Vec<f32> = e_af.data.iter().map(|x| *x as f32).collect();
        // 2026-09-25: Only query rows at or after `pad` are compared; a padded query row
        // selects no key.
        let per_row = h * vd;
        let keep_row = |flat_idx: usize| -> bool { (flat_idx / per_row) >= pad };
        let sa: Vec<f32> = attn
            .iter()
            .enumerate()
            .step_by(e_a.stride)
            .filter(|(i, _)| keep_row(*i))
            .map(|(_, v)| *v)
            .collect();
        let ga2: Vec<f32> = ga
            .iter()
            .enumerate()
            .filter(|(j, _)| keep_row(j * e_a.stride))
            .map(|(_, v)| *v)
            .collect();
        let gaf2: Vec<f32> = gaf
            .iter()
            .enumerate()
            .filter(|(j, _)| keep_row(j * e_a.stride))
            .map(|(_, v)| *v)
            .collect();
        let attn_err = maxabs(&sa, &ga2);
        let floor_b = maxabs(&ga2, &gaf2);

        let final_out = round_bf16(&gemm(&round_bf16(&attn), s, h * vd, &w_o, hid));
        let e_f = entry(gb, "final_out")?;
        e_f.expect_n(s * hid, "final_out")?;
        let e_ff = entry(gf, "final_out")?;
        let gfin: Vec<f32> = e_f.data.iter().map(|x| *x as f32).collect();
        let gfinf: Vec<f32> = e_ff.data.iter().map(|x| *x as f32).collect();
        let keep_fin = |flat_idx: usize| -> bool { (flat_idx / hid) >= pad };
        let sf: Vec<f32> = final_out
            .iter()
            .enumerate()
            .step_by(e_f.stride)
            .filter(|(i, _)| keep_fin(*i))
            .map(|(_, v)| *v)
            .collect();
        let gfin2: Vec<f32> = gfin
            .iter()
            .enumerate()
            .filter(|(j, _)| keep_fin(j * e_f.stride))
            .map(|(_, v)| *v)
            .collect();
        let gfinf2: Vec<f32> = gfinf
            .iter()
            .enumerate()
            .filter(|(j, _)| keep_fin(j * e_f.stride))
            .map(|(_, v)| *v)
            .collect();
        let fin_err = maxabs(&sf, &gfin2);
        let fin_floor = maxabs(&gfin2, &gfinf2);
        let mag = gfinf2.iter().fold(0.0f64, |m, x| m.max((*x as f64).abs()));

        let e_v = entry(gb, "visible_per_row")?;
        let vis_ok = e_v.data.iter().enumerate().all(|(rr, x)| {
            mask[rr * s..(rr + 1) * s]
                .iter()
                .filter(|y| **y != 0)
                .count()
                == *x as usize
        });
        let pass = vis_ok
            && attn_err <= (floor_b * 8.0).max(1e-3)
            && fin_err <= (fin_floor * 8.0).max(mag * 0.01);
        all_ok &= pass;
        println!(
            "           {rname:<12} {s:>6} {visible:>9} {attn_err:>11.3e} {floor_b:>11.3e} \
             {fin_err:>11.3e} {:>8}",
            if pass { "ok" } else { "FAIL" }
        );
        if !pass {
            println!("             vis_ok={vis_ok} fin_floor={fin_floor:.3e} mag={mag:.3e}");
        }
        let _ = checksum(&attn);
    }
    Ok(all_ok)
}
