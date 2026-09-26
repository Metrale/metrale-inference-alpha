// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: QSA prefill-side GPU tests.
//!
//! Owner: model-layers (QSA).
//! Invariants: none beyond the types.
//!
//! A child module: `super` is the `qsa_tests` module (fixture helpers), and
//! `super::super` is `qsa` itself.

#![allow(unused_imports)]

use super::*;

/// 2026-09-25: Prefill selection sets vs the golden mask rows. Ingest all T
/// tokens, run `prefill_select` (its attention output is not checked: q and
/// the pools are uninitialised, and `qsa_prefill_attn_matches_cpu` covers the
/// kernel), read back the uploaded block lists and compare each selective
/// row's set against the golden `mask_sel_rows` row.
#[test]
#[ignore]
fn qsa_prefill_select_sets_match_reference() {
    let meta: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(format!("{}/meta.json", bins_dir())).unwrap(),
    )
    .unwrap();
    let t_all = meta["num_tokens"].as_u64().unwrap() as usize;
    let n_heads = meta["index_n_heads"].as_u64().unwrap() as usize;
    let hd = meta["index_head_dim"].as_u64().unwrap() as usize;
    let ratio = meta["compress_ratio"].as_u64().unwrap() as usize;
    let budget = meta["token_budget"].as_u64().unwrap() as usize;
    let topk = meta["block_topk"].as_u64().unwrap() as usize;
    let hidden = meta["hidden_size"].as_u64().unwrap() as usize;
    let rot = meta["rotary_dim"].as_u64().unwrap() as usize;
    let eps = meta["rms_norm_eps"].as_f64().unwrap() as f32;

    let set = metrale_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4")
        .expect("build with METRALE_TARGET_MODEL='*'");
    let gpu = metrale_gpu_runtime::cuda_backend::MetraleCudaBackend::new(0, &set.modules)
        .expect("CUDA backend");
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();

    let (nq, nkv, hd_attn, bs) = (24u32, 2usize, 256usize, 16usize);
    let qsa = QsaIndexer::new(
        upload(g, &bin("w_qk_proj")),
        upload(g, &bin("w_q_norm")),
        upload(g, &bin("w_k_norm")),
        n_heads,
        hd,
        ratio,
        budget,
        rot,
        1.0e7,
        eps,
        hidden,
        nkv,
        hd_attn,
        g,
    )
    .unwrap();
    let hidden_dev = upload(g, &bin("hidden"));
    let mut qst = qsa.new_seq_state(g).unwrap();
    qsa.prefill_ingest(&mut qst, hidden_dev, t_all, 0, g, stream)
        .unwrap();

    let q_row = nq as usize * hd_attn;
    let q_roped = g.alloc(t_all * q_row * 2).unwrap();
    let attn_ctx = g.alloc(t_all * q_row * 2).unwrap();
    let pages = t_all.div_ceil(bs);
    let kpool = g.alloc(pages * bs * nkv * hd_attn * 2).unwrap();
    let vpool = g.alloc(pages * bs * nkv * hd_attn * 2).unwrap();
    let host_table: Vec<u32> = (0..pages as u32).collect();

    const ROWS: usize = 2048;
    let qkw = (n_heads + 1) * hd;
    let stride = t_all.div_ceil(ratio);
    let scratch = g
        .alloc(ROWS * qkw * 2 + ROWS * n_heads * hd * 4 + ROWS * stride * 4 + ROWS * topk * 4)
        .unwrap();

    qsa.prefill_select(
        &mut qst,
        hidden_dev,
        q_roped,
        attn_ctx,
        kpool,
        vpool,
        &host_table,
        0,
        t_all,
        nq,
        bs as u32,
        1.0 / (hd_attn as f32).sqrt(),
        scratch,
        g,
        stream,
    )
    .unwrap();
    g.synchronize(stream).unwrap();

    let bound = qsa.inert_bound();
    let n_sel = t_all - bound;
    assert!(n_sel <= ROWS, "test assumes one slab");
    let lists_off = ROWS * qkw * 2 + ROWS * n_heads * hd * 4 + ROWS * stride * 4;
    let mut raw = vec![0u8; n_sel * topk * 4];
    g.copy_d2h(scratch.offset(lists_off), &mut raw).unwrap();
    let lists: Vec<i32> = raw
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // 2026-09-25: The engine's own scores for the (single) slab, to judge flips.
    let scores_off = ROWS * qkw * 2 + ROWS * n_heads * hd * 4;
    let mut sraw = vec![0u8; n_sel * stride * 4];
    g.copy_d2h(scratch.offset(scores_off), &mut sraw).unwrap();
    let eng_scores: Vec<f32> = sraw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // 2026-09-25: Golden: [n_sel, T] u8 mask rows for pos in [bound, T).
    let mask = bin("mask_sel_rows");
    assert_eq!(mask.len(), n_sel * t_all, "mask_sel_rows shape");
    let mut rows_exact = 0usize;
    let mut total_flips = 0usize;
    for r in 0..n_sel {
        let pos = bound + r;
        let complete = (pos + 1) / ratio;
        let want: std::collections::BTreeSet<i32> = (0..complete * ratio)
            .filter(|&t| mask[r * t_all + t] != 0)
            .map(|t| (t / ratio) as i32)
            .collect();
        for t in complete * ratio..=pos {
            assert_eq!(
                mask[r * t_all + t],
                1,
                "row {pos}: tail token {t} not visible"
            );
        }
        let got: std::collections::BTreeSet<i32> =
            lists[r * topk..(r + 1) * topk].iter().copied().collect();
        assert_eq!(want.len(), topk, "row {pos}: golden block count");
        assert_eq!(got.len(), topk, "row {pos}: duplicate blocks in list");
        let flips = want.difference(&got).count();
        total_flips += flips;
        if flips == 0 {
            rows_exact += 1;
        }
        // 2026-09-25: BF16 key storage against the FP32 golden lets blocks near
        // the top-k cutoff swap. Each flip is judged against the engine's own
        // scores: a golden-selected block the engine dropped must sit less
        // than 0.05 below the engine's cutoff.
        if flips > 0 {
            let row_sc = &eng_scores[r * stride..r * stride + complete];
            let mut sorted: Vec<f32> = row_sc.to_vec();
            sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
            let cutoff = sorted[topk - 1];
            for b in want.difference(&got) {
                let d = cutoff - row_sc[*b as usize];
                assert!(
                    (0.0..0.05).contains(&d),
                    "row {pos}: golden block {b} dropped with engine score \
                     {:.5} vs cutoff {cutoff:.5} (gap {d:.5}) — not a \
                     near-tie",
                    row_sc[*b as usize]
                );
            }
        }
    }
    println!(
        "qsa prefill sets: {rows_exact}/{n_sel} rows EXACT, {total_flips} total near-tie flips"
    );
}

/// 2026-09-25: `qsa_prefill_attn` vs a CPU reference on synthetic data, covering the
/// GQA head mapping, paged addressing and the tail tokens at a small topk.
#[test]
#[ignore]
fn qsa_prefill_attn_matches_cpu() {
    let set = metrale_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4")
        .expect("build with METRALE_TARGET_MODEL='*'");
    let gpu = metrale_gpu_runtime::cuda_backend::MetraleCudaBackend::new(0, &set.modules)
        .expect("CUDA backend");
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();
    let k = g.kernel("qsa_indexer", "qsa_prefill_attn").unwrap();

    let (rows, nq, nkv, hd, ratio, topk, bs) =
        (3usize, 24usize, 2usize, 256usize, 4usize, 8usize, 16usize);
    // 2026-09-25: Row 0 has 10 complete blocks (more than topk = 8) and a tail of 2.
    let first_pos = 41usize;
    let n_pos = first_pos + rows;
    let pages = n_pos.div_ceil(bs);

    let mut seed = 0x12345u32;
    let mut nextf = move || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        ((seed >> 8) as f32 / (1 << 24) as f32) - 0.5
    };
    let bf = |v: f32| -> u16 { (v.to_bits() >> 16) as u16 };
    let unbf = |u: u16| -> f32 { f32::from_bits((u as u32) << 16) };

    let q_host: Vec<u16> = (0..rows * nq * hd).map(|_| bf(nextf())).collect();
    let kv_elems = pages * bs * nkv * hd;
    let k_host: Vec<u16> = (0..kv_elems).map(|_| bf(nextf())).collect();
    let v_host: Vec<u16> = (0..kv_elems).map(|_| bf(nextf())).collect();
    let lists_host: Vec<i32> = (0..rows)
        .flat_map(|r| (0..topk as i32).map(move |i| (i * 5 + r as i32) % 10))
        .collect();

    let as_bytes = |v: &[u16]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let q_dev = upload(g, &as_bytes(&q_host));
    let k_dev = upload(g, &as_bytes(&k_host));
    let v_dev = upload(g, &as_bytes(&v_host));
    let lists_dev = upload(
        g,
        &lists_host
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<u8>>(),
    );
    let ident: Vec<u8> = (0..pages as i32).flat_map(|v| v.to_le_bytes()).collect();
    let table = upload(g, &ident);
    let out_dev = g.alloc(rows * nq * hd * 2).unwrap();
    let scale = 1.0 / (hd as f32).sqrt();

    ops::qsa_prefill_attn(
        g,
        k,
        q_dev,
        k_dev,
        v_dev,
        table,
        lists_dev,
        out_dev,
        rows as u32,
        first_pos as u32,
        topk as u32,
        ratio as u32,
        bs as u32,
        nq as u32,
        nkv as u32,
        hd as u32,
        scale,
        stream,
    )
    .unwrap();
    g.synchronize(stream).unwrap();
    let got = dl_bf16(g, out_dev, rows * nq * hd);

    let group = nq / nkv;
    let mut worst_cos = 1.0f64;
    for r in 0..rows {
        let pos = first_pos + r;
        let complete = (pos + 1) / ratio;
        let tail = (pos + 1) - complete * ratio;
        let mut toks: Vec<usize> = lists_host[r * topk..(r + 1) * topk]
            .iter()
            .flat_map(|&b| (0..ratio).map(move |i| b as usize * ratio + i))
            .collect();
        toks.extend(complete * ratio..complete * ratio + tail);
        for h in 0..nq {
            let kvh = h / group;
            let qv: Vec<f32> = (0..hd)
                .map(|d| unbf(q_host[(r * nq + h) * hd + d]))
                .collect();
            let scores: Vec<f32> = toks
                .iter()
                .map(|&t| {
                    let base = (t * nkv + kvh) * hd;
                    (0..hd).map(|d| qv[d] * unbf(k_host[base + d])).sum::<f32>() * scale
                })
                .collect();
            let m = scores.iter().cloned().fold(f32::MIN, f32::max);
            let exps: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
            let l: f32 = exps.iter().sum();
            let mut refv = vec![0.0f32; hd];
            for (i, &t) in toks.iter().enumerate() {
                let base = (t * nkv + kvh) * hd;
                let w = exps[i] / l;
                for d in 0..hd {
                    refv[d] += w * unbf(v_host[base + d]);
                }
            }
            let gv = &got[(r * nq + h) * hd..(r * nq + h + 1) * hd];
            let dot: f64 = gv
                .iter()
                .zip(&refv)
                .map(|(a, b)| *a as f64 * *b as f64)
                .sum();
            let ng: f64 = gv.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
            let nr: f64 = refv.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
            let cos = dot / (ng * nr).max(1e-30);
            worst_cos = worst_cos.min(cos);
        }
    }
    println!("qsa_prefill_attn vs CPU: worst cos = {worst_cos:.9}");
    assert!(worst_cos > 0.999, "attention kernel diverges: {worst_cos}");
}

/// 2026-09-25: Prints the cosine of `attn_prefill_64` against a CPU reference
/// (head 0) at probe rows of a causal 2809-token sequence with nq=24, nkv=2,
/// hd=256 and synthetic q/k/v. It asserts nothing.
#[test]
#[ignore]
fn flash64_long_seq_rows_repro() {
    let set = metrale_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4")
        .expect("build with METRALE_TARGET_MODEL='*'");
    let gpu = metrale_gpu_runtime::cuda_backend::MetraleCudaBackend::new(0, &set.modules)
        .expect("CUDA backend");
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();
    let k = g.kernel("attn_prefill", "attn_prefill_64").unwrap();

    let (n, nq, nkv, hd) = (2809usize, 24usize, 2usize, 256usize);
    let mut seed = 0xBEEFu32;
    let mut nextf = move || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        ((seed >> 8) as f32 / (1 << 24) as f32) - 0.5
    };
    let bf = |v: f32| -> u16 { (v.to_bits() >> 16) as u16 };
    let unbf = |u: u16| -> f32 { f32::from_bits((u as u32) << 16) };
    let q_host: Vec<u16> = (0..n * nq * hd).map(|_| bf(nextf())).collect();
    let k_host: Vec<u16> = (0..n * nkv * hd).map(|_| bf(nextf())).collect();
    let v_host: Vec<u16> = (0..n * nkv * hd).map(|_| bf(nextf())).collect();
    let as_bytes = |v: &[u16]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let q_dev = upload(g, &as_bytes(&q_host));
    let k_dev = upload(g, &as_bytes(&k_host));
    let v_dev = upload(g, &as_bytes(&v_host));
    let out_dev = g.alloc(n * nq * hd * 2).unwrap();
    // 2026-09-25: Poison the output so unwritten rows are detectable.
    let poison = vec![0x3Fu8; n * nq * hd * 2];
    g.copy_h2d_async(&poison, out_dev, stream).unwrap();
    let scale = 1.0 / (hd as f32).sqrt();

    ops::prefill_attention_64(
        g, k, q_dev, k_dev, v_dev, out_dev, n as u32, 1, nq as u32, nkv as u32, hd as u32, scale,
        true, 0, stream,
    )
    .unwrap();
    g.synchronize(stream).unwrap();
    let got = dl_bf16(g, out_dev, n * nq * hd);

    let group = nq / nkv;
    for &row in &[100usize, 1024, 1200, 1279, 1280, 1290, 1500, 2051, 2808] {
        // 2026-09-25: CPU reference for head 0 only.
        let h = 0usize;
        let kvh = h / group;
        let qv: Vec<f32> = (0..hd)
            .map(|d| unbf(q_host[(row * nq + h) * hd + d]))
            .collect();
        let mut m = f32::MIN;
        let scores: Vec<f32> = (0..=row)
            .map(|t| {
                let base = (t * nkv + kvh) * hd;
                let s: f32 = (0..hd).map(|d| qv[d] * unbf(k_host[base + d])).sum::<f32>() * scale;
                m = m.max(s);
                s
            })
            .collect();
        let exps: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
        let l: f32 = exps.iter().sum();
        let mut refv = vec![0.0f32; hd];
        for (t, e) in exps.iter().enumerate() {
            let base = (t * nkv + kvh) * hd;
            let w = e / l;
            for d in 0..hd {
                refv[d] += w * unbf(v_host[base + d]);
            }
        }
        let gv = &got[(row * nq + h) * hd..(row * nq + h) * hd + hd];
        let dot: f64 = gv
            .iter()
            .zip(&refv)
            .map(|(a, b)| *a as f64 * *b as f64)
            .sum();
        let ng: f64 = gv.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
        let nr: f64 = refv.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
        let cos = dot / (ng * nr).max(1e-30);
        println!("  flash64 row {row:>4}: cos={cos:.6} |got|={ng:.4} |ref|={nr:.4}");
    }
}
