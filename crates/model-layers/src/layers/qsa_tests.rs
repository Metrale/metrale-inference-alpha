// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: GPU parity for the QSA indexer against the reference module.
//!
//! Owner: model-layers (QSA).
//! Invariants: none beyond the types.
//!
//! Fixtures come from `bench/qwen4_exp/qsa_golden.py`, which builds the
//! reference `Qwen4ExpTextQSAIndexer` in FP32 with the checkpoint's layer-3
//! weights over T=2200 tokens (550 complete blocks, more than the checkpoint's
//! `block_topk`, so selection prunes).
//!
//! The engine stores raw and block keys in BF16, so scores near the top-k
//! cutoff can flip against the FP32 golden. The selection-set compare allows
//! up to 3 block swaps and asserts each is within tolerance of the cutoff
//! score; the selected count must match exactly.
//!
//! GPU test, `#[ignore]`d. Run with
//! ```text
//! METRALE_QSA_TEST_DATA=<fixture dir> \
//!   cargo test -p metrale-model-layers --release qsa_matches -- --ignored --nocapture
//! ```

use super::*;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

fn bins_dir() -> String {
    std::env::var("METRALE_QSA_TEST_DATA").expect(
        "set METRALE_QSA_TEST_DATA — generate with \
         `bench/qwen4_exp/qsa_golden.py --out .../qsa_golden.npz`",
    )
}

fn bin(name: &str) -> Vec<u8> {
    let p = format!("{}/{name}.bin", bins_dir());
    std::fs::read(&p).unwrap_or_else(|e| panic!("{p}: {e}"))
}

fn f32s(name: &str) -> Vec<f32> {
    bin(name)
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn i32s(name: &str) -> Vec<i32> {
    bin(name)
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn upload(g: &dyn GpuBackend, bytes: &[u8]) -> DevicePtr {
    let p = g.alloc(bytes.len()).unwrap();
    g.copy_h2d_async(bytes, p, g.default_stream()).unwrap();
    p
}

fn dl_bf16(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Vec<f32> {
    let mut raw = vec![0u8; n * 2];
    g.copy_d2h(p, &mut raw).unwrap();
    raw.chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect()
}

fn dl_f32(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Vec<f32> {
    let mut raw = vec![0u8; n * 4];
    g.copy_d2h(p, &mut raw).unwrap();
    raw.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn compare(label: &str, got: &[f32], want: &[f32], rel: f32) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    let rms =
        (want.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / want.len() as f64).sqrt() as f32;
    let tol = (rms * rel).max(1e-3);
    let mut max_abs = 0.0f32;
    let mut dot = 0.0f64;
    let (mut ng, mut nw) = (0.0f64, 0.0f64);
    for (&a, &b) in got.iter().zip(want) {
        max_abs = max_abs.max((a - b).abs());
        dot += a as f64 * b as f64;
        ng += a as f64 * a as f64;
        nw += b as f64 * b as f64;
    }
    let cos = dot / (ng.sqrt() * nw.sqrt()).max(1e-30);
    println!("  {label:<12} max|diff|={max_abs:.4e} cos={cos:.9} ref_rms={rms:.4e}");
    assert!(
        max_abs <= tol,
        "{label}: max|diff| {max_abs:.4e} > {tol:.4e}"
    );
    assert!(cos > 0.9999, "{label}: cosine {cos:.9}");
}

#[test]
#[ignore]
fn qsa_matches_reference() {
    let meta: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(format!("{}/meta.json", bins_dir())).unwrap(),
    )
    .unwrap();
    assert_eq!(
        meta["norm_convention"].as_str().unwrap(),
        "normed * (1.0 + weight)"
    );
    let t_all = meta["num_tokens"].as_u64().unwrap() as usize;
    let n_heads = meta["index_n_heads"].as_u64().unwrap() as usize;
    let hd = meta["index_head_dim"].as_u64().unwrap() as usize;
    let ratio = meta["compress_ratio"].as_u64().unwrap() as usize;
    let budget = meta["token_budget"].as_u64().unwrap() as usize;
    let topk = meta["block_topk"].as_u64().unwrap() as usize;
    let hidden = meta["hidden_size"].as_u64().unwrap() as usize;
    let rot = meta["rotary_dim"].as_u64().unwrap() as usize;
    let eps = meta["rms_norm_eps"].as_f64().unwrap() as f32;
    // 2026-09-25: The golden's rope uses the checkpoint config's rope_theta;
    // a wrong value here fails the q_post and block_keys compares.
    let theta = 1.0e7f32;

    let set = metrale_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4")
        .expect("build with METRALE_TARGET_MODEL='*'");
    let gpu = metrale_gpu_runtime::cuda_backend::MetraleCudaBackend::new(0, &set.modules)
        .expect("CUDA backend");
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();

    let qsa = QsaIndexer::new(
        upload(g, &bin("w_qk_proj")),
        upload(g, &bin("w_q_norm")),
        upload(g, &bin("w_k_norm")),
        n_heads,
        hd,
        ratio,
        budget,
        rot,
        theta,
        eps,
        hidden,
        2,
        256,
        g,
    )
    .unwrap();

    let hidden_dev = upload(g, &bin("hidden"));
    let mut qst = qsa.new_seq_state(g).unwrap();

    // 2026-09-25: Prefill ingest of T-1 tokens; the last token goes through
    // `decode_select`.
    let t_pre = t_all - 1;
    qsa.prefill_ingest(&mut qst, hidden_dev, t_pre, 0, g, stream)
        .unwrap();
    g.synchronize(stream).unwrap();

    let want_raw = f32s("raw_keys");
    compare(
        "raw_keys",
        &dl_bf16(g, qst.raw_keys, t_pre * hd),
        &want_raw[..t_pre * hd],
        0.05,
    );
    let n_blocks_pre = t_pre / ratio;
    let want_bk = f32s("block_keys");
    compare(
        "block_keys",
        &dl_bf16(g, qst.block_keys, n_blocks_pre * hd),
        &want_bk[..n_blocks_pre * hd],
        0.05,
    );

    // 2026-09-25: Decode step for the last token: ingest + select + gather.
    // The paged pools are uninitialised gather sources; their contents are
    // not checked.
    let bs = 16usize;
    let pages = t_all.div_ceil(bs);
    let row = 2 * 256;
    let kpool = g.alloc(pages * bs * row * 2).unwrap();
    let vpool = g.alloc(pages * bs * row * 2).unwrap();
    let ident: Vec<u8> = (0..pages as i32).flat_map(|v| v.to_le_bytes()).collect();
    let table = upload(g, &ident);

    let last_row = hidden_dev.offset((t_all - 1) * hidden * 2);
    let sel = qsa
        .decode_select(
            &mut qst,
            last_row,
            t_all - 1,
            kpool,
            vpool,
            table,
            bs as u32,
            g,
            stream,
        )
        .unwrap()
        .expect("selection must be ACTIVE at T=2200");
    g.synchronize(stream).unwrap();

    let want_q: Vec<f32> = f32s("q_post")[(t_all - 1) * n_heads * hd..].to_vec();
    compare(
        "q_post",
        &dl_f32(g, qsa.q_post, n_heads * hd),
        &want_q,
        0.05,
    );

    let n_blocks = t_all / ratio;
    let want_scores = f32s("scores_last");
    let got_scores = dl_f32(g, qsa.scores_dev, n_blocks);
    compare("scores", &got_scores, &want_scores, 0.05);

    // 2026-09-25: Selection set: exact up to near-tie flips at the top-k cutoff.
    // Each mismatched block's golden score must sit within tolerance of the
    // cutoff.
    let want_sel = i32s("selected_last");
    assert_eq!(sel.n_sel as usize, want_sel.len(), "selected count");
    let got_sel = {
        let mut b = vec![0u8; sel.n_sel as usize * 4];
        g.copy_d2h(qsa.sel_dev, &mut b).unwrap();
        b.chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect::<Vec<i32>>()
    };
    let got_set: std::collections::BTreeSet<i32> = got_sel.iter().copied().collect();
    let want_set: std::collections::BTreeSet<i32> = want_sel.iter().copied().collect();
    let missing: Vec<i32> = want_set.difference(&got_set).copied().collect();
    let extra: Vec<i32> = got_set.difference(&want_set).copied().collect();
    let flipped_blocks = missing.len().div_ceil(ratio);
    println!(
        "  selection    {} tokens, {} missing / {} extra (<= {} near-tie \
         block flips allowed)",
        want_sel.len(),
        missing.len(),
        extra.len(),
        3
    );
    assert_eq!(missing.len(), extra.len(), "selection: count drift");
    assert!(
        flipped_blocks <= 3,
        "selection: {flipped_blocks} block flips — more than near-tie noise \
         (missing {missing:?})"
    );
    if !missing.is_empty() {
        let mut sorted: Vec<f32> = want_scores.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let cutoff = sorted[topk - 1];
        for m in missing.iter().chain(extra.iter()) {
            let b = (*m as usize) / ratio;
            let d = (want_scores[b] - cutoff).abs();
            assert!(
                d < 0.02 * cutoff.abs().max(1.0),
                "selection: block {b} flipped with score {:.5} vs cutoff \
                 {cutoff:.5} — not a near-tie",
                want_scores[b]
            );
        }
    }
    println!("qsa parity OK: n_sel={} (budget {})", sel.n_sel, budget);
}

// 2026-09-25: Prefill-side tests, as a child module so the fixture helpers
// above are reachable through `super::*`.
#[path = "qsa_tests_prefill.rs"]
mod prefill;
