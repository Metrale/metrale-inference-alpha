// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: GPU parity tests of the Qwen3.8-Flash-Next low-rank mHC launchers against reference outputs.
//!
//! Owner: model-layers ops.
//! Invariants: none beyond the types.
//!
//! The fixtures come from bench/qwen4_exp/hc_golden.py, which runs `Qwen4ExpTextGatedResidual`
//! from the vendored bench/qwen4_exp/ref/modeling_qwen4_exp.py.
//!
//! Both tests are `#[ignore]` and need a GPU. Run with
//! ```text
//! METRALE_HC_TEST_DATA=/tank/metrale-testdata/qwen4exp_hc \
//!   cargo test -p metrale-model-layers --release hc_lowrank -- --ignored --nocapture
//! ```

use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use crate::layers::ops;
use crate::layers::qwen3_attention::HcLowRank;

struct Fixture {
    dir: String,
    hc: usize,
    h: usize,
    rank: usize,
    eps: f32,
    tokens: usize,
}

impl Fixture {
    fn load() -> Self {
        let dir = std::env::var("METRALE_HC_TEST_DATA").expect(
            "set METRALE_HC_TEST_DATA — generate with \
             `python3 -u bench/qwen4_exp/hc_golden.py --bin-dir <dir>`",
        );
        let meta: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(format!("{dir}/meta.json")).unwrap())
                .unwrap();
        // 2026-09-25: The fixture records the RMSNorm convention it was generated under.
        // `Qwen4ExpTextRMSNorm` scales by `1 + weight` and `Qwen4ExpTextRMSNormGated` by `weight`,
        // so a fixture generated under the other form is refused here rather than compared.
        assert_eq!(
            meta["norm_convention"].as_str().unwrap(),
            "normed * (1.0 + weight)",
            "fixture was generated under a different norm convention"
        );
        Self {
            dir,
            hc: meta["hc_count"].as_u64().unwrap() as usize,
            h: meta["hidden_size"].as_u64().unwrap() as usize,
            rank: meta["hc_lowrank"].as_u64().unwrap() as usize,
            eps: meta["rms_norm_eps"].as_f64().unwrap() as f32,
            tokens: meta["num_tokens"].as_u64().unwrap() as usize,
        }
    }

    fn bytes(&self, name: &str) -> Vec<u8> {
        let p = format!("{}/{name}.bin", self.dir);
        std::fs::read(&p).unwrap_or_else(|e| panic!("{p}: {e}"))
    }

    fn f32s(&self, name: &str) -> Vec<f32> {
        self.bytes(name)
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
}

fn upload(g: &dyn GpuBackend, bytes: &[u8]) -> DevicePtr {
    let p = g.alloc(bytes.len()).unwrap();
    g.copy_h2d_async(bytes, p, g.default_stream()).unwrap();
    p
}

fn download_bf16(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Vec<f32> {
    let mut raw = vec![0u8; n * 2];
    g.copy_d2h(p, &mut raw).unwrap();
    raw.chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect()
}

fn download_f32(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Vec<f32> {
    let mut raw = vec![0u8; n * 4];
    g.copy_d2h(p, &mut raw).unwrap();
    raw.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// 2026-09-25: Report the max absolute difference, the cosine similarity and the reference's RMS
/// together, and assert the first two: a near-zero output can pass a max-abs bound against a
/// small reference, and a correctly shaped but mis-scaled one has cosine 1.0.
fn compare(label: &str, got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    let mut max_abs = 0.0f32;
    let mut dot = 0.0f64;
    let (mut ng, mut nw) = (0.0f64, 0.0f64);
    let mut worst = 0usize;
    for (i, (&a, &b)) in got.iter().zip(want).enumerate() {
        let d = (a - b).abs();
        if d > max_abs {
            max_abs = d;
            worst = i;
        }
        dot += a as f64 * b as f64;
        ng += a as f64 * a as f64;
        nw += b as f64 * b as f64;
    }
    let cos = dot / (ng.sqrt() * nw.sqrt()).max(1e-30);
    let rms = (nw / want.len() as f64).sqrt();
    println!(
        "  {label:<22} max|diff|={max_abs:.4e} cos={cos:.9} \
         ref_rms={rms:.4e}  worst[{worst}] got={:.5} want={:.5}",
        got[worst], want[worst]
    );
    assert!(
        max_abs <= tol,
        "{label}: max|diff| {max_abs:.4e} exceeds {tol:.4e}"
    );
    assert!(cos > 0.9999, "{label}: cosine {cos:.9} — shape differs");
}

/// 2026-09-25: One site's low-rank weights, uploaded.
fn site_weights(g: &dyn GpuBackend, f: &Fixture, site: &str, inject: bool) -> HcLowRank {
    HcLowRank {
        norm_w: upload(g, &f.bytes(&format!("{site}_w_hc_norm"))),
        down_w: upload(g, &f.bytes(&format!("{site}_w_down"))),
        up_w: upload(g, &f.bytes(&format!("{site}_w_up"))),
        inject_w: if inject {
            upload(g, &f.bytes(&format!("{site}_w_inject")))
        } else {
            DevicePtr::NULL
        },
        rank: f.rank,
    }
}

/// 2026-09-25: Tolerance of 5% of the reference's RMS (at least 1e-3). A BF16 output has 8
/// significant bits, about 4e-3 relative, and the kernel sums its `hc * H`-term dot products in
/// FP32 in a different order than the reference.
fn tol_for(ref_vals: &[f32]) -> f32 {
    let rms = (ref_vals.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / ref_vals.len() as f64)
        .sqrt() as f32;
    (rms * 0.05).max(1e-3)
}

/// 2026-09-25: Tolerance of 12% of the reference's RMS for the small-T cuBLASLt path, which stages
/// `normed` in BF16 like the GEMM path and sums in cuBLASLt's order. [`compare`] still applies its
/// cosine bound.
fn tol_gemm(ref_vals: &[f32]) -> f32 {
    let rms = (ref_vals.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / ref_vals.len() as f64)
        .sqrt() as f32;
    (rms * 0.12).max(1e-3)
}

#[test]
#[ignore]
fn hc_lowrank_matches_reference() {
    let f = Fixture::load();
    // 2026-09-25: Load this target by identity, not through `ptx_modules()`: in a multi-target
    // build that is the first target's set, and another target's `hyper_connection` module may be
    // the DeepSeek-V4 file, whose kernels have the same names and different arguments.
    let set = metrale_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4").expect(
        "qwen3.8-flash-next/nvfp4 is not in this build — \
         build with METRALE_TARGET_MODEL='*' or =qwen3.8-flash-next",
    );
    let gpu = metrale_gpu_runtime::cuda_backend::MetraleCudaBackend::new(0, &set.modules)
        .expect("CUDA backend");
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();
    let (t, h, hc) = (f.tokens, f.h, f.hc);
    println!(
        "hc={hc} hidden={h} rank={} tokens={t} eps={:e}",
        f.rank, f.eps
    );

    let k_pre = g.kernel("hyper_connection", "hc_pre").unwrap();
    let k_head = g.kernel("hyper_connection", "hc_head").unwrap();
    let k_post = g.kernel("hyper_connection", "hc_post").unwrap();
    for (name, k) in [("hc_pre", k_pre), ("hc_head", k_head), ("hc_post", k_post)] {
        assert!(
            k.0 != 0,
            "{name} resolved to handle 0 — the qwen3.8-flash-next shadow is \
             not the one loaded, so this would be testing DeepSeek's kernel"
        );
    }

    let streams = upload(g, &f.bytes("streams"));
    let y_out = g.alloc(t * h * 2).unwrap();
    let inj_out = g.alloc(t * hc * 4).unwrap();
    // 2026-09-25: The split layout `sizes.rs` reserves at 64 rows, FP32 [64, hc*h + rank]. The
    // cuBLASLt GEMM layout at the fixture's T fits inside it.
    let scratch = g.alloc(64 * (hc * h + f.rank) * 4).unwrap();

    // 2026-09-25: hc_pre for both sites, on both small-T paths: the public entry takes the
    // cuBLASLt path (checked with `tol_gemm`), and the split path is called directly (checked with
    // `tol_for`).
    for site in ["attn", "mlp"] {
        let w = site_weights(g, &f, site, true);
        let want_mixed = f.f32s(&format!("{site}_mixed"));
        let want_inj = f.f32s(&format!("{site}_inj"));

        ops::hc_pre_lowrank(
            g, k_pre, streams, &w, y_out, inj_out, scratch, t as u32, h as u32, hc as u32, f.eps,
            stream,
        )
        .unwrap();
        g.synchronize(stream).unwrap();
        println!("{site}_hyper_connection (cublas arm):");
        compare(
            "mixed_input",
            &download_bf16(g, y_out, t * h),
            &want_mixed,
            tol_gemm(&want_mixed),
        );
        compare(
            "injection_weights",
            &download_f32(g, inj_out, t * hc),
            &want_inj,
            tol_gemm(&want_inj),
        );

        super::hyper_connection_lowrank::hc_pre_split(
            g, streams, &w, y_out, inj_out, scratch, t as u32, h as u32, hc as u32, f.eps, true,
            stream,
        )
        .unwrap();
        g.synchronize(stream).unwrap();
        println!("{site}_hyper_connection (split arm):");
        compare(
            "mixed_input",
            &download_bf16(g, y_out, t * h),
            &want_mixed,
            tol_for(&want_mixed),
        );
        compare(
            "injection_weights",
            &download_f32(g, inj_out, t * hc),
            &want_inj,
            tol_for(&want_inj),
        );
    }

    // 2026-09-25: hc_head, the model-level mixer (`use_combine=False` in the reference).
    let w_head = site_weights(g, &f, "head", false);
    ops::hc_head_lowrank(
        g, k_head, streams, &w_head, y_out, scratch, t as u32, h as u32, hc as u32, f.eps, stream,
    )
    .unwrap();
    g.synchronize(stream).unwrap();
    let want_head = f.f32s("head_mixed");
    println!("hyper_connection_mixer (cublas arm):");
    compare(
        "mixed_input",
        &download_bf16(g, y_out, t * h),
        &want_head,
        tol_gemm(&want_head),
    );

    super::hyper_connection_lowrank::hc_pre_split(
        g,
        streams,
        &w_head,
        y_out,
        DevicePtr::NULL,
        scratch,
        t as u32,
        h as u32,
        hc as u32,
        f.eps,
        false,
        stream,
    )
    .unwrap();
    g.synchronize(stream).unwrap();
    println!("hyper_connection_mixer (split arm):");
    compare(
        "mixed_input",
        &download_bf16(g, y_out, t * h),
        &want_head,
        tol_for(&want_head),
    );

    let block_out = upload(g, &f.bytes("post_block_out"));
    let inj = upload(
        g,
        &f.f32s("attn_inj")
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    let post_out = g.alloc(t * hc * h * 4).unwrap();
    ops::hc_post_lowrank(
        g, k_post, block_out, streams, inj, post_out, t as u32, h as u32, hc as u32, stream,
    )
    .unwrap();
    g.synchronize(stream).unwrap();
    let want_post = f.f32s("post_expected");
    println!("hc_post:");
    compare(
        "residual",
        &download_f32(g, post_out, t * hc * h),
        &want_post,
        tol_for(&want_post),
    );
}

/// 2026-09-25: The GEMM path (T > 64) against the same reference outputs. The collapse has no
/// cross-token term, so the fixture tiled 12 times is a valid large-T golden in which every copy
/// must reproduce the reference. At this T `hc_pre_lowrank` takes `hc_pre_gemm`, which rounds
/// `normed` to BF16; the check uses `tol_for`.
#[test]
#[ignore]
fn hc_pre_gemm_matches_reference() {
    const TILE: usize = 12;
    let f = Fixture::load();
    let set = metrale_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4").expect(
        "qwen3.8-flash-next/nvfp4 is not in this build — \
         build with METRALE_TARGET_MODEL='*' or =qwen3.8-flash-next",
    );
    let gpu = metrale_gpu_runtime::cuda_backend::MetraleCudaBackend::new(0, &set.modules)
        .expect("CUDA backend");
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();
    let (t, h, hc) = (f.tokens, f.h, f.hc);
    let big_t = t * TILE;
    assert!(
        big_t > 64,
        "tiled fixture must exceed the split-path ceiling"
    );

    let k_pre = g.kernel("hyper_connection", "hc_pre").unwrap();
    for name in ["hc_pre_stage_bf16", "hc_silu_scale", "hc_pre_mix"] {
        let k = g.kernel("hyper_connection", name).unwrap();
        assert!(k.0 != 0, "{name} resolved to handle 0");
    }

    let stream_bytes = f.bytes("streams");
    let tiled: Vec<u8> = stream_bytes
        .iter()
        .copied()
        .cycle()
        .take(stream_bytes.len() * TILE)
        .collect();
    let streams = upload(g, &tiled);
    let y_out = g.alloc(big_t * h * 2).unwrap();
    let inj_out = g.alloc(big_t * hc * 4).unwrap();
    // 2026-09-25: The GEMM layout `sizes.rs` reserves for `L = min(T, 2048) = big_t` rows.
    let scratch = g.alloc(big_t * (2 * hc * h + f.rank + hc) * 2).unwrap();

    for site in ["attn", "mlp"] {
        let w = site_weights(g, &f, site, true);
        ops::hc_pre_lowrank(
            g,
            k_pre,
            streams,
            &w,
            y_out,
            inj_out,
            scratch,
            big_t as u32,
            h as u32,
            hc as u32,
            f.eps,
            stream,
        )
        .unwrap();
        g.synchronize(stream).unwrap();

        let want_mixed: Vec<f32> = {
            let one = f.f32s(&format!("{site}_mixed"));
            one.iter().copied().cycle().take(one.len() * TILE).collect()
        };
        let want_inj: Vec<f32> = {
            let one = f.f32s(&format!("{site}_inj"));
            one.iter().copied().cycle().take(one.len() * TILE).collect()
        };
        println!("{site}_hyper_connection (GEMM path, T={big_t}):");
        compare(
            "mixed_input",
            &download_bf16(g, y_out, big_t * h),
            &want_mixed,
            tol_for(&want_mixed),
        );
        compare(
            "injection_weights",
            &download_f32(g, inj_out, big_t * hc),
            &want_inj,
            tol_for(&want_inj),
        );
    }

    // 2026-09-25: The model-level mixer takes the same GEMM path with `inject = false`.
    let k_head = g.kernel("hyper_connection", "hc_head").unwrap();
    let w_head = site_weights(g, &f, "head", false);
    ops::hc_head_lowrank(
        g,
        k_head,
        streams,
        &w_head,
        y_out,
        scratch,
        big_t as u32,
        h as u32,
        hc as u32,
        f.eps,
        stream,
    )
    .unwrap();
    g.synchronize(stream).unwrap();
    let want_head: Vec<f32> = {
        let one = f.f32s("head_mixed");
        one.iter().copied().cycle().take(one.len() * TILE).collect()
    };
    println!("hyper_connection_mixer (GEMM path, T={big_t}):");
    compare(
        "mixed_input",
        &download_bf16(g, y_out, big_t * h),
        &want_head,
        tol_for(&want_head),
    );
}
