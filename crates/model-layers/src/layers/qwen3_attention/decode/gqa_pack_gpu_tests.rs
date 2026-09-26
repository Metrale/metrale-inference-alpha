// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: GPU equality test: the GQA-packed paged-decode kernels
//! `paged_decode_attn_{fp8,bf16}_gqa` against the unpacked `paged_decode_attn_fp8` /
//! `paged_decode_attn`, over the same Q, KV pool, block table and seq lens, compared byte for byte
//! with no tolerance.
//!
//! The test also asserts that the comparison is not vacuous:
//! - the shape is one [`attn_splitk::gqa_pack_shape_ok`] accepts;
//! - [`attn_splitk::gqa_pack_enabled`] is true in this process (it is false unless
//!   `METRALE_ATTN_DECODE_GQA_PACK` arms it), so the run fails rather than passing vacuously;
//! - all four entry points resolve to non-zero handles, and the packed and unpacked handles differ;
//! - [`splitk_dispatch::gqa_pack_kernel`], the production route, returns the packed handle for this
//!   shape and refuses a head ratio the kernel cannot index;
//! - each output buffer starts at the `UNWRITTEN` sentinel, and every head of both outputs left it.
//!   Under the packed grid `(num_kv_heads, num_seqs)` an unpacked kernel writes only the first
//!   `NKV` heads.
//!
//! `#[ignore]`: it needs a GPU and a built PTX set. Run with:
//! ```text
//! METRALE_ATTN_DECODE_GQA_PACK=1 cargo test -p metrale-model-layers --release \
//!   gqa_packed_decode_is_byte_identical -- --ignored --nocapture
//! ```
//!
//! Owner: model-layers attention decode.
//! Invariants: none beyond the types.

use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_kernels::attn_splitk;

use super::gqa_pack_fixture::{
    BANDS, BLOCK_SIZE, Band, CASES, Case, Fixture, HD, K_SCALE, NKV, NQ, UNWRITTEN, V_SCALE,
    all_finite, bf16_bits, fixture, fp8_byte, heads_written, uniform,
};
use super::splitk_dispatch;
use crate::layers::ops;

#[test]
#[ignore]
#[allow(clippy::too_many_lines)]
fn gqa_packed_decode_is_byte_identical_to_unpacked() {
    const NEEDED: [&str; 4] = [
        "paged_decode",
        "paged_decode_fp8",
        "paged_decode_attn_bf16_gqa",
        "paged_decode_attn_fp8_gqa",
    ];
    let set = metrale_kernels::all_ptx_sets()
        .into_iter()
        .find(|t| {
            NEEDED
                .iter()
                .all(|m| t.modules.iter().any(|(name, _)| name == m))
        })
        .expect(
            "no built PTX target carries paged_decode + paged_decode_fp8 + both \
             GQA-packed twins; build the GB10 kernels first",
        );
    println!(
        "target {}/{} arch {}",
        set.target.model, set.target.quant, set.ptx_arch
    );

    // 2026-09-25: The shape gate, asserted against the predicate the dispatch uses.
    assert_eq!(
        NQ / NKV,
        attn_splitk::DECODE_GQA_PACK_WIDTH,
        "nq/nkv must be the pack width"
    );
    assert_eq!(HD, attn_splitk::DECODE_GQA_PACK_HEAD_DIM);
    assert!(
        attn_splitk::gqa_pack_shape_ok(NQ, NKV, HD),
        "nq={NQ} nkv={NKV} head_dim={HD} is not a shape the packed kernels serve",
    );

    // 2026-09-25: The lever. Unarmed, `gqa_pack_kernel` returns None and the comparisons below
    // would test nothing.
    assert!(
        attn_splitk::gqa_pack_enabled(),
        "METRALE_ATTN_DECODE_GQA_PACK is not armed in this process — re-run with \
         METRALE_ATTN_DECODE_GQA_PACK=1. Without it the packed kernel is never \
         the one production would pick and this test would be measuring a route \
         nobody takes.",
    );

    let gpu = metrale_gpu_runtime::cuda_backend::MetraleCudaBackend::new(0, &set.modules)
        .expect("CUDA backend");
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();

    let k_fp8 = g
        .kernel("paged_decode_fp8", "paged_decode_attn_fp8")
        .unwrap();
    let k_fp8_gqa = g
        .kernel("paged_decode_attn_fp8_gqa", "paged_decode_attn_fp8_gqa")
        .unwrap();
    let k_bf16 = g.kernel("paged_decode", "paged_decode_attn").unwrap();
    let k_bf16_gqa = g
        .kernel("paged_decode_attn_bf16_gqa", "paged_decode_attn_bf16_gqa")
        .unwrap();

    for (n, h) in [
        ("paged_decode_fp8::paged_decode_attn_fp8", k_fp8),
        (
            "paged_decode_attn_fp8_gqa::paged_decode_attn_fp8_gqa",
            k_fp8_gqa,
        ),
        ("paged_decode::paged_decode_attn", k_bf16),
        (
            "paged_decode_attn_bf16_gqa::paged_decode_attn_bf16_gqa",
            k_bf16_gqa,
        ),
    ] {
        assert!(h.0 != 0, "{n} resolved to handle 0");
        println!("resolved {n} -> handle {}", h.0);
    }
    assert_ne!(k_fp8.0, k_fp8_gqa.0, "FP8 arms resolved the SAME kernel");
    assert_ne!(k_bf16.0, k_bf16_gqa.0, "BF16 arms resolved the SAME kernel");

    for (arm, h) in [("fp8", k_fp8_gqa), ("bf16", k_bf16_gqa)] {
        let routed = splitk_dispatch::gqa_pack_kernel(Some(h), NQ, NKV, HD);
        assert_eq!(
            routed.map(|r| r.0),
            Some(h.0),
            "{arm}: gqa_pack_kernel refused the packed handle at the production shape",
        );
        // 2026-09-25: And it refuses a head ratio the kernel cannot index.
        assert!(
            splitk_dispatch::gqa_pack_kernel(Some(h), NQ + 1, NKV, HD).is_none(),
            "{arm}: gqa_pack_kernel accepted nq={} over nkv={NKV}",
            NQ + 1,
        );
    }

    let upload = |b: &[u8]| -> DevicePtr {
        let p = g.alloc(b.len().max(256)).unwrap();
        g.copy_h2d_async(b, p, stream).unwrap();
        p
    };
    let out_pair = |bytes: usize| -> (DevicePtr, DevicePtr) {
        let a = g.alloc(bytes).unwrap();
        let b = g.alloc(bytes).unwrap();
        g.memset_async(a, UNWRITTEN, bytes, stream).unwrap();
        g.memset_async(b, UNWRITTEN, bytes, stream).unwrap();
        (a, b)
    };

    let mut compared = 0usize;
    for (ci, case) in CASES.iter().enumerate() {
        for (bi, band) in BANDS.iter().enumerate() {
            let seed = 0x9E37_79B9_0000_0001 ^ ((ci as u64) << 8) ^ (bi as u64);
            let inv_sqrt_d = 1.0f32 / (HD as f32).sqrt();
            let cache_stride = u64::from(BLOCK_SIZE * NKV * HD);

            let (lo, hi) = band.fp8_exp;
            let f = fixture(case, seed, 1, |s| vec![fp8_byte(s, lo, hi)]);
            let (q, kp, vp, bt, sl) = (
                upload(&f.q),
                upload(&f.k),
                upload(&f.v),
                upload(&f.block_table),
                upload(&f.seq_lens),
            );
            let (o_ref, o_gqa) = out_pair(f.out_bytes);
            ops::paged_decode_attn_fp8(
                g,
                k_fp8,
                q,
                kp,
                vp,
                o_ref,
                bt,
                sl,
                f.max_blocks_per_seq,
                f.num_seqs,
                NQ,
                NKV,
                HD,
                BLOCK_SIZE,
                inv_sqrt_d,
                K_SCALE,
                V_SCALE,
                NQ * HD,
                cache_stride,
                case.sliding,
                stream,
            )
            .unwrap();
            ops::paged_decode_attn_fp8_gqa(
                g,
                k_fp8_gqa,
                q,
                kp,
                vp,
                o_gqa,
                bt,
                sl,
                f.max_blocks_per_seq,
                f.num_seqs,
                NQ,
                NKV,
                HD,
                BLOCK_SIZE,
                inv_sqrt_d,
                K_SCALE,
                V_SCALE,
                NQ * HD,
                cache_stride,
                case.sliding,
                stream,
            )
            .unwrap();
            g.synchronize(stream).unwrap();
            compare("fp8", case, band, &f, g, o_ref, o_gqa);
            compared += 1;
            for p in [q, kp, vp, bt, sl, o_ref, o_gqa] {
                g.free(p).unwrap();
            }

            let mag = band.bf16_mag;
            let f = fixture(case, seed ^ 0xBF16, 2, |s| {
                bf16_bits(uniform(s, mag)).to_le_bytes().to_vec()
            });
            let (q, kp, vp, bt, sl) = (
                upload(&f.q),
                upload(&f.k),
                upload(&f.v),
                upload(&f.block_table),
                upload(&f.seq_lens),
            );
            let (o_ref, o_gqa) = out_pair(f.out_bytes);
            ops::paged_decode_attn_bf16(
                g,
                k_bf16,
                q,
                kp,
                vp,
                o_ref,
                bt,
                sl,
                f.max_blocks_per_seq,
                f.num_seqs,
                NQ,
                NKV,
                HD,
                BLOCK_SIZE,
                inv_sqrt_d,
                NQ * HD,
                case.sliding,
                stream,
            )
            .unwrap();
            ops::paged_decode_attn_bf16_gqa(
                g,
                k_bf16_gqa,
                q,
                kp,
                vp,
                o_gqa,
                bt,
                sl,
                f.max_blocks_per_seq,
                f.num_seqs,
                NQ,
                NKV,
                HD,
                BLOCK_SIZE,
                inv_sqrt_d,
                NQ * HD,
                case.sliding,
                stream,
            )
            .unwrap();
            g.synchronize(stream).unwrap();
            compare("bf16", case, band, &f, g, o_ref, o_gqa);
            compared += 1;
            for p in [q, kp, vp, bt, sl, o_ref, o_gqa] {
                g.free(p).unwrap();
            }
        }
    }
    println!("{compared} (case, band, dtype) launches compared byte-for-byte");
    assert_eq!(
        compared,
        CASES.len() * BANDS.len() * 2,
        "not every cell ran"
    );
}

/// 2026-09-25: Download both buffers, check that both arms wrote every head and that the reference
/// is finite, then compare bytes.
fn compare(
    dtype: &str,
    case: &Case,
    band: &Band,
    f: &Fixture,
    g: &dyn GpuBackend,
    o_ref: DevicePtr,
    o_gqa: DevicePtr,
) {
    let mut a = vec![0u8; f.out_bytes];
    let mut b = vec![0u8; f.out_bytes];
    g.copy_d2h(o_ref, &mut a).unwrap();
    g.copy_d2h(o_gqa, &mut b).unwrap();
    let tag = format!("{dtype} {} {}", case.label, band.label);

    heads_written(&a, f.num_seqs)
        .unwrap_or_else(|e| panic!("{tag}: the UNPACKED arm did not run: {e}"));
    heads_written(&b, f.num_seqs).unwrap_or_else(|e| {
        panic!(
            "{tag}: the PACKED arm did not write every head: {e}. Under \
             grid=(nkv={NKV}, num_seqs={}) only a kernel writing PD_GQA heads per \
             CTA can fill this buffer — an unpacked kernel launched here would \
             leave heads {NKV}..{NQ} untouched.",
            f.num_seqs,
        )
    });
    all_finite(&a).unwrap_or_else(|e| panic!("{tag}: {e}"));

    if let Some((i, (x, y))) = a
        .iter()
        .zip(b.iter())
        .enumerate()
        .find(|(_, (x, y))| x != y)
    {
        panic!(
            "{tag}: output differs at byte {i} (unpacked 0x{x:02X} vs packed 0x{y:02X}) — \
             the packed kernel is NOT bit-identical",
        );
    }
    println!("{tag}: {} output bytes byte-identical", f.out_bytes);
}
