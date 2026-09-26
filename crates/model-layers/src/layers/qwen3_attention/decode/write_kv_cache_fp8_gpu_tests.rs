// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: GPU half of `write_kv_cache_fp8_tests`: the unfused three-kernel chain and the fused
//! kernel over the same inputs into separate FP8 pools, compared byte for byte.
//!
//! Owner: model-layers attention decode.
//! Invariants: none beyond the types.

use super::{fixture, shapes};

// 2026-09-25: `#[ignore]`: needs a GPU and a built PTX set. Run with:
// ```text
// cargo test -p metrale-model-layers --release gpu_parity_fused_fp8_kv -- --ignored --nocapture
// ```

#[test]
#[ignore]
fn gpu_parity_fused_fp8_kv_write() {
    use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

    use crate::layers::ops;

    const NEEDED: [&str; 4] = [
        "norm",
        "rope",
        "reshape_and_cache",
        "reshape_and_cache_fused_k_fp8",
    ];
    let set = metrale_kernels::all_ptx_sets()
        .into_iter()
        .find(|t| {
            NEEDED
                .iter()
                .all(|m| t.modules.iter().any(|(name, _)| name == m))
        })
        .expect(
            "no built PTX target carries norm + rope + reshape_and_cache + \
             reshape_and_cache_fused_k_fp8; build the GB10 kernels first",
        );
    println!(
        "target {}/{} arch {}",
        set.target.model, set.target.quant, set.ptx_arch
    );
    let gpu = metrale_gpu_runtime::cuda_backend::MetraleCudaBackend::new(0, &set.modules)
        .expect("CUDA backend");
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();

    let k_rms = g.kernel("norm", "rms_norm").unwrap();
    let k_rope = g.kernel("rope", "rope_forward").unwrap();
    let k_write = g
        .kernel("reshape_and_cache", "reshape_and_cache_flash_fp8")
        .unwrap();
    let k_fused = g
        .kernel(
            "reshape_and_cache_fused_k_fp8",
            "fused_k_norm_rope_cache_write_fp8_kv",
        )
        .unwrap();
    for (n, h) in [
        ("rms_norm", k_rms),
        ("rope_forward", k_rope),
        ("reshape_and_cache_flash_fp8", k_write),
        ("fused_k_norm_rope_cache_write_fp8_kv", k_fused),
    ] {
        assert!(h.0 != 0, "{n} resolved to handle 0");
    }

    let upload = |b: &[u8]| -> DevicePtr {
        let p = g.alloc(b.len().max(256)).unwrap();
        g.copy_h2d_async(b, p, stream).unwrap();
        p
    };
    let le16 = |v: &[u16]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };

    for (si, sh) in shapes().iter().enumerate() {
        for mag in [0.5f32, 8.0, 448.0] {
            let (k_host, v_host, w_host) = fixture(sh, 0x5EED_0000_0000_0001 ^ si as u64, mag);
            let (nkv, hd) = (sh.num_kv_heads as u32, sh.head_dim as u32);
            let n_elems = sh.num_kv_heads * sh.head_dim;
            let num_blocks = sh.slot / sh.block_size + 2;
            let pool_bytes = num_blocks * sh.cache_stride;
            // 2026-09-25: One Q head: the RoPE launch needs a Q buffer, and its content does not
            // affect K.
            let nq: u32 = 1;

            let k_raw = upload(&le16(&k_host));
            let v_raw = upload(&le16(&v_host));
            let w = upload(&le16(&w_host));
            let q = upload(&vec![0u8; hd as usize * 2]);
            let pos = upload(&sh.pos.to_le_bytes());
            let slot = upload(&(sh.slot as i64).to_le_bytes());
            // 2026-09-25: Working copy: the unfused chain mutates K in place.
            let k_work = upload(&le16(&k_host));

            let pools: Vec<DevicePtr> = (0..4)
                .map(|_| {
                    let p = g.alloc(pool_bytes).unwrap();
                    // 2026-09-25: Every pool starts at 0xA5 and the whole pool is compared, so a
                    // write outside the target slot is caught too.
                    g.memset_async(p, 0xA5, pool_bytes, stream).unwrap();
                    p
                })
                .collect();
            let (ka, va, kb, vb) = (pools[0], pools[1], pools[2], pools[3]);

            // 2026-09-25: Unfused: rms_norm -> rope_forward -> reshape_and_cache_flash_fp8.
            ops::rms_norm(
                g,
                k_rms,
                k_work,
                &crate::weight_map::DenseWeight { weight: w },
                k_work,
                nkv,
                hd,
                sh.eps,
                stream,
            )
            .unwrap();
            ops::rope(
                g,
                k_rope,
                q,
                k_work,
                pos,
                1,
                nq,
                nkv,
                hd,
                sh.rotary_dim as u32,
                sh.theta,
                stream,
            )
            .unwrap();
            ops::reshape_and_cache_fp8(
                g,
                k_write,
                k_work,
                v_raw,
                ka,
                va,
                slot,
                1,
                nkv,
                hd,
                sh.block_size as u32,
                sh.k_scale,
                sh.v_scale,
                n_elems as u32,
                n_elems as u32,
                sh.cache_stride as u64,
                stream,
            )
            .unwrap();

            // 2026-09-25: Fused, from the untouched raw K.
            ops::fused_k_norm_rope_cache_write_fp8_kv(
                g,
                k_fused,
                k_raw,
                v_raw,
                w,
                pos,
                kb,
                vb,
                slot,
                1,
                nkv,
                hd,
                sh.rotary_dim as u32,
                sh.block_size as u32,
                sh.k_scale,
                sh.v_scale,
                n_elems as u32,
                n_elems as u32,
                sh.cache_stride as u64,
                sh.eps,
                sh.theta,
                stream,
            )
            .unwrap();

            let dl = |p: DevicePtr| {
                let mut b = vec![0u8; pool_bytes];
                g.copy_d2h(p, &mut b).unwrap();
                b
            };
            for (label, a, b) in [("K", dl(ka), dl(kb)), ("V", dl(va), dl(vb))] {
                let diff = a
                    .iter()
                    .zip(b.iter())
                    .enumerate()
                    .find(|(_, (x, y))| x != y);
                assert!(
                    diff.is_none(),
                    "shape {si} mag {mag}: {label} pool differs at byte {:?} \
                     (unfused {:?} vs fused {:?}) — the fused kernel is NOT \
                     bit-identical; every committed FP8-KV record is at risk",
                    diff.map(|(i, _)| i),
                    diff.map(|(_, (x, _))| *x),
                    diff.map(|(_, (_, y))| *y),
                );
            }
            println!("shape {si} mag {mag}: K and V pools byte-identical");
        }
    }
}
