// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Bench of the tile-geometry variants of the GLM routed-MoE grouped W4A16 GEMM
//! (`moe_w4a16_grouped_gemm_ptrtable`), on the GLM-5.3 prefill shape, without a serve.
//!
//! Owner: model-arch examples (GLM-5.3 routed MoE).
//! Invariants: none beyond the types. A variant whose output differs from the base kernel's
//! is reported, not fatal; the run fails only when the base kernel does not resolve.
//!
//! Shape: 288 experts, `top_k = 8`, `hidden = 4096`, `moe_intermediate = 2048`
//! (`kernels/gb10/glm-5.3-flash/MODEL.toml`). Half the experts are local and the rest carry a
//! NULL weight pointer, as the GLM loader's pointer table gives remote experts. Gate/up are
//! `N=2048, K=4096` and down is `N=4096, K=2048`.
//!
//! Each variant's output is compared byte for byte with the base kernel's, and the first
//! difference is printed. A variant stages `lut * (e4m3 * scale2)` where the base kernel stages
//! `(lut * e4m3) * scale2`, so equality is not guaranteed.
//!
//! Bandwidth counts only experts that are both local and non-empty: a CTA whose expert has a
//! NULL weight pointer, or no rows, returns before it reads a weight byte.
//!
//!   cargo run -p metrale-model-arch --release --example glm5next_moe_grouped_tile_bench \
//!       --features cuda,gpu-examples -- `[rows]`

use anyhow::{Result, bail};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

mod device;
mod variants;

use device::{dn_i32, dn_raw, launch, lcg, up, up_f32, up_i32, up_u64};
use variants::VARIANTS;

const NUM_EXPERTS: usize = 288;
const TOP_K: usize = 8;
const HIDDEN: usize = 4096;
const MOE_INTER: usize = 2048;
/// 2026-09-25: Expert `e` is local when `e % EP == 0`; every other expert gets a NULL pointer.
const EP: usize = 2;
const GROUP_SIZE: usize = 16;

const WARMUP: usize = 2;
const ITERS: usize = 10;

fn main() -> Result<()> {
    let rows: usize = std::env::args()
        .nth(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(256);

    // 2026-09-25: `ptx_modules()` is the first target in (model, quant) order, which in a build
    // of every gb10 model is deepseek-v4-flash. That directory has its own
    // `moe_w4a16_grouped_gemm.cu` without the tile variants, so the bench loads the
    // (glm-5.3-flash, nvfp4) set, which compiles the common file.
    let set = metrale_kernels::all_ptx_sets()
        .into_iter()
        .find(|s| s.target.model == "glm-5.3-flash" && s.target.quant == "nvfp4")
        .ok_or_else(|| anyhow::anyhow!("no (glm-5.3-flash, nvfp4) PTX set in this build"))?;
    println!(
        "PTX set: ({}, {}, {})  {} modules",
        set.target.arch,
        set.target.model,
        set.target.quant,
        set.modules.len()
    );
    let g = MetraleCudaBackend::new(0, &set.modules)?;
    let gpu: &dyn GpuBackend = &g;
    let k_sort: KernelHandle = gpu.kernel("moe", "moe_sort_by_expert")?;

    let mut s = 0xA713_5EED_u64;
    let te = rows * TOP_K;

    let mut ids: Vec<u32> = Vec::with_capacity(te);
    for _ in 0..rows {
        let mut picked: Vec<u32> = Vec::with_capacity(TOP_K);
        while picked.len() < TOP_K {
            let e = (lcg(&mut s) as usize % NUM_EXPERTS) as u32;
            if !picked.contains(&e) {
                picked.push(e);
            }
        }
        ids.extend(picked);
    }

    // 2026-09-25: One weight buffer per local expert, sized for the larger of the two
    // projection shapes, serves both.
    let max_bytes_packed = MOE_INTER.max(HIDDEN) * HIDDEN.max(MOE_INTER) / 2;
    let max_bytes_scale = MOE_INTER.max(HIDDEN) * (HIDDEN.max(MOE_INTER) / GROUP_SIZE);
    let mut packed_ptrs = vec![0u64; NUM_EXPERTS];
    let mut scale_ptrs = vec![0u64; NUM_EXPERTS];
    let mut scale2 = vec![0f32; NUM_EXPERTS];
    let mut n_local = 0usize;
    // 2026-09-25: The local experts' buffers are slices of one arena;
    // `GLM_TILE_BENCH_SEPARATE_ALLOC=1` gives each its own allocation instead.
    let separate = std::env::var("GLM_TILE_BENCH_SEPARATE_ALLOC").as_deref() == Ok("1");
    let n_local_total = NUM_EXPERTS.div_ceil(EP);
    let arena_p = if separate {
        DevicePtr(0)
    } else {
        gpu.alloc(n_local_total * max_bytes_packed)?
    };
    let arena_s = if separate {
        DevicePtr(0)
    } else {
        gpu.alloc(n_local_total * max_bytes_scale)?
    };
    {
        let mut pbuf = vec![0u8; max_bytes_packed];
        let mut sbuf = vec![0u8; max_bytes_scale];
        for b in pbuf.iter_mut() {
            *b = lcg(&mut s) as u8;
        }
        for b in sbuf.iter_mut() {
            // 2026-09-25: E4M3 codes 0x30..=0x47, which decode to 0.5..=3.75: no zero, no NaN.
            *b = 0x30 + (lcg(&mut s) % 0x18) as u8;
        }
        for e in 0..NUM_EXPERTS {
            if e % EP != 0 {
                continue;
            }
            // 2026-09-25: Flip one byte per expert so no two experts hold identical weights.
            pbuf[e] ^= 0x5A;
            sbuf[e % max_bytes_scale] = 0x30 + (e % 0x18) as u8;
            if separate {
                packed_ptrs[e] = up(gpu, &pbuf)?.0;
                scale_ptrs[e] = up(gpu, &sbuf)?.0;
            } else {
                let pp = arena_p.offset(n_local * max_bytes_packed);
                let sp = arena_s.offset(n_local * max_bytes_scale);
                gpu.copy_h2d(&pbuf, pp)?;
                gpu.copy_h2d(&sbuf, sp)?;
                packed_ptrs[e] = pp.0;
                scale_ptrs[e] = sp.0;
            }
            n_local += 1;
            scale2[e] = 0.5 + (e % 64) as f32 / 64.0;
        }
    }
    println!(
        "weights: {} local experts, {} arena",
        n_local,
        if separate {
            "SEPARATE allocs"
        } else {
            "ONE contiguous"
        }
    );
    let d_packed_ptrs = up_u64(gpu, &packed_ptrs)?;
    let d_scale_ptrs = up_u64(gpu, &scale_ptrs)?;
    let d_scale2 = up_f32(gpu, &scale2)?;

    let a_bytes: Vec<u8> = (0..rows.max(te) * HIDDEN.max(MOE_INTER))
        .flat_map(|_| {
            // 2026-09-25: BF16 with a biased exponent in 96..=127: magnitudes below 2, and no
            // inf or NaN reaches the mma.
            let m = (lcg(&mut s) % 0x8000) as u16;
            (0x3000u16 | (m & 0x0FFF) | ((lcg(&mut s) as u16 & 1) << 15)).to_le_bytes()
        })
        .collect();
    let d_a = up(gpu, &a_bytes)?;

    let d_ids = up_i32(gpu, &ids.iter().map(|x| *x as i32).collect::<Vec<_>>())?;
    let d_stid = gpu.alloc(te * 4)?;
    let d_seid = gpu.alloc(te * 4)?;
    let d_off = gpu.alloc((NUM_EXPERTS + 1) * 4)?;
    let d_t2p = gpu.alloc(te * 4)?;
    KernelLaunch::new(gpu, k_sort)
        .grid([1, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(d_ids)
        .arg_ptr(d_stid)
        .arg_ptr(d_seid)
        .arg_ptr(d_off)
        .arg_ptr(d_t2p)
        .arg_u32(te as u32)
        .arg_u32(NUM_EXPERTS as u32)
        .arg_u32(TOP_K as u32)
        .launch(0)?;
    gpu.synchronize(0)?;
    let off = dn_i32(gpu, d_off, NUM_EXPERTS + 1)?;

    let counts: Vec<i32> = (0..NUM_EXPERTS).map(|e| off[e + 1] - off[e]).collect();
    let busiest = counts.iter().copied().max().unwrap_or(0) as u32;
    let swept: usize = (0..NUM_EXPERTS)
        .filter(|e| e % EP == 0 && counts[*e] > 0)
        .count();
    let mean = te as f64 / NUM_EXPERTS as f64;

    println!(
        "rows={rows} top_k={TOP_K} slots={te} experts={NUM_EXPERTS} local={n_local} \
         swept(local & non-empty)={swept}  mean rows/expert {mean:.2}  busiest {busiest}"
    );

    for (label, n_out, kk, gather) in [
        ("gate/up  N=2048 K=4096", MOE_INTER, HIDDEN, true),
        ("down     N=4096 K=2048", HIDDEN, MOE_INTER, false),
    ] {
        // 2026-09-25: Weight bytes of one sweep over the swept experts: NVFP4 is 0.5 B packed
        // plus 1/16 B of E4M3 block scale per element.
        let bytes = swept as f64 * (n_out * kk) as f64 * (0.5 + 1.0 / 16.0);
        let stid = if gather { d_stid } else { DevicePtr(0) };
        let c_bytes = te * n_out * 2;
        let d_c = gpu.alloc(c_bytes)?;

        println!("\n  {label}   one sweep = {:.1} MB", bytes / 1.0e6);
        println!(
            "  {:<18} {:>8} {:>10} {:>9} {:>8}  identity",
            "variant", "ms", "GB/s", "% of 273", "vs base"
        );

        // 2026-09-25: Control: `moe_w4a16_grouped_stream_probe` reads the same weight bytes with
        // no dequant and no mma. It runs before the variants, and is skipped when it does not
        // resolve.
        if let Ok(kp) = gpu.kernel("moe_w4a16", "moe_w4a16_grouped_stream_probe") {
            let pb = (n_out * kk / 2) as u32;
            let sb = (n_out * kk / GROUP_SIZE) as u32;
            let d_sink = gpu.alloc(NUM_EXPERTS * 4)?;
            let gx = (pb / 16).div_ceil(256 * 8).max(1);
            let go = |_: usize| -> Result<()> {
                KernelLaunch::new(gpu, kp)
                    .grid([gx, 1, NUM_EXPERTS as u32])
                    .block([256, 1, 1])
                    .arg_ptr(d_packed_ptrs)
                    .arg_ptr(d_scale_ptrs)
                    .arg_ptr(d_sink)
                    .arg_u32(NUM_EXPERTS as u32)
                    .arg_u32(pb)
                    .arg_u32(sb)
                    .launch(0)
            };
            for i in 0..WARMUP {
                go(i)?;
            }
            gpu.synchronize(0)?;
            let t0 = std::time::Instant::now();
            for i in 0..ITERS {
                go(i)?;
            }
            gpu.synchronize(0)?;
            let ms = t0.elapsed().as_secs_f64() * 1.0e3 / ITERS as f64;
            let gbs = bytes / (ms * 1.0e6);
            println!(
                "  {:<18} {ms:>8.3} {gbs:>10.1} {:>8.1}%          MEASURED CEILING (no dequant, no mma)",
                "STREAM control",
                100.0 * gbs / 273.0
            );
        }

        let mut base_out: Option<Vec<u8>> = None;
        let mut base_ms = 0f64;
        for v in VARIANTS {
            let k = match gpu.kernel("moe_w4a16", v.kernel) {
                Ok(k) => k,
                Err(e) => {
                    println!("  {:<18} UNRESOLVED: {e}", v.name);
                    continue;
                }
            };
            let max_m_tiles = busiest.div_ceil(v.m_tile).max(1);

            gpu.memset_async(d_c, 0, c_bytes, 0)?;
            for _ in 0..WARMUP {
                launch(
                    gpu,
                    k,
                    d_a,
                    d_packed_ptrs,
                    d_scale_ptrs,
                    d_scale2,
                    d_c,
                    d_off,
                    stid,
                    n_out,
                    kk,
                    max_m_tiles,
                    v.n_tile,
                    v.threads,
                )?;
            }
            gpu.synchronize(0)?;

            let t0 = std::time::Instant::now();
            for _ in 0..ITERS {
                launch(
                    gpu,
                    k,
                    d_a,
                    d_packed_ptrs,
                    d_scale_ptrs,
                    d_scale2,
                    d_c,
                    d_off,
                    stid,
                    n_out,
                    kk,
                    max_m_tiles,
                    v.n_tile,
                    v.threads,
                )?;
            }
            gpu.synchronize(0)?;
            let ms = t0.elapsed().as_secs_f64() * 1.0e3 / ITERS as f64;
            let gbs = bytes / (ms * 1.0e6);

            let out = dn_raw(gpu, d_c, c_bytes)?;
            let identity = match &base_out {
                None => {
                    base_out = Some(out);
                    base_ms = ms;
                    "—".to_string()
                }
                Some(b) => match b.iter().zip(&out).position(|(x, y)| x != y) {
                    None => "BYTE-IDENTICAL".to_string(),
                    Some(i) => format!(
                        "🔴 DIFFERS at elem {} (row {}, n {}): base {:02x?} got {:02x?}",
                        i / 2,
                        i / 2 / n_out,
                        (i / 2) % n_out,
                        &b[i & !1..(i & !1) + 2],
                        &out[i & !1..(i & !1) + 2]
                    ),
                },
            };
            println!(
                "  {:<18} {ms:>8.3} {gbs:>10.1} {:>8.1}% {:>7.2}x  {identity}",
                v.name,
                100.0 * gbs / 273.0,
                if base_ms > 0.0 { base_ms / ms } else { 1.0 }
            );
        }
        if base_out.is_none() {
            bail!("the base kernel did not resolve — nothing to compare against");
        }
    }

    Ok(())
}
