// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Load step: the per-head views of wkv_b and wq_b: W_UK_T (transposed on the host), W_UV and wq_b_rope (device-to-device copies).
//!
//! Owner: model-arch (Mistral loader).
//! Invariants:
//! - The context is written only after every copy and the final stream synchronise succeeded.

use anyhow::Result;

use super::super::gpu_alloc_or_managed;
use super::ctx::MistralLayerCtx;
use metrale_model_layers::weight_map::DenseWeight;

pub(crate) fn build_per_head_views(ctx: &mut MistralLayerCtx<'_>) -> Result<()> {
    let t_phase = std::time::Instant::now();
    let n_kv = ctx.n_kv;
    let kv_lora = ctx.kv_lora;
    let nope = ctx.nope;
    let rope = ctx.rope;
    let v_dim = ctx.v_dim;
    let hd = ctx.hd;
    let q_lora = ctx.q_lora;
    let bf16 = ctx.bf16;
    let gpu = ctx.gpu;
    let stream = ctx.stream;
    let stride = nope + v_dim;
    let wkv_b = ctx.wkv_b.as_ref().expect("phase A must precede");
    let wq_b = ctx.wq_b.as_ref().expect("phase A must precede");

    // 2026-09-25: wkv_b is `[n_kv * (nope + v_dim), kv_lora]`: per head,
    // `nope` K rows then `v_dim` V rows. The K block `[nope, kv_lora]` is
    // transposed on the host to W_UK_T `[kv_lora, nope]` per head.
    let wkv_b_total_rows = n_kv * stride;
    let wkv_b_bytes = wkv_b_total_rows * kv_lora * bf16;
    let mut wkv_b_host = vec![0u8; wkv_b_bytes];
    let t = std::time::Instant::now();
    gpu.copy_d2h(wkv_b.weight, &mut wkv_b_host)?;
    let t_d2h = t.elapsed();

    let w_uk_per_head = kv_lora * nope * bf16;
    let mut w_uk_host = vec![0u8; n_kv * w_uk_per_head];
    let t = std::time::Instant::now();
    for head in 0..n_kv {
        for p in 0..nope {
            for lkv in 0..kv_lora {
                let src_off = ((head * stride + p) * kv_lora + lkv) * bf16;
                let dst_off = (head * kv_lora * nope + lkv * nope + p) * bf16;
                w_uk_host[dst_off..dst_off + bf16]
                    .copy_from_slice(&wkv_b_host[src_off..src_off + bf16]);
            }
        }
    }
    let t_transpose = t.elapsed();
    let t = std::time::Instant::now();
    let w_uk_t_ptr = gpu_alloc_or_managed(gpu, n_kv * w_uk_per_head)?;
    gpu.copy_h2d(&w_uk_host, w_uk_t_ptr)?;
    let t_h2d = t.elapsed();

    // 2026-09-25: W_UV is stored `[n_kv, v_dim, kv_lora]`. Within a head the
    // source rows (`head*stride + nope + v`, v ascending) and the destination
    // rows are both contiguous, so each head is one async copy
    // (`per_head_runs_cover_the_same_bytes` checks the ranges).
    let t = std::time::Instant::now();
    let w_uv_ptr = gpu_alloc_or_managed(gpu, n_kv * kv_lora * v_dim * bf16)?;
    let uv_run = v_dim * kv_lora * bf16;
    for head in 0..n_kv {
        let src = wkv_b.weight.offset((head * stride + nope) * kv_lora * bf16);
        let dst = w_uv_ptr.offset(head * uv_run);
        gpu.copy_d2d_async(src, dst, uv_run, stream)?;
    }
    let t_uv_d2d = t.elapsed();

    // 2026-09-25: `wq_b_rope[n*rope + r, l] = wq_b[n*hd + nope + r, l]` for
    // `r` in `0..rope`: again one contiguous run per head.
    let t = std::time::Instant::now();
    let wqbr_ptr = gpu_alloc_or_managed(gpu, n_kv * rope * q_lora * bf16)?;
    let rope_run = rope * q_lora * bf16;
    for head in 0..n_kv {
        let src = wq_b.weight.offset((head * hd + nope) * q_lora * bf16);
        let dst = wqbr_ptr.offset(head * rope_run);
        gpu.copy_d2d_async(src, dst, rope_run, stream)?;
    }
    // 2026-09-25: One synchronise covers all `2 * n_kv` async copies.
    gpu.synchronize(stream)?;

    let t_rope_d2d = t.elapsed();

    tracing::info!(
        "MLA phase B (per-head views) L{}: total={:.1}ms | wkv_b d2h ({:.1} MB)={:.1}ms, \
         cpu-transpose ({} elems)={:.1}ms, w_uk h2d={:.1}ms, \
         w_uv d2d ({} runs)={:.1}ms, wq_b_rope d2d ({} runs)+sync={:.1}ms",
        ctx.layer_idx,
        t_phase.elapsed().as_secs_f64() * 1e3,
        wkv_b_bytes as f64 / 1e6,
        t_d2h.as_secs_f64() * 1e3,
        n_kv * nope * kv_lora,
        t_transpose.as_secs_f64() * 1e3,
        t_h2d.as_secs_f64() * 1e3,
        n_kv,
        t_uv_d2d.as_secs_f64() * 1e3,
        n_kv,
        t_rope_d2d.as_secs_f64() * 1e3,
    );

    ctx.wq_b_rope = Some(DenseWeight { weight: wqbr_ptr });
    ctx.w_uk_t = Some(DenseWeight { weight: w_uk_t_ptr });
    ctx.w_uv = Some(DenseWeight { weight: w_uv_ptr });
    ctx.w_uk_host = w_uk_host;
    Ok(())
}

#[cfg(test)]
mod tests {
    /// 2026-09-25: `build_per_head_views` copies W_UV and wq_b_rope as one run
    /// per head. Expanded row by row, each run must give the same
    /// `(source, destination, length)` triples as a per-row copy.
    #[test]
    fn per_head_runs_cover_the_same_bytes() {
        for &(n_kv, kv_lora, nope, rope, v_dim, hd, q_lora) in &[
            (
                32usize, 512usize, 192usize, 64usize, 256usize, 256usize, 1536usize,
            ),
            (8, 256, 128, 64, 128, 192, 1024),
            (3, 5, 7, 2, 11, 9, 13),
        ] {
            let bf16 = 2usize;
            let stride = nope + v_dim;

            let mut want: Vec<(usize, usize, usize)> = Vec::new();
            for head in 0..n_kv {
                for v in 0..v_dim {
                    let src = (head * stride + nope + v) * kv_lora * bf16;
                    let dst = (head * v_dim * kv_lora + v * kv_lora) * bf16;
                    want.push((src, dst, kv_lora * bf16));
                }
            }
            let uv_run = v_dim * kv_lora * bf16;
            let mut got: Vec<(usize, usize, usize)> = Vec::new();
            for head in 0..n_kv {
                let src = (head * stride + nope) * kv_lora * bf16;
                let dst = head * uv_run;
                for v in 0..v_dim {
                    got.push((
                        src + v * kv_lora * bf16,
                        dst + v * kv_lora * bf16,
                        kv_lora * bf16,
                    ));
                }
            }
            assert_eq!(
                got, want,
                "w_uv run/row mismatch (n_kv={n_kv} v_dim={v_dim})"
            );

            let mut want: Vec<(usize, usize, usize)> = Vec::new();
            for head in 0..n_kv {
                for r in 0..rope {
                    let src = (head * hd + nope + r) * q_lora * bf16;
                    let dst = (head * rope + r) * q_lora * bf16;
                    want.push((src, dst, q_lora * bf16));
                }
            }
            let rope_run = rope * q_lora * bf16;
            let mut got: Vec<(usize, usize, usize)> = Vec::new();
            for head in 0..n_kv {
                let src = (head * hd + nope) * q_lora * bf16;
                let dst = head * rope_run;
                for r in 0..rope {
                    got.push((
                        src + r * q_lora * bf16,
                        dst + r * q_lora * bf16,
                        q_lora * bf16,
                    ));
                }
            }
            assert_eq!(
                got, want,
                "wq_b_rope run/row mismatch (n_kv={n_kv} rope={rope})"
            );
        }
    }
}
