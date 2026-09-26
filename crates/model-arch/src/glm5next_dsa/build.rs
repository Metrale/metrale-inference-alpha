// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Loading one DSA block for this rank: tensor-parallel sharding plus four
//! load-time weight transforms.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants: none beyond the types.
//!
//! [`build_dsa_weights`] takes a `load` closure that returns a checkpoint tensor by
//! layer-relative name, so the transforms are testable without a `WeightStore`.
//!
//! # The transforms, each done once at load
//!
//! 1. **`q_absorb`** ([`absorb_q`]): `q_b_proj` multiplied through `kv_b_proj`'s K half, so
//!    Q is in the `kv_lora_rank`-wide latent space the decode kernel reads.
//! 2. **`weights_proj` scaled by `index_heads^-0.5`**: `dsa_index_scores` does not apply the
//!    factor, and a positive scale folds into the weight exactly.
//! 3. **`o_absorb`** ([`absorb_o`]): `o_proj` multiplied through `kv_b_proj`'s V half. The
//!    decode kernel's output is `[local_heads, kv_lora_rank]` in latent space, while the
//!    checkpoint `o_proj` takes `[local_heads, v_head_dim]` (256 per head against 512 on
//!    GLM-5.3).
//! 4. **`ape` uploaded as f32**: the `dsa_kpool_compress` parameter is `const float*` and
//!    the checkpoint stores BF16.

use anyhow::{Result, bail};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::Glm5NextDsaConfig;
use super::layer::Glm5NextDsaWeights;
use super::tp::{DsaShard, DsaTpPlan};

/// 2026-09-25: Returns a checkpoint tensor by layer-relative name: the full (unsharded)
/// BF16 values, widened to f32 on the host.
pub type LoadFn<'a> = &'a dyn Fn(&str) -> Result<Vec<f32>>;

/// 2026-09-25: Worker threads for [`absorb_q`]: `METRALE_MLA_ABSORB_THREADS` when it parses
/// to a positive integer, else the available parallelism, clamped to `1..=rows`.
/// `mistral_loader::loader_impl::phase_qk_absorbed` reads the variable the same way.
fn absorb_threads(rows: usize) -> usize {
    let want = std::env::var("METRALE_MLA_ABSORB_THREADS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        });
    want.clamp(1, rows.max(1))
}

/// 2026-09-25: A disjoint span of `absorb_q` output rows, row `row0` onward.
///
/// Each output element sums over `r` in ascending order inside one thread, so the result
/// does not depend on the thread count (`absorb_q_bit_exact` in the tests).
#[allow(clippy::too_many_arguments)]
fn absorb_q_rows(
    out: &mut [f32],
    row0: usize,
    kv_b: &[f32],
    q_b: &[f32],
    kvl: usize,
    ql: usize,
    nope: usize,
    vd: usize,
) {
    for (i, acc) in out.chunks_exact_mut(ql).enumerate() {
        let row = row0 + i;
        let h = row / kvl;
        let c = row % kvl;
        let kv_base = h * (nope + vd);
        let qb_base = h * nope;
        acc.fill(0.0);
        for r in 0..nope {
            let w = kv_b[(kv_base + r) * kvl + c];
            let q_row = &q_b[(qb_base + r) * ql..][..ql];
            for (a, &q) in acc.iter_mut().zip(q_row.iter()) {
                *a += q * w;
            }
        }
    }
}

/// 2026-09-25: `q_absorb[h*kvl + c][k] = Σ_r kv_b[h*(nope+vd) + r][c] · q_b[h*nope + r][k]`
/// over all `full_heads` heads, computed on the host.
///
/// `q_b_proj` has `qk_head_dim` rows per head (256 on GLM-5.3) and `kv_b_proj` has
/// `qk_nope_head_dim + v_head_dim` (512); the element-count checks refuse either tensor at
/// the other's width. NoPE only: a nonzero `qk_rope_head_dim` is refused.
pub fn absorb_q(
    cfg: &Glm5NextDsaConfig,
    q_b: &[f32],
    kv_b: &[f32],
    full_heads: usize,
) -> Result<Vec<f32>> {
    let (nope, vd, kvl, ql) = (
        cfg.qk_nope_head_dim,
        cfg.v_head_dim,
        cfg.kv_lora_rank,
        cfg.q_lora_rank,
    );
    let qk = cfg.qk_head_dim();
    if q_b.len() != full_heads * qk * ql {
        bail!(
            "absorb_q: q_b_proj has {} elems, expected {}",
            q_b.len(),
            full_heads * qk * ql
        );
    }
    if kv_b.len() != full_heads * (nope + vd) * kvl {
        bail!(
            "absorb_q: kv_b_proj has {} elems, expected {}",
            kv_b.len(),
            full_heads * (nope + vd) * kvl
        );
    }
    // 2026-09-25: NoPE: qk_head_dim == qk_nope_head_dim, so the K half of kv_b lines up with
    // the whole of q_b. A rope section would need its rows carried separately.
    if cfg.qk_rope_head_dim != 0 {
        bail!(
            "absorb_q: NoPE only; qk_rope_head_dim is {}",
            cfg.qk_rope_head_dim
        );
    }

    let rows = full_heads * kvl;
    let mut out = vec![0f32; rows * ql];
    let threads = absorb_threads(rows);
    if threads <= 1 {
        absorb_q_rows(&mut out, 0, kv_b, q_b, kvl, ql, nope, vd);
    } else {
        let rows_per = rows.div_ceil(threads);
        std::thread::scope(|scope| {
            for (chunk_idx, chunk) in out.chunks_mut(rows_per * ql).enumerate() {
                let row0 = chunk_idx * rows_per;
                scope.spawn(move || {
                    absorb_q_rows(chunk, row0, kv_b, q_b, kvl, ql, nope, vd);
                });
            }
        });
    }
    Ok(out)
}

/// 2026-09-25: The output-side absorption:
/// `o_absorb[i][h*kvl + c] = Σ_r o_proj[i][h*vd + r] · kv_b[h*(nope+vd) + nope + r][c]`.
///
/// The output-side counterpart of [`absorb_q`]: it maps the decode kernel's latent output
/// back through `kv_b_proj`'s V half. It runs on this rank's heads only (`o_local`,
/// `kv_b_local`): output columns of head `h` read only head `h`'s `o_proj` columns and
/// `kv_b` rows. Each head's V half starts at row `qk_nope_head_dim` of that head's `kv_b`
/// block.
pub fn absorb_o(
    cfg: &Glm5NextDsaConfig,
    o_local: &[f32],
    kv_b_local: &[f32],
    local_heads: usize,
) -> Result<Vec<f32>> {
    let (nope, vd, kvl, hidden) = (
        cfg.qk_nope_head_dim,
        cfg.v_head_dim,
        cfg.kv_lora_rank,
        cfg.hidden,
    );
    if cfg.qk_rope_head_dim != 0 {
        bail!(
            "absorb_o: NoPE only; qk_rope_head_dim is {}",
            cfg.qk_rope_head_dim
        );
    }
    let in_w = local_heads * vd;
    let out_w = local_heads * kvl;
    if o_local.len() != hidden * in_w {
        bail!(
            "absorb_o: o_proj has {} elems, expected {} ([{hidden}, {in_w}])",
            o_local.len(),
            hidden * in_w
        );
    }
    if kv_b_local.len() != local_heads * (nope + vd) * kvl {
        bail!(
            "absorb_o: kv_b_proj slice has {} elems, expected {}",
            kv_b_local.len(),
            local_heads * (nope + vd) * kvl
        );
    }

    let mut out = vec![0f32; hidden * out_w];
    // 2026-09-25: Output rows are independent and contiguous, so they are split across
    // threads: the work is `hidden * local_heads * kvl * vd` MACs per layer.
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .clamp(1, hidden);
    let rows = hidden.div_ceil(threads);
    std::thread::scope(|sc| {
        for (o_chunk, i_chunk) in out
            .chunks_mut(rows * out_w)
            .zip(o_local.chunks(rows * in_w))
        {
            sc.spawn(move || {
                for (dst_row, src_row) in o_chunk.chunks_mut(out_w).zip(i_chunk.chunks(in_w)) {
                    for h in 0..local_heads {
                        let dst = &mut dst_row[h * kvl..(h + 1) * kvl];
                        let v_base = h * (nope + vd) + nope;
                        for r in 0..vd {
                            let w = src_row[h * vd + r];
                            let src = &kv_b_local[(v_base + r) * kvl..(v_base + r) * kvl + kvl];
                            for (d, k) in dst.iter_mut().zip(src) {
                                *d += w * k;
                            }
                        }
                    }
                }
            });
        }
    });
    Ok(out)
}

/// 2026-09-25: Rows `[start, end)` of a `[rows, row_elems]` row-major tensor.
fn row_slice(v: &[f32], row_elems: usize, start: usize, end: usize) -> Vec<f32> {
    v[start * row_elems..end * row_elems].to_vec()
}

/// 2026-09-25: Column range `[start, end)` of every row: the `DsaShard::HeadCols` case
/// (`o_proj`).
fn col_slice(v: &[f32], row_elems: usize, start: usize, end: usize) -> Vec<f32> {
    v.chunks(row_elems)
        .flat_map(|r| r[start..end].iter().copied())
        .collect()
}

/// 2026-09-25: Apply one tensor's shard plan to full host values.
pub fn shard_host(plan: &super::tp::DsaTensorPlan, full: &[f32]) -> Vec<f32> {
    match plan.kind {
        DsaShard::Replicated => full.to_vec(),
        DsaShard::HeadRows => row_slice(
            full,
            plan.full_row_elems,
            plan.src_row_offset,
            plan.src_row_offset + plan.local_rows,
        ),
        DsaShard::HeadCols => col_slice(
            full,
            plan.full_row_elems,
            plan.src_col_offset,
            plan.src_col_offset + plan.local_row_elems,
        ),
    }
}

fn up_bf16(gpu: &dyn GpuBackend, v: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = v
        .iter()
        .flat_map(|x| half::bf16::from_f32(*x).to_le_bytes())
        .collect();
    let p = gpu.alloc(b.len().max(1))?;
    gpu.copy_h2d(&b, p)?;
    Ok(p)
}
fn up_f32(gpu: &dyn GpuBackend, v: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    let p = gpu.alloc(b.len().max(1))?;
    gpu.copy_h2d(&b, p)?;
    Ok(p)
}

/// 2026-09-25: Load, transform, shard and upload one DSA block for this rank.
pub fn build_dsa_weights(
    gpu: &dyn GpuBackend,
    cfg: &Glm5NextDsaConfig,
    plan: &DsaTpPlan,
    load: LoadFn<'_>,
) -> Result<Glm5NextDsaWeights> {
    let get = |n: &str| -> Result<Vec<f32>> { load(&format!("self_attn.{n}")) };
    let shard = |n: &'static str, full: Vec<f32>| -> Result<Vec<f32>> {
        let p = plan
            .get(n)
            .ok_or_else(|| anyhow::anyhow!("no shard plan for {n}"))?;
        Ok(shard_host(p, &full))
    };

    let kv_b = get("kv_b_proj.weight")?;

    // 2026-09-25: Transform 1 over all heads, then this rank's rows: heads
    // `tp_rank * local_heads ..`, `kv_lora_rank` rows each.
    let q_absorb_full = absorb_q(cfg, &get("q_b_proj.weight")?, &kv_b, plan.full_heads)?;
    let per_head = cfg.kv_lora_rank;
    let start = plan.tp_rank * plan.local_heads * per_head;
    let len = plan.local_heads * per_head;
    let q_absorb = row_slice(&q_absorb_full, cfg.q_lora_rank, start, start + len);

    // 2026-09-25: Transform 3 on this rank's heads: its `kv_b` rows and its `o_proj`
    // column slice.
    let kv_b_rows = cfg.qk_nope_head_dim + cfg.v_head_dim;
    let kv_b_local = row_slice(
        &kv_b,
        cfg.kv_lora_rank,
        plan.tp_rank * plan.local_heads * kv_b_rows,
        (plan.tp_rank + 1) * plan.local_heads * kv_b_rows,
    );
    let o_absorb = absorb_o(
        cfg,
        &shard("o_proj", get("o_proj.weight")?)?,
        &kv_b_local,
        plan.local_heads,
    )?;

    // 2026-09-25: Transform 2: fold index_heads^-0.5 into weights_proj.
    let scale = (cfg.index_heads as f32).powf(-0.5);
    let weights_proj: Vec<f32> = get("indexer.weights_proj.weight")?
        .iter()
        .map(|x| x * scale)
        .collect();

    // 2026-09-25: Transform 4: ape stays f32 for the kernel.
    let ape = get("indexer.index_kpool_compress_ape")?;

    Ok(Glm5NextDsaWeights {
        q_a_proj: up_bf16(gpu, &get("q_a_proj.weight")?)?,
        q_a_layernorm: up_bf16(gpu, &get("q_a_layernorm.weight")?)?,
        q_absorb: up_bf16(gpu, &q_absorb)?,
        kv_a_proj: up_bf16(gpu, &get("kv_a_proj_with_mqa.weight")?)?,
        kv_a_layernorm: up_bf16(gpu, &get("kv_a_layernorm.weight")?)?,
        o_absorb: up_bf16(gpu, &o_absorb)?,
        wk: up_bf16(gpu, &get("indexer.wk.weight")?)?,
        k_norm_weight: up_bf16(gpu, &get("indexer.k_norm.weight")?)?,
        k_norm_bias: up_bf16(gpu, &get("indexer.k_norm.bias")?)?,
        compress_gate: up_bf16(gpu, &get("indexer.index_kpool_compress_gate")?)?,
        wq_b: up_bf16(gpu, &get("indexer.wq_b.weight")?)?,
        weights_proj: up_bf16(gpu, &weights_proj)?,
        ape: up_f32(gpu, &ape)?,
    })
}

#[cfg(test)]
mod tests;
