// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Load step: the MLA LoRA projections wq_a, wq_b, wkv_a, wkv_b and the norms q_a_norm, kv_a_norm, with NVFP4 copies and the TP shard.
//!
//! Owner: model-arch (Mistral loader).
//! Invariants: the context is written only after every load succeeded.

use anyhow::Result;

use super::ctx::MistralLayerCtx;
use crate::tp_shard::{TpShardKind, shard_dense_bf16};
use metrale_model_layers::weight_map::{DenseWeight, dense, quantize_to_nvfp4};

pub(super) fn load_lora_qkv(ctx: &mut MistralLayerCtx<'_>) -> Result<()> {
    let ap = ctx.ap();
    let h = ctx.h;
    let q_lora = ctx.q_lora;
    let kv_lora = ctx.kv_lora;
    let nope = ctx.nope;
    let rope = ctx.rope;
    let v_dim = ctx.v_dim;
    let n_heads = ctx.n_heads;
    let n_kv = ctx.n_kv;
    let hd = ctx.hd;
    let bf16 = ctx.bf16;
    let gpu = ctx.gpu;

    // 2026-09-25: wq_a, wq_b and wkv_a are always quantised to NVFP4 here;
    // `phase_assemble` drops the NVFP4 copies when METRALE_NVFP4_MLA is 0,
    // false, no or off.
    let wq_a_dense = dense(ctx.store, &format!("{ap}.wq_a.weight"))?;
    let wq_a_nvfp4 = Some(quantize_to_nvfp4(
        &wq_a_dense,
        q_lora,
        h,
        gpu,
        ctx.absmax_k,
        ctx.quantize_k,
        ctx.stream,
    )?);
    let mut wq_b = dense(ctx.store, &format!("{ap}.wq_b.weight"))?;

    // 2026-09-25: wq_b is column-parallel on the head axis. `n_heads` is
    // already this rank's count, so the unsharded row count is
    // `n_heads * tp_size * hd`.
    let tp_rank = ctx.config.tp_rank;
    let tp_size = ctx.config.tp_world_size.max(1);
    if tp_size > 1 {
        let full_rows = n_heads * tp_size * hd;
        let (sharded, _, _) = shard_dense_bf16(
            wq_b.weight,
            full_rows,
            q_lora,
            TpShardKind::ColumnParallel,
            tp_rank,
            tp_size,
            gpu,
        )?;
        if sharded != wq_b.weight {
            gpu.free(wq_b.weight)?;
        }
        wq_b.weight = sharded;
    }
    let wq_b_nvfp4 = Some(quantize_to_nvfp4(
        &wq_b,
        n_heads * hd,
        q_lora,
        gpu,
        ctx.absmax_k,
        ctx.quantize_k,
        ctx.stream,
    )?);
    let q_a_norm = dense(ctx.store, &format!("{ap}.q_a_norm.weight"))?;

    // 2026-09-25: wkv_a is `[kv_lora + rope, h]`: the first `kv_lora` rows
    // project to the latent, the last `rope` rows to K_rope.
    let wkv_a_dense = dense(ctx.store, &format!("{ap}.wkv_a_with_mqa.weight"))?;
    let wkv_a_nvfp4 = Some(quantize_to_nvfp4(
        &wkv_a_dense,
        kv_lora + rope,
        h,
        gpu,
        ctx.absmax_k,
        ctx.quantize_k,
        ctx.stream,
    )?);
    let wkv_a_rope_dense = DenseWeight {
        weight: wkv_a_dense.weight.offset(kv_lora * h * bf16),
    };
    let mut wkv_b = dense(ctx.store, &format!("{ap}.wkv_b.weight"))?;
    if tp_size > 1 {
        let full_rows = n_kv * tp_size * (nope + v_dim);
        let (sharded, _, _) = shard_dense_bf16(
            wkv_b.weight,
            full_rows,
            kv_lora,
            TpShardKind::ColumnParallel,
            tp_rank,
            tp_size,
            gpu,
        )?;
        if sharded != wkv_b.weight {
            gpu.free(wkv_b.weight)?;
        }
        wkv_b.weight = sharded;
    }
    let kv_a_norm = dense(ctx.store, &format!("{ap}.kv_a_norm.weight"))?;

    ctx.wq_a_dense = Some(wq_a_dense);
    ctx.wq_a_nvfp4 = wq_a_nvfp4;
    ctx.wq_b = Some(wq_b);
    ctx.wq_b_nvfp4 = wq_b_nvfp4;
    ctx.q_a_norm = Some(q_a_norm);
    ctx.wkv_a_dense = Some(wkv_a_dense);
    ctx.wkv_a_nvfp4 = wkv_a_nvfp4;
    ctx.wkv_a_rope_dense = Some(wkv_a_rope_dense);
    ctx.wkv_b = Some(wkv_b);
    ctx.kv_a_norm = Some(kv_a_norm);
    Ok(())
}
