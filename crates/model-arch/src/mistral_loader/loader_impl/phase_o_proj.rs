// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Load step: the attention output projection `wo`, in BF16 and as an NVFP4 copy.
//!
//! Owner: model-arch (Mistral loader).
//! Invariants: the context is written only after both loads succeeded.

use anyhow::Result;

use super::ctx::MistralLayerCtx;
use metrale_model_layers::weight_map::{dense, quantize_to_nvfp4};

pub(super) fn load_o_proj(ctx: &mut MistralLayerCtx<'_>) -> Result<()> {
    let ap = ctx.ap();
    let h = ctx.h;
    let n_heads = ctx.n_heads;
    let hd = ctx.hd;
    let gpu = ctx.gpu;

    let o_dense_bf16 = dense(ctx.store, &format!("{ap}.wo.weight"))?;
    let o_nvfp4 = Some(quantize_to_nvfp4(
        &o_dense_bf16,
        h,
        n_heads * hd,
        gpu,
        ctx.absmax_k,
        ctx.quantize_k,
        ctx.stream,
    )?);

    ctx.o_dense_bf16 = Some(o_dense_bf16);
    ctx.o_nvfp4 = o_nvfp4;
    Ok(())
}
