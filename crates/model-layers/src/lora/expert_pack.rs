// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Router and routed-expert LoRA: checks on an adapter's audited
//! router and expert tensors, and their pack into the expert pool. That pool is
//! a separate allocation from the slot pool, sized per adapter from its
//! audited keys (`expert_router_bytes`, `loading.rs`).
//!
//! Owner: model-layers (lora).
//! Invariants:
//! - `pack_into` and `expert_router_bytes` use the same padded stride,
//!   `packed_stride(max_rank)`.

use std::collections::BTreeMap;

use anyhow::{Result, bail};
use metrale_config::{ModelConfig, PeftAdapterConfig};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_weights::weights::WeightStore;

use super::*;
use crate::layers::ops::lora_delta::LoraPair;
use crate::weight_map::DenseWeight;

/// 2026-09-25: Router audit map: global layer → `[a_key, b_key]`.
pub(crate) type RouterMap = BTreeMap<usize, [Option<String>; 2]>;
/// 2026-09-25: Expert audit map: `(global layer, expert, proj)` →
/// `[a_key, b_key]`.
pub(crate) type ExpertMap = BTreeMap<(usize, u16, ExpertProj), [Option<String>; 2]>;

/// 2026-09-25: True when the adapter has any router or expert tensor.
pub(crate) fn present(router: &RouterMap, experts: &ExpertMap) -> bool {
    !router.is_empty() || !experts.is_empty()
}

/// 2026-09-25: Store-free checks on the router and expert maps: refused when
/// `METRALE_LORA_EXPERTS` is off or `r` exceeds `max_lora_expert_rank()`.
/// Pair completeness and shapes are checked by `validate_shapes`.
pub(crate) fn validate(
    cfg: &ModelConfig,
    peft: &PeftAdapterConfig,
    router: &RouterMap,
    experts: &ExpertMap,
) -> Result<()> {
    if !present(router, experts) {
        return Ok(());
    }
    if !lora_experts_env() {
        bail!(
            "REJECT[expert-lora-disabled]: adapter targets {} router + {} expert \
             projection(s), but MoE expert/router LoRA is off. Set METRALE_LORA_EXPERTS=1 \
             to opt into the correctness-first (single-active, host-synced) expert path.",
            router.len(),
            experts.len()
        );
    }
    let cap = max_lora_expert_rank();
    if peft.r > cap {
        bail!(
            "REJECT[expert-rank-exceeds-cap]: r={} > METRALE_LORA_EXPERT_RANK={} \
             (the expert pool grows ~num_experts×num_layers faster than attention; \
             raise the cap only with the VRAM headroom for it)",
            peft.r,
            cap
        );
    }
    for (layer, pair) in router {
        let (out, inp) = router_dims(cfg);
        audit_pair_shape(peft, pair, "router", *layer, None, out, inp)?;
    }
    for ((layer, n, proj), pair) in experts {
        let (out, inp) = proj.dims(cfg, *layer);
        audit_pair_shape(peft, pair, proj.peft_name(), *layer, Some(*n), out, inp)?;
    }
    Ok(())
}

fn audit_pair_shape(
    peft: &PeftAdapterConfig,
    _pair: &[Option<String>; 2],
    _module: &str,
    _layer: usize,
    _expert: Option<u16>,
    _out: usize,
    _inp: usize,
) -> Result<()> {
    // 2026-09-25: Does nothing; `validate_shapes` checks shapes against the
    // store.
    let _ = peft;
    Ok(())
}

/// 2026-09-25: Store-backed checks: each pair has both tensors, A is
/// `[r, in]` and B is `[out, r]`.
pub(crate) fn validate_shapes(
    store: &WeightStore,
    cfg: &ModelConfig,
    peft: &PeftAdapterConfig,
    router: &RouterMap,
    experts: &ExpertMap,
) -> Result<()> {
    for (layer, pair) in router {
        let (out, inp) = router_dims(cfg);
        check_shape(
            store,
            peft,
            pair,
            &format!("router(layer {layer})"),
            out,
            inp,
        )?;
    }
    for ((layer, n, proj), pair) in experts {
        let (out, inp) = proj.dims(cfg, *layer);
        check_shape(
            store,
            peft,
            pair,
            &format!("expert {n} {:?}(layer {layer})", proj),
            out,
            inp,
        )?;
    }
    Ok(())
}

fn check_shape(
    store: &WeightStore,
    peft: &PeftAdapterConfig,
    pair: &[Option<String>; 2],
    what: &str,
    out_dim: usize,
    in_dim: usize,
) -> Result<()> {
    let [Some(a_key), Some(b_key)] = pair else {
        bail!("REJECT[unpaired-tensor]: {what} has only one of lora_A/lora_B");
    };
    let a = store.get(a_key)?;
    let b = store.get(b_key)?;
    if a.shape != vec![peft.r, in_dim] {
        bail!(
            "REJECT[shape-mismatch]: '{a_key}' is {:?}, expected [{}, {}] (r, in_dim)",
            a.shape,
            peft.r,
            in_dim
        );
    }
    if b.shape != vec![out_dim, peft.r] {
        bail!(
            "REJECT[shape-mismatch]: '{b_key}' is {:?}, expected [{}, {}] (out_dim, r)",
            b.shape,
            out_dim,
            peft.r
        );
    }
    Ok(())
}

/// 2026-09-25: The expert pool's padded rank: `max_rank` rounded up to a
/// multiple of 8 BF16 elements (16 bytes). `moe_lora_gather_bgmv` reads B
/// rows, `max_rank` BF16 long, as `uint4`, so each row must start 16-byte
/// aligned; a raw r = 12 gives 24-byte rows. It is B's row
/// stride, A's padded row count and `LoraPair.max_rank`; `LoraPair.rank`
/// stays `peft.r`. Used by `expert_router_bytes` and `pack_into`.
pub(crate) fn packed_stride(max_rank: usize) -> usize {
    max_rank.div_ceil(8) * 8
}

/// 2026-09-25: The (layer, proj) and router-layer lists
/// [`expert_router_bytes`] sizes one adapter's expert pool from.
pub(crate) fn key_lists(
    router: &RouterMap,
    experts: &ExpertMap,
) -> (Vec<(usize, ExpertProj)>, Vec<usize>) {
    let ek = experts.keys().map(|(l, _, p)| (*l, *p)).collect();
    let rl = router.keys().copied().collect();
    (ek, rl)
}

/// 2026-09-25: Copy one A/B pair from the store to `(a_ptr, b_ptr)` and build
/// its [`LoraPair`]: A (`[r, in]`) is copied as is, B's rows are re-strided
/// from r to `stride`. The padding is not written, so the caller must pass
/// zeroed memory.
#[allow(clippy::too_many_arguments)]
fn pack_pair(
    store: &WeightStore,
    a_key: &str,
    b_key: &str,
    peft: &PeftAdapterConfig,
    out_dim: usize,
    in_dim: usize,
    stride: usize,
    gpu: &dyn GpuBackend,
    a_ptr: DevicePtr,
    b_ptr: DevicePtr,
) -> Result<LoraPair> {
    const BF16_BYTES: usize = 2;
    let a_t = store.get(a_key)?;
    let mut a_host = vec![0u8; peft.r * in_dim * BF16_BYTES];
    gpu.copy_d2h(a_t.ptr, &mut a_host)?;
    gpu.copy_h2d(&a_host, a_ptr)?;

    let b_t = store.get(b_key)?;
    let mut b_src = vec![0u8; out_dim * peft.r * BF16_BYTES];
    gpu.copy_d2h(b_t.ptr, &mut b_src)?;
    let mut b_host = vec![0u8; out_dim * stride * BF16_BYTES];
    for row in 0..out_dim {
        let d = row * stride * BF16_BYTES;
        let s = row * peft.r * BF16_BYTES;
        b_host[d..d + peft.r * BF16_BYTES].copy_from_slice(&b_src[s..s + peft.r * BF16_BYTES]);
    }
    gpu.copy_h2d(&b_host, b_ptr)?;

    Ok(LoraPair {
        a: DenseWeight { weight: a_ptr },
        b: DenseWeight { weight: b_ptr },
        rank: peft.r as u32,
        k_in: in_dim as u32,
        n_out: out_dim as u32,
        scale: peft.scaling(),
        max_rank: stride as u32,
    })
}

/// 2026-09-25: Pack the adapter's router and expert pairs into the expert pool
/// from byte offset `*off`, advancing it, and set `layers[l].router` and
/// `layers[l].experts`, creating the layer entry when it is `None`. A pair
/// missing a tensor is skipped. Returns the number of pairs packed.
#[allow(clippy::too_many_arguments)]
pub(crate) fn pack_into(
    layers: &mut [Option<LoraLayerWeights>],
    store: &WeightStore,
    peft: &PeftAdapterConfig,
    router: &RouterMap,
    experts: &ExpertMap,
    cfg: &ModelConfig,
    gpu: &dyn GpuBackend,
    pool: DevicePtr,
    max_rank: usize,
    off: &mut usize,
) -> Result<usize> {
    const BF16_BYTES: usize = 2;
    let stride = packed_stride(max_rank);
    let mut packed = 0usize;
    let ensure = |layers: &mut [Option<LoraLayerWeights>], l: usize| {
        if layers[l].is_none() {
            layers[l] = Some(LoraLayerWeights::empty(l));
        }
    };

    for (layer, pair) in router {
        let [Some(a_key), Some(b_key)] = pair else {
            continue;
        };
        let (out_dim, in_dim) = router_dims(cfg);
        let a_ptr = DevicePtr(pool.0 + *off as u64);
        let b_ptr = DevicePtr(pool.0 + (*off + stride * in_dim * BF16_BYTES) as u64);
        *off += (stride * in_dim + out_dim * stride) * BF16_BYTES;
        let lp = pack_pair(
            store, a_key, b_key, peft, out_dim, in_dim, stride, gpu, a_ptr, b_ptr,
        )?;
        ensure(layers, *layer);
        layers[*layer].as_mut().unwrap().router = Some(lp);
        packed += 1;
    }

    for ((layer, n, proj), pair) in experts {
        let [Some(a_key), Some(b_key)] = pair else {
            continue;
        };
        let (out_dim, in_dim) = proj.dims(cfg, *layer);
        let a_ptr = DevicePtr(pool.0 + *off as u64);
        let b_ptr = DevicePtr(pool.0 + (*off + stride * in_dim * BF16_BYTES) as u64);
        *off += (stride * in_dim + out_dim * stride) * BF16_BYTES;
        let lp = pack_pair(
            store, a_key, b_key, peft, out_dim, in_dim, stride, gpu, a_ptr, b_ptr,
        )?;
        ensure(layers, *layer);
        let el = layers[*layer]
            .as_mut()
            .unwrap()
            .experts
            .get_or_insert_with(ExpertLoraLayer::default);
        el.pairs.insert((*n, *proj), lp);
        packed += 1;
    }
    Ok(packed)
}

#[cfg(test)]
#[path = "expert_pack_tests.rs"]
mod tests;
