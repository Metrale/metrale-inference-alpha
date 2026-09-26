# metrale-model-engine

**Path:** `crates/model-engine/`
**Role:** the model assembly crate. It turns a loaded checkpoint into a `TransformerModel`, defines the `Model` trait the scheduler drives, and holds the model factory and the single-request generate loop.
**Key files:** `traits/model.rs` (the `Model` trait), `model/` (`TransformerModel`), `factory.rs` (`loader_for_config`), `engine.rs` (the generate loop), `kimi_k3_loader/`, `rank_agree.rs` in `crates/model-engine/src/`.

The pieces a model is built from live in the crates below it: the layer types and the `TransformerLayer` trait in [metrale-model-layers](./metrale-model-layers.md) (`crates/model-layers/src/layer/`, `layers/`, `weight_map/`, `quant_format/`, `vision_preprocess.rs`); the `ModelWeightLoader` trait, the per-family loaders and the family-specific layers in [metrale-model-arch](./metrale-model-arch.md) (`crates/model-arch/src/weight_loader/`, `nemotron_mamba2.rs`, `mistral_loader/`, `glm5next_*`, `kimi_k3/`, …); the weight store, the fast loader and preflight in [metrale-model-weights](./metrale-model-weights.md).

## The central traits

### `Model`

`Model` is the whole scheduler-facing interface. It is a set of supertraits, each covering one concern, and an implementor writes one `impl` per supertrait and an empty `impl Model`:

```rust
pub trait Model:
    Send + Sync
    + ModelLifecycle + ModelForward + ModelLogits + ModelAdapters
    + ModelSsmState + ModelVerify + ModelDraft + ModelVision
    + ModelEp + ModelStreams + ModelDeviceFeed
{
}
```

One model per server. `TransformerModel` (`model/types.rs`) is the concrete type: it owns the loaded `Vec<Box<dyn TransformerLayer>>`, the embedding and LM head, the KV cache and the SSM state pools, and is shared with the scheduler behind an `Arc`.

### `TransformerLayer`

`TransformerLayer` (`crates/model-layers/src/layer/transformer_layer.rs`) is likewise a composite of capability traits (`LayerCapabilities`, `LayerWeightSetup`, `LayerGraphHooks`, …) plus the per-token `decode` and the prefill entry points. Every layer is a trait object; the model's layer loop calls each one in turn, and there is no `match` on layer kind in the hot loop — the virtual call is the entire dispatch.

## Layers

The generic layer implementations are in `crates/model-layers/src/layers/`: `qwen3_attention` (full attention for the Qwen families), `qwen3_ssm` (the Qwen SSM and gated-delta-rule layers), `moe/` and `moe_grouped_decode`, `dense_ffn*`, the MTP heads (`mtp_head`, `mtp_multi`), `vision_encoder` / `vision_tower` / `glm_vit`, and `ops/` (the kernel launch wrappers). Family-specific layers live in `crates/model-arch/src/`: `nemotron_mamba2` and `nemotron_moe`, the GLM-5 Next set (`glm5next_dsa`, `glm5next_kda`, `glm5next_mhc`, `glm5next_mlp`, …), the DeepSeek V4.1 set (`attn_v41`, `moe_v41`, …), `kimi_k3/` and the DFlash head.

New models reuse these where possible; a new layer type is written only when the math genuinely differs.

## Weight loaders

One `ModelWeightLoader` per model family, in `crates/model-arch/src/weight_loader/` (plus `mistral_loader/` and the Kimi K3 loader in this crate). The trait's core methods are `load_layers`, `load_embedding`, `load_final_norm` and `load_lm_head`; the rest are optional hooks with defaults — MTP, DFlash and n-gram embedding weights, LoRA adapters, the vision encoder, KV layer dimensions, the precision schedule and tensor-parallel support.

Each loader knows the HF weight-name patterns for its family and builds its layers through the helpers in `crates/model-layers/src/weight_map/` and the quant-format dispatch in `crates/model-layers/src/quant_format/` (compressed-tensors, FP8 block-scaled, ModelOpt).

## The factory (`factory.rs`)

The single point where a model type becomes a loader:

```rust
pub fn loader_for_config(config: &ModelConfig) -> Result<Box<dyn ModelWeightLoader>> {
    let normalized = config.model_type.to_lowercase().replace(['-', '.'], "_");
    match normalized.as_str() {
        "kimi_k3" | "kimi_linear" => Ok(Box::new(KimiK3WeightLoader)),
        "qwen3_next" => Ok(Box::new(Qwen3WeightLoader)),
        "qwen3_vl_moe" => Ok(Box::new(Qwen3VLWeightLoader)),
        // qwen3_5*, qwen3_6_moe, nemotron_h, gemma4, mistral, minimax_m2,
        // deepseek_v4, glm5_next, laguna, step3p7, longcat_flash, nllb, …
        _ => bail!("Unsupported model type: …"),
    }
}
```

This is the **single code site where `model_type` strings are matched to a loader**. Everything downstream holds a `Model` and is model-agnostic. See [Adding a new model family](https://github.com/Metrale/metrale-inference-alpha/blob/main/docs/HARDWARE.md#adding-a-new-model-family).

## The generate loop (`engine.rs`)

`generate`, `generate_streaming` and `generate_speculative` run prefill, the decode loop, sampling and finish-reason detection for one request against a `&dyn Model`. The server's scheduler drives batched serving itself; these functions serve single-request callers.

## Speculative decoding

MTP draft-then-verify lives in the model (`ModelDraft`, `ModelVerify`) and in the speculative contract in `crates/model-layers/src/speculative/`; the policy — the MTP gate, the adaptive and DFlash rungs, the n-gram proposer — is in [metrale-speculative](./metrale-speculative.md). See the [MTP chapter](../deep-dives/mtp.md).

## Vision preprocessing

`crates/model-layers/src/vision_preprocess.rs` (and `video_preprocess.rs`, `vision_preprocess_glm.rs`) turns image and video input into the model's patch grid: resize and normalise, pixel-values tensor, MRoPE position IDs.

## Preflight

`crates/model-weights/src/preflight.rs` checks the weight store against the config **before** NCCL init and model construction, so an obvious checkpoint mismatch (wrong expert count, missing `lm_head`, MTP tensors the loader cannot consume) fails fast with a readable error rather than as a collective-init hang or a late build error.

## Rank agreement (`rank_agree.rs`)

During model construction, rank 0 broadcasts the environment-read scalars that shape the collective schedule and every rank compares them with its own; a mismatch fails construction naming each disagreeing entry, rather than surfacing later as a hang or a reduce over the wrong extent.

## What's explicitly not here

- **No HTTP.** That's `metrale-server`.
- **No GPU ops.** Every GPU touch is via `metrale-gpu-runtime::GpuBackend`.
- **No collective ops.** Every multi-GPU touch is via `metrale-comm::CommBackend`.

Adding a new model is almost always: one new loader under `crates/model-arch/src/weight_loader/`, one match arm in `factory.rs`, reuse of existing layers, and a new layer module only if the math genuinely differs.
