// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: LoRA tensor-name classification and adapter identity:
//! `classify_key` maps a PEFT tensor name to (layer, target, A or B), and
//! `adapter_id_hash` is the adapter's cache-identity key.
//!
//! Owner: model-layers (lora).
//! Invariants:
//! - `adapter_id_hash` never returns 0, the base (no-adapter) id.

use anyhow::{Result, anyhow, bail};
use metrale_config::{LayerType, ModelConfig};

use super::*;

/// 2026-09-25: The adapter's cache-identity key, which keeps prefix and KV
/// reuse within one adapter. It is FNV-1a over the adapter's name, then over
/// `generation`'s little-endian bytes when `generation != 0`. It never depends
/// on the pool slot, so an adapter moved to another slot keeps its id. A swap
/// into a slot bumps that slot's generation, which changes the id. `0` is
/// reserved for the base model, so a hash of 0 becomes 1.
pub fn adapter_id_hash(name: &str, generation: u64) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in name.as_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    if generation != 0 {
        for &b in generation.to_le_bytes().iter() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    if h == 0 { 1 } else { h }
}

/// 2026-09-25: Whether `key` names a `linear_attn.*` tensor other than
/// `linear_attn.out_proj`, the family `classify_key` refuses with
/// `REJECT[gdn-target]`.
/// `audit_adapter` skips these under `METRALE_LORA_ALLOW_PARTIAL` instead of
/// matching on the refusal message.
pub fn is_gdn_key(key: &str) -> bool {
    // 2026-09-25: The same splits as `classify_key`: PEFT prefix, then
    // `.layers.`, then the layer index.
    let Some(stripped) = key.strip_prefix("base_model.model.") else {
        return false;
    };
    let Some((_prefix, rest)) = stripped.split_once(".layers.") else {
        return false;
    };
    let Some((_idx, tail)) = rest.split_once('.') else {
        return false;
    };
    // 2026-09-25: `linear_attn.out_proj` has a delta path, so it is not skipped.
    tail.starts_with("linear_attn.") && tail != "linear_attn.out_proj"
}

/// 2026-09-25: PEFT tensor name → (layer, target, A or B). Every name it
/// cannot place is an error naming the reason (`REJECT[...]`), never a skip.
/// Whatever precedes `.layers.` is ignored, so `model.layers.{i}.*` and
/// `model.language_model.layers.{i}.*` both parse; the layer index follows
/// `.layers.`.
pub fn classify_key(key: &str, cfg: &ModelConfig) -> Result<(usize, LoraTarget, AdapterAb)> {
    let stripped = key.strip_prefix("base_model.model.").ok_or_else(|| {
        anyhow!("REJECT[not-peft-key]: '{key}' lacks the 'base_model.model.' PEFT prefix")
    })?;
    if stripped.contains("lora_embedding_") {
        bail!("REJECT[embedding-lora]: '{key}' — embed_tokens/lm_head LoRA is out of v0 scope");
    }
    let (module_path, ab) = if let Some(p) = stripped.strip_suffix(".lora_A.weight") {
        (p, AdapterAb::A)
    } else if let Some(p) = stripped.strip_suffix(".lora_B.weight") {
        (p, AdapterAb::B)
    } else {
        bail!(
            "REJECT[unrecognized-tensor]: '{key}' is not a lora_A/lora_B weight \
             (modules_to_save exports and old '.lora_A.<adapter>.weight' layouts \
             are not supported in v0)"
        );
    };
    let (_prefix, rest) = module_path.split_once(".layers.").ok_or_else(|| {
        anyhow!("REJECT[non-layer-module]: '{key}' targets '{module_path}' outside the layer stack")
    })?;
    let (idx_str, tail) = rest
        .split_once('.')
        .ok_or_else(|| anyhow!("REJECT[malformed-key]: '{key}'"))?;
    let layer_idx: usize = idx_str
        .parse()
        .map_err(|_| anyhow!("REJECT[malformed-layer-index]: '{key}'"))?;
    if layer_idx >= cfg.num_hidden_layers {
        bail!(
            "REJECT[layer-out-of-range]: '{key}' targets layer {layer_idx} \
             (model has {})",
            cfg.num_hidden_layers
        );
    }
    let target = match tail {
        "self_attn.q_proj" => LoraTarget::Attn(LoraModule::QProj),
        "self_attn.k_proj" => LoraTarget::Attn(LoraModule::KProj),
        "self_attn.v_proj" => LoraTarget::Attn(LoraModule::VProj),
        "self_attn.o_proj" => LoraTarget::Attn(LoraModule::OProj),
        "mlp.gate_proj" => LoraTarget::Attn(LoraModule::GateProj),
        "mlp.up_proj" => LoraTarget::Attn(LoraModule::UpProj),
        "mlp.down_proj" => LoraTarget::Attn(LoraModule::DownProj),
        // 2026-09-25: The MoE router, not the dense `mlp.gate_proj` above.
        "mlp.gate" => LoraTarget::Router,
        t if t.starts_with("mlp.experts.") => {
            classify_expert_tail(key, &t["mlp.experts.".len()..], cfg)?
        }
        // 2026-09-25: The GDN output projection (value_dim -> hidden) runs after
        // the recurrence, so its delta never enters the state update.
        // `in_proj_*` and `conv1d` feed the recurrence and are refused.
        "linear_attn.out_proj" => LoraTarget::Attn(LoraModule::OutProj),
        t if t.starts_with("linear_attn.") => bail!(
            "REJECT[gdn-target]: '{key}' — GDN/linear-attention INPUT-side \
             projections (in_proj_qkv / in_proj_z / in_proj_a / in_proj_b / \
             conv1d) feed the recurrence and stay rejected until an \
             exact-replay parity harness exists. `out_proj` IS supported."
        ),
        other => bail!("REJECT[unsupported-module]: '{key}' targets '{other}'"),
    };
    // 2026-09-25: Layer-type rules. Router and expert targets have none: a
    // linear-attention layer (`Qwen3SsmLayer`) can hold a MoE FFN too.
    match target {
        // 2026-09-25: A dense-FFN target on a dense model (`num_experts == 0`)
        // is allowed on any layer: `Qwen3SsmLayer` installs it on its
        // `FfnComponent::Dense` (`layers/qwen3_ssm/lora.rs`). On a MoE model it
        // falls to the full-attention rule below.
        LoraTarget::Attn(m) if m.is_dense_ffn() && cfg.num_experts == 0 => {}
        // 2026-09-25: GDN `out_proj` only on linear-attention layers.
        LoraTarget::Attn(m)
            if m.is_gdn_out() && cfg.layer_type(layer_idx) == LayerType::FullAttention =>
        {
            bail!(
                "REJECT[gdn-out-on-attention-layer]: '{key}' targets layer \
                 {layer_idx}, which is a full-attention layer with no GDN out_proj"
            )
        }
        LoraTarget::Attn(m) if m.is_gdn_out() => {}
        LoraTarget::Attn(_) => match cfg.layer_type(layer_idx) {
            LayerType::FullAttention => {}
            lt => bail!(
                "REJECT[non-full-attention-layer]: '{key}' targets layer {layer_idx} \
                 ({lt:?}); attention-projection LoRA applies only on the full-attention \
                 layers {:?}",
                full_attention_layers(cfg),
            ),
        },
        LoraTarget::Router | LoraTarget::Expert { .. } => {}
    }
    Ok((layer_idx, target, ab))
}

/// 2026-09-25: Parse the `{N}.{proj}` remainder of an `mlp.experts.` tail
/// (e.g. `"7.gate_proj"`) into a routed-expert [`LoraTarget`]. Refused: a dense
/// model (`num_experts == 0`), a fused or unindexed layout, a non-numeric or
/// out-of-range expert index, and any projection but gate/up/down_proj.
fn classify_expert_tail(key: &str, rest: &str, cfg: &ModelConfig) -> Result<LoraTarget> {
    if cfg.num_experts == 0 {
        bail!(
            "REJECT[expert-lora-on-dense-model]: '{key}' targets a routed expert but \
             the model has num_experts=0 (dense) — use mlp.{{gate,up,down}}_proj instead"
        );
    }
    let (n_str, proj_str) = rest.split_once('.').ok_or_else(|| {
        anyhow!(
            "REJECT[fused-expert-lora]: '{key}' — fused/unindexed expert layout \
             (e.g. experts.gate_up_proj via target_parameters) is deferred to \
             Feature-1 phase 3; export per-expert mlp.experts.{{N}}.{{proj}} tensors"
        )
    })?;
    let n: usize = n_str.parse().map_err(|_| {
        anyhow!("REJECT[malformed-expert-index]: '{key}' — '{n_str}' is not an index")
    })?;
    if n >= cfg.num_experts {
        bail!(
            "REJECT[expert-out-of-range]: '{key}' targets expert {n} \
             (model has {} experts)",
            cfg.num_experts
        );
    }
    let proj = match proj_str {
        "gate_proj" => ExpertProj::Gate,
        "up_proj" => ExpertProj::Up,
        "down_proj" => ExpertProj::Down,
        "gate_up_proj" => bail!(
            "REJECT[fused-expert-lora]: '{key}' — fused gate_up_proj is deferred to \
             Feature-1 phase 3 (needs a per-expert decomposer)"
        ),
        other => bail!("REJECT[unsupported-expert-proj]: '{key}' proj '{other}'"),
    };
    Ok(LoraTarget::Expert { n: n as u16, proj })
}

#[cfg(test)]
#[path = "key_tests.rs"]
mod tests;
