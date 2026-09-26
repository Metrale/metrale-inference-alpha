// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Parser for a PEFT `adapter_config.json` (runtime LoRA adapters).
//!
//! Unlike `parse_quantization_config`, which returns `Option`, this parser returns an error
//! with a `REJECT(<field>)` reason for every setting it cannot apply, so a requested adapter
//! is never partly skipped without notice.
//!
//! The `ModelConfig` fields `kv_lora_rank`, `q_lora_rank` and `o_lora_rank` are MLA
//! attention compression ranks, unrelated to adapters; names here use `peft_` / `adapter_`.
//!
//! Owner: config (LoRA adapters).
//! Invariants:
//! - A parsed `PeftAdapterConfig` has `r > 0`, a finite positive `lora_alpha`, an explicit
//!   `use_rslora`, and at least one target module, target pattern or token overlay.

use anyhow::{Context, Result, bail};
use serde::Deserialize;

/// 2026-09-26: The module names (last `.`-segment of a `target_modules` entry) this parser
/// accepts; `validate_target_module` also accepts the MoE router leaf `gate`. On a
/// gated-attention model `q_proj` is the whole Q-and-gate projection,
/// `2 * num_attention_heads * head_dim` wide (`metrale_model_layers` `LoraModule::dims`).
/// Which layers a module may target is decided per tensor by `classify_key` in the same
/// crate.
pub const PEFT_SUPPORTED_TARGET_MODULES: &[&str] = &[
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    // 2026-09-26: The GDN / linear-attention block's output projection (value_dim -> hidden).
    "out_proj",
];

/// 2026-09-26: The fields of a PEFT `adapter_config.json` that the engine uses, as validated
/// by [`parse_peft_adapter_config`]. `lora_dropout`, a training-only setting, is ignored.
#[derive(Debug, Clone)]
pub struct PeftAdapterConfig {
    /// 2026-09-26: LoRA rank, greater than 0.
    pub r: usize,
    /// 2026-09-26: LoRA alpha, finite and positive; the JSON may hold an integer or a float.
    pub lora_alpha: f64,
    /// 2026-09-26: The `target_modules` list as written: bare module names or full paths, each
    /// validated on its last `.`-segment. Per-layer matching is the weight loader's.
    pub target_modules: Vec<String>,
    /// 2026-09-26: PEFT's regex form of `target_modules` (a JSON string other than
    /// `all-linear`), kept as written and not expanded. It skips the name-level allow-list,
    /// but every adapter tensor is still checked by `classify_key`, which accepts only the
    /// projections it names.
    pub target_modules_pattern: Option<String>,
    /// 2026-09-26: rsLoRA: scaling is `alpha/sqrt(r)` instead of `alpha/r`. The config must
    /// state it; an absent key is rejected rather than read as false.
    pub use_rslora: bool,
    /// 2026-09-26: The `layers_to_transform` list, if present. Nothing reads it; the weight
    /// loader decides which layers receive deltas.
    pub layers_to_transform: Option<Vec<usize>>,
    /// 2026-09-26: Token ids for the trainable-token overlay, unique and ascending, from the
    /// `trainable_token_indices` list or the list shared by its `embed_tokens` and `lm_head`
    /// entries. Empty means no overlay.
    pub trainable_token_indices: Vec<u32>,
    /// 2026-09-26: The accepted `modules_to_save` leaves, `embed_tokens` and `lm_head`, sorted
    /// and deduplicated. Any other leaf is a `REJECT(modules_to_save)`. Empty means no
    /// full-module overlay.
    pub modules_to_save: Vec<String>,
    /// 2026-09-26: Always `false` from this parser. Low-rank embedding LoRA
    /// (`lora_embedding_A/B`) is detected from tensor names by the overlay loader, which
    /// rejects it.
    pub lora_embedding: bool,
}

impl PeftAdapterConfig {
    /// 2026-09-26: The delta scale, `y += scaling() * (x @ Aᵀ) @ Bᵀ`: `alpha/r`, or
    /// `alpha/sqrt(r)` with `use_rslora`.
    pub fn scaling(&self) -> f32 {
        debug_assert!(self.r > 0, "validated at parse");
        if self.use_rslora {
            (self.lora_alpha / (self.r as f64).sqrt()) as f32
        } else {
            (self.lora_alpha / self.r as f64) as f32
        }
    }
}

/// 2026-09-26: Deserialization target with PEFT's field names. Unknown keys are ignored
/// (no `deny_unknown_fields`), so any key that changes inference must be named here to be
/// checked.
#[derive(Deserialize)]
struct RawPeftAdapterConfig {
    /// 2026-09-26: Absent or `LORA` (any case); anything else is rejected.
    #[serde(default)]
    peft_type: Option<String>,
    r: usize,
    lora_alpha: f64,
    /// 2026-09-26: A list of module names, or a string: `all-linear` is rejected and any other
    /// string is a regex pattern. Absent or null is accepted for a token-overlay adapter.
    #[serde(default)]
    target_modules: serde_json::Value,
    /// 2026-09-26: `None` (absent) is rejected, not read as false.
    #[serde(default)]
    use_rslora: Option<bool>,
    #[serde(default)]
    use_dora: bool,
    /// 2026-09-26: Absent or `none`; any other value is rejected.
    #[serde(default)]
    bias: Option<String>,
    #[serde(default)]
    rank_pattern: Option<serde_json::Map<String, serde_json::Value>>,
    #[serde(default)]
    alpha_pattern: Option<serde_json::Map<String, serde_json::Value>>,
    /// 2026-09-26: Full (not low-rank) modules saved with the adapter. The leaves
    /// `embed_tokens` and `lm_head` are accepted as a token overlay; any other is rejected.
    #[serde(default)]
    modules_to_save: Option<Vec<String>>,
    /// 2026-09-26: Layer subset. Null or an array of non-negative integers is accepted;
    /// anything else is rejected.
    #[serde(default)]
    layers_to_transform: Option<serde_json::Value>,
    /// 2026-09-26: Token ids whose `embed_tokens` / `lm_head` rows the adapter replaces, as a
    /// list or as a per-module object; [`parse_trainable_tokens`] reads both forms.
    #[serde(default)]
    trainable_token_indices: Option<serde_json::Value>,
    /// 2026-09-26: LoRA on fused parameter tensors (routed MoE experts). A non-empty value is
    /// rejected; without this field it would be ignored as an unknown key.
    #[serde(default)]
    target_parameters: Option<Vec<String>>,
}

/// 2026-09-26: Parse a PEFT `adapter_config.json` payload.
///
/// Errors with a `REJECT(<field>)` message for each unsupported setting: a non-LoRA
/// `peft_type`, DoRA, a bias other than `none`, non-empty `rank_pattern` / `alpha_pattern` /
/// `target_parameters`, a `modules_to_save` leaf other than `embed_tokens` / `lm_head`, an
/// absent `use_rslora`, `r == 0`, a non-positive `lora_alpha`, a malformed
/// `layers_to_transform` or `trainable_token_indices`, a rejected target module, and an
/// adapter that targets nothing. The caller adds the file path to the error.
pub fn parse_peft_adapter_config(json: &str) -> Result<PeftAdapterConfig> {
    let raw: RawPeftAdapterConfig = serde_json::from_str(json)
        .context("Parsing PEFT adapter_config.json (r / lora_alpha / target_modules required)")?;

    if let Some(ref pt) = raw.peft_type
        && !pt.eq_ignore_ascii_case("LORA")
    {
        bail!("REJECT(peft_type): adapter declares peft_type='{pt}'; only LORA is supported");
    }
    if raw.use_dora {
        bail!(
            "REJECT(use_dora): DoRA adapters are unsupported (magnitude decomposition has no runtime-delta form)"
        );
    }
    if let Some(ref b) = raw.bias
        && b != "none"
    {
        bail!("REJECT(bias): bias='{b}' ships trained bias deltas; only bias='none' is supported");
    }
    if raw.rank_pattern.as_ref().is_some_and(|m| !m.is_empty()) {
        bail!(
            "REJECT(rank_pattern): per-module rank overrides are unsupported in v0 (uniform r only)"
        );
    }
    if raw.alpha_pattern.as_ref().is_some_and(|m| !m.is_empty()) {
        bail!(
            "REJECT(alpha_pattern): per-module alpha overrides are unsupported in v0 (uniform lora_alpha only)"
        );
    }
    let modules_to_save = partition_modules_to_save(raw.modules_to_save.as_deref())?;

    if raw
        .target_parameters
        .as_ref()
        .is_some_and(|v| !v.is_empty())
    {
        bail!(
            "REJECT(target_parameters): fused-parameter LoRA {:?} (routed MoE experts) \
             is deferred to Feature 1 phase 3",
            raw.target_parameters.as_deref().unwrap_or_default()
        );
    }

    let use_rslora = raw.use_rslora.ok_or_else(|| {
        anyhow::anyhow!(
            "REJECT(use_rslora): field absent — scaling inputs are never defaulted \
             (PEFT <0.7 config; re-export the adapter with peft>=0.7)"
        )
    })?;

    let layers_to_transform = parse_layers_to_transform(&raw.layers_to_transform)?;

    if raw.r == 0 {
        bail!("REJECT(r): LoRA rank must be > 0");
    }
    if !(raw.lora_alpha.is_finite() && raw.lora_alpha > 0.0) {
        bail!(
            "REJECT(lora_alpha): must be a finite positive number, got {}",
            raw.lora_alpha
        );
    }

    let trainable_token_indices = parse_trainable_tokens(&raw.trainable_token_indices)?;

    let (target_modules, target_modules_pattern) = parse_target_modules(&raw.target_modules)?;
    for entry in &target_modules {
        validate_target_module(entry)?;
    }

    // 2026-09-26: An adapter with only a token overlay may list no target module.
    let has_overlay = !trainable_token_indices.is_empty() || !modules_to_save.is_empty();
    if target_modules.is_empty() && target_modules_pattern.is_none() && !has_overlay {
        bail!("REJECT(target_modules): empty list — adapter targets nothing");
    }

    Ok(PeftAdapterConfig {
        r: raw.r,
        lora_alpha: raw.lora_alpha,
        target_modules,
        target_modules_pattern,
        use_rslora,
        layers_to_transform,
        trainable_token_indices,
        modules_to_save,
        lora_embedding: false,
    })
}

/// 2026-09-26: The accepted `modules_to_save` leaves (`embed_tokens`, `lm_head`, matched on
/// the last `.`-segment), sorted and deduplicated. Any other leaf is an error.
fn partition_modules_to_save(mods: Option<&[String]>) -> Result<Vec<String>> {
    let Some(mods) = mods else {
        return Ok(Vec::new());
    };
    let mut accepted = Vec::new();
    for m in mods {
        let leaf = m.rsplit('.').next().unwrap_or(m);
        match leaf {
            "embed_tokens" | "lm_head" => accepted.push(leaf.to_string()),
            other => bail!(
                "REJECT(modules_to_save): adapter saves full module '{other}'; only the \
                 token-overlay subset {{embed_tokens, lm_head}} is supported"
            ),
        }
    }
    accepted.sort();
    accepted.dedup();
    Ok(accepted)
}

/// 2026-09-26: Parse `trainable_token_indices` into one list of ids, shared by the embedding
/// and lm-head overlays.
///
/// Absent or null gives an empty list. A list `[id, …]` is taken as is. An object
/// `{"embed_tokens": […], "lm_head": […]}` gives its non-null list, and the two lists must
/// be equal. Errors on an id that is negative, above `u32::MAX`, repeated or out of
/// ascending order, on another module key, and on differing lists.
fn parse_trainable_tokens(v: &Option<serde_json::Value>) -> Result<Vec<u32>> {
    fn parse_ids(arr: &[serde_json::Value]) -> Result<Vec<u32>> {
        let mut ids = Vec::with_capacity(arr.len());
        let mut seen = std::collections::HashSet::new();
        let mut previous: Option<u32> = None;
        for e in arr {
            let n = e.as_u64().context(
                "REJECT(trainable_token_indices): entries must be non-negative integers",
            )?;
            if n > u32::MAX as u64 {
                bail!("REJECT(trainable_token_indices): id {n} exceeds u32 range");
            }
            let id = n as u32;
            if !seen.insert(id) {
                bail!("REJECT(trainable_token_indices): duplicate id {id}");
            }
            if let Some(prev) = previous
                && id < prev
            {
                bail!(
                    "REJECT(trainable_token_indices): ids must be ascending to preserve \
                     trainable_tokens_delta row order; {id} follows {prev}"
                );
            }
            previous = Some(id);
            ids.push(id);
        }
        Ok(ids)
    }

    match v {
        None | Some(serde_json::Value::Null) => Ok(Vec::new()),
        Some(serde_json::Value::Array(arr)) => parse_ids(arr),
        Some(serde_json::Value::Object(map)) => {
            let mut shared: Option<Vec<u32>> = None;
            for (module, val) in map {
                if module != "embed_tokens" && module != "lm_head" {
                    bail!(
                        "REJECT(trainable_token_indices): unsupported module '{module}'; only \
                         embed_tokens and lm_head are supported"
                    );
                }
                let module_ids = match val {
                    serde_json::Value::Array(arr) => parse_ids(arr)?,
                    serde_json::Value::Null => continue,
                    other => bail!(
                        "REJECT(trainable_token_indices): dict value must be an array, got {other}"
                    ),
                };
                if let Some(ref expected) = shared
                    && expected != &module_ids
                {
                    bail!(
                        "REJECT(trainable_token_indices): per-module token lists differ; \
                         Metrale Engine requires one shared embed_tokens/lm_head order"
                    );
                }
                shared = Some(module_ids);
            }
            Ok(shared.unwrap_or_default())
        }
        Some(other) => bail!(
            "REJECT(trainable_token_indices): expected null, an array, or a per-module \
             object, got {other}"
        ),
    }
}

fn parse_layers_to_transform(v: &Option<serde_json::Value>) -> Result<Option<Vec<usize>>> {
    match v {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(serde_json::Value::Array(arr)) => {
            let layers = arr
                .iter()
                .map(|e| {
                    e.as_u64().map(|n| n as usize).context(
                        "REJECT(layers_to_transform): entries must be non-negative integers",
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(Some(layers))
        }
        Some(other) => bail!(
            "REJECT(layers_to_transform): expected null or an array of layer indices, got {other}"
        ),
    }
}

fn parse_target_modules(v: &serde_json::Value) -> Result<(Vec<String>, Option<String>)> {
    match v {
        // 2026-09-26: `all-linear` is a PEFT keyword for every linear layer, which cannot be
        // enumerated here: a fused or packed tensor stands for several linear layers.
        serde_json::Value::String(s) if s == "all-linear" => bail!(
            "REJECT(target_modules): string form '{s}' is unsupported — Metrale Engine cannot \
             enumerate 'all linear' against fused/quantized layouts; re-export the \
             adapter with an explicit module list or a regex"
        ),
        // 2026-09-26: Any other string is a regex pattern; see
        // `PeftAdapterConfig::target_modules_pattern`.
        serde_json::Value::String(s) => Ok((Vec::new(), Some(s.clone()))),
        serde_json::Value::Array(arr) => {
            let mods: Vec<String> = arr
                .iter()
                .map(|e| {
                    e.as_str()
                        .map(str::to_string)
                        .context("REJECT(target_modules): entries must be strings")
                })
                .collect::<Result<_>>()?;
            // 2026-09-26: The caller judges an empty list.
            Ok((mods, None))
        }
        serde_json::Value::Null => Ok((Vec::new(), None)),
        other => bail!("REJECT(target_modules): expected an array of module names, got {other}"),
    }
}

/// 2026-09-26: `METRALE_LORA_ALLOW_PARTIAL` (`1` or `true`, any case), read once per process:
/// load an adapter that names target modules the engine cannot apply, skipping those.
///
/// `metrale_model_layers::lora::env` re-exports this function; it is defined in this crate
/// because `validate_target_module` reads it at parse time.
///
/// Off by default: a partly applied adapter cannot be told apart from a bad one.
pub fn allow_partial_targets() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("METRALE_LORA_ALLOW_PARTIAL")
            .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
    })
}

/// 2026-09-26: The name-level check of one `target_modules` entry, a bare name (`"k_proj"`)
/// or a full path (`"model.layers.3.self_attn.k_proj"`), on its last `.`-segment. Which
/// layers it may target is checked per tensor by the weight loader.
fn validate_target_module(entry: &str) -> Result<()> {
    let leaf = entry.rsplit('.').next().unwrap_or(entry);
    match leaf {
        // 2026-09-26: The GDN / linear-attention inputs, fused and split spellings, and
        // `conv1d` feed the recurrence and are rejected unless `allow_partial_targets()`.
        // The block's `out_proj` is in the allow-list.
        "in_proj_qkvz" | "in_proj_ba" | "in_proj_qkv" | "in_proj_z" | "in_proj_a" | "in_proj_b"
        | "conv1d"
            if !allow_partial_targets() =>
        {
            bail!(
                "REJECT(gdn): target module '{leaf}' is a GDN/linear-attention projection; GDN \
                 layers are unsupported in v0 (full-attention layers only). Set \
                 METRALE_LORA_ALLOW_PARTIAL=1 to load the adapter anyway, applying only its \
                 supported modules — it will then be PARTIALLY applied."
            )
        }
        "in_proj_qkvz" | "in_proj_ba" | "in_proj_qkv" | "in_proj_z" | "in_proj_a" | "in_proj_b"
        | "conv1d" => Ok(()),
        "embed_tokens" | "lm_head" => {
            bail!("REJECT(embedding): target module '{leaf}' is unsupported in v0")
        }
        // 2026-09-26: The MoE router (`mlp.gate`), not the dense `gate_proj`. Expert
        // projections (`mlp.experts.N.gate_proj`) use the dense leaves. Whether router and
        // expert deltas load is decided by `classify_key` and `METRALE_LORA_EXPERTS`.
        "gate" => Ok(()),
        m if PEFT_SUPPORTED_TARGET_MODULES.contains(&m) => Ok(()),
        other => bail!(
            "REJECT(unknown_module): target module '{other}' is not in the v0 allow-list \
             {PEFT_SUPPORTED_TARGET_MODULES:?}"
        ),
    }
}

#[cfg(test)]
#[path = "lora_tests.rs"]
mod tests;
