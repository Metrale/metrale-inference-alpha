// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Config-derived model fingerprint and the SSM tier namespaces folded into tier keys.
//!
//! Owner: model-engine SSM tier.
//! Invariants:
//! - [`ModelFingerprint::derive_with_id`] is a pure function of the config fields it
//!   encodes, `blob_bytes` and `model_id`. The input is tagged records, u64s as
//!   `[tag][8-byte LE]` and strings as `[tag][4-byte LE len][bytes]`, hashed with
//!   FNV-1a/64 from `metrale_storage::tier::hash`, which documents why the hash is vendored.
//! - Every fingerprint and namespace returned here is non-zero.
//! - An explicit namespace override that does not parse, or parses to 0, is an error.
//!
//! The paging peer drives one residency per blob kind and shape for all its clients
//! (`metrale_storage::cache_peer::registry`), so the namespace folded into each key is
//! what keeps two models apart there. The KV cache dtype is not encoded: SSM h/conv
//! state is FP32 whatever the KV dtype. The KV paging tier folds its own dtype and
//! block geometry in `metrale_storage::kv_paging::ns`.

use std::num::NonZeroU64;

use anyhow::{Result, anyhow, bail};
use metrale_config::ModelConfig;

pub(crate) use metrale_storage::tier::hash::{FNV_OFFSET, fnv1a_64, mix64};

/// 2026-09-25: Encoding version, hashed as the tag-0x00 record. Bump it whenever the
/// encoded field set or order changes; every key changes with it.
pub(crate) const FP_VERSION: u64 = 2;

fn put_u64(buf: &mut Vec<u8>, tag: u8, v: u64) {
    buf.push(tag);
    buf.extend_from_slice(&v.to_le_bytes());
}

fn put_str(buf: &mut Vec<u8>, tag: u8, s: &str) {
    buf.push(tag);
    buf.extend_from_slice(&(s.len() as u32).to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
}

/// 2026-09-25: Identity of the bytes this model's SSM tier produces, derived from the
/// loaded [`ModelConfig`] geometry, its quantization identity and `blob_bytes`.
/// Non-zero by construction.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct ModelFingerprint(NonZeroU64);

impl ModelFingerprint {
    /// 2026-09-25: Derive the fingerprint and log it at INFO. `METRALE_MODEL_ID` is an
    /// optional extra salt for checkpoints whose config is identical; unset, it is
    /// encoded as an empty string.
    pub(crate) fn derive(cfg: &ModelConfig, blob_bytes: usize) -> Result<Self> {
        let model_id = std::env::var("METRALE_MODEL_ID").unwrap_or_default();
        let fp = Self::derive_with_id(cfg, blob_bytes, &model_id)?;
        tracing::info!(
            "SSM tier model fingerprint = {:#018x} (model_type={}, blob_bytes={blob_bytes}, \
             METRALE_MODEL_ID={model_id:?}); pin with METRALE_SSM_SWAP_NS / METRALE_SSM_DECODE_NS",
            fp.get(),
            cfg.model_type,
        );
        // 2026-09-25: The config carries no weight identity, so checkpoints with an
        // identical config derive the same namespace. Warn when a shared store is
        // selected and neither `METRALE_MODEL_ID` nor a namespace override is set.
        let shared_store = std::env::var("METRALE_SSM_SWAP").ok().as_deref() == Some("1")
            || std::env::var("METRALE_SSM_DECODE_TIER").ok().as_deref() == Some("peer");
        let overridden = std::env::var_os("METRALE_SSM_SWAP_NS").is_some()
            || std::env::var_os("METRALE_SSM_DECODE_NS").is_some();
        if shared_store && model_id.is_empty() && !overridden {
            tracing::warn!(
                "a SHARED SSM cache store is selected (METRALE_SSM_SWAP=1 local swap file, or \
                 METRALE_SSM_DECODE_TIER=peer) but METRALE_MODEL_ID is unset: the fingerprint is \
                 derived from config GEOMETRY ONLY, so two checkpoints with identical config \
                 (fine-tunes, RL variants, continued pre-train of one base) will SHARE cache \
                 keys and silently cross-serve recurrent state. Set METRALE_MODEL_ID to a stable \
                 per-checkpoint string (or set METRALE_SSM_SWAP_NS / METRALE_SSM_DECODE_NS \
                 explicitly) when co-locating such models on one store."
            );
        }
        Ok(fp)
    }

    /// 2026-09-25: KV paging fingerprint: the same encoding as the SSM tier with
    /// `blob_bytes = 0`. The SSM tier is built only for models with SSM layers, so its
    /// tag-0x40 record is non-zero and the two never share an input. Models without
    /// SSM layers still get a fingerprint. KV fields belong in
    /// `metrale_storage::kv_paging::ns`; adding them here would change every SSM key.
    pub(crate) fn derive_kv(cfg: &ModelConfig) -> Result<Self> {
        let model_id = std::env::var("METRALE_MODEL_ID").unwrap_or_default();
        let fp = Self::derive_with_id(cfg, 0, &model_id)?;
        tracing::info!(
            "KV paging model fingerprint = {:#018x} (model_type={}, \
             METRALE_MODEL_ID={model_id:?}); folded into the METRALE_KV_PAGING namespace",
            fp.get(),
            cfg.model_type,
        );
        Ok(fp)
    }

    /// 2026-09-25: The encoding itself, with no env reads and no logging. Changing the
    /// field set or order requires an [`FP_VERSION`] bump.
    pub(crate) fn derive_with_id(
        cfg: &ModelConfig,
        blob_bytes: usize,
        model_id: &str,
    ) -> Result<Self> {
        if cfg.model_type.is_empty() && cfg.num_hidden_layers == 0 {
            bail!(
                "cannot derive a model fingerprint: empty model_type and zero geometry; \
                 fix the model config or set METRALE_SSM_SWAP_NS / METRALE_SSM_DECODE_NS \
                 to explicit non-zero u64 namespaces"
            );
        }
        let (qm, qa, qf) = cfg.quantization_config.as_ref().map_or(("", "", ""), |q| {
            (
                q.quant_method.as_str(),
                q.quant_algo.as_str(),
                q.format.as_str(),
            )
        });
        let mut buf = Vec::with_capacity(256);
        put_u64(&mut buf, 0x00, FP_VERSION);
        put_str(&mut buf, 0x01, &cfg.model_type);
        // 2026-09-25: Quantization identity: builds of one checkpoint that differ only
        // in quantization have the same geometry.
        put_str(&mut buf, 0x02, qm);
        put_str(&mut buf, 0x03, qa);
        put_str(&mut buf, 0x04, qf);
        put_str(&mut buf, 0x05, model_id);
        for (tag, v) in [
            (0x10, cfg.num_hidden_layers),
            (0x11, cfg.num_ssm_layers()),
            (0x12, cfg.num_attention_layers()),
            (0x13, cfg.head_dim),
            (0x14, cfg.num_key_value_heads),
            (0x15, cfg.num_experts),
            // 2026-09-25: Residual-stream and FFN widths. `blob_bytes` does not capture
            // them: it is `num_ssm_layers * (h + conv)` bytes (`spill_blob_bytes`).
            (0x16, cfg.hidden_size),
            (0x17, cfg.num_attention_heads),
            (0x18, cfg.intermediate_size),
            (0x19, cfg.moe_intermediate_size),
            (0x1a, cfg.num_experts_per_tok),
            // 2026-09-25: SSM state geometry; the blob size is a lossy product of these.
            (0x20, cfg.linear_num_key_heads),
            (0x21, cfg.linear_key_head_dim),
            (0x22, cfg.linear_num_value_heads),
            (0x23, cfg.linear_value_head_dim),
            (0x24, cfg.linear_conv_kernel_dim),
            (0x25, cfg.mamba_num_heads),
            (0x26, cfg.mamba_head_dim),
            (0x27, cfg.ssm_state_size),
            (0x28, cfg.n_groups),
            // 2026-09-25: SSM h/conv element size in bytes: the FP32 factor 4 in
            // `ssm_h_state_bytes` and `ssm_conv_state_bytes`.
            (0x30, 4),
            (0x40, blob_bytes),
        ] {
            put_u64(&mut buf, tag, v as u64);
        }
        // 2026-09-25: Per-layer (kv_heads, head_dim) overrides, filled by the loader
        // (`factory/build.rs`), in layer order.
        put_u64(&mut buf, 0x50, cfg.kv_layer_dims.len() as u64);
        for &(kvh, hd) in &cfg.kv_layer_dims {
            put_u64(&mut buf, 0x51, kvh as u64);
            put_u64(&mut buf, 0x52, hd as u64);
        }
        let h = fnv1a_64(&buf);
        // 2026-09-25: A zero hash maps to `FNV_OFFSET`, keeping the result non-zero.
        Ok(Self(
            NonZeroU64::new(h).unwrap_or(NonZeroU64::new(FNV_OFFSET).unwrap()),
        ))
    }

    pub(crate) fn get(self) -> u64 {
        self.0.get()
    }

    pub(crate) fn nonzero(self) -> NonZeroU64 {
        self.0
    }
}

/// 2026-09-25: Marconi swap namespace: `METRALE_SSM_SWAP_NS` when set (parsed by
/// [`parse_ns`], so junk or 0 is an error), else the fingerprint.
pub(crate) fn resolve_swap_ns(fp: ModelFingerprint) -> Result<NonZeroU64> {
    resolve_ns_from(
        std::env::var("METRALE_SSM_SWAP_NS").ok().as_deref(),
        "METRALE_SSM_SWAP_NS",
        fp.nonzero(),
    )
}

/// 2026-09-25: Per-process decode client salt, stored once so every decode store
/// built in this process folds the same value. `METRALE_SSM_DECODE_CLIENT_ID` pins
/// it ([`parse_u64_strict`], 0 accepted); unset, it comes from
/// `metrale_storage::tier::entropy::random_u64`, whose error is returned.
fn decode_client_salt() -> Result<u64> {
    use std::sync::OnceLock;
    // 2026-09-25: A process-wide static because the salt identifies the process on a
    // shared tier, not a model or a run configuration.
    static SALT: OnceLock<u64> = OnceLock::new();
    if let Some(&s) = SALT.get() {
        return Ok(s);
    }
    let salt = match std::env::var("METRALE_SSM_DECODE_CLIENT_ID").ok() {
        Some(raw) => parse_u64_strict("METRALE_SSM_DECODE_CLIENT_ID", &raw)?,
        None => metrale_storage::tier::entropy::random_u64()?,
    };
    // 2026-09-25: If another thread stored first, `get_or_init` returns its value.
    Ok(*SALT.get_or_init(|| salt))
}

/// 2026-09-25: Env-free core of the decode namespace:
/// `mix64(mix64(fingerprint, DECODE_DOMAIN), client_salt)`. `DECODE_DOMAIN` separates
/// decode keys from the same model's Marconi keys, whose namespace is the bare
/// fingerprint ([`resolve_swap_ns`]); the fingerprint separates models; the salt
/// separates processes. A zero from the first fold falls back to `DECODE_DOMAIN`, a
/// zero from the second to the first fold's value, so the result is never zero.
pub(crate) fn derive_decode_ns_salted(fp: u64, salt: u64) -> NonZeroU64 {
    let base = NonZeroU64::new(mix64(fp, metrale_kernels::DECODE_DOMAIN)).unwrap_or_else(|| {
        NonZeroU64::new(metrale_kernels::DECODE_DOMAIN)
            .expect("DECODE_DOMAIN is a non-zero constant")
    });
    NonZeroU64::new(mix64(base.get(), salt)).unwrap_or(base)
}

/// 2026-09-25: Decode namespace: `METRALE_SSM_DECODE_NS` when set, used unsalted and
/// parsed by [`parse_ns`]; else [`derive_decode_ns_salted`] with the process salt,
/// which is logged at INFO.
pub(crate) fn resolve_decode_ns(fp: ModelFingerprint) -> Result<NonZeroU64> {
    if let Ok(raw) = std::env::var("METRALE_SSM_DECODE_NS") {
        tracing::warn!(
            "METRALE_SSM_DECODE_NS={raw:?} bypasses the per-process decode client salt: decode \
             keys are SLOT COORDINATES (not content hashes), so two processes sharing this \
             value on one peer WILL cross-serve and cross-delete each other's rollback state \
             (silent corruption + spurious 'cold MISS on live target'). Ensure a DISTINCT \
             value per process, or unset it and pin METRALE_SSM_DECODE_CLIENT_ID instead."
        );
        if std::env::var_os("METRALE_SSM_DECODE_CLIENT_ID").is_some() {
            tracing::warn!(
                "METRALE_SSM_DECODE_CLIENT_ID is IGNORED while METRALE_SSM_DECODE_NS is set \
                 (the explicit namespace fully determines the wire keys)"
            );
        }
        return parse_ns("METRALE_SSM_DECODE_NS", &raw);
    }
    let salt = decode_client_salt()?;
    let ns = derive_decode_ns_salted(fp.get(), salt);
    tracing::info!(
        "SSM decode ns = {ns:#018x} (fp {:#018x} ⊕ DECODE_DOMAIN ⊕ client_salt \
         {salt:#018x}); decode keys are CLIENT-PRIVATE on a shared peer; pin \
         METRALE_SSM_DECODE_CLIENT_ID={salt:#x} to reproduce this namespace",
        fp.get(),
    );
    Ok(ns)
}

/// 2026-09-25: Env-free core of the namespace overrides: `override_raw` parsed by
/// [`parse_ns`] when present, else `derived`.
pub(crate) fn resolve_ns_from(
    override_raw: Option<&str>,
    var: &str,
    derived: NonZeroU64,
) -> Result<NonZeroU64> {
    match override_raw {
        Some(raw) => parse_ns(var, raw),
        None => Ok(derived),
    }
}

/// 2026-09-25: Strict u64 parser, decimal or `0x`/`0X` hex after trimming; anything
/// else is an error. 0 is accepted; [`parse_ns`] adds the non-zero check.
pub(crate) fn parse_u64_strict(var: &str, raw: &str) -> Result<u64> {
    let s = raw.trim();
    let parsed = match s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        Some(hex) => u64::from_str_radix(hex, 16),
        None => s.parse::<u64>(),
    };
    parsed.map_err(|e| anyhow!("{var}={raw:?} is not a valid u64 (decimal or 0x-hex): {e}"))
}

/// 2026-09-25: Namespace override parser: [`parse_u64_strict`], and 0 is an error.
pub(crate) fn parse_ns(var: &str, raw: &str) -> Result<NonZeroU64> {
    let v = parse_u64_strict(var, raw)?;
    NonZeroU64::new(v).ok_or_else(|| {
        anyhow!(
            "{var}=0 is invalid: the ns=0 passthrough is removed (it silently \
             cross-served state between models on a shared peer); unset {var} \
             to use the derived model fingerprint (logged at INFO on startup)"
        )
    })
}

#[cfg(test)]
#[path = "fingerprint_tests.rs"]
mod tests;
