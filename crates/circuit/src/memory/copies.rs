// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: Load-time derived weight copies (`kernels/circuits/COPIES.toml`): every
//! device allocation the loader makes for a weight beyond the checkpoint tensors it stores
//! (requantized serving copies, transposed prefill twins, repacks, widened scales, and the
//! leaked intermediates nothing reads). A rule matches nodes by op, by the checkpoint's declared
//! weight format, by the served format and by serve settings, and states the byte law of its copy.
//!
//! Owner: metrale-circuit (memory).
//! Invariants:
//! - A copy's bytes come from the node's own shape (`out x k`, per expert for routed experts) and
//!   the rule's law; nothing is sized from a measured total.
//! - Every rule cites the loader site that allocates it; a rule that matches no node of a circuit
//!   is not an error (it belongs to another checkpoint), but an unknown op, format, law or
//!   setting spelling is refused at parse.

use std::collections::BTreeMap;

use serde::Deserialize;

use crate::format::Format;
use crate::ir::{OpKind, Section};

/// 2026-10-02: How a copy's bytes follow from the weight's `out x k` shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CopyLaw {
    /// 2026-10-02: A whole weight in this format ([`Format::weight_bytes`]).
    Weight(Format),
    /// 2026-10-02: One F32 scale per output row (`out x 4`).
    RowScaleF32,
    /// 2026-10-02: One F32 scale per `r x c` block (`ceil(out/r) x ceil(k/c) x 4`).
    BlockGridF32(u32, u32),
}

impl CopyLaw {
    /// 2026-10-02: `bf16`, `nvfp4/g16`, `fp8/...` (a [`Format`]), `row_scale_f32` or
    /// `block_grid_f32/<r>x<c>`.
    pub fn parse(s: &str) -> Option<Self> {
        if s == "row_scale_f32" {
            return Some(Self::RowScaleF32);
        }
        if let Some(rc) = s.strip_prefix("block_grid_f32/") {
            let (r, c) = rc.split_once('x')?;
            let pos = |v: &str| v.parse::<u32>().ok().filter(|&v| v > 0);
            return Some(Self::BlockGridF32(pos(r)?, pos(c)?));
        }
        Format::parse(s).ok().map(Self::Weight)
    }

    /// 2026-10-02: Bytes of one copy of an `out x k` weight; `None` when the shape does not fit
    /// the format's groups, or on overflow.
    pub fn bytes(self, out: u64, k: u64) -> Option<u64> {
        match self {
            Self::Weight(f) => f.weight_bytes(out, k),
            Self::RowScaleF32 => out.checked_mul(4),
            Self::BlockGridF32(r, c) => out
                .div_ceil(u64::from(r))
                .checked_mul(k.div_ceil(u64::from(c)))?
                .checked_mul(4),
        }
    }
}

/// 2026-10-02: One copy rule.
#[derive(Debug, Clone, PartialEq)]
pub struct CopyRule {
    /// 2026-10-02: Rule id (unique).
    pub id: String,
    /// 2026-10-02: Arches it applies to.
    pub arch: Vec<String>,
    /// 2026-10-02: Op spellings it matches ([`OpKind::name`]: `linear:q`, `lm_head`, ...).
    pub ops: Vec<String>,
    /// 2026-10-02: The section it matches (`main` | `draft`).
    pub section: Section,
    /// 2026-10-02: Declared (stored) weight formats it matches; empty matches any.
    pub stored: Vec<Format>,
    /// 2026-10-02: Served weight formats it matches; empty matches any.
    pub served: Vec<Format>,
    /// 2026-10-02: Serve settings that must hold (`speculative = "on"`).
    pub when: BTreeMap<String, String>,
    /// 2026-10-02: The copy's byte law.
    pub law: CopyLaw,
    /// 2026-10-02: Copies per matched weight.
    pub count: u64,
    /// 2026-10-02: Nothing reads it and nothing frees it (a loader defect, resident all the same).
    pub leaked: bool,
    /// 2026-10-02: The loader site that allocates it.
    pub site: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct File {
    schema: u32,
    copy: Vec<RuleFile>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RuleFile {
    id: String,
    arch: Vec<String>,
    ops: Vec<String>,
    section: String,
    #[serde(default)]
    stored: Vec<String>,
    #[serde(default)]
    served: Vec<String>,
    #[serde(default)]
    when: BTreeMap<String, String>,
    law: String,
    count: u64,
    #[serde(default)]
    leaked: bool,
    site: String,
}

/// 2026-10-02: Why the copy rules did not load.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("COPIES.toml: {0}")]
pub struct CopyError(pub String);

/// 2026-10-02: The settings a rule's `when` may read: the serve facts the loader branches on.
/// `latent_moe` is `on` when the checkpoint's MoE runs through a latent projection
/// (`moe_latent_size > 0`, Nemotron-3-Super), `off` otherwise. `moe_expert_tables` is the serve's
/// plan-based decision on the MoE's transposed prefill tables (`build` or `skip`).
pub const COPY_SETTINGS: [&str; 6] = [
    "speculative",
    "weight_quantization",
    "lm_head_dtype",
    "expert_quantization",
    "latent_moe",
    "moe_expert_tables",
];

/// 2026-10-02: Parse the rules text.
pub fn parse_copies(text: &str) -> Result<Vec<CopyRule>, CopyError> {
    let file: File = toml::from_str(text).map_err(|e| CopyError(e.to_string()))?;
    if file.schema != 1 {
        return Err(CopyError(format!(
            "schema {} (this build reads 1)",
            file.schema
        )));
    }
    let mut out: Vec<CopyRule> = Vec::with_capacity(file.copy.len());
    for r in file.copy {
        let bad = |what: String| CopyError(format!("copy `{}`: {what}", r.id));
        if out.iter().any(|o| o.id == r.id) {
            return Err(bad("listed twice".into()));
        }
        for op in &r.ops {
            let (name, q) = op
                .split_once(':')
                .map_or((op.as_str(), None), |(a, b)| (a, Some(b)));
            let parsed = match name {
                "linear" => OpKind::parse(name, q, None),
                _ if q.is_some() => return Err(bad(format!("op `{op}` takes no qualifier"))),
                _ => OpKind::parse(name, None, None),
            };
            if !parsed.is_ok_and(|k| k.reads_linear_weight()) {
                return Err(bad(format!("op `{op}` is no weight-reading op")));
            }
        }
        let formats = |v: &[String]| -> Result<Vec<Format>, CopyError> {
            v.iter()
                .map(|f| Format::parse(f).map_err(|e| bad(e.to_string())))
                .collect()
        };
        for k in r.when.keys() {
            if !COPY_SETTINGS.contains(&k.as_str()) {
                return Err(bad(format!(
                    "setting `{k}` is not one of {}",
                    COPY_SETTINGS.join(", ")
                )));
            }
        }
        let section = match r.section.as_str() {
            "main" => Section::Main,
            "draft" => Section::Draft,
            s => return Err(bad(format!("section `{s}` is not main or draft"))),
        };
        if r.count == 0 || r.arch.is_empty() || r.ops.is_empty() || r.site.is_empty() {
            return Err(bad("needs arch, ops, a site and a positive count".into()));
        }
        out.push(CopyRule {
            law: CopyLaw::parse(&r.law).ok_or_else(|| bad(format!("law `{}`", r.law)))?,
            stored: formats(&r.stored)?,
            served: formats(&r.served)?,
            id: r.id,
            arch: r.arch,
            ops: r.ops,
            section,
            when: r.when,
            count: r.count,
            leaked: r.leaked,
            site: r.site,
        });
    }
    Ok(out)
}

impl CopyRule {
    /// 2026-10-02: The rule applies to a node of `op` in `section` of `arch`, stored as `stored`
    /// and served as `served`, under `settings`.
    pub fn matches(
        &self,
        arch: &str,
        op: &OpKind,
        section: Section,
        stored: Format,
        served: Format,
        settings: &BTreeMap<String, String>,
    ) -> bool {
        self.arch.iter().any(|a| a == arch)
            && self.section == section
            && self.ops.iter().any(|o| *o == op.name())
            && (self.stored.is_empty() || self.stored.contains(&stored))
            && (self.served.is_empty() || self.served.contains(&served))
            && self.when.iter().all(|(k, v)| settings.get(k) == Some(v))
    }
}
