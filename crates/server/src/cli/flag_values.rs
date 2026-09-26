// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The closed value sets of the enumerated `met serve` string
//! flags, and the parsers for `--kv-high-precision-layers` and the tristate
//! flags.
//!
//! `validate_serve_args` refuses a value outside a set (`check_enum`), and
//! [`options_for_flag`] offers the sets to the TUI option picker and the CLI
//! manifest, so what is offered and what is accepted come from one list.
//!
//! `--kv-cache-dtype` has no const here: its list is
//! `metrale_cache::kv_cache::KvCacheDtype::ALL`.
//!
//! The sets are not wired into clap as `PossibleValuesParser`s:
//! `validate_serve_args` reports every violation at once, where clap exits on
//! the first.
//!
//! Owner: server CLI (`met serve`).
//! Invariants: none beyond the types.

/// 2026-09-26: What `--kv-high-precision-layers auto` resolves to. The flag's
/// help text states the same number as "recommended".
pub(crate) const AUTO_KV_HIGH_PRECISION_LAYERS: usize = 2;

/// 2026-09-26: The accepted forms of `--kv-high-precision-layers`: a count or
/// a keyword, so not a `check_enum` list. `validate_serve_args` and
/// `serve_phases::kv_cache` both parse through this type, so they accept the
/// same forms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum KvHighPrecisionLayers {
    /// 2026-09-26: `max` / `all`: every attention layer stays BF16.
    All,
    /// 2026-09-26: `auto`: [`AUTO_KV_HIGH_PRECISION_LAYERS`].
    Auto,
    /// 2026-09-26: An explicit first-N-and-last-N count. `0`, the flag's
    /// default, defers to `auto_high_precision_layers` for the KV dtype.
    Count(usize),
}

impl std::str::FromStr for KvHighPrecisionLayers {
    type Err = String;

    fn from_str(raw: &str) -> Result<Self, String> {
        match raw.trim().to_lowercase().as_str() {
            "max" | "all" => Ok(Self::All),
            "auto" => Ok(Self::Auto),
            s => s.parse::<usize>().map(Self::Count).map_err(|_| {
                "expected a whole number of layers, or one of: auto, max, all".to_string()
            }),
        }
    }
}

impl KvHighPrecisionLayers {
    pub(crate) fn resolve(self, num_attn_layers: usize) -> usize {
        match self {
            Self::All => num_attn_layers,
            Self::Auto => AUTO_KV_HIGH_PRECISION_LAYERS,
            Self::Count(n) => n,
        }
    }
}

/// 2026-09-26: The value set of a lever whose default is decided off the
/// command line and which the command line may pin either way. `auto` pins
/// nothing. A presence flag could move it in one direction only.
pub(crate) const TRISTATES: &[&str] = &["auto", "on", "off"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Tristate {
    Auto,
    On,
    Off,
}

impl std::str::FromStr for Tristate {
    type Err = String;

    fn from_str(raw: &str) -> Result<Self, String> {
        match raw {
            "auto" => Ok(Self::Auto),
            "on" => Ok(Self::On),
            "off" => Ok(Self::Off),
            other => Err(format!("expected one of auto, on, off; got {other:?}")),
        }
    }
}

impl Tristate {
    /// 2026-09-26: Parse a value `validate_serve_args` has already accepted;
    /// panics on any other.
    pub(crate) fn validated(raw: &str) -> Self {
        raw.parse().expect("validated by validate_serve_args")
    }

    /// 2026-09-26: What the command line pins: `None` for `auto`, which leaves
    /// the lever's own default in charge.
    pub(crate) fn pinned(self) -> Option<bool> {
        match self {
            Self::Auto => None,
            Self::On => Some(true),
            Self::Off => Some(false),
        }
    }
}

pub(crate) const LM_HEAD_DTYPES: &[&str] = &["default", "bf16", "nvfp4", "fp8"];
pub(crate) const MTP_QUANTS: &[&str] = &["bf16", "fp8", "nvfp4"];
pub(crate) const SCHEDULERS: &[&str] = &["fifo", "slai"];
/// 2026-09-26: The device routers: `sync` settles every device step before
/// the next host decision; `async` lets one plain decode step run ahead of the
/// host, and falls back to `sync` with a warning at startup when the router
/// cannot be built, as on a model without a device token feed.
pub(crate) const SCHEDULER_CONFIGS: &[&str] = &["sync", "async"];
/// 2026-09-26: `--telemetry`: the telemetry crate's own level names.
pub(crate) const TELEMETRY_LEVELS: &[&str] = &metrale_telemetry::Level::NAMES;
pub(crate) const SSM_H_DTYPES: &[&str] = &["f32", "f16", "f16-pool"];
pub(crate) const MTP_GATES: &[&str] = &["auto", "force"];
pub(crate) const TOOL_CALL_PARSERS: &[&str] = &[
    "hermes",
    "qwen3_coder",
    "qwen3_xml",
    "gemma4",
    "mistral",
    "minimax_xml",
    "bare_json",
    "poolside_v1",
];

/// 2026-09-26: The closed value set for a `met serve` flag, by its long name,
/// or `None` for a free-form flag. For `--kv-cache-dtype` it lists each
/// dtype's canonical name only; parse aliases such as `fp8k2v` for
/// `fp8k_turbo2v` are not offered.
pub(crate) fn options_for_flag(flag: &str) -> Option<Vec<String>> {
    let owned = |list: &[&str]| list.iter().map(|s| s.to_string()).collect();
    match flag {
        "lm-head-dtype" => Some(owned(LM_HEAD_DTYPES)),
        "mtp-quantization" => Some(owned(MTP_QUANTS)),
        "scheduler" => Some(owned(SCHEDULERS)),
        "scheduler-config" => Some(owned(SCHEDULER_CONFIGS)),
        "ssm-h-dtype" => Some(owned(SSM_H_DTYPES)),
        "telemetry" => Some(owned(TELEMETRY_LEVELS)),
        "mtp-gate" => Some(owned(MTP_GATES)),
        "tool-call-parser" => Some(owned(TOOL_CALL_PARSERS)),
        "ssm-batched-recurrent" | "content-loop-watchdog" | "tool-grammar" => {
            Some(owned(TRISTATES))
        }
        "kv-cache-dtype" => Some(
            metrale_cache::kv_cache::KvCacheDtype::ALL
                .iter()
                .map(|d| d.name().to_string())
                .collect(),
        ),
        _ => None,
    }
}

#[cfg(test)]
#[path = "flag_values_tests.rs"]
mod tests;
