// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The serve matrix's parameter schema.
//!
//! Owner: bench, serve matrix.
//! Invariants:
//! - The parameter defaults are the `ParamSpec` defaults here;
//!   `ParamValues::defaults` derives a run's starting values from them.

use crate::params::{ParamKind, ParamSpec, ParamValue};

/// 2026-09-26: Every serve-matrix parameter.
pub fn specs() -> Vec<ParamSpec> {
    vec![
        ParamSpec::new(
            "include",
            "Model filter",
            "Case-insensitive substring of the HF id. `all` runs every checkpoint the box can serve.",
            ParamKind::Text,
            ParamValue::Text("all".into()),
        ),
        ParamSpec::new(
            "max_seq_len",
            "Max sequence length",
            "Context each round is served with. Must fit the long-context probe plus its output.",
            ParamKind::Int {
                min: 2048,
                max: 262_144,
            },
            ParamValue::Int(32_768),
        ),
        ParamSpec::new(
            "long_ctx_tokens",
            "Long-context probe",
            "Prompt size for the needle-in-a-haystack recall probe. 0 turns it off.",
            ParamKind::Int {
                min: 0,
                max: 131_072,
            },
            ParamValue::Int(16_384),
        ),
        ParamSpec::new(
            "tps_tokens",
            "Throughput budget",
            "Output tokens the throughput probe asks for. Too few and the reply arrives in one SSE delta, leaving decode unmeasurable.",
            ParamKind::Int { min: 16, max: 4096 },
            ParamValue::Int(256),
        ),
        ParamSpec::new(
            "probe_budget",
            "Probe output tokens",
            "Output budget for the codegen and tool-call probes.",
            ParamKind::Int { min: 32, max: 4096 },
            ParamValue::Int(512),
        ),
        ParamSpec::new(
            "speculative",
            "Speculative decoding",
            "Serve each round with MTP on. Off by default: a checkpoint with no MTP head falls back to single-token decode and reports the baseline's numbers under a +MTP label.",
            ParamKind::Bool,
            ParamValue::Bool(false),
        ),
        ParamSpec::new(
            "request_timeout_s",
            "Request timeout",
            "Seconds before a single probe request is abandoned.",
            ParamKind::Int { min: 10, max: 3600 },
            ParamValue::Int(300),
        ),
        ParamSpec::new(
            "update_baselines",
            "Update baselines",
            "Record this run's tok/s as the new bar instead of gating against it. Deliberate refresh only — review the numbers first.",
            ParamKind::Bool,
            ParamValue::Bool(false),
        ),
    ]
}
