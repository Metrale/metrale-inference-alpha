// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The shapes `metralectl bench … --json` writes, as this side reads them.
//!
//! Owner: server CLI (`met benchmark certify`).
//! Every `NodeInfo` field has a serde default, so a report that adds or drops
//! a field still parses; `node::admit` decides which absent facts refuse a node.
//! Invariants: none beyond the types.

use std::path::PathBuf;

use serde::Deserialize;

/// 2026-09-26: metralectl's exit codes 0-8, by name (`from_code`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Exit {
    Done,
    Usage,
    Unreachable,
    NotPaired,
    Refused,
    JobFailed,
    Cancelled,
    StreamLost,
    Unsupported,
    /// 2026-09-26: Killed by a signal (`None`), or a code this build does not know.
    Other(Option<i32>),
}

impl Exit {
    pub fn from_code(code: Option<i32>) -> Self {
        match code {
            Some(0) => Self::Done,
            Some(1) => Self::Usage,
            Some(2) => Self::Unreachable,
            Some(3) => Self::NotPaired,
            Some(4) => Self::Refused,
            Some(5) => Self::JobFailed,
            Some(6) => Self::Cancelled,
            Some(7) => Self::StreamLost,
            Some(8) => Self::Unsupported,
            other => Self::Other(other),
        }
    }
}

/// 2026-09-26: metralectl's error document.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Default)]
pub struct ErrorObj {
    pub code: String,
    pub message: String,
    #[serde(default)]
    pub node: Option<String>,
    #[serde(default)]
    pub node_id: Option<String>,
    #[serde(default)]
    pub fix: Option<String>,
    #[serde(default)]
    pub retryable: bool,
}

impl std::fmt::Display for ErrorObj {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.code, self.message)?;
        if let Some(fix) = &self.fix {
            write!(f, " (fix: {fix})")?;
        }
        Ok(())
    }
}

/// 2026-09-26: A `{"state":"reading","value":x}` / `{"state":"unsupported"}` metric;
/// absent means `Unsupported`.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Default)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum Metric {
    Reading {
        value: f64,
    },
    #[default]
    Unsupported,
}

impl Metric {
    pub fn value(self) -> Option<f64> {
        match self {
            Self::Reading { value } => Some(value),
            Self::Unsupported => None,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Default)]
pub struct GpuInfo {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub count: u32,
    #[serde(default)]
    pub driver_version: String,
    #[serde(default)]
    pub cuda_version: String,
    #[serde(default)]
    pub sm_clock_mhz: Metric,
    #[serde(default)]
    pub temperature_c: Metric,
    #[serde(default)]
    pub memory_total_bytes: Metric,
    #[serde(default)]
    pub memory_used_frac: Metric,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Default)]
pub struct HostThermal {
    #[serde(default)]
    pub chassis_temps_c: Vec<f64>,
    #[serde(default)]
    pub throttle_thermal: Option<bool>,
    #[serde(default)]
    pub sm_clock_max_mhz: Option<f64>,
    #[serde(default)]
    pub mem_total_kb: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Default)]
pub struct RepoInfo {
    #[serde(default)]
    pub path: String,
    #[serde(default)]
    pub remote_name: String,
    #[serde(default)]
    pub remote_url: String,
    #[serde(default)]
    pub head_sha: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Default)]
pub struct BuiltSha {
    pub sha: String,
}

/// 2026-09-26: What a node says about itself. Every field has a serde default; see
/// `node::admit` for which absent facts refuse the node.
#[derive(Clone, Debug, Deserialize, PartialEq, Default)]
pub struct NodeInfo {
    #[serde(default)]
    pub node: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub agent_version: String,
    #[serde(default)]
    pub bench_enabled: bool,
    #[serde(default)]
    pub disabled_reason: Option<String>,
    #[serde(default)]
    pub gpu: Option<GpuInfo>,
    #[serde(default)]
    pub thermal: Option<HostThermal>,
    #[serde(default)]
    pub hardware_class: Option<String>,
    #[serde(default)]
    pub metrale_repo: Option<RepoInfo>,
    #[serde(default)]
    pub metrale_home: Option<String>,
    #[serde(default)]
    pub signer_fp: Option<String>,
    #[serde(default)]
    pub built_shas: Vec<BuiltSha>,
    #[serde(default)]
    pub busy: bool,
    #[serde(default)]
    pub busy_reason: Option<String>,
    #[serde(default)]
    pub queued: u32,
    #[serde(default)]
    pub queue_depth: u32,
    #[serde(default)]
    pub host_free_fraction: Metric,
    #[serde(default)]
    pub disk_free_bytes: Metric,
    #[serde(default)]
    pub min_free_disk_bytes: u64,
    #[serde(default)]
    pub max_run_s: u32,
}

/// 2026-09-26: One row of `bench nodes --json`.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct NodeRow {
    pub node: String,
    pub ok: bool,
    #[serde(default)]
    pub info: Option<NodeInfo>,
    #[serde(default)]
    pub error: Option<ErrorObj>,
}

/// 2026-09-26: What `submit` sends.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SubmitSpec {
    pub job_key: String,
    pub sha: String,
    pub gate: String,
    /// 2026-09-26: `k=v` pairs, each passed to `metralectl bench submit` as `--param`;
    /// a shard's `shard=i/n`.
    pub params: Vec<String>,
    pub hardware: String,
    pub max_run_s: Option<u32>,
    pub note: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub struct Submitted {
    pub node_id: String,
    pub job_id: String,
    pub existing: bool,
}

/// 2026-09-26: One line of an attached stream. Only the fields the driver reads are
/// named; serde ignores the rest.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct StreamEvent {
    pub seq: u64,
    pub kind: String,
    #[serde(default)]
    pub phase: Option<String>,
    #[serde(default)]
    pub detail: Option<String>,
    #[serde(default)]
    pub lines: Vec<String>,
    #[serde(default)]
    pub outcome: Option<String>,
    #[serde(default)]
    pub exit_code: Option<i32>,
    #[serde(default)]
    pub reason: Option<String>,
    #[serde(default)]
    pub stage: Option<String>,
    #[serde(default)]
    pub verdict: Option<serde_json::Value>,
    #[serde(default)]
    pub cached: Option<bool>,
}

/// 2026-09-26: How an attach ended.
#[derive(Clone, Debug, PartialEq)]
pub enum AttachEnd {
    /// 2026-09-26: Exit 0: the job passed. `exit_code` is the `done` event's.
    Passed { exit_code: Option<i32> },
    /// 2026-09-26: Exit 5: the job ended without a pass; the `done` event says how.
    JobFailed {
        outcome: Option<String>,
        exit_code: Option<i32>,
        detail: String,
    },
    /// 2026-09-26: Exit 6, or the driver's cancel flag killed the attach.
    Cancelled,
    /// 2026-09-26: Exit 7: the stream was lost; `last_seq` is the highest seq seen (or
    /// `from_seq - 1`).
    StreamLost { last_seq: u64 },
    /// 2026-09-26: Any other exit, with the error document if the last line was one.
    Failed { exit: Exit, error: Option<ErrorObj> },
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub struct FetchedFile {
    pub name: String,
    pub relative_path: String,
    pub path: PathBuf,
    pub bytes: u64,
    pub sha256: String,
}
