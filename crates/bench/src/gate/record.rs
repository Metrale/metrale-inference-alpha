// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The committed shape of one gate run, the baseline shapes it is
//! judged against, and reading both.
//!
//! Owner: bench gate (records).
//! Invariants:
//! - `GateRecord::from_run` never builds a record with an empty `git_sha` or
//!   from a `Running` frame.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::hardware::Hardware;
use crate::history::RunRecord;
use crate::result::{RunStatus, VerdictKind};

pub use super::record_env::resolve_perf_env;
pub use super::record_path::{date_of, record_path, record_path_for, variant_slug};
pub use super::record_summary::now_secs;
use super::record_summary::summarize;
pub use super::record_write::write_record;

/// 2026-09-26: One run record, as committed under `.benchmarks/<id>/`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GateRecord {
    pub schema: u32,
    pub benchmark_id: String,
    pub benchmark_name: String,
    /// 2026-09-26: The commit checked out when the run started. `from_run`
    /// refuses an empty or whitespace-only sha.
    pub git_sha: String,
    /// 2026-09-26: The uncommitted `PERF_PATHS` files present when the run
    /// started ([`super::dirty_perf_paths`]). Empty, and absent from the JSON,
    /// for a clean tree; `check_one` fails a record whose list is not empty.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub dirty_paths: Vec<String>,
    pub recorded_at: u64,
    pub target_model: String,
    /// 2026-09-26: The run record's `params`, rendered into `command` as
    /// `--param KEY=VALUE`.
    pub params: BTreeMap<String, String>,
    /// 2026-09-26: The `met benchmark run` invocation `from_run` rebuilds from
    /// the recorded inputs, ending in `--pull-request-gate`.
    pub command: Vec<String>,
    /// 2026-09-26: The recipe (`<family>/<stem>`) that served this run when the
    /// gate started its own server; `None` for an endpoint the operator ran.
    /// When set, `command` carries no `--url` or `--model`, because that URL
    /// named a port chosen for the run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub served_by: Option<String>,
    /// 2026-09-26: Recipe keys changed for this run: the baseline entry's
    /// `serve_overrides` merged with the operator's `--serve-override`, copied
    /// from the run record. Empty, and absent from the JSON, means the recipe
    /// served as written; otherwise `served_by` alone does not describe the
    /// server's config.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub serve_overrides: BTreeMap<String, String>,

    pub metrale_version: String,
    /// 2026-09-26: The fingerprint the serving endpoint's `/hardware` returned.
    pub hardware: Hardware,
    /// 2026-09-26: The box's state before and after the run, with the delta
    /// and the pre- and post-check verdicts, carried from the terminal frame.
    /// Absent when the frame captured none: absent means unmeasured, not
    /// healthy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hardware_state: Option<crate::hardware::HardwareStateReport>,
    /// 2026-09-26: The terminal frame's metrics, by name.
    pub metrics: BTreeMap<String, f64>,
    /// 2026-09-26: Content identity of the dataset the run scored against,
    /// from the terminal frame (the MLPerf agentic leg writes
    /// `file-sha256:…;draw-sha256:…`). Absent when the frame carries none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dataset_fingerprint: Option<String>,
    /// 2026-09-26: The run's terminal status. `check_one` fails a record whose
    /// status is `Failed`, whatever its metrics.
    pub frame_status: RunStatus,
    /// 2026-09-26: `PASS`, `FAIL` or `info` from the frame's verdict kind;
    /// `None` when the frame has no verdict. Its reason is `verdict_reason`.
    pub verdict: Option<String>,
    pub verdict_reason: String,
    /// 2026-09-26: The one line `record_summary::summarize` builds: model,
    /// metrics, verdict, and the first warning or error logged.
    pub summary: String,
    /// 2026-09-26: The performance controls in `record_env`'s `PERF_CONTROLS`,
    /// resolved: an unset or empty variable is recorded as its default.
    /// `from_run` reads this process's environment, which an in-process serve
    /// shares; `with_serve_env` re-resolves it through the levers applied to
    /// the server. Disclosure only: `check_record` does not read it.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub perf_env: BTreeMap<String, String>,
    /// 2026-09-26: The serve settings the gate resolved from its recipe; the
    /// keys and rules are in [`super::record_serve`]. Empty, and absent from
    /// the JSON, for a run against an operator's own endpoint. Disclosure
    /// only: `check_record` does not read it.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub serve_resolved: BTreeMap<String, String>,
    /// 2026-09-26: The `METRALE_*` levers the gate applied to the server it
    /// measured: the recipe's `env:` block with the entry's
    /// `[benchmarks.serve_env]` on top (`serve_env::Reconciled::env`). Empty,
    /// and absent from the JSON, for an operator's own endpoint or a recipe
    /// that declares none. Disclosure only: `check_record` does not read it.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub serve_env: BTreeMap<String, String>,
    /// 2026-09-26: What each kernel target compiled to when this was measured.
    /// Lets a later `kernels/`-only diff keep this record for the targets whose
    /// device code did not change; see [`super::closure`]. Empty, and absent
    /// from the JSON, excuses no diff.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub closure: super::closure::Attestation,
}

/// 2026-09-26: One metric's threshold: `min` alone is a floor, `max` alone a
/// ceiling, both a range, and equal values an exact pin (`scoring::compare`).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Bound {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max: Option<f64>,
    /// 2026-09-26: Slack allowed beyond the bound, in the metric's units; 0
    /// when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub noise: Option<f64>,
}

/// 2026-09-26: One (hardware, model) pair's thresholds, and the recipe that
/// serves the model.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ModelBaseline {
    /// 2026-09-26: The recipe that serves this model, as `<family>/<stem>`
    /// (e.g. `qwen3.6/qwen3.6-27b-nvfp4-unsloth`). `None` means the gate cannot
    /// start its own server and must be given a live endpoint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recipe: Option<String>,
    /// 2026-09-26: Display name for this variant in the TUI's variant list;
    /// empty shows the checkpoint id.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub label: String,
    /// 2026-09-26: The BENCH.toml entry's free-text note.
    #[serde(default)]
    pub note: String,
    #[serde(default)]
    pub metrics: BTreeMap<String, Bound>,
    /// 2026-09-26: Recipe keys self-start applies for this gate, as
    /// `--serve-override KEY=VALUE` strings. Empty, and omitted, means the
    /// recipe serves as written. `port` is refused at parse; see
    /// `bench::BenchEntry`.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub serve_overrides: BTreeMap<String, String>,
    /// 2026-09-26: Benchmark parameter values the gate pins for this entry.
    /// Empty, and omitted, means the schema defaults. Each value is parsed by
    /// its parameter's `ParamKind::parse`; an explicit `--param` wins, and
    /// `check_record` requires every pin on the record.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub param_overrides: BTreeMap<String, String>,
    /// 2026-09-26: `METRALE_*` serve levers the gate pins for this entry,
    /// applied on top of the recipe's own `env:` block. Validated by
    /// `serve_env::declared` at parse and disclosed on the record as
    /// `GateRecord::serve_env`; `check_record` does not read them.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub serve_env: BTreeMap<String, String>,
}

/// 2026-09-26: Baseline pins first; the operator's `--serve-override` wins on
/// a clash.
pub fn merge_serve_overrides(
    baseline: BTreeMap<String, String>,
    requested: BTreeMap<String, String>,
) -> BTreeMap<String, String> {
    let mut out = baseline;
    out.extend(requested);
    out
}

/// 2026-09-26: Every measured model on one box class, and which one to serve
/// by default.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct HardwareBaseline {
    /// 2026-09-26: The model to use when the caller does not name one: the
    /// BENCH.toml entry marked `default = true`. Two defaults on one box class
    /// are refused when the baseline is assembled.
    pub default: String,
    #[serde(default)]
    pub models: BTreeMap<String, ModelBaseline>,
}

/// 2026-09-26: The thresholds a benchmark's gate records must meet, assembled
/// by [`super::bench`] from the measured entries in every
/// `kernels/<hw>/<model>/BENCH.toml`. Keyed hardware, then model, and
/// `check_record` scores a record only against its own pair. The records
/// themselves live under `.benchmarks/<id>/`.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GateBaseline {
    /// 2026-09-26: Schema version; an assembled baseline is 2.
    #[serde(default)]
    pub schema: u32,
    pub hardware: BTreeMap<String, HardwareBaseline>,
}

impl GateBaseline {
    /// 2026-09-26: Resolve one (hardware, model) entry. `model: None` takes the
    /// hardware's declared default. An unknown hardware or model is an error
    /// naming what was asked for and what exists.
    pub fn resolve(&self, hardware: &str, model: Option<&str>) -> Result<(String, &ModelBaseline)> {
        let hw = self.hardware.get(hardware).ok_or_else(|| {
            anyhow::anyhow!(
                "no baseline for hardware {hardware:?}; this benchmark has entries for [{}]",
                self.hardware.keys().cloned().collect::<Vec<_>>().join(", ")
            )
        })?;
        let want = model.unwrap_or(&hw.default);
        let entry = hw.models.get(want).ok_or_else(|| {
            anyhow::anyhow!(
                "no baseline for model {want:?} on {hardware:?}; it has [{}]",
                hw.models.keys().cloned().collect::<Vec<_>>().join(", ")
            )
        })?;
        Ok((want.to_string(), entry))
    }
}

/// 2026-09-26: Read and parse one record file.
pub fn read_record(path: &Path) -> Result<GateRecord> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))
}

/// 2026-09-26: The benchmark's baseline, assembled from the BENCH.toml files.
pub fn read_baseline(root: &Path, benchmark_id: &str) -> Result<GateBaseline> {
    super::bench::baseline_for(root, benchmark_id)
}

impl GateRecord {
    /// 2026-09-26: `(index, count)` from the `shard.index` / `shard.count`
    /// metrics, or `None` for a whole-draw run or for values that are not
    /// integers with `1 <= count` and `index < count`. `record_path_for`
    /// copies it into the file name.
    pub fn shard(&self) -> Option<(usize, usize)> {
        let index = *self.metrics.get("shard.index")?;
        let count = *self.metrics.get("shard.count")?;
        if index.fract() != 0.0 || count.fract() != 0.0 || count < 1.0 || index >= count {
            return None;
        }
        Some((index as usize, count as usize))
    }

    /// 2026-09-26: Build a gate record from a finished run. `hardware` is the
    /// serving endpoint's `/hardware` fingerprint. With `served_by` set,
    /// `command` omits `--url` and `--model`: replaying asks for the benchmark
    /// again and the recipe supplies the endpoint. `dirty_paths` is the
    /// [`super::dirty_perf_paths`] result from when the run started.
    ///
    /// Errors on an empty `git_sha` or a `Running` frame.
    pub fn from_run(
        record: &RunRecord,
        hardware: Hardware,
        git_sha: String,
        dirty_paths: Vec<String>,
        served_by: Option<String>,
    ) -> Result<Self> {
        // 2026-09-26: Read off the run record, so the run record and the gate
        // record cannot disagree about the serve overrides.
        let serve_overrides = record.serve_overrides.clone();
        if git_sha.trim().is_empty() {
            bail!("a gate record needs the commit sha it was measured from");
        }
        let frame = &record.frame;
        if frame.status == RunStatus::Running {
            bail!("the run never reached a terminal frame — nothing to gate");
        }
        let mut params = Vec::new();
        if served_by.is_none() {
            if !record.target_url.is_empty() {
                params.push(("--url".to_string(), record.target_url.clone()));
            }
            if !record.target_model.is_empty() {
                params.push(("--model".to_string(), record.target_model.clone()));
            }
        }
        for (k, v) in &record.params {
            params.push(("--param".to_string(), format!("{k}={v}")));
        }
        // 2026-09-26: The overrides go into `command` too, so replaying it
        // serves the same config.
        for (k, v) in &serve_overrides {
            params.push(("--serve-override".to_string(), format!("{k}={v}")));
        }
        if record.benchmark_id == "agentic-webserver" {
            params.push(("--yes".to_string(), String::new()));
        }
        let mut command: Vec<String> = vec![
            "met".into(),
            "benchmark".into(),
            "run".into(),
            record.benchmark_id.clone(),
        ];
        for (flag, value) in &params {
            command.push(flag.clone());
            if !value.is_empty() {
                command.push(value.clone());
            }
        }
        command.push("--pull-request-gate".into());

        let verdict = frame.verdict.as_ref().map(|v| match v.kind {
            VerdictKind::Pass => "PASS".to_string(),
            VerdictKind::Fail => "FAIL".to_string(),
            VerdictKind::Info => "info".to_string(),
        });
        let verdict_reason = frame
            .verdict
            .as_ref()
            .map(|v| v.reason.clone())
            .unwrap_or_default();
        Ok(Self {
            schema: 1,
            // 2026-09-26: Attached afterwards by `with_closure`. Left empty it
            // excuses no later diff (`closure::excuses`), so a caller that
            // skips it costs re-runs, not soundness.
            closure: Default::default(),
            benchmark_id: record.benchmark_id.clone(),
            benchmark_name: record.benchmark_name.clone(),
            git_sha,
            dirty_paths,
            recorded_at: record.recorded_at,
            target_model: record.target_model.clone(),
            params: record.params.clone(),
            command,
            served_by,
            serve_overrides,
            metrale_version: record.metrale_version.clone(),
            hardware,
            // 2026-09-26: Taken from the terminal frame rather than probed
            // here: this runs after the benchmark finished.
            hardware_state: frame.hardware_state.clone(),
            metrics: frame.metrics.clone(),
            dataset_fingerprint: frame.dataset_fingerprint.clone(),
            frame_status: frame.status,
            verdict,
            verdict_reason,
            summary: summarize(record),
            // 2026-09-26: This process's environment; `with_serve_env`
            // re-resolves it through the levers applied to the server.
            perf_env: resolve_perf_env(|k| std::env::var(k).ok()),
            // 2026-09-26: Attached afterwards by `with_serve_resolved` and
            // `with_serve_env`.
            serve_resolved: BTreeMap::new(),
            serve_env: BTreeMap::new(),
        })
    }

    /// 2026-09-26: Attach what each kernel target in the measuring binary
    /// compiled from. `baked` is `metrale_kernels::TARGET_CLOSURES`, which the
    /// kernels build script emits, so it describes the built sources rather
    /// than the working tree. Unparseable input attaches an empty attestation,
    /// which excuses no later diff.
    #[must_use]
    pub fn with_closure(mut self, baked: &str) -> Self {
        self.closure = serde_json::from_str(baked).unwrap_or_default();
        self
    }

    /// 2026-09-26: True only when `verdict` is `PASS`; FAIL, info and no
    /// verdict are false.
    pub fn verdict_passes(&self) -> bool {
        self.verdict.as_deref() == Some("PASS")
    }

    /// 2026-09-26: True when the frame status is `Failed`.
    pub fn frame_status_failed(&self) -> bool {
        self.frame_status == RunStatus::Failed
    }
}
