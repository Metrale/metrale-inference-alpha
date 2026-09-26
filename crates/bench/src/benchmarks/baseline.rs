// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Per-benchmark baselines: the metrics of an earlier run, stored as
//! JSON in the artifact store's `runs/<benchmark_id>/` directory, for the
//! benchmarks that compare a run against their own history (`ttft`,
//! `serve_matrix`). Each file records the endpoint and model it was measured
//! against.
//!
//! Owner: bench.
//! Invariants: `load_for` with a model named never returns a baseline whose
//! `model` is a different one.

use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::artifacts::ArtifactStore;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Baseline {
    /// 2026-09-26: Unix seconds; `age_text` shows it.
    pub recorded_at: u64,
    /// 2026-09-26: The endpoint and model the numbers were measured against.
    /// `ttft_verdict` reports a baseline from another box or model instead of
    /// comparing against it.
    pub target: String,
    pub model: String,
    pub metrics: BTreeMap<String, f64>,
}

impl Baseline {
    pub fn get(&self, key: &str) -> Option<f64> {
        self.metrics.get(key).copied()
    }

    /// 2026-09-26: Age as text: `just now`, `N min old`, `N h old` or `N d old`.
    pub fn age_text(&self) -> String {
        let now = now_secs();
        let secs = now.saturating_sub(self.recorded_at);
        match secs {
            0..=90 => "just now".into(),
            91..=5400 => format!("{} min old", secs / 60),
            5401..=172_800 => format!("{} h old", secs / 3600),
            _ => format!("{} d old", secs / 86_400),
        }
    }
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// 2026-09-26: Where one model's baseline for `benchmark_id` lives:
/// `baseline-<variant_slug>.json`, or `baseline.json` when no model is named.
/// Keyed by model because one gate can serve several checkpoints, and with one
/// shared file each variant's run would replace the other's baseline.
fn path(
    store: &ArtifactStore,
    benchmark_id: &str,
    model: Option<&str>,
) -> Result<std::path::PathBuf> {
    let name = match model {
        Some(m) => format!("baseline-{}.json", crate::gate::record::variant_slug(m)),
        None => "baseline.json".to_string(),
    };
    Ok(store.runs_dir(benchmark_id)?.join(name))
}

/// 2026-09-26: The stored baseline, if any. An unreadable or corrupt file reads
/// as absent: a run without a baseline still works, a refusal to start would
/// not.
pub fn load(store: &ArtifactStore, benchmark_id: &str) -> Option<Baseline> {
    // 2026-09-26: `baseline.json` first, else the model-keyed file if exactly
    // one exists. With two or more, none is returned rather than a guess.
    if let Ok(p) = path(store, benchmark_id, None)
        && let Ok(text) = std::fs::read_to_string(p)
        && let Ok(b) = serde_json::from_str::<Baseline>(&text)
    {
        return Some(b);
    }
    let dir = store.runs_dir(benchmark_id).ok()?;
    let mut candidate = None;
    for e in std::fs::read_dir(dir).ok()? {
        let path = e.ok()?.path();
        let name = path.file_name()?.to_string_lossy().to_string();
        if !(name.starts_with("baseline-") && name.ends_with(".json")) {
            continue;
        }
        if candidate.is_some() {
            return None;
        }
        candidate = Some(path);
    }
    let text = std::fs::read_to_string(candidate?).ok()?;
    serde_json::from_str(&text).ok()
}

/// 2026-09-26: The baseline for one model: its model-keyed file, else
/// `baseline.json` if that records the same model. With no model named,
/// `baseline.json` alone.
pub fn load_for(
    store: &ArtifactStore,
    benchmark_id: &str,
    model: Option<&str>,
) -> Option<Baseline> {
    if let Some(m) = model
        && let Ok(p) = path(store, benchmark_id, Some(m))
        && let Ok(text) = std::fs::read_to_string(p)
        && let Ok(b) = serde_json::from_str::<Baseline>(&text)
        && b.model == m
    {
        return Some(b);
    }
    let p = path(store, benchmark_id, None).ok()?;
    let text = std::fs::read_to_string(p).ok()?;
    let b: Baseline = serde_json::from_str(&text).ok()?;
    match model {
        Some(m) if b.model != m => None,
        _ => Some(b),
    }
}

/// 2026-09-26: Writes the model-keyed baseline, replacing any earlier one. Call
/// only after a run whose numbers should become the bar: a failed or partial
/// leg stored here is compared against by every later run.
pub fn save(
    store: &ArtifactStore,
    benchmark_id: &str,
    target: &str,
    model: &str,
    metrics: BTreeMap<String, f64>,
) -> Result<()> {
    let baseline = Baseline {
        recorded_at: now_secs(),
        target: target.to_string(),
        model: model.to_string(),
        metrics,
    };
    let p = path(store, benchmark_id, Some(model))?;
    std::fs::write(&p, serde_json::to_string_pretty(&baseline)?)
        .with_context(|| format!("writing {}", p.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store(name: &str) -> ArtifactStore {
        let d =
            std::env::temp_dir().join(format!("metrale-baseline-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        ArtifactStore::with_root(d)
    }

    /// 2026-09-26: Two checkpoints of one gate keep separate baselines.
    #[test]
    fn two_models_of_one_gate_keep_separate_baselines() {
        let s = store("variants");
        let mut a = BTreeMap::new();
        a.insert("median_ms".into(), 1677.9);
        save(&s, "ttft-cold", "http://h:1", "Qwen/Qwen3.6-35B-A3B-FP8", a).unwrap();
        let mut b = BTreeMap::new();
        b.insert("median_ms".into(), 442.3);
        save(
            &s,
            "ttft-cold",
            "http://h:1",
            "nvidia/Qwen3.6-35B-A3B-NVFP4",
            b,
        )
        .unwrap();
        let fp8 = load_for(&s, "ttft-cold", Some("Qwen/Qwen3.6-35B-A3B-FP8")).unwrap();
        assert_eq!(fp8.get("median_ms"), Some(1677.9));
        let nv = load_for(&s, "ttft-cold", Some("nvidia/Qwen3.6-35B-A3B-NVFP4")).unwrap();
        assert_eq!(nv.get("median_ms"), Some(442.3));
    }

    /// 2026-09-26: A model with no baseline of its own reads as absent rather
    /// than inheriting `baseline.json` from a different checkpoint.
    #[test]
    fn legacy_shared_baseline_is_not_inherited_by_another_model() {
        let s = store("legacy");
        let p = s.runs_dir("ttft-warm").unwrap().join("baseline.json");
        let legacy = Baseline {
            recorded_at: now_secs(),
            target: "http://h:1".into(),
            model: "Qwen/Qwen3.6-35B-A3B-FP8".into(),
            metrics: BTreeMap::from([("median_ms".to_string(), 1600.0)]),
        };
        std::fs::write(&p, serde_json::to_string(&legacy).unwrap()).unwrap();
        assert!(load_for(&s, "ttft-warm", Some("Qwen/Qwen3.6-35B-A3B-FP8")).is_some());
        assert!(load_for(&s, "ttft-warm", Some("nvidia/Qwen3.6-35B-A3B-NVFP4")).is_none());
    }

    #[test]
    fn round_trips_and_reports_the_target_it_came_from() {
        let s = store("rt");
        assert!(load(&s, "ttft-warm").is_none());
        let mut m = BTreeMap::new();
        m.insert("median_ms".into(), 812.5);
        save(&s, "ttft-warm", "http://127.0.0.1:8888", "qwen", m).unwrap();
        let b = load(&s, "ttft-warm").unwrap();
        assert_eq!(b.get("median_ms"), Some(812.5));
        assert_eq!(b.target, "http://127.0.0.1:8888");
        assert_eq!(b.model, "qwen");
        assert_eq!(b.age_text(), "just now");
    }

    #[test]
    fn a_model_keyed_file_must_belong_to_the_model_in_its_name() {
        let s = store("misfiled");
        let p = path(&s, "ttft-warm", Some("wanted-model")).unwrap();
        let wrong = Baseline {
            recorded_at: now_secs(),
            target: "http://h:1".into(),
            model: "another-model".into(),
            metrics: BTreeMap::from([("median_ms".to_string(), 1.0)]),
        };
        std::fs::write(p, serde_json::to_string(&wrong).unwrap()).unwrap();
        assert!(load_for(&s, "ttft-warm", Some("wanted-model")).is_none());
    }

    #[test]
    fn an_unkeyed_load_never_guesses_among_multiple_model_files() {
        let s = store("ambiguous");
        let corrupt = path(&s, "ttft-warm", Some("created-first")).unwrap();
        std::fs::write(corrupt, "{ not json").unwrap();
        save(
            &s,
            "ttft-warm",
            "http://h:1",
            "created-second",
            BTreeMap::from([("median_ms".to_string(), 1.0)]),
        )
        .unwrap();
        assert!(
            load(&s, "ttft-warm").is_none(),
            "two model-keyed files are ambiguous even when only one parses"
        );
    }

    #[test]
    fn a_corrupt_baseline_reads_as_absent() {
        let s = store("corrupt");
        let p = s.runs_dir("x").unwrap().join("baseline.json");
        std::fs::write(p, "{ not json").unwrap();
        assert!(load(&s, "x").is_none());
    }
}
