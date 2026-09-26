// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Resolve a model or LoRA adapter specifier (a local directory or a Hub id) to a directory on disk, reading the local Hub cache.
//!
//! Owner: server.
//! Invariants: none beyond the types.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

/// 2026-09-26: The command that fetches `id` into the cache this resolver
/// reads, written once for every "not found" error. It names `hf download` and
/// also the older `huggingface-cli`.
fn download_hint(id: &str) -> String {
    format!(
        "  hf download {id}\n\
         (`hf` is the huggingface_hub CLI; before 1.0 it was `huggingface-cli`. \
         The dashboard's Library downloads into the same cache.)"
    )
}

/// 2026-09-26: Resolve a model specifier to a local directory.
///
/// An existing directory is used as is when it holds `config.json` or a
/// `.gguf` file (`find_gguf`); anything else is looked up as a Hub id in the
/// local cache.
pub fn resolve_model_dir(model: &str, cache_dir: Option<&Path>) -> Result<PathBuf> {
    let as_path = Path::new(model);
    if as_path.is_dir()
        && (as_path.join("config.json").exists()
            || metrale_model_weights::weights::find_gguf(as_path).is_some())
    {
        tracing::info!("Model path: {} (local directory)", as_path.display());
        return Ok(as_path.to_path_buf());
    }

    resolve_from_hf_cache(model, cache_dir)
}

/// 2026-09-26: Look up a Hub id in the local cache:
/// `{cache_root}/models--{org}--{name}/snapshots/{hash}/`, where `hash` is read
/// from `refs/main`. Fails without `refs/main`, without that snapshot, or
/// without `config.json`/`params.json` in it.
fn resolve_from_hf_cache(model_id: &str, cache_dir: Option<&Path>) -> Result<PathBuf> {
    let cache_root = resolve_cache_root(cache_dir)?;

    let dir_name = format!("models--{}", model_id.replace('/', "--"));
    let model_cache = cache_root.join(&dir_name);

    if !model_cache.is_dir() {
        bail!(
            "Model '{}' not found in HF cache at {}.\n\
             Download it first:\n{}",
            model_id,
            cache_root.display(),
            download_hint(model_id),
        );
    }

    let ref_path = model_cache.join("refs/main");
    let snapshot_hash = std::fs::read_to_string(&ref_path)
        .with_context(|| {
            format!(
                "No default revision for '{}'. Expected refs/main at {}.\n\
                 The model may not have been fully downloaded.",
                model_id,
                ref_path.display(),
            )
        })?
        .trim()
        .to_string();

    let snapshot_dir = model_cache.join("snapshots").join(&snapshot_hash);
    if !snapshot_dir.is_dir() {
        bail!(
            "Snapshot directory not found: {}\n\
             refs/main points to hash '{}' but that snapshot doesn't exist.",
            snapshot_dir.display(),
            snapshot_hash,
        );
    }

    if !snapshot_dir.join("config.json").exists() && !snapshot_dir.join("params.json").exists() {
        bail!(
            "Model directory exists but is missing config.json or params.json: {}\n\
             The download may be incomplete.",
            snapshot_dir.display(),
        );
    }

    // 2026-09-26: `refs/main` can point at a snapshot with metadata but no
    // weights. Then the most recently modified sibling snapshot that has
    // weights is used instead, and the error below is raised only when none
    // has.
    if snapshot_has_weights(&snapshot_dir) {
        tracing::info!(
            "Model: {} (resolved to {})",
            model_id,
            snapshot_dir.display(),
        );
        return Ok(snapshot_dir);
    }

    tracing::warn!(
        "Snapshot '{}' for {} has no weight files (metadata-only); \
         scanning {} for a sibling snapshot with weights.",
        snapshot_hash,
        model_id,
        model_cache.join("snapshots").display(),
    );

    if let Some(fallback) = find_snapshot_with_weights(&model_cache.join("snapshots")) {
        tracing::info!(
            "Model: {} (resolved to {} — fell back from refs/main snapshot {} which had no weights)",
            model_id,
            fallback.display(),
            snapshot_hash,
        );
        return Ok(fallback);
    }

    bail!(
        "Snapshot '{}' for {} has no weight files (no model.safetensors / \
         consolidated.safetensors / *.safetensors found in {}). Sibling \
         snapshots in {} also lack weights — refresh the cache:\n{}",
        snapshot_hash,
        model_id,
        snapshot_dir.display(),
        model_cache.join("snapshots").display(),
        download_hint(model_id),
    );
}

/// 2026-09-26: Resolve a LoRA adapter specifier (local path or Hub id) to a
/// directory holding `adapter_config.json` and `adapter_model.safetensors`.
/// Like `resolve_model_dir`, but with the adapter's marker files and without
/// the sibling-snapshot fallback.
pub fn resolve_adapter_dir(spec: &str, cache_dir: Option<&Path>) -> Result<PathBuf> {
    let as_path = Path::new(spec);
    if as_path.is_dir() {
        if as_path.join("adapter_config.json").exists() {
            tracing::info!("Adapter path: {} (local directory)", as_path.display());
            return validate_adapter_dir(as_path.to_path_buf(), spec);
        }
        bail!(
            "Adapter directory {} has no adapter_config.json — not a PEFT adapter",
            as_path.display(),
        );
    }

    let cache_root = resolve_cache_root(cache_dir)?;
    let dir_name = format!("models--{}", spec.replace('/', "--"));
    let model_cache = cache_root.join(&dir_name);
    if !model_cache.is_dir() {
        bail!(
            "Adapter '{}' not found in HF cache at {}.\n\
             Download it first:\n{}",
            spec,
            cache_root.display(),
            download_hint(spec),
        );
    }

    let ref_path = model_cache.join("refs/main");
    let snapshot_hash = std::fs::read_to_string(&ref_path)
        .with_context(|| {
            format!(
                "No default revision for adapter '{}'. Expected refs/main at {}.\n\
                 The adapter may not have been fully downloaded.",
                spec,
                ref_path.display(),
            )
        })?
        .trim()
        .to_string();

    let snapshot_dir = model_cache.join("snapshots").join(&snapshot_hash);
    if !snapshot_dir.is_dir() {
        bail!(
            "Snapshot directory not found: {}\n\
             refs/main points to hash '{}' but that snapshot doesn't exist.",
            snapshot_dir.display(),
            snapshot_hash,
        );
    }

    if !snapshot_dir.join("adapter_config.json").exists() {
        bail!(
            "'{}' resolved to {} but it has no adapter_config.json — not a PEFT adapter repo",
            spec,
            snapshot_dir.display(),
        );
    }

    tracing::info!("Adapter: {} (resolved to {})", spec, snapshot_dir.display());
    validate_adapter_dir(snapshot_dir, spec)
}

/// 2026-09-26: Require `adapter_model.safetensors`. An `adapter_model.bin` gets
/// its own error, so it is not reported as missing weights.
fn validate_adapter_dir(dir: PathBuf, spec: &str) -> Result<PathBuf> {
    if dir.join("adapter_model.safetensors").exists() {
        return Ok(dir);
    }
    if dir.join("adapter_model.bin").exists() {
        bail!(
            "Adapter '{}' ships adapter_model.bin (torch pickle) — unsupported. \
             Re-export with save_pretrained(..., safe_serialization=True).",
            spec,
        );
    }
    bail!(
        "Adapter '{}' has no adapter_model.safetensors in {}",
        spec,
        dir.display(),
    );
}

/// 2026-09-26: True when the directory holds `model.safetensors`,
/// `consolidated.safetensors`, either one's `.index.json`, or any `*.safetensors`
/// or `*.gguf` file. An index alone counts.
pub(crate) fn snapshot_has_weights(dir: &Path) -> bool {
    let direct = [
        "model.safetensors",
        "model.safetensors.index.json",
        "consolidated.safetensors",
        "consolidated.safetensors.index.json",
    ];
    if direct.iter().any(|n| dir.join(n).exists()) {
        return true;
    }
    let Ok(read) = std::fs::read_dir(dir) else {
        return false;
    };
    read.filter_map(|e| e.ok()).any(|e| {
        e.file_name()
            .to_str()
            .is_some_and(|n| n.ends_with(".safetensors") || n.ends_with(".gguf"))
    })
}

/// 2026-09-26: The most recently modified directory under `snapshots/` for
/// which `snapshot_has_weights` holds, or `None`.
pub(crate) fn find_snapshot_with_weights(snapshots_root: &Path) -> Option<PathBuf> {
    let entries: Vec<_> = std::fs::read_dir(snapshots_root)
        .ok()?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .filter_map(|e| {
            let p = e.path();
            if !snapshot_has_weights(&p) {
                return None;
            }
            let mtime = e.metadata().ok()?.modified().ok()?;
            Some((mtime, p))
        })
        .collect();
    entries.into_iter().max_by_key(|(t, _)| *t).map(|(_, p)| p)
}

/// 2026-09-26: The Hub cache root, in order: the `cache_dir` argument
/// (`--cache-dir`), `$HF_HUB_CACHE`, `$HF_HOME/hub`, then
/// `$HOME/.cache/huggingface/hub`. Fails only when all are unset.
pub(crate) fn resolve_cache_root(cache_dir: Option<&Path>) -> Result<PathBuf> {
    if let Some(dir) = cache_dir {
        return Ok(dir.to_path_buf());
    }

    if let Ok(hub_cache) = std::env::var("HF_HUB_CACHE") {
        return Ok(PathBuf::from(hub_cache));
    }

    if let Ok(hf_home) = std::env::var("HF_HOME") {
        return Ok(PathBuf::from(hf_home).join("hub"));
    }

    let home =
        std::env::var("HOME").context("Cannot determine home directory: $HOME is not set")?;
    Ok(PathBuf::from(home).join(".cache/huggingface/hub"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// 2026-09-26: A cache entry with `refs/main`, `config.json` and
    /// `model.safetensors`.
    fn setup_mock_cache(tmp: &Path, org: &str, name: &str, hash: &str) -> PathBuf {
        let model_id = format!("{org}/{name}");
        let dir_name = format!("models--{}", model_id.replace('/', "--"));
        let model_cache = tmp.join(&dir_name);

        let snapshot_dir = model_cache.join("snapshots").join(hash);
        fs::create_dir_all(&snapshot_dir).unwrap();
        fs::create_dir_all(model_cache.join("refs")).unwrap();
        fs::write(model_cache.join("refs/main"), hash).unwrap();
        fs::write(snapshot_dir.join("config.json"), "{}").unwrap();
        fs::write(snapshot_dir.join("model.safetensors"), b"weights").unwrap();

        snapshot_dir
    }

    /// 2026-09-26: A cache entry whose `refs/main` snapshot has only
    /// `config.json` and `tokenizer.json`.
    fn setup_mock_cache_no_weights(tmp: &Path, org: &str, name: &str, hash: &str) -> PathBuf {
        let model_id = format!("{org}/{name}");
        let dir_name = format!("models--{}", model_id.replace('/', "--"));
        let model_cache = tmp.join(&dir_name);
        let snapshot_dir = model_cache.join("snapshots").join(hash);
        fs::create_dir_all(&snapshot_dir).unwrap();
        fs::create_dir_all(model_cache.join("refs")).unwrap();
        fs::write(model_cache.join("refs/main"), hash).unwrap();
        fs::write(snapshot_dir.join("config.json"), "{}").unwrap();
        fs::write(snapshot_dir.join("tokenizer.json"), "{}").unwrap();
        snapshot_dir
    }

    #[test]
    fn resolve_local_directory() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("config.json"), "{}").unwrap();

        let result = resolve_model_dir(tmp.path().to_str().unwrap(), None);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), tmp.path());
    }

    #[test]
    fn resolve_hf_model_id() {
        let tmp = tempfile::tempdir().unwrap();
        let expected = setup_mock_cache(
            tmp.path(),
            "nvidia",
            "Qwen3-Next-80B-A3B-Instruct-NVFP4",
            "abc123",
        );

        let result =
            resolve_model_dir("nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4", Some(tmp.path()));
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), expected);
    }

    #[test]
    fn resolve_missing_model_gives_download_hint() {
        let tmp = tempfile::tempdir().unwrap();
        let result = resolve_model_dir("nonexistent/model", Some(tmp.path()));
        let err = result.unwrap_err().to_string();
        assert!(err.contains("not found in HF cache"));
        assert!(err.contains("hf download"), "{err}");
        // 2026-09-26: The older CLI name is still mentioned.
        assert!(err.contains("huggingface-cli"), "{err}");
    }

    #[test]
    fn resolve_missing_config_json() {
        let tmp = tempfile::tempdir().unwrap();
        let dir_name = "models--org--model";
        let snapshot_dir = tmp.path().join(dir_name).join("snapshots/abc");
        fs::create_dir_all(&snapshot_dir).unwrap();
        fs::create_dir_all(tmp.path().join(dir_name).join("refs")).unwrap();
        fs::write(tmp.path().join(dir_name).join("refs/main"), "abc").unwrap();

        let result = resolve_model_dir("org/model", Some(tmp.path()));
        let err = result.unwrap_err().to_string();
        assert!(err.contains("missing config.json"));
    }

    #[test]
    fn cache_dir_arg_takes_precedence() {
        let custom = tempfile::tempdir().unwrap();
        let _expected = setup_mock_cache(custom.path(), "org", "model", "hash1");

        let result = resolve_model_dir("org/model", Some(custom.path()));
        assert!(result.is_ok());
        assert!(result.unwrap().starts_with(custom.path()));
    }

    #[test]
    fn falls_back_to_sibling_snapshot_with_weights() {
        // 2026-09-26: `refs/main` points at a snapshot without weights; the
        // sibling with weights is returned.
        let tmp = tempfile::tempdir().unwrap();
        let bad =
            setup_mock_cache_no_weights(tmp.path(), "nvidia", "Gemma-4-31B-IT-NVFP4", "05fa17");
        let model_cache = bad.parent().unwrap().parent().unwrap();
        let good_hash = "1365cf";
        let good = model_cache.join("snapshots").join(good_hash);
        fs::create_dir_all(&good).unwrap();
        fs::write(good.join("config.json"), "{}").unwrap();
        fs::write(good.join("model-00001-of-00004.safetensors"), b"shard").unwrap();
        std::thread::sleep(std::time::Duration::from_millis(20));
        fs::write(
            good.join("model-00001-of-00004.safetensors"),
            b"shard-newer",
        )
        .unwrap();

        let result = resolve_model_dir("nvidia/Gemma-4-31B-IT-NVFP4", Some(tmp.path()));
        assert!(result.is_ok(), "expected fallback to succeed: {:?}", result);
        assert_eq!(result.unwrap(), good);
    }

    #[test]
    fn bails_when_all_snapshots_lack_weights() {
        let tmp = tempfile::tempdir().unwrap();
        setup_mock_cache_no_weights(tmp.path(), "org", "model", "h1");
        let result = resolve_model_dir("org/model", Some(tmp.path()));
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("no weight files") || err.contains("metadata-only"),
            "expected weight-files error, got: {err}"
        );
        assert!(err.contains("hf download"), "{err}");
    }
}
