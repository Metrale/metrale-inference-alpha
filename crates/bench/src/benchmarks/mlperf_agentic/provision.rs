// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Locate and verify the trajectory file, `dataset.jsonl` in the
//! artifact store's `mlperf-agentic` directory. Nothing downloads it: an
//! operator places it there.
//!
//! Owner: bench, mlperf_agentic.
//! Invariants: `ensure` returns `Ok` only when the file exists and, if a pin
//! is set, its full-file SHA256 matches the pin.

use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use crate::artifacts::ArtifactStore;
use crate::plugin::PluginHandle;

pub const PLUGIN_ID: &str = "mlperf-agentic";

/// 2026-09-26: The dataset status the absent-dataset error quotes.
pub const UPSTREAM_DATASET_STATUS: &str = "mlcommons/endpoints@7935df4 examples/10_Agentic_Inference/README.md: \"The official \
     MLPerf dataset can be downloaded from MLCommons storage (link TBD)\"";

#[derive(Clone, Debug)]
pub struct Artifacts {
    pub dir: PathBuf,
    pub dataset: PathBuf,
    /// 2026-09-26: Full-file SHA256 of `dataset`, hex. Written to
    /// `dataset_summary.json` and into the run's dataset fingerprint.
    pub file_sha256: String,
}

/// 2026-09-26: Locate and verify the dataset. `expected_sha256` is the pin:
/// empty means record only, non-empty means the file must hash to exactly
/// that (compared case-insensitively).
pub fn ensure(
    store: &ArtifactStore,
    handle: &PluginHandle,
    expected_sha256: &str,
) -> Result<Artifacts> {
    let dir = store.plugin_dir(PLUGIN_ID)?;
    let dataset = dir.join("dataset.jsonl");
    if !dataset.is_file() {
        // 2026-09-26: No proxy or reconstructed dataset: a score from a
        // different draw is comparable to nothing.
        bail!(
            "the MLPerf Agentic Inference dataset is not provisioned: {} does not exist.\n\
             The official dataset (613 trajectories: 500 Workato workflow + 113 DeepSWE \
             coding, 30,335 client turns) is NOT yet published — {}. There is no download \
             URL, license, or auth story to automate, and this leg deliberately refuses to \
             substitute a proxy or reconstructed dataset: a score from a different draw is \
             comparable to nothing, however official it looks.\n\
             When MLCommons publishes the file: place it at the path above, record its \
             SHA256 as this leg's expected_sha256 parameter (and in BENCH.toml), and run \
             the calibration protocol in the BENCH.toml note before trusting any number.",
            dataset.display(),
            UPSTREAM_DATASET_STATUS,
        );
    }

    let (file_sha256, bytes) = sha256_file(&dataset)?;
    if !expected_sha256.is_empty() && !expected_sha256.eq_ignore_ascii_case(&file_sha256) {
        bail!(
            "{} does not match its pin: sha256 {file_sha256} ({bytes} bytes), expected \
             {expected_sha256}. A replay of the wrong file scores against the wrong ground \
             truth; refusing to run.",
            dataset.display()
        );
    }
    handle.info(format!(
        "dataset {} — {bytes} bytes, sha256 {file_sha256}{}",
        dataset.display(),
        if expected_sha256.is_empty() {
            " (UNPINNED: no expected_sha256 — record-only until the official file ships)"
        } else {
            " (pin verified)"
        }
    ));
    std::fs::write(
        dir.join("dataset_summary.json"),
        serde_json::to_string_pretty(&serde_json::json!({
            "file_sha256": file_sha256,
            "bytes": bytes,
        }))? + "\n",
    )?;
    Ok(Artifacts {
        dir,
        dataset,
        file_sha256,
    })
}

/// 2026-09-26: Hex SHA256 and byte count of a file, read in 64 KiB blocks.
fn sha256_file(path: &Path) -> Result<(String, u64)> {
    use sha2::{Digest, Sha256};
    let mut file =
        std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut digest = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    let mut total = 0u64;
    loop {
        let n = file
            .read(&mut buf)
            .with_context(|| format!("reading {}", path.display()))?;
        if n == 0 {
            break;
        }
        total += n as u64;
        digest.update(&buf[..n]);
    }
    Ok((format!("{:x}", digest.finalize()), total))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let d =
            std::env::temp_dir().join(format!("metrale-mlperf-prov-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn sha256_matches_a_known_vector() {
        let dir = tmp("sha");
        let p = dir.join("f");
        std::fs::write(&p, b"abc").unwrap();
        let (sha, n) = sha256_file(&p).unwrap();
        assert_eq!(n, 3);
        assert_eq!(
            sha,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    // 2026-09-26: The absent-dataset failure is tested in `mlperf_tests.rs`,
    // where a PluginHandle exists to call ensure() with.
}
