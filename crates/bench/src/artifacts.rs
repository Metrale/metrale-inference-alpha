// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The Metrale Engine home (`ArtifactStore`): the per-plugin
//! artifact and per-benchmark run directories, plus write-if-changed assets and
//! provisioning stamps.
//!
//! Layout this module creates under the root:
//! ```text
//!   <root>/
//!     artifacts/<plugin-id>/     provisioned material (venvs, datasets, scripts)
//!     runs/<benchmark-id>/       run records (history.rs) and baselines (baseline.rs)
//! ```
//!
//! Owner: bench (artifacts).
//! Invariants:
//! - `write_asset_bytes` never rewrites a file whose bytes already match; an
//!   existing file it cannot read counts as different.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

mod home;

pub use home::{HomeFault, HomeSource, MetraleHome};

/// 2026-09-26: Handle to the artifact area: the resolved root directory.
#[derive(Clone, Debug)]
pub struct ArtifactStore {
    root: PathBuf,
}

impl ArtifactStore {
    /// 2026-09-26: The store at the resolved home: `METRALE_HOME` when set,
    /// otherwise `$HOME/.metrale`. Neither set is an error; there is no
    /// fallback directory. See [`MetraleHome::resolve`].
    pub fn discover() -> Result<Self> {
        Ok(Self {
            root: MetraleHome::resolve()?.root,
        })
    }

    /// 2026-09-26: The resolved home together with where it came from
    /// ([`HomeSource`]).
    pub fn discover_with_provenance() -> Result<MetraleHome> {
        MetraleHome::resolve()
    }

    /// 2026-09-26: Point the store at an explicit root, bypassing resolution.
    pub fn with_root(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// 2026-09-26: `<root>/artifacts/<plugin_id>`, created.
    pub fn plugin_dir(&self, plugin_id: &str) -> Result<PathBuf> {
        let dir = self.root.join("artifacts").join(plugin_id);
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("creating artifact dir {}", dir.display()))?;
        Ok(dir)
    }

    /// 2026-09-26: `<root>/runs/<benchmark_id>`, created.
    pub fn runs_dir(&self, benchmark_id: &str) -> Result<PathBuf> {
        let dir = self.root.join("runs").join(benchmark_id);
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("creating runs dir {}", dir.display()))?;
        Ok(dir)
    }
}

/// 2026-09-26: Write a compiled-in asset into `dir`, but only when the bytes
/// differ, so a provisioned script always matches the binary that ships it and
/// an unchanged file keeps its mtime.
///
/// Returns `true` when the file was written.
pub fn write_asset(dir: &Path, name: &str, contents: &str) -> Result<bool> {
    write_asset_bytes(dir, name, contents.as_bytes())
}

/// 2026-09-26: [`write_asset`] for assets that are not text: the same
/// compare-then-write contract, over bytes. The comparison reads bytes because
/// `read_to_string` fails on non-UTF-8 files and would make every binary asset
/// look changed.
pub fn write_asset_bytes(dir: &Path, name: &str, contents: &[u8]) -> Result<bool> {
    let path = dir.join(name);
    if let Ok(existing) = std::fs::read(&path)
        && existing == contents
    {
        return Ok(false);
    }
    std::fs::write(&path, contents).with_context(|| format!("writing {}", path.display()))?;
    Ok(true)
}

/// 2026-09-26: A provisioning stamp: a marker file whose contents identify the
/// inputs that produced an artifact. `is_current` is true only when the file's
/// trimmed contents equal the expected value, so changed inputs make it stale.
pub struct Stamp {
    path: PathBuf,
    expected: String,
}

impl Stamp {
    pub fn new(dir: &Path, name: &str, expected: impl Into<String>) -> Self {
        Self {
            path: dir.join(name),
            expected: expected.into(),
        }
    }

    pub fn is_current(&self) -> bool {
        std::fs::read_to_string(&self.path).is_ok_and(|s| s.trim() == self.expected.trim())
    }

    /// 2026-09-26: Record that provisioning succeeded. Call it last: a stamp
    /// written before the work completes marks a half-provisioned directory as
    /// current.
    pub fn commit(&self) -> Result<()> {
        std::fs::write(&self.path, &self.expected)
            .with_context(|| format!("writing stamp {}", self.path.display()))
    }
}

#[cfg(test)]
mod tests {
    use super::home::resolve_from;
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let d =
            std::env::temp_dir().join(format!("metrale-plugin-test-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn dirs_are_created_under_the_configured_root() {
        let root = tmp("dirs");
        let store = ArtifactStore::with_root(&root);
        let p = store.plugin_dir("bfcl").unwrap();
        let r = store.runs_dir("bfcl-subset").unwrap();
        assert!(p.is_dir() && r.is_dir());
        assert_eq!(p, root.join("artifacts/bfcl"));
        assert_eq!(r, root.join("runs/bfcl-subset"));
    }

    #[test]
    fn write_asset_rewrites_only_on_change() {
        let dir = tmp("asset");
        let path = dir.join("s.py");
        assert!(write_asset(&dir, "s.py", "print(1)").unwrap());

        let old = std::time::SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(1);
        std::fs::File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_times(std::fs::FileTimes::new().set_modified(old))
            .unwrap();
        let pinned_mtime = std::fs::metadata(&path).unwrap().modified().unwrap();

        assert!(!write_asset(&dir, "s.py", "print(1)").unwrap());
        assert_eq!(
            std::fs::metadata(&path).unwrap().modified().unwrap(),
            pinned_mtime,
            "unchanged contents must not rewrite the asset"
        );
        assert!(write_asset(&dir, "s.py", "print(2)").unwrap());
        assert_eq!(std::fs::read_to_string(path).unwrap(), "print(2)");
    }

    #[test]
    fn binary_assets_are_compared_as_bytes() {
        let dir = tmp("binary-asset");
        let path = dir.join("image.bin");
        let bytes = [0xff, 0x00, 0xfe];

        assert!(write_asset_bytes(&dir, "image.bin", &bytes).unwrap());
        assert!(!write_asset_bytes(&dir, "image.bin", &bytes).unwrap());
        assert_eq!(std::fs::read(path).unwrap(), bytes);
    }

    /// 2026-09-26: A scratch `$HOME` with the given directories in it. It is
    /// passed to `resolve_from`, never exported with `set_var`, so these tests
    /// share no process state.
    fn home_with(tag: &str, dirs: &[&str]) -> PathBuf {
        let home = tmp(tag);
        for d in dirs {
            std::fs::create_dir_all(home.join(d)).unwrap();
        }
        home
    }

    fn resolved(home: &Path) -> MetraleHome {
        resolve_from(None, Some(home.as_os_str().to_owned())).unwrap()
    }

    #[test]
    fn a_fresh_box_gets_the_default_home() {
        let home = home_with("neither", &[]);
        let got = resolved(&home);
        assert_eq!(got.root, home.join(".metrale"));
        assert_eq!(got.source, HomeSource::HomeDefault);
    }

    /// 2026-09-26: `METRALE_HOME` wins even when `$HOME/.metrale` exists.
    #[test]
    fn metrale_home_wins_over_the_default_directory() {
        let home = home_with("env-wins", &[".metrale"]);
        let explicit = home.join("elsewhere");
        let got = resolve_from(
            Some(explicit.as_os_str().to_owned()),
            Some(home.as_os_str().to_owned()),
        )
        .unwrap();
        assert_eq!(got.root, explicit);
        assert_eq!(got.source, HomeSource::Env);
    }

    /// 2026-09-26: `resolve_from` creates nothing on disk.
    #[test]
    fn resolving_creates_nothing() {
        let home = home_with("no-side-effects", &[]);
        let _ = resolved(&home);
        assert!(
            !home.join(".metrale").exists(),
            "resolve must not create the home it names"
        );
        let entries: Vec<_> = std::fs::read_dir(&home)
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert!(entries.is_empty(), "expected nothing, got {entries:?}");
    }

    #[test]
    fn stamp_is_stale_until_committed_and_tracks_its_inputs() {
        let dir = tmp("stamp");
        let s = Stamp::new(&dir, ".provisioned", "v1");
        assert!(!s.is_current());
        s.commit().unwrap();
        assert!(s.is_current());
        assert!(!Stamp::new(&dir, ".provisioned", "v2").is_current());
    }
}
