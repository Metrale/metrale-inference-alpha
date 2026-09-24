// SPDX-License-Identifier: AGPL-3.0-only

//! `~/.metrale` — where a plugin keeps everything it had to fetch or build.
//!
//! Layout:
//! ```text
//!   ~/.metrale/
//!     artifacts/<plugin-id>/     downloaded + provisioned material (venvs, datasets)
//!     runs/<benchmark-id>/       persisted run frames, read by the History pane
//! ```
//!
//! Nothing here writes into the repo or the CWD: a benchmark run must not
//! mutate the tree it is measuring.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

mod home;

pub use home::{HomeFault, HomeSource, MetraleHome};

/// Handle to the on-disk artifact area. Cheap to clone.
#[derive(Clone, Debug)]
pub struct ArtifactStore {
    root: PathBuf,
}

impl ArtifactStore {
    /// Resolve the Metrale Engine home. `METRALE_HOME` wins when set (the escape hatch for
    /// a read-only or shared `$HOME`); otherwise `$HOME/.metrale`. A missing
    /// `$HOME` is an error, not a fallback to `/tmp` — a benchmark silently
    /// provisioning several GB somewhere unexpected is worse than a clear stop.
    ///
    /// See [`MetraleHome::resolve`] for the full rule.
    pub fn discover() -> Result<Self> {
        Ok(Self {
            root: MetraleHome::resolve()?.root,
        })
    }

    /// The resolved home together with WHERE it came from.
    ///
    /// `discover()` throws the provenance away, which is why nothing could ever
    /// tell an operator that two commands had disagreed about the root.
    pub fn discover_with_provenance() -> Result<MetraleHome> {
        MetraleHome::resolve()
    }

    /// Point the store at an explicit root (tests, and the `METRALE_HOME` path).
    pub fn with_root(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// `~/.metrale/artifacts/<plugin_id>`, created.
    pub fn plugin_dir(&self, plugin_id: &str) -> Result<PathBuf> {
        let dir = self.root.join("artifacts").join(plugin_id);
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("creating artifact dir {}", dir.display()))?;
        Ok(dir)
    }

    /// `~/.metrale/runs/<benchmark_id>`, created.
    pub fn runs_dir(&self, benchmark_id: &str) -> Result<PathBuf> {
        let dir = self.root.join("runs").join(benchmark_id);
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("creating runs dir {}", dir.display()))?;
        Ok(dir)
    }
}

/// Write a compiled-in asset into `dir`, but only when the bytes differ.
///
/// Provisioned scripts must track the binary that ships them — a Metrale Engine upgrade
/// that changes the BFCL scorer has to overwrite the copy in `~/.metrale`, or the
/// run would be scored by the previous release. Comparing content (rather than
/// checking existence) keeps mtimes stable so downstream stamps stay valid.
///
/// Returns `true` when the file was written.
pub fn write_asset(dir: &Path, name: &str, contents: &str) -> Result<bool> {
    write_asset_bytes(dir, name, contents.as_bytes())
}

/// [`write_asset`] for assets that are not text.
///
/// Same contract — compare content, write only on a difference, return whether
/// it wrote — but over bytes, because `read_to_string` fails on any file that
/// is not valid UTF-8 and would therefore report every binary asset as
/// "differs" and rewrite it on each `load()`, churning mtimes that downstream
/// stamps depend on. The vision benchmark provisions PNGs.
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

/// A provisioning stamp: a marker file whose contents identify the inputs that
/// produced the artifact. Provisioning is skipped iff the stamp matches, so a
/// changed pin (requirements, script, dataset digest) re-provisions by itself
/// instead of needing anyone to remember to clear a cache.
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

    /// Record that provisioning succeeded. Call this LAST — a stamp written
    /// before the work completes turns a half-provisioned directory into a
    /// permanent "already done".
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

    /// A scratch `$HOME` with a chosen set of home directories already in it.
    ///
    /// Passed to `resolve_from` rather than exported through `set_var`, so
    /// these cases run concurrently with the rest of the suite and with
    /// each other.
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

    /// A fresh install lands on the default name.
    #[test]
    fn a_fresh_box_gets_the_default_home() {
        let home = home_with("neither", &[]);
        let got = resolved(&home);
        assert_eq!(got.root, home.join(".metrale"));
        assert_eq!(got.source, HomeSource::HomeDefault);
    }

    /// The explicit escape hatch outranks the default, including when the
    /// default directory is sitting right there. An operator who named a root
    /// gets that root.
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

    /// `resolve_from` MUST NOT write to the filesystem: nobody's home is
    /// created as a side effect of reading a path.
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
        // A changed pin invalidates it without anyone clearing a cache.
        assert!(!Stamp::new(&dir, ".provisioned", "v2").is_current());
    }
}
