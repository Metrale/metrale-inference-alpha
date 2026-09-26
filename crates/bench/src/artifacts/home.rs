// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Which directory is the Metrale Engine home, where that answer came
//! from, and whether this process can write there.
//!
//! Owner: bench (artifacts).
//! Invariants:
//! - `resolve_from` reads only its two arguments: it neither reads nor writes
//!   the filesystem.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

/// 2026-09-26: Where the Metrale Engine home came from. It matters because the
/// signing identity lives under the home (`<root>/identity/ed25519.pk8`,
/// `gate::signing`), so two homes sign as two signers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HomeSource {
    /// 2026-09-26: `METRALE_HOME` was set.
    Env,
    /// 2026-09-26: `$HOME/.metrale`.
    HomeDefault,
}

impl HomeSource {
    /// 2026-09-26: The provenance as operator-facing text.
    pub fn describe(self) -> &'static str {
        match self {
            Self::Env => "from METRALE_HOME",
            Self::HomeDefault => "default, $HOME/.metrale",
        }
    }
}

/// 2026-09-26: A resolved Metrale Engine home and its provenance.
#[derive(Clone, Debug)]
pub struct MetraleHome {
    pub root: PathBuf,
    pub source: HomeSource,
}

/// 2026-09-26: Why a Metrale Engine home cannot be used.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HomeFault {
    /// 2026-09-26: The directory exists but the write probe failed.
    NotWritable {
        /// 2026-09-26: The directory's owner uid, when it could be read.
        owner_uid: Option<u32>,
        /// 2026-09-26: This process's uid, when it could be read.
        process_uid: Option<u32>,
    },
    /// 2026-09-26: The path exists and is not a directory.
    NotADirectory,
    /// 2026-09-26: The directory did not exist and `create_dir_all` failed.
    Uncreatable(String),
}

impl std::fmt::Display for HomeFault {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotWritable {
                owner_uid: Some(o),
                process_uid: Some(p),
            } if o != p => write!(
                f,
                "not writable: owned by uid {o}, this process is uid {p}. A `sudo` \
                 or container run most likely created it. Fix with \
                 `sudo chown -R {p} <path>`, or point METRALE_HOME somewhere this \
                 user owns."
            ),
            Self::NotWritable { .. } => write!(
                f,
                "not writable by this process. Fix the permissions, or point \
                 METRALE_HOME somewhere this user owns."
            ),
            Self::NotADirectory => write!(f, "exists but is not a directory"),
            Self::Uncreatable(e) => write!(f, "does not exist and could not be created: {e}"),
        }
    }
}

impl MetraleHome {
    /// 2026-09-26: Resolve from the environment: a non-empty `METRALE_HOME`
    /// wins, then `$HOME/.metrale`. An empty `METRALE_HOME` is an error, and so
    /// is having neither variable. Touches no file.
    pub fn resolve() -> Result<Self> {
        resolve_from(std::env::var_os("METRALE_HOME"), std::env::var_os("HOME"))
    }

    /// 2026-09-26: Can this process use the home? `None` means yes.
    ///
    /// Creates the directory if it is missing, then probes by writing and
    /// removing `.metrale-write-probe`, so ownership, ACLs, read-only mounts
    /// and a full disk all show up as the one failure that matters.
    pub fn fault(&self) -> Option<HomeFault> {
        check_usable(&self.root)
    }

    /// 2026-09-26: One line naming the root and where it came from.
    pub fn describe(&self) -> String {
        format!("{} ({})", self.root.display(), self.source.describe())
    }
}

/// 2026-09-26: [`MetraleHome::resolve`] over explicit values of `METRALE_HOME`
/// and `HOME`, so tests need not call `set_var`, which is unsafe and
/// process-global.
pub(super) fn resolve_from(
    metrale_home: Option<std::ffi::OsString>,
    home: Option<std::ffi::OsString>,
) -> Result<MetraleHome> {
    if let Some(explicit) = metrale_home {
        let root = PathBuf::from(explicit);
        if root.as_os_str().is_empty() {
            bail!("METRALE_HOME is set but empty");
        }
        return Ok(MetraleHome {
            root,
            source: HomeSource::Env,
        });
    }
    let home = home
        .map(PathBuf::from)
        .filter(|h| !h.as_os_str().is_empty())
        .context("neither METRALE_HOME nor HOME is set — cannot place ~/.metrale")?;
    Ok(MetraleHome {
        root: home.join(".metrale"),
        source: HomeSource::HomeDefault,
    })
}

pub(super) fn check_usable(root: &Path) -> Option<HomeFault> {
    if root.exists() && !root.is_dir() {
        return Some(HomeFault::NotADirectory);
    }
    if !root.exists()
        && let Err(e) = std::fs::create_dir_all(root)
    {
        return Some(HomeFault::Uncreatable(e.to_string()));
    }
    let probe = root.join(".metrale-write-probe");
    match std::fs::write(&probe, b"") {
        Ok(()) => {
            let _ = std::fs::remove_file(&probe);
            None
        }
        Err(_) => Some(HomeFault::NotWritable {
            owner_uid: owner_uid_of(root),
            process_uid: current_uid(),
        }),
    }
}

#[cfg(unix)]
fn owner_uid_of(p: &Path) -> Option<u32> {
    use std::os::unix::fs::MetadataExt;
    std::fs::metadata(p).ok().map(|m| m.uid())
}

#[cfg(not(unix))]
fn owner_uid_of(_p: &Path) -> Option<u32> {
    None
}

/// 2026-09-26: This process's uid, read as the owner of `/proc/self` because
/// `libc` is not a dependency of this crate. Checked 2026-09-26 on Linux:
/// `stat` of `/proc/self`, following the link, gives the effective uid.
#[cfg(unix)]
fn current_uid() -> Option<u32> {
    use std::os::unix::fs::MetadataExt;
    std::fs::metadata("/proc/self").ok().map(|m| m.uid())
}

#[cfg(not(unix))]
fn current_uid() -> Option<u32> {
    None
}
