// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Subprocess runner, Python preflight and venv provisioning.
//! BFCL provisions its venv with these during `load()`; `run` is also used to
//! call other tools.
//!
//! A `load()` error becomes the run's failed `setup` frame, so the failures
//! of `find_python`, `ensure_venv` and `pip_install` name the missing piece
//! and a fix.
//!
//! Owner: bench (provisioning).
//! Invariants:
//! - A command that `run` sees exit unsuccessfully is an error carrying at
//!   most the last 12 lines of its stderr.

use std::path::{Path, PathBuf};
use std::process::Stdio;

use anyhow::{Context, Result, anyhow, bail};
use tokio::process::Command;

/// 2026-09-26: Captured output of a command.
#[derive(Debug)]
pub struct Output {
    pub stdout: String,
    pub stderr: String,
}

/// 2026-09-26: Run a command to completion with stdin closed, capturing both
/// streams. A failure carries the tail of stderr, since a bare exit code is
/// not actionable.
pub async fn run(program: &Path, args: &[&str], cwd: Option<&Path>) -> Result<Output> {
    let mut cmd = Command::new(program);
    cmd.args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(dir) = cwd {
        cmd.current_dir(dir);
    }
    let out = cmd
        .output()
        .await
        .with_context(|| format!("spawning {}", program.display()))?;
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    if !out.status.success() {
        let tail: Vec<&str> = stderr.lines().rev().take(12).collect();
        let tail: Vec<&str> = tail.into_iter().rev().collect();
        bail!(
            "{} {} failed ({}):\n{}",
            program.display(),
            args.join(" "),
            out.status,
            tail.join("\n")
        );
    }
    Ok(Output { stdout, stderr })
}

/// 2026-09-26: Locate a Python at least `min_major.min_minor`: `python3`,
/// then `python`.
///
/// `METRALE_PYTHON`, when set, is the only candidate: the box may keep the
/// usable interpreter off PATH.
pub async fn find_python(min_major: u32, min_minor: u32) -> Result<PathBuf> {
    let candidates: Vec<PathBuf> = match std::env::var_os("METRALE_PYTHON") {
        Some(p) => vec![PathBuf::from(p)],
        None => vec![PathBuf::from("python3"), PathBuf::from("python")],
    };
    let mut tried = Vec::new();
    for cand in candidates {
        match run(&cand, &["--version"], None).await {
            Ok(out) => {
                let text = if out.stdout.trim().is_empty() {
                    out.stderr
                } else {
                    out.stdout
                };
                let (major, minor) = parse_version(&text)
                    .ok_or_else(|| anyhow!("could not parse {:?} from {}", text, cand.display()))?;
                if (major, minor) >= (min_major, min_minor) {
                    return Ok(cand);
                }
                tried.push(format!(
                    "{} is {major}.{minor}, need >= {min_major}.{min_minor}",
                    cand.display()
                ));
            }
            Err(e) => tried.push(format!("{}: {e}", cand.display())),
        }
    }
    bail!(
        "no usable python found (need >= {min_major}.{min_minor}). Tried:\n  {}\n\
         Install python3 or point METRALE_PYTHON at an interpreter.",
        tried.join("\n  ")
    )
}

fn parse_version(text: &str) -> Option<(u32, u32)> {
    // 2026-09-26: "Python 3.12.3": take the first token whose leading
    // dot-separated part is all digits; a missing minor reads as 0.
    let token = text.split_whitespace().find(|t| {
        t.split('.')
            .next()
            .is_some_and(|h| !h.is_empty() && h.chars().all(|c| c.is_ascii_digit()))
    })?;
    let mut parts = token.split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next().unwrap_or("0").parse().ok()?;
    Some((major, minor))
}

/// 2026-09-26: The interpreter inside a venv, per platform layout.
pub fn venv_python(venv: &Path) -> PathBuf {
    if cfg!(windows) {
        venv.join("Scripts").join("python.exe")
    } else {
        venv.join("bin").join("python")
    }
}

/// 2026-09-26: Create `venv` with `python` if its interpreter is not already
/// there, and return the interpreter.
pub async fn ensure_venv(python: &Path, venv: &Path) -> Result<PathBuf> {
    let interpreter = venv_python(venv);
    if interpreter.is_file() {
        return Ok(interpreter);
    }
    // 2026-09-26: Probe the `venv` module first, so a box missing
    // `python3-venv` is told which package to install.
    run(python, &["-c", "import venv"], None)
        .await
        .context("python is present but the `venv` module is not — install python3-venv")?;
    let venv_str = venv
        .to_str()
        .ok_or_else(|| anyhow!("venv path is not valid UTF-8: {}", venv.display()))?;
    run(python, &["-m", "venv", venv_str], None)
        .await
        .with_context(|| format!("creating venv at {}", venv.display()))?;
    if !interpreter.is_file() {
        bail!(
            "venv created at {} but {} is missing",
            venv.display(),
            interpreter.display()
        );
    }
    Ok(interpreter)
}

/// 2026-09-26: `pip install -r <requirements>` with the venv's interpreter.
pub async fn pip_install(interpreter: &Path, requirements: &Path) -> Result<()> {
    let req = requirements
        .to_str()
        .ok_or_else(|| anyhow!("requirements path is not valid UTF-8"))?;
    run(
        interpreter,
        &[
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "-r",
            req,
        ],
        None,
    )
    .await
    .context(
        "pip install failed — this step needs network access to PyPI \
         (air-gapped boxes must pre-populate ~/.metrale/artifacts)",
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_parses_from_either_stream_format() {
        assert_eq!(parse_version("Python 3.12.3\n"), Some((3, 12)));
        assert_eq!(parse_version("Python 3.9"), Some((3, 9)));
        assert_eq!(parse_version("no version here"), None);
    }

    #[test]
    fn venv_interpreter_path_matches_the_platform_layout() {
        let p = venv_python(Path::new("/tmp/v"));
        if cfg!(windows) {
            assert!(p.ends_with("Scripts/python.exe"));
        } else {
            assert_eq!(p, Path::new("/tmp/v/bin/python"));
        }
    }

    #[tokio::test]
    async fn a_failing_command_reports_stderr_not_just_the_code() {
        let script =
            "i=1; while [ $i -le 14 ]; do printf 'line-%02d\\n' $i >&2; i=$((i + 1)); done; exit 3";
        let err = run(Path::new("sh"), &["-c", script], None)
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("failed (exit status: 3)"), "{err}");
        assert!(!err.contains("line-01\n"), "tail excludes line 1: {err}");
        assert!(!err.contains("line-02\n"), "tail excludes line 2: {err}");
        assert!(err.contains("line-03\n"), "tail begins at line 3: {err}");
        assert!(err.ends_with("line-14"), "tail preserves order: {err}");
    }

    #[tokio::test]
    async fn a_successful_command_captures_both_streams_and_honors_cwd() {
        let dir = std::env::temp_dir().join(format!("metrale-python-run-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("scratch");
        let out = run(
            Path::new("sh"),
            &["-c", "pwd; printf warning >&2"],
            Some(&dir),
        )
        .await
        .expect("runs");
        assert_eq!(
            PathBuf::from(out.stdout.trim()),
            dir.canonicalize().expect("canonical scratch")
        );
        assert_eq!(out.stderr, "warning");
        let _ = std::fs::remove_dir_all(dir);
    }
}
