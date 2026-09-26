// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The shared warm cargo target dir, a port of the harness's
//! `warm_cargo_cache.sh`. With a cold dir every generated Axum project compiles
//! its whole dependency tree inside the scorer's build timeout. This builds a
//! template project in the debug profile (for the agent's own `cargo test` and
//! `cargo run`) and the release profile (for the scorer's `cargo build
//! --release`), and returns the dir that the agent shell and the scorer both
//! set as `CARGO_TARGET_DIR`.
//!
//! Owner: bench, agentic.
//! Invariants: `prepare` returns the dir only after both profile builds
//! succeeded; any failure is an error, never a fall back to a cold dir.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use anyhow::{Context, Result, bail};

use crate::plugin::PluginHandle;

/// 2026-09-26: Per `cargo` invocation; a guard so a wedged `cargo` cannot hang
/// the benchmark.
const PREWARM_TIMEOUT: Duration = Duration::from_secs(1800);

/// 2026-09-26: Parses to the same TOML as the `Cargo.toml` heredoc in
/// `warm_cargo_cache.sh` (asserted in the tests).
pub const TEMPLATE_MANIFEST: &str = r#"[package]
name = "metrale-warm-template"
version = "0.1.0"
edition = "2021"

[dependencies]
axum = { version = "0.8", features = ["json", "macros", "ws", "multipart"] }
tokio = { version = "1", features = ["full"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
tower = { version = "0.5", features = ["full"] }
tower-http = { version = "0.6", features = ["full"] }
hyper = { version = "1", features = ["full"] }
reqwest = { version = "0.12", features = ["json"] }
anyhow = "1"
thiserror = "2"
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }

[dev-dependencies]
reqwest = { version = "0.12", features = ["json"] }
"#;

/// 2026-09-26: The template's `main.rs`, a ping/pong server on
/// `METRALE_HARNESS_PORT`; the same text as the `main.rs` heredoc in
/// `warm_cargo_cache.sh` after its first comment line (asserted in the tests).
pub const TEMPLATE_MAIN: &str = r#"use axum::{routing::get, Router};

async fn ping() -> &'static str {
    "pong"
}

#[tokio::main]
async fn main() {
    let _ = serde_json::json!({"ok": true});
    let _v: tower::ServiceBuilder<tower::layer::util::Identity> = tower::ServiceBuilder::new();
    let app = Router::new().route("/ping", get(ping));
    let port: u16 = std::env::var("METRALE_HARNESS_PORT")
        .unwrap_or_else(|_| "3001".to_string())
        .parse()
        .unwrap();
    let listener = tokio::net::TcpListener::bind(("0.0.0.0", port)).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
"#;

/// 2026-09-26: Resolves `${VAR:-${HOME}/.cargo/<leaf>}` as the harness does: an
/// empty `explicit` counts as unset, and a missing or empty `home` is an error.
/// The callers read the environment.
fn dir_from(explicit: Option<OsString>, home: Option<OsString>, leaf: &str) -> Result<PathBuf> {
    if let Some(p) = explicit.filter(|p| !p.is_empty()) {
        return Ok(PathBuf::from(p));
    }
    let home = home
        .filter(|h| !h.is_empty())
        .context("HOME is not set, so the shared warm cargo dir has no home")?;
    Ok(PathBuf::from(home).join(".cargo").join(leaf))
}

/// 2026-09-26: `METRALE_WARM_TARGET_DIR`, else
/// `$HOME/.cargo/metrale-warm-target`: the variable and default path of
/// `warm_cargo_cache.sh`, `run_tier.sh` and `score_run.py`'s
/// `_warm_target_dir` (asserted in the tests).
pub fn warm_target_dir() -> Result<PathBuf> {
    dir_from(
        std::env::var_os("METRALE_WARM_TARGET_DIR"),
        std::env::var_os("HOME"),
        "metrale-warm-target",
    )
}

/// 2026-09-26: `METRALE_WARM_TEMPLATE_DIR`, else
/// `$HOME/.cargo/metrale-warm-template`, as in `warm_cargo_cache.sh`.
pub fn template_dir() -> Result<PathBuf> {
    dir_from(
        std::env::var_os("METRALE_WARM_TEMPLATE_DIR"),
        std::env::var_os("HOME"),
        "metrale-warm-template",
    )
}

/// 2026-09-26: Writes the template project. A file whose content already
/// matches is left untouched.
pub fn write_template(dir: &Path) -> Result<()> {
    let src = dir.join("src");
    std::fs::create_dir_all(&src)
        .with_context(|| format!("creating warm template dir {}", src.display()))?;
    write_if_changed(&dir.join("Cargo.toml"), TEMPLATE_MANIFEST)?;
    write_if_changed(&src.join("main.rs"), TEMPLATE_MAIN)?;
    Ok(())
}

/// 2026-09-26: Rewriting an identical file would bump its mtime and make cargo
/// rebuild the template crate on every run.
fn write_if_changed(path: &Path, content: &str) -> Result<()> {
    if std::fs::read_to_string(path).is_ok_and(|old| old == content) {
        return Ok(());
    }
    std::fs::write(path, content).with_context(|| format!("writing {}", path.display()))
}

/// 2026-09-26: One `cargo` invocation against the template, output into the
/// warm dir. A non-zero exit, a failed start or `PREWARM_TIMEOUT` is an error.
async fn cargo(args: &[&str], warm: &Path, manifest: &Path) -> Result<()> {
    let mut cmd = tokio::process::Command::new("cargo");
    cmd.args(args)
        .arg("--manifest-path")
        .arg(manifest)
        .env("CARGO_TARGET_DIR", warm)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    let out = tokio::time::timeout(PREWARM_TIMEOUT, cmd.output())
        .await
        .with_context(|| {
            format!(
                "`cargo {}` exceeded {}s while warming {}",
                args.join(" "),
                PREWARM_TIMEOUT.as_secs(),
                warm.display()
            )
        })?
        .with_context(|| format!("`cargo {}` could not start", args.join(" ")))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        let tail = err.lines().rev().take(8).collect::<Vec<_>>().join(" ");
        bail!(
            "`cargo {}` failed while warming {}: {}",
            args.join(" "),
            warm.display(),
            super::super::one_line(tail)
        );
    }
    Ok(())
}

/// 2026-09-26: Prepares the shared warm target dir and returns it. A failure is
/// an error rather than a fall back to a cold dir, which would show only as
/// slow or timed-out builds.
pub async fn prepare(handle: &PluginHandle) -> Result<PathBuf> {
    let warm = warm_target_dir()?;
    let template = template_dir()?;
    std::fs::create_dir_all(&warm)
        .with_context(|| format!("creating warm target dir {}", warm.display()))?;
    write_template(&template)?;
    let manifest = template.join("Cargo.toml");

    // 2026-09-26: `cargo test --no-run` builds the debug dependencies the
    // agent's own `cargo test` and `cargo run` use; `cargo build --release`
    // builds the release ones the scorer uses.
    handle.check_cancelled()?;
    handle.status(format!("warming cargo debug profile in {}", warm.display()));
    cargo(&["test", "--no-run"], &warm, &manifest).await?;
    handle.check_cancelled()?;
    handle.status(format!(
        "warming cargo release profile in {}",
        warm.display()
    ));
    cargo(&["build", "--release"], &warm, &manifest).await?;
    Ok(warm)
}

#[cfg(test)]
#[path = "warm_tests.rs"]
mod tests;
