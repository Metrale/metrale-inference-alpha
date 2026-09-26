// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Scoring for the agentic webserver task, ported from the
//! harness's `score_run.py` and `followed_directions.py`. Two separate axes:
//!
//! * `webserver_ok`, the outcome: the scorer builds and runs the code the agent
//!   left behind and asks `/ping` for `pong`, whether or not the agent built or
//!   checked anything itself.
//! * `followed_directions`, the process: whether the commands the agent ran and
//!   the tree it left evidence each step in `REQUIRED_STEPS`.
//!
//! Owner: bench, agentic.
//! Invariants: `followed_directions` returns one entry per `REQUIRED_STEPS`
//! name, in that order.

use std::path::Path;
use std::process::Stdio;
use std::time::Duration;

use anyhow::Result;

/// 2026-09-26: The prompt-mandated process steps; `Directions::overall` is
/// their AND.
pub const REQUIRED_STEPS: [&str; 6] = [
    "wrote_project",
    "wrote_tests",
    "ran_tests",
    "ran_server",
    "curled",
    "tore_down",
];

#[derive(Clone, Debug, Default)]
pub struct WebserverResult {
    pub webserver_ok: bool,
    pub build_ok: bool,
    pub error: String,
    pub port_used: u16,
}

#[derive(Clone, Debug, Default)]
pub struct Directions {
    pub steps: Vec<(&'static str, bool)>,
}

impl Directions {
    /// 2026-09-26: True only when there are steps and every one is evidenced.
    pub fn overall(&self) -> bool {
        !self.steps.is_empty() && self.steps.iter().all(|(_, ok)| *ok)
    }
    /// 2026-09-26: The steps that were not evidenced, in declaration order.
    pub fn missing(&self) -> Vec<&'static str> {
        self.steps
            .iter()
            .filter(|(_, ok)| !*ok)
            .map(|(name, _)| *name)
            .collect()
    }
    pub fn met(&self) -> usize {
        self.steps.iter().filter(|(_, ok)| *ok).count()
    }
}

/// 2026-09-26: An OS-assigned port, released on return, for a caller that binds
/// it soon after (the self-started gate server in the server crate's
/// `bench_selfstart` and `bench_lease`). The scorer holds its port through the
/// build with `reserve_port` instead.
pub fn free_port() -> Result<u16> {
    let listener = reserve_port()?;
    Ok(listener.local_addr()?.port())
}

/// 2026-09-26: An OS-assigned port on 127.0.0.1, held as a listener. A fresh
/// port per iteration means a server left over from an earlier iteration can
/// neither collide with this one nor answer its probe. Holding the listener
/// through the build stops another process from taking the port meanwhile.
fn reserve_port() -> Result<std::net::TcpListener> {
    Ok(std::net::TcpListener::bind("127.0.0.1:0")?)
}

/// 2026-09-26: Builds the project in release, runs it on a reserved port and
/// checks that `/ping` answers `pong` within `serve_timeout`. Every failure is
/// reported in `WebserverResult::error`, not as an `Err`.
pub async fn webserver_test(
    sandbox: &Path,
    cargo_target_dir: Option<&Path>,
    build_timeout: Duration,
    serve_timeout: Duration,
) -> WebserverResult {
    let mut out = WebserverResult::default();
    // 2026-09-26: As in `score_run.py`'s `webserver_test`, no build is tried
    // without a `Cargo.toml` and a `src/`.
    if !sandbox.join("Cargo.toml").is_file() {
        out.error = "no Cargo.toml was written".into();
        return out;
    }
    if !sandbox.join("src").is_dir() {
        out.error = "no src/ was written — skipping webserver test".into();
        return out;
    }
    let port_reservation = match reserve_port() {
        Ok(reservation) => reservation,
        Err(e) => {
            out.error = format!("could not reserve a port: {e}");
            return out;
        }
    };
    let port = match port_reservation.local_addr() {
        Ok(address) => address.port(),
        Err(e) => {
            out.error = format!("could not inspect the reserved port: {e}");
            return out;
        }
    };
    out.port_used = port;

    let mut build = tokio::process::Command::new("cargo");
    build
        .args(["build", "--release"])
        .current_dir(sandbox)
        // 2026-09-26: Set for the build too, as `score_run.py` does, since a
        // project may read it with `env!`. The listener still holds the port.
        .env("METRALE_HARNESS_PORT", port.to_string())
        // 2026-09-26: The agent's shell sets METRALE_AGENT_SHELL, and the
        // harness's cargo shim detaches `cargo run` when it is set; the scorer
        // must not inherit it, or the server it supervises would detach.
        .env_remove("METRALE_AGENT_SHELL")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    if let Some(dir) = cargo_target_dir {
        build.env("CARGO_TARGET_DIR", dir);
    }
    match tokio::time::timeout(build_timeout, build.output()).await {
        Ok(Ok(o)) if o.status.success() => out.build_ok = true,
        Ok(Ok(o)) => {
            let err = String::from_utf8_lossy(&o.stderr);
            out.error =
                super::super::one_line(err.lines().rev().take(6).collect::<Vec<_>>().join(" "));
            return out;
        }
        Ok(Err(e)) => {
            out.error = format!("cargo build could not start: {e}");
            return out;
        }
        Err(_) => {
            out.error = format!("cargo build exceeded {}s", build_timeout.as_secs());
            return out;
        }
    }

    // 2026-09-26: The server's stderr goes to a file, as in `score_run.py`, so
    // a bind panic shows in the error instead of a bare `/ping` timeout.
    let err_log = std::env::temp_dir().join(format!(
        "metrale-ws-stderr-{}-{port}.log",
        std::process::id()
    ));
    // 2026-09-26: `create_new`, because the name is predictable and the code
    // being scored may still have processes running: `create` would follow a
    // symlink planted at that path and truncate its target. If the file cannot
    // be created, stderr is discarded.
    let sink = match std::fs::File::options()
        .write(true)
        .create_new(true)
        .open(&err_log)
    {
        Ok(f) => Stdio::from(f),
        Err(_) => Stdio::null(),
    };

    let mut serve = tokio::process::Command::new("cargo");
    serve
        .args(["run", "--release"])
        .current_dir(sandbox)
        // 2026-09-26: The prompt tells the model to read this variable. A
        // project that hardcodes a port binds elsewhere and correctly fails.
        .env("METRALE_HARNESS_PORT", port.to_string())
        // 2026-09-26: As `score_run.py` sets for the server.
        .env("RUST_LOG", "warn")
        .env_remove("METRALE_AGENT_SHELL")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(sink)
        .kill_on_drop(true);
    if let Some(dir) = cargo_target_dir {
        serve.env("CARGO_TARGET_DIR", dir);
    }
    // 2026-09-26: The port is released just before the spawn; a short race
    // with other processes remains.
    drop(port_reservation);
    let mut child = match serve.spawn() {
        Ok(c) => c,
        Err(e) => {
            out.error = format!("cargo run could not start: {e}");
            let _ = std::fs::remove_file(&err_log);
            return out;
        }
    };
    // 2026-09-26: `child` is killed on drop (`kill_on_drop` above), so every
    // return below also kills it.
    let deadline = tokio::time::Instant::now() + serve_timeout;
    let mut exited = None;
    while tokio::time::Instant::now() < deadline {
        if let Some(body) = ping(port).await
            && body.to_lowercase().contains("pong")
        {
            out.webserver_ok = true;
            let _ = child.kill().await;
            let _ = std::fs::remove_file(&err_log);
            return out;
        }
        // 2026-09-26: An exited process will never answer; stop now and report
        // its exit status instead of a timeout.
        if let Ok(Some(status)) = child.try_wait() {
            exited = Some(status);
            break;
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    let _ = child.kill().await;
    let detail = server_stderr(&err_log);
    let _ = std::fs::remove_file(&err_log);
    out.error = match exited {
        Some(status) => format!("server exited ({status}) before answering /ping{detail}"),
        None => format!(
            "/ping did not answer 'pong' within {}s{detail}",
            serve_timeout.as_secs()
        ),
    };
    out
}

/// 2026-09-26: The last 800 characters of the server's stderr, with a `port in
/// use` note when it contains `Address already in use` or `EADDRINUSE`, as in
/// `score_run.py`. Empty when the file is empty or unreadable.
fn server_stderr(path: &Path) -> String {
    let text = std::fs::read_to_string(path).unwrap_or_default();
    if text.trim().is_empty() {
        return String::new();
    }
    let mut note = String::new();
    if text.contains("Address already in use") || text.contains("EADDRINUSE") {
        note.push_str(" | server bind failed (port in use)");
    }
    let mut tail: Vec<char> = text.chars().rev().take(800).collect();
    tail.reverse();
    let tail: String = tail.into_iter().collect();
    format!("{note} | stderr: {}", super::super::one_line(tail))
}

/// 2026-09-26: One `GET /ping`. `None` when the connection or the request
/// write fails, which is normal while the server is starting.
async fn ping(port: u16) -> Option<String> {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let mut sock = tokio::time::timeout(
        Duration::from_millis(500),
        tokio::net::TcpStream::connect(("127.0.0.1", port)),
    )
    .await
    .ok()?
    .ok()?;
    let req = format!("GET /ping HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n");
    sock.write_all(req.as_bytes()).await.ok()?;
    // 2026-09-26: Raw bytes for up to 2 s, then whatever arrived, as `curl -sS
    // -m 2` in `score_run.py` does, so a server that ignores `Connection:
    // close` is still read.
    let mut buf = Vec::new();
    let read = async {
        let mut chunk = [0u8; 4096];
        while let Ok(n) = sock.read(&mut chunk).await {
            if n == 0 || buf.len() > 64 * 1024 {
                break;
            }
            buf.extend_from_slice(&chunk[..n]);
        }
    };
    let _ = tokio::time::timeout(Duration::from_secs(2), read).await;
    Some(String::from_utf8_lossy(&buf).into_owned())
}

/// 2026-09-26: Whether the agent performed each step in `REQUIRED_STEPS`. The
/// evidence is the shell commands it ran and the tree it left, the two sources
/// `followed_directions.py` uses.
pub fn followed_directions(commands: &[String], sandbox: &Path) -> Directions {
    let joined = commands.join("\n").to_lowercase();
    // 2026-09-26: Ports of the `_RE_*` detectors in `followed_directions.py`,
    // matched on the lowercased commands. Each is anchored on word boundaries,
    // so "killed" or "skill" is not evidence of `kill`.
    let wrote_project = regular(&sandbox.join("Cargo.toml")) && has_main(sandbox);
    let wrote_tests = has_tests(sandbox);
    let ran_tests = contains_cargo(&joined, &["test", "nextest"]);
    let ran_server = contains_cargo(&joined, &["run"]) || ran_binary(&joined);
    let curled = ["curl", "wget", "httpie", "httpx"]
        .iter()
        .any(|k| word(&joined, k))
        || word_then(&joined, "nc", "-z");
    let tore_down =
        word(&joined, "kill") || word(&joined, "pkill") || word_then(&joined, "fuser", "-k");
    Directions {
        steps: REQUIRED_STEPS
            .iter()
            .zip([
                wrote_project,
                wrote_tests,
                ran_tests,
                ran_server,
                curled,
                tore_down,
            ])
            .map(|(name, ok)| (*name, ok))
            .collect(),
    }
}

fn is_word(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// 2026-09-26: Byte offsets just past every occurrence of `needle` that starts
/// on a word boundary.
fn after_word_start<'a>(hay: &'a str, needle: &'a str) -> impl Iterator<Item = usize> + 'a {
    hay.match_indices(needle)
        .filter(|(i, _)| !hay[..*i].chars().next_back().is_some_and(is_word))
        .map(|(i, m)| i + m.len())
}

/// 2026-09-26: `\bneedle\b`.
fn word(hay: &str, needle: &str) -> bool {
    after_word_start(hay, needle).any(|end| !hay[end..].starts_with(is_word))
}

/// 2026-09-26: `\bfirst\s+second\b`, the shape of `_RE_KILL`'s `fuser -k` and
/// `_RE_CURL`'s `nc -z`.
fn word_then(hay: &str, first: &str, second: &str) -> bool {
    after_word_start(hay, first).any(|end| {
        let rest = &hay[end..];
        let head = rest.trim_start();
        head.len() < rest.len()
            && head
                .strip_prefix(second)
                .is_some_and(|after| !after.starts_with(is_word))
    })
}

/// 2026-09-26: `\btarget/(?:debug|release)/\S`, the part of `_RE_RUN` for a
/// binary run directly instead of through `cargo run`.
fn ran_binary(hay: &str) -> bool {
    ["target/debug/", "target/release/"].iter().any(|p| {
        after_word_start(hay, p).any(|end| hay[end..].starts_with(|c: char| !c.is_whitespace()))
    })
}

/// 2026-09-26: `\bcargo\s+<sub>\b` for any of `subs`, the shape of `_RE_TEST`
/// and `_RE_RUN`.
fn contains_cargo(haystack: &str, subs: &[&str]) -> bool {
    after_word_start(haystack, "cargo").any(|end| {
        let rest = &haystack[end..];
        let head = rest.trim_start();
        head.len() < rest.len()
            && subs.iter().any(|s| {
                head.strip_prefix(s)
                    .is_some_and(|after| !after.starts_with(is_word))
            })
    })
}

/// 2026-09-26: A regular `src/main.rs`, or a `main.rs` anywhere in the walked
/// tree, as `followed_directions.py` requires for `wrote_project`.
fn has_main(sandbox: &Path) -> bool {
    regular(&sandbox.join("src/main.rs"))
        || walk(sandbox).any(|p| p.file_name().is_some_and(|n| n == "main.rs"))
}

/// 2026-09-26: A real `tests/` directory holding a `.rs` file, or `#[test]`,
/// `#[cfg(test)]` or `#[tokio::test]` in any walked `.rs` file, as
/// `followed_directions.py`'s `_has_tests` checks.
fn has_tests(sandbox: &Path) -> bool {
    let tests = sandbox.join("tests");
    let real_dir = std::fs::symlink_metadata(&tests).is_ok_and(|m| m.is_dir());
    if real_dir && walk(&tests).any(|p| p.extension().is_some_and(|e| e == "rs")) {
        return true;
    }
    walk(sandbox)
        .filter(|p| p.extension().is_some_and(|e| e == "rs"))
        .filter_map(|p| std::fs::read_to_string(p).ok())
        .any(|s| {
            s.contains("#[test]") || s.contains("#[cfg(test)]") || s.contains("#[tokio::test]")
        })
}

/// 2026-09-26: Every regular file under `root`, recursively, skipping any
/// directory named `target` or `.git`; unreadable directories are skipped.
///
/// Symlinks are neither followed nor collected, because the tree was written by
/// the agent being scored. A few `ln -s . a` links would make the walk explode
/// combinatorially, and it has no timeout. And a link is not evidence: `ln -s
/// ~/metrale/tests/foo.rs tests/foo.rs` would otherwise credit `wrote_tests` to
/// an agent that wrote no tests.
fn walk(root: &Path) -> impl Iterator<Item = std::path::PathBuf> {
    let mut stack = vec![root.to_path_buf()];
    let mut files = Vec::new();
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let name = entry.file_name();
            if matches!(name.to_str(), Some("target") | Some(".git")) {
                continue;
            }
            let Ok(kind) = entry.file_type() else {
                continue;
            };
            match (kind.is_dir(), kind.is_file()) {
                (true, _) => stack.push(entry.path()),
                (_, true) => files.push(entry.path()),
                _ => {}
            }
        }
    }
    files.into_iter()
}

/// 2026-09-26: A regular file, not a symlink to one.
fn regular(path: &Path) -> bool {
    std::fs::symlink_metadata(path).is_ok_and(|m| m.is_file())
}

#[cfg(test)]
#[path = "score_tests.rs"]
mod tests;

/// 2026-09-26: How many iterations evidenced each step, keyed `step:<name>`, in
/// the first iteration's step order; empty for no iterations.
/// `followed_directions` is all-or-nothing per iteration, so without this the
/// tier record cannot say which step failed.
pub fn per_step_tallies(all: &[&Directions]) -> Vec<(String, f64)> {
    let Some(first) = all.first() else {
        return Vec::new();
    };
    first
        .steps
        .iter()
        .map(|(name, _)| {
            let met = all
                .iter()
                .filter(|d| d.steps.iter().any(|(n, ok)| n == name && *ok))
                .count();
            (format!("step:{name}"), met as f64)
        })
        .collect()
}
