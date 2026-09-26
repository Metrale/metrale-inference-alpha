// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Running the shell command the model wrote, and bounding its time,
//! memory and output.
//!
//! Owner: bench, agentic.
//! Invariants:
//! - Each command runs as `sh -c` with the sandbox as its working directory
//!   and, on unix, in its own process group.
//! - While a command runs, at most `2 * CAPTURE_END` bytes of each stream are
//!   held.
//! - [`truncate`] returns at most `MAX_TOOL_OUTPUT` bytes.

use std::collections::VecDeque;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use parking_lot::Mutex;
use tokio::io::AsyncReadExt;

use super::{AgentConfig, DRAIN_GRACE, MAX_TOOL_OUTPUT, norm};

/// 2026-09-26: Bytes of one stream kept at each end while the command runs
/// (see [`Capture`]); the middle is counted and dropped. [`truncate`] later
/// cuts the result to `MAX_TOOL_OUTPUT`.
const CAPTURE_END: usize = 16 * MAX_TOOL_OUTPUT;

/// 2026-09-26: Room reserved for the elision note, so [`truncate`] returns at
/// most `MAX_TOOL_OUTPUT` bytes and a second `truncate` (the agent loop
/// truncates every tool result) returns its input unchanged.
const ELISION_NOTE: usize = 96;

#[cfg(test)]
pub(super) const TEST_ELISION_NOTE: usize = ELISION_NOTE;

pub(crate) async fn run_shell(cfg: &AgentConfig, command: &str, limit: Duration) -> Result<String> {
    let mut cmd = tokio::process::Command::new("sh");
    cmd.arg("-c")
        .arg(command)
        .current_dir(&cfg.sandbox)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        // 2026-09-26: The child is killed if this future is dropped before it
        // exits.
        .kill_on_drop(true)
        // 2026-09-26: `run_tier.sh` sets the same variable for opencode.
        // `bench/fp8_dgx2_drift/harness/cargo-shim/cargo`, when it is on PATH,
        // reads it to detach `cargo run`. It is set on this child only, so the
        // scorer's own `cargo run` is not detached.
        .env("METRALE_AGENT_SHELL", "1");
    // 2026-09-26: Its own process group, so the timeout can kill everything the
    // command started. A `setsid` server leaves the group; `agent::reap`
    // handles those.
    #[cfg(unix)]
    cmd.process_group(0);
    if let Some(dir) = &cfg.cargo_target_dir {
        cmd.env("CARGO_TARGET_DIR", dir);
    }
    let mut child = cmd.spawn()?;
    let pid = child.id();
    let (out, err) = (Arc::default(), Arc::default());
    // 2026-09-26: Drain concurrently with the wait: output past the pipe buffer
    // blocks the writer until someone reads it.
    let pumps = (
        tokio::spawn(pump(child.stdout.take(), Arc::clone(&out))),
        tokio::spawn(pump(child.stderr.take(), Arc::clone(&err))),
    );
    // 2026-09-26: Wait for the process, not for end-of-pipe: a `setsid cmd &`
    // that inherits stdout keeps the pipe open after `sh` has exited.
    let status = match tokio::time::timeout(limit, child.wait()).await {
        Ok(s) => Some(s?),
        Err(_) => {
            // 2026-09-26: Kill the whole group, so a child the command forked
            // does not keep running until `agent::reap` at the end of the
            // iteration.
            if let Some(pid) = pid {
                kill_group(pid).await;
            }
            let _ = child.kill().await;
            None
        }
    };
    let aborts = (pumps.0.abort_handle(), pumps.1.abort_handle());
    let _ = tokio::time::timeout(DRAIN_GRACE, async {
        let _ = pumps.0.await;
        let _ = pumps.1.await;
    })
    .await;
    // 2026-09-26: A pump still running after the grace is reading a pipe that
    // something not killed holds open; abort it so it does not live on.
    aborts.0.abort();
    aborts.1.abort();
    let mut text = out.lock().text();
    let stderr = err.lock().text();
    if !stderr.trim().is_empty() {
        // 2026-09-26: Non-empty stderr is appended on every path, including the
        // timeout.
        text.push_str("\n[stderr]\n");
        text.push_str(&stderr);
    }
    match status {
        Some(s) if !s.success() => text.push_str(&format!("\n[exit {s}]")),
        Some(_) => {}
        None => text.push_str(&format!(
            "\n[timed out after {}s and was killed; the output above is what it had produced. \
             If this was a server, start it detached with its output redirected to a file.]",
            limit.as_secs()
        )),
    }
    // 2026-09-26: Normalise before truncating: `truncate` cuts at byte offsets
    // and reports an elided count, so both would vary with the text that
    // [`norm`] rewrites.
    Ok(truncate(&norm::normalize(&text)))
}

/// 2026-09-26: Copies raw bytes into the capture, which decodes them once, so a
/// UTF-8 sequence split across two reads is not mangled.
async fn pump<R: AsyncReadExt + Unpin>(reader: Option<R>, sink: Arc<Mutex<Capture>>) {
    let Some(mut reader) = reader else { return };
    let mut buf = [0u8; 8192];
    while let Ok(n) = reader.read(&mut buf).await {
        if n == 0 {
            return;
        }
        sink.lock().push(&buf[..n]);
    }
}

/// 2026-09-26: A bounded window over one stream: the first [`CAPTURE_END`]
/// bytes and the last [`CAPTURE_END`], with the bytes between counted and
/// dropped. Reading never stops to enforce the bound, because a writer blocked
/// on a full pipe would be reported as a timeout.
#[derive(Default)]
pub(crate) struct Capture {
    head: Vec<u8>,
    tail: VecDeque<u8>,
    dropped: usize,
}

impl Capture {
    fn push(&mut self, mut bytes: &[u8]) {
        if self.head.len() < CAPTURE_END {
            let n = (CAPTURE_END - self.head.len()).min(bytes.len());
            self.head.extend_from_slice(&bytes[..n]);
            bytes = &bytes[n..];
        }
        if bytes.len() >= CAPTURE_END {
            self.dropped += self.tail.len() + bytes.len() - CAPTURE_END;
            self.tail.clear();
            self.tail.extend(&bytes[bytes.len() - CAPTURE_END..]);
            return;
        }
        let overflow = (self.tail.len() + bytes.len()).saturating_sub(CAPTURE_END);
        self.dropped += overflow;
        drop(self.tail.drain(..overflow));
        self.tail.extend(bytes);
    }

    /// 2026-09-26: Decoded once, over the concatenation, so a multi-byte
    /// character split across the head/tail seam is not mangled.
    fn text(&self) -> String {
        let mut bytes = self.head.clone();
        if self.dropped > 0 {
            bytes.extend_from_slice(
                format!("\n… [{} bytes dropped from the middle] …\n", self.dropped).as_bytes(),
            );
        }
        bytes.extend(self.tail.iter().copied());
        String::from_utf8_lossy(&bytes).into_owned()
    }

    #[cfg(test)]
    fn held(&self) -> usize {
        self.head.len() + self.tail.len()
    }
}

/// 2026-09-26: SIGKILL the process group led by `pid` (the child was spawned
/// with `process_group(0)`). The `--` ends option parsing, so `-<pid>` is read
/// as a group. `a_timed_out_command_takes_the_children_it_forked_with_it`
/// (agent_shell_tests.rs) checks that the group dies.
#[cfg(unix)]
async fn kill_group(pid: u32) {
    let _ = tokio::process::Command::new("kill")
        .arg("-9")
        .arg("--")
        .arg(format!("-{pid}"))
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .await;
}

#[cfg(not(unix))]
async fn kill_group(_pid: u32) {}

/// 2026-09-26: Keep the head and tail of text longer than `MAX_TOOL_OUTPUT`
/// bytes, with a note giving the elided length in bytes. The result is at
/// most `MAX_TOOL_OUTPUT` bytes, so truncating it again returns it unchanged.
pub fn truncate(text: &str) -> String {
    if text.len() <= MAX_TOOL_OUTPUT {
        return text.to_string();
    }
    let keep = MAX_TOOL_OUTPUT - ELISION_NOTE;
    let (mut cut, mut from) = (keep / 2, text.len() - keep / 2);
    while !text.is_char_boundary(cut) {
        cut -= 1;
    }
    while !text.is_char_boundary(from) {
        from += 1;
    }
    let (head, tail) = (&text[..cut], &text[from..]);
    let elided = text.len() - head.len() - tail.len();
    format!("{head}\n… [{elided} characters elided from the middle] …\n{tail}")
}

#[cfg(test)]
#[path = "agent_shell_tests.rs"]
mod tests;
