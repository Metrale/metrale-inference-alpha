// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Per-iteration trajectory file: the prompt, then each turn's
//! reasoning, text, tool calls and finish reason, and each tool result, as
//! plain text so `diff` of two runs shows the first turn at which they split.
//! The outcome columns (`webserver_ok`, steps, turns) can agree while the
//! trajectories differ.
//!
//! Owner: bench, agentic.
//! Invariants:
//! - The file is written beside the sandbox (`run-07` gives
//!   `run-07.trajectory.txt`), never inside it: the agent's `read`, `glob` and
//!   `grep` tools and the scorer's `has_tests` walk the sandbox.
//! - A write failure is logged and never returned as an error.

use std::io::Write;
use std::path::{Path, PathBuf};

use crate::http::ChatOutcome;

pub struct Trace {
    path: Option<PathBuf>,
}

impl Trace {
    /// 2026-09-26: Truncates a trace left by an earlier run of the same index,
    /// so the file holds one run.
    pub fn start(sandbox: &Path, prompt: &str) -> Self {
        let path = path_for(sandbox);
        let trace = Self { path: Some(path) };
        match std::fs::write(trace.path.as_ref().expect("just set"), header(prompt)) {
            Ok(()) => trace,
            Err(e) => {
                tracing::warn!("agentic: trajectory trace disabled: {e}");
                Self { path: None }
            }
        }
    }

    /// 2026-09-26: One model turn: its reasoning, its reply, the calls it asked
    /// for and its finish reason. Empty reasoning or text sections are omitted.
    pub fn turn(&self, index: usize, outcome: &ChatOutcome) {
        let mut s = format!("\n── turn {} ───────────────────────────────\n", index + 1);
        section(&mut s, "reasoning", &outcome.reasoning);
        section(&mut s, "text", &outcome.text);
        for call in &outcome.tool_calls {
            s.push_str(&format!(
                "[call {}] {} {}\n",
                call.id, call.name, call.arguments
            ));
        }
        if let Some(reason) = &outcome.finish_reason {
            s.push_str(&format!("[finish] {reason}\n"));
        }
        self.append(&s);
    }

    /// 2026-09-26: One tool result, as the content sent back to the model
    /// (truncated, and for shell output normalised), with trailing whitespace
    /// trimmed.
    pub fn result(&self, tool: &str, content: &str) {
        let mut s = format!("[result {tool}]\n");
        s.push_str(content.trim_end());
        s.push('\n');
        self.append(&s);
    }

    fn append(&self, text: &str) {
        let Some(path) = &self.path else { return };
        let written = std::fs::OpenOptions::new()
            .append(true)
            .open(path)
            .and_then(|mut f| f.write_all(text.as_bytes()));
        if let Err(e) = written {
            tracing::warn!("agentic: could not extend {}: {e}", path.display());
        }
    }
}

fn path_for(sandbox: &Path) -> PathBuf {
    let mut name = sandbox.file_name().unwrap_or_default().to_os_string();
    name.push(".trajectory.txt");
    sandbox.with_file_name(name)
}

fn header(prompt: &str) -> String {
    format!("[prompt]\n{}\n", prompt.trim_end())
}

fn section(out: &mut String, name: &str, body: &str) {
    if body.trim().is_empty() {
        return;
    }
    out.push_str(&format!("[{name}]\n{}\n", body.trim_end()));
}

#[cfg(test)]
#[path = "trace_tests.rs"]
mod tests;
