// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The agentic benchmark's agent loop: tool-calling turns against the
//! served endpoint, with the tools run inside a per-iteration sandbox directory.
//! Its tools, prompt and caps are modelled on the opencode client that
//! `bench/fp8_dgx2_drift/harness/run_tier.sh` drives.
//!
//! Owner: bench, agentic.
//! Invariants:
//! - The loop runs at most `max_turns` turns.
//! - Every tool result sent back to the model has passed through [`truncate`].
//! - `run_task` reaps processes whose working directory is in the sandbox after
//!   the loop returns, whether it returned `Ok` or `Err`.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use anyhow::Result;
use serde_json::{Value, json};

#[path = "agent_history.rs"]
mod history;
#[path = "norm.rs"]
pub mod norm;
#[path = "agent_path.rs"]
mod path_guard;
#[path = "agent_shell.rs"]
pub mod shell;
#[path = "agent_tools.rs"]
pub mod tools;
#[path = "trace.rs"]
pub mod trace;
use history::{assistant_message, call_id, compact, preserve_thinking};
pub use path_guard::resolve;
pub(crate) use shell::run_shell;
pub use shell::truncate;
pub use tools::{glob_match, tool_schema};

/// 2026-09-26: Bytes of one tool result; [`truncate`] keeps the head and tail of a
/// longer one.
pub(super) const MAX_TOOL_OUTPUT: usize = 8192;

/// 2026-09-26: Characters of conversation (content plus reasoning) above which
/// [`compact`] elides old reasoning, then old tool results.
const HISTORY_BUDGET: usize = 96_000;

/// 2026-09-26: The most recent tool results, which [`compact`] never elides.
const LIVE_TOOL_RESULTS: usize = 4;

/// 2026-09-26: The most recent assistant turns whose reasoning [`compact`] never
/// elides; older reasoning is elided before any tool result.
const LIVE_REASONING: usize = 4;

/// 2026-09-26: The request temperature: greedy, so that two runs of the same
/// build can repeat. `request_body` sends it unless `METRALE_AGENTIC_SAMPLING`
/// is `model-card`.
const TEMPERATURE: f64 = 0.0;

/// 2026-09-26: The request seed, sent with [`TEMPERATURE`].
const SEED: u64 = 0;

/// 2026-09-26: How long `run_shell` waits for its output pumps after the command
/// exits; it then aborts them, because a pipe inherited by a detached child
/// never reaches EOF.
pub(super) const DRAIN_GRACE: Duration = Duration::from_secs(2);

/// 2026-09-26: The agent prompt. The six tools it lists are the ones
/// [`tool_schema`] defines.
const AGENT_PROMPT: &str = "\
You are a coding assistant running locally on Metrale Engine. No data leaves this machine.

You have access to tools for interacting with the filesystem and running commands:
- **bash**: Execute shell commands (ls, cat, grep, find, git, etc.)
- **read**: Read file contents
- **write**: Create or overwrite files
- **edit**: Edit existing files (find and replace)
- **glob**: Find files matching a pattern
- **grep**: Search file contents with regex

When asked to list files, check directories, or run commands, use the **bash** tool.
When asked to read a file, use the **read** tool.

IMPORTANT: Think briefly, then act. Do NOT describe tool calls in your thinking — just make \
them directly. Keep thinking short (under 50 words). Never put tool calls inside thinking tags. \
Use the write tool (not edit) when creating new files.";

/// 2026-09-26: What one agent run did, for scoring and diagnostics.
#[derive(Default)]
pub struct Transcript {
    /// 2026-09-26: Every shell command the agent issued, in order; the input of
    /// `score::followed_directions`.
    pub commands: Vec<String>,
    pub turns: usize,
    pub tool_calls: usize,
    /// 2026-09-26: True when the loop ended at the turn cap rather than because
    /// the agent stopped calling tools.
    pub hit_turn_cap: bool,
    /// 2026-09-26: Turns that stopped at `max_tokens` with no tool call and were
    /// resumed instead of ending the run.
    pub truncated_turns: usize,
    /// 2026-09-26: Turns with tool-call syntax in the content and no parsed tool
    /// call, which were re-asked instead of ending the run.
    pub unparsed_call_turns: usize,
    /// 2026-09-26: Sum over turns of each stream's `ChatOutcome::completion_tokens`.
    pub completion_tokens: usize,
    pub final_text: String,
}

pub struct AgentConfig {
    pub sandbox: PathBuf,
    pub max_turns: usize,
    pub command_timeout: Duration,
    pub request_timeout: Duration,
    pub max_tokens: usize,
    /// 2026-09-26: Exported as `CARGO_TARGET_DIR` to every shell command, so the
    /// agent's builds reuse the dependencies `warm::prepare` compiled.
    pub cargo_target_dir: Option<PathBuf>,
}

/// 2026-09-26: The system message: the agent prompt, the model name, and an
/// environment block naming the sandbox as the working directory. It carries
/// no date or other per-run value except the sandbox path.
fn system_prompt(sandbox: &Path, model: &str) -> String {
    let dir = sandbox.display();
    format!(
        "{AGENT_PROMPT}\nYou are powered by the model named {model}. The exact model ID is \
         {model}\nHere is some useful information about the environment you are running in:\n\
         <env>\n  Working directory: {dir}\n  Workspace root folder: {dir}\n  \
         Is directory a git repo: no\n  Platform: linux\n</env>"
    )
}

/// 2026-09-26: Run one agentic task until the agent stops calling tools or the
/// turn cap is reached.
pub async fn run_task(
    handle: &crate::plugin::PluginHandle,
    cfg: &AgentConfig,
    prompt: &str,
) -> Result<Transcript> {
    let mut transcript = Transcript::default();
    let outcome = agent_loop(handle, cfg, prompt, &mut transcript).await;
    // 2026-09-26: Reap on every path, including a transport error, so nothing the
    // agent started outlives the iteration.
    reap(&cfg.sandbox).await;
    outcome.map(|()| transcript)
}

/// 2026-09-26: SIGKILL every process, other than this one, whose working
/// directory is inside the sandbox. A server started with `setsid`, as the
/// prompt asks, is not our child, so `kill_on_drop` cannot reach it.
/// `run_tier.sh` reaps by working directory the same way. Without `/proc` this
/// does nothing.
async fn reap(sandbox: &Path) {
    let real = std::fs::canonicalize(sandbox).unwrap_or_else(|_| sandbox.to_path_buf());
    let Ok(entries) = std::fs::read_dir("/proc") else {
        return;
    };
    let me = std::process::id().to_string();
    let victims: Vec<String> = entries
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()) && *n != me)
        .filter(|pid| {
            std::fs::read_link(format!("/proc/{pid}/cwd")).is_ok_and(|c| c.starts_with(&real))
        })
        .collect();
    if !victims.is_empty() {
        let _ = tokio::process::Command::new("kill")
            .arg("-9")
            .args(&victims)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .await;
    }
}

async fn agent_loop(
    handle: &crate::plugin::PluginHandle,
    cfg: &AgentConfig,
    prompt: &str,
    transcript: &mut Transcript,
) -> Result<()> {
    let target = handle.target();
    let mut messages = vec![
        json!({"role": "system", "content": system_prompt(&cfg.sandbox, &target.model)}),
        json!({"role": "user", "content": prompt}),
    ];
    let tools = tool_schema();
    let trace = trace::Trace::start(&cfg.sandbox, prompt);

    for turn in 0..cfg.max_turns {
        handle.check_cancelled()?;
        handle.status(format!("agent turn {}/{}", turn + 1, cfg.max_turns));
        compact(&mut messages);
        let body = request_body(&target.model, &messages, &tools, cfg.max_tokens);
        let outcome = crate::http::chat_stream(target, &body, cfg.request_timeout).await?;
        transcript.turns = turn + 1;
        transcript.final_text = outcome.text.clone();
        transcript.completion_tokens += outcome.completion_tokens;
        trace.turn(turn, &outcome);

        if outcome.tool_calls.is_empty() {
            // 2026-09-26: A turn that stopped at `max_tokens` did not finish. Its
            // text goes back as the assistant turn, followed by a user message
            // saying it was cut off, and the loop continues.
            if was_cut_off(&outcome) {
                transcript.truncated_turns += 1;
                messages.push(json!({"role": "assistant", "content": outcome.text}));
                messages.push(json!({"role": "user", "content":
                    "Your previous message was cut off at the output limit before you \
                     finished. Do not repeat it. Continue from where it stopped, and make \
                     the tool call you intended."}));
                continue;
            }
            // 2026-09-26: A turn with tool-call syntax in its text and no parsed
            // call is re-asked instead of ending the run.
            if tools::emitted_unparsed_call(&outcome) {
                transcript.unparsed_call_turns += 1;
                messages.push(json!({"role": "assistant", "content": outcome.text}));
                messages.push(json!({"role": "user", "content":
                    "Your previous message contained tool-call syntax in the message body, so \
                     no tool actually ran. Re-issue exactly that one call as a real tool call, \
                     with nothing else in the message. If you are finished, say so in plain \
                     text with no tool-call syntax."}));
                continue;
            }
            return Ok(());
        }

        messages.push(assistant_message(&outcome, turn));
        for (i, call) in outcome.tool_calls.iter().enumerate() {
            handle.check_cancelled()?;
            transcript.tool_calls += 1;
            // 2026-09-26: A tool error becomes the tool result; it does not end
            // the run.
            let content = match tools::execute(cfg, call, &mut transcript.commands).await {
                Ok(text) => text,
                Err(e) => format!("error: {e:#}"),
            };
            let content = truncate(&content);
            trace.result(&call.name, &content);
            messages.push(json!({"role": "tool", "content": content,
                "tool_call_id": call_id(turn, i)}));
        }
    }
    transcript.hit_turn_cap = true;
    Ok(())
}

/// 2026-09-26: True when the turn has no tool call and stopped with
/// `finish_reason` `length`: it ran out of room rather than finishing.
fn was_cut_off(outcome: &crate::http::ChatOutcome) -> bool {
    outcome.tool_calls.is_empty() && outcome.finish_reason.as_deref() == Some("length")
}

/// 2026-09-26: One chat request. `the_gate_request_pins_sampling_messages_and_tools`
/// (agent_tests.rs) asserts its fields.
fn request_body(model: &str, messages: &[Value], tools: &Value, max_tokens: usize) -> Value {
    let mut body = json!({
        "model": model, "stream": true,
        "max_tokens": max_tokens, "messages": messages,
        "tools": tools, "tool_choice": "auto",
    });
    // 2026-09-26: With METRALE_AGENTIC_SAMPLING=model-card the request carries no
    // temperature or seed, so the server's defaults apply.
    if std::env::var("METRALE_AGENTIC_SAMPLING").as_deref() != Ok("model-card") {
        body["temperature"] = json!(TEMPERATURE);
        body["seed"] = json!(SEED);
    }
    // 2026-09-26: Only `preserve_thinking` is sent; other template kwargs such as
    // `reasoning_effort` are left to the server.
    if preserve_thinking() {
        body["chat_template_kwargs"] = json!({"preserve_thinking": true});
    }
    body
}

#[cfg(test)]
#[path = "agent_loop_tests.rs"]
mod loop_tests;
#[cfg(test)]
#[path = "agent_tests.rs"]
mod tests;
#[cfg(test)]
#[path = "agent_truncation_tests.rs"]
mod truncation_tests;
