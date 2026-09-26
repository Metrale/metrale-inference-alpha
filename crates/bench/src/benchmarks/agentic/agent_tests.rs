// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for the agent loop: path containment, truncation, the
//! request and message shapes, compaction, and the shell runner.
//!
//! Owner: bench, agentic.
//! Invariants: none beyond the types.

use super::*;
use crate::http;

/// 2026-09-26: A fresh directory under the temp dir, named by this process's id
/// and `name`, so tests do not share one.
pub fn sandbox(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("metrale-agent-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

pub fn cfg(sandbox: PathBuf) -> AgentConfig {
    AgentConfig {
        sandbox,
        max_turns: 1,
        command_timeout: Duration::from_secs(20),
        request_timeout: Duration::from_secs(1),
        max_tokens: 16,
        cargo_target_dir: None,
    }
}

#[test]
fn paths_cannot_escape_the_sandbox() {
    let sb = Path::new("/tmp/sandbox");
    assert_eq!(resolve(sb, "src/main.rs").unwrap(), sb.join("src/main.rs"));
    assert_eq!(resolve(sb, "./Cargo.toml").unwrap(), sb.join("Cargo.toml"));
    assert!(resolve(sb, "../../etc/passwd").is_err());
    assert!(resolve(sb, "/etc/passwd").is_err());
    assert!(resolve(sb, "src/../../../etc/shadow").is_err());
}

#[test]
fn an_absolute_path_inside_the_sandbox_is_accepted() {
    let sb = Path::new("/tmp/sandbox");
    assert_eq!(
        resolve(sb, "/tmp/sandbox/src/main.rs").unwrap(),
        sb.join("src/main.rs")
    );
    assert!(resolve(sb, "/tmp/sandbox/../escape").is_err());
}

#[cfg(unix)]
#[test]
fn a_symlink_out_of_the_sandbox_is_not_a_way_out_of_it() {
    // 2026-09-26: `escape/passwd` is lexically inside the sandbox; the symlink
    // check must refuse it.
    let sb = sandbox("symlink-escape");
    std::os::unix::fs::symlink("/etc", sb.join("escape")).unwrap();
    assert!(resolve(&sb, "escape/passwd").is_err());
    assert!(resolve(&sb, "escape").is_err());
    // 2026-09-26: A symlink that stays inside still resolves, and so does a path
    // that does not exist yet, as for every `write` of a new file.
    std::fs::create_dir(sb.join("src")).unwrap();
    std::os::unix::fs::symlink(sb.join("src"), sb.join("inside")).unwrap();
    assert!(resolve(&sb, "inside/main.rs").is_ok());
    assert!(resolve(&sb, "src/deep/new.rs").is_ok());
}

#[test]
fn truncation_keeps_both_ends() {
    let text = format!("{}ERROR_AT_END", "a".repeat(20_000));
    let t = truncate(&text);
    assert!(t.ends_with("ERROR_AT_END"), "tail must survive");
    assert!(t.starts_with("aaa"));
    assert!(t.contains("elided"));
    assert!(t.chars().count() < 20_100);
}

#[test]
fn output_at_or_below_the_cap_is_untouched() {
    assert_eq!(truncate("hello"), "hello");
    let at_cap = "x".repeat(MAX_TOOL_OUTPUT);
    assert_eq!(truncate(&at_cap), at_cap);
}

#[test]
fn truncation_caps_at_the_harness_output_cap_and_never_splits_a_char() {
    let text = "€".repeat(20_000);
    let cut = (MAX_TOOL_OUTPUT - shell::TEST_ELISION_NOTE) / 2;
    assert!(
        !text.is_char_boundary(cut),
        "fixture head cut is a character boundary"
    );
    assert!(
        !text.is_char_boundary(text.len() - cut),
        "fixture tail cut is a character boundary"
    );
    let t = truncate(&text);
    assert!(t.len() < MAX_TOOL_OUTPUT + 200, "{}", t.len());
    assert!(t.contains("characters elided from the middle"));
}

#[test]
fn assistant_message_substitutes_empty_arguments_with_an_object() {
    let outcome = http::ChatOutcome {
        tool_calls: vec![http::ToolCall {
            id: String::new(),
            name: "bash".into(),
            arguments: String::new(),
        }],
        ..Default::default()
    };
    let m = assistant_message(&outcome, 0);
    assert_eq!(m["tool_calls"][0]["function"]["arguments"], "{}");
    assert_eq!(m["tool_calls"][0]["id"], "call_0_0");
    assert!(m["content"].is_null());
}

#[test]
fn tool_call_ids_are_positional_and_never_the_servers() {
    // 2026-09-26: The ids the server returned are replaced by positional ones.
    let outcome = http::ChatOutcome {
        tool_calls: vec![
            http::ToolCall {
                id: "call_0000000000000004".into(),
                name: "bash".into(),
                arguments: "{}".into(),
            },
            http::ToolCall {
                id: "call_0000000000000005".into(),
                name: "read".into(),
                arguments: "{}".into(),
            },
        ],
        ..Default::default()
    };
    let m = assistant_message(&outcome, 3);
    assert_eq!(m["tool_calls"][0]["id"], "call_3_0");
    assert_eq!(m["tool_calls"][1]["id"], "call_3_1");
    // 2026-09-26: The assistant message's id is `call_id`, which the tool reply
    // also uses.
    assert_eq!(m["tool_calls"][1]["id"].as_str().unwrap(), call_id(3, 1));
    // 2026-09-26: Two turns never collide, so an old reply cannot pair with a
    // new call.
    assert_ne!(call_id(3, 1), call_id(4, 1));
}

#[test]
fn the_system_prompt_is_the_harness_agent_prompt_plus_the_environment() {
    let p = system_prompt(Path::new("/tmp/run-03"), "Qwen/Qwen3.6-35B-A3B-FP8");
    assert!(p.starts_with("You are a coding assistant running locally on Metrale Engine."));
    assert!(p.contains("Keep thinking short (under 50 words)"));
    assert!(p.contains("Working directory: /tmp/run-03"));
    assert!(p.contains("Qwen/Qwen3.6-35B-A3B-FP8"));
    // 2026-09-26: Every tool the prompt lists must exist in the schema.
    for name in ["bash", "read", "write", "edit", "glob", "grep"] {
        assert!(p.contains(&format!("**{name}**")), "{name}");
        assert!(
            tool_schema()
                .as_array()
                .unwrap()
                .iter()
                .any(|t| t["function"]["name"] == name),
            "{name}"
        );
    }
}

#[test]
fn the_gate_request_pins_sampling_messages_and_tools() {
    const { assert!(TEMPERATURE == 0.0) };
    let messages = [json!({"role": "user", "content": "hi"})];
    let body = request_body("Qwen/Qwen3.6-35B-A3B-FP8", &messages, &tool_schema(), 8192);
    assert_eq!(body["temperature"], 0.0);
    assert_eq!(body["seed"], SEED);
    assert_eq!(body["model"], "Qwen/Qwen3.6-35B-A3B-FP8");
    assert_eq!(body["stream"], true);
    assert_eq!(body["max_tokens"], 8192);
    assert_eq!(body["messages"], json!(messages));
    assert_eq!(body["tool_choice"], "auto");
    assert_eq!(body["tools"], tool_schema());
}

#[test]
fn compaction_elides_the_oldest_tool_results_and_keeps_the_pairing() {
    let big = "x".repeat(20_000);
    let mut msgs = vec![json!({"role": "system", "content": "s"})];
    for i in 0..10 {
        msgs.push(json!({"role": "assistant", "content": Value::Null,
            "tool_calls": [{"id": format!("c{i}")}]}));
        msgs.push(json!({"role": "tool", "tool_call_id": format!("c{i}"), "content": big}));
    }
    let before = msgs.len();
    compact(&mut msgs);

    assert_eq!(
        msgs.len(),
        before,
        "a dropped tool reply is a 400, not a saving"
    );
    let total: usize = msgs
        .iter()
        .map(|m| m["content"].as_str().map_or(64, str::len))
        .sum();
    assert!(total <= HISTORY_BUDGET, "{total}");
    assert!(msgs[2]["content"].as_str().unwrap().contains("elided"));
    // 2026-09-26: The newest tool result is kept whole.
    let last = msgs.last().unwrap()["content"].as_str().unwrap();
    assert_eq!(last.len(), big.len(), "the live window must survive intact");

    // 2026-09-26: Even when the budget stays exceeded, the last
    // `LIVE_TOOL_RESULTS` results are not elided.
    let live = "y".repeat(HISTORY_BUDGET);
    let mut pressured = vec![json!({"role": "system", "content": "s"})];
    for i in 0..5 {
        pressured.push(json!({"role": "assistant", "content": Value::Null,
            "tool_calls": [{"id": format!("p{i}")}]}));
        pressured.push(json!({"role": "tool", "tool_call_id": format!("p{i}"), "content": live}));
    }
    compact(&mut pressured);
    assert!(pressured[2]["content"].as_str().unwrap().contains("elided"));
    for i in 1..5 {
        assert_eq!(
            pressured[2 + 2 * i]["content"].as_str().unwrap().len(),
            live.len(),
            "live tool result {i} was compacted"
        );
    }
}

#[test]
fn a_session_below_the_history_budget_is_left_alone() {
    let mut msgs = vec![json!({"role": "system", "content": "s"})];
    for i in 0..5 {
        msgs.push(json!({"role": "assistant", "content": Value::Null,
            "tool_calls": [{"id": format!("c{i}")}]}));
        msgs.push(json!({"role": "tool", "tool_call_id": format!("c{i}"),
            "content": format!("small-{i}")}));
    }
    let before = msgs.clone();
    compact(&mut msgs);
    assert_eq!(msgs, before);
}

#[tokio::test]
async fn stderr_and_a_non_zero_exit_are_both_reported() {
    let c = cfg(std::env::temp_dir());
    let out = run_shell(&c, "echo hi; echo bad >&2; exit 7", Duration::from_secs(5))
        .await
        .unwrap();
    assert!(out.contains("hi"), "stdout is missing: {out}");
    assert!(out.contains("bad"), "stderr is missing: {out}");
    assert!(
        out.contains("exit status: 7"),
        "the exact exit status is missing: {out}"
    );
}

#[tokio::test]
async fn a_backgrounded_process_holding_the_pipe_does_not_stall_the_command() {
    // 2026-09-26: A background process holding the pipe must not make a
    // finished command wait for its timeout.
    let c = cfg(std::env::temp_dir());
    let started = std::time::Instant::now();
    let out = run_shell(&c, "sleep 25 & echo started", Duration::from_secs(20))
        .await
        .unwrap();
    assert!(out.contains("started"), "{out}");
    assert!(
        started.elapsed() < Duration::from_secs(10),
        "took {:?}",
        started.elapsed()
    );
}

#[tokio::test]
async fn a_timed_out_command_still_returns_what_it_printed() {
    let c = cfg(std::env::temp_dir());
    let out = run_shell(
        &c,
        "echo early; echo late >&2; sleep 30",
        Duration::from_millis(400),
    )
    .await
    .unwrap();
    assert!(
        out.contains("early"),
        "stdout before the kill is lost: {out}"
    );
    assert!(
        out.contains("late"),
        "stderr before the kill is lost: {out}"
    );
}

#[tokio::test]
async fn shell_output_is_normalised_before_it_is_truncated() {
    // 2026-09-26: The wiring, not the rules (those are in `norm_tests.rs`):
    // `run_shell` output goes through the normaliser.
    let c = cfg(std::env::temp_dir());
    let out = run_shell(
        &c,
        "echo '   Compiling pingpong v0.1.0 (/tmp/x)'; \
         echo '    Finished `test` profile [unoptimized] target(s) in 1.23s'; \
         echo 'kill: (1417733) - No such process'",
        Duration::from_secs(10),
    )
    .await
    .unwrap();
    assert!(!out.contains("Compiling"), "{out}");
    assert!(out.contains("target(s) in <elapsed>"), "{out}");
    assert!(out.contains("kill: (<pid>)"), "{out}");

    // 2026-09-26: This output fits under `MAX_TOOL_OUTPUT` only after the
    // progress line is dropped, so it shows normalising runs before truncating.
    let out = run_shell(
        &c,
        "printf '%08000d\\n' 0; printf '   Compiling %0300d\\n' 0",
        Duration::from_secs(10),
    )
    .await
    .unwrap();
    assert!(!out.contains("Compiling"), "{out}");
    assert!(!out.contains("elided from the middle"), "{out}");
    assert_eq!(out.len(), 8001);
}

#[cfg(target_os = "linux")]
#[tokio::test]
async fn a_detached_survivor_in_the_sandbox_is_reaped() {
    // 2026-09-26: A process whose working directory is in the sandbox is ended
    // by `reap`.
    let sb = sandbox("reap");
    let mut victim = tokio::process::Command::new("sh")
        .arg("-c")
        .arg("exec sleep 45")
        .current_dir(&sb)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .kill_on_drop(false)
        .spawn()
        .unwrap();
    let pid = victim.id().unwrap();
    reap(&sb).await;
    let seen = tokio::time::timeout(Duration::from_secs(5), victim.wait()).await;
    assert!(seen.is_ok(), "pid {pid} survived the reap");
    assert!(std::path::Path::new("/proc").exists());
}

#[tokio::test]
async fn output_past_the_pipe_buffer_does_not_deadlock_the_writer() {
    // 2026-09-26: Draining only after the process exits would block a command
    // that writes more than the pipe buffer holds.
    let c = cfg(std::env::temp_dir());
    let out = run_shell(
        &c,
        "head -c 8000000 /dev/zero | tr '\\0' 'a'",
        Duration::from_secs(2),
    )
    .await
    .unwrap();
    assert!(!out.contains("timed out"), "{}", &out[..80.min(out.len())]);
    assert!(!out.contains("[exit"), "the writer did not finish: {out}");
    assert!(out.contains("elided"), "the cap still applies");
}
