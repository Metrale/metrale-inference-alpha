// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Hints appended to tool results in the chat prompt: a recovery
//! hint for an error-shaped result (`inject_hints`), and the bash-wander hint
//! when the conversation's tool calls have produced nothing
//! (`bash_wander_hint`, on only with `METRALE_BASH_WANDER_WATCHDOG=1`).
//! `api/chat/msg_entry.rs` calls both.
//!
//! Owner: server chat API.
//! Invariants: `inject_hints` appends at most one hint.

/// 2026-09-26: What an injector sees of one tool result.
pub struct HintContext<'a> {
    /// 2026-09-26: The tool result text; a failed result starts with a
    /// `[tool error]` line (`api/chat/msg_entry.rs`).
    pub text: &'a str,
    /// 2026-09-26: Error-shaped tool results in a row up to and including
    /// this one, as `api/chat/msg_entry.rs` counts them; `looks_like_error`
    /// passes 0.
    pub consecutive_errors: u32,
}

/// 2026-09-26: One error class: `is_relevant` recognises it and `hint` gives
/// the text to append.
pub trait HintInjector: Send + Sync {
    /// 2026-09-26: Whether the text shows this error class. `looks_like_error`
    /// asks every injector about every tool result.
    fn is_relevant(&self, ctx: &HintContext) -> bool;

    /// 2026-09-26: The hint to append, called only after `is_relevant`
    /// returned true. An empty string appends nothing.
    fn hint(&self, ctx: &HintContext) -> String;
}

/// 2026-09-26: `EISDIR` or "illegal operation on a directory": the path is a
/// directory. Escalates from 3 errors in a row.
pub struct WritePathHint;

impl HintInjector for WritePathHint {
    fn is_relevant(&self, ctx: &HintContext) -> bool {
        ctx.text.contains("EISDIR") || ctx.text.contains("illegal operation on a directory")
    }

    fn hint(&self, ctx: &HintContext) -> String {
        if ctx.consecutive_errors >= 3 {
            "\n\n<CRITICAL>\n\
             STOP using the Write tool — it has failed 3+ times with a directory path.\n\
             You MUST use Bash instead: cat > ./path/to/file.ext << 'EOF'\n\
             ...content...\nEOF\n\
             Write/Edit file_path MUST be a FILE (e.g. ./dir/file.txt), NEVER a directory (./dir).\n\
             </CRITICAL>"
                .to_string()
        } else {
            "\n\nHint: file_path is a directory, not a file. \
             Use a full path like ./dir/file.txt (not ./dir). \
             If Write keeps failing, use Bash: cat > file << 'EOF'"
                .to_string()
        }
    }
}

/// 2026-09-26: `ENOENT` or "No such file or directory": suggests creating the
/// parent directory. Escalates from 3 errors in a row.
pub struct FileNotFoundHint;

impl HintInjector for FileNotFoundHint {
    fn is_relevant(&self, ctx: &HintContext) -> bool {
        ctx.text.contains("ENOENT") || ctx.text.contains("No such file or directory")
    }

    fn hint(&self, ctx: &HintContext) -> String {
        if ctx.consecutive_errors >= 3 {
            "\n\n<CRITICAL>\n\
             The file or directory does not exist. Create the parent directory first:\n\
             mkdir -p ./parent/dir && cat > ./parent/dir/file.ext << 'EOF'\n\
             ...content...\nEOF\n\
             </CRITICAL>"
                .to_string()
        } else {
            "\n\nHint: File or directory not found. \
             Create the parent directory first with: mkdir -p ./parent/dir"
                .to_string()
        }
    }
}

/// 2026-09-26: A missing file (`ENOENT` or "No such file") in a text that also
/// contains `read` or `open`. Escalates from 3 errors in a row.
pub struct ReadErrorHint;

impl HintInjector for ReadErrorHint {
    fn is_relevant(&self, ctx: &HintContext) -> bool {
        (ctx.text.contains("ENOENT") || ctx.text.contains("No such file"))
            && (ctx.text.contains("read") || ctx.text.contains("open"))
    }

    fn hint(&self, ctx: &HintContext) -> String {
        if ctx.consecutive_errors >= 3 {
            "\n\nHint: File does not exist. Check the path with: ls ./dir/ \
             or find . -name 'filename'. Do NOT keep reading non-existent files."
                .to_string()
        } else {
            "\n\nHint: File not found. Verify the path exists before reading.".to_string()
        }
    }
}

/// 2026-09-26: Any text containing "agent type", which covers the
/// unknown-agent errors of a delegation tool. Escalates from 2 errors in a row.
pub struct TaskDelegationHint;

impl HintInjector for TaskDelegationHint {
    fn is_relevant(&self, ctx: &HintContext) -> bool {
        ctx.text.contains("Unknown agent type")
            || ctx.text.contains("not a valid agent type")
            || ctx.text.contains("agent type")
    }

    fn hint(&self, ctx: &HintContext) -> String {
        if ctx.consecutive_errors >= 2 {
            "\n\nHint: STOP using the Task tool — use Bash, Write, Read, and Glob directly instead. \
             Do NOT delegate to sub-agents."
                .to_string()
        } else {
            "\n\nHint: Task delegation failed. Use direct tools (Bash, Write, Read) instead."
                .to_string()
        }
    }
}

/// 2026-09-26: An edit whose `old_string` did not match ("not found in file",
/// "old_string", "not unique", "No match found"). Escalates from 3 errors in a
/// row.
pub struct EditMismatchHint;

impl HintInjector for EditMismatchHint {
    fn is_relevant(&self, ctx: &HintContext) -> bool {
        ctx.text.contains("not found in file")
            || ctx.text.contains("old_string")
            || ctx.text.contains("not unique")
            || ctx.text.contains("No match found")
    }

    fn hint(&self, ctx: &HintContext) -> String {
        if ctx.consecutive_errors >= 3 {
            "\n\nHint: Edit keeps failing. Read the file first to see exact content, \
             then retry with the correct old_string. Or use Write to replace the entire file."
                .to_string()
        } else {
            "\n\nHint: Edit failed — old_string doesn't match file content. \
             Read the file first to see the exact text."
                .to_string()
        }
    }
}

/// 2026-09-26: A command that is not installed ("command not found",
/// "Exit code 127", ": not found"). Retrying cannot fix it, so the first hint
/// already says not to retry, and it escalates from 2 errors in a row.
pub struct NotInstalledHint;

impl HintInjector for NotInstalledHint {
    fn is_relevant(&self, ctx: &HintContext) -> bool {
        ctx.text.contains("command not found")
            || ctx.text.contains("Exit code 127")
            || ctx.text.contains(": not found")
            || ctx.text.contains("[tool error]\nExit code 127")
    }

    fn hint(&self, ctx: &HintContext) -> String {
        if ctx.consecutive_errors >= 2 {
            "\n\n<CRITICAL>\n\
             STOP retrying. The command is NOT installed in this \
             environment. Cosmetic variations (different mkdir, cd, \
             flag order) will not change the outcome. Reply to the \
             user about the missing dependency and ask whether to \
             proceed differently. Do NOT call Bash with the same \
             command again.\n\
             </CRITICAL>"
                .to_string()
        } else {
            "\n\nHint: that command is not installed. Do NOT retry \
             with cosmetic variations (mkdir, cd, &&). Either tell \
             the user the tool is missing, or use a different \
             approach that doesn't require it."
                .to_string()
        }
    }
}

/// 2026-09-26: Any other error-shaped text; last in both injector lists, so
/// it applies only when no specific injector matched. Escalates from 3 errors
/// in a row.
pub struct GenericErrorHint;

/// 2026-09-26: Whether the text, after leading whitespace, starts with an
/// error-code word of at least 4 ASCII letters or digits followed by `:`:
/// mixed case with at least two capitals (`BadResource:`, `NotFound:`) or all
/// capitals (`EISDIR:`). A word with one capital, such as `Note:`, does not
/// count.
fn pascal_error_prefix(text: &str) -> bool {
    let t = text.trim_start();
    let Some(first) = t.split(':').next() else {
        return false;
    };
    // 2026-09-26: When `first` is shorter than `t`, the next byte is the `:`
    // that `split` cut at.
    if first.len() < 4
        || first.len() >= t.len()
        || !first.chars().all(|c| c.is_ascii_alphanumeric())
    {
        return false;
    }
    let caps = first.chars().filter(|c| c.is_ascii_uppercase()).count();
    let has_lower = first.chars().any(|c| c.is_ascii_lowercase());
    (has_lower && caps >= 2) || (!has_lower && caps == first.chars().count())
}

impl HintInjector for GenericErrorHint {
    fn is_relevant(&self, ctx: &HintContext) -> bool {
        ctx.text.starts_with("Error")
            || ctx.text.starts_with("error")
            || ctx.text.contains("Error:")
            || ctx.text.contains("error:")
            || ctx.text.contains("failed")
            || ctx.text.contains("Permission denied")
            || ctx.text.starts_with("BadResource")
            || pascal_error_prefix(ctx.text)
    }

    fn hint(&self, ctx: &HintContext) -> String {
        if ctx.consecutive_errors >= 3 {
            "\n\n<CRITICAL>\n\
             This tool has failed 3+ times. STOP retrying the same approach.\n\
             Use Bash as a fallback for file operations.\n\
             Do NOT retry the exact same call — change your strategy.\n\
             </CRITICAL>"
                .to_string()
        } else {
            "\n\nHint: Tool call failed. If it keeps failing, \
             try Bash as a fallback. Do NOT retry the exact same call."
                .to_string()
        }
    }
}

/// 2026-09-26: Append the hint of the first injector in the list below that
/// matches and gives a non-empty hint. Specific injectors come first and
/// `GenericErrorHint` last; nothing is returned.
pub fn inject_hints(text: &mut String, consecutive_errors: u32) {
    let ctx = HintContext {
        text,
        consecutive_errors,
    };

    let injectors: &[&dyn HintInjector] = &[
        &WritePathHint,
        &ReadErrorHint,
        &EditMismatchHint,
        &TaskDelegationHint,
        &FileNotFoundHint,
        &NotInstalledHint,
        &GenericErrorHint,
    ];

    for injector in injectors {
        if injector.is_relevant(&ctx) {
            let hint = injector.hint(&ctx);
            if !hint.is_empty() {
                text.push_str(&hint);
                return;
            }
        }
    }
}

/// 2026-09-26: Whether any injector's `is_relevant` matches the text.
pub fn looks_like_error(text: &str) -> bool {
    let ctx = HintContext {
        text,
        consecutive_errors: 0,
    };
    let injectors: &[&dyn HintInjector] = &[
        &WritePathHint,
        &ReadErrorHint,
        &EditMismatchHint,
        &TaskDelegationHint,
        &FileNotFoundHint,
        &NotInstalledHint,
        &GenericErrorHint,
    ];
    injectors.iter().any(|i| i.is_relevant(&ctx))
}

/// 2026-09-26: Whether a tool call counts as progress for the bash-wander
/// hint: a tool whose lowercased name contains `write` or `edit` or is
/// `create` or `patch`; or a shell tool (name contains `bash` or `exec`, or is
/// `shell` or `run`) whose `command`, `cmd` or `script` argument contains an
/// entry of `WRITE_VERBS`. Every other call is exploration.
pub fn tool_call_is_productive(name: &str, args: &serde_json::Value) -> bool {
    let n = name.to_ascii_lowercase();
    if n.contains("write") || n.contains("edit") || n == "create" || n == "patch" {
        return true;
    }
    if n.contains("bash") || n == "shell" || n == "run" || n.contains("exec") {
        let cmd = args
            .get("command")
            .or_else(|| args.get("cmd"))
            .or_else(|| args.get("script"))
            .and_then(|v| v.as_str())
            .unwrap_or("");
        const WRITE_VERBS: &[&str] = &[
            "cat >",
            "cat>",
            "tee ",
            ">>",
            " > ",
            "cargo build",
            "cargo run",
            "cargo test",
            "rustc ",
            "go build",
            "go run",
            "npm run",
            "npm install",
            "make ",
            "python ",
            "node ",
            "touch ",
        ];
        return WRITE_VERBS.iter().any(|v| cmd.contains(v));
    }
    false
}

/// 2026-09-26: The steering hint for the latest tool result: `Some` when
/// `enabled` (`ChatLevers::bash_wander`) and the conversation has at least
/// `MIN_CALLS` (5) tool calls with none productive, with the stronger text
/// from 9 calls; `None` otherwise.
pub fn bash_wander_hint(
    total_tool_calls: usize,
    productive_calls: usize,
    enabled: bool,
) -> Option<String> {
    if !enabled {
        return None;
    }
    bash_wander_hint_inner(total_tool_calls, productive_calls)
}

/// 2026-09-26: [`bash_wander_hint`] without the `enabled` check.
fn bash_wander_hint_inner(total_tool_calls: usize, productive_calls: usize) -> Option<String> {
    const MIN_CALLS: usize = 5;
    if productive_calls > 0 || total_tool_calls < MIN_CALLS {
        return None;
    }
    let n = total_tool_calls;
    let body = if n >= 9 {
        format!(
            "<CRITICAL PROGRESS WATCHDOG>\n\
             You have run {n} tool calls and have NOT written or edited a single file. \
             Exploration will not complete the task. In your next message, call the write \
             tool to create the required source file(s), then build and run to verify. \
             Do not run any more read-only commands.\n</CRITICAL PROGRESS WATCHDOG>"
        )
    } else {
        format!(
            "<PROGRESS WATCHDOG>\n\
             You have run {n} tool calls without writing or editing any file yet. If the \
             task asks you to create or modify files, do that now with the write/edit tool \
             instead of more exploration, then verify by building/running.\n</PROGRESS WATCHDOG>"
        )
    };
    Some(format!("\n\n{body}"))
}

#[cfg(test)]
#[path = "hint_injector_tests.rs"]
mod hint_injector_tests;
