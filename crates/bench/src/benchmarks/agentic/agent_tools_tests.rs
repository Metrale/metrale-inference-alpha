// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for the agent's six tools.
//!
//! Owner: bench, agentic.
//! Invariants: none beyond the types.

use super::*;
use crate::benchmarks::agentic::agent::tests::{cfg, sandbox};

async fn tool(cfg: &AgentConfig, name: &str, args: Value) -> Result<String> {
    let call = crate::http::ToolCall {
        id: String::new(),
        name: name.into(),
        arguments: args.to_string(),
    };
    execute(cfg, &call, &mut Vec::new()).await
}

#[cfg(unix)]
#[tokio::test]
async fn write_distinguishes_dangling_symlinks_by_destination() {
    let c = cfg(sandbox("dangling-write"));
    let outside = sandbox("dangling-write-outside");
    let target = outside.join("created.rs");
    let relative_target = Path::new("..")
        .join(outside.file_name().unwrap())
        .join("created.rs");
    std::os::unix::fs::symlink(relative_target, c.sandbox.join("escape")).unwrap();

    let result = tool(
        &c,
        "write",
        json!({"filePath": "escape", "content": "outside"}),
    )
    .await;

    assert!(result.is_err(), "outside write was accepted: {result:?}");
    assert!(!target.exists(), "write created {}", target.display());

    let inside_target = c.sandbox.join("created-inside.rs");
    std::os::unix::fs::symlink(&inside_target, c.sandbox.join("inside")).unwrap();
    tool(
        &c,
        "write",
        json!({"filePath": "inside", "content": "inside"}),
    )
    .await
    .unwrap();
    assert_eq!(std::fs::read_to_string(inside_target).unwrap(), "inside");
}

#[test]
fn the_tool_surface_is_the_six_the_harness_agent_enables() {
    let names: Vec<String> = tool_schema()
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["function"]["name"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(names, ["bash", "read", "write", "edit", "glob", "grep"]);
}

#[tokio::test]
async fn a_model_supplied_timeout_can_shorten_but_never_raise_the_ceiling() {
    let mut c = cfg(std::env::temp_dir());
    c.command_timeout = Duration::from_secs(1);

    let started = std::time::Instant::now();
    let out = tool(&c, "bash", json!({"command": "sleep 30", "timeout": 50}))
        .await
        .unwrap();
    assert!(out.contains("timed out"), "{out}");
    assert!(
        started.elapsed() < Duration::from_millis(500),
        "the model's shorter timeout was ignored: {:?}",
        started.elapsed()
    );

    let started = std::time::Instant::now();
    let out = tool(
        &c,
        "bash",
        json!({"command": "sleep 30", "timeout": 600_000}),
    )
    .await
    .unwrap();
    assert!(out.contains("timed out"), "{out}");
    assert!(
        started.elapsed() < Duration::from_secs(2),
        "the model raised the harness ceiling: {:?}",
        started.elapsed()
    );
}

#[tokio::test]
async fn the_agent_shell_is_marked_so_the_cargo_shim_detaches_cargo_run() {
    // 2026-09-26: `bench/fp8_dgx2_drift/harness/cargo-shim/cargo` detaches
    // `cargo run` only when METRALE_AGENT_SHELL is set.
    let c = cfg(sandbox("shellenv"));
    let out = tool(
        &c,
        "bash",
        json!({"command": "echo ${METRALE_AGENT_SHELL:-unset}"}),
    )
    .await
    .unwrap();
    assert_eq!(out.trim(), "1");
}

#[tokio::test]
async fn read_lists_a_directory_so_a_missing_src_is_visible() {
    let c = cfg(sandbox("readdir"));
    tool(
        &c,
        "write",
        json!({"filePath": "Cargo.toml", "content": "[package]\n"}),
    )
    .await
    .unwrap();
    assert_eq!(
        tool(&c, "read", json!({"filePath": "."})).await.unwrap(),
        "Cargo.toml"
    );

    tool(
        &c,
        "write",
        json!({"filePath": "z.txt", "content": "created first\n"}),
    )
    .await
    .unwrap();
    tool(
        &c,
        "write",
        json!({"filePath": "src/main.rs", "content": "fn main() {}\n"}),
    )
    .await
    .unwrap();
    let listing = tool(&c, "read", json!({"filePath": "."})).await.unwrap();
    assert_eq!(listing, "Cargo.toml\nsrc/\nz.txt");
}

#[tokio::test]
async fn read_numbers_lines_and_honours_offset() {
    let c = cfg(sandbox("readlines"));
    tool(
        &c,
        "write",
        json!({"filePath": "a.rs", "content": "one\ntwo\nthree\n"}),
    )
    .await
    .unwrap();
    assert_eq!(
        tool(&c, "read", json!({"filePath": "a.rs"})).await.unwrap(),
        "1: one\n2: two\n3: three"
    );
    assert_eq!(
        tool(
            &c,
            "read",
            json!({"filePath": "a.rs", "offset": 2, "limit": 1})
        )
        .await
        .unwrap(),
        "2: two"
    );
}

#[tokio::test]
async fn a_missing_read_names_the_path_it_could_not_open() {
    let c = cfg(sandbox("readmiss"));
    let e = format!(
        "{:#}",
        tool(&c, "read", json!({"filePath": "cargo.toml"}))
            .await
            .unwrap_err()
    );
    assert!(e.contains("cargo.toml"), "{e}");
}

#[tokio::test]
async fn an_absolute_path_from_the_environment_block_is_usable() {
    // 2026-09-26: The system prompt names the sandbox as the working directory,
    // so an absolute path built from it must work.
    let c = cfg(sandbox("abs"));
    let abs = c.sandbox.join("src/main.rs");
    tool(
        &c,
        "write",
        json!({"filePath": abs, "content": "fn main() {}\n"}),
    )
    .await
    .unwrap();
    assert!(abs.is_file());
    assert!(
        tool(
            &c,
            "write",
            json!({"filePath": "/etc/passwd", "content": "x"})
        )
        .await
        .is_err()
    );
}

#[tokio::test]
async fn edit_replaces_exactly_once_and_says_how_to_retry_when_it_cannot() {
    let c = cfg(sandbox("edit"));
    tool(
        &c,
        "write",
        json!({"filePath": "src/main.rs",
        "content": "fn main() {\n    axum::serve(l, app);\n}\n"}),
    )
    .await
    .unwrap();
    let said = tool(
        &c,
        "edit",
        json!({"filePath": "src/main.rs",
        "oldString": "axum::serve(l, app);", "newString": "axum::serve(l, app).await.unwrap();"}),
    )
    .await
    .unwrap();
    // 2026-09-26: Sandbox-relative, like `write`.
    assert_eq!(said, "replaced 1 occurrence(s) in src/main.rs");
    assert!(
        std::fs::read_to_string(c.sandbox.join("src/main.rs"))
            .unwrap()
            .contains(".await.unwrap()")
    );

    let e = format!(
        "{:#}",
        tool(
            &c,
            "edit",
            json!({"filePath": "src/main.rs",
            "oldString": "nope", "newString": "x"})
        )
        .await
        .unwrap_err()
    );
    assert_eq!(e, "oldString not found in content");
}

#[tokio::test]
async fn edit_refuses_an_ambiguous_match_unless_replace_all_is_set() {
    let c = cfg(sandbox("editall"));
    tool(
        &c,
        "write",
        json!({"filePath": "a.rs", "content": "let x = 1;\nlet x = 2;\n"}),
    )
    .await
    .unwrap();
    let e = format!(
        "{:#}",
        tool(
            &c,
            "edit",
            json!({"filePath": "a.rs", "oldString": "let x", "newString": "let y"})
        )
        .await
        .unwrap_err()
    );
    assert!(
        e.contains("multiple matches") && e.contains("replaceAll"),
        "{e}"
    );

    tool(
        &c,
        "edit",
        json!({"filePath": "a.rs", "oldString": "let x",
        "newString": "let y", "replaceAll": true}),
    )
    .await
    .unwrap();
    let text = std::fs::read_to_string(c.sandbox.join("a.rs")).unwrap();
    assert_eq!(text, "let y = 1;\nlet y = 2;\n");
}

#[tokio::test]
async fn edit_rejects_an_empty_match_without_touching_the_file() {
    let c = cfg(sandbox("edit-empty"));
    tool(&c, "write", json!({"filePath": "a.rs", "content": "abc\n"}))
        .await
        .unwrap();
    let result = tool(
        &c,
        "edit",
        json!({"filePath": "a.rs", "oldString": "", "newString": "x", "replaceAll": true}),
    )
    .await;
    assert!(result.is_err(), "empty oldString was accepted: {result:?}");
    assert_eq!(
        std::fs::read_to_string(c.sandbox.join("a.rs")).unwrap(),
        "abc\n"
    );
}

#[tokio::test]
async fn glob_and_grep_see_the_project_but_not_the_build_tree() {
    let c = cfg(sandbox("search"));
    tool(
        &c,
        "write",
        json!({"filePath": "src/main.rs", "content": "async fn ping() {}\n"}),
    )
    .await
    .unwrap();
    tool(
        &c,
        "write",
        json!({"filePath": "Cargo.toml", "content": "[package]\nname=\"p\"\n"}),
    )
    .await
    .unwrap();
    std::fs::create_dir_all(c.sandbox.join("target/debug")).unwrap();
    std::fs::write(c.sandbox.join("target/debug/junk.rs"), "async fn ping() {}").unwrap();

    assert_eq!(
        tool(&c, "glob", json!({"pattern": "**/*.rs"}))
            .await
            .unwrap(),
        "src/main.rs"
    );

    let g = tool(&c, "grep", json!({"pattern": "fn ping"}))
        .await
        .unwrap();
    assert!(g.contains("src/main.rs:1:"), "{g}");
    assert!(!g.contains("target/"), "{g}");

    let none = tool(
        &c,
        "grep",
        json!({"pattern": "fn ping", "include": "*.toml"}),
    )
    .await
    .unwrap();
    assert_eq!(none, "No files found");
}

#[tokio::test]
async fn read_and_grep_do_not_return_content_past_the_file_cap() {
    // 2026-09-26: A file larger than the cap, with its only match past it.
    let c = cfg(sandbox("bigfile"));
    let mut big = "x".repeat(READ_LINES * READ_LINE_CHARS);
    big.push_str("\nNEEDLE_AT_THE_END\n");
    std::fs::write(c.sandbox.join("server.log"), &big).unwrap();

    let read = tool(&c, "read", json!({"filePath": "server.log"}))
        .await
        .unwrap();
    assert!(
        read.contains("(file truncated at this point)"),
        "not capped"
    );
    assert!(!read.contains("NEEDLE_AT_THE_END"));
    assert_eq!(
        tool(&c, "grep", json!({"pattern": "NEEDLE_AT_THE_END"}))
            .await
            .unwrap(),
        "No files found"
    );

    // 2026-09-26: Non-UTF-8 is an error, except for a multi-byte character the
    // cap itself cut in half.
    std::fs::write(c.sandbox.join("a.bin"), [0xff, 0xfe, 0x00, 0x01]).unwrap();
    assert!(
        tool(&c, "read", json!({"filePath": "a.bin"}))
            .await
            .is_err()
    );
    // 2026-09-26: The leading byte puts the cap's cut in the middle of a
    // two-byte `é`.
    let wide = "a".to_string() + &"é".repeat(READ_LINES * READ_LINE_CHARS);
    std::fs::write(c.sandbox.join("wide.txt"), &wide).unwrap();
    let read = tool(&c, "read", json!({"filePath": "wide.txt"}))
        .await
        .unwrap();
    assert!(!read.contains('\u{fffd}'), "a cut character was mangled");
}

#[cfg(unix)]
#[tokio::test]
async fn the_file_reader_returns_at_the_cap_without_waiting_for_eof() {
    use std::io::Write;

    let fifo = sandbox("read-fifo").join("stream");
    assert!(
        std::process::Command::new("mkfifo")
            .arg(&fifo)
            .status()
            .unwrap()
            .success()
    );
    let (release, held) = std::sync::mpsc::channel();
    let writer_path = fifo.clone();
    let writer = std::thread::spawn(move || {
        let mut stream = std::fs::OpenOptions::new()
            .write(true)
            .open(writer_path)
            .unwrap();
        stream
            .write_all(&vec![b'x'; READ_LINES * READ_LINE_CHARS])
            .unwrap();
        let _ = held.recv_timeout(Duration::from_secs(8));
    });
    let read = tokio::task::spawn_blocking(move || read_capped(&fifo));
    let result = tokio::time::timeout(Duration::from_secs(4), read).await;
    let _ = release.send(());
    writer.join().unwrap();
    let text = result
        .expect("reader waited for EOF after reaching its cap")
        .unwrap()
        .unwrap();
    assert!(text.contains("(file truncated at this point)"));
}

#[cfg(unix)]
#[tokio::test]
async fn search_tools_neither_follow_symlink_directories_nor_read_symlink_files() {
    // 2026-09-26: A finite directory alias, so code that followed symlinks
    // fails this test instead of looping.
    let c = cfg(sandbox("cycle"));
    tool(
        &c,
        "write",
        json!({"filePath": "src/main.rs", "content": "async fn ping() {}\n"}),
    )
    .await
    .unwrap();
    std::os::unix::fs::symlink("src", c.sandbox.join("alias")).unwrap();
    std::os::unix::fs::symlink("/etc/hostname", c.sandbox.join("host.rs")).unwrap();
    assert_eq!(
        tool(&c, "glob", json!({"pattern": "**/*.rs"}))
            .await
            .unwrap(),
        "src/main.rs"
    );
    let g = tool(&c, "grep", json!({"pattern": "fn ping"}))
        .await
        .unwrap();
    assert_eq!(g, "Found 1 matches\nsrc/main.rs:1: async fn ping() {}");
}

#[tokio::test]
async fn an_unknown_tool_tells_the_model_what_it_may_call() {
    let c = cfg(std::env::temp_dir());
    let e = format!("{:#}", tool(&c, "todowrite", json!({})).await.unwrap_err());
    assert!(e.contains("bash, read, write, edit, glob, grep"), "{e}");
}

#[test]
fn glob_patterns_cover_basenames_segments_and_recursive_paths() {
    let m = |p: &str, t: &str| glob_match(p.as_bytes(), t.as_bytes());
    assert!(m("**/*.rs", "src/main.rs"));
    assert!(m("**/*.rs", "main.rs"));
    assert!(
        m("*.rs", "src/deep/main.rs"),
        "a bare pattern matches the basename"
    );
    assert!(m("src/*.rs", "src/main.rs"));
    assert!(
        !m("src/*.rs", "src/a/main.rs"),
        "* must not cross a separator"
    );
    assert!(m("src/**/*.rs", "src/a/b/main.rs"));
    assert!(m("Cargo.to?l", "Cargo.toml"));
    assert!(!m("*.rs", "Cargo.toml"));
}
