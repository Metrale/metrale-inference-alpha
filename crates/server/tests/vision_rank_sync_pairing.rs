// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Every scheduler site that sends the `0xFFFFFFF0` prefill
//! command must follow it with `ep_sync_vision_embeds`, because the worker's
//! `0xFFFFFFF0` arm (`model-engine/src/model/impl_a2.rs`) reads the vision
//! rows right after the prompt tokens. Without it, on a model with a vision
//! tower, the head's and the workers' broadcasts fall out of step.
//!
//! The check scans source rather than running two ranks: the two halves of
//! the protocol are in different files, and a runtime test would cover only
//! the paths it exercised.
//!
//! Owner: server tests.
//! Invariants: none beyond the types.

use std::path::{Path, PathBuf};

/// 2026-09-26: Lines allowed between the command send and the vision sync:
/// room for the argument words, the token broadcast and their comments. The
/// widest gap in the tree today is 15 (`prefill_a_step.rs`).
const MAX_GAP: usize = 40;

const SEND: &str = "ep_broadcast_cmd_for_seq";
const PREFILL_CMD: &str = "0xFFFFFFF0";
const SYNC: &str = "ep_sync_vision_embeds";

fn scheduler_sources() -> Vec<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/scheduler");
    let mut out = Vec::new();
    let mut stack = vec![root];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).expect("scheduler sources must be readable") {
            let path = entry.expect("readable dir entry").path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "rs") {
                out.push(path);
            }
        }
    }
    assert!(!out.is_empty(), "scanned no scheduler sources");
    out
}

/// 2026-09-26: A line whose code, not its trailing comment, sends the
/// prefill command.
fn is_prefill_send(line: &str) -> bool {
    let code = line.split("//").next().unwrap_or(line);
    code.contains(SEND) && code.contains(PREFILL_CMD)
}

fn is_sync(line: &str) -> bool {
    let code = line.split("//").next().unwrap_or(line);
    code.contains(SYNC)
}

#[test]
fn every_prefill_broadcast_is_followed_by_the_vision_row_sync() {
    let mut senders = 0usize;
    let mut unpaired = Vec::new();
    for path in scheduler_sources() {
        let text = std::fs::read_to_string(&path).expect("source readable");
        let lines: Vec<&str> = text.lines().collect();
        for (i, line) in lines.iter().enumerate() {
            if !is_prefill_send(line) {
                continue;
            }
            senders += 1;
            let end = (i + 1 + MAX_GAP).min(lines.len());
            if !lines[i + 1..end].iter().any(|l| is_sync(l)) {
                unpaired.push(format!("{}:{}", path.display(), i + 1));
            }
        }
    }
    assert!(
        senders >= 5,
        "expected at least the five known prefill-command senders, found {senders} — if a send \
         site moved, this scan is no longer looking where the protocol is written"
    );
    assert!(
        unpaired.is_empty(),
        "these prefill-command send sites do not hand the workers their vision rows within \
         {MAX_GAP} lines:\n  {}\n\nAdd `model.ep_sync_vision_embeds(&<the prompt tokens>)?;` \
         after the token broadcast. The worker reads that word for EVERY 0xFFFFFFF0 on a \
         vision-capable model; omitting it blocks the worker, and in the orderings where it \
         does not, leaves the ranks all-reducing different image embeddings while the model \
         stays fluent.",
        unpaired.join("\n  ")
    );
}

/// 2026-09-26: The worker half: exactly one `ep_sync_vision_embeds` call,
/// after the start of the `0xFFFFFFF0` arm.
#[test]
fn the_worker_reads_the_vision_rows_once_per_prefill_command() {
    let worker = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../model-engine/src/model/impl_a2.rs")
        .canonicalize()
        .expect("the worker command loop must exist");
    let text = std::fs::read_to_string(&worker).expect("source readable");
    let calls = text.lines().filter(|l| is_sync(l)).count();
    assert_eq!(
        calls, 1,
        "the worker's command loop must call {SYNC} exactly once — found {calls}. Two calls \
         read two words off the wire for one the head sent; none leaves the workers spliceless."
    );
    let arm = text
        .find("0xFFFFFFF0 => {")
        .expect("the prefill arm must exist");
    let sync_at = text.find(SYNC).expect("the sync call must exist");
    assert!(
        sync_at > arm,
        "the vision sync must live INSIDE the 0xFFFFFFF0 arm; reading it anywhere else pairs \
         it with a different command"
    );
}
