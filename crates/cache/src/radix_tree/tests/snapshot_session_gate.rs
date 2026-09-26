// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests for `session_gate_blocks`, with and without `--hermetic`, and a source scan that keeps it the only spelling of the gate.
//!
//! Owner: cache.
//! Invariants: none beyond the types.

use super::super::snapshot::SnapshotEntry;
use super::super::snapshot_session::session_gate_blocks;

fn entry(is_tail: bool, session_hash: u64) -> SnapshotEntry {
    sibling_entry(is_tail, false, session_hash)
}

fn sibling_entry(is_tail: bool, is_tail_sibling: bool, session_hash: u64) -> SnapshotEntry {
    SnapshotEntry {
        snapshot_id: 1,
        session_hash,
        token_count: 64,
        prefix_hash: 0xdead_beef,
        last_access: 0,
        tiered: false,
        is_tail,
        is_tail_sibling,
    }
}

#[test]
fn without_hermetic_a_non_tail_entry_crosses_sessions() {
    assert!(!session_gate_blocks(&entry(false, 11), 22, false));
}

#[test]
fn under_hermetic_a_non_tail_entry_is_confined_to_its_session() {
    assert!(session_gate_blocks(&entry(false, 11), 22, true));
}

/// 2026-09-25: Under `--hermetic`, an entry from the same non-zero session is
/// still allowed, tail or not.
#[test]
fn hermetic_still_allows_an_entry_from_the_same_session() {
    assert!(!session_gate_blocks(&entry(false, 42), 42, true));
    assert!(!session_gate_blocks(&entry(true, 42), 42, true));
}

#[test]
fn a_tail_from_another_session_is_blocked_either_way() {
    assert!(session_gate_blocks(&entry(true, 11), 22, false));
    assert!(session_gate_blocks(&entry(true, 11), 22, true));
}

#[test]
fn an_unsessioned_lookup_gets_nothing_it_should_not() {
    assert!(session_gate_blocks(&entry(true, 11), 0, false));
    assert!(!session_gate_blocks(&entry(false, 11), 0, false));
    assert!(session_gate_blocks(&entry(false, 11), 0, true));
}

#[test]
fn hermetic_also_confines_the_tail_sibling() {
    assert!(
        !session_gate_blocks(&sibling_entry(false, true, 11), 22, false),
        "production must keep serving warm turns from the sibling"
    );
    assert!(
        session_gate_blocks(&sibling_entry(false, true, 11), 22, true),
        "a KAT must not restore a mid-chunk capture from another request"
    );
}

/// 2026-09-25: No `.rs` file under `src/radix_tree` other than
/// `snapshot_session.rs` and this file may spell the gate condition itself.
/// `lookup` and `lookup_tiered` both call `session_gate_blocks`, so a change to
/// the gate reaches both.
#[test]
fn the_session_gate_is_not_spelled_out_anywhere_but_the_predicate() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/radix_tree");
    let mut offenders = Vec::new();
    let mut scanned = 0usize;
    let mut walk = vec![dir.clone()];
    while let Some(d) = walk.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else {
            continue;
        };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                walk.push(p);
                continue;
            }
            if p.extension().is_none_or(|x| x != "rs") {
                continue;
            }
            let rel = p
                .strip_prefix(&dir)
                .unwrap_or(&p)
                .to_string_lossy()
                .to_string();
            // 2026-09-25: Only the predicate's module and this test are
            // excused; `snapshot.rs` and `snapshot_tier.rs` hold the two
            // callers and are scanned.
            if rel == "snapshot_session.rs" || rel.ends_with("snapshot_session_gate.rs") {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(&p) else {
                continue;
            };
            scanned += 1;
            for (n, line) in text.lines().enumerate() {
                // 2026-09-25: Match the shape of the condition, not a name: a
                // copy would read both fields on one line of code.
                let code = line.split("//").next().unwrap_or("");
                if code.contains("entry.session_hash") && code.contains("session_hash ==") {
                    offenders.push(format!("{rel}:{}", n + 1));
                }
            }
        }
    }
    assert!(
        scanned > 5,
        "the scan visited {scanned} files — it is not scanning the tree"
    );
    assert!(
        offenders.is_empty(),
        "the session gate must be read through `session_gate_blocks`, not \
         re-spelled — otherwise --hermetic reaches one copy and not the other:\n  {}",
        offenders.join("\n  ")
    );
}
