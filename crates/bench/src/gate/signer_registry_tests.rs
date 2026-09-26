// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for `committed_signers`, which counts only keys git
//! tracks, and for the wording of `signer_notice`.
//!
//! Owner: bench gate (signing).
//! Invariants: none beyond the types.

use super::signing::committed_signers;
use super::tests::tempdir;
use std::process::Command;

fn git(dir: &std::path::Path, args: &[&str]) {
    let ok = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .output()
        .expect("git runs")
        .status
        .success();
    assert!(ok, "git {args:?} failed in {}", dir.display());
}

fn repo() -> tempdir::Dir {
    let d = tempdir::Dir::new();
    let p = d.path();
    git(p, &["init", "-q"]);
    git(p, &["config", "user.email", "t@example.invalid"]);
    git(p, &["config", "user.name", "t"]);
    std::fs::create_dir_all(p.join(".github/record-signers")).expect("mkdir");
    d
}

fn put(dir: &std::path::Path, fp: &str) {
    std::fs::write(
        dir.join(".github/record-signers").join(format!("{fp}.pub")),
        "# header\nAAAA\n",
    )
    .expect("write key");
}

/// 2026-09-26: A `.pub` on disk that git does not track is not returned.
#[test]
fn an_auto_registered_but_uncommitted_key_does_not_count() {
    let d = repo();
    put(d.path(), "aaaaaaaaaaaaaaaa");
    let got = committed_signers(d.path()).expect("reads");
    assert!(
        got.is_empty(),
        "an untracked .pub must not read as a committed signer, got {got:?}"
    );
}

#[test]
fn a_committed_key_counts() {
    let d = repo();
    put(d.path(), "bbbbbbbbbbbbbbbb");
    git(d.path(), &["add", ".github/record-signers"]);
    git(d.path(), &["commit", "-qm", "register"]);
    assert_eq!(
        committed_signers(d.path()).expect("reads"),
        vec!["bbbbbbbbbbbbbbbb".to_string()]
    );
}

/// 2026-09-26: With a committed and an uncommitted key, only the committed
/// one is returned.
#[test]
fn only_the_committed_one_of_two_is_returned() {
    let d = repo();
    put(d.path(), "cccccccccccccccc");
    git(d.path(), &["add", ".github/record-signers"]);
    git(d.path(), &["commit", "-qm", "register"]);
    put(d.path(), "dddddddddddddddd");
    let got = committed_signers(d.path()).expect("reads");
    assert_eq!(got, vec!["cccccccccccccccc".to_string()], "got {got:?}");
}

/// 2026-09-26: Outside a git repository the result is an error, not an empty
/// list.
#[test]
fn an_unreadable_tree_is_an_error_not_an_empty_list() {
    let d = tempdir::Dir::new();
    assert!(
        committed_signers(d.path()).is_err(),
        "a non-repo must not report zero committed signers"
    );
}

use super::signing::signer_notice;

#[test]
fn a_committed_signer_gets_no_notice() {
    assert!(
        signer_notice(&["aaaa".to_string(), "bbbb".to_string()], "bbbb").is_none(),
        "the ordinary case must be silent, or operators learn to ignore it"
    );
}

/// 2026-09-26: The notice for an uncommitted signer names the fingerprint, the
/// registry directory, and that one PR's records need the same signer.
#[test]
fn an_uncommitted_signer_is_named_along_with_the_consequence() {
    let msg = signer_notice(&["aaaa".to_string()], "zzzz").expect("must warn");
    assert!(msg.contains("zzzz"), "must name the fingerprint: {msg}");
    assert!(
        msg.contains(".github/record-signers"),
        "must name where to commit it: {msg}"
    );
    assert!(
        msg.contains("same") || msg.contains("SAME"),
        "must say every record needs the same signer: {msg}"
    );
    assert!(
        msg.contains("split across boxes") || msg.contains("spanning signers"),
        "must name the campaign-splitting consequence: {msg}"
    );
}

#[test]
fn an_empty_registry_still_warns_rather_than_waving_through() {
    assert!(
        signer_notice(&[], "zzzz").is_some(),
        "no committed signers must not read as permission"
    );
}
