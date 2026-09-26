// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests of the content-pinned amnesty: it excuses exactly the
//! pinned blob, fails closed on everything else, is wired into
//! `invalidating_paths_with_amnesty`, and the production table obeys its
//! expiry rule.
//!
//! Owner: bench gate.
//! Invariants: none beyond the types.

use super::amnesty::{AMNESTY_EPOCH, AmnestyEntry, ONE_TIME_AMNESTY, excused_by};
use super::check_paths::invalidating_paths_with_amnesty;
use super::coverage_tests::{any_gate, scratch_repo};
use super::tests::tempdir;
use super::{REQUIRED_GATES, read_record, records_newest_first};

const TAXONOMY: &str = ".github/pr-taxonomy.json";
const GRANTED_COVERAGE: &str = "crates/bench/src/gate/coverage.rs";

fn blob_oid(root: &std::path::Path, head: &str, path: &str) -> String {
    let out = std::process::Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["rev-parse", &format!("{head}:{path}")])
        .output()
        .expect("git runs");
    assert!(out.status.success(), "rev-parse {head}:{path}: {out:?}");
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

/// 2026-09-26: A test entry pinning a real blob. The OID string is leaked to
/// get the `&'static str` the entry holds.
fn entry(path: &'static str, oid: String) -> AmnestyEntry {
    AmnestyEntry {
        path,
        head_blob_oid: Box::leak(oid.into_boxed_str()),
        grant: "test grant",
    }
}

/// 2026-09-26: The pinned content is excused at the commit that carries it,
/// and a later edit to the same path is not, because the blob OID changed.
#[test]
fn pinned_content_is_excused_and_a_later_edit_reinvalidates() {
    let dir = tempdir::Dir::new();
    let root = dir.path();
    scratch_repo::init(root);
    scratch_repo::commit(root, TAXONOMY, r#"{ "a": {}, "b": {} }"#, "the landing");
    let landed = scratch_repo::head(root);
    let table = [entry(TAXONOMY, blob_oid(root, &landed, TAXONOMY))];

    assert!(
        excused_by(root, &landed, TAXONOMY, &table),
        "the exact landed bytes must be excused at the landing commit"
    );

    scratch_repo::commit(
        root,
        TAXONOMY,
        r#"{ "a": {}, "b": {}, "c": {} }"#,
        "a later edit",
    );
    let later = scratch_repo::head(root);
    assert!(
        !excused_by(root, &later, TAXONOMY, &table),
        "an edit after the grant changes the blob OID — the amnesty must not \
         stretch to cover bytes nobody reviewed"
    );
}

/// 2026-09-26: A path the table does not list is never excused, whatever its
/// content.
#[test]
fn an_unlisted_path_is_never_excused() {
    let dir = tempdir::Dir::new();
    let root = dir.path();
    scratch_repo::init(root);
    scratch_repo::commit(root, TAXONOMY, "{}", "taxonomy");
    scratch_repo::commit(root, "crates/engine.rs", "{}", "engine");
    let head = scratch_repo::head(root);
    // 2026-09-26: Both files hold `{}`, so the pinned OID is also
    // engine.rs's own; only the path keeps it out.
    let table = [entry(TAXONOMY, blob_oid(root, &head, TAXONOMY))];
    assert!(
        !excused_by(root, &head, "crates/engine.rs", &table),
        "only listed paths participate; content is checked second, not instead"
    );
}

/// 2026-09-26: No repository, an unknown commit and a path absent at the head
/// all read as not excused.
#[test]
fn git_failure_fails_closed() {
    let bare = tempdir::Dir::new();
    let table = [entry(TAXONOMY, "0".repeat(40))];
    assert!(
        !excused_by(bare.path(), "HEAD", TAXONOMY, &table),
        "no repo means no answer, and no answer must not excuse"
    );

    let dir = tempdir::Dir::new();
    let root = dir.path();
    scratch_repo::init(root);
    let head = scratch_repo::head(root);
    assert!(
        !excused_by(root, "ffffffffff", TAXONOMY, &table),
        "an unknown commit must fail closed"
    );
    assert!(
        !excused_by(root, &head, TAXONOMY, &table),
        "a path absent at head has no blob to match — fail closed"
    );
}

/// 2026-09-26: The invalidating-path filter consults the table it is given:
/// the granted blob is dropped, a later edit is kept. An explicit table keeps
/// both arms testable while the production table is empty.
#[test]
fn invalidating_paths_drops_exactly_what_the_grant_excuses() {
    let dir = tempdir::Dir::new();
    let root = dir.path();
    scratch_repo::init(root);
    scratch_repo::commit(root, GRANTED_COVERAGE, "// baseline", "baseline");
    let record_sha = scratch_repo::head(root);

    scratch_repo::commit(root, GRANTED_COVERAGE, "// granted bytes", "the grant");
    let granted = scratch_repo::head(root);
    let table = [entry(
        GRANTED_COVERAGE,
        blob_oid(root, &granted, GRANTED_COVERAGE),
    )];
    let dropped = invalidating_paths_with_amnesty(root, &granted, &record_sha, &any_gate(), &table)
        .expect("the granted diff runs");
    assert_eq!(dropped, Vec::<String>::new());

    scratch_repo::commit(root, GRANTED_COVERAGE, "// later edit", "later edit");
    let later = scratch_repo::head(root);
    let kept = invalidating_paths_with_amnesty(root, &later, &record_sha, &any_gate(), &table)
        .expect("the later diff runs");
    assert_eq!(
        kept,
        vec![GRANTED_COVERAGE.to_string()],
        "editing the granted path must restore invalidation"
    );
}

/// 2026-09-26: The production table is empty, or holds only
/// `crates/bench/src/gate/coverage.rs` with a 40-hex OID and a grant naming
/// "PR #816".
#[test]
fn the_table_is_exactly_the_pr_816_grant() {
    let paths: Vec<&str> = ONE_TIME_AMNESTY.iter().map(|e| e.path).collect();
    if !paths.is_empty() {
        assert_eq!(
            paths,
            vec!["crates/bench/src/gate/coverage.rs"],
            "the grant must not grow beyond PR #816's coverage-policy blob"
        );
    }
    for entry in &ONE_TIME_AMNESTY {
        assert_eq!(
            entry.head_blob_oid.len(),
            40,
            "{} is not pinned",
            entry.path
        );
        assert!(
            entry.head_blob_oid.chars().all(|c| c.is_ascii_hexdigit()),
            "{} has a non-hex blob OID",
            entry.path
        );
        assert!(
            entry.grant.contains("PR #816"),
            "{} lacks its grant",
            entry.path
        );
    }
}

/// 2026-09-26: With entries in the table, this fails once every required gate
/// that has a record has one newer than [`AMNESTY_EPOCH`]. With the table
/// empty, it fails if any such gate's newest record is not newer.
#[test]
fn amnesty_expires_once_every_gate_has_a_fresh_record() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root is two levels above the crate");
    let stale: Vec<&str> = REQUIRED_GATES
        .iter()
        .copied()
        .filter(|id| {
            // 2026-09-26: A gate with no record is skipped: the grant never
            // covered it. A record that exists but cannot be read counts as
            // stale.
            let records = records_newest_first(root, id);
            let Some(newest) = records.first() else {
                return false;
            };
            read_record(newest).ok().map(|r| r.recorded_at).unwrap_or(0) <= AMNESTY_EPOCH
        })
        .collect();
    if ONE_TIME_AMNESTY.is_empty() {
        assert!(
            stale.is_empty(),
            "the PR #648 grant was removed before every required gate had a fresh record: {stale:?}"
        );
    } else {
        assert!(
            !stale.is_empty(),
            "every required gate now has a record newer than AMNESTY_EPOCH \
             (end of 2026-08-27 UTC): empty the fully re-earned one-time grant"
        );
    }
}
