// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests of `Freshness` and `short` that need no network.
//!
//! Owner: server (model download).
//! Invariants: none beyond the types.

use super::*;

#[test]
fn nothing_on_disk_is_missing_not_stale() {
    let dir = std::env::temp_dir().join("metrale-stale-empty");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    assert_eq!(
        check("org/none", &dir).unwrap_or(Freshness::Unknown),
        Freshness::Missing,
        "an absent model is not an out-of-date one"
    );
}

#[test]
fn unknown_is_not_stale() {
    // 2026-09-26: Only `Stale` is stale; a failed check (`Unknown`) is not.
    assert!(!Freshness::Unknown.is_stale());
    assert!(!Freshness::Current.is_stale());
    assert!(!Freshness::Missing.is_stale());
    assert!(
        Freshness::Stale {
            local: "a".into(),
            remote: "b".into()
        }
        .is_stale()
    );
}

#[test]
fn short_revisions_are_seven_chars_and_never_panic() {
    assert_eq!(short("a1b2c3d4e5f6"), "a1b2c3d");
    assert_eq!(short("abc"), "abc");
    assert_eq!(short(""), "");
}
